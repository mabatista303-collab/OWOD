import logging
from types import MethodType

import torch


logger = logging.getLogger(__name__)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _freeze_module(module, module_name):
    if module is None:
        return False

    for parameter in module.parameters():
        parameter.requires_grad = False

    logger.info("Parameter isolation: froze %s", module_name)
    return True


def _register_row_mask(parameter, frozen_rows):
    if parameter is None or frozen_rows <= 0:
        return None

    def hook(grad):
        if grad is None:
            return grad
        grad = grad.clone()
        grad[:frozen_rows] = 0
        return grad

    return parameter.register_hook(hook)


def _count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _get_protected_rows(model):
    model = _unwrap_model(model)
    protected_rows = getattr(model, "_param_iso_protected_rows", None)
    if protected_rows is None:
        protected_rows = []
        model._param_iso_protected_rows = protected_rows
    return protected_rows


def _track_protected_rows(model, parameter, frozen_rows, name):
    if parameter is None or frozen_rows <= 0:
        return

    _get_protected_rows(model).append(
        {
            "parameter": parameter,
            "frozen_rows": frozen_rows,
            "name": name,
            "reference": None,
        }
    )


def initialize_parameter_isolation(model):
    model = _unwrap_model(model)
    protected_rows = _get_protected_rows(model)

    if not protected_rows:
        return

    with torch.no_grad():
        for item in protected_rows:
            item["reference"] = item["parameter"].detach().clone()[: item["frozen_rows"]].clone()

    logger.info(
        "Parameter isolation: captured reference values for %d protected parameter tensors",
        len(protected_rows),
    )


def _restore_protected_rows(model):
    restored = 0
    with torch.no_grad():
        for item in _get_protected_rows(model):
            reference = item.get("reference")
            if reference is None:
                continue
            item["parameter"].data[: item["frozen_rows"]].copy_(reference)
            restored += 1
    return restored


def _clear_optimizer_state(optimizer, parameter, frozen_rows):
    state = optimizer.state.get(parameter, None)
    if not state:
        return

    for _, value in state.items():
        if not torch.is_tensor(value):
            continue
        if value.ndim == 0 or value.shape[0] < frozen_rows:
            continue
        value[:frozen_rows].zero_()


def enforce_parameter_isolation(model, optimizer=None):
    model = _unwrap_model(model)
    protected_rows = _get_protected_rows(model)
    if not protected_rows:
        return

    restored = _restore_protected_rows(model)

    if optimizer is not None:
        for item in protected_rows:
            if item.get("reference") is None:
                continue
            _clear_optimizer_state(optimizer, item["parameter"], item["frozen_rows"])

    if restored:
        logger.debug("Parameter isolation: restored %d protected parameter tensors", restored)


def install_parameter_isolation_guard(model, optimizer):
    if getattr(optimizer, "_param_iso_guard_installed", False):
        return

    original_step = optimizer.step

    def guarded_step(self, *args, **kwargs):
        result = original_step(*args, **kwargs)
        enforce_parameter_isolation(model, self)
        return result

    optimizer.step = MethodType(guarded_step, optimizer)
    optimizer._param_iso_guard_installed = True
    logger.info("Parameter isolation: installed optimizer step guard")


def apply_parameter_isolation(model, cfg):
    model = _unwrap_model(model)
    isolation_cfg = cfg.OWOD.INCREMENTAL.PARAM_ISOLATION

    if not isolation_cfg.ENABLED:
        return

    before = _count_trainable_parameters(model)

    if isolation_cfg.FREEZE_BACKBONE:
        _freeze_module(getattr(model, "backbone", None), "backbone")

    if isolation_cfg.FREEZE_PROPOSAL_GENERATOR:
        _freeze_module(getattr(model, "proposal_generator", None), "proposal_generator")

    roi_heads = getattr(model, "roi_heads", None)
    if isolation_cfg.FREEZE_ROI_BOX_FEATURE_EXTRACTOR and roi_heads is not None:
        frozen = False
        if hasattr(roi_heads, "res5"):
            frozen = _freeze_module(roi_heads.res5, "roi_heads.res5") or frozen
        if hasattr(roi_heads, "box_head"):
            frozen = _freeze_module(roi_heads.box_head, "roi_heads.box_head") or frozen
        if not frozen:
            logger.warning(
                "Parameter isolation requested ROI box feature extractor freezing, "
                "but no supported feature extractor was found."
            )

    prev_intro_cls = cfg.OWOD.PREV_INTRODUCED_CLS
    if prev_intro_cls > 0 and roi_heads is not None and hasattr(roi_heads, "box_predictor"):
        box_predictor = roi_heads.box_predictor
        hook_handles = []

        if isolation_cfg.FREEZE_OLD_CLASSIFIER and hasattr(box_predictor, "cls_score"):
            cls_layer = box_predictor.cls_score
            hook_handles.append(_register_row_mask(cls_layer.weight, prev_intro_cls))
            _track_protected_rows(model, cls_layer.weight, prev_intro_cls, "cls_score.weight")
            if cls_layer.bias is not None:
                hook_handles.append(_register_row_mask(cls_layer.bias, prev_intro_cls))
                _track_protected_rows(model, cls_layer.bias, prev_intro_cls, "cls_score.bias")
            logger.info(
                "Parameter isolation: masked gradients for %d old classifier rows",
                prev_intro_cls,
            )

        if isolation_cfg.FREEZE_OLD_BBOX_REG and hasattr(box_predictor, "bbox_pred"):
            bbox_layer = box_predictor.bbox_pred
            box_dim = len(box_predictor.box2box_transform.weights)
            frozen_rows = prev_intro_cls * box_dim
            if bbox_layer.weight.shape[0] > box_dim:
                hook_handles.append(_register_row_mask(bbox_layer.weight, frozen_rows))
                _track_protected_rows(model, bbox_layer.weight, frozen_rows, "bbox_pred.weight")
                if bbox_layer.bias is not None:
                    hook_handles.append(_register_row_mask(bbox_layer.bias, frozen_rows))
                    _track_protected_rows(model, bbox_layer.bias, frozen_rows, "bbox_pred.bias")
                logger.info(
                    "Parameter isolation: masked gradients for %d old bbox rows",
                    frozen_rows,
                )
            else:
                logger.info(
                    "Parameter isolation: bbox regression is class agnostic, "
                    "skipping old-class bbox masking."
                )

        model._param_iso_hook_handles = [handle for handle in hook_handles if handle is not None]

    after = _count_trainable_parameters(model)
    logger.info(
        "Parameter isolation enabled: trainable params %d -> %d",
        before,
        after,
    )
