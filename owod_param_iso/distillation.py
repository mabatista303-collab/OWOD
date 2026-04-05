import torch
from torch.nn import functional as F


class DetectionDistillationHelper:
    def __init__(self, cfg):
        distill_cfg = cfg.OWOD.INCREMENTAL.DISTILLATION
        self.enabled = distill_cfg.ENABLED
        self.teacher_weights = distill_cfg.TEACHER_WEIGHTS or cfg.MODEL.WEIGHTS
        self.temperature = float(distill_cfg.TEMPERATURE)
        self.cls_weight = float(distill_cfg.CLS_WEIGHT)
        self.bbox_weight = float(distill_cfg.BBOX_WEIGHT)
        self.prev_classes = int(cfg.OWOD.PREV_INTRODUCED_CLS)
        self.cls_agnostic_bbox_reg = bool(cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG)
        self.box_dim = len(cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _zero_loss(self, tensor):
        return tensor.sum() * 0.0

    def get_distillation_losses(self, student_predictions, teacher_predictions):
        student_logits, student_deltas = student_predictions
        teacher_logits, teacher_deltas = teacher_predictions

        if not self.enabled or self.prev_classes <= 0 or student_logits.numel() == 0:
            zero = self._zero_loss(student_logits)
            return {
                "loss_distill_cls": zero,
                "loss_distill_box": zero,
            }

        losses = {}
        if self.cls_weight > 0:
            student_old_logits = torch.cat(
                [student_logits[:, : self.prev_classes], student_logits[:, -1:]], dim=1
            )
            teacher_old_logits = torch.cat(
                [teacher_logits[:, : self.prev_classes], teacher_logits[:, -1:]], dim=1
            )
            temperature = self.temperature
            student_log_probs = F.log_softmax(student_old_logits / temperature, dim=1)
            teacher_probs = F.softmax(teacher_old_logits / temperature, dim=1)
            cls_loss = (
                F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                * (temperature ** 2)
            )
            losses["loss_distill_cls"] = cls_loss * self.cls_weight
        else:
            losses["loss_distill_cls"] = self._zero_loss(student_logits)

        if self.bbox_weight > 0:
            if self.cls_agnostic_bbox_reg:
                student_old_deltas = student_deltas
                teacher_old_deltas = teacher_deltas
            else:
                old_bbox_channels = self.prev_classes * self.box_dim
                student_old_deltas = student_deltas[:, :old_bbox_channels]
                teacher_old_deltas = teacher_deltas[:, :old_bbox_channels]
            bbox_loss = F.smooth_l1_loss(
                student_old_deltas,
                teacher_old_deltas,
                reduction="mean",
            )
            losses["loss_distill_box"] = bbox_loss * self.bbox_weight
        else:
            losses["loss_distill_box"] = self._zero_loss(student_logits)

        return losses
