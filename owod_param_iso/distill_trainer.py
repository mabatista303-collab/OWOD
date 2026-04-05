import logging
import os
import time
from collections import OrderedDict
from contextlib import nullcontext

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads

from .distillation import DetectionDistillationHelper


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class DistillationTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.distillation_helper = DetectionDistillationHelper(cfg)
        self.teacher_model = self._build_teacher_model(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name, cfg)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        return OrderedDict({k + "_TTA": v for k, v in res.items()})

    def _build_teacher_model(self, cfg):
        teacher_model = type(self).build_model(cfg)
        DetectionCheckpointer(teacher_model).resume_or_load(
            self.distillation_helper.teacher_weights, resume=False
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        return teacher_model

    def _forward_detection_branch(self, model, batched_inputs):
        student = _unwrap_model(model)
        if not isinstance(student.roi_heads, Res5ROIHeads):
            raise NotImplementedError("DistillationTrainer currently supports Res5ROIHeads only.")
        if student.roi_heads.mask_on:
            raise NotImplementedError("DistillationTrainer does not support MASK_ON models.")

        images = student.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(student.device) for x in batched_inputs]
        features = student.backbone(images.tensor)
        proposals, proposal_losses = student.proposal_generator(images, features, gt_instances)

        roi_heads = student.roi_heads
        proposals = roi_heads.label_and_sample_proposals(proposals, gt_instances)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = roi_heads._shared_roi_transform(
            [features[f] for f in roi_heads.in_features], proposal_boxes
        )
        input_features = box_features.mean(dim=[2, 3])
        predictions = roi_heads.box_predictor(input_features)

        if roi_heads.enable_clustering:
            roi_heads.box_predictor.update_feature_store(input_features, proposals)
        if roi_heads.compute_energy_flag:
            roi_heads.compute_energy(predictions, proposals)

        detector_losses = roi_heads.box_predictor.losses(predictions, proposals, input_features)
        return proposals, proposal_losses, predictions, detector_losses, proposal_boxes, images

    def _forward_teacher_predictions(self, proposal_boxes, batched_inputs):
        teacher = self.teacher_model
        images = teacher.preprocess_image(batched_inputs)
        features = teacher.backbone(images.tensor)
        roi_heads = teacher.roi_heads
        box_features = roi_heads._shared_roi_transform(
            [features[f] for f in roi_heads.in_features], proposal_boxes
        )
        input_features = box_features.mean(dim=[2, 3])
        return roi_heads.box_predictor(input_features)

    def run_step(self):
        assert self.model.training, "[DistillationTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        _, proposal_losses, student_predictions, detector_losses, proposal_boxes, _ = (
            self._forward_detection_branch(self.model, data)
        )

        with torch.no_grad():
            teacher_predictions = self._forward_teacher_predictions(proposal_boxes, data)

        loss_dict = {}
        loss_dict.update(detector_losses)
        loss_dict.update(proposal_losses)
        loss_dict.update(
            self.distillation_helper.get_distillation_losses(
                student_predictions, teacher_predictions
            )
        )
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        with torch.cuda.stream(torch.cuda.Stream()) if losses.device.type == "cuda" else nullcontext():
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        self.optimizer.step()
