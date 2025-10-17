"""
Evaluation metrics for medical image segmentation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    per_class: bool = True,
) -> torch.Tensor:
    """Compute Dice Similarity Coefficient."""
    if pred.dim() == 4:
        num_classes = pred.max().item() + 1
        pred = F.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3).float()

    if target.dim() == 4:
        num_classes = target.max().item() + 1
        target = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice if per_class else dice.mean(dim=1)


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    per_class: bool = True,
) -> torch.Tensor:
    """Compute Intersection over Union (IoU)."""
    if pred.dim() == 4:
        num_classes = pred.max().item() + 1
        pred = F.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3).float()

    if target.dim() == 4:
        num_classes = target.max().item() + 1
        target = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou if per_class else iou.mean(dim=1)


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
) -> float:
    """Compute 95th percentile Hausdorff Distance."""
    spacing = spacing or tuple([1.0] * pred.ndim)

    pred_edge = pred ^ ndimage.binary_erosion(pred)
    target_edge = target ^ ndimage.binary_erosion(target)

    if not np.any(pred_edge) or not np.any(target_edge):
        return np.inf

    pred_dt = distance_transform_edt(~pred_edge, sampling=spacing)
    target_dt = distance_transform_edt(~target_edge, sampling=spacing)

    pred_to_target = pred_dt[target_edge]
    target_to_pred = target_dt[pred_edge]

    all_distances = np.concatenate([pred_to_target, target_to_pred])
    return float(np.percentile(all_distances, 95))


def surface_dice(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
    tolerance: float = 1.0,
) -> float:
    """Compute Surface Dice at a given tolerance."""
    spacing = spacing or tuple([1.0] * pred.ndim)

    pred_surface = pred ^ ndimage.binary_erosion(pred)
    target_surface = target ^ ndimage.binary_erosion(target)

    if not np.any(pred_surface) or not np.any(target_surface):
        return 0.0

    pred_dt = distance_transform_edt(~pred_surface, sampling=spacing)
    target_dt = distance_transform_edt(~target_surface, sampling=spacing)

    pred_in_tolerance = np.sum(pred_dt[target_surface] <= tolerance)
    target_in_tolerance = np.sum(target_dt[pred_surface] <= tolerance)

    total_surface = np.sum(pred_surface) + np.sum(target_surface)
    if total_surface == 0:
        return 0.0

    return (pred_in_tolerance + target_in_tolerance) / total_surface


class SegmentationMetrics:
    """Utility for computing a suite of segmentation metrics."""

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        include_background: bool = False,
        compute_hd95: bool = True,
        compute_surface_dice: bool = True,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.include_background = include_background
        self.compute_hd95 = compute_hd95
        self.compute_surface_dice = compute_surface_dice
        self.start_class = 0 if include_background else 1

    def compute_all_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        spacing: Optional[Tuple[float, ...]] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        dice = dice_coefficient(pred, target, per_class=True)
        iou = iou_score(pred, target, per_class=True)

        dice_mean = dice.mean(dim=0)
        iou_mean = iou.mean(dim=0)

        for i in range(self.start_class, self.num_classes):
            metrics[f"dice_{self.class_names[i]}"] = dice_mean[i].item()
            metrics[f"iou_{self.class_names[i]}"] = iou_mean[i].item()

        metrics["dice_mean"] = dice_mean[self.start_class :].mean().item()
        metrics["iou_mean"] = iou_mean[self.start_class :].mean().item()

        if (self.compute_hd95 or self.compute_surface_dice) and spacing is not None:
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            for i in range(self.start_class, self.num_classes):
                pred_class = pred_np == i
                target_class = target_np == i

                if not np.any(target_class):
                    continue

                hd95_list = []
                surface_dice_list = []

                for b in range(pred_np.shape[0]):
                    if not np.any(target_class[b]):
                        continue

                    if self.compute_hd95:
                        try:
                            hd = hausdorff_distance_95(pred_class[b], target_class[b], spacing)
                            if not np.isinf(hd):
                                hd95_list.append(hd)
                        except Exception as exc:
                            logger.debug(
                                "Failed to compute HD95 for class %s: %s",
                                self.class_names[i],
                                exc,
                            )

                    if self.compute_surface_dice:
                        try:
                            sd = surface_dice(
                                pred_class[b], target_class[b], spacing, tolerance=1.0
                            )
                            surface_dice_list.append(sd)
                        except Exception as exc:
                            logger.debug(
                                "Failed to compute surface Dice for class %s: %s",
                                self.class_names[i],
                                exc,
                            )

                if hd95_list:
                    metrics[f"hd95_{self.class_names[i]}"] = float(np.mean(hd95_list))
                if surface_dice_list:
                    metrics[f"surface_dice_{self.class_names[i]}"] = float(
                        np.mean(surface_dice_list)
                    )

        return metrics


def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute confusion matrix for multiclass segmentation."""
    mask = (target >= 0) & (target < num_classes)
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    indices = num_classes * target[mask].long() + pred[mask].long()
    conf_mat = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf_mat


__all__ = [
    "dice_coefficient",
    "iou_score",
    "hausdorff_distance_95",
    "surface_dice",
    "SegmentationMetrics",
    "compute_confusion_matrix",
]
