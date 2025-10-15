"""
Evaluation metrics for medical image segmentation
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


def dice_coefficient(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-5,
        per_class: bool = True
) -> torch.Tensor:
    """
    Compute Dice Similarity Coefficient

    Args:
        pred: Predicted segmentation [B, C, H, W, D] or [B, H, W, D]
        target: Ground truth [B, C, H, W, D] or [B, H, W, D]
        smooth: Smoothing factor
        per_class: If True, compute per-class Dice

    Returns:
        Dice coefficient(s)
    """
    # Convert to one-hot if needed
    if pred.dim() == 4:  # [B, H, W, D]
        num_classes = pred.max().item() + 1
        pred = F.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3).float()

    if target.dim() == 4:
        num_classes = target.max().item() + 1
        target = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    # Compute Dice
    dice = (2.0 * intersection + smooth) / (union + smooth)

    if per_class:
        return dice  # [B, C]
    else:
        return dice.mean(dim=1)  # [B]


def iou_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-5,
        per_class: bool = True
) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU)

    Args:
        pred: Predicted segmentation
        target: Ground truth
        smooth: Smoothing factor
        per_class: If True, compute per-class IoU

    Returns:
        IoU score(s)
    """
    # Convert to one-hot if needed
    if pred.dim() == 4:
        num_classes = pred.max().item() + 1
        pred = F.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3).float()

    if target.dim() == 4:
        num_classes = target.max().item() + 1
        target = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection

    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)

    if per_class:
        return iou  # [B, C]
    else:
        return iou.mean(dim=1)  # [B]


def hausdorff_distance_95(
        pred: np.ndarray,
        target: np.ndarray,
        spacing: Optional[Tuple[float, ...]] = None
) -> float:
    """
    Compute 95th percentile Hausdorff Distance

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        spacing: Voxel spacing for distance computation

    Returns:
        HD95 in mm (if spacing provided) or pixels
    """
    if spacing is None:
        spacing = tuple([1.0] * pred.ndim)

    # Compute edges
    pred_edge = pred ^ ndimage.binary_erosion(pred)
    target_edge = target ^ ndimage.binary_erosion(target)

    if not np.any(pred_edge) or not np.any(target_edge):
        return np.inf

    # Distance transforms
    pred_dt = distance_transform_edt(~pred_edge, sampling=spacing)
    target_dt = distance_transform_edt(~target_edge, sampling=spacing)

    # Distances from pred to target and vice versa
    pred_to_target = pred_dt[target_edge]
    target_to_pred = target_dt[pred_edge]

    # Combine and compute 95th percentile
    all_distances = np.concatenate([pred_to_target, target_to_pred])
    hd95 = np.percentile(all_distances, 95)

    return hd95


def surface_dice(
        pred: np.ndarray,
        target: np.ndarray,
        spacing: Optional[Tuple[float, ...]] = None,
        tolerance: float = 1.0
) -> float:
    """
    Compute Surface Dice at specified tolerance

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        spacing: Voxel spacing
        tolerance: Distance tolerance in mm (or pixels if no spacing)

    Returns:
        Surface Dice score
    """
    if spacing is None:
        spacing = tuple([1.0] * pred.ndim)

    # Compute surfaces
    pred_surface = pred ^ ndimage.binary_erosion(pred)
    target_surface = target ^ ndimage.binary_erosion(target)

    if not np.any(pred_surface) or not np.any(target_surface):
        return 0.0

    # Distance transforms
    pred_dt = distance_transform_edt(~pred_surface, sampling=spacing)
    target_dt = distance_transform_edt(~target_surface, sampling=spacing)

    # Count surface points within tolerance
    pred_in_tolerance = np.sum(pred_dt[target_surface] <= tolerance)
    target_in_tolerance = np.sum(target_dt[pred_surface] <= tolerance)

    # Compute surface Dice
    total_surface = np.sum(pred_surface) + np.sum(target_surface)

    if total_surface == 0:
        return 0.0

    surface_dice = (pred_in_tolerance + target_in_tolerance) / total_surface

    return surface_dice


class SegmentationMetrics:
    """Compute multiple segmentation metrics"""

    def __init__(
            self,
            num_classes: int,
            class_names: Optional[List[str]] = None,
            include_background: bool = False,
            compute_hd95: bool = True,
            compute_surface_dice: bool = True
    ):
        """
        Initialize metrics computer

        Args:
            num_classes: Number of classes (including background)
            class_names: Names of classes
            include_background: Whether to include background in metrics
            compute_hd95: Whether to compute Hausdorff Distance
            compute_surface_dice: Whether to compute Surface Dice
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.include_background = include_background
        self.compute_hd95 = compute_hd95
        self.compute_surface_dice = compute_surface_dice

        # Start from class 1 if not including background
        self.start_class = 0 if include_background else 1

    def compute_all_metrics(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            spacing: Optional[Tuple[float, ...]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics

        Args:
            pred: Predicted segmentation [B, H, W, D] (class indices)
            target: Ground truth [B, H, W, D]
            spacing: Voxel spacing for distance metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Compute Dice per class
        dice = dice_coefficient(pred, target, per_class=True)

        # Compute IoU per class
        iou = iou_score(pred, target, per_class=True)

        # Average across batch
        dice_mean = dice.mean(dim=0)  # [C]
        iou_mean = iou.mean(dim=0)  # [C]

        # Store per-class metrics
        for i in range(self.start_class, self.num_classes):
            metrics[f'dice_{self.class_names[i]}'] = dice_mean[i].item()
            metrics[f'iou_{self.class_names[i]}'] = iou_mean[i].item()

        # Overall metrics (excluding background)
        metrics['dice_mean'] = dice_mean[self.start_class:].mean().item()
        metrics['iou_mean'] = iou_mean[self.start_class:].mean().item()

        # Distance-based metrics (computed on CPU with numpy)
        if (self.compute_hd95 or self.compute_surface_dice) and spacing is not None:
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            # Compute for each class (excluding background)
            for i in range(self.start_class, self.num_classes):
                pred_class = (pred_np == i)
                target_class = (target_np == i)

                # Skip if class not present in target
                if not np.any(target_class):
                    continue

                # Process each sample in batch
                hd95_list = []
                surface_dice_list = []

                for b in range(pred_np.shape[0]):
                    if not np.any(target_class[b]):
                        # Skip samples without this class in the target
                        continue

                    if self.compute_hd95:
                        try:
                            hd = hausdorff_distance_95(
                                pred_class[b],
                                target_class[b],
                                spacing
                            )
                            if not np.isinf(hd):
                                hd95_list.append(hd)
                        except Exception as exc:
                            logger.debug(f"Failed to compute HD95 for class {self.class_names[i]}: {exc}")

                    if self.compute_surface_dice:
                        try:
                            sd = surface_dice(
                                pred_class[b],
                                target_class[b],
                                spacing,
                                tolerance=1.0
                            )
                            surface_dice_list.append(sd)
                        except Exception as exc:
                            logger.debug(f"Failed to compute surface Dice for class {self.class_names[i]}: {exc}")

                if hd95_list:
                    metrics[f'hd95_{self.class_names[i]}'] = np.mean(hd95_list)

                if surface_dice_list:
                    metrics[f'surface_dice_{self.class_names[i]}'] = np.mean(surface_dice_list)

        return metrics


def compute_confusion_matrix(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int
) -> torch.Tensor:
    """
    Compute confusion matrix

    Args:
        pred: Predicted class indices [B, H, W, D]
        target: Ground truth [B, H, W, D]
        num_classes: Number of classes

    Returns:
        Confusion matrix [C, C]
    """
    mask = (target >= 0) & (target < num_classes)
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    indices = num_classes * target[mask].long() + pred[mask].long()
    conf_mat = torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    return conf_mat


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    num_classes = 4
    shape = (128, 128, 32)

    # Create dummy predictions and targets
    pred = torch.randint(0, num_classes, (batch_size,) + shape)
    target = torch.randint(0, num_classes, (batch_size,) + shape)

    # Initialize metrics computer
    class_names = ['background', 'uterus', 'ovary', 'endometrioma']
    metrics_computer = SegmentationMetrics(
        num_classes=num_classes,
        class_names=class_names,
        include_background=False,
        compute_hd95=True,
        compute_surface_dice=True
    )

    # Compute metrics
    spacing = (5.0, 5.0, 5.0)
    metrics = metrics_computer.compute_all_metrics(pred, target, spacing)

    print("Computed metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
