"""
Preprocessing utilities for UT-EndoMRI MRI volumes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class MRIPreprocessor:
    """Preprocessing pipeline for MRI images."""

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (5.0, 5.0, 5.0),
        target_size: Optional[Tuple[int, int, int]] = None,
        intensity_clip_percentiles: Tuple[float, float] = (1, 99),
        normalize_method: str = "min_max",
        resampling_order: int = 3,
    ):
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.intensity_clip_percentiles = intensity_clip_percentiles
        self.normalize_method = normalize_method
        self.resampling_order = resampling_order

    def clip_intensity(
        self,
        image: np.ndarray,
        percentiles: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        percentiles = percentiles or self.intensity_clip_percentiles
        lower = np.percentile(image, percentiles[0])
        upper = np.percentile(image, percentiles[1])
        clipped = np.clip(image, lower, upper)
        logger.debug("Clipped intensity to [%f, %f]", lower, upper)
        return clipped

    def normalize_intensity(
        self,
        image: np.ndarray,
        method: Optional[str] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        method = method or self.normalize_method
        masked_image = image[mask > 0] if mask is not None else image

        if method == "min_max":
            min_val = masked_image.min()
            max_val = masked_image.max()
            normalized = (image - min_val) / (max_val - min_val + 1e-8)
        elif method == "z_score":
            mean = masked_image.mean()
            std = masked_image.std()
            normalized = (image - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.debug("Normalized intensity using %s method", method)
        return normalized

    def resample_image(
        self,
        image: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Optional[Tuple[float, float, float]] = None,
        is_label: bool = False,
    ) -> np.ndarray:
        target_spacing = target_spacing or self.target_spacing
        zoom_factors = [
            orig / target for orig, target in zip(original_spacing, target_spacing)
        ]
        order = 0 if is_label else self.resampling_order
        resampled = ndimage.zoom(image, zoom_factors, order=order, mode="nearest")
        logger.debug("Resampled from %s to %s", image.shape, resampled.shape)
        return resampled

    def crop_or_pad(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int, int],
        mode: str = "constant",
    ) -> np.ndarray:
        current_size = image.shape
        pad_width = []
        for curr, target in zip(current_size, target_size):
            diff = target - curr
            if diff > 0:
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))

        if any(p != (0, 0) for p in pad_width):
            image = np.pad(image, pad_width, mode=mode)

        slices = []
        for curr, target in zip(image.shape, target_size):
            if curr > target:
                start = (curr - target) // 2
                slices.append(slice(start, start + target))
            else:
                slices.append(slice(None))

        return image[tuple(slices)]

    def preprocess_image(
        self,
        image: np.ndarray,
        original_spacing: Tuple[float, float, float],
        is_label: bool = False,
    ) -> np.ndarray:
        if not is_label:
            image = self.clip_intensity(image)
            image = self.normalize_intensity(image)

        image = self.resample_image(image, original_spacing, is_label=is_label)

        if self.target_size is not None:
            image = self.crop_or_pad(image, self.target_size)

        return image

    def preprocess_pair(
        self,
        image: np.ndarray,
        label: np.ndarray,
        original_spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        image_processed = self.preprocess_image(image, original_spacing, is_label=False)
        label_processed = self.preprocess_image(label, original_spacing, is_label=True)
        return image_processed, label_processed


def compute_intensity_statistics(
    images: List[np.ndarray],
    masks: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """Compute summary statistics over a collection of images."""
    all_intensities = []

    for i, img in enumerate(images):
        if masks is not None and i < len(masks) and masks[i] is not None:
            masked = img[masks[i] > 0]
        else:
            masked = img.flatten()
        all_intensities.append(masked)

    all_intensities = np.concatenate(all_intensities)

    return {
        "mean": float(np.mean(all_intensities)),
        "std": float(np.std(all_intensities)),
        "min": float(np.min(all_intensities)),
        "max": float(np.max(all_intensities)),
        "median": float(np.median(all_intensities)),
        "percentile_1": float(np.percentile(all_intensities, 1)),
        "percentile_99": float(np.percentile(all_intensities, 99)),
    }


def create_foreground_mask(image: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Create a binary foreground mask via intensity thresholding."""
    mask = (image > threshold).astype(np.uint8)
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=2)
    return mask


__all__ = [
    "MRIPreprocessor",
    "compute_intensity_statistics",
    "create_foreground_mask",
]
