"""Training-time data augmentation utilities built on MONAI transforms."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandZoomd,
)

logger = logging.getLogger(__name__)


def _build_label_crop_transform(
    cfg: Dict,
    fallback_roi: Sequence[int],
) -> RandCropByLabelClassesd:
    spatial_size = tuple(int(v) for v in cfg.get("roi_size", fallback_roi))

    ratios = list(cfg.get("ratios", [0.05, 0.2, 0.4, 0.35]))
    if not ratios:
        raise ValueError("label_crop ratios must contain at least one entry")

    num_classes = cfg.get("num_classes")
    if num_classes is None or num_classes <= 0:
        num_classes = len(ratios)

    if len(ratios) != num_classes:
        logger.warning(
            "label_crop ratios length (%d) mismatches num_classes (%d); adjusting ratios.",
            len(ratios),
            num_classes,
        )
        if num_classes < len(ratios):
            ratios = ratios[:num_classes]
        else:
            ratios.extend([ratios[-1]] * (num_classes - len(ratios)))

    return RandCropByLabelClassesd(
        keys=("image", "label"),
        label_key="label",
        spatial_size=spatial_size,
        ratios=ratios,
        num_classes=num_classes,
        num_samples=max(1, int(cfg.get("num_samples", 1))),
        allow_smaller=cfg.get("allow_smaller", True),
        warn=False,
    )


def get_train_transforms(
    config: Dict,
    roi_size: Optional[Sequence[int]] = None,
) -> Optional[Compose]:
    """Create a MONAI ``Compose`` of training-time augmentations.

    Parameters
    ----------
    config:
        Dictionary containing augmentation settings (typically ``config['augmentation']['train']``).
    roi_size:
        Optional spatial size used for label-aware cropping. If omitted, the crop transform
        falls back to the ROI specified in ``config['label_crop']['roi_size']``.
    """

    transforms: List = [EnsureChannelFirstd(keys="image", channel_dim=0)]

    label_crop_cfg = config.get("label_crop", {})
    if label_crop_cfg.get("enabled"):
        if roi_size is None and "roi_size" not in label_crop_cfg:
            raise ValueError("label_crop requires either roi_size argument or roi_size in config")

        transforms.append(_build_label_crop_transform(label_crop_cfg, roi_size or label_crop_cfg["roi_size"]))

    flip_prob = config.get("random_flip_prob", 0.0)
    if flip_prob > 0:
        for axis in (0, 1, 2):
            transforms.append(
                RandFlipd(keys=("image", "label"), prob=flip_prob, spatial_axis=axis)
            )

    rotation_deg = config.get("random_rotation", 0)
    if rotation_deg > 0:
        radians = np.deg2rad(rotation_deg)
        transforms.append(
            RandRotated(
                keys=("image", "label"),
                range_x=radians,
                range_y=radians,
                range_z=radians,
                prob=0.5,
                padding_mode="border",
                mode=("bilinear", "nearest"),
            )
        )

    if config.get("random_translation", 0) > 0:
        transforms.append(
            RandZoomd(
                keys=("image", "label"),
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.5,
                mode=("trilinear", "nearest"),
            )
        )

    if config.get("random_elastic_deform", False):
        transforms.append(
            Rand3DElasticd(
                keys=("image", "label"),
                sigma_range=(4, 8),
                magnitude_range=(50, 150),
                prob=0.3,
                mode=("trilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    gamma_range = config.get("random_gamma")
    if gamma_range:
        transforms.append(
            RandAdjustContrastd(
                keys="image",
                prob=0.5,
                gamma=gamma_range,
            )
        )

    noise_std = config.get("random_gaussian_noise", 0.0)
    if noise_std > 0:
        transforms.append(
            RandGaussianNoised(
                keys="image",
                prob=0.3,
                std=noise_std,
            )
        )

    if not transforms:
        return None

    transforms.append(EnsureTyped(keys=("image", "label")))
    return Compose(transforms)


__all__ = ["get_train_transforms"]
