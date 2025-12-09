# """
# Data augmentation transforms for medical images.
# """

# from __future__ import annotations

# import random
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# import numpy as np
# import torch
# from scipy import ndimage
# from monai.transforms import RandCropByLabelClassesd

# import logging

# logger = logging.getLogger(__name__)


# class Compose:
#     """Compose multiple transforms together."""

#     def __init__(self, transforms: List):
#         self.transforms = transforms

#     def __call__(self, sample: Dict) -> Dict:
#         for transform in self.transforms:
#             sample = transform(sample)
#         return sample


# class RandomFlip:
#     """Random flip along specified axes."""

#     def __init__(self, axes: Tuple[int, ...] = (0, 1, 2), prob: float = 0.5):
#         self.axes = axes
#         self.prob = prob

#     def __call__(self, sample: Dict) -> Dict:
#         if random.random() > self.prob:
#             return sample

#         image = sample["image"]
#         label = sample["label"]

#         axis = random.choice(self.axes)
#         image = torch.flip(image, dims=[axis + 1])
#         label = torch.flip(label, dims=[axis])

#         sample["image"] = image
#         sample["label"] = label
#         return sample


# class RandomRotation:
#     """Random rotation in specified plane."""

#     def __init__(
#         self,
#         angle_range: Tuple[float, float] = (-25, 25),
#         axes: Tuple[int, int] = (0, 1),
#         prob: float = 0.5,
#         order: int = 3,
#     ):
#         self.angle_range = angle_range
#         self.axes = axes
#         self.prob = prob
#         self.order = order

#     def __call__(self, sample: Dict) -> Dict:
#         if random.random() > self.prob:
#             return sample

#         angle = random.uniform(*self.angle_range)

#         image = sample["image"].numpy()
#         label = sample["label"].numpy()

#         rotated_image = []
#         for c in range(image.shape[0]):
#             rotated = ndimage.rotate(
#                 image[c],
#                 angle,
#                 axes=self.axes,
#                 reshape=False,
#                 order=self.order,
#                 mode="nearest",
#             )
#             rotated_image.append(rotated)

#         rotated_image = np.stack(rotated_image, axis=0)
#         rotated_label = ndimage.rotate(
#             label,
#             angle,
#             axes=self.axes,
#             reshape=False,
#             order=0,
#             mode="nearest",
#         )

#         sample["image"] = torch.from_numpy(rotated_image).float()
#         sample["label"] = torch.from_numpy(rotated_label).long()
#         return sample


# class RandomTranslation:
#     """Random translation (shift)."""

#     def __init__(self, max_shift: int = 25, prob: float = 0.5):
#         self.max_shift = max_shift
#         self.prob = prob

#     def __call__(self, sample: Dict) -> Dict:
#         if random.random() > self.prob:
#             return sample

#         image = sample["image"].numpy()
#         label = sample["label"].numpy()

#         shifts = [random.randint(-self.max_shift, self.max_shift) for _ in range(len(image.shape) - 1)]
#         shifts = [0] + shifts

#         shifted_image = ndimage.shift(image, shift=shifts, order=3, mode="nearest")
#         shifted_label = ndimage.shift(label, shift=shifts[1:], order=0, mode="nearest")

#         sample["image"] = torch.from_numpy(shifted_image).float()
#         sample["label"] = torch.from_numpy(shifted_label).long()
#         return sample


# class RandomElasticDeformation:
#     """Random elastic deformation."""

#     def __init__(self, alpha: float = 10.0, sigma: float = 3.0, prob: float = 0.3):
#         self.alpha = alpha
#         self.sigma = sigma
#         self.prob = prob

#     def __call__(self, sample: Dict) -> Dict:
#         if random.random() > self.prob:
#             return sample

#         image = sample["image"].numpy()
#         label = sample["label"].numpy()

#         shape = image.shape[1:]

#         dx = ndimage.gaussian_filter(
#             (np.random.rand(*shape) * 2 - 1),
#             self.sigma,
#             mode="constant",
#             cval=0,
#         ) * self.alpha

#         dy = ndimage.gaussian_filter(
#             (np.random.rand(*shape) * 2 - 1),
#             self.sigma,
#             mode="constant",
#             cval=0,
#         ) * self.alpha

#         if len(shape) == 3:
#             dz = ndimage.gaussian_filter(
#                 (np.random.rand(*shape) * 2 - 1),
#                 self.sigma,
#                 mode="constant",
#                 cval=0,
#             ) * self.alpha

#         if len(shape) == 3:
#             x, y, z = np.meshgrid(
#                 np.arange(shape[0]),
#                 np.arange(shape[1]),
#                 np.arange(shape[2]),
#                 indexing="ij",
#             )
#             indices = [
#                 np.reshape(x + dx, (-1, 1)),
#                 np.reshape(y + dy, (-1, 1)),
#                 np.reshape(z + dz, (-1, 1)),
#             ]
#         else:
#             x, y = np.meshgrid(
#                 np.arange(shape[0]),
#                 np.arange(shape[1]),
#                 indexing="ij",
#             )
#             indices = [
#                 np.reshape(x + dx, (-1, 1)),
#                 np.reshape(y + dy, (-1, 1)),
#             ]

#         deformed_image = []
#         for c in range(image.shape[0]):
#             deformed = ndimage.map_coordinates(
#                 image[c], indices, order=3, mode="nearest"
#             ).reshape(shape)
#             deformed_image.append(deformed)

#         deformed_image = np.stack(deformed_image, axis=0)
#         deformed_label = ndimage.map_coordinates(
#             label, indices, order=0, mode="nearest"
#         ).reshape(shape)

#         sample["image"] = torch.from_numpy(deformed_image).float()
#         sample["label"] = torch.from_numpy(deformed_label).long()
#         return sample


# class RandomGamma:
#     """Random gamma correction for intensity augmentation."""

#     def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5):
#         self.gamma_range = gamma_range
#         self.prob = prob

#     def __call__(self, sample: Dict) -> Dict:
#         if random.random() > self.prob:
#             return sample

#         gamma = random.uniform(*self.gamma_range)
#         image = torch.pow(sample["image"], gamma)
#         sample["image"] = image
#         return sample


# class RandomGaussianNoise:
#     """Add random Gaussian noise."""

#     def __init__(self, noise_std: float = 0.01, prob: float = 0.3):
#         self.noise_std = noise_std
#         self.prob = prob

#     def __call__(self, sample: Dict) -> Dict:
#         if random.random() > self.prob:
#             return sample

#         image = sample["image"]
#         noise = torch.randn_like(image) * self.noise_std
#         image = torch.clamp(image + noise, 0, 1)

#         sample["image"] = image
#         return sample


# class RandomCrop:
#     """Random crop to specified size."""

#     def __init__(self, crop_size: Tuple[int, ...], prob: float = 1.0):
#         self.crop_size = crop_size
#         self.prob = prob

#     def __call__(self, sample: Dict) -> Dict:
#         if random.random() > self.prob:
#             return sample

#         image = sample["image"]
#         label = sample["label"]

#         current_size = image.shape[1:]
#         starts = []
#         for curr, crop in zip(current_size, self.crop_size):
#             start = random.randint(0, curr - crop) if curr > crop else 0
#             starts.append(start)

#         slices = [slice(None)]
#         for start, crop in zip(starts, self.crop_size):
#             slices.append(slice(start, start + crop))

#         image = image[tuple(slices)]
#         label = label[tuple(slices[1:])]

#         sample["image"] = image
#         sample["label"] = label
#         return sample


# def get_train_transforms(config: Dict, roi_size: Optional[Sequence[int]] = None) -> Optional[Compose]:
#     """Construct training augmentation pipeline from configuration."""
#     transforms: List = []

#     label_crop_cfg = config.get("label_crop", {})
#     if label_crop_cfg.get("enabled") and roi_size is not None:
#         crop_roi = tuple(int(v) for v in label_crop_cfg.get("roi_size", roi_size))
#         ratios = list(label_crop_cfg.get("ratios", [0.05, 0.2, 0.4, 0.35]))
#         num_classes = label_crop_cfg.get("num_classes")
#         if num_classes is None or num_classes <= 0:
#             num_classes = len(ratios)
#         if len(ratios) != num_classes:
#             logger.warning(
#                 "label-aware crop ratios length (%d) mismatches num_classes (%d); adjusting ratios.",
#                 len(ratios),
#                 num_classes,
#             )
#             if num_classes < len(ratios):
#                 ratios = ratios[:num_classes]
#             else:
#                 ratios.extend([ratios[-1]] * (num_classes - len(ratios)))

#         transforms.append(
#             RandCropByLabelClassesd(
#                 keys=("image", "label"),
#                 label_key="label",
#                 spatial_size=crop_roi,
#                 ratios=ratios,
#                 num_classes=num_classes,
#                 num_samples=max(1, int(label_crop_cfg.get("num_samples", 1))),
#                 allow_smaller=label_crop_cfg.get("allow_smaller", True),
#             )
#         )

#     if config.get("random_flip_prob", 0) > 0:
#         transforms.append(RandomFlip(prob=config["random_flip_prob"]))

#     if config.get("random_rotation", 0) > 0:
#         transforms.append(
#             RandomRotation(
#                 angle_range=(-config["random_rotation"], config["random_rotation"]),
#                 prob=0.5,
#             )
#         )

#     if config.get("random_translation", 0) > 0:
#         transforms.append(
#             RandomTranslation(max_shift=config["random_translation"], prob=0.5)
#         )

#     if config.get("random_elastic_deform", False):
#         transforms.append(RandomElasticDeformation(prob=0.3))

#     if config.get("random_gamma"):
#         transforms.append(
#             RandomGamma(gamma_range=tuple(config["random_gamma"]), prob=0.5)
#         )

#     if config.get("random_gaussian_noise", 0) > 0:
#         transforms.append(
#             RandomGaussianNoise(noise_std=config["random_gaussian_noise"], prob=0.3)
#         )

#     return Compose(transforms) if transforms else None


# __all__ = [
#     "Compose",
#     "RandomFlip",
#     "RandomRotation",
#     "RandomTranslation",
#     "RandomElasticDeformation",
#     "RandomGamma",
#     "RandomGaussianNoise",
#     "RandomCrop",
#     "get_train_transforms",
# ]

from monai.transforms import (
    Compose,
    LoadImaged,
    RandCropByLabelClassesd,
    RandFlipd,
    RandSpatialRotationd,
    RandZoomd,
    ScaleIntensityRanged,
    SpatialPadd,
    CropForegroundd,
    Orientationd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    Spacingd,
    EnsureTyped,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
    RandBiasFieldd,
    RandStdShiftIntensityd,
    RandGaussianSmoothd,
    RandElasticDeformationd,
    RandGammaContrastd,
)
from monai.data import decollate_batch
import numpy as np

def get_train_transforms(augmentation_cfg: dict, roi_size: tuple[int, int, int]):
    transforms_list = [
        # Common preprocessing that might be here before augmentation
        # EnsureChannelFirstd(keys=["image", "label"]),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        # CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=[roi_size[0], roi_size[1], roi_size[2]]),
        # SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode=("reflect", "constant")), 
    ]

    label_crop_cfg = augmentation_cfg.get("label_crop", {"enabled": False})
    if label_crop_cfg["enabled"]:
        # Ensure that only valid parameters for RandCropByLabelClassesd are passed
        transforms_list.append(
            RandCropByLabelClassesd(
                keys=["image", "label"],
                roi_size=label_crop_cfg["roi_size"],
                ratios=label_crop_cfg["ratios"],
                num_samples=label_crop_cfg["num_samples"],
                allow_smaller=label_crop_cfg["allow_smaller"],
            )
        )

    # Add other augmentation transforms based on config
    if augmentation_cfg.get("random_flip_prob", 0) > 0:
        transforms_list.append(RandFlipd(keys=["image", "label"], prob=augmentation_cfg["random_flip_prob"], spatial_axis=0))
        transforms_list.append(RandFlipd(keys=["image", "label"], prob=augmentation_cfg["random_flip_prob"], spatial_axis=1))
        transforms_list.append(RandFlipd(keys=["image", "label"], prob=augmentation_cfg["random_flip_prob"], spatial_axis=2))

    if augmentation_cfg.get("random_rotation", 0) > 0:
        transforms_list.append(RandSpatialRotationd(
            keys=["image", "label"],
            range_x=np.pi / 180 * augmentation_cfg["random_rotation"],
            range_y=np.pi / 180 * augmentation_cfg["random_rotation"],
            range_z=np.pi / 180 * augmentation_cfg["random_rotation"],
            prob=0.5, 
            padding_mode="border",
            mode=("bilinear", "nearest"),
        ))
    
    if augmentation_cfg.get("random_translation", 0) > 0:
        transforms_list.append(RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5))

    if augmentation_cfg.get("random_elastic_deform", False):
        transforms_list.append(RandElasticDeformationd(
            keys=["image", "label"],
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            prob=0.3,
            padding_mode="border",
            mode=("bilinear", "nearest"),
        ))

    if augmentation_cfg.get("random_gamma"):
        transforms_list.append(RandGammaContrastd(keys="image", prob=0.5, gamma=augmentation_cfg["random_gamma"])) 

    if augmentation_cfg.get("random_gaussian_noise", 0) > 0:
        transforms_list.append(RandGaussianNoised(keys="image", prob=0.3, mean=0.0, std=augmentation_cfg["random_gaussian_noise"])) 

    # Final type assurance
    transforms_list.append(EnsureTyped(keys=["image", "label"]))

    return Compose(transforms_list)
