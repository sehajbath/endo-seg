"""
Data augmentation transforms for medical images.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import logging

logger = logging.getLogger(__name__)


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomFlip:
    """Random flip along specified axes."""

    def __init__(self, axes: Tuple[int, ...] = (0, 1, 2), prob: float = 0.5):
        self.axes = axes
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample["image"]
        label = sample["label"]

        axis = random.choice(self.axes)
        image = torch.flip(image, dims=[axis + 1])
        label = torch.flip(label, dims=[axis])

        sample["image"] = image
        sample["label"] = label
        return sample


class RandomRotation:
    """Random rotation in specified plane."""

    def __init__(
        self,
        angle_range: Tuple[float, float] = (-25, 25),
        axes: Tuple[int, int] = (0, 1),
        prob: float = 0.5,
        order: int = 3,
    ):
        self.angle_range = angle_range
        self.axes = axes
        self.prob = prob
        self.order = order

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        angle = random.uniform(*self.angle_range)

        image = sample["image"].numpy()
        label = sample["label"].numpy()

        rotated_image = []
        for c in range(image.shape[0]):
            rotated = ndimage.rotate(
                image[c],
                angle,
                axes=self.axes,
                reshape=False,
                order=self.order,
                mode="nearest",
            )
            rotated_image.append(rotated)

        rotated_image = np.stack(rotated_image, axis=0)
        rotated_label = ndimage.rotate(
            label,
            angle,
            axes=self.axes,
            reshape=False,
            order=0,
            mode="nearest",
        )

        sample["image"] = torch.from_numpy(rotated_image).float()
        sample["label"] = torch.from_numpy(rotated_label).long()
        return sample


class RandomTranslation:
    """Random translation (shift)."""

    def __init__(self, max_shift: int = 25, prob: float = 0.5):
        self.max_shift = max_shift
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample["image"].numpy()
        label = sample["label"].numpy()

        shifts = [random.randint(-self.max_shift, self.max_shift) for _ in range(len(image.shape) - 1)]
        shifts = [0] + shifts

        shifted_image = ndimage.shift(image, shift=shifts, order=3, mode="nearest")
        shifted_label = ndimage.shift(label, shift=shifts[1:], order=0, mode="nearest")

        sample["image"] = torch.from_numpy(shifted_image).float()
        sample["label"] = torch.from_numpy(shifted_label).long()
        return sample


class RandomElasticDeformation:
    """Random elastic deformation."""

    def __init__(self, alpha: float = 10.0, sigma: float = 3.0, prob: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample["image"].numpy()
        label = sample["label"].numpy()

        shape = image.shape[1:]

        dx = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma,
            mode="constant",
            cval=0,
        ) * self.alpha

        dy = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma,
            mode="constant",
            cval=0,
        ) * self.alpha

        if len(shape) == 3:
            dz = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0,
            ) * self.alpha

        if len(shape) == 3:
            x, y, z = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            )
            indices = [
                np.reshape(x + dx, (-1, 1)),
                np.reshape(y + dy, (-1, 1)),
                np.reshape(z + dz, (-1, 1)),
            ]
        else:
            x, y = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                indexing="ij",
            )
            indices = [
                np.reshape(x + dx, (-1, 1)),
                np.reshape(y + dy, (-1, 1)),
            ]

        deformed_image = []
        for c in range(image.shape[0]):
            deformed = ndimage.map_coordinates(
                image[c], indices, order=3, mode="nearest"
            ).reshape(shape)
            deformed_image.append(deformed)

        deformed_image = np.stack(deformed_image, axis=0)
        deformed_label = ndimage.map_coordinates(
            label, indices, order=0, mode="nearest"
        ).reshape(shape)

        sample["image"] = torch.from_numpy(deformed_image).float()
        sample["label"] = torch.from_numpy(deformed_label).long()
        return sample


class RandomGamma:
    """Random gamma correction for intensity augmentation."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5):
        self.gamma_range = gamma_range
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        gamma = random.uniform(*self.gamma_range)
        image = torch.pow(sample["image"], gamma)
        sample["image"] = image
        return sample


class RandomGaussianNoise:
    """Add random Gaussian noise."""

    def __init__(self, noise_std: float = 0.01, prob: float = 0.3):
        self.noise_std = noise_std
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample["image"]
        noise = torch.randn_like(image) * self.noise_std
        image = torch.clamp(image + noise, 0, 1)

        sample["image"] = image
        return sample


class RandomCrop:
    """Random crop to specified size."""

    def __init__(self, crop_size: Tuple[int, ...], prob: float = 1.0):
        self.crop_size = crop_size
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample["image"]
        label = sample["label"]

        current_size = image.shape[1:]
        starts = []
        for curr, crop in zip(current_size, self.crop_size):
            start = random.randint(0, curr - crop) if curr > crop else 0
            starts.append(start)

        slices = [slice(None)]
        for start, crop in zip(starts, self.crop_size):
            slices.append(slice(start, start + crop))

        image = image[tuple(slices)]
        label = label[tuple(slices[1:])]

        sample["image"] = image
        sample["label"] = label
        return sample


class ForegroundPatchSampler:
    """Foreground-biased sampler that oversamples ovaries and endometriomas."""

    def __init__(
        self,
        roi_size: Sequence[int],
        positive_fraction: float = 0.8,
        priority_labels: Optional[Iterable[int]] = None,
        foreground_labels: Optional[Iterable[int]] = None,
        num_attempts: int = 6,
    ) -> None:
        self.roi_size = tuple(int(v) for v in roi_size)
        self.positive_fraction = float(positive_fraction)
        self.priority_labels = tuple(priority_labels or ())
        self.foreground_labels = tuple(foreground_labels or ())
        self.num_attempts = max(1, int(num_attempts))

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]
        label = sample["label"]
        if image.ndim != label.ndim + 1:
            raise ValueError("Expected label without channel dim when sampling patches")

        cropped = self._sample_patch(image, label)
        if cropped is not None:
            sample["image"], sample["label"] = cropped
        return sample

    def _sample_patch(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(self.num_attempts):
            want_positive = random.random() < self.positive_fraction
            if want_positive:
                cropped = self._sample_positive(image, label)
                if cropped is not None:
                    return cropped
            else:
                return self._random_crop(image, label)
        return self._random_crop(image, label)

    def _sample_positive(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        masks = []
        if self.priority_labels:
            masks.append(self._mask_for_values(label, self.priority_labels))
        if self.foreground_labels:
            masks.append(self._mask_for_values(label, self.foreground_labels))
        masks.append(label > 0)

        for mask in masks:
            coords = mask.nonzero(as_tuple=False)
            if coords.numel() == 0:
                continue
            center = coords[random.randrange(coords.size(0))]
            return self._crop_around_center(image, label, center)
        return None

    def _mask_for_values(self, label: torch.Tensor, values: Iterable[int]) -> torch.Tensor:
        mask = torch.zeros_like(label, dtype=torch.bool)
        for value in values:
            mask |= label == int(value)
        return mask

    def _random_crop(self, image: torch.Tensor, label: torch.Tensor):
        starts = []
        for dim, roi in zip(label.shape, self.roi_size):
            if dim <= roi:
                starts.append(0)
            else:
                starts.append(random.randint(0, dim - roi))
        return self._crop(image, label, starts)

    def _crop_around_center(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        center: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        starts: List[int] = []
        for dim, roi, c in zip(label.shape, self.roi_size, center.tolist()):
            if dim <= roi:
                starts.append(0)
                continue
            c = int(c)
            min_start = max(0, c - roi + 1)
            max_start = min(c, dim - roi)
            if max_start < min_start:
                min_start = max_start
            start = random.randint(min_start, max_start) if max_start > min_start else min_start
            starts.append(start)
        return self._crop(image, label, starts)

    def _crop(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        starts: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_slices = [slice(None)]
        lbl_slices = []
        for start, roi, dim in zip(starts, self.roi_size, label.shape):
            if dim <= roi:
                img_slices.append(slice(0, dim))
                lbl_slices.append(slice(0, dim))
            else:
                img_slices.append(slice(start, start + roi))
                lbl_slices.append(slice(start, start + roi))

        cropped_img = image[tuple(img_slices)]
        cropped_lbl = label[tuple(lbl_slices)]

        if cropped_img.shape[1:] != tuple(self.roi_size):
            cropped_img, cropped_lbl = self._pad_to_roi(cropped_img, cropped_lbl)

        return cropped_img, cropped_lbl

    def _pad_to_roi(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_sizes: List[int] = []
        for dim, roi in zip(reversed(image.shape[1:]), reversed(self.roi_size)):
            pad_amt = max(0, roi - dim)
            pad_sizes.extend([0, pad_amt])

        if any(pad_sizes):
            image = F.pad(image, pad_sizes, value=0)
            label = F.pad(label.unsqueeze(0), pad_sizes, value=0).squeeze(0)

        slices = [slice(None)]
        lbl_slices = []
        for roi in self.roi_size:
            slices.append(slice(0, roi))
            lbl_slices.append(slice(0, roi))

        return image[tuple(slices)], label[tuple(lbl_slices)]


def get_train_transforms(config: Dict, roi_size: Optional[Sequence[int]] = None) -> Optional[Compose]:
    """Construct training augmentation pipeline from configuration."""
    transforms: List = []

    sampler_cfg = config.get("patch_sampler", {})
    if sampler_cfg.get("enabled") and roi_size is not None:
        transforms.append(
            ForegroundPatchSampler(
                roi_size=sampler_cfg.get("roi_size", roi_size),
                positive_fraction=sampler_cfg.get("positive_fraction", 0.8),
                priority_labels=sampler_cfg.get("priority_labels"),
                foreground_labels=sampler_cfg.get("foreground_labels"),
                num_attempts=sampler_cfg.get("num_attempts", 6),
            )
        )

    if config.get("random_flip_prob", 0) > 0:
        transforms.append(RandomFlip(prob=config["random_flip_prob"]))

    if config.get("random_rotation", 0) > 0:
        transforms.append(
            RandomRotation(
                angle_range=(-config["random_rotation"], config["random_rotation"]),
                prob=0.5,
            )
        )

    if config.get("random_translation", 0) > 0:
        transforms.append(
            RandomTranslation(max_shift=config["random_translation"], prob=0.5)
        )

    if config.get("random_elastic_deform", False):
        transforms.append(RandomElasticDeformation(prob=0.3))

    if config.get("random_gamma"):
        transforms.append(
            RandomGamma(gamma_range=tuple(config["random_gamma"]), prob=0.5)
        )

    if config.get("random_gaussian_noise", 0) > 0:
        transforms.append(
            RandomGaussianNoise(noise_std=config["random_gaussian_noise"], prob=0.3)
        )

    return Compose(transforms) if transforms else None


__all__ = [
    "Compose",
    "RandomFlip",
    "RandomRotation",
    "RandomTranslation",
    "RandomElasticDeformation",
    "RandomGamma",
    "RandomGaussianNoise",
    "RandomCrop",
    "ForegroundPatchSampler",
    "get_train_transforms",
]
