"""
Data augmentation transforms for medical images.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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


def get_train_transforms(config: Dict) -> Optional[Compose]:
    """Construct training augmentation pipeline from configuration."""
    transforms: List = []

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
    "get_train_transforms",
]
