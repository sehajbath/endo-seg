"""
Data augmentation transforms for medical images
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import random
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class Compose:
    """Compose multiple transforms together"""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomFlip:
    """Random flip along specified axes"""

    def __init__(self, axes: Tuple[int, ...] = (0, 1, 2), prob: float = 0.5):
        """
        Args:
            axes: Axes to flip (0=H, 1=W, 2=D for 3D)
            prob: Probability of applying flip
        """
        self.axes = axes
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample['image']
        label = sample['label']

        # Randomly select an axis to flip
        axis = random.choice(self.axes)

        # Flip (add 1 to axis to account for channel dimension)
        image = torch.flip(image, dims=[axis + 1])
        label = torch.flip(label, dims=[axis])

        sample['image'] = image
        sample['label'] = label

        return sample


class RandomRotation:
    """Random rotation in specified plane"""

    def __init__(
            self,
            angle_range: Tuple[float, float] = (-25, 25),
            axes: Tuple[int, int] = (0, 1),
            prob: float = 0.5,
            order: int = 3
    ):
        """
        Args:
            angle_range: Range of rotation angles in degrees
            axes: Plane of rotation (0=H, 1=W, 2=D)
            prob: Probability of applying rotation
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        """
        self.angle_range = angle_range
        self.axes = axes
        self.prob = prob
        self.order = order

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        angle = random.uniform(*self.angle_range)

        image = sample['image'].numpy()
        label = sample['label'].numpy()

        # Rotate each channel of image
        rotated_image = []
        for c in range(image.shape[0]):
            rotated = ndimage.rotate(
                image[c],
                angle,
                axes=self.axes,
                reshape=False,
                order=self.order,
                mode='nearest'
            )
            rotated_image.append(rotated)

        rotated_image = np.stack(rotated_image, axis=0)

        # Rotate label with nearest neighbor
        rotated_label = ndimage.rotate(
            label,
            angle,
            axes=self.axes,
            reshape=False,
            order=0,
            mode='nearest'
        )

        sample['image'] = torch.from_numpy(rotated_image).float()
        sample['label'] = torch.from_numpy(rotated_label).long()

        return sample


class RandomTranslation:
    """Random translation (shift)"""

    def __init__(
            self,
            max_shift: int = 25,
            prob: float = 0.5
    ):
        """
        Args:
            max_shift: Maximum shift in pixels
            prob: Probability of applying translation
        """
        self.max_shift = max_shift
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample['image'].numpy()
        label = sample['label'].numpy()

        # Random shifts for each spatial dimension
        shifts = [
            random.randint(-self.max_shift, self.max_shift)
            for _ in range(len(image.shape) - 1)  # Exclude channel dimension
        ]
        shifts = [0] + shifts  # No shift for channel dimension

        # Apply shift
        shifted_image = ndimage.shift(
            image,
            shift=shifts,
            order=3,
            mode='nearest'
        )

        shifted_label = ndimage.shift(
            label,
            shift=shifts[1:],  # Exclude channel dimension
            order=0,
            mode='nearest'
        )

        sample['image'] = torch.from_numpy(shifted_image).float()
        sample['label'] = torch.from_numpy(shifted_label).long()

        return sample


class RandomElasticDeformation:
    """Random elastic deformation"""

    def __init__(
            self,
            alpha: float = 10.0,
            sigma: float = 3.0,
            prob: float = 0.3
    ):
        """
        Args:
            alpha: Deformation intensity
            sigma: Smoothness of deformation
            prob: Probability of applying deformation
        """
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample['image'].numpy()
        label = sample['label'].numpy()

        shape = image.shape[1:]  # Exclude channel dimension

        # Generate random displacement fields
        dx = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma,
            mode="constant",
            cval=0
        ) * self.alpha

        dy = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma,
            mode="constant",
            cval=0
        ) * self.alpha

        if len(shape) == 3:
            dz = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            ) * self.alpha

        # Create meshgrid
        if len(shape) == 3:
            x, y, z = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij'
            )
            indices = [
                np.reshape(x + dx, (-1, 1)),
                np.reshape(y + dy, (-1, 1)),
                np.reshape(z + dz, (-1, 1))
            ]
        else:
            x, y = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                indexing='ij'
            )
            indices = [
                np.reshape(x + dx, (-1, 1)),
                np.reshape(y + dy, (-1, 1))
            ]

        # Apply deformation to each channel
        deformed_image = []
        for c in range(image.shape[0]):
            deformed = ndimage.map_coordinates(
                image[c],
                indices,
                order=3,
                mode='nearest'
            ).reshape(shape)
            deformed_image.append(deformed)

        deformed_image = np.stack(deformed_image, axis=0)

        # Apply to label
        deformed_label = ndimage.map_coordinates(
            label,
            indices,
            order=0,
            mode='nearest'
        ).reshape(shape)

        sample['image'] = torch.from_numpy(deformed_image).float()
        sample['label'] = torch.from_numpy(deformed_label).long()

        return sample


class RandomGamma:
    """Random gamma correction for intensity augmentation"""

    def __init__(
            self,
            gamma_range: Tuple[float, float] = (0.8, 1.2),
            prob: float = 0.5
    ):
        """
        Args:
            gamma_range: Range of gamma values
            prob: Probability of applying gamma correction
        """
        self.gamma_range = gamma_range
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        gamma = random.uniform(*self.gamma_range)

        image = sample['image']

        # Apply gamma correction
        image = torch.pow(image, gamma)

        sample['image'] = image

        return sample


class RandomGaussianNoise:
    """Add random Gaussian noise"""

    def __init__(
            self,
            noise_std: float = 0.01,
            prob: float = 0.3
    ):
        """
        Args:
            noise_std: Standard deviation of noise
            prob: Probability of adding noise
        """
        self.noise_std = noise_std
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample['image']

        # Add Gaussian noise
        noise = torch.randn_like(image) * self.noise_std
        image = image + noise

        # Clip to valid range
        image = torch.clamp(image, 0, 1)

        sample['image'] = image

        return sample


class RandomCrop:
    """Random crop to specified size"""

    def __init__(
            self,
            crop_size: Tuple[int, ...],
            prob: float = 1.0
    ):
        """
        Args:
            crop_size: Size of crop (H, W, D)
            prob: Probability of applying crop
        """
        self.crop_size = crop_size
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.prob:
            return sample

        image = sample['image']
        label = sample['label']

        # Get current size (exclude channel dimension)
        current_size = image.shape[1:]

        # Calculate crop indices
        starts = []
        for curr, crop in zip(current_size, self.crop_size):
            if curr > crop:
                start = random.randint(0, curr - crop)
            else:
                start = 0
            starts.append(start)

        # Create slice objects
        slices = [slice(None)]  # Keep all channels
        for start, crop in zip(starts, self.crop_size):
            slices.append(slice(start, start + crop))

        image = image[tuple(slices)]
        label = label[tuple(slices[1:])]  # Exclude channel dimension

        sample['image'] = image
        sample['label'] = label

        return sample


def get_train_transforms(config: Dict) -> Compose:
    """
    Get training augmentation pipeline from config

    Args:
        config: Augmentation configuration dictionary

    Returns:
        Composed transforms
    """
    transforms = []

    if config.get('random_flip_prob', 0) > 0:
        transforms.append(
            RandomFlip(prob=config['random_flip_prob'])
        )

    if config.get('random_rotation', 0) > 0:
        transforms.append(
            RandomRotation(
                angle_range=(-config['random_rotation'], config['random_rotation']),
                prob=0.5
            )
        )

    if config.get('random_translation', 0) > 0:
        transforms.append(
            RandomTranslation(
                max_shift=config['random_translation'],
                prob=0.5
            )
        )

    if config.get('random_elastic_deform', False):
        transforms.append(
            RandomElasticDeformation(prob=0.3)
        )

    if config.get('random_gamma'):
        transforms.append(
            RandomGamma(
                gamma_range=config['random_gamma'],
                prob=0.5
            )
        )

    if config.get('random_gaussian_noise', 0) > 0:
        transforms.append(
            RandomGaussianNoise(
                noise_std=config['random_gaussian_noise'],
                prob=0.3
            )
        )

    return Compose(transforms) if transforms else None


if __name__ == "__main__":
    # Example usage
    sample = {
        'image': torch.randn(1, 128, 128, 32),
        'label': torch.randint(0, 4, (128, 128, 32))
    }

    # Create augmentation pipeline
    transforms = Compose([
        RandomFlip(prob=0.5),
        RandomRotation(prob=0.5),
        RandomGamma(prob=0.5)
    ])

    augmented = transforms(sample)

    print(f"Original image shape: {sample['image'].shape}")
    print(f"Augmented image shape: {augmented['image'].shape}")