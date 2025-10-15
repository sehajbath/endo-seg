"""
Preprocessing functions for UT-EndoMRI dataset
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class MRIPreprocessor:
    """Preprocessing pipeline for MRI images"""

    def __init__(
            self,
            target_spacing: Tuple[float, float, float] = (5.0, 5.0, 5.0),
            target_size: Optional[Tuple[int, int, int]] = None,
            intensity_clip_percentiles: Tuple[float, float] = (1, 99),
            normalize_method: str = "min_max",
            resampling_order: int = 3
    ):
        """
        Initialize MRI preprocessor

        Args:
            target_spacing: Target voxel spacing in mm (x, y, z)
            target_size: Target size (H, W, D). If None, uses original size after resampling
            intensity_clip_percentiles: Percentiles for intensity clipping
            normalize_method: Normalization method ('min_max' or 'z_score')
            resampling_order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.intensity_clip_percentiles = intensity_clip_percentiles
        self.normalize_method = normalize_method
        self.resampling_order = resampling_order

    def clip_intensity(
            self,
            image: np.ndarray,
            percentiles: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Clip intensity values to specified percentiles

        Args:
            image: Input image array
            percentiles: (lower, upper) percentiles for clipping

        Returns:
            Clipped image
        """
        if percentiles is None:
            percentiles = self.intensity_clip_percentiles

        lower = np.percentile(image, percentiles[0])
        upper = np.percentile(image, percentiles[1])

        clipped = np.clip(image, lower, upper)
        logger.debug(f"Clipped intensity to [{lower:.2f}, {upper:.2f}]")

        return clipped

    def normalize_intensity(
            self,
            image: np.ndarray,
            method: Optional[str] = None,
            mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Normalize image intensity

        Args:
            image: Input image array
            method: Normalization method ('min_max' or 'z_score')
            mask: Optional mask to compute stats only within mask

        Returns:
            Normalized image
        """
        if method is None:
            method = self.normalize_method

        if mask is not None:
            masked_image = image[mask > 0]
        else:
            masked_image = image

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

        logger.debug(f"Normalized using {method} method")
        return normalized

    def resample_image(
            self,
            image: np.ndarray,
            original_spacing: Tuple[float, float, float],
            target_spacing: Optional[Tuple[float, float, float]] = None,
            is_label: bool = False
    ) -> np.ndarray:
        """
        Resample image to target spacing

        Args:
            image: Input image array
            original_spacing: Original voxel spacing (x, y, z)
            target_spacing: Target voxel spacing
            is_label: Whether this is a label image (use nearest neighbor)

        Returns:
            Resampled image
        """
        if target_spacing is None:
            target_spacing = self.target_spacing

        # Calculate zoom factors
        zoom_factors = [
            orig / target
            for orig, target in zip(original_spacing, target_spacing)
        ]

        # Use appropriate interpolation order
        order = 0 if is_label else self.resampling_order

        resampled = ndimage.zoom(image, zoom_factors, order=order, mode='nearest')

        logger.debug(f"Resampled from {image.shape} to {resampled.shape}")
        logger.debug(f"Spacing: {original_spacing} -> {target_spacing}")

        return resampled

    def resize_image(
            self,
            image: np.ndarray,
            target_size: Optional[Tuple[int, int, int]] = None,
            is_label: bool = False
    ) -> np.ndarray:
        """
        Resize image to target size

        Args:
            image: Input image array
            target_size: Target size (H, W, D)
            is_label: Whether this is a label image

        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size

        if target_size is None:
            return image

        # Calculate zoom factors
        zoom_factors = [
            target / orig
            for target, orig in zip(target_size, image.shape)
        ]

        # Use appropriate interpolation order
        order = 0 if is_label else self.resampling_order

        resized = ndimage.zoom(image, zoom_factors, order=order, mode='nearest')

        logger.debug(f"Resized from {image.shape} to {resized.shape}")

        return resized

    def crop_or_pad(
            self,
            image: np.ndarray,
            target_size: Tuple[int, int, int],
            mode: str = 'constant'
    ) -> np.ndarray:
        """
        Crop or pad image to target size

        Args:
            image: Input image
            target_size: Target size (H, W, D)
            mode: Padding mode for np.pad

        Returns:
            Cropped/padded image
        """
        current_size = image.shape

        # Calculate padding/cropping for each dimension
        pad_width = []
        for curr, target in zip(current_size, target_size):
            diff = target - curr
            if diff > 0:
                # Need padding
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                # Need cropping
                pad_width.append((0, 0))

        # Pad if needed
        if any(p != (0, 0) for p in pad_width):
            image = np.pad(image, pad_width, mode=mode)

        # Crop if needed
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
            is_label: bool = False
    ) -> np.ndarray:
        """
        Full preprocessing pipeline for an image

        Args:
            image: Input image array
            original_spacing: Original voxel spacing
            is_label: Whether this is a label image

        Returns:
            Preprocessed image
        """
        if not is_label:
            # Clip and normalize intensity (only for images, not labels)
            image = self.clip_intensity(image)
            image = self.normalize_intensity(image)

        # Resample to target spacing
        image = self.resample_image(image, original_spacing, is_label=is_label)

        # Resize to target size if specified
        if self.target_size is not None:
            image = self.crop_or_pad(image, self.target_size)

        return image

    def preprocess_pair(
            self,
            image: np.ndarray,
            label: np.ndarray,
            original_spacing: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image-label pair with same transformations

        Args:
            image: Input image
            label: Label mask
            original_spacing: Original voxel spacing

        Returns:
            Preprocessed image and label
        """
        image_processed = self.preprocess_image(image, original_spacing, is_label=False)
        label_processed = self.preprocess_image(label, original_spacing, is_label=True)

        return image_processed, label_processed


def compute_intensity_statistics(
        images: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None
) -> dict:
    """
    Compute intensity statistics across multiple images

    Args:
        images: List of image arrays
        masks: Optional list of masks to compute stats within

    Returns:
        Dictionary with statistics
    """
    all_intensities = []

    for i, img in enumerate(images):
        if masks is not None and i < len(masks) and masks[i] is not None:
            masked = img[masks[i] > 0]
        else:
            masked = img.flatten()
        all_intensities.append(masked)

    all_intensities = np.concatenate(all_intensities)

    stats = {
        'mean': float(np.mean(all_intensities)),
        'std': float(np.std(all_intensities)),
        'min': float(np.min(all_intensities)),
        'max': float(np.max(all_intensities)),
        'median': float(np.median(all_intensities)),
        'percentile_1': float(np.percentile(all_intensities, 1)),
        'percentile_99': float(np.percentile(all_intensities, 99)),
    }

    return stats


def create_foreground_mask(image: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Create binary foreground mask by thresholding

    Args:
        image: Input image
        threshold: Threshold value

    Returns:
        Binary mask
    """
    mask = (image > threshold).astype(np.uint8)

    # Remove small components
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=2)

    return mask


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create preprocessor
    preprocessor = MRIPreprocessor(
        target_spacing=(5.0, 5.0, 5.0),
        target_size=(128, 128, 32),
        intensity_clip_percentiles=(1, 99),
        normalize_method="min_max"
    )

    # Example with dummy data
    dummy_image = np.random.randn(100, 100, 20) * 100 + 500
    dummy_label = np.random.randint(0, 2, (100, 100, 20))
    original_spacing = (1.5, 1.5, 3.0)

    processed_image, processed_label = preprocessor.preprocess_pair(
        dummy_image, dummy_label, original_spacing
    )

    print(f"Original image shape: {dummy_image.shape}")
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Processed label shape: {processed_label.shape}")
    print(f"Image intensity range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")