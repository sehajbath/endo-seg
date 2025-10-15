"""
DataLoader utilities for training and validation
"""
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from typing import Optional, Dict, Any
import logging

from .dataset import EndoMRIDataset
from .preprocessing import MRIPreprocessor
from .augmentation import get_train_transforms
from .utils import (
    EndoMRIDataInfo,
    canonicalize_structure_list,
    merge_structure_labels,
    get_subject_data_dict,
    load_nifti
)

logger = logging.getLogger(__name__)


def get_dataloaders(
        data_root: str,
        splits: Dict[str, list],
        config: Dict[str, Any],
        dataset_name: str = "D2_TCPW",
        num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Args:
        data_root: Root directory of dataset
        splits: Dictionary with 'train', 'val', 'test' subject IDs
        config: Configuration dictionary
        dataset_name: Dataset name ('D1_MHS' or 'D2_TCPW')
        num_workers: Number of workers for data loading

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Create preprocessor
    preprocess_config = config.get('preprocessing', {})
    preprocessor = MRIPreprocessor(
        target_spacing=tuple(preprocess_config.get('target_spacing', [5.0, 5.0, 5.0])),
        target_size=tuple(preprocess_config.get('target_size', [128, 128, 32])),
        intensity_clip_percentiles=tuple(preprocess_config.get('intensity_clip_percentiles', [1, 99])),
        normalize_method=preprocess_config.get('normalize_method', 'min_max'),
        resampling_order=preprocess_config.get('resampling_order', 3)
    )

    # Get sequences and structures from config
    sequences = [seq for seq, enabled in config.get('sequences', {}).items() if enabled]

    raw_structures = [struct for struct, enabled in config.get('structures', {}).items() if enabled]
    try:
        structures = canonicalize_structure_list(raw_structures)
    except ValueError as exc:
        raise ValueError(f"Invalid structure name in configuration: {exc}") from exc

    logger.info(f"Using sequences: {sequences}")
    if raw_structures != structures:
        logger.info(f"Structures (configured): {raw_structures}")
    logger.info(f"Segmenting structures (canonical): {structures}")

    # Create augmentation transforms
    aug_config = config.get('augmentation', {}).get('train', {})
    train_transform = get_train_transforms(aug_config) if aug_config else None

    # Create datasets
    train_dataset = EndoMRIDataset(
        data_root=data_root,
        subject_ids=splits['train'],
        sequences=sequences,
        structures=structures,
        dataset_name=dataset_name,
        preprocessor=preprocessor,
        transform=train_transform,
        cache_data=False
    )

    val_dataset = EndoMRIDataset(
        data_root=data_root,
        subject_ids=splits['val'],
        sequences=sequences,
        structures=structures,
        dataset_name=dataset_name,
        preprocessor=preprocessor,
        transform=None,  # No augmentation for validation
        cache_data=False
    )

    test_dataset = EndoMRIDataset(
        data_root=data_root,
        subject_ids=splits['test'],
        sequences=sequences,
        structures=structures,
        dataset_name=dataset_name,
        preprocessor=preprocessor,
        transform=None,
        cache_data=False
    )

    # Get batch size
    batch_size = config.get('training', {}).get('batch_size', 2)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch_size=1 for validation to avoid padding issues
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def compute_class_weights(
        data_root: str,
        subject_ids: list,
        sequences: list,
        structures: list,
        dataset_name: str = "D2_TCPW",
        num_classes: int = 4
) -> torch.Tensor:
    """
    Compute class weights based on frequency (for weighted loss)

    Args:
        data_root: Root directory
        subject_ids: List of subject IDs to analyze
        sequences: MRI sequences
        structures: Structures to segment
        dataset_name: Dataset name
        num_classes: Number of classes (including background)

    Returns:
        Class weights tensor
    """
    logger.info("Computing class weights from training data...")

    from pathlib import Path

    class_counts = np.zeros(num_classes)
    dataset_path = Path(data_root) / dataset_name

    canonical_structures = canonicalize_structure_list(structures)
    structure_to_idx = {}
    for struct in canonical_structures:
        class_idx = EndoMRIDataInfo.structure_to_index(struct)
        if class_idx >= num_classes:
            logger.warning(
                f"Structure '{struct}' maps to class index {class_idx}, "
                f"but only {num_classes} classes are configured. Skipping."
            )
            continue
        structure_to_idx[struct] = class_idx

    for subject_id in subject_ids:
        subject_dir = dataset_path / subject_id

        if not subject_dir.exists():
            continue

        data_dict = get_subject_data_dict(
            subject_dir,
            sequences,
            structures,
            rater_id=None
        )

        # Load labels
        label_dict = {}
        for struct in canonical_structures:
            label_path = data_dict.get(f'label_{struct}')
            if label_path is not None:
                label_data, _ = load_nifti(str(label_path))
                label_dict[struct] = label_data

        # Merge labels
        if label_dict:
            merged_label = merge_structure_labels(label_dict, structure_to_idx)

            # Count classes
            for c in range(num_classes):
                class_counts[c] += np.sum(merged_label == c)

    # Compute weights (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)

    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes

    logger.info(f"Class weights computed:")
    index_to_structure = {idx: name for name, idx in EndoMRIDataInfo.STRUCTURE_CLASS_INDEX.items()}
    class_names = ['background'] + [
        index_to_structure.get(i, f'class_{i}') for i in range(1, num_classes)
    ]

    for i, weight in enumerate(class_weights):
        name = class_names[i] if i < len(class_names) else f'class_{i}'
        logger.info(f"  {name}: {weight:.4f} (count: {class_counts[i]:.0f})")

    return torch.tensor(class_weights, dtype=torch.float32)


class InfiniteDataLoader:
    """
    Wrapper for DataLoader that loops infinitely
    Useful for training with step-based (not epoch-based) training
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Reset iterator when epoch ends
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

    def __len__(self):
        return len(self.dataloader)


def collate_fn_with_metadata(batch: list) -> Dict[str, Any]:
    """
    Custom collate function that handles metadata

    Args:
        batch: List of samples from dataset

    Returns:
        Batched data with metadata preserved
    """
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    subject_ids = [item['subject_id'] for item in batch]
    spacings = torch.stack([item['spacing'] for item in batch])

    return {
        'image': images,
        'label': labels,
        'subject_id': subject_ids,
        'spacing': spacings
    }


def create_weighted_sampler(
        dataset: EndoMRIDataset,
        class_weights: Optional[torch.Tensor] = None
) -> WeightedRandomSampler:
    """
    Create weighted random sampler for imbalanced datasets

    Args:
        dataset: Dataset to sample from
        class_weights: Optional class weights

    Returns:
        WeightedRandomSampler
    """
    # If no class weights provided, compute from dataset
    if class_weights is None:
        class_counts = torch.zeros(4)  # Assume 4 classes

        for idx in range(len(dataset)):
            sample = dataset[idx]
            label = sample['label']

            for c in range(4):
                class_counts[c] += (label == c).sum().item()

        # Inverse frequency
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()

    # Compute sample weights based on primary structure present
    sample_weights = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample['label']

        # Find dominant class (excluding background)
        unique_classes = torch.unique(label)
        unique_classes = unique_classes[unique_classes > 0]  # Exclude background

        if len(unique_classes) > 0:
            # Use weight of the most frequent foreground class
            dominant_class = unique_classes[0].item()
            weight = class_weights[dominant_class].item()
        else:
            # Background only
            weight = class_weights[0].item()

        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def prefetch_to_device(
        dataloader: DataLoader,
        device: torch.device
):
    """
    Generator that prefetches batches to GPU

    Args:
        dataloader: DataLoader to prefetch from
        device: Target device (usually GPU)

    Yields:
        Batches on target device
    """
    for batch in dataloader:
        # Move all tensors to device
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.to(device, non_blocking=True)
            else:
                batch_on_device[key] = value

        yield batch_on_device


if __name__ == "__main__":
    import yaml
    from .utils import load_data_splits

    logging.basicConfig(level=logging.INFO)

    # Load config
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load splits
    splits = load_data_splits('../data/splits/split_info.json')

    # Create dataloaders
    dataloaders = get_dataloaders(
        data_root='../data/raw/UT-EndoMRI',
        splits=splits,
        config=config,
        dataset_name='D2_TCPW',
        num_workers=2
    )

    # Test train loader
    print("Testing train dataloader...")
    train_batch = next(iter(dataloaders['train']))
    print(f"Batch image shape: {train_batch['image'].shape}")
    print(f"Batch label shape: {train_batch['label'].shape}")
    print(f"Subject IDs: {train_batch['subject_id']}")

    print("\n✓ DataLoader test successful!")
