"""
DataLoader utilities for training and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import EndoMRIDataset
from .preprocessing import MRIPreprocessor
from .structures import (
    EndoMRIDataInfo,
    canonicalize_structure_list,
    merge_structure_labels,
)
from ..augment.transforms import get_train_transforms
from ..io.files import get_subject_data_dict, load_nifti

logger = logging.getLogger(__name__)


def get_dataloaders(
    data_root: str,
    splits: Dict[str, list],
    config: Dict[str, Any],
    dataset_name: str = "D2_TCPW",
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders based on configuration."""
    preprocess_config = config.get("preprocessing", {})
    preprocessor = MRIPreprocessor(
        target_spacing=tuple(preprocess_config.get("target_spacing", [5.0, 5.0, 5.0])),
        target_size=tuple(preprocess_config.get("target_size", [128, 128, 32])),
        intensity_clip_percentiles=tuple(
            preprocess_config.get("intensity_clip_percentiles", [1, 99])
        ),
        normalize_method=preprocess_config.get("normalize_method", "min_max"),
        resampling_order=preprocess_config.get("resampling_order", 3),
    )

    sequences = [seq for seq, enabled in config.get("sequences", {}).items() if enabled]

    raw_structures = [struct for struct, enabled in config.get("structures", {}).items() if enabled]
    structures = canonicalize_structure_list(raw_structures)

    logger.info("Using sequences: %s", sequences)
    if raw_structures != structures:
        logger.info("Structures (configured): %s", raw_structures)
    logger.info("Segmenting structures (canonical): %s", structures)

    aug_config = config.get("augmentation", {}).get("train", {})
    train_transform = get_train_transforms(aug_config) if aug_config else None

    datasets = {
        split: EndoMRIDataset(
            data_root=data_root,
            subject_ids=splits[split],
            sequences=sequences,
            structures=structures,
            dataset_name=dataset_name,
            preprocessor=preprocessor,
            transform=train_transform if split == "train" else None,
            cache_data=False,
        )
        for split in ("train", "val", "test")
    }

    batch_size = config.get("training", {}).get("batch_size", 2)

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    }

    logger.info(
        "Created dataloaders: train=%d, val=%d, test=%d",
        len(loaders["train"]),
        len(loaders["val"]),
        len(loaders["test"]),
    )

    return loaders


def compute_class_weights(
    data_root: str,
    subject_ids: list,
    sequences: list,
    structures: list,
    dataset_name: str = "D2_TCPW",
    num_classes: int = 4,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training split."""
    logger.info("Computing class weights from training data...")

    class_counts = np.zeros(num_classes)
    dataset_path = Path(data_root) / dataset_name

    canonical_structures = canonicalize_structure_list(structures)
    structure_to_idx: Dict[str, int] = {}
    for struct in canonical_structures:
        class_idx = EndoMRIDataInfo.structure_to_index(struct)
        if class_idx >= num_classes:
            logger.warning(
                "Structure '%s' maps to class index %d but only %d classes configured. Skipping.",
                struct,
                class_idx,
                num_classes,
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
            rater_id=None,
        )

        label_dict: Dict[str, np.ndarray] = {}
        for struct in canonical_structures:
            label_path = data_dict.get(f"label_{struct}")
            if label_path is not None:
                label_data, _ = load_nifti(str(label_path))
                label_dict[struct] = label_data

        if label_dict:
            merged_label = merge_structure_labels(label_dict, structure_to_idx)
            for c in range(num_classes):
                class_counts[c] += np.sum(merged_label == c)

    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes

    index_to_structure = {
        idx: name for name, idx in EndoMRIDataInfo.STRUCTURE_CLASS_INDEX.items()
    }
    class_names = ["background"] + [
        index_to_structure.get(i, f"class_{i}") for i in range(1, num_classes)
    ]

    for i, weight in enumerate(class_weights):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        logger.info("  %s: %.4f (count: %.0f)", name, weight, class_counts[i])

    return torch.tensor(class_weights, dtype=torch.float32)


class InfiniteDataLoader:
    """Wrapper that keeps yielding batches indefinitely."""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def __len__(self):
        return len(self.dataloader)


def collate_fn_with_metadata(batch: list) -> Dict[str, Any]:
    """Collate function that preserves metadata alongside tensors."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    subject_ids = [item["subject_id"] for item in batch]
    spacings = torch.stack([item["spacing"] for item in batch])

    return {
        "image": images,
        "label": labels,
        "subject_id": subject_ids,
        "spacing": spacings,
    }


def prefetch_to_device(dataloader: DataLoader, device: torch.device):
    """Generator that prefetches batches to the specified device."""
    for batch in dataloader:
        batch_on_device = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        yield batch_on_device


__all__ = [
    "get_dataloaders",
    "compute_class_weights",
    "InfiniteDataLoader",
    "collate_fn_with_metadata",
    "prefetch_to_device",
]
