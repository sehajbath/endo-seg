"""
DataLoader utilities for training and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .dataset import EndoMRIDataset
from .preprocessing import MRIPreprocessor
from .structures import (
    EndoMRIDataInfo,
    canonicalize_structure_list,
    merge_structure_labels,
)
from ..augment.transforms import get_train_transforms
from ..io.files import get_subject_data_dict, load_nifti
from ..io.splits import summarize_split_counts

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

    split_summary = splits.get("split_summary")
    patient_stats = splits.get("patient_stats")
    if split_summary:
        logger.info("Patient-level split composition:")
        for split_name in ("train", "val", "test"):
            stats = split_summary.get(split_name)
            if stats:
                logger.info(
                    "  %s: %d patients (ovary+=%d, endometrioma+=%d)",
                    split_name,
                    stats.get("num_patients", 0),
                    stats.get("has_ovary", 0),
                    stats.get("has_endo", 0),
                )
    elif patient_stats:
        logger.info("Patient-level split composition:")
        for split_name in ("train", "val", "test"):
            ids = splits.get(split_name, [])
            stats = summarize_split_counts(ids, patient_stats)
            logger.info(
                "  %s: %d patients (ovary+=%d, endometrioma+=%d)",
                split_name,
                stats.get("num_patients", 0),
                stats.get("has_ovary", 0),
                stats.get("has_endo", 0),
            )

    logger.info("Using sequences: %s", sequences)
    if raw_structures != structures:
        logger.info("Structures (configured): %s", raw_structures)
    logger.info("Segmenting structures (canonical): %s", structures)

    aug_config = config.get("augmentation", {}).get("train", {})
    train_transform = (
        get_train_transforms(aug_config, roi_size=preprocessor.target_size)
        if aug_config
        else None
    )

    # Optional per-subject dataset mapping for combined splits
    dataset_map: Dict[str, str] = {}
    if patient_stats:
        for sid, stats in patient_stats.items():
            ds = stats.get("dataset")
            if ds:
                dataset_map[sid] = ds

    # Auto-detect which sequence labels were annotated on (per-subject)
    auto_detect_sequences = config.get("auto_detect_sequences", True)
    sequence_map = {}
    if auto_detect_sequences and len(sequences) > 1:
        from .sequence_detector import build_sequence_map

        logger.info("Auto-detecting label-sequence alignment for each subject...")

        all_subjects = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
        sequence_map = build_sequence_map(
            data_root=Path(data_root),
            subject_ids=all_subjects,
            sequences=sequences,
            dataset_name=dataset_name,
            rater_id=None,  # Will be determined by dataset
        )

        logger.info(
            "Sequence auto-detection complete: %d subjects mapped to optimal sequences",
            len(sequence_map)
        )
    elif len(sequences) == 1:
        logger.info("Only one sequence configured, skipping auto-detection")
    else:
        logger.info("Sequence auto-detection disabled (auto_detect_sequences=False)")

    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = EndoMRIDataset(
            data_root=data_root,
            subject_ids=splits[split],
            sequences=sequences,
            structures=structures,
            dataset_name=dataset_name,
            preprocessor=preprocessor,
            transform=train_transform if split == "train" else None,
            cache_data=False,
            dataset_map=dataset_map if dataset_map else None,
            sequence_map=sequence_map if sequence_map else None,
        )

    batch_size = config.get("training", {}).get("batch_size", 2)
    subject_sampler_cfg = config.get("training", {}).get("subject_sampler", {})
    train_sampler = None
    if subject_sampler_cfg.get("enabled"):
        if not patient_stats:
            logger.warning("Subject sampler enabled but patient_stats missing; falling back to uniform sampling.")
        else:
            weights: List[float] = []
            endo_weight = subject_sampler_cfg.get("endo_weight", 4.0)
            ovary_weight = subject_sampler_cfg.get("ovary_weight", 2.0)
            default_weight = subject_sampler_cfg.get("default_weight", 1.0)
            for entry in datasets["train"].data_index:
                subject_id = entry.get("subject_id")
                stats = patient_stats.get(subject_id, {}) if subject_id else {}
                if stats.get("has_endo"):
                    weights.append(endo_weight)
                elif stats.get("has_ovary"):
                    weights.append(ovary_weight)
                else:
                    weights.append(default_weight)
            if weights:
                train_sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True,
                )
                logger.info(
                    "Using weighted subject sampler (endo weight %.2f, ovary weight %.2f, default %.2f)",
                    endo_weight,
                    ovary_weight,
                    default_weight,
                )

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
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


def get_dataloaders_multi_dataset(
    data_root: str,
    splits: Dict[str, list],
    config: Dict[str, Any],
    datasets: List[str] = None,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders combining multiple datasets.

    Args:
        data_root: Root directory containing all datasets
        splits: Dict with train/val/test subject IDs (may include multiple dataset prefixes)
        config: Configuration dict with preprocessing, augmentation, training params
        datasets: List of dataset names to combine (e.g., ["D1_MHS", "D2_TCPW"])
                 If None, inferred from subject ID prefixes in splits
        num_workers: Number of dataloader workers

    Returns:
        Dict mapping split name to DataLoader with concatenated datasets
    """
    if datasets is None:
        # Infer datasets from subject ID prefixes
        all_subjects = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
        dataset_prefixes = set(sid.split("-")[0] for sid in all_subjects if "-" in sid)
        datasets = [f"{prefix}_MHS" if prefix == "D1" else f"{prefix}_TCPW" for prefix in dataset_prefixes]
        logger.info(f"Inferred datasets from subject IDs: {datasets}")

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

    split_summary = splits.get("split_summary")
    patient_stats = splits.get("patient_stats")
    if split_summary:
        logger.info("Patient-level split composition:")
        for split_name in ("train", "val", "test"):
            stats = split_summary.get(split_name)
            if stats:
                logger.info(
                    "  %s: %d patients (D1=%d, D2=%d, ovary+=%d, endo+=%d)",
                    split_name,
                    stats.get("num_patients", 0),
                    stats.get("d1_count", 0),
                    stats.get("d2_count", 0),
                    stats.get("has_ovary", 0),
                    stats.get("has_endo", 0),
                )
    elif patient_stats:
        logger.info("Patient-level split composition:")
        for split_name in ("train", "val", "test"):
            ids = splits.get(split_name, [])
            stats = summarize_split_counts(ids, patient_stats)
            logger.info(
                "  %s: %d patients (ovary+=%d, endo+=%d)",
                split_name,
                stats.get("num_patients", 0),
                stats.get("has_ovary", 0),
                stats.get("has_endo", 0),
            )

    logger.info("Using sequences: %s", sequences)
    if raw_structures != structures:
        logger.info("Structures (configured): %s", raw_structures)
    logger.info("Segmenting structures (canonical): %s", structures)

    aug_config = config.get("augmentation", {}).get("train", {})
    train_transform = (
        get_train_transforms(aug_config, roi_size=preprocessor.target_size)
        if aug_config
        else None
    )

    # Auto-detect which sequence labels were annotated on (per-subject)
    auto_detect_sequences = config.get("auto_detect_sequences", True)
    sequence_map = {}
    if auto_detect_sequences and len(sequences) > 1:
        from .sequence_detector import build_sequence_map_multi_dataset

        logger.info("Auto-detecting label-sequence alignment for each subject...")

        # Build dataset_map from subject IDs
        all_subjects = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
        temp_dataset_map = {}
        for sid in all_subjects:
            if "-" in sid:
                prefix = sid.split("-")[0]
                if prefix == "D1":
                    temp_dataset_map[sid] = "D1_MHS"
                elif prefix == "D2":
                    temp_dataset_map[sid] = "D2_TCPW"

        sequence_map = build_sequence_map_multi_dataset(
            data_root=Path(data_root),
            subject_ids=all_subjects,
            sequences=sequences,
            dataset_map=temp_dataset_map,
            rater_map={"D1_MHS": "r3", "D2_TCPW": None},
        )

        logger.info(
            "Sequence auto-detection complete: %d subjects mapped to optimal sequences",
            len(sequence_map)
        )
    elif len(sequences) == 1:
        logger.info("Only one sequence configured, skipping auto-detection")
    else:
        logger.info("Sequence auto-detection disabled (auto_detect_sequences=False)")

    def split_subjects_by_dataset(subject_ids: List[str]) -> Dict[str, List[str]]:
        """Separate subject IDs by dataset based on prefix."""
        dataset_subjects = {ds: [] for ds in datasets}
        for sid in subject_ids:
            if "-" in sid:
                prefix = sid.split("-")[0]
                if prefix == "D1" and "D1_MHS" in datasets:
                    dataset_subjects["D1_MHS"].append(sid)
                elif prefix == "D2" and "D2_TCPW" in datasets:
                    dataset_subjects["D2_TCPW"].append(sid)
                else:
                    logger.warning(f"Unknown dataset prefix for subject {sid}")
        return dataset_subjects

    # Create concatenated datasets for each split
    concat_datasets = {}
    for split_name in ("train", "val", "test"):
        split_ids = splits.get(split_name, [])
        dataset_subjects = split_subjects_by_dataset(split_ids)

        split_datasets = []
        for dataset_name, dataset_ids in dataset_subjects.items():
            if not dataset_ids:
                logger.info(f"No subjects from {dataset_name} in {split_name} split")
                continue

            # Use rater 1 for D1_MHS, None for others
            rater_id = "r3" if dataset_name == "D1_MHS" else None

            ds = EndoMRIDataset(
                data_root=data_root,
                subject_ids=dataset_ids,
                sequences=sequences,
                structures=structures,
                dataset_name=dataset_name,
                rater_id=rater_id,
                preprocessor=preprocessor,
                transform=train_transform if split_name == "train" else None,
                cache_data=False,
                sequence_map=sequence_map if sequence_map else None,
            )
            split_datasets.append(ds)
            logger.info(
                f"{split_name.capitalize()} split: {len(dataset_ids)} subjects from {dataset_name}"
            )

        # Concatenate datasets for this split
        if len(split_datasets) == 1:
            concat_datasets[split_name] = split_datasets[0]
        elif len(split_datasets) > 1:
            concat_datasets[split_name] = ConcatDataset(split_datasets)
        else:
            raise ValueError(f"No datasets found for {split_name} split")

    batch_size = config.get("training", {}).get("batch_size", 2)
    subject_sampler_cfg = config.get("training", {}).get("subject_sampler", {})
    train_sampler = None
    if subject_sampler_cfg.get("enabled"):
        if not patient_stats:
            logger.warning("Subject sampler enabled but patient_stats missing; falling back to uniform sampling.")
        else:
            weights: List[float] = []
            endo_weight = subject_sampler_cfg.get("endo_weight", 4.0)
            ovary_weight = subject_sampler_cfg.get("ovary_weight", 2.0)
            default_weight = subject_sampler_cfg.get("default_weight", 1.0)

            train_dataset = concat_datasets["train"]
            # Handle both single dataset and ConcatDataset
            if isinstance(train_dataset, ConcatDataset):
                # Iterate through all constituent datasets
                for ds in train_dataset.datasets:
                    for entry in ds.data_index:
                        subject_id = entry.get("subject_id")
                        stats = patient_stats.get(subject_id, {}) if subject_id else {}
                        if stats.get("has_endo"):
                            weights.append(endo_weight)
                        elif stats.get("has_ovary"):
                            weights.append(ovary_weight)
                        else:
                            weights.append(default_weight)
            else:
                for entry in train_dataset.data_index:
                    subject_id = entry.get("subject_id")
                    stats = patient_stats.get(subject_id, {}) if subject_id else {}
                    if stats.get("has_endo"):
                        weights.append(endo_weight)
                    elif stats.get("has_ovary"):
                        weights.append(ovary_weight)
                    else:
                        weights.append(default_weight)

            if weights:
                train_sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True,
                )
                logger.info(
                    "Using weighted subject sampler (endo weight %.2f, ovary weight %.2f, default %.2f)",
                    endo_weight,
                    ovary_weight,
                    default_weight,
                )

    loaders = {
        "train": DataLoader(
            concat_datasets["train"],
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            concat_datasets["val"],
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        "test": DataLoader(
            concat_datasets["test"],
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    }

    logger.info(
        "Created multi-dataset dataloaders: train=%d, val=%d, test=%d",
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
    dataset_map: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training split."""
    logger.info("Computing class weights from training data...")

    class_counts = np.zeros(num_classes)
    dataset_map = dataset_map or {}

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
        dataset_for_subject = dataset_map.get(subject_id, dataset_name)
        subject_dir = Path(data_root) / dataset_for_subject / subject_id
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
            merged_label = merge_structure_labels(label_dict, structure_to_idx, subject_id)
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
    "get_dataloaders_multi_dataset",
    "compute_class_weights",
    "InfiniteDataLoader",
    "collate_fn_with_metadata",
    "prefetch_to_device",
]
