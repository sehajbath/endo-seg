"""
Train/val/test split utilities for UT-EndoMRI.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .files import get_subject_data_dict, load_nifti

logger = logging.getLogger(__name__)


PatientStats = Dict[str, Dict[str, bool]]
SplitSummary = Dict[str, Dict[str, int]]


def _load_structure_labels(subject_dir: Path, structures: Sequence[str]) -> Dict[str, np.ndarray]:
    data_dict = get_subject_data_dict(subject_dir, sequences=[], structures=list(structures))
    label_dict: Dict[str, np.ndarray] = {}

    for struct in structures:
        label_path = data_dict.get(f"label_{struct}")
        if label_path is None:
            continue
        try:
            label_data, _ = load_nifti(str(label_path))
        except FileNotFoundError:
            logger.warning("Missing label file for %s: %s", struct, label_path)
            continue

        if label_data is not None:
            label_dict[struct] = label_data

    return label_dict


def _has_primary_sequence(subject_dir: Path, primary_sequence: Optional[str]) -> bool:
    if primary_sequence is None:
        return True

    data_dict = get_subject_data_dict(
        subject_dir,
        sequences=[primary_sequence],
        structures=[],
        rater_id=None,
    )
    return data_dict.get(f"image_{primary_sequence}") is not None


def compute_patient_label_stats(
    data_root: str,
    dataset_name: str = "D2_TCPW",
    subject_ids: Optional[Sequence[str]] = None,
    strict_shapes: bool = True,
) -> PatientStats:
    """Detect which patients contain ovary/endometrioma annotations.

    Notes
    -----
    UT-EndoMRI labels can be stored on different underlying sequences per structure
    (e.g., uterus/ovary on T2, endometrioma on T1FS in D1_MHS). These label volumes
    may therefore have different shapes/affines. For split stratification and sampling
    we only need label *presence*, so we assess each label volume independently rather
    than merging them into a single grid.
    """

    dataset_path = Path(data_root) / dataset_name
    if subject_ids is None:
        subject_ids = sorted(
            subject_dir.name for subject_dir in dataset_path.iterdir() if subject_dir.is_dir()
        )

    target_structures = ("ovary", "endometrioma")
    patient_stats: PatientStats = {}

    for subject_id in subject_ids:
        subject_dir = dataset_path / subject_id
        if not subject_dir.exists():
            logger.warning("Subject directory not found while computing stats: %s", subject_dir)
            continue

        label_dict = _load_structure_labels(subject_dir, target_structures)
        ovary = label_dict.get("ovary")
        endo = label_dict.get("endometrioma")
        patient_stats[subject_id] = {
            "has_ovary": bool(np.any(ovary > 0)) if ovary is not None else False,
            "has_endo": bool(np.any(endo > 0)) if endo is not None else False,
        }

    total_ovary = sum(1 for stats in patient_stats.values() if stats["has_ovary"])
    total_endo = sum(1 for stats in patient_stats.values() if stats["has_endo"])
    logger.info(
        "Computed patient label stats for %d subjects (ovary+=%d, endometrioma+=%d)",
        len(patient_stats),
        total_ovary,
        total_endo,
    )

    return patient_stats


def summarize_split_counts(split_ids: Sequence[str], patient_stats: PatientStats) -> Dict[str, int]:
    return {
        "num_patients": len(split_ids),
        "has_ovary": sum(
            1 for subject_id in split_ids if patient_stats.get(subject_id, {}).get("has_ovary")
        ),
        "has_endo": sum(
            1 for subject_id in split_ids if patient_stats.get(subject_id, {}).get("has_endo")
        ),
    }


def log_split_summary(split_summary: SplitSummary) -> None:
    for split_name in ("train", "val", "test"):
        stats = split_summary.get(split_name)
        if not stats:
            continue
        logger.info(
            "%s split -> %d patients (ovary+=%d, endometrioma+=%d)",
            split_name.capitalize(),
            stats.get("num_patients", 0),
            stats.get("has_ovary", 0),
            stats.get("has_endo", 0),
        )


def stratified_patient_split(
    patient_stats: PatientStats,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split patient IDs while preserving label-positive cases in each set."""

    if not patient_stats:
        raise ValueError("Patient stats dictionary is empty.")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    categories = {
        "endometrioma": [],
        "ovary": [],
        "negative": [],
    }

    for subject_id, stats in sorted(patient_stats.items()):
        if stats.get("has_endo"):
            categories["endometrioma"].append(subject_id)
        elif stats.get("has_ovary"):
            categories["ovary"].append(subject_id)
        else:
            categories["negative"].append(subject_id)

    logger.info(
        "Patient category counts (endo+=%d, ovary_only=%d, negative=%d)",
        len(categories["endometrioma"]),
        len(categories["ovary"]),
        len(categories["negative"]),
    )

    rng = np.random.default_rng(seed)

    def split_category(ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        if not ids:
            return [], [], []

        ids = list(ids)
        rng.shuffle(ids)
        desired = np.array([train_ratio, val_ratio, test_ratio]) * len(ids)
        counts = np.floor(desired).astype(int)
        remainder = len(ids) - counts.sum()
        if remainder > 0:
            fractions = desired - counts
            order = np.argsort(-fractions)
            for idx in order[:remainder]:
                counts[idx] += 1

        train_end = counts[0]
        val_end = train_end + counts[1]
        return ids[:train_end], ids[train_end:val_end], ids[val_end:]

    splits = {"train": [], "val": [], "test": []}
    for cat_ids in categories.values():
        train_ids, val_ids, test_ids = split_category(cat_ids)
        splits["train"].extend(train_ids)
        splits["val"].extend(val_ids)
        splits["test"].extend(test_ids)

    for key in splits:
        rng.shuffle(splits[key])

    return splits["train"], splits["val"], splits["test"]


def create_data_splits(
    data_root: str,
    output_file: str,
    dataset_name: str = "D2_TCPW",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratified: bool = True,
    primary_sequence: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Create (optionally stratified) data splits and persist them to disk."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    dataset_path = Path(data_root) / dataset_name
    subject_ids: List[str] = []
    for subject_dir in sorted(dataset_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        if not _has_primary_sequence(subject_dir, primary_sequence):
            logger.warning(
                "Skipping %s because sequence %s is unavailable",
                subject_dir.name,
                primary_sequence,
            )
            continue
        subject_ids.append(subject_dir.name)

    split_summary: Optional[SplitSummary] = None
    patient_stats: Optional[PatientStats] = None

    if stratified:
        patient_stats = compute_patient_label_stats(
            data_root=data_root,
            dataset_name=dataset_name,
            subject_ids=subject_ids,
        )
        train_ids, val_ids, test_ids = stratified_patient_split(
            patient_stats,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        split_summary = {
            "train": summarize_split_counts(train_ids, patient_stats),
            "val": summarize_split_counts(val_ids, patient_stats),
            "test": summarize_split_counts(test_ids, patient_stats),
        }
        log_split_summary(split_summary)
    else:
        np.random.seed(seed)
        indices = np.random.permutation(len(subject_ids))
        n_train = int(len(subject_ids) * train_ratio)
        n_val = int(len(subject_ids) * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        train_ids = [subject_ids[i] for i in train_indices]
        val_ids = [subject_ids[i] for i in val_indices]
        test_ids = [subject_ids[i] for i in test_indices]

    splits: Dict[str, List[str]] = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "dataset": dataset_name,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "stratified": stratified,
    }

    if patient_stats is not None:
        splits["patient_stats"] = patient_stats
    if split_summary is not None:
        splits["split_summary"] = split_summary

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        "Saved data splits to %s (train=%d, val=%d, test=%d)",
        output_file,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits


def load_data_splits(split_file: str) -> Dict[str, List[str]]:
    """Load data split metadata from disk."""
    with open(split_file, "r") as f:
        return json.load(f)


__all__ = [
    "create_data_splits",
    "load_data_splits",
    "compute_patient_label_stats",
    "stratified_patient_split",
    "summarize_split_counts",
    "log_split_summary",
]
