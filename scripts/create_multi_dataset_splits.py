#!/usr/bin/env python3
"""Generate stratified train/val/test splits for multi-dataset training.

Combines D1_MHS and D2_TCPW datasets, filtering for T2 sequence availability
and stratifying by endometrioma presence to maintain rare class distribution.

Usage:
    python scripts/create_multi_dataset_splits.py \
        --data_root /path/to/UT-EndoMRI \
        --output data/splits/combined_d1_d2_t2_strat_seed42_t70_v15.json \
        --datasets D1_MHS D2_TCPW \
        --required_sequences T2 \
        --stratify_by endometrioma \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def scan_dataset_subjects(
    data_root: Path,
    dataset_name: str,
    required_sequences: List[str],
) -> Tuple[List[str], List[str], Dict[str, Dict]]:
    """Scan a dataset for subjects with required sequences and compute label statistics.

    Note: Subjects are included if they have *any* of the required sequences. Missing
    sequences can be zero-filled downstream for modality fusion.

    Args:
        data_root: Root directory containing dataset
        dataset_name: Name of dataset (D1_MHS or D2_TCPW)
        required_sequences: List of required sequences (e.g., ["T2"])

    Returns:
        Tuple of (valid_subjects, excluded_subjects, subject_stats)
        - valid_subjects: List of subject IDs with all required sequences
        - excluded_subjects: List of subject IDs missing sequences
        - subject_stats: Dict mapping subject_id -> {has_ovary, has_endo, dataset}
    """
    dataset_dir = data_root / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    valid_subjects = []
    excluded_subjects = []
    subject_stats = {}

    # Get all subject directories
    subject_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        # Include subject if it has any of the required sequences (not necessarily all)
        has_any_required = False
        for seq in required_sequences:
            seq_file = subject_dir / f"{subject_id}_{seq}.nii.gz"
            if seq_file.exists():
                has_any_required = True
                break

        # Determine label availability
        label_dir = subject_dir / "labels"
        has_ovary = False
        has_endo = False

        if label_dir.exists():
            # Check for ovary labels
            ovary_labels = list(label_dir.glob("*ovary*.nii.gz"))
            has_ovary = len(ovary_labels) > 0

            # Check for endometrioma labels
            endo_labels = list(label_dir.glob("*endometrioma*.nii.gz"))
            has_endo = len(endo_labels) > 0

        subject_stats[subject_id] = {
            "dataset": dataset_name,
            "has_ovary": has_ovary,
            "has_endo": has_endo,
        }

        if has_any_required:
            valid_subjects.append(subject_id)
        else:
            excluded_subjects.append(subject_id)

    logger.info(
        f"{dataset_name}: {len(valid_subjects)} valid, {len(excluded_subjects)} excluded"
    )

    return valid_subjects, excluded_subjects, subject_stats


def stratified_split(
    subject_ids: List[str],
    subject_stats: Dict[str, Dict],
    stratify_by: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Perform stratified split maintaining rare class distribution.

    Args:
        subject_ids: List of subject IDs to split
        subject_stats: Dict mapping subject_id -> {has_ovary, has_endo, dataset}
        stratify_by: "endometrioma" or "ovary"
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Create stratification labels
    if stratify_by == "endometrioma":
        strat_labels = [int(subject_stats[sid]["has_endo"]) for sid in subject_ids]
    elif stratify_by == "ovary":
        strat_labels = [int(subject_stats[sid]["has_ovary"]) for sid in subject_ids]
    else:
        raise ValueError(f"Unknown stratify_by: {stratify_by}")

    # First split: train vs (val + test)
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        subject_ids,
        strat_labels,
        test_size=(val_ratio + test_ratio),
        stratify=strat_labels,
        random_state=seed,
    )

    # Second split: val vs test
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=val_test_ratio,
        stratify=temp_labels,
        random_state=seed,
    )

    return train_ids, val_ids, test_ids


def compute_split_summary(
    split_ids: List[str],
    subject_stats: Dict[str, Dict],
) -> Dict:
    """Compute statistics for a split.

    Args:
        split_ids: List of subject IDs in the split
        subject_stats: Dict mapping subject_id -> {has_ovary, has_endo, dataset}

    Returns:
        Dict with summary statistics
    """
    summary = {
        "num_patients": len(split_ids),
        "has_ovary": sum(subject_stats[sid]["has_ovary"] for sid in split_ids),
        "has_endo": sum(subject_stats[sid]["has_endo"] for sid in split_ids),
        "d1_count": sum(subject_stats[sid]["dataset"] == "D1_MHS" for sid in split_ids),
        "d2_count": sum(subject_stats[sid]["dataset"] == "D2_TCPW" for sid in split_ids),
    }
    return summary


def create_multi_dataset_splits(
    data_root: str,
    output_path: str,
    datasets: List[str],
    required_sequences: List[str],
    stratify_by: str = "endometrioma",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Generate stratified splits across multiple datasets.

    Args:
        data_root: Root directory containing datasets
        output_path: Path to save split JSON
        datasets: List of dataset names (e.g., ["D1_MHS", "D2_TCPW"])
        required_sequences: List of required sequences (e.g., ["T2"])
        stratify_by: "endometrioma" or "ovary"
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    data_root = Path(data_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning datasets: {datasets}")
    logger.info(f"Required sequences: {required_sequences}")

    # Scan all datasets
    all_valid_subjects = []
    all_excluded_subjects = []
    all_subject_stats = {}

    for dataset_name in datasets:
        valid, excluded, stats = scan_dataset_subjects(
            data_root, dataset_name, required_sequences
        )
        all_valid_subjects.extend(valid)
        all_excluded_subjects.extend(excluded)
        all_subject_stats.update(stats)

    logger.info(f"Total valid subjects: {len(all_valid_subjects)}")
    logger.info(f"Total excluded subjects: {len(all_excluded_subjects)}")

    # Compute overall statistics
    num_ovary = sum(stats["has_ovary"] for stats in all_subject_stats.values() if stats["has_ovary"])
    num_endo = sum(stats["has_endo"] for stats in all_subject_stats.values() if stats["has_endo"])
    logger.info(f"Subjects with ovary labels: {num_ovary}")
    logger.info(f"Subjects with endometrioma labels: {num_endo}")

    # Perform stratified split
    logger.info(f"Splitting with stratification by: {stratify_by}")
    train_ids, val_ids, test_ids = stratified_split(
        all_valid_subjects,
        all_subject_stats,
        stratify_by,
        train_ratio,
        val_ratio,
        test_ratio,
        seed,
    )

    # Compute split summaries
    train_summary = compute_split_summary(train_ids, all_subject_stats)
    val_summary = compute_split_summary(val_ids, all_subject_stats)
    test_summary = compute_split_summary(test_ids, all_subject_stats)

    logger.info(f"Train: {train_summary}")
    logger.info(f"Val: {val_summary}")
    logger.info(f"Test: {test_summary}")

    # Create output JSON
    output_data = {
        "datasets": datasets,
        "sequences_required": required_sequences,
        "stratify_by": stratify_by,
        "total_subjects": len(all_valid_subjects),
        "excluded_subjects": all_excluded_subjects,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "patient_stats": all_subject_stats,
        "split_summary": {
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        },
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "seed": seed,
    }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Splits saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified multi-dataset splits for endometrioma segmentation"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing datasets (e.g., /path/to/UT-EndoMRI)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for split JSON file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["D1_MHS", "D2_TCPW"],
        help="List of datasets to combine (default: D1_MHS D2_TCPW)",
    )
    parser.add_argument(
        "--required_sequences",
        type=str,
        nargs="+",
        default=["T2"],
        help="Required sequences (default: T2)",
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        choices=["endometrioma", "ovary"],
        default="endometrioma",
        help="Stratify by structure (default: endometrioma)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    create_multi_dataset_splits(
        data_root=args.data_root,
        output_path=args.output,
        datasets=args.datasets,
        required_sequences=args.required_sequences,
        stratify_by=args.stratify_by,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
