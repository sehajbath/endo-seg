"""
Train/val/test split utilities for UT-EndoMRI.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def create_data_splits(
    data_root: str,
    output_file: str,
    dataset_name: str = "D2_TCPW",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Create random data splits and persist them to disk."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    np.random.seed(seed)

    dataset_path = Path(data_root) / dataset_name
    subject_ids = sorted(
        subject_dir.name for subject_dir in dataset_path.iterdir() if subject_dir.is_dir()
    )

    indices = np.random.permutation(len(subject_ids))
    n_train = int(len(subject_ids) * train_ratio)
    n_val = int(len(subject_ids) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    splits = {
        "train": [subject_ids[i] for i in train_indices],
        "val": [subject_ids[i] for i in val_indices],
        "test": [subject_ids[i] for i in test_indices],
        "dataset": dataset_name,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
    }

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


__all__ = ["create_data_splits", "load_data_splits"]
