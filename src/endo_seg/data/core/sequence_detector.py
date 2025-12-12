"""Automatically detect which MRI sequence labels were annotated on."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from .structures import EndoMRIDataInfo

logger = logging.getLogger(__name__)


def check_affine_alignment(
    img_nii: nib.Nifti1Image,
    label_nii: nib.Nifti1Image,
    tolerance: float = 0.01,
) -> Tuple[bool, Dict[str, any]]:
    """Check if image and label are aligned in physical space.

    Args:
        img_nii: Image NIfTI object
        label_nii: Label NIfTI object
        tolerance: Relative tolerance for spacing comparison

    Returns:
        Tuple of (is_aligned, alignment_info)
        - is_aligned: True if shape, spacing, and affine match
        - alignment_info: Dict with detailed comparison results
    """
    img_shape = img_nii.shape[:3]
    label_shape = label_nii.shape[:3]
    img_spacing = img_nii.header.get_zooms()[:3]
    label_spacing = label_nii.header.get_zooms()[:3]
    img_affine = img_nii.affine
    label_affine = label_nii.affine

    # Check shape match
    shape_match = img_shape == label_shape

    # Check spacing match (within tolerance)
    spacing_match = all(
        abs(i_s - l_s) / max(i_s, 1e-6) < tolerance
        for i_s, l_s in zip(img_spacing, label_spacing)
    )

    # Check affine match
    affine_match = np.allclose(img_affine, label_affine, atol=1e-3)

    # Overall alignment
    is_aligned = shape_match and spacing_match and affine_match

    alignment_info = {
        "shape_match": shape_match,
        "spacing_match": spacing_match,
        "affine_match": affine_match,
        "img_shape": img_shape,
        "label_shape": label_shape,
        "img_spacing": img_spacing,
        "label_spacing": label_spacing,
    }

    return is_aligned, alignment_info


def detect_label_sequence(
    subject_dir: Path,
    subject_id: str,
    sequences: List[str],
    dataset_name: str = "D1_MHS",
    rater_id: Optional[str] = None,
) -> Optional[str]:
    """Detect which sequence a label was annotated on.

    Args:
        subject_dir: Path to subject directory
        subject_id: Subject ID
        sequences: List of sequences to check (e.g., ["T1FS", "T2"])
        dataset_name: Dataset name for label file naming convention
        rater_id: Rater ID for D1_MHS (e.g., "r3")

    Returns:
        Sequence name that label aligns with, or None if no match found
    """
    # Find label file (check most common structures)
    label_file = None
    for struct_abbrev in ["em", "ov", "ut"]:  # endometrioma, ovary, uterus
        if dataset_name == "D1_MHS":
            suffix = f"_{rater_id}" if rater_id else "_r3"
            label_path = subject_dir / f"{subject_id}_{struct_abbrev}{suffix}.nii.gz"
        else:
            label_path = subject_dir / f"{subject_id}_{struct_abbrev}.nii.gz"

        if label_path.exists():
            label_file = label_path
            break

    if label_file is None:
        logger.debug(f"No label file found for {subject_id}")
        return None

    try:
        label_nii = nib.load(str(label_file))
    except Exception as e:
        logger.warning(f"Failed to load label for {subject_id}: {e}")
        return None

    # Check alignment with each sequence
    best_sequence = None
    best_score = -1

    for seq in sequences:
        img_path = subject_dir / f"{subject_id}_{seq}.nii.gz"
        if not img_path.exists():
            continue

        try:
            img_nii = nib.load(str(img_path))
            is_aligned, info = check_affine_alignment(img_nii, label_nii)

            # Score alignment (3 points max: shape + spacing + affine)
            score = sum([
                info["shape_match"],
                info["spacing_match"],
                info["affine_match"],
            ])

            if is_aligned:
                logger.debug(
                    f"{subject_id}: Label perfectly aligned with {seq} "
                    f"(shape={info['label_shape']})"
                )
                return seq  # Perfect match, return immediately

            if score > best_score:
                best_score = score
                best_sequence = seq

        except Exception as e:
            logger.warning(f"Failed to check {seq} for {subject_id}: {e}")
            continue

    # If no perfect match, return best partial match (if any criterion matched)
    if best_score > 0:
        logger.debug(
            f"{subject_id}: Label best matches {best_sequence} (score={best_score}/3)"
        )
        return best_sequence

    logger.warning(f"{subject_id}: Label doesn't align with any sequence")
    return None


def build_sequence_map(
    data_root: Path,
    subject_ids: List[str],
    sequences: List[str],
    dataset_name: str = "D1_MHS",
    rater_id: Optional[str] = None,
) -> Dict[str, str]:
    """Build mapping of subject_id -> best sequence for labels.

    Args:
        data_root: Root directory containing datasets
        subject_ids: List of subject IDs to process
        sequences: List of available sequences
        dataset_name: Dataset name (for multi-dataset, use dataset_map instead)
        rater_id: Rater ID for D1_MHS

    Returns:
        Dict mapping subject_id to sequence name
    """
    sequence_map = {}

    for subject_id in subject_ids:
        subject_dir = data_root / dataset_name / subject_id
        if not subject_dir.exists():
            continue

        best_seq = detect_label_sequence(
            subject_dir,
            subject_id,
            sequences,
            dataset_name=dataset_name,
            rater_id=rater_id,
        )

        if best_seq is not None:
            sequence_map[subject_id] = best_seq

    # Log statistics
    seq_counts = {}
    for seq in sequence_map.values():
        seq_counts[seq] = seq_counts.get(seq, 0) + 1

    logger.info(
        "Detected label-sequence alignment for %d subjects: %s",
        len(sequence_map),
        ", ".join(f"{seq}={count}" for seq, count in sorted(seq_counts.items()))
    )

    return sequence_map


def build_sequence_map_multi_dataset(
    data_root: Path,
    subject_ids: List[str],
    sequences: List[str],
    dataset_map: Dict[str, str],
    rater_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build sequence mapping for multi-dataset training.

    Args:
        data_root: Root directory containing datasets
        subject_ids: List of all subject IDs (from all datasets)
        sequences: List of available sequences
        dataset_map: Mapping of subject_id -> dataset_name
        rater_map: Optional mapping of dataset_name -> rater_id

    Returns:
        Dict mapping subject_id to best sequence
    """
    rater_map = rater_map or {"D1_MHS": "r3", "D2_TCPW": None}
    sequence_map = {}

    for subject_id in subject_ids:
        dataset_name = dataset_map.get(subject_id)
        if dataset_name is None:
            logger.warning(f"No dataset mapping for {subject_id}")
            continue

        subject_dir = data_root / dataset_name / subject_id
        if not subject_dir.exists():
            continue

        rater_id = rater_map.get(dataset_name)
        best_seq = detect_label_sequence(
            subject_dir,
            subject_id,
            sequences,
            dataset_name=dataset_name,
            rater_id=rater_id,
        )

        if best_seq is not None:
            sequence_map[subject_id] = best_seq

    # Log statistics by dataset
    dataset_seq_counts = {}
    for subject_id, seq in sequence_map.items():
        dataset_name = dataset_map[subject_id]
        if dataset_name not in dataset_seq_counts:
            dataset_seq_counts[dataset_name] = {}
        dataset_seq_counts[dataset_name][seq] = dataset_seq_counts[dataset_name].get(seq, 0) + 1

    for dataset_name, seq_counts in sorted(dataset_seq_counts.items()):
        logger.info(
            "%s label-sequence alignment: %s",
            dataset_name,
            ", ".join(f"{seq}={count}" for seq, count in sorted(seq_counts.items()))
        )

    total_count = len(sequence_map)
    logger.info(f"Total subjects with detected sequences: {total_count}")

    return sequence_map


__all__ = [
    "check_affine_alignment",
    "detect_label_sequence",
    "build_sequence_map",
    "build_sequence_map_multi_dataset",
]
