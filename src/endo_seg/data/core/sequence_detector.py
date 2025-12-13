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
    tolerance: float = 0.05,
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
        rater_id: Rater ID for D1_MHS (e.g., "r3"). If None, tries all available raters.

    Returns:
        Sequence name that label aligns with, or None if no match found
    """
    # Find label file (check most common structures, rater-agnostic for D1_MHS)
    label_file = None
    for struct_abbrev in ["em", "ov", "ut"]:  # endometrioma, ovary, uterus
        if dataset_name == "D1_MHS":
            # Try specified rater first, then fall back to all available raters
            raters_to_try = []
            if rater_id:
                raters_to_try.append(rater_id)
            for r in ["r3", "r2", "r1"]:
                if r != rater_id:
                    raters_to_try.append(r)

            for rater in raters_to_try:
                label_path = subject_dir / f"{subject_id}_{struct_abbrev}_{rater}.nii.gz"
                if label_path.exists():
                    label_file = label_path
                    break

            if label_file is not None:
                break
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

    # Check alignment with each sequence; prefer perfect alignment, otherwise return best partial
    best_sequence = None
    best_score = -1
    for seq in sequences:
        img_path = subject_dir / f"{subject_id}_{seq}.nii.gz"
        if not img_path.exists():
            continue

        try:
            img_nii = nib.load(str(img_path))
            is_aligned, info = check_affine_alignment(img_nii, label_nii)

            if is_aligned:
                logger.debug(
                    f"{subject_id}: Label perfectly aligned with {seq} "
                    f"(shape={info['label_shape']})"
                )
                return seq  # Perfect match, return immediately
            # Score alignment (shape + spacing + affine)
            score = sum([
                info["shape_match"],
                info["spacing_match"],
                info["affine_match"],
            ])
            if score > best_score:
                best_score = score
                best_sequence = seq

        except Exception as e:
            logger.warning(f"Failed to check {seq} for {subject_id}: {e}")
            continue

    # If no perfect match, return best partial match if any criterion matched
    if best_score > 0 and best_sequence is not None:
        logger.debug(
            f"{subject_id}: Label best matches {best_sequence} (score={best_score}/3)"
        )
        return best_sequence

    logger.warning(f"{subject_id}: Label doesn't align with any sequence")
    return None


def detect_structure_sequences(
    subject_dir: Path,
    subject_id: str,
    sequences: List[str],
    structures: List[str],
    dataset_name: str = "D1_MHS",
    rater_id: Optional[str] = None,
) -> Optional[Dict[str, any]]:
    """Detect which sequence each structure aligns with.

    Args:
        subject_dir: Path to subject directory
        subject_id: Subject ID
        sequences: List of sequences to check (e.g., ["T1FS", "T2", "T2FS"])
        structures: List of structures to check (e.g., ["uterus", "ovary", "endometrioma"])
        dataset_name: Dataset name for label file naming convention
        rater_id: Rater ID for D1_MHS (e.g., "r3"). If None, tries all available raters.

    Returns:
        Dict with:
          - "reference_sequence": str (majority vote winner)
          - "structure_sequences": Dict[structure_name, sequence_name]
          - "sequence_counts": Dict[sequence_name, int] (for debugging)
        Or None if no structures found
    """
    from .structures import EndoMRIDataInfo

    structure_sequences = {}
    sequence_votes = {}

    # Detect sequence for each structure individually
    for struct in structures:
        struct_abbrev = EndoMRIDataInfo.get_structure_abbrev(struct)

        # Find label file for this structure (rater-agnostic for D1_MHS)
        label_path = None
        actual_rater = None

        if dataset_name == "D1_MHS":
            # Try specified rater first, then fall back to all available raters
            raters_to_try = []
            if rater_id:
                raters_to_try.append(rater_id)
            for r in ["r3", "r2", "r1"]:
                if r != rater_id:
                    raters_to_try.append(r)

            for rater in raters_to_try:
                candidate_path = subject_dir / f"{subject_id}_{struct_abbrev}_{rater}.nii.gz"
                if candidate_path.exists():
                    label_path = candidate_path
                    actual_rater = rater
                    if rater != rater_id and rater_id is not None:
                        logger.debug(f"{subject_id}/{struct}: Using rater {rater} (default {rater_id} not available)")
                    break
        else:
            label_path = subject_dir / f"{subject_id}_{struct_abbrev}.nii.gz"
            if not label_path.exists():
                label_path = None

        if label_path is None or not label_path.exists():
            logger.debug(f"No label file found for {struct} in {subject_id}")
            continue

        try:
            label_nii = nib.load(str(label_path))
        except Exception as e:
            logger.warning(f"Failed to load {struct} label for {subject_id}: {e}")
            continue

        # Check alignment with each sequence
        best_seq = None
        best_score = -1

        for seq in sequences:
            img_path = subject_dir / f"{subject_id}_{seq}.nii.gz"
            if not img_path.exists():
                continue

            try:
                img_nii = nib.load(str(img_path))
                is_aligned, info = check_affine_alignment(img_nii, label_nii)

                if is_aligned:
                    logger.debug(
                        f"{subject_id}/{struct}: Label perfectly aligned with {seq}"
                    )
                    best_seq = seq
                    break  # Perfect match

                # Score alignment
                score = sum([
                    info["shape_match"],
                    info["spacing_match"],
                    info["affine_match"],
                ])
                if score > best_score:
                    best_score = score
                    best_seq = seq

            except Exception as e:
                logger.warning(f"Failed to check {seq} for {subject_id}/{struct}: {e}")
                continue

        if best_seq is not None:
            structure_sequences[struct] = best_seq
            sequence_votes[best_seq] = sequence_votes.get(best_seq, 0) + 1
            logger.debug(f"{subject_id}/{struct}: Best sequence = {best_seq} (score={best_score}/3)")

    if not structure_sequences:
        logger.warning(f"No structure alignments found for {subject_id}")
        return None

    # Choose reference sequence by majority vote
    reference_sequence = max(sequence_votes.items(), key=lambda x: x[1])[0]

    logger.info(
        f"{subject_id}: Detected sequences per structure: {structure_sequences}, "
        f"reference={reference_sequence} (votes: {sequence_votes})"
    )

    return {
        "reference_sequence": reference_sequence,
        "structure_sequences": structure_sequences,
        "sequence_counts": sequence_votes,
    }


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
    structures: Optional[List[str]] = None,
    rater_map: Optional[Dict[str, str]] = None,
    dataset_sequences: Optional[Dict[str, List[str]]] = None,
    per_structure_detection: bool = True,
) -> Dict[str, any]:
    """Build sequence mapping for multi-dataset training.

    Args:
        data_root: Root directory containing datasets
        subject_ids: List of all subject IDs (from all datasets)
        sequences: List of available sequences
        dataset_map: Mapping of subject_id -> dataset_name
        structures: List of structures to detect sequences for (for per-structure detection)
        rater_map: Optional mapping of dataset_name -> rater_id
        dataset_sequences: Optional mapping of dataset_name -> candidate sequences
        per_structure_detection: If True, detect sequence per structure;
                                 if False, use legacy single-sequence detection

    Returns:
        If per_structure_detection=True:
            Dict mapping subject_id to structure detection results with keys:
              - "reference_sequence": str (majority vote winner)
              - "structure_sequences": Dict[structure_name, sequence_name]
              - "sequence_counts": Dict[sequence_name, int]
        If per_structure_detection=False:
            Dict mapping subject_id to single sequence name (legacy behavior)
    """
    rater_map = rater_map or {"D1_MHS": "r3", "D2_TCPW": None}
    dataset_sequences = dataset_sequences or {}
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
        seq_candidates = dataset_sequences.get(dataset_name, sequences)

        if per_structure_detection and structures is not None:
            # NEW: Per-structure detection
            detection_result = detect_structure_sequences(
                subject_dir,
                subject_id,
                seq_candidates,
                structures,
                dataset_name=dataset_name,
                rater_id=rater_id,
            )
            if detection_result is not None:
                sequence_map[subject_id] = detection_result
        else:
            # Legacy: Single sequence per subject
            best_seq = detect_label_sequence(
                subject_dir,
                subject_id,
                seq_candidates,
                dataset_name=dataset_name,
                rater_id=rater_id,
            )
            if best_seq is not None:
                sequence_map[subject_id] = best_seq

    # Log statistics
    if per_structure_detection and structures is not None:
        # Per-structure detection: log reference sequences
        ref_seq_counts = {}
        for subject_id, detection in sequence_map.items():
            if isinstance(detection, dict):
                ref_seq = detection["reference_sequence"]
                ref_seq_counts[ref_seq] = ref_seq_counts.get(ref_seq, 0) + 1

        logger.info(
            f"Per-structure detection complete: {len(sequence_map)} subjects, "
            f"reference sequences: {dict(sorted(ref_seq_counts.items()))}"
        )
    else:
        # Legacy: log by dataset
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
    "detect_structure_sequences",
    "build_sequence_map",
    "build_sequence_map_multi_dataset",
]
