"""
File-system utilities for UT-EndoMRI datasets.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from ..core.structures import EndoMRIDataInfo

logger = logging.getLogger(__name__)

KNOWN_SEQUENCE_TOKENS = {
    abbrev.lower() for abbrev in EndoMRIDataInfo.SEQUENCE_ABBREV.values()
}
KNOWN_STRUCTURE_TOKENS = {
    abbrev.lower() for abbrev in EndoMRIDataInfo.STRUCTURE_ABBREV.values()
}
IGNORED_TOKENS = {"pat"}


def load_nifti(file_path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load a NIfTI file and return the numpy array and image object."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img


def save_nifti(data: np.ndarray, affine: np.ndarray, file_path: str) -> None:
    """Save a numpy array as a NIfTI file."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse UT-EndoMRI filename to extract metadata

    Format: D[dataset_id]-[patient_id]_[sequence/structure]_r[rater_id].nii.gz

    Args:
        filename: Filename to parse

    Returns:
        Dictionary with parsed components
    """
    # Remove extension
    name = filename.replace('.nii.gz', '').replace('.nii', '')
    parts = name.split('_')

    # Parse dataset and patient ID
    dataset_patient = parts[0]
    dataset_id = dataset_patient.split('-')[0]
    patient_id = dataset_patient.split('-')[1]

    info: Dict[str, Optional[str]] = {
        'dataset_id': dataset_id,
        'patient_id': patient_id,
        'full_id': dataset_patient
    }

    # Check if it's a label or image
    if len(parts) >= 2:
        # Could be sequence (T1, T2, etc.) or structure (ut, ov, etc.)
        token = parts[1].strip()
        token_lower = token.lower()
        info['type'] = token

        if token_lower in IGNORED_TOKENS:
            info['is_ignored'] = True
            info['is_label'] = False
            info['is_sequence'] = False
            info['is_structure'] = False
            info['is_unknown'] = True
            return info

        is_sequence = token_lower in KNOWN_SEQUENCE_TOKENS
        is_structure = token_lower in KNOWN_STRUCTURE_TOKENS

        info['is_sequence'] = is_sequence
        info['is_structure'] = is_structure

        if is_sequence:
            info['sequence'] = EndoMRIDataInfo.get_sequence_abbrev(token)

        if is_structure:
            canonical = EndoMRIDataInfo.canonical_structure_name(token_lower)
            info['structure'] = canonical
            info['is_label'] = True
        else:
            info['is_label'] = False
    else:
        info['is_sequence'] = False
        info['is_structure'] = False
        info['is_label'] = False

    # Check for rater ID (labels only)
    if len(parts) >= 3 and parts[2].startswith('r'):
        info['rater_id'] = parts[2]
        info['is_label'] = True

    info['is_unknown'] = not info.get('is_label', False) and not info.get('is_sequence', False)

    return info


def get_subject_files(subject_dir: Path) -> Dict[str, List[Path]]:
    """Group subject files into image and label lists."""
    files: Dict[str, List[Path]] = {"images": [], "labels": []}

    for file in subject_dir.glob("*.nii.gz"):
        info = parse_filename(file.name)
        if info.get('is_ignored'):
            continue
        files["labels" if info["is_label"] else "images"].append(file)

    return files


def get_dataset_statistics(data_root: str, dataset_name: str) -> Dict[str, object]:
    """Compute high-level dataset statistics."""
    dataset_path = Path(data_root) / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    stats = {
        "num_subjects": 0,
        "sequences": set(),
        "structures": set(),
        "raters": set(),
    }

    for subject_dir in dataset_path.iterdir():
        if not subject_dir.is_dir():
            continue

        stats["num_subjects"] += 1
        files = get_subject_files(subject_dir)

        for img_file in files["images"]:
            info = parse_filename(img_file.name)
            stats["sequences"].add(info["type"])

        for label_file in files["labels"]:
            info = parse_filename(label_file.name)
            stats["structures"].add(info["type"])
            if "rater_id" in info:
                stats["raters"].add(info["rater_id"])

    stats["sequences"] = sorted(stats["sequences"])
    stats["structures"] = sorted(stats["structures"])
    stats["raters"] = sorted(stats["raters"])
    return stats


def get_subject_data_dict(
    subject_dir: Path,
    sequences: List[str],
    structures: List[str],
    rater_id: Optional[str] = None,
) -> Dict[str, Optional[Path]]:
    """
    Map sequence/structure names to concrete file-system paths for a subject.
    """
    data_dict: Dict[str, Optional[Path]] = {}
    files = get_subject_files(subject_dir)

    for seq in sequences:
        seq_file = None
        for img_file in files["images"]:
            info = parse_filename(img_file.name)
            if info["type"] == seq:
                seq_file = img_file
                break
        data_dict[f"image_{seq}"] = seq_file

    for struct in structures:
        canonical = EndoMRIDataInfo.canonical_structure_name(struct)
        struct_abbrev = EndoMRIDataInfo.get_structure_abbrev(canonical)
        label_file = None

        for lbl_file in files["labels"]:
            info = parse_filename(lbl_file.name)
            if info["type"] == struct_abbrev:
                if rater_id is None or info.get("rater_id") == rater_id:
                    label_file = lbl_file
                    break

        data_dict[f"label_{canonical}"] = label_file
        if canonical != struct:
            data_dict[f"label_{struct}"] = label_file

    return data_dict


__all__ = [
    "load_nifti",
    "save_nifti",
    "parse_filename",
    "get_subject_files",
    "get_dataset_statistics",
    "get_subject_data_dict",
]
