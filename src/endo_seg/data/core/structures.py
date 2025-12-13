"""
Structure metadata and utilities for UT-EndoMRI segmentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage

logger = logging.getLogger(__name__)


class EndoMRIDataInfo:
    """Constants and structure metadata for the UT-EndoMRI dataset."""

    STRUCTURE_ABBREV: Dict[str, str] = {
        "uterus": "ut",
        "ovary": "ov",
        "endometrioma": "em",
        "cyst": "cy",
        "cul_de_sac": "cds",
    }

    STRUCTURE_ALIASES: Dict[str, Set[str]] = {
        "uterus": {"uterus", "uteri"},
        "ovary": {"ovary", "ovaries"},
        "endometrioma": {"endometrioma", "endometriomas"},
        "cyst": {"cyst", "cysts"},
        "cul_de_sac": {"cul_de_sac", "cul-de-sac", "culdesac", "cul de sac"},
    }

    # Global label indices used across datasets, referenced by patient stats,
    # samplers, and training scripts to keep label IDs consistent.
    STRUCTURE_CLASS_INDEX: Dict[str, int] = {
        "uterus": 1,
        "ovary": 2,
        "endometrioma": 3,
        "cyst": 4,
        "cul_de_sac": 5,
    }

    SEQUENCE_ABBREV: Dict[str, str] = {
        "T1": "T1",
        "T1FS": "T1FS",
        "T2": "T2",
        "T2FS": "T2FS",
    }

    DATASET_1_DIR = "D1_MHS"
    DATASET_2_DIR = "D2_TCPW"

    @classmethod
    def canonical_structure_name(cls, structure: str) -> str:
        """Return canonical structure name for a given alias."""
        if structure is None:
            raise ValueError("Structure name cannot be None")

        structure_lower = structure.lower()

        for canonical, abbrev in cls.STRUCTURE_ABBREV.items():
            if structure_lower == abbrev.lower():
                return canonical

        for canonical, aliases in cls.STRUCTURE_ALIASES.items():
            if structure_lower in aliases:
                return canonical

        raise ValueError(
            f"Unknown structure '{structure}'. "
            f"Supported structures: {list(cls.STRUCTURE_ALIASES.keys())}"
        )

    @classmethod
    def structure_to_index(cls, structure: str) -> int:
        """Return class index for canonical structure."""
        canonical = cls.canonical_structure_name(structure)
        return cls.STRUCTURE_CLASS_INDEX[canonical]

    @classmethod
    def get_structure_abbrev(cls, structure: str) -> str:
        """Return UT-EndoMRI abbreviation for a structure."""
        canonical = cls.canonical_structure_name(structure)
        return cls.STRUCTURE_ABBREV.get(canonical, canonical)

    @staticmethod
    def get_sequence_abbrev(sequence: str) -> str:
        """Return UT-EndoMRI abbreviation for an MRI sequence."""
        return EndoMRIDataInfo.SEQUENCE_ABBREV.get(sequence.upper(), sequence)


def canonicalize_structure_list(structures: List[str]) -> List[str]:
    """
    Canonicalize and deduplicate structure names while preserving order.
    """
    canonical_structures: List[str] = []
    seen: Set[str] = set()

    for struct in structures:
        canonical = EndoMRIDataInfo.canonical_structure_name(struct)
        if canonical not in seen:
            seen.add(canonical)
            canonical_structures.append(canonical)

    return canonical_structures


def _resize_label(label: np.ndarray, target_shape: np.ndarray) -> np.ndarray:
    zoom = [t / s for s, t in zip(label.shape, target_shape)]
    if any(z <= 0 for z in zoom):
        raise ValueError(f"Invalid zoom factors {zoom} for shapes {label.shape}->{target_shape}")
    return ndimage.zoom(label, zoom=zoom, order=0, mode="nearest")


def resample_label_to_reference(
    label: np.ndarray,
    source_affine: np.ndarray,
    target_affine: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Resample label from source sequence space to target sequence space.

    Uses affine transformation to properly handle different voxel sizes and orientations.

    Args:
        label: Label array from source sequence
        source_affine: 4x4 affine matrix of source sequence
        target_affine: 4x4 affine matrix of target (reference) sequence
        target_shape: Target shape (H, W, D)

    Returns:
        Resampled label array at target shape
    """
    # Create NIfTI object for source label
    source_nii = nib.Nifti1Image(label.astype(np.int16), source_affine)

    # Create dummy target NIfTI with desired shape/affine
    target_nii = nib.Nifti1Image(np.zeros(target_shape, dtype=np.int16), target_affine)

    # Resample using nibabel's affine-aware resampling
    resampled_nii = resample_from_to(
        source_nii,
        target_nii,
        order=0,  # Nearest neighbor for labels
        mode='constant',
        cval=0
    )

    return resampled_nii.get_fdata().astype(label.dtype)


def merge_structure_labels(
    label_dict: Dict[str, Optional[np.ndarray]],
    structure_to_index: Optional[Dict[str, int]] = None,
    subject_id: Optional[str] = None,
    strict_shapes: bool = False,
    resize_tolerance: float = 0.2,
    affine_dict: Optional[Dict[str, np.ndarray]] = None,
    reference_affine: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Merge structure-specific label volumes into a multi-class label map.

    Args:
        label_dict: Dict mapping structure names to label arrays
        structure_to_index: Optional mapping from structure name to class index
        subject_id: Optional subject ID for logging
        strict_shapes: If True, skip structures with mismatched shapes
        resize_tolerance: Relative shape tolerance before skipping (if not strict)
        affine_dict: Optional dict mapping structure names to their affine matrices
        reference_affine: Optional affine matrix of reference sequence

    Returns:
        Merged multi-class label array
    """
    mapping = structure_to_index or EndoMRIDataInfo.STRUCTURE_CLASS_INDEX

    shape = None
    for label in label_dict.values():
        if label is not None:
            shape = np.array(label.shape, dtype=np.int64)
            break

    if shape is None:
        raise ValueError("No valid labels found to merge.")

    merged = np.zeros(tuple(shape), dtype=np.int32)

    for struct_name, label in label_dict.items():
        if label is None:
            continue

        canonical = EndoMRIDataInfo.canonical_structure_name(struct_name)
        if canonical not in mapping:
            continue

        if label.shape != tuple(shape):
            # Try affine-aware resampling if affine info provided
            if affine_dict is not None and reference_affine is not None and struct_name in affine_dict:
                logger.info(
                    f"Resampling {struct_name}"
                    + (f" (subject {subject_id})" if subject_id else "")
                    + f" from {label.shape} to {tuple(shape)} using affine transformation"
                )
                label = resample_label_to_reference(
                    label,
                    source_affine=affine_dict[struct_name],
                    target_affine=reference_affine,
                    target_shape=tuple(shape),
                )
            else:
                # Fall back to scipy zoom (legacy behavior)
                msg = (
                    f"Label shape mismatch for {struct_name}"
                    + (f" (subject {subject_id})" if subject_id else "")
                    + f" (got {label.shape}, expected {tuple(shape)})"
                )
                # Compute relative shape difference; if too large, skip
                rel_diff = max(abs(a - b) / max(b, 1) for a, b in zip(label.shape, shape))
                if strict_shapes or rel_diff > resize_tolerance:
                    logger.warning(msg + "; skipping structure")
                    continue
                else:
                    affine_note = " (NO affine)" if affine_dict is not None else ""
                    logger.warning(msg + f"; resizing with nearest neighbor{affine_note}")
                    label = _resize_label(label, shape)

        merged[label > 0] = mapping[canonical]

    return merged


__all__ = [
    "EndoMRIDataInfo",
    "canonicalize_structure_list",
    "resample_label_to_reference",
    "merge_structure_labels",
]
