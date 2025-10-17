"""
Structure metadata and utilities for UT-EndoMRI segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np


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


def merge_structure_labels(
    label_dict: Dict[str, Optional[np.ndarray]],
    structure_to_index: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Merge structure-specific label volumes into a multi-class label map.
    """
    mapping = structure_to_index or EndoMRIDataInfo.STRUCTURE_CLASS_INDEX

    shape = None
    for label in label_dict.values():
        if label is not None:
            shape = label.shape
            break

    if shape is None:
        raise ValueError("No valid labels found to merge.")

    merged = np.zeros(shape, dtype=np.int32)

    for struct_name, label in label_dict.items():
        if label is None:
            continue

        canonical = EndoMRIDataInfo.canonical_structure_name(struct_name)
        if canonical not in mapping:
            continue

        merged[label > 0] = mapping[canonical]

    return merged


__all__ = [
    "EndoMRIDataInfo",
    "canonicalize_structure_list",
    "merge_structure_labels",
]
