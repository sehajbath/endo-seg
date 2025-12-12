"""
PyTorch datasets for the UT-EndoMRI collection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import MRIPreprocessor
from .structures import (
    EndoMRIDataInfo,
    canonicalize_structure_list,
    merge_structure_labels,
)
from ..io.files import (
    get_subject_data_dict,
    load_nifti,
    parse_filename,
)

logger = logging.getLogger(__name__)


class EndoMRIDataset(Dataset):
    """PyTorch dataset for UT-EndoMRI subjects.

    Each ``subject_id`` corresponds to a directory named like ``D2-012``
    under ``<data_root>/<dataset_name>``. The dataset looks up MRI sequences
    (e.g., ``T2FS``) and structure labels (``uterus``, ``ovary``, ``endometrioma``)
    using :func:`endo_seg.data.io.files.get_subject_data_dict`, ensuring that
    every returned sample contains at least one requested label volume.
    """

    def __init__(
        self,
        data_root: str,
        subject_ids: List[str],
        sequences: List[str],
        structures: List[str],
        dataset_name: str = "D2_TCPW",
        preprocessor: Optional[MRIPreprocessor] = None,
        transform: Optional[Callable] = None,
        rater_id: Optional[str] = None,
        cache_data: bool = False,
        dataset_map: Optional[Dict[str, str]] = None,
        sequence_map: Optional[Dict[str, str]] = None,
    ):
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids
        self.sequences = sequences
        self.structures = canonicalize_structure_list(structures)
        self.original_structures = list(structures)
        self.dataset_name = dataset_name
        self.preprocessor = preprocessor
        self.transform = transform
        self.rater_id = rater_id
        self.cache_data = cache_data
        # Optional mapping of subject_id -> dataset_name (for combined splits)
        self.dataset_map = dataset_map or {}
        # Optional mapping of subject_id -> best sequence for labels
        self.sequence_map = sequence_map or {}

        self.data_index = self._build_data_index()
        self.cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = {} if cache_data else None

        logger.info("Initialized dataset with %d subjects", len(self.data_index))
        logger.info("Sequences: %s", self.sequences)
        if self.original_structures != self.structures:
            logger.info("Structures (configured): %s", self.original_structures)
        logger.info("Structures (canonical): %s", self.structures)
        if self.sequence_map:
            logger.info("Using per-subject sequence selection (%d subjects mapped)", len(self.sequence_map))

    def _build_data_index(self) -> List[Dict[str, Path]]:
        data_index: List[Dict[str, Path]] = []

        for subject_id in self.subject_ids:
            dataset_for_subject = self.dataset_map.get(subject_id, self.dataset_name)
            dataset_path = self.data_root / dataset_for_subject
            subject_dir = dataset_path / subject_id
            if not subject_dir.exists():
                logger.warning("Subject directory not found: %s", subject_dir)
                continue

            data_dict = get_subject_data_dict(
                subject_dir,
                self.sequences,
                self.structures,
                self.rater_id,
            )

            # Allow inclusion if any enabled sequence is present
            available_seqs = [seq for seq in self.sequences if data_dict.get(f"image_{seq}") is not None]
            if not available_seqs:
                logger.warning("No configured sequences found for %s", subject_id)
                continue

            # Track a reference sequence for shape/spacing
            # If sequence_map is provided, use the detected sequence for this subject
            if subject_id in self.sequence_map:
                preferred_seq = self.sequence_map[subject_id]
                if preferred_seq in available_seqs:
                    data_dict["_ref_seq"] = preferred_seq
                    logger.debug(
                        f"Using {preferred_seq} as reference for {subject_id} (label alignment detected)"
                    )
                else:
                    logger.warning(
                        f"Preferred sequence {preferred_seq} not available for {subject_id}, using {available_seqs[0]}"
                    )
                    data_dict["_ref_seq"] = available_seqs[0]
            else:
                # No sequence map provided, use first available sequence
                data_dict["_ref_seq"] = available_seqs[0]

            has_label = any(
                data_dict.get(f"label_{struct}") is not None for struct in self.structures
            )
            if not has_label:
                logger.warning("No labels found for %s", subject_id)
                continue

            data_dict["subject_id"] = subject_id
            data_index.append(data_dict)

        return data_index

    def __len__(self) -> int:
        return len(self.data_index)

    def _load_image(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        data, img = load_nifti(str(file_path))
        affine = img.affine
        spacing = np.abs(np.diag(affine)[:3])
        return data, spacing

    def _load_label(self, file_path: Path) -> np.ndarray:
        data, _ = load_nifti(str(file_path))
        return data

    def _merge_labels(self, label_dict: Dict[str, np.ndarray], subject_id: Optional[str]) -> np.ndarray:
        return merge_structure_labels(label_dict, subject_id=subject_id)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_data and self.cache is not None and idx in self.cache:
            return self.cache[idx]

        data_info = self.data_index[idx]
        subject_id = data_info["subject_id"]

        # Choose reference sequence (first available) to establish shape/spacing
        ref_seq = data_info.get("_ref_seq", self.sequences[0])
        image_path = data_info.get(f"image_{ref_seq}")
        if image_path is None:
            raise ValueError(f"No available reference sequence for {subject_id}")
        image_ref, spacing = self._load_image(image_path)

        # Build a fixed-channel tensor for all configured sequences, zero-filling missing ones
        images = []
        modality_mask: List[float] = []
        for seq in self.sequences:
            seq_path = data_info.get(f"image_{seq}")
            if seq_path is not None:
                seq_data, _ = self._load_image(seq_path)
                if seq_data.shape != image_ref.shape:
                    # Harmonize shape to reference; prefer preprocessor crop/pad if available
                    if self.preprocessor is not None:
                        seq_data = self.preprocessor.crop_or_pad(seq_data, image_ref.shape)
                    else:
                        logger.warning(
                            "Image shape mismatch for %s (subject %s); expected %s got %s. Zero-filling.",
                            seq,
                            subject_id,
                            image_ref.shape,
                            seq_data.shape,
                        )
                        seq_data = np.zeros_like(image_ref)
                if seq_data.shape != image_ref.shape:
                    logger.warning(
                        "Image shape still mismatched for %s (subject %s); forcing zero-fill to %s.",
                        seq,
                        subject_id,
                        image_ref.shape,
                    )
                    seq_data = np.zeros_like(image_ref)
                images.append(seq_data)
                modality_mask.append(1.0)
            else:
                images.append(np.zeros_like(image_ref))
                modality_mask.append(0.0)

        image = np.stack(images, axis=0)

        label_dict: Dict[str, Optional[np.ndarray]] = {}
        for struct in self.structures:
            label_path = data_info.get(f"label_{struct}")
            label_dict[struct] = self._load_label(label_path) if label_path is not None else None

        label = self._merge_labels(label_dict, subject_id)

        if self.preprocessor is not None:
            processed_images = []
            for i in range(image.shape[0]):
                proc_img, proc_label = self.preprocessor.preprocess_pair(
                    image[i], label, spacing
                )
                processed_images.append(proc_img)

            image = np.stack(processed_images, axis=0)
            label = proc_label

        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).long()

        if self.transform is not None:
            sample = self.transform({"image": image_tensor, "label": label_tensor})
            # RandCropByLabelClassesd returns a list even with num_samples=1
            if isinstance(sample, list):
                sample = sample[0]
            image_tensor = sample["image"]
            label_tensor = sample["label"]

        output = {
            "image": image_tensor,
            "label": label_tensor,
            "subject_id": subject_id,
            "spacing": torch.tensor(spacing).float(),
            "modality_mask": torch.tensor(modality_mask).float(),
        }

        if self.cache_data and self.cache is not None:
            self.cache[idx] = output

        return output


class EndoMRIMultiRaterDataset(Dataset):
    """Placeholder for dataset that returns multi-rater annotations."""

    def __init__(
        self,
        data_root: str,
        subject_ids: List[str],
        sequences: List[str],
        structures: List[str],
        preprocessor: Optional[MRIPreprocessor] = None,
        transform: Optional[Callable] = None,
        return_all_raters: bool = False,
    ):
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids
        self.sequences = sequences
        self.structures = canonicalize_structure_list(structures)
        self.preprocessor = preprocessor
        self.transform = transform
        self.return_all_raters = return_all_raters

        self.data_index = self._build_data_index()

    def _build_data_index(self) -> List[Dict[str, object]]:
        data_index: List[Dict[str, object]] = []
        dataset_path = self.data_root / EndoMRIDataInfo.DATASET_1_DIR

        for subject_id in self.subject_ids:
            subject_dir = dataset_path / subject_id
            if not subject_dir.exists():
                continue

            raters = set()
            for file in subject_dir.glob("*.nii.gz"):
                info = parse_filename(file.name)
                if info["is_label"] and "rater_id" in info:
                    raters.add(info["rater_id"])

            if not raters:
                continue

            data_dict: Dict[str, object] = {
                "subject_id": subject_id,
                "raters": sorted(raters),
            }

            for seq in self.sequences:
                seq_files = list(subject_dir.glob(f"{subject_id}_{seq}.nii.gz"))
                data_dict[f"image_{seq}"] = seq_files[0] if seq_files else None

            for rater in data_dict["raters"]:
                for struct in self.structures:
                    struct_abbrev = EndoMRIDataInfo.get_structure_abbrev(struct)
                    label_files = list(
                        subject_dir.glob(f"{subject_id}_{struct_abbrev}_{rater}.nii.gz")
                    )
                    data_dict[f"label_{struct}_{rater}"] = label_files[0] if label_files else None

            data_index.append(data_dict)

        return data_index

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int):
        raise NotImplementedError("Multi-rater dataset implementation pending.")


__all__ = ["EndoMRIDataset", "EndoMRIMultiRaterDataset"]
