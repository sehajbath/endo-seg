"""
PyTorch Dataset classes for UT-EndoMRI
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging

from .utils import (
    load_nifti,
    parse_filename,
    get_subject_data_dict,
    EndoMRIDataInfo,
    merge_structure_labels,
    canonicalize_structure_list
)
from .preprocessing import MRIPreprocessor

logger = logging.getLogger(__name__)


class EndoMRIDataset(Dataset):
    """PyTorch Dataset for UT-EndoMRI"""

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
            cache_data: bool = False
    ):
        """
        Initialize EndoMRI Dataset

        Args:
            data_root: Root directory of UT-EndoMRI
            subject_ids: List of subject IDs to include
            sequences: List of MRI sequences to use (e.g., ['T2FS'])
            structures: List of structures to segment (e.g., ['uterus', 'ovary'])
            dataset_name: 'D1_MHS' or 'D2_TCPW'
            preprocessor: MRIPreprocessor instance
            transform: Optional augmentation transforms
            rater_id: Specific rater ID for Dataset 1 (e.g., 'r1')
            cache_data: Whether to cache preprocessed data in memory
        """
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

        # Build data index
        self.data_index = self._build_data_index()

        # Cache for preprocessed data
        self.cache = {} if cache_data else None

        logger.info(f"Initialized dataset with {len(self.data_index)} subjects")
        logger.info(f"Sequences: {self.sequences}")
        if self.original_structures != self.structures:
            logger.info(f"Structures (input): {self.original_structures}")
        logger.info(f"Structures (canonical): {self.structures}")

    def _build_data_index(self) -> List[Dict]:
        """Build index of all data files"""
        data_index = []
        dataset_path = self.data_root / self.dataset_name

        for subject_id in self.subject_ids:
            subject_dir = dataset_path / subject_id

            if not subject_dir.exists():
                logger.warning(f"Subject directory not found: {subject_dir}")
                continue

            # Get file paths for this subject
            data_dict = get_subject_data_dict(
                subject_dir,
                self.sequences,
                self.structures,
                self.rater_id
            )

            # Check if primary sequence exists
            primary_seq = self.sequences[0]
            if data_dict[f'image_{primary_seq}'] is None:
                logger.warning(f"Primary sequence {primary_seq} not found for {subject_id}")
                continue

            # Check if at least one label exists
            has_label = any(
                data_dict.get(f'label_{struct}') is not None
                for struct in self.structures
            )

            if not has_label:
                logger.warning(f"No labels found for {subject_id}")
                continue

            data_dict['subject_id'] = subject_id
            data_index.append(data_dict)

        return data_index

    def __len__(self) -> int:
        return len(self.data_index)

    def _load_image(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and return data with affine matrix"""
        data, img = load_nifti(str(file_path))
        affine = img.affine
        spacing = np.abs(np.diag(affine)[:3])
        return data, spacing

    def _load_label(self, file_path: Path) -> np.ndarray:
        """Load label mask"""
        data, _ = load_nifti(str(file_path))
        return data

    def _merge_labels(self, label_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Merge multiple structure labels into single multi-class mask."""
        return merge_structure_labels(label_dict)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index"""
        # Check cache
        if self.cache_data and idx in self.cache:
            return self.cache[idx]

        data_info = self.data_index[idx]
        subject_id = data_info['subject_id']

        # Load primary sequence image
        primary_seq = self.sequences[0]
        image_path = data_info[f'image_{primary_seq}']
        image, spacing = self._load_image(image_path)

        # Load additional sequences if specified
        images = [image]
        for seq in self.sequences[1:]:
            seq_path = data_info[f'image_{seq}']
            if seq_path is not None:
                seq_data, _ = self._load_image(seq_path)
                images.append(seq_data)
            else:
                # If sequence missing, use zeros
                images.append(np.zeros_like(image))

        # Stack sequences along channel dimension
        if len(images) > 1:
            image = np.stack(images, axis=0)  # Shape: (C, H, W, D)
        else:
            image = image[np.newaxis, ...]  # Shape: (1, H, W, D)

        # Load labels
        label_dict = {}
        for struct in self.structures:
            label_path = data_info.get(f'label_{struct}')
            if label_path is not None:
                label_dict[struct] = self._load_label(label_path)
            else:
                label_dict[struct] = None

        # Merge labels into single mask
        label = self._merge_labels(label_dict)

        # Apply preprocessing
        if self.preprocessor is not None:
            processed_images = []
            for i in range(image.shape[0]):
                proc_img, proc_label = self.preprocessor.preprocess_pair(
                    image[i], label, spacing
                )
                processed_images.append(proc_img)

            image = np.stack(processed_images, axis=0)
            label = proc_label

        # Convert to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        # Apply transforms/augmentation
        if self.transform is not None:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            image = sample['image']
            label = sample['label']

        output = {
            'image': image,
            'label': label,
            'subject_id': subject_id,
            'spacing': torch.tensor(spacing).float()
        }

        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = output

        return output


class EndoMRIMultiRaterDataset(Dataset):
    """Dataset for multi-rater annotations (Dataset 1)"""

    def __init__(
            self,
            data_root: str,
            subject_ids: List[str],
            sequences: List[str],
            structures: List[str],
            preprocessor: Optional[MRIPreprocessor] = None,
            transform: Optional[Callable] = None,
            return_all_raters: bool = False
    ):
        """
        Initialize multi-rater dataset

        Args:
            data_root: Root directory
            subject_ids: Subject IDs
            sequences: MRI sequences
            structures: Structures to segment
            preprocessor: Preprocessor
            transform: Augmentation
            return_all_raters: If True, return all rater annotations
        """
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids
        self.sequences = sequences
        self.structures = canonicalize_structure_list(structures)
        self.preprocessor = preprocessor
        self.transform = transform
        self.return_all_raters = return_all_raters

        self.data_index = self._build_data_index()

    def _build_data_index(self) -> List[Dict]:
        """Build index with all raters"""
        data_index = []
        dataset_path = self.data_root / "D1_MHS"

        for subject_id in self.subject_ids:
            subject_dir = dataset_path / subject_id

            if not subject_dir.exists():
                continue

            # Find all raters for this subject
            raters = set()
            for file in subject_dir.glob('*.nii.gz'):
                info = parse_filename(file.name)
                if info['is_label'] and 'rater_id' in info:
                    raters.add(info['rater_id'])

            if not raters:
                continue

            data_dict = {
                'subject_id': subject_id,
                'raters': sorted(list(raters))
            }

            # Get image paths
            for seq in self.sequences:
                seq_files = list(subject_dir.glob(f"{subject_id}_{seq}.nii.gz"))
                data_dict[f'image_{seq}'] = seq_files[0] if seq_files else None

            # Get label paths for each rater
            for rater in data_dict['raters']:
                for struct in self.structures:
                    struct_abbrev = EndoMRIDataInfo.get_structure_abbrev(struct)
                    label_files = list(subject_dir.glob(
                        f"{subject_id}_{struct_abbrev}_{rater}.nii.gz"
                    ))
                    data_dict[f'label_{struct}_{rater}'] = label_files[0] if label_files else None

            data_index.append(data_dict)

        return data_index

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Dict:
        """Get item with multi-rater labels"""
        # Implementation similar to EndoMRIDataset but returns all rater labels
        # Left as exercise - can be implemented based on needs
        raise NotImplementedError("Multi-rater dataset implementation pending")


if __name__ == "__main__":
    # Example usage
    from .utils import load_data_splits

    logging.basicConfig(level=logging.INFO)

    # Load splits
    splits = load_data_splits("data/splits/split_info.json")

    # Create preprocessor
    preprocessor = MRIPreprocessor(
        target_spacing=(5.0, 5.0, 5.0),
        target_size=(128, 128, 32)
    )

    # Create dataset
    train_dataset = EndoMRIDataset(
        data_root="data/raw/UT-EndoMRI",
        subject_ids=splits['train'],
        sequences=['T2FS'],
        structures=['uterus', 'ovary', 'endometrioma'],
        dataset_name="D2_TCPW",
        preprocessor=preprocessor
    )

    print(f"Dataset size: {len(train_dataset)}")

    # Get sample
    sample = train_dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label shape: {sample['label'].shape}")
    print(f"Subject ID: {sample['subject_id']}")
