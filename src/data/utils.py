"""
Data utility functions for UT-EndoMRI dataset
"""
import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EndoMRIDataInfo:
    """Constants and information about UT-EndoMRI dataset"""

    # Structure abbreviations
    STRUCTURE_ABBREV = {
        'uterus': 'ut',
        'ovary': 'ov',
        'endometrioma': 'em',
        'cyst': 'cy',
        'cul_de_sac': 'cds'
    }

    # Structure aliases to support pluralized or alternate spellings
    STRUCTURE_ALIASES = {
        'uterus': {'uterus', 'uteri'},
        'ovary': {'ovary', 'ovaries'},
        'endometrioma': {'endometrioma', 'endometriomas'},
        'cyst': {'cyst', 'cysts'},
        'cul_de_sac': {'cul_de_sac', 'cul-de-sac', 'culdesac', 'cul de sac'}
    }

    # Canonical structure -> class index mapping (background=0)
    STRUCTURE_CLASS_INDEX = {
        'uterus': 1,
        'ovary': 2,
        'endometrioma': 3,
        'cyst': 4,
        'cul_de_sac': 5
    }

    # Sequence abbreviations
    SEQUENCE_ABBREV = {
        'T1': 'T1',
        'T1FS': 'T1FS',
        'T2': 'T2',
        'T2FS': 'T2FS'
    }

    # Dataset paths
    DATASET_1_DIR = "D1_MHS"
    DATASET_2_DIR = "D2_TCPW"

    @classmethod
    def canonical_structure_name(cls, structure: str) -> str:
        """
        Convert a structure name (possibly plural or aliased) to its canonical form.
        """
        if structure is None:
            raise ValueError("Structure name cannot be None")

        structure_lower = structure.lower()

        # Accept abbreviations directly (e.g., 'ut' -> 'uterus')
        for canonical, abbrev in cls.STRUCTURE_ABBREV.items():
            if structure_lower == abbrev.lower():
                return canonical
        for canonical, aliases in cls.STRUCTURE_ALIASES.items():
            if structure_lower in aliases:
                return canonical

        raise ValueError(
            f"Unknown structure '{structure}'. Supported structures: {list(cls.STRUCTURE_ALIASES.keys())}"
        )

    @classmethod
    def structure_to_index(cls, structure: str) -> int:
        """Get class index for structure."""
        canonical = cls.canonical_structure_name(structure)
        return cls.STRUCTURE_CLASS_INDEX[canonical]

    @classmethod
    def get_structure_abbrev(cls, structure: str) -> str:
        """Get abbreviation for structure name"""
        canonical = cls.canonical_structure_name(structure)
        return cls.STRUCTURE_ABBREV.get(canonical, canonical)

    @staticmethod
    def get_sequence_abbrev(sequence: str) -> str:
        """Get abbreviation for sequence name"""
        return EndoMRIDataInfo.SEQUENCE_ABBREV.get(sequence.upper(), sequence)


def load_nifti(file_path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load NIfTI file and return data array and image object

    Args:
        file_path: Path to .nii or .nii.gz file

    Returns:
        data: Numpy array of image data
        img: NiBabel image object
    """
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise


def save_nifti(data: np.ndarray, affine: np.ndarray, file_path: str):
    """
    Save numpy array as NIfTI file

    Args:
        data: Numpy array to save
        affine: Affine transformation matrix
        file_path: Output file path
    """
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)
    logger.info(f"Saved NIfTI file to {file_path}")


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

    info = {
        'dataset_id': dataset_id,
        'patient_id': patient_id,
        'full_id': dataset_patient
    }

    is_structure_token = False

    # Check if it's a label or image
    if len(parts) >= 2:
        # Could be sequence (T1, T2, etc.) or structure (ut, ov, etc.)
        info['type'] = parts[1]
        try:
            EndoMRIDataInfo.canonical_structure_name(parts[1])
            is_structure_token = True
        except ValueError:
            is_structure_token = False

    # Check for rater ID (labels only)
    if len(parts) >= 3 and parts[2].startswith('r'):
        info['rater_id'] = parts[2]
        info['is_label'] = True
    else:
        info['is_label'] = is_structure_token

    return info


def get_subject_files(subject_dir: Path) -> Dict[str, List[Path]]:
    """
    Get all files for a subject organized by type

    Args:
        subject_dir: Path to subject directory

    Returns:
        Dictionary with 'images' and 'labels' lists
    """
    files = {'images': [], 'labels': []}

    for file in subject_dir.glob('*.nii.gz'):
        info = parse_filename(file.name)
        if info['is_label']:
            files['labels'].append(file)
        else:
            files['images'].append(file)

    return files


def get_dataset_statistics(data_root: str, dataset_name: str) -> Dict:
    """
    Calculate statistics for dataset (number of subjects, sequences, structures)

    Args:
        data_root: Root directory of UT-EndoMRI
        dataset_name: Either 'D1_MHS' or 'D2_TCPW'

    Returns:
        Dictionary with dataset statistics
    """
    dataset_path = Path(data_root) / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    stats = {
        'num_subjects': 0,
        'sequences': set(),
        'structures': set(),
        'raters': set()
    }

    # Iterate through subject directories
    for subject_dir in dataset_path.iterdir():
        if not subject_dir.is_dir():
            continue

        stats['num_subjects'] += 1
        files = get_subject_files(subject_dir)

        # Analyze images
        for img_file in files['images']:
            info = parse_filename(img_file.name)
            stats['sequences'].add(info['type'])

        # Analyze labels
        for label_file in files['labels']:
            info = parse_filename(label_file.name)
            stats['structures'].add(info['type'])
            if 'rater_id' in info:
                stats['raters'].add(info['rater_id'])

    # Convert sets to sorted lists
    stats['sequences'] = sorted(list(stats['sequences']))
    stats['structures'] = sorted(list(stats['structures']))
    stats['raters'] = sorted(list(stats['raters']))

    return stats


def create_data_splits(
        data_root: str,
        output_file: str,
        dataset_name: str = "D2_TCPW",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create train/val/test splits for the dataset

    Args:
        data_root: Root directory of UT-EndoMRI
        output_file: Path to save split information
        dataset_name: Dataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility

    Returns:
        Dictionary with train/val/test subject IDs
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    np.random.seed(seed)

    dataset_path = Path(data_root) / dataset_name

    # Get all subject IDs
    subject_ids = []
    for subject_dir in sorted(dataset_path.iterdir()):
        if subject_dir.is_dir():
            subject_ids.append(subject_dir.name)

    # Shuffle and split
    subject_ids = np.array(subject_ids)
    indices = np.random.permutation(len(subject_ids))

    n_train = int(len(subject_ids) * train_ratio)
    n_val = int(len(subject_ids) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    splits = {
        'train': subject_ids[train_indices].tolist(),
        'val': subject_ids[val_indices].tolist(),
        'test': subject_ids[test_indices].tolist(),
        'dataset': dataset_name,
        'seed': seed,
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        }
    }

    # Save splits
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Data splits saved to {output_file}")
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    return splits


def load_data_splits(split_file: str) -> Dict[str, List[str]]:
    """
    Load train/val/test splits from JSON file

    Args:
        split_file: Path to split JSON file

    Returns:
        Dictionary with split information
    """
    with open(split_file, 'r') as f:
        splits = json.load(f)
    return splits


def get_subject_data_dict(
        subject_dir: Path,
        sequences: List[str],
        structures: List[str],
        rater_id: Optional[str] = None
) -> Dict[str, Optional[Path]]:
    """
    Get dictionary mapping sequences/structures to file paths for a subject

    Args:
        subject_dir: Path to subject directory
        sequences: List of sequences to include
        structures: List of structures to include
        rater_id: Specific rater ID to use (for Dataset 1)

    Returns:
        Dictionary with file paths
    """
    data_dict = {}
    files = get_subject_files(subject_dir)

    # Get sequence files
    for seq in sequences:
        seq_file = None
        for img_file in files['images']:
            info = parse_filename(img_file.name)
            if info['type'] == seq:
                seq_file = img_file
                break
        data_dict[f'image_{seq}'] = seq_file

    # Get structure labels
    for struct in structures:
        canonical_struct = EndoMRIDataInfo.canonical_structure_name(struct)
        struct_abbrev = EndoMRIDataInfo.get_structure_abbrev(canonical_struct)
        label_file = None

        for lbl_file in files['labels']:
            info = parse_filename(lbl_file.name)
            if info['type'] == struct_abbrev:
                # Check rater if specified
                if rater_id is None or info.get('rater_id') == rater_id:
                    label_file = lbl_file
                    break

        data_dict[f'label_{canonical_struct}'] = label_file
        if canonical_struct != struct:
            data_dict[f'label_{struct}'] = label_file

    return data_dict


def canonicalize_structure_list(structures: List[str]) -> List[str]:
    """
    Return canonical structure names, preserving order and removing duplicates.
    """
    canonical_structures = []
    seen = set()

    for struct in structures:
        canonical = EndoMRIDataInfo.canonical_structure_name(struct)
        if canonical not in seen:
            seen.add(canonical)
            canonical_structures.append(canonical)

    return canonical_structures


def merge_structure_labels(
        label_dict: Dict[str, Optional[np.ndarray]],
        structure_to_index: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Merge separate structure masks into a single multi-class label volume.

    Args:
        label_dict: Mapping of structure -> label array
        structure_to_index: Optional structure -> class index override

    Returns:
        Multi-class label array.
    """
    mapping = structure_to_index or EndoMRIDataInfo.STRUCTURE_CLASS_INDEX

    # Determine output shape
    shape = None
    for label in label_dict.values():
        if label is not None:
            shape = label.shape
            break

    if shape is None:
        raise ValueError("No valid labels found to merge.")

    merged_label = np.zeros(shape, dtype=np.int32)

    for struct_name, label in label_dict.items():
        if label is None:
            continue

        canonical = EndoMRIDataInfo.canonical_structure_name(struct_name)

        if canonical not in mapping:
            logger.warning(f"No class index configured for structure '{canonical}'. Skipping.")
            continue

        mask = label > 0
        merged_label[mask] = mapping[canonical]

    return merged_label


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Example: Get dataset statistics
    data_root = "data/raw/UT-EndoMRI"

    print("Dataset 1 (D1_MHS) Statistics:")
    stats_d1 = get_dataset_statistics(data_root, "D1_MHS")
    print(json.dumps(stats_d1, indent=2))

    print("\nDataset 2 (D2_TCPW) Statistics:")
    stats_d2 = get_dataset_statistics(data_root, "D2_TCPW")
    print(json.dumps(stats_d2, indent=2))
