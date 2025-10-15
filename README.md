# Uncertainty-Aware Transformer Models for Pelvic MRI Segmentation in Endometriosis

Deep learning models for pelvic MRI segmentation in endometriosis with uncertainty quantification using Transformer architectures.

## Project Overview

This project extends the work from [Liang et al. (2025)](https://www.nature.com/articles/s41597-025-05623-3) by developing uncertainty-aware Transformer-based segmentation models for the UT-EndoMRI dataset.

**Key Features:**
- Uncertainty quantification using Monte Carlo Dropout, Deep Ensembles, and Evidential Learning
- Transformer-based architecture (Swin UNETR) for improved global context
- Multi-structure segmentation (uterus, ovaries, endometriomas)
- Comprehensive preprocessing and augmentation pipeline
- Correlation analysis between uncertainty and inter-rater disagreement

## Installation

### Prerequisites
- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended: 24GB+ VRAM)
- 50GB+ free disk space for dataset

### Option 1: Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/
cd endo-uncertainty-seg

# Create conda environment
conda env create -f environment.yml
conda activate endo-uncertainty

# Install package in development mode
pip install -e .
```

### Option 2: Using pip

```bash
# Clone repository
git clone https://github.com/
cd endo-uncertainty-seg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Dataset Setup

### 1. Download UT-EndoMRI Dataset

Download the dataset from Zenodo:
- **Link:** https://zenodo.org/records/15750762
- **Size:** 8.0 GB

```bash
# Create data directory
mkdir -p data/raw

# Download and extract (manual or using wget/curl)
cd data/raw
wget https://zenodo.org/records/15750762/files/UT-EndoMRI.zip
unzip UT-EndoMRI.zip
cd ../..
```

Your data directory should now look like:
```
data/raw/UT-EndoMRI/
├── D1_MHS/          # Dataset 1: Multi-center, multi-rater
│   ├── D1-000/
│   ├── D1-001/
│   └── ...
├── D2_TCPW/         # Dataset 2: Single-center, single-rater
│   ├── D2-000/
│   ├── D2-001/
│   └── ...
└── SiteScannerInfo.csv
```

### 2. Create Data Splits

```bash
# Create train/val/test splits using paper's split
python scripts/create_splits.py \
    --data_root data/raw/UT-EndoMRI \
    --dataset D2_TCPW \
    --output data/splits/split_info.json \
    --use_paper_split

# Or create custom random splits
python scripts/create_splits.py \
    --data_root data/raw/UT-EndoMRI \
    --dataset D2_TCPW \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### 3. Explore the Dataset

```bash
# Run exploratory data analysis
python scripts/explore_data.py \
    --data_root data/raw/UT-EndoMRI \
    --dataset both \
    --output_dir data/eda_results

# This will print statistics about:
# - Number of subjects, sequences, structures
# - Image dimensions and spacing
# - Label volumes and distribution
# - Missing data analysis
```

## Usage

### Basic Dataset Loading

```python
from src.data.dataset import EndoMRIDataset
from src.data.preprocessing import MRIPreprocessor
from src.data.utils import load_data_splits

# Load splits
splits = load_data_splits("data/splits/split_info.json")

# Create preprocessor
preprocessor = MRIPreprocessor(
    target_spacing=(5.0, 5.0, 5.0),
    target_size=(128, 128, 32),
    intensity_clip_percentiles=(1, 99),
    normalize_method="min_max"
)

# Create dataset
train_dataset = EndoMRIDataset(
    data_root="data/raw/UT-EndoMRI",
    subject_ids=splits['train'],
    sequences=['T2FS'],  # T2-weighted fat suppression
    structures=['uterus', 'ovary', 'endometrioma'],
    dataset_name="D2_TCPW",
    preprocessor=preprocessor
)

# Get a sample
sample = train_dataset[0]
print(f"Image shape: {sample['image'].shape}")  # (C, H, W, D)
print(f"Label shape: {sample['label'].shape}")  # (H, W, D)
print(f"Subject ID: {sample['subject_id']}")
```

### Data Preprocessing

```python
from src.data.preprocessing import MRIPreprocessor
import numpy as np

# Initialize preprocessor
preprocessor = MRIPreprocessor(
    target_spacing=(5.0, 5.0, 5.0),
    target_size=(128, 128, 32),
    intensity_clip_percentiles=(1, 99),
    normalize_method="min_max"
)

# Preprocess image-label pair
image = np.random.randn(100, 100, 20)  # Example
label = np.random.randint(0, 4, (100, 100, 20))
original_spacing = (1.5, 1.5, 3.0)

processed_img, processed_lbl = preprocessor.preprocess_pair(
    image, label, original_spacing
)
```

## Dataset Information

### UT-EndoMRI Dataset Details

**Dataset 1 (D1_MHS):**
- 51 subjects from multiple centers
- Multi-rater annotations (up to 3 raters)
- Multiple MRI scanners (GE, Philips, Siemens)
- Field strengths: 1.5T and 3T
- Sequences: T2-weighted, T1-weighted fat suppression

**Dataset 2 (D2_TCPW):**
- 81-82 subjects from single center
- Single rater annotation
- Philips Ingenia 1.5T scanner
- Sequences: T1, T1FS, T2, T2FS
- 12 subjects with endometriomas

**Structures:**
- Uterus (ut)
- Ovary (ov)
- Endometrioma (em)
- Cyst (cy)
- Cul-de-sac (cds)

**Key Findings from Paper:**
- Inter-rater agreement (Krippendorff's α): 0.73 for uterus, 0.46 for ovaries
- Average DSC: 0.73 ± 0.18 for uterus, 0.48 ± 0.24 for ovaries
- Baseline nnU-Net: 0.272 DSC for ovaries
- RAovSeg: 0.290 DSC for ovaries

## Project Structure

```
endometriosis-uncertainty-seg/
├── configs/              # Configuration files
├── data/                 # Data directory (add to .gitignore)
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architectures
│   ├── training/        # Training logic
│   ├── inference/       # Inference pipeline
│   ├── visualization/   # Visualization utilities
│   └── utils/           # General utilities
├── scripts/             # Executable scripts
├── notebooks/           # Jupyter notebooks
├── experiments/         # Experiment outputs
└── tests/              # Unit tests
```

## Next Steps

After completing Phase 1 setup:

1. **Phase 2:** Reproduce nnU-Net baseline
2. **Phase 3:** Implement Transformer architecture (Swin UNETR)
3. **Phase 4:** Add uncertainty quantification (MC Dropout, Ensembles)
4. **Phase 5:** Training and evaluation
5. **Phase 6:** Clinical validation and analysis

## License

This project is licensed under the MIT License. The UT-EndoMRI dataset is available for free use exclusively in non-commercial scientific research.