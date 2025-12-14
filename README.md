# Uncertainty-Aware Multi‑Modal SwinUNETR for UT‑EndoMRI Pelvic Segmentation

This repository contains an end‑to‑end training pipeline for pelvic MRI segmentation on **UT‑EndoMRI** using a MONAI **SwinUNETR** backbone, with utilities for uncertainty‑aware inference.

## Project Overview (Current State)

This project builds on UT‑EndoMRI (Liang et al., 2025) and focuses on **multi‑modal** training across **D1_MHS + D2_TCPW** while handling real‑world missing modalities and label/sequence misalignment.

**What’s implemented and actively used**
- **Multi‑modal SwinUNETR** training with **fixed** `in_channels=4` and canonical modality order: `T1, T1FS, T2, T2FS`
- **Missing modality support** per patient via **sentinel fill** (`missing_modality_value`, default `-1.0`) + `modality_mask`
- **Per‑subject label↔sequence alignment detection** (optional) with **per‑structure** detection and a voted reference sequence
- **Affine‑aware label resampling** when labels are annotated in different sequence spaces
- Two‑stage recipe used in the notebook:
  - **Stage 1 pretrain** (3‑class: background/uterus/ovary)
  - **Stage 2 finetune** (4‑class: + endometrioma)

## Documentation

- `UPDATED_PIPELINE_DATA_FLOW.md` — report‑ready end‑to‑end data flow (splits → dataset → transforms → training)
- `notebooks/swin_unetr_colab.ipynb` — reference run pipeline (D1+D2, multi‑modal, pretrain → finetune)

## Installation

### Prerequisites
- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended: 24GB+ VRAM)
- 50GB+ free disk space for dataset

### Option 1: Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/sehajbath/endo-seg
cd endo-seg

# Create conda environment
conda env create -f environment.yml
conda activate endo-uncertainty

# Install package in development mode
pip install -e .
```

### Option 2: Using pip

```bash
# Clone repository
git clone https://github.com/sehajbath/endo-seg
cd endo-seg

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

The data directory should now look like:
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
# Create train/val/test splits for a single dataset (D2_TCPW example)
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

## Usage

### Quick Start (Notebook, D1+D2 Multi‑Modal)

The most up‑to‑date reference pipeline is the notebook:
- `notebooks/swin_unetr_colab.ipynb`

It builds combined D1+D2 splits, constructs multi‑dataset dataloaders, and runs:
- Stage 1 pretraining (uterus+ovary)
- Stage 2 finetuning (+endometrioma)

### Basic Dataset Loading (Single Dataset)

```python
from endo_seg.data import EndoMRIDataset, MRIPreprocessor
from endo_seg.data import load_data_splits

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
    # Multi-modal runs use the canonical order: ["T1","T1FS","T2","T2FS"]
    sequences=['T2FS'],
    structures=['uterus', 'ovary', 'endometrioma'],
    dataset_name="D2_TCPW",
    preprocessor=preprocessor,
    missing_modality_value=-1.0,
)

# Get a sample
sample = train_dataset[0]
print(f"Image shape: {sample['image'].shape}")  # (C, H, W, D)
print(f"Label shape: {sample['label'].shape}")  # (H, W, D)
print(f"Subject ID: {sample['subject_id']}")
print(f"Modality mask: {sample['modality_mask']}")  # 1=present+valid, 0=missing/invalid
```

### Training (CLI, Single Dataset)

`scripts/train_swin_unetr.py` supports end‑to‑end training driven by YAML config (single dataset):

```bash
python scripts/train_swin_unetr.py --config configs/config.yaml --run-name swin_unetr_run
```

For combined D1+D2 multi‑modal training, use `notebooks/swin_unetr_colab.ipynb` as the reference pipeline.

### Key Multi‑Modal Concepts

**Canonical channel order**
- Multi‑modal training assumes channels are always ordered as: `T1, T1FS, T2, T2FS`.

**Missing modalities**
- Per patient, any missing/invalid modality channel is filled with a constant sentinel
  (`missing_modality_value`, default `-1.0`) and recorded in `modality_mask`.

**Label/sequence alignment**
- When enabled, the pipeline can auto‑detect which image sequence each label aligns to and means labels can be resampled into a reference space using affine information.

## Configuration Files

- `configs/config.yaml` — baseline config for CLI training
- `configs/pretrain_uterus_ovary.yaml` — stage‑1 style pretraining template
- `configs/finetune_endometrioma.yaml` — stage‑2 style finetuning template
- `configs/model_config.yaml` — model defaults

## Project Structure

```
endo-seg/
├── configs/              # YAML configs
├── data/                 # Data directory (not committed)
├── notebooks/            # Reference notebooks (Colab pipeline)
├── scripts/              # CLI utilities and training entrypoints
├── src/endo_seg/         # Installable package
│   ├── config/           # Config loading/merging
│   ├── data/             # IO, datasets, preprocessing, augmentation
│   ├── models/           # SwinUNETR wrapper + utilities
│   └── training/         # Training loops, losses, metrics, checkpointing
└── experiments/          # Output checkpoints/logs
```

## License

This project is licensed under the MIT License. The UT-EndoMRI dataset is available for free use exclusively in non-commercial scientific research.
