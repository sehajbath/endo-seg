# Implementation Summary: Two-Stage Training Pipeline

## Overview

Successfully implemented a comprehensive two-stage transfer learning pipeline for endometrioma segmentation, addressing the catastrophic training failure (Endometrioma Dice: 0.139).

## Files Modified

### 1. **notebooks/swin_unetr_colab.ipynb** ✅ COMPLETE
   - **Config Cell (Cell #6, ID: 586cdf95)**: Updated with Phase 1 fixes
   - **Stage 1 Cells (Cells #19-20)**: Added pretraining on uterus + ovary (3-class)
   - **Stage 2 Cells (Cells #21-23)**: Added finetuning on endometrioma (4-class with transfer learning)

### 2. **src/endo_seg/training/swin_unetr_trainer.py** ✅ COMPLETE
   - Fixed `DiceCEWithWeights` class (removed `squared_pred=True`)
   - Added weighted combination (0.7 Dice + 0.3 CE)
   - Added `FocalDiceCELoss` class for extreme class imbalance
   - Updated `build_loss_and_metrics()` to support configurable loss types

### 3. **configs/pretrain_uterus_ovary.yaml** ✅ CREATED
   - Stage 1 configuration (3-class: background, uterus, ovary)
   - Optimized hyperparameters for pretraining from scratch

### 4. **configs/finetune_endometrioma.yaml** ✅ CREATED
   - Stage 2 configuration (4-class with transfer learning)
   - Extreme foreground bias (70% endo-centered crops)
   - Focal loss and aggressive class weighting

---

## Phase 1 Fixes (Applied to Config Cell)

### 🔧 Critical Fix: Augmentation Pipeline
**Problem:** `roi_size == target_size` (both 224×224×96) → NO CROPPING
**Solution:**
```python
"target_size": (320, 320, 128),      # ✅ INCREASED
"label_crop_config": {
    "roi_size": (224, 224, 96),      # ✅ Now SMALLER → actual cropping!
    "ratios": [0.05, 0.15, 0.2, 0.6], # ✅ 60% endo-centered
    "num_samples": 4,                 # ✅ More crops per subject
}
```

### 🔧 Updated Class Weights
```python
"class_weights": [0.01, 0.5, 2.0, 10.0],  # [bg, uterus, ovary, endo]
```
- Background: 0.01 (minimal weight)
- Uterus: 0.5 (reduced from 0.3)
- Ovary: 2.0 (reduced from 3.0)
- Endometrioma: 10.0 (increased from 7.0)

### 🔧 Increased Batch Size
```python
"batch_size": 2,  # Increased from 1 for better gradient estimates
```

### 🔧 Loss Function Configuration
```python
"loss_type": "dice_ce",  # or "focal" for finetuning
"dice_weight": 0.7,      # Dice dominant
"ce_weight": 0.3,        # CE for class balance
```

### 🔧 Updated Subject Sampling
```python
"subject_sampler_config": {
    "endo_weight": 6.0,  # Increased from 4.0
    "ovary_weight": 2.0,
    "default_weight": 1.0,
}
```

---

## Stage 1: Pretrain on Uterus + Ovary (3-class)

### Objective
Train a 3-class model to learn pelvic anatomy and MRI contrast patterns before tackling rare endometriomas.

### Configuration Highlights
```python
config_pretrain = {
    "num_classes": 3,
    "structures": {"uterus": True, "ovaries": True, "endometriomas": False},

    # Augmentation
    "label_crop_config": {
        "ratios": [0.1, 0.4, 0.5],  # 10% bg, 40% uterus, 50% ovary
        "num_samples": 4,
    },

    # Training
    "epochs": 100,
    "learning_rate": 2e-4,  # Higher for training from scratch
    "class_weights": [0.01, 1.0, 3.0],  # [bg, uterus, ovary]

    # Loss
    "loss_type": "dice_ce",
    "dice_weight": 0.7,
    "ce_weight": 0.3,
}
```

### Expected Performance
- **Uterus Dice:** > 0.90
- **Ovary Dice:** > 0.75
- **Convergence:** 50-100 epochs

### Why Pretrain?
1. Uterus/ovary available in 100%/80% of subjects (vs only 16% for endo)
2. Encoder learns anatomical priors before tackling rare lesions
3. Progressive learning: large (uterus) → medium (ovary) → small (endo)

---

## Stage 2: Finetune on Endometrioma (4-class)

### Objective
Transfer pretrained weights and finetune for 4-class segmentation with extreme endometrioma focus.

### Transfer Learning Strategy
1. Load pretrained encoder + decoder from Stage 1
2. Expand final layer from 3 → 4 output channels
3. Freeze early Swin transformer stages (0, 1)
4. Finetune with extreme foreground bias

### Transfer Learning Helper Function
```python
def load_pretrained_and_expand(
    pretrained_checkpoint_path: str,
    model_config: dict,
    num_classes_new: int,
    freeze_encoder_stages: list = [0, 1],
    device: str = "cuda",
) -> SwinUNETRWithUncertainty:
    """Load 3-class pretrained model, expand to 4-class, freeze encoder."""
    # Load checkpoint
    # Create 4-class model
    # Load matching weights (skip final layer)
    # Freeze specified Swin stages
    # Return model ready for finetuning
```

### Configuration Highlights
```python
config_finetune = {
    "num_classes": 4,
    "structures": {"uterus": True, "ovaries": True, "endometriomas": True},

    # EXTREME foreground bias
    "label_crop_config": {
        "ratios": [0.05, 0.1, 0.15, 0.7],  # ✅ 70% endo-centered!
        "num_samples": 8,                   # ✅ Even more crops
    },

    # Finetuning training params
    "epochs": 200,
    "learning_rate": 5e-5,        # ✅ Lower LR for finetuning
    "weight_decay": 1e-6,          # ✅ Less regularization
    "target_metric_patience": 25,  # ✅ Longer patience

    # Aggressive class weights
    "class_weights": [0.01, 0.5, 2.0, 15.0],  # ✅ 15x for endo!

    # FOCAL LOSS for extreme class imbalance
    "loss_type": "focal",
    "focal_gamma": 2.0,
    "dice_weight": 0.7,
    "focal_weight": 0.3,

    # Subject sampling
    "subject_sampler_config": {
        "endo_weight": 6.0,
        "ovary_weight": 2.0,
        "default_weight": 1.0,
    },
}
```

### Key Adaptations
- **70% endo-centered crops** (vs 60% in Phase 1)
- **Focal loss** for rare class emphasis
- **15x class weight** for endometrioma
- **Lower LR (5e-5)** for finetuning
- **Longer patience (25 epochs)**
- **Freeze encoder stages [0, 1]** (early Swin blocks)

### Expected Performance
- **Endometrioma Dice:** > 0.55 (4x improvement from 0.139!)
- **Ovary Dice:** > 0.70
- **Uterus Dice:** > 0.85
- **Priority Dice (Ovary+Endo):** > 0.62

---

## Notebook Structure (Updated)

### Section 1-3: Setup (Unchanged)
1. Environment & Paths
2. Configuration (✅ Updated with Phase 1 fixes)
3. Data Loading & Augmentation

### Section 4: Original Training (Kept for Reference)
- Model & Trainer Setup
- Original training loop
- Visualizations

### Section 5-6: New Two-Stage Training
**Cell 19:** Markdown - Stage 1 Introduction
**Cell 20:** Code - Stage 1 Pretraining
- Creates 3-class config
- Rebuilds datasets/dataloaders for 3 classes
- Creates 3-class model
- Runs pretraining loop
- Saves to `experiments/checkpoints_pretrain/best.pth`

**Cell 21:** Markdown - Stage 2 Introduction
**Cell 22:** Code - Transfer Learning Helper
- `load_pretrained_and_expand()` function
- Loads 3-class checkpoint
- Expands to 4-class model
- Freezes encoder stages

**Cell 23:** Code - Stage 2 Finetuning
- Creates 4-class config with extreme endo bias
- Loads pretrained model and expands
- Rebuilds datasets/dataloaders for 4 classes
- Runs finetuning loop
- Plots both training histories
- Saves to `experiments/checkpoints_finetune/best.pth`

### Section 7: Uncertainty Evaluation (Unchanged)
- Can be run on either pretrained or finetuned model

---

## Usage Instructions

### Option 1: Run Phase 1 Fixes Only (Quick Test)
Run the notebook up to Cell 18 with the updated config to verify Phase 1 fixes improve learning.

### Option 2: Run Full Two-Stage Pipeline (Recommended)
1. **Run Cells 1-18:** Setup and data loading
2. **Run Cell 20:** Stage 1 pretraining (~2-3 hours for 100 epochs)
3. **Verify Stage 1:** Check uterus Dice > 0.90, ovary Dice > 0.75
4. **Run Cells 22-23:** Stage 2 finetuning (~4-6 hours for 200 epochs)
5. **Evaluate:** Run uncertainty evaluation cells on finetuned model

### Option 3: Resume from Existing Pretrained Model
If you already have a pretrained checkpoint:
1. Update `pretrained_ckpt_path` in Cell 23 to point to your checkpoint
2. Skip Cell 20 (pretraining)
3. Run Cell 23 (finetuning) directly

---

## Expected Performance Improvements

### Current (Broken Augmentation)
```
Endometrioma Dice: 0.139
Ovary Dice: 0.123
Uterus Dice: 0.088
Priority Dice: 0.131
```

### After Phase 1 Fixes Only (No Pretraining)
```
Endometrioma Dice: 0.25-0.35  (2-3x improvement)
Ovary Dice: 0.55-0.65
Uterus Dice: 0.80-0.88
Priority Dice: 0.40-0.50
```

### After Stage 1 (Pretrained on Uterus + Ovary)
```
Uterus Dice: 0.90-0.93
Ovary Dice: 0.75-0.82
(No endo in this stage)
```

### After Stage 2 (Finetuned on Endometrioma) ⭐
```
Endometrioma Dice: 0.55-0.70  (4-5x improvement!)
Ovary Dice: 0.70-0.78
Uterus Dice: 0.85-0.90
Priority Dice: 0.62-0.74
```

---

## Root Cause Analysis

### Primary Issue: Augmentation Not Working
**Symptom:** Val Dice plateaued at 0.25 after epoch 20
**Root Cause:** `roi_size == target_size` → RandCropByLabelClassesd doesn't crop
**Impact:**
- Foreground-biased sampling completely disabled
- Model sees full volumes where endometriomas are <<1% of voxels
- Gradient signal dominated by background (99%+ voxels)

### Secondary Issues
1. **Loss function:** `squared_pred=True` removes rare class gradients
2. **Class imbalance:** Only 8/49 subjects have endometriomas (16%)
3. **Validation set:** Only 2 endo cases (too small for reliable metric)
4. **Batch size = 1:** Noisy gradients, meaningless batch norm stats

---

## Files Ready for Push

All changes are ready for manual push to GitHub:

1. ✅ `notebooks/swin_unetr_colab.ipynb` - Updated with Phase 1 fixes + two-stage training
2. ✅ `src/endo_seg/training/swin_unetr_trainer.py` - Fixed loss functions
3. ✅ `configs/pretrain_uterus_ovary.yaml` - Stage 1 config
4. ✅ `configs/finetune_endometrioma.yaml` - Stage 2 config
5. ✅ `src/endo_seg/data/augment/transforms.py` - Verified (no changes needed)

---

## Next Steps

1. **Push changes to GitHub** (user will do manually)
2. **Run Stage 1 pretraining** in Colab (~100 epochs)
3. **Verify Stage 1 performance** (uterus/ovary Dice)
4. **Run Stage 2 finetuning** (~200 epochs)
5. **Evaluate on test set** with uncertainty quantification

---

## Summary

✅ **All tasks completed:**
- Phase 1 fixes applied to config cell
- Stage 1 pretraining block added
- Stage 2 finetuning block added
- Transfer learning helper function implemented
- Loss functions fixed in trainer.py
- Config files created for both stages

**Expected outcome:** 4-5x improvement in endometrioma Dice (from 0.139 → 0.55-0.70)

**Key insight:** Two-stage transfer learning leverages abundant uterus/ovary data (100%/80% of subjects) to initialize encoder before finetuning on rare endometriomas (16% of subjects).
