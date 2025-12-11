# Quick Start Guide: Two-Stage Training in Colab

## 🎯 Goal
Improve endometrioma segmentation from Dice 0.139 → 0.55-0.70 using two-stage transfer learning.

---

## 📋 Prerequisites

1. **Google Drive Setup:**
   - Colab notebook: `notebooks/swin_unetr_colab.ipynb`
   - Dataset: `UT-EndoMRI/` in your Google Drive
   - Repository: `endo-seg/` synced from GitHub

2. **GitHub Push Required:**
   - Push latest changes to `refactor-structure` branch
   - Colab will pull from GitHub on startup
   - Changes include fixed `trainer.py` with new loss functions

---

## 🚀 Option 1: Quick Test (Phase 1 Fixes Only)

**Estimated Time:** ~3 hours (150 epochs)

**Steps:**
1. Open `notebooks/swin_unetr_colab.ipynb` in Google Colab
2. Run cells 1-18 (setup, data loading)
3. Run cell 18 or the `run_stratified_training()` cell
4. Monitor training:
   - Should see Dice improving (not plateauing at 0.25)
   - Endometrioma Dice should reach 0.25-0.35

**What to Check:**
- Config cell shows: `target_size: (320, 320, 128)`, `roi_size: (224, 224, 96)`
- Training loss decreases steadily (not plateauing at 0.38)
- Validation Dice increases beyond 0.25

**If this works:** Proceed to Option 2 (two-stage training)

---

## 🎯 Option 2: Full Two-Stage Pipeline (Recommended)

**Estimated Time:** ~6-9 hours total
- Stage 1: 2-3 hours (100 epochs)
- Stage 2: 4-6 hours (200 epochs)

### Step 1: Setup
```python
# Run cells 1-18
# This sets up environment, loads data, creates splits
```

### Step 2: Stage 1 - Pretrain on Uterus + Ovary
```python
# Run Cell 20 (Stage 1 Pretraining)
# This will:
#   - Create 3-class config (background, uterus, ovary)
#   - Train for 100 epochs
#   - Save to experiments/checkpoints_pretrain/best.pth
```

**Monitor Stage 1:**
- Uterus Dice should reach > 0.90
- Ovary Dice should reach > 0.75
- Training should converge in 50-100 epochs

**Stage 1 Output:**
```
✅ Stage 1 pretraining complete!
Best checkpoint saved to: /content/drive/MyDrive/endo-seg/experiments/checkpoints_pretrain/best.pth
```

### Step 3: Stage 2 - Finetune on Endometrioma
```python
# Run Cells 22-23 (Transfer Learning + Finetuning)
# This will:
#   - Load pretrained 3-class model
#   - Expand to 4-class model
#   - Freeze encoder stages [0, 1]
#   - Finetune for 200 epochs with extreme endo bias
#   - Save to experiments/checkpoints_finetune/best.pth
```

**Monitor Stage 2:**
- Initial Dice should be better than random (pretrained init)
- Endometrioma Dice should reach > 0.55
- Ovary/Uterus Dice should remain high (> 0.70 / > 0.85)

**Stage 2 Output:**
```
✅ Stage 2 finetuning complete!
Best checkpoint saved to: /content/drive/MyDrive/endo-seg/experiments/checkpoints_finetune/best.pth

Training history plot saved with both stages.
```

### Step 4: Evaluation
```python
# Run uncertainty evaluation cells (Cell 21-26)
# Load the finetuned model:
best_ckpt = os.path.join(
    config["checkpoint_dir"].replace("checkpoints_swin_unetr", "checkpoints_finetune"),
    "best.pth"
)
```

---

## 🔧 Troubleshooting

### Issue: "Module 'endo_seg' not found"
**Cause:** GitHub not pulled or not installed
**Fix:**
```python
# Re-run cell 2 (git pull and pip install)
subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
%pip install -e .
```

### Issue: Loss function errors
**Cause:** Old `trainer.py` without fixed loss functions
**Fix:** Push latest changes to GitHub, then re-pull in Colab

### Issue: "FocalDiceCELoss not found" in Stage 2
**Cause:** Need updated `trainer.py` from GitHub
**Fix:**
```python
# Check if FocalDiceCELoss exists
from endo_seg.training.swin_unetr_trainer import FocalDiceCELoss
# If this fails, push updated trainer.py to GitHub and re-pull
```

### Issue: Stage 1 Dice still low (< 0.70)
**Possible causes:**
1. Augmentation still not working → check `target_size` > `roi_size`
2. Learning rate too low → try 5e-4 instead of 2e-4
3. Need more epochs → increase to 150-200

### Issue: Stage 2 not improving
**Possible causes:**
1. Pretrained model not loaded → check checkpoint path
2. Encoder not frozen → verify freeze_encoder_stages=[0,1]
3. Not enough endo bias → try ratios=[0.03, 0.07, 0.1, 0.8] (80% endo)

---

## 📊 Expected Training Curves

### Stage 1 (Pretraining)
```
Training Loss: 0.5 → 0.15 (smooth decrease)
Val Dice: 0.3 → 0.85 (steady increase)
  - Uterus: 0.5 → 0.92
  - Ovary: 0.2 → 0.78
```

### Stage 2 (Finetuning)
```
Training Loss: 0.3 → 0.18 (starts lower due to pretrain)
Val Dice: 0.5 → 0.70 (steady increase)
  - Uterus: 0.85 → 0.88 (maintains)
  - Ovary: 0.70 → 0.75 (maintains/improves)
  - Endometrioma: 0.15 → 0.60 (major improvement!)
```

---

## 💾 Checkpoints

### Stage 1 Checkpoints
**Location:** `experiments/checkpoints_pretrain/`
- `best.pth` - Best validation Dice (use for Stage 2)
- `latest.pth` - Most recent epoch

**What's saved:**
- Model state (3-class output)
- Optimizer state
- Epoch number
- Best Dice score

### Stage 2 Checkpoints
**Location:** `experiments/checkpoints_finetune/`
- `best.pth` - Best validation Dice (final model)
- `latest.pth` - Most recent epoch

**What's saved:**
- Model state (4-class output)
- Optimizer state
- Epoch number
- Best Dice score

---

## 📈 Performance Targets

### Minimum Acceptable Performance
- Endometrioma Dice: > 0.50
- Ovary Dice: > 0.65
- Uterus Dice: > 0.80

### Target Performance
- Endometrioma Dice: > 0.60
- Ovary Dice: > 0.75
- Uterus Dice: > 0.88

### Excellent Performance
- Endometrioma Dice: > 0.70
- Ovary Dice: > 0.80
- Uterus Dice: > 0.92

---

## 🔄 Alternative: Resume from Existing Checkpoint

If you have a pretrained checkpoint from a previous run:

```python
# In Cell 23 (Stage 2 Finetuning), update:
pretrained_ckpt_path = "/content/drive/MyDrive/endo-seg/experiments/checkpoints_pretrain/best.pth"

# Or use a custom path:
pretrained_ckpt_path = "/path/to/your/checkpoint.pth"

# Then skip Cell 20 and run Cells 22-23 directly
```

---

## 📝 Logging

### WandB (if enabled)
- Project: `endo-uncertainty-seg`
- Run names:
  - Stage 1: `pretrain_uterus_ovary`
  - Stage 2: `finetune_endometrioma`

### Local Logs
- TensorBoard logs: `experiments/logs_pretrain/` and `experiments/logs_finetune/`
- Training plots: Saved in checkpoint directories

---

## ⏱️ Time Estimates (A100 GPU)

| Task | Epochs | Time per Epoch | Total Time |
|------|--------|----------------|------------|
| Stage 1 Pretrain | 100 | ~90s | ~2.5 hours |
| Stage 2 Finetune | 200 | ~120s | ~6.5 hours |
| **Total** | 300 | - | **~9 hours** |

*Note: Times may vary based on GPU availability and batch size*

---

## ✅ Success Checklist

After completing both stages:

- [ ] Stage 1 uterus Dice > 0.90
- [ ] Stage 1 ovary Dice > 0.75
- [ ] Stage 2 endometrioma Dice > 0.55
- [ ] Stage 2 ovary Dice maintained > 0.70
- [ ] Stage 2 uterus Dice maintained > 0.85
- [ ] Training curves show steady improvement (not plateauing)
- [ ] Checkpoints saved successfully
- [ ] Uncertainty evaluation runs without errors

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs:** Look for error messages in Colab output
2. **Verify config:** Print `config_pretrain` and `config_finetune` to check settings
3. **Check dimensions:** Verify `target_size > roi_size` in augmentation config
4. **Monitor GPU:** Ensure GPU is allocated and not running out of memory
5. **Review plan:** Reference `/Users/sehajroopbath/.claude/plans/inherited-crafting-dragonfly.md`

---

## 🎉 Next Steps After Success

1. **Test Set Evaluation:** Run on held-out test set
2. **Uncertainty Analysis:** Analyze MC-dropout + TTA uncertainty maps
3. **Hyperparameter Tuning:**
   - Try 80% endo bias: `ratios=[0.03, 0.07, 0.1, 0.8]`
   - Try higher focal gamma: `focal_gamma=3.0`
   - Try more class weight: `class_weights=[0.01, 0.5, 2.0, 20.0]`
4. **Multi-Modal Fusion:** Enable T1, T2 sequences
5. **Dataset Expansion:** Try D1_MHS for more endo cases

---

**Good luck! 🚀**
