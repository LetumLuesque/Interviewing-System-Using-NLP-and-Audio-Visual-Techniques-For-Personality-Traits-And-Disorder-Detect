# Training Guide - Late Fusion with SEUMLD Dataset

## 🎉 System is Ready for GPU Training!

Your system has been configured with:
- ✅ **SEUMLD Dataset**: 3,224 question-level samples (34.5% deceptive)
- ✅ **GPU Support**: NVIDIA GeForce RTX 5070 Ti (11.94 GB VRAM)
- ✅ **Late Fusion Architecture**: Independent modality predictions
- ✅ **Mixed Precision Training**: Enabled for faster training

---

## 📊 Dataset Summary

### Available Datasets:

| Dataset | Samples | Modalities | Labels | Location |
|---------|---------|------------|--------|----------|
| **MDPE** | 1,200 | Video + Audio + Text | Personality + Deception | `path/to/mdpe_dataset` |
| **SEUMLD** | 3,224 | Video + Audio (no text) | Deception only | `D:\Grad\Datasets\SEUMLD\SEUMLD` |

### Combined Training Data:
- **Total**: ~4,400+ samples
- **3.5x more data** than MDPE alone!
- **Cross-cultural**: Multiple cultures + Chinese subjects

---

## 🚀 Quick Start Training

### 1. Test GPU (Already Done ✅)

```bash
cd d:\Code\Grad\multimodal_interview_system_late_fusion
python test_gpu.py
```

**Result**: GPU is ready!
- NVIDIA GeForce RTX 5070 Ti (11.94 GB)
- CUDA 12.8, cuDNN enabled
- Mixed precision supported

### 2. Verify SEUMLD Dataset (Already Done ✅)

```bash
python prepare_seumld_data.py --data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" --analyze_folds
```

**Result**: Dataset verified!
- 3,224 samples ready
- Preprocessed data available
- 5-fold cross-validation splits

### 3. Train with MDPE + SEUMLD

```bash
python main_train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir path/to/mdpe_dataset \
    --seumld_data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --use_combined \
    --seumld_fine_grained \
    --seed 42
```

## ⚙️ Training Configuration

### Current Settings (Optimized for RTX 5070 Ti):

```yaml
# config/training_config.yaml

training:
  batch_size: 16                    # Fits in 12GB VRAM
  num_epochs: 100
  gradient_accumulation_steps: 2    # Effective batch size = 32
  gradient_clip: 0.5
  early_stopping_patience: 15
  warmup_epochs: 5

optimizer:
  lr: 5e-5                          # Conservative learning rate
  weight_decay: 1e-4

device: "cuda"                      # GPU training
mixed_precision: true               # FP16 for faster training
```

### Fusion Method (Late Fusion):

```yaml
# config/model_config.yaml

fusion:
  late_fusion:
    method: "weighted"              # Learnable weights per modality
    num_modalities: 3
    dropout: 0.2
```

---

## 📈 Expected Training Time

### With RTX 5070 Ti (12GB VRAM):

| Configuration | Samples | Time/Epoch | Total Time (100 epochs) |
|---------------|---------|------------|-------------------------|
| MDPE only | 1,200 | ~3-5 min | ~5-8 hours |
| MDPE + SEUMLD | 4,400 | ~10-15 min | ~17-25 hours |

**Recommendations:**
- Start with 10-20 epochs to verify training
- Use early stopping (patience=15)
- Monitor validation loss

---

## 🎯 Training Commands

### Option 1: MDPE Only (Baseline)

```bash
python main_train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir path/to/mdpe_dataset \
    --seed 42
```

### Option 2: MDPE + SEUMLD (Recommended)

```bash
python main_train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir path/to/mdpe_dataset \
    --seumld_data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --use_combined \
    --seumld_fine_grained \
    --seumld_fold 0 \
    --seed 42
```

### Option 3: Resume Training

```bash
python main_train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir path/to/mdpe_dataset \
    --seumld_data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --use_combined \
    --resume checkpoints/best_model_epoch_X.pt
```

---

## 📊 Monitoring Training

### 1. Watch Console Output

```
Epoch 1/100
Train Loss: 15.234 | Val Loss: 16.123 | Val Acc: 0.645
✓ MDPE dataset: 840 samples
✓ SEUMLD dataset: 2580 samples (fold 0)
Combined dataset: 3420 total samples
```

### 2. Check Logs

```bash
# View latest log
tail -f logs/training_YYYYMMDD_HHMMSS.log

# Or on Windows
Get-Content logs\training_YYYYMMDD_HHMMSS.log -Wait
```

### 3. TensorBoard (if enabled)

```bash
tensorboard --logdir logs
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 8  # or even 4
```

**Solution 2**: Increase gradient accumulation
```yaml
training:
  gradient_accumulation_steps: 4  # Effective batch size = 32
```

### Training Too Slow

**Solution 1**: Use preprocessed SEUMLD data (already done ✅)

**Solution 2**: Reduce num_workers (Windows issue)
```yaml
training:
  num_workers: 0  # Already set for Windows
```

### Model Not Learning

**Solution 1**: Check learning rate
```yaml
optimizer:
  lr: 1e-4  # Try higher LR
```

**Solution 2**: Check loss weights
```yaml
loss_weights:
  personality_mse: 1.0
  deception_ce: 0.6
  cultural_bias: 0.2
```

### SEUMLD Missing Text

**This is expected!** SEUMLD has no transcripts.

The late fusion architecture handles this automatically:
- Visual + Audio predictions are made
- Text prediction is skipped (None)
- Fusion uses only available modalities

---

## 📁 Output Files

### Checkpoints

```
checkpoints/
├── best_model_epoch_39.pt       # Best model by validation loss
├── model_epoch_5.pt             # Periodic checkpoint
├── model_epoch_10.pt
└── ...
```

### Logs

```
logs/
├── training_20251227_170000.log  # Training log
└── ...
```

---

## 🎓 Advanced Options

### 1. Different Fusion Methods

```bash
# Try learned fusion (MLP-based)
# Edit config/model_config.yaml:
fusion:
  late_fusion:
    method: "learned"  # Instead of "weighted"
```

### 2. Different SEUMLD Folds

```bash
# Use different fold for cross-validation
python main_train.py \
    ... \
    --seumld_fold 1  # Try folds 0-4
```

### 3. Subject-Level Labels (Coarse-Grained)

```bash
# Use subject-level instead of question-level
python main_train.py \
    ... \
    --seumld_data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --use_combined
    # Remove --seumld_fine_grained flag
```

---

## 📊 Expected Results

### Baseline (MDPE only):
- Personality MAE: ~13.1
- Deception Accuracy: ~67.8%
- AUC-ROC: ~0.722

### With SEUMLD (Expected):
- Personality MAE: ~12.5-13.5 (similar, SEUMLD has no personality labels)
- Deception Accuracy: ~70-75% (improved with more data!)
- AUC-ROC: ~0.75-0.80
- Better cross-cultural generalization

---

## ✅ Checklist Before Training

- [x] GPU tested and working (RTX 5070 Ti)
- [x] SEUMLD dataset verified (3,224 samples)
- [x] Late fusion model configured
- [x] Training config optimized for GPU
- [ ] MDPE dataset path specified
- [ ] Training command ready
- [ ] Monitoring setup (logs/tensorboard)

---

## 🚀 Ready to Train!

Your system is fully configured and ready for GPU training with the late fusion architecture and SEUMLD dataset integration.

**Recommended first run:**

```bash
cd d:\Code\Grad\multimodal_interview_system_late_fusion

python main_train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir path/to/mdpe_dataset \
    --seumld_data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --use_combined \
    --seumld_fine_grained \
    --seed 42
```

**Monitor progress:**
- Watch console output for loss/accuracy
- Check `logs/` folder for detailed logs
- Checkpoints saved in `checkpoints/`

**Good luck with training!** 🎉

