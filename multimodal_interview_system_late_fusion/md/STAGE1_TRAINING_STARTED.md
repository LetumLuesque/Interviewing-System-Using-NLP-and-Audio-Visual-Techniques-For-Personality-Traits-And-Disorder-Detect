# Stage 1 Training Started

## ✅ Setup Complete

**Date**: January 12, 2026  
**Stage**: Stage 1 (MDPE Only - Baseline)  
**Status**: Training initiated

---

## Configuration

### Dataset
- **MDPE Dataset**: `D:\Grad\Datasets\MDPE-Dataset`
- **Pre-extracted Features**: `features/mdpe/` (936 video, 936 audio, 936 text features)
- **Using Pre-extracted**: ✅ YES (saves significant time)

### Training Config
- **Config File**: `config/training_config_stage1.yaml`
- **Model Config**: `config/model_config.yaml`
- **Batch Size**: 24 (effective batch size: 48 with gradient accumulation)
- **Epochs**: 50
- **Learning Rate**: 1e-4
- **Early Stopping**: Patience = 10 epochs
- **Mixed Precision**: ✅ Enabled (FP16)

### Hardware
- **GPU**: RTX 5070 Ti (12GB VRAM)
- **Workers**: 2 (Windows multiprocessing)
- **Persistent Workers**: ✅ Enabled (faster data loading)

---

## Command Used

```bash
python train_staged.py \
    --stage 1 \
    --data_dir "D:\Grad\Datasets\MDPE-Dataset" \
    --seed 42 \
    --features_dir features
```

---

## Expected Results

### Training Time
- **Estimated**: 5-8 hours (with pre-extracted features)
- **Without features**: ~15-20 hours

### Performance Targets
- **Personality MAE**: ~13.1
- **Deception Accuracy**: ~67.8%
- **AUC-ROC**: ~0.722

---

## Monitoring

### Check Logs
```bash
# View latest log
Get-Content logs\training_*.log -Wait -Tail 50

# Or find latest
Get-ChildItem logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### Check Checkpoints
```bash
# List checkpoints
Get-ChildItem checkpoints\*.pt

# Best model will be saved as:
# checkpoints/best_model_epoch_X.pt
```

### Monitor GPU
```bash
# Windows PowerShell
while ($true) { nvidia-smi; Start-Sleep -Seconds 2; Clear-Host }
```

---

## Next Steps After Stage 1

Once Stage 1 completes:

1. **Evaluate Results**: Check validation metrics
2. **Review Checkpoint**: Best model saved automatically
3. **Proceed to Stage 2**: 
   ```bash
   python train_staged.py \
       --stage 2 \
       --data_dir "D:\Grad\Datasets\MDPE-Dataset" \
       --seumld_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
       --seed 42 \
       --features_dir features
   ```

---

## Notes

- ✅ All old checkpoints removed
- ✅ Pre-extracted features enabled (faster training)
- ✅ Stage 1 config optimized for MDPE-only training
- ✅ **Loss Weights Optimized for Personality Focus**:
  - **Personality Weight**: 3.0 (high priority - Stage 1 focuses on personality detection)
  - **Deception Weight**: 0.2 (low priority - deception will be improved in Stage 2/3)
  - **Cultural Bias**: 0.1 (lower weight for Stage 1)
- ⚠️ **Data Balancing Issue**: Stage 1 does NOT balance classes (see `DATA_BALANCING_ISSUES.md`)
  - This may cause bias toward majority class
  - Will be addressed in future updates

---

## Troubleshooting

### Training Stopped Unexpectedly
```bash
# Resume from latest checkpoint
python train_staged.py \
    --stage 1 \
    --data_dir "D:\Grad\Datasets\MDPE-Dataset" \
    --resume checkpoints/best_model_epoch_X.pt \
    --features_dir features
```

### Out of Memory
- Reduce batch size in `config/training_config_stage1.yaml`:
  ```yaml
  training:
    batch_size: 16  # Reduce from 24
    gradient_accumulation_steps: 3  # Increase to maintain effective batch size
  ```

### Slow Training
- Verify pre-extracted features are being used (check logs for "Loading pre-extracted")
- Check GPU utilization: `nvidia-smi`
- Ensure `num_workers: 2` is set (Windows limitation)

---

**Training is now running in the background!**
