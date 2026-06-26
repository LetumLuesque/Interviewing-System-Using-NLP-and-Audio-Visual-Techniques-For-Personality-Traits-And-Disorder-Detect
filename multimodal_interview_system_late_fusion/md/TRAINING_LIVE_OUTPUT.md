# Live Training Output Guide

## ✅ Training is Running Successfully!

**Current Status**: Stage 1 training is active and progressing.

---

## How to See Live Training Output

### Option 1: Watch Log File (Recommended)
```powershell
# In a separate terminal window:
cd d:\Code\Grad\multimodal_interview_system_late_fusion
Get-Content logs\training_20260112_162352.log -Wait -Tail 20
```

This will show you:
- Real-time batch progress
- Loss values every 5 batches
- Epoch completion
- Validation metrics
- Model checkpoints

### Option 2: Check Latest Log Entries
```powershell
# Quick status check:
Get-Content logs\training_20260112_162352.log -Tail 30
```

### Option 3: Monitor Checkpoints
```powershell
# Watch for new checkpoints:
Get-ChildItem checkpoints\best_model_epoch_*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

---

## What You'll See

### During Training (Every 5 Batches):
```
Epoch 2 [Train] Batch 5/84 | Loss: 108.4058
Epoch 2 [Train] Batch 10/84 | Loss: 110.7870
Epoch 2 [Train] Batch 15/84 | Loss: 109.2341
...
```

### At End of Epoch:
```
Completed training for epoch 2
Validating epoch 2...
Epoch 2 [Val] Batch 5/18 | Loss: 112.3456
...
Epoch 2 Metrics:
  val_mae_mean: 37.7171
  val_loss: 113.4236
Saved best model at epoch 2
```

---

## Current Training Status

- **Epoch**: 2/50 (in progress)
- **Batch Size**: 8
- **Effective Batch Size**: 24 (with gradient accumulation)
- **Progress**: Visible every 5 batches
- **Checkpoints**: Saved automatically when validation improves

---

## Troubleshooting

### If you don't see output:
1. **Check if training is running**:
   ```powershell
   Get-Process python | Where-Object {$_.WorkingSet -gt 100000000}
   ```

2. **Find the latest log**:
   ```powershell
   Get-ChildItem logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
   ```

3. **View live log**:
   ```powershell
   Get-Content logs\<latest_log_file> -Wait -Tail 20
   ```

---

## Expected Training Time

- **Per Epoch**: ~5-10 minutes
- **Total (50 epochs)**: ~4-8 hours
- **Early Stopping**: May stop earlier if validation doesn't improve for 10 epochs

---

**Training is running! Check the log file for live updates.** ✅
