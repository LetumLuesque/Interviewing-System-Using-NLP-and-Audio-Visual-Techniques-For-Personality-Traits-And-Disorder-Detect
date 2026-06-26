# SEUMLD Fine-Grained Feature Extraction

## ✅ Status: Running

**Started**: January 12, 2026  
**Mode**: Fine-grained (question-level)  
**Total Samples**: 3,224  
**Output**: `features/seumld/video/` and `features/seumld/audio/`

---

## What Was Fixed

### Problem
- Only 76 subject-level features existed
- Fine-grained training needs ~3,224 question-level features
- Preprocess folder contains frame directories (JPG images), not video files

### Solution
1. ✅ Updated `extract_features.py` to handle frame directories
2. ✅ Updated `seumld_loader.py` to check for fine-grained features first
3. ✅ Added frame directory processing (loads JPG frames and processes them)

---

## Extraction Process

### Source Data
- **Location**: `D:\Grad\Datasets\SEUMLD\SEUMLD\Preprocess\Video\`
- **Format**: Frame directories (e.g., `005_01/` contains JPG frames)
- **Samples**: 3,224 question-level clips

### Processing
1. For each sample directory (e.g., `005_01/`):
   - Load all JPG/PNG frames
   - Sample evenly to get 10 frames
   - Resize to 224x224
   - Normalize (ImageNet stats)
   - Save as `features/seumld/video/005_01.pt`

2. For audio:
   - Look for matching audio in `Preprocess/Audio/005_01/`
   - Extract MFCC features
   - Save as `features/seumld/audio/005_01.pt`

---

## Command

```bash
python extract_features.py \
    --dataset seumld \
    --data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --output_dir features \
    --seumld_fine_grained
```

---

## Expected Output

### Video Features
- **Location**: `features/seumld/video/`
- **Format**: `{sample_id}.pt` (e.g., `005_01.pt`)
- **Shape**: `[10, 3, 224, 224]` (10 frames, RGB, 224x224)
- **Total**: 3,224 files

### Audio Features  
- **Location**: `features/seumld/audio/`
- **Format**: `{sample_id}.pt` (e.g., `005_01.pt`)
- **Shape**: `[300, 40]` (300 timesteps, 40 MFCC coefficients)
- **Total**: ~3,224 files (may be fewer if audio missing)

---

## Progress Monitoring

### Check Log File
```bash
Get-Content seumld_extraction.log -Wait -Tail 20
```

### Check Feature Count
```bash
# Video features
Get-ChildItem features\seumld\video | Measure-Object | Select-Object Count

# Audio features
Get-ChildItem features\seumld\audio | Measure-Object | Select-Object Count
```

### Estimated Time
- **Per sample**: ~1-2 seconds
- **Total (3,224 samples)**: ~1-2 hours
- **With face detection**: ~2-4 hours

---

## Loader Updates

The `seumld_loader.py` now checks for features in this order:

1. **Fine-grained mode**:
   - First: `features/seumld/video/{sample_name}.pt` (e.g., `005_01.pt`)
   - Fallback: `features/seumld/video/{subject_id}.pt` (e.g., `5.pt`)
   - Final: On-the-fly processing from Preprocess frames

2. **Coarse-grained mode**:
   - First: `features/seumld/video/{subject_id}.pt` (e.g., `5.pt`)
   - Final: On-the-fly processing from Original videos

---

## Benefits

✅ **Faster Training**: Pre-extracted features load instantly  
✅ **Consistent Processing**: Same features every time  
✅ **Fine-Grained Support**: Question-level features for better training  
✅ **Backward Compatible**: Still works with subject-level features

---

## After Extraction Completes

Once extraction finishes:

1. **Verify Count**:
   ```bash
   Get-ChildItem features\seumld\video | Measure-Object
   # Should show ~3,224 files
   ```

2. **Test Loading**:
   ```python
   from data.loaders.seumld_loader import create_dataloader
   loader = create_dataloader(
       data_dir=r"D:\Grad\Datasets\SEUMLD\SEUMLD",
       split='train',
       batch_size=4,
       features_dir='features',
       use_fine_grained=True
   )
   batch = next(iter(loader))
   print("✅ Features loaded successfully!")
   ```

3. **Start Stage 2 Training**:
   ```bash
   python train_staged.py \
       --stage 2 \
       --data_dir "D:\Grad\Datasets\MDPE-Dataset" \
       --seumld_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
       --features_dir features \
       --seed 42
   ```

---

## Troubleshooting

### Extraction Stops/Fails
- Check log: `seumld_extraction.log`
- Resume: Script skips existing files, just re-run

### Missing Audio Features
- Some samples may not have audio
- Training will work with video-only samples
- Check: `Get-ChildItem features\seumld\audio | Measure-Object`

### Out of Disk Space
- Each feature file: ~50-100 KB
- Total: ~150-300 MB for all features
- Check: `Get-ChildItem features\seumld -Recurse | Measure-Object -Property Length -Sum`

---

**Extraction is running in the background. Check `seumld_extraction.log` for progress!**
