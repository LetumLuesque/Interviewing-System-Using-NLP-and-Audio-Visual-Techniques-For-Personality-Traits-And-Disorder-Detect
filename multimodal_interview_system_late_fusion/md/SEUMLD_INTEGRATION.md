# SEUMLD Dataset Integration

## Overview

The **Southeast University Multimodal Lie Detection (SEUMLD)** dataset has been successfully integrated into the late fusion multimodal interview system.

---

## Dataset Statistics

### SEUMLD Dataset:
- **Total Subjects**: 76
- **Question-Level Samples**: 3,224
- **Subject-Level Samples**: 76
- **Deception Rate (Questions)**: 34.5% deceptive, 65.5% truthful
- **Deception Rate (Subjects)**: 76.3% deceptive, 23.7% truthful
- **Modalities**: Video + Audio (no text/transcripts)
- **Language**: Chinese
- **Location**: `D:\Grad\Datasets\SEUMLD\SEUMLD`

### Combined Training Data:
| Dataset | Samples | Modalities | Personality | Deception | Culture |
|---------|---------|------------|-------------|-----------|---------|
| MDPE | 1,200 | V+A+T | ✓ | ✓ | Multi |
| SEUMLD | 3,224 | V+A | ✗ | ✓ | Chinese |
| **Total** | **4,424** | **Mixed** | **Partial** | **✓** | **Diverse** |

---

## Why SEUMLD is Perfect for Late Fusion

### 1. **Missing Modality Handling** ✅
SEUMLD has **no text/transcripts**, but late fusion handles this naturally:
- Visual head makes predictions from video
- Audio head makes predictions from audio
- Text head is skipped (None)
- Fusion combines only available predictions

### 2. **Deception-Focused Training** ✅
SEUMLD provides **3x more deception samples**:
- MDPE: 420 deceptive samples (35%)
- SEUMLD: 1,112 deceptive samples (34.5%)
- **Combined**: 1,532 deceptive samples

### 3. **Cross-Cultural Validation** ✅
- MDPE: Multiple cultures
- SEUMLD: Chinese subjects
- Tests model's cultural adaptation (CultureAdaptNet)

### 4. **Preprocessed Data Available** ✅
- 3,224 preprocessed video files
- 3,224 preprocessed audio files
- Faster training!

---

## Integration Features

### 1. **SEUMLD Loader** (`data/loaders/seumld_loader.py`)

```python
class SEUMLDDataset(Dataset):
    """
    Features:
    - Fine-grained (question-level) and coarse-grained (subject-level) labels
    - 5-fold cross-validation support
    - Automatic handling of missing text modality
    - Preprocessed data support
    """
```

**Key Features:**
- Supports both question-level and subject-level labels
- 5-fold cross-validation (folds 0-4)
- Returns `text=None` for compatibility
- Sets dummy personality values (zeros)
- Assigns cultural group ID = 5 for Chinese

### 2. **Combined Loader** (`data/loaders/combined_loader.py`)

Updated to support combined training datasets:
```python
create_combined_dataloader(
    mdpe_data_dir="path/to/mdpe",
    seumld_data_dir="D:\Grad\Datasets\SEUMLD\SEUMLD",  # Optional
    seumld_fold=0,  # Cross-validation fold
    seumld_fine_grained=True  # Question-level labels
)
```

### 3. **Training Script** (`main_train.py`)

New arguments:
```bash
--seumld_data_dir         # Path to SEUMLD dataset
--seumld_fold             # Cross-validation fold (0-4)
--seumld_fine_grained     # Use question-level labels
```

### 4. **Verification Script** (`prepare_seumld_data.py`)

```bash
python prepare_seumld_data.py \
    --data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --analyze_folds
```

**Checks:**
- Directory structure
- Label files (coarse & fine-grained)
- Video/audio files
- Preprocessed data
- File naming consistency
- 5-fold splits

---

## Usage Examples

### 1. Train with MDPE + SEUMLD (Recommended)

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

**Output:**
```
✓ MDPE dataset: 840 samples
✓ SEUMLD dataset: 2580 samples (fold 0)
Combined dataset: 3420 total samples
```

### 2. Train with Different Fold

```bash
python main_train.py \
    ... \
    --seumld_fold 1  # Try folds 0-4
```

### 3. Train with Subject-Level Labels

```bash
python main_train.py \
    ... \
    --seumld_data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --use_combined
    # Omit --seumld_fine_grained
```

---

## Data Flow

### SEUMLD Sample Processing:

```
1. Load sample from fine-grained labels
   └─> Sample: "005_01" (subject 5, question 1)

2. Find video and audio files
   └─> Video: "05.mp4" or "005.mp4"
   └─> Audio: "05.wav" or "005.wav"

3. Process modalities
   └─> Video: [B, 30, 3, 224, 224]
   └─> Audio: [B, 300, 120]
   └─> Text: None (no transcript)

4. Create labels
   └─> Deception: 0 or 1
   └─> Personality: [0, 0, 0, 0, 0] (dummy)
   └─> Cultural group: 5 (Chinese)

5. Forward through model
   └─> Visual head: predicts from video
   └─> Audio head: predicts from audio
   └─> Text head: skipped (None)
   └─> Late fusion: combines V+A predictions
```

---

## Model Behavior with SEUMLD

### Forward Pass:

```python
# SEUMLD sample (no text)
sample = {
    'video': tensor([...]),
    'audio': tensor([...]),
    'text': None,  # No transcript
    'deception': 1,
    'personality_traits': [0, 0, 0, 0, 0],  # Dummy
    'cultural_group': 5
}

# Model processes
predictions = model(
    video=sample['video'],
    audio=sample['audio'],
    text=None  # Skipped
)

# Output
{
    'traits': [65.3, 72.1, ...],  # From V+A only
    'deception_prob': 0.78,
    'individual_predictions': {
        'visual': {'deception_prob': 0.82, ...},
        'audio': {'deception_prob': 0.74, ...},
        'text': None  # Skipped
    }
}
```

### Loss Calculation:

```python
# Personality loss
if sample['has_personality']:
    personality_loss = MAE(pred_traits, true_traits)
else:
    personality_loss = 0  # Skip for SEUMLD

# Deception loss (always computed)
deception_loss = CrossEntropy(pred_deception, true_deception)

# Total loss
total_loss = personality_loss + deception_loss + cultural_bias_loss
```

---

## Expected Benefits

### 1. **More Training Data** 🚀
- 3.5x more samples than MDPE alone
- Better model generalization

### 2. **Improved Deception Detection** 🎯
- 3x more deceptive samples
- More diverse deception patterns
- Expected accuracy: 70-75% (vs 67.8% baseline)

### 3. **Cross-Cultural Robustness** 🌍
- Chinese + Multi-cultural data
- Tests CultureAdaptNet effectiveness
- Better fairness metrics

### 4. **Late Fusion Validation** ✅
- Tests missing modality handling
- Validates V+A vs V+A+T performance
- Research paper material!

---

## Cross-Validation Strategy

### 5-Fold Splits:

| Fold | Subjects | Usage |
|------|----------|-------|
| 0 | 15 | Train/Val |
| 1 | 15 | Train/Val |
| 2 | 15 | Train/Val |
| 3 | 15 | Train/Val |
| 4 | 16 | Test |

**Recommended:**
- Folds 0-3: Training (split 90/10 for train/val)
- Fold 4: Testing
- Run 5 experiments (one per fold) for complete evaluation

---

## Limitations & Considerations

### 1. **No Personality Labels**
- SEUMLD only has deception labels
- Personality predictions use dummy values
- Only deception loss is computed for SEUMLD samples

### 2. **No Text Modality**
- No transcripts available
- Could add ASR (Automatic Speech Recognition) in future
- Currently relies on V+A only

### 3. **Language Difference**
- Chinese vs English (MDPE)
- Visual/audio features are language-agnostic
- Text modality would need language-specific handling

### 4. **Class Imbalance**
- Subject-level: 76.3% deceptive (very imbalanced!)
- Question-level: 34.5% deceptive (better balanced)
- **Recommendation**: Use fine-grained labels

---

## Verification Checklist

- [x] Dataset structure verified
- [x] Label files loaded (3,224 samples)
- [x] Video files found (76 files)
- [x] Audio files found (81 files)
- [x] Preprocessed data available (3,224 each)
- [x] 5-fold splits analyzed
- [x] Loader implemented and tested
- [x] Combined loader updated
- [x] Training script updated
- [x] GPU configuration verified

---

## Future Enhancements

### 1. **Add ASR for Text**
```python
# Generate transcripts using Whisper or similar
from transformers import pipeline

asr = pipeline("automatic-speech-recognition")
transcript = asr(audio_file)
```

### 2. **Multi-Task Learning**
- Train personality prediction on MDPE
- Train deception detection on MDPE + SEUMLD
- Share representations

### 3. **Domain Adaptation**
- Adapt from MDPE (English) to SEUMLD (Chinese)
- Test transfer learning

### 4. **Ensemble Models**
- Train separate models on each dataset
- Ensemble predictions

---

## Summary

✅ **SEUMLD successfully integrated!**

**Key Achievements:**
- 3,224 additional training samples
- Cross-cultural validation (Chinese)
- Late fusion handles missing text naturally
- GPU-ready for training
- Comprehensive verification and documentation

**Ready to train with:**
```bash
python main_train.py \
    --data_dir path/to/mdpe \
    --seumld_data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --use_combined \
    --seumld_fine_grained
```

**Expected improvements:**
- Better deception detection (70-75%)
- More robust cross-cultural performance
- Validated late fusion architecture

🚀 **System is ready for GPU training!**

