# Quick Start Guide - Late Fusion System

## Installation

1. Navigate to the late fusion project directory:
```bash
cd d:\Code\Grad\multimodal_interview_system_late_fusion
```

2. Install dependencies (if not already installed):
```bash
pip install -r requirements.txt
```

## Training

### Basic Training
```bash
python main_train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir path/to/mdpe_dataset \
    --seed 42
```

### Training with Combined Dataset
```bash
python main_train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir path/to/mdpe_dataset \
    --seumld_data_dir "D:\Grad\Datasets\SEUMLD\SEUMLD" \
    --use_combined \
    --seed 42
```

### Resume Training from Checkpoint
```bash
python main_train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir path/to/mdpe_dataset \
    --resume checkpoints/best_model_epoch_X.pt
```

## Inference

### Single Interview Analysis
```bash
python main_inference.py \
    --config config/model_config.yaml \
    --checkpoint checkpoints/best_model_epoch_X.pt \
    --video path/to/interview.mp4 \
    --audio path/to/interview.wav \
    --transcript path/to/transcript.txt \
    --candidate_name "John Doe" \
    --position "Software Engineer" \
    --output_dir reports
```

### Inference with Missing Modalities
The late fusion system works even with missing modalities:

```bash
# Only video and text (no audio)
python main_inference.py \
    --config config/model_config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --video path/to/video.mp4 \
    --transcript path/to/transcript.txt \
    --output_dir reports

# Only audio and text (no video)
python main_inference.py \
    --config config/model_config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/audio.wav \
    --transcript path/to/transcript.txt \
    --output_dir reports
```

## Configuration Options

### Fusion Methods

Edit `config/model_config.yaml` to change the fusion strategy:

```yaml
fusion:
  late_fusion:
    method: "weighted"  # Options: 'weighted', 'average', 'max', 'learned'
```

#### Weighted (Default - Recommended)
```yaml
method: "weighted"
```
- Learns optimal weights for each modality
- Best balance of performance and interpretability

#### Average (Simple baseline)
```yaml
method: "average"
```
- Equal weights for all modalities
- Good baseline, no extra parameters

#### Max (Conservative)
```yaml
method: "max"
```
- Takes maximum across modalities
- Useful for detecting extreme values

#### Learned (Most flexible)
```yaml
method: "learned"
```
- MLP learns combination strategy
- More parameters, potentially better performance

### Cultural Adaptation

Enable or disable cultural bias correction:

```yaml
fusion:
  use_culture_adapt: true  # or false
  culture_adapt_net:
    input_dim: 5
    hidden_dim: 256
    num_cultural_groups: 6
```

## Understanding the Output

### Training Output
```
Epoch 1/100
Train Loss: 15.234 | Val Loss: 16.123 | Val Acc: 0.645
Best model saved at epoch 1

Epoch 5/100
Train Loss: 12.456 | Val Loss: 13.234 | Val Acc: 0.678
Best model saved at epoch 5
...
```

### Inference Output

The system generates three types of outputs:

1. **HTML Report** (`reports/John_Doe_analysis.html`)
   - Comprehensive analysis with visualizations
   - Individual modality predictions
   - Modality contribution weights
   - Personality profile and deception risk

2. **JSON Results** (`reports/John_Doe_results.json`)
   ```json
   {
     "candidate_name": "John Doe",
     "personality_traits": {
       "openness": 65.3,
       "conscientiousness": 72.1,
       ...
     },
     "individual_predictions": {
       "visual": {"openness": 63.2, ...},
       "audio": {"openness": 68.1, ...},
       "text": {"openness": 64.7, ...}
     },
     "deception_probability": 0.23,
     "modality_weights": {
       "visual": 0.32,
       "audio": 0.38,
       "text": 0.30
     }
   }
   ```

3. **Visualizations** (embedded in HTML)
   - Personality radar chart (overall + per-modality)
   - Deception probability gauge
   - Modality contribution breakdown
   - 3D trait space visualization

## Testing Individual Modalities

### Test Visual-Only Performance
Modify the forward call to pass only video:
```python
predictions = model(video=video_tensor, audio=None, text=None)
```

### Test Audio-Only Performance
```python
predictions = model(video=None, audio=audio_tensor, text=None)
```

### Test Text-Only Performance
```python
predictions = model(video=None, audio=None, text=text_dict)
```

## Analyzing Modality Contributions

After training, analyze which modalities contribute most:

```python
# Load model
model = MultimodalInterviewModel(config)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Get modality weights
weights = model.get_modality_contributions()
print(f"Visual contribution: {weights['trait_weights'][0]}")
print(f"Audio contribution: {weights['trait_weights'][1]}")
print(f"Text contribution: {weights['trait_weights'][2]}")
```

## Common Issues and Solutions

### Issue: Out of Memory
**Solution**: Reduce batch size in `config/training_config.yaml`:
```yaml
training:
  batch_size: 8  # or even 4
```

### Issue: Model not learning
**Solution**: 
1. Check learning rate (try 1e-3 or 5e-5)
2. Verify loss weights are balanced
3. Check if data is normalized properly

### Issue: One modality dominates
**Solution**:
1. Use 'learned' fusion method for more flexibility
2. Check if data quality differs across modalities
3. Add regularization to fusion weights

### Issue: Missing modality at inference
**Solution**: Late fusion handles this automatically! Just pass None for missing modalities.

## Evaluation

### Run Evaluation on Test Set
```bash
python evaluation/accuracy_test.py \
    --config config/model_config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --data_dir path/to/mdpe_dataset \
    --split test
```

### Generate Visualizations
```bash
python evaluation/visualization.py \
    --results results/test_predictions.json \
    --output_dir plots
```

## Comparing with Early Fusion

To compare performance with the early fusion version:

1. Train both models on the same data
2. Evaluate both on the same test set
3. Compare:
   - Overall performance (MAE, accuracy)
   - Robustness to missing modalities (late fusion should win)
   - Interpretability (late fusion provides per-modality insights)
   - Model size (early fusion is smaller)

## Next Steps

1. **Experiment with fusion methods**: Try 'learned' vs 'weighted'
2. **Analyze modality importance**: Which contributes most to each trait?
3. **Test robustness**: How well does it work with missing modalities?
4. **Cultural fairness**: Compare SPD with and without CultureAdaptNet
5. **Hyperparameter tuning**: Optimize learning rate, dropout, etc.

## Support

For issues or questions:
- Check `LATE_FUSION_EXPLANATION.md` for technical details
- Review `README.md` for full documentation
- Compare with original early fusion version in `../multimodal_interview_system/`

