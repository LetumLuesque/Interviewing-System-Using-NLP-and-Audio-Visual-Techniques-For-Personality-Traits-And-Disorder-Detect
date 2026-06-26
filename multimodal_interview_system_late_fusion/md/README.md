# Multimodal Interview Analysis System (Late Fusion)

A comprehensive deep learning system for analyzing interview videos using multimodal data (video, audio, and text) to predict Big Five personality traits and detect deception. This version implements **Late Fusion** architecture, where each modality makes independent predictions that are combined at the decision level.

## Key Difference: Late Fusion vs Early Fusion

**Late Fusion Architecture:**
- Each modality (visual, audio, text) processes data independently through its complete pipeline
- Each modality has its own prediction head that outputs personality traits and deception predictions
- Predictions from all modalities are combined at the final decision level
- Advantages:
  - Each modality learns specialized prediction strategies
  - More interpretable - can see individual modality contributions
  - Robust to missing modalities
  - Can use different fusion strategies (weighted, learned, ensemble)

**Early Fusion (Original):**
- Features from all modalities are combined early in the pipeline
- Single shared prediction head processes fused features
- Advantages:
  - Can learn cross-modal interactions
  - More compact model

## System Architecture

The system processes interview videos through three parallel and independent streams:

### 1. Visual Stream
- **Feature Extraction**: Face detection, landmark alignment, and Expr3DNet (3D-CNN) + Vision Transformer
- **Visual Prediction Head**: Independent RNN-based prediction module
- **Output**: Personality traits (OCEAN) + Deception probability

### 2. Audio Stream
- **Feature Extraction**: Noise reduction, MFCC extraction, AcousticFeatureNet (3D-CNN + GRU) + WavLM
- **Audio Prediction Head**: Independent RNN-based prediction module
- **Output**: Personality traits (OCEAN) + Deception probability

### 3. Text Stream
- **Feature Extraction**: Transcript processing with TextTraitNet (LSTM + Attention) + Sentence-BERT
- **Text Prediction Head**: Independent RNN-based prediction module
- **Output**: Personality traits (OCEAN) + Deception probability

### 4. Late Fusion Layer
- **Input**: Independent predictions from all modalities
- **Methods**: 
  - Weighted fusion (learnable weights per modality)
  - Average fusion (simple averaging)
  - Learned fusion (MLP-based combination)
  - Ensemble fusion (combination of multiple strategies)
- **Output**: Final combined predictions

### 5. Cultural Adaptation (Optional)
- **CultureAdaptNet**: Applied to final predictions for bias correction
- **Purpose**: Ensure fairness across cultural groups

## Architecture Diagram

```
Video Input → Expr3DNet + ViT → Visual Head → Traits + Deception ─┐
                                                                     │
Audio Input → AcousticNet + WavLM → Audio Head → Traits + Deception ├→ Late Fusion → Final Predictions
                                                                     │        ↓
Text Input → TextTraitNet + SBERT → Text Head → Traits + Deception ─┘   CultureAdaptNet (optional)
                                                                              ↓
                                                                      Personality + Deception
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multimodal_interview_system_late_fusion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download additional resources:
```bash
# Download NLTK data
python -m nltk.downloader punkt stopwords

# Download SpaCy model (optional)
python -m spacy download en_core_web_sm
```

## Dataset Setup

The system expects the MDPE dataset in the following structure:

```
data/mdpe_dataset/
├── metadata.csv
└── samples/
    ├── sample_001/
    │   ├── video.mp4
    │   ├── audio.wav
    │   ├── transcript.txt
    │   └── labels.json
    └── ...
```

The `metadata.csv` should contain columns:
- `sample_id`: Unique sample identifier
- `video_path`, `audio_path`, `transcript_path`: Paths to data files
- `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`: Big Five trait scores (0-100)
- `deception`: Deception label (0 or 1)
- `cultural_group`: Cultural group identifier
- `split`: Dataset split (train/val/test)

## Training

Train the model using:

```bash
python main_train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir data/mdpe_dataset \
    --seed 42
```

### Configuration

Edit `config/model_config.yaml` to customize:
- **Fusion method**: `weighted`, `average`, `max`, or `learned`
- Model architecture parameters for each modality
- Training hyperparameters
- Data augmentation settings
- Loss function weights

Key configuration options for late fusion:
```yaml
fusion:
  late_fusion:
    method: "weighted"  # or 'average', 'max', 'learned'
    num_modalities: 3
    dropout: 0.2
```

## Inference

Run inference on a new interview video:

```bash
python main_inference.py \
    --config config/model_config.yaml \
    --checkpoint checkpoints/best_model_epoch_X.pt \
    --video path/to/video.mp4 \
    --audio path/to/audio.wav \
    --transcript path/to/transcript.txt \
    --candidate_name "John Doe" \
    --position "Software Engineer" \
    --output_dir reports
```

## Output

The system generates:
1. **HTML Report**: Comprehensive analysis with:
   - Big Five personality profile
   - Individual modality predictions (visual, audio, text)
   - Modality contribution weights
   - Deception risk assessment
   - Confidence scores per modality
   - Recommendations
2. **Visualizations**: Interactive Plotly charts including:
   - Big Five personality profile (combined and per-modality)
   - 3D trait space visualization
   - Deception probability gauge
   - Modality contribution breakdown
3. **JSON Insights**: Machine-readable analysis results with individual modality outputs

## Project Structure

```
multimodal_interview_system_late_fusion/
├── config/              # Configuration files
│   ├── model_config.yaml   # Model architecture config (late fusion)
│   └── training_config.yaml
├── data/                # Data processing modules
│   ├── processors/      # Video, audio, text processors
│   ├── loaders/         # Dataset loaders
│   └── augmentation/    # Data augmentation
├── models/              # Model definitions
│   ├── unimodal/        # Unimodal feature extractors
│   │   ├── visual/      # Visual models (Expr3DNet, ViT)
│   │   ├── audio/       # Audio models (AcousticNet, WavLM)
│   │   └── text/        # Text models (TextTraitNet, SBERT)
│   ├── fusion/          # Fusion models
│   │   ├── late_fusion.py        # Late fusion implementation (NEW)
│   │   ├── early_fusion.py       # Early fusion (for comparison)
│   │   └── culture_adapt_net.py  # Cultural bias correction
│   └── prediction/      # Prediction heads
│       ├── unimodal_heads.py  # Per-modality prediction heads (NEW)
│       └── bigfive_net.py     # Legacy prediction module
├── training/            # Training infrastructure
├── evaluation/          # Evaluation metrics and visualization
├── inference/           # Inference pipeline
├── utils/               # Utility functions
├── main_train.py        # Training script
├── main_inference.py    # Inference script
└── requirements.txt     # Dependencies
```

## Features

- **Late Fusion Architecture**: Independent modality predictions combined at decision level
- **Multimodal Processing**: Integrates video, audio, and text modalities
- **Multiple Fusion Strategies**: Weighted, average, max, or learned fusion
- **Modality Analysis**: See individual contributions and predictions per modality
- **Cultural Bias Correction**: CultureAdaptNet for fair predictions across cultural groups
- **Deception Detection**: Binary classification for deception probability (per modality and combined)
- **Big Five Prediction**: OCEAN personality trait scoring (per modality and combined)
- **Comprehensive Reporting**: HR-ready reports with modality breakdown
- **Real-world Adaptations**: Noise-robust audio, low-light video enhancement
- **Robustness**: Works even with missing modalities

## Model Components

### Visual Stream
- **Expr3DNet**: 3D CNN for temporal micro-expression modeling
- **ViT Encoder**: Vision Transformer for frame-level features
- **Visual Prediction Head**: RNN-based head for trait and deception prediction

### Audio Stream
- **AcousticFeatureNet**: 3D-CNN + GRU for MFCC feature extraction
- **WavLM Encoder**: Pre-trained WavLM for speech pattern analysis
- **Audio Prediction Head**: RNN-based head for trait and deception prediction

### Text Stream
- **TextTraitNet**: LSTM with attention for text-based trait analysis
- **Sentence-BERT**: Semantic text embeddings
- **Text Prediction Head**: RNN-based head for trait and deception prediction

### Fusion & Cultural Adaptation
- **Late Fusion**: Decision-level combination of modality predictions
  - Weighted: Learnable weights per modality
  - Average: Simple averaging
  - Learned: MLP-based combination
  - Ensemble: Multiple strategies combined
- **CultureAdaptNet**: Cultural normalization and bias correction

## Evaluation Metrics

- **Personality Prediction**: MAE per trait, overall MAE, per-modality MAE
- **Deception Detection**: Accuracy, Precision, Recall, F1-score (per-modality and combined)
- **Cultural Fairness**: Statistical parity difference
- **Modality Contribution**: Learned weights and importance scores
- **Ablation Studies**: Performance with individual modalities vs combined

## Comparison with Early Fusion

You can compare this late fusion system with the original early fusion version:
- **Late Fusion**: `multimodal_interview_system_late_fusion/`
- **Early Fusion**: `multimodal_interview_system/`

Expected differences:
- Late fusion provides better interpretability (individual modality predictions)
- Late fusion is more robust to missing modalities
- Early fusion may capture cross-modal interactions better
- Performance depends on dataset characteristics

## Advantages of Late Fusion

1. **Interpretability**: Can analyze which modalities contribute most to predictions
2. **Modularity**: Easy to add/remove modalities or update individual streams
3. **Robustness**: Works with missing modalities (uses only available ones)
4. **Flexibility**: Can use different fusion strategies without retraining feature extractors
5. **Analysis**: Can study disagreements between modalities (e.g., when visual suggests truth but audio suggests deception)

## License

[Your License Here]

## Citation

If you use this system, please cite:

```bibtex
@software{multimodal_interview_analysis_late_fusion,
  title={Multimodal Interview Analysis System with Late Fusion},
  author={Ali M. Negm, Mazen Ebrahim, Tamer M. Nassef},
  year={2025},
  note={Late fusion architecture for personality assessment and deception detection}
}
```

## Contact

For questions or issues, please contact:
- Ali M. Negm: ali.mohamed49@msa.edu.eg
- Mazen Ebrahim: maashraf@msa.edu.eg  
- Tamer M. Nassef: tnassef@msa.edu.eg

## Acknowledgments

This late fusion implementation extends the original multimodal interview analysis system by implementing decision-level fusion instead of feature-level fusion, providing enhanced interpretability and modularity.
