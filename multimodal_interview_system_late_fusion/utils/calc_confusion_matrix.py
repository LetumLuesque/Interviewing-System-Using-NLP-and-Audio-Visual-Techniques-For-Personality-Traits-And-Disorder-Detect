import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion.late_fusion import LateFusionMultimodal
from data.loaders.combined_loader import create_combined_dataloader
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def calculate_cm():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_config = load_config(os.path.join(base_dir, 'config', 'training_config_stage2.yaml'))
    model_config = load_config(os.path.join(base_dir, 'config', 'model_config.yaml'))
    
    # Load Data (Use Stage 2 Val set as it's the "Clean" set we optimized for)
    print("Loading validation data...")
    val_loader = create_combined_dataloader(
        mdpe_data_dir=r"D:\Grad\Datasets\MDPE-Dataset",
        seumld_data_dir=r"D:\Grad\Datasets\SEUMLD\SEUMLD",
        split='val',
        batch_size=32,
        shuffle=False,
        num_workers=0,
        use_preprocessed=True, # Use extracted features
        balance_classes=False, # Validation shouldn't be artificially balanced
        features_dir='features',
        seumld_fold=0,
        seumld_fine_grained=True
    )
    
    # Initialize Model
    model = LateFusionMultimodal(model_config)
    model.to(device)
    
    # Load Checkpoint
    ckpt_path = os.path.join(base_dir, 'checkpoints', 'best_model_epoch_36.pt')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            video = batch['video'].to(device) if batch['video'] is not None else None
            audio = batch['audio'].to(device) if batch['audio'] is not None else None
            text = {k: v.to(device) for k, v in batch['text'].items()} if batch['text'] is not None else None
            deception_target = batch['deception'].to(device)
            
            # Forward
            outputs = model(video, audio, text)
            probs = torch.sigmoid(outputs['deception_logits'])
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(deception_target.cpu().numpy())
            
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "="*40)
    print(f"CONFUSION MATRIX RESULTS (Epoch 36)")
    print("="*40)
    print(f"True Negatives (Truthful  -> Truthful) : {tn}")
    print(f"False Positives (Truthful -> Deceptive): {fp}")
    print(f"False Negatives (Deceptive -> Truthful): {fn}")
    print(f"True Positives  (Deceptive -> Deceptive): {tp}")
    print("-"*40)
    print(f"Total Samples: {len(all_targets)}")
    print(f"Accuracy: {(tp+tn)/len(all_targets):.4f}")
    print("="*40)

if __name__ == "__main__":
    calculate_cm()
