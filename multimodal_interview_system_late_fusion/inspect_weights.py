
import torch
import sys
import os

# Add path
sys.path.append(os.getcwd())

from models.multimodal_model import MultimodalInterviewModel
from utils.helpers import load_config

def inspect_weights():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint on {device}...")
    
    checkpoint = torch.load('checkpoints/best_model_epoch_36.pt', map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Extract LateFusion weights
    # Keys usually: 'late_fusion.deception_weights', 'late_fusion.trait_weights'
    
    print("\nKeys in state dict being checked:")
    found = False
    for k in state_dict.keys():
        if 'late_fusion' in k and 'weight' in k:
            print(f"{k}: {state_dict[k]}")
            found = True
            
    if not found:
        print("No late_fusion weights found directly. Maybe inside module?")
        
if __name__ == '__main__':
    inspect_weights()
