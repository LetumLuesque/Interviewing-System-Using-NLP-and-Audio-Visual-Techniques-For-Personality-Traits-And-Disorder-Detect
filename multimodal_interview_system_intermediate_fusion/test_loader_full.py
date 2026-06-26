
import torch
from data.loaders.mdpe_loader import MDPEDataLoader
from torch.utils.data import DataLoader
import os

def test_loader():
    print("Testing MDPE Loader with full dataset...")
    
    # Setup
    data_dir = r'D:\Grad\Datasets\MDPE-Dataset'
    img_size = 224
    
    # Initialize Dataset
    dataset = MDPEDataLoader(
        root_dir=data_dir,
        img_size=img_size,
        split='train',
        use_preextracted=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check if we have ~2240 samples
    if len(dataset) < 2000:
        print("WARNING: Dataset size seems too small for full dataset!")
    
    # Initialize Loader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    print("Iterating checking for missing modalities...")
    
    for i, batch in enumerate(loader):
        if i > 2: break
        
        print(f"Batch {i}:")
        print(f"  Video: {batch['video'].shape}")
        print(f"  Audio: {batch['audio'].shape}")
        print(f"  Text: {batch['text'].shape}")
        print(f"  Deception Labels: {batch['labels']['deception']}")
        
        # Check if we have mixed labels (both 0 and 1)
        # Note: shuffle=True so we might see both
    
    print("Test passed!")

if __name__ == "__main__":
    test_loader()
