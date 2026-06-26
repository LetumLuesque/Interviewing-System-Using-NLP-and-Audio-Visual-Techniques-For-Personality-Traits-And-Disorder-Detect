"""
Test script to verify combined dataset loading
"""
from data.loaders.combined_loader import create_combined_dataloader
from data.processors.video_processor import VideoProcessor
from data.processors.audio_processor import AudioProcessor
from data.processors.text_processor import TextProcessor

def test_combined_dataset():
    """Test loading combined MDPE + SEUMLD dataset"""
    
    mdpe_dir = "D:\\Grad\\Datasets\\MDPE-Dataset"
    seumld_dir = "D:\\Grad\\Datasets\\SEUMLD\\SEUMLD"
    
    print("=" * 80)
    print("Testing Combined Dataset Loader")
    print("=" * 80)
    
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    
    try:
        # Test train loader
        print("\n1. Creating train loader...")
        train_loader = create_combined_dataloader(
            mdpe_data_dir=mdpe_dir,
            seumld_data_dir=seumld_dir,
            split='train',
            batch_size=2,
            shuffle=False,
            num_workers=0,
            video_processor=video_processor,
            audio_processor=audio_processor,
            text_processor=text_processor
        )
        
        print(f"   Train dataset size: {len(train_loader.dataset)}")
        
        # Get a batch
        print("\n2. Getting a batch...")
        batch = next(iter(train_loader))
        
        print(f"   Batch size: {len(batch['personality'])}")
        print(f"   Personality shape: {batch['personality'].shape}")
        print(f"   Deception shape: {batch['deception'].shape}")
        print(f"   Deception values: {batch['deception'].tolist()}")
        
        # Check deception distribution
        print("\n3. Checking deception distribution in train set...")
        all_deception = []
        for i, batch in enumerate(train_loader):
            all_deception.extend(batch['deception'].tolist())
            if i >= 10:  # Check first 11 batches
                break
        
        import torch
        all_deception = torch.tensor(all_deception)
        unique_deception = torch.unique(all_deception).tolist()
        value_counts = torch.bincount(all_deception.long()).tolist()
        
        print(f"   Total samples checked: {len(all_deception)}")
        print(f"   Unique values: {unique_deception}")
        for val, count in enumerate(value_counts):
            if count > 0:
                pct = (count / len(all_deception)) * 100
                print(f"      Label {val}: {count} samples ({pct:.1f}%)")
        
        if 0 in unique_deception and 1 in unique_deception:
            print("\n   [SUCCESS] Found both deceptive (1) and truthful (0) samples!")
        else:
            print("\n   [WARNING] Missing one class!")
        
        # Test val loader
        print("\n4. Creating validation loader...")
        val_loader = create_combined_dataloader(
            mdpe_data_dir=mdpe_dir,
            seumld_data_dir=seumld_dir,
            split='val',
            batch_size=2,
            shuffle=False,
            num_workers=0,
            video_processor=video_processor,
            audio_processor=audio_processor,
            text_processor=text_processor
        )
        
        print(f"   Val dataset size: {len(val_loader.dataset)}")
        
        print("\n" + "=" * 80)
        print("Combined dataset test completed successfully!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load combined dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_combined_dataset()

