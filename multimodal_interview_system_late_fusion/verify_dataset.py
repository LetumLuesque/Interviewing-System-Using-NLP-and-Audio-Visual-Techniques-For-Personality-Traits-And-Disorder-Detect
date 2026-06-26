"""
Diagnostic script to verify MDPE dataset loading and deception labels
"""
import pandas as pd
from pathlib import Path
import torch
from data.loaders.mdpe_loader import create_dataloader
from data.processors.video_processor import VideoProcessor
from data.processors.audio_processor import AudioProcessor
from data.processors.text_processor import TextProcessor

def verify_dataset(data_dir: str):
    """Verify dataset extraction and deception labels"""
    
    data_path = Path(data_dir)
    
    print("=" * 80)
    print("DATASET VERIFICATION")
    print("=" * 80)
    
    # 1. Check if metadata.csv exists
    metadata_path = data_path / "metadata.csv"
    print(f"\n1. Checking metadata file...")
    print(f"   Path: {metadata_path}")
    
    if not metadata_path.exists():
        print(f"   [ERROR] metadata.csv not found at {metadata_path}")
        print(f"   [TIP] Run prepare_mdpe_data.py first to create metadata.csv")
        return False
    
    print(f"   [OK] metadata.csv exists")
    
    # 2. Load and inspect metadata
    print(f"\n2. Loading metadata...")
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"   [OK] Loaded {len(metadata)} rows")
        print(f"   Columns: {list(metadata.columns)}")
    except Exception as e:
        print(f"   [ERROR] loading metadata: {e}")
        return False
    
    # 3. Check for deception column
    print(f"\n3. Checking deception labels...")
    if 'deception' not in metadata.columns:
        print(f"   [ERROR] 'deception' column not found in metadata.csv")
        print(f"   Available columns: {list(metadata.columns)}")
        print(f"   [TIP] Run prepare_mdpe_data.py to add deception labels")
        return False
    
    print(f"   [OK] 'deception' column found")
    
    # 4. Analyze deception distribution
    print(f"\n4. Deception label distribution (ALL samples):")
    deception_counts = metadata['deception'].value_counts().sort_index()
    print(f"   {deception_counts.to_dict()}")
    
    total = len(metadata)
    for label, count in deception_counts.items():
        pct = (count / total) * 100
        print(f"   Label {label}: {count} samples ({pct:.1f}%)")
    
    # Check if all are 0
    if len(deception_counts) == 1 and 0 in deception_counts:
        print(f"   [WARNING] All samples have deception=0!")
        print(f"   [TIP] This explains why model predicts all 0s")
    
    # 5. Check split distribution
    print(f"\n5. Split distribution:")
    if 'split' in metadata.columns:
        split_counts = metadata['split'].value_counts()
        print(f"   Overall: {split_counts.to_dict()}")
        
        for split in ['train', 'val', 'test']:
            if split in split_counts:
                split_data = metadata[metadata['split'] == split]
                split_deception = split_data['deception'].value_counts().sort_index()
                print(f"   {split}: {len(split_data)} samples")
                print(f"      Deception: {split_deception.to_dict()}")
    else:
        print(f"   [WARNING] No 'split' column found")
    
    # 6. Check sample_id range
    print(f"\n6. Sample ID analysis:")
    if 'sample_id' in metadata.columns:
        sample_ids = metadata['sample_id'].astype(int)
        print(f"   Range: {sample_ids.min()} to {sample_ids.max()}")
        print(f"   Expected deception=1 for IDs 194-233")
        
        # Check if deception labels match sample IDs
        expected_deceptive = metadata[(metadata['sample_id'] >= 194) & (metadata['sample_id'] <= 233)]
        actual_deceptive = metadata[metadata['deception'] == 1]
        
        print(f"   Samples with ID 194-233: {len(expected_deceptive)}")
        print(f"   Samples with deception=1: {len(actual_deceptive)}")
        
        if len(expected_deceptive) > 0:
            print(f"   Expected deceptive IDs: {sorted(expected_deceptive['sample_id'].unique().tolist())[:10]}...")
        
        if len(actual_deceptive) > 0:
            print(f"   Actual deceptive IDs: {sorted(actual_deceptive['sample_id'].unique().tolist())[:10]}...")
        
        if len(expected_deceptive) != len(actual_deceptive):
            print(f"   [WARNING] Mismatch between sample IDs and deception labels!")
            print(f"   Expected {len(expected_deceptive)} deceptive samples, but found {len(actual_deceptive)}")
    else:
        print(f"   [WARNING] No 'sample_id' column found")
    
    # 7. Check for any NaN values in deception
    print(f"\n7. Data quality check:")
    nan_count = metadata['deception'].isna().sum()
    print(f"   NaN values in deception: {nan_count}")
    if nan_count > 0:
        print(f"   [WARNING] Found {nan_count} NaN values in deception column!")
    
    # 8. Test data loader
    print(f"\n8. Testing data loader...")
    try:
        video_processor = VideoProcessor()
        audio_processor = AudioProcessor()
        text_processor = TextProcessor()
        
        train_loader = create_dataloader(
            data_dir=data_dir,
            split='train',
            batch_size=2,
            shuffle=False,
            num_workers=0,
            video_processor=video_processor,
            audio_processor=audio_processor,
            text_processor=text_processor
        )
        
        print(f"   [OK] DataLoader created successfully")
        print(f"   Dataset size: {len(train_loader.dataset)}")
        
        # Get a batch and check labels
        print(f"\n9. Checking loaded labels from DataLoader...")
        batch = next(iter(train_loader))
        
        print(f"   Batch size: {len(batch['personality'])}")
        print(f"   Personality shape: {batch['personality'].shape}")
        print(f"   Deception shape: {batch['deception'].shape}")
        print(f"   Deception values in first batch: {batch['deception'].tolist()}")
        print(f"   Deception unique values in first batch: {torch.unique(batch['deception']).tolist()}")
        
        # Check all batches
        all_deception = []
        all_personality = []
        sample_ids_loaded = []
        for i, batch in enumerate(train_loader):
            all_deception.extend(batch['deception'].tolist())
            all_personality.append(batch['personality'])
            sample_ids_loaded.extend(batch.get('sample_ids', []))
            if i >= 10:  # Check first 11 batches
                break
        
        all_deception = torch.tensor(all_deception)
        unique_deception = torch.unique(all_deception).tolist()
        print(f"\n   First 11 batches deception distribution:")
        print(f"   Total samples checked: {len(all_deception)}")
        print(f"   Unique values: {unique_deception}")
        if len(all_deception) > 0:
            value_counts = torch.bincount(all_deception.long())
            print(f"   Value counts: {value_counts.tolist()}")
            for val, count in enumerate(value_counts.tolist()):
                if count > 0:
                    pct = (count / len(all_deception)) * 100
                    print(f"      Label {val}: {count} samples ({pct:.1f}%)")
        
        if len(unique_deception) == 1 and unique_deception[0] == 0:
            print(f"   [PROBLEM] All deception labels are 0!")
            print(f"   [TIP] This is why the model can't learn deception detection")
        elif 1 in unique_deception:
            print(f"   [OK] Good: Found both 0 and 1 deception labels")
        
    except Exception as e:
        print(f"   [ERROR] creating DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "D:\\Grad\\Datasets\\MDPE-Dataset"
    verify_dataset(data_dir)

