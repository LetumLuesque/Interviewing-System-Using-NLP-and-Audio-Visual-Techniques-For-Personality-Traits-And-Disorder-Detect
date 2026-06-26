"""
MDPE Dataset Preparation Script
Prepares the MDPE dataset for training by creating metadata.csv from existing label files.

MDPE Structure:
- Subjects are in folders named by ID (194, 195, ..., 233)
- Each subject folder contains multiple clips (one per question)
- Format: {subject_id}-{question_id}-{something}.mp4/.wav
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from glob import glob
from sklearn.model_selection import train_test_split


def prepare_mdpe_dataset(data_dir: str, output_dir: str = None):
    """
    Prepare MDPE dataset by creating metadata.csv that maps to extracted files.
    This version creates a separate sample for EACH clip and ensures valid splits.
    
    Args:
        data_dir: Path to MDPE-Dataset folder
        output_dir: Output directory (defaults to data_dir)
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir) if output_dir else data_path
    
    print(f"Preparing MDPE dataset from: {data_path}")
    print("=" * 60)
    
    # Load personality labels (subject-level)
    personality_file = data_path / "all_label_personality.csv"
    if not personality_file.exists():
        personality_file = data_path / "all_label_personality_normalized.csv"
    
    if not personality_file.exists():
        raise FileNotFoundError(f"Personality labels not found in {data_path}")
    
    personality_df = pd.read_csv(personality_file)
    print(f"Loaded {len(personality_df)} subjects with personality labels")
    
    # Load partition info
    partition_file = data_path / "partition.csv"
    if partition_file.exists():
        partition_df = pd.read_csv(partition_file)
        print(f"Loaded {len(partition_df)} partition entries")
    else:
        print("No partition.csv found")
        partition_df = pd.DataFrame(columns=['id', 'partition'])
    
    # Merge personality and partition
    subject_metadata = personality_df.merge(partition_df, on='id', how='left')
    
    # Identify subjects without partition (likely deception subset 194+)
    missing_partition = subject_metadata['partition'].isna()
    if missing_partition.any():
        print(f"Warning: {missing_partition.sum()} subjects missing partition info. Assigning randomly.")
        
        # Get indices of missing partitions
        indices = subject_metadata.index[missing_partition].tolist()
        
        # Split indices into train/val/test (70/15/15)
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        subject_metadata.loc[train_idx, 'partition'] = 'train'
        subject_metadata.loc[val_idx, 'partition'] = 'val'
        subject_metadata.loc[test_idx, 'partition'] = 'test'
    
    # Rename columns to match expected format
    column_mapping = {
        'id': 'subject_id',
        'Extraversion': 'extraversion',
        'Agreeableness': 'agreeableness',
        'Conscientiousnes': 'conscientiousness',
        'Conscientiousness': 'conscientiousness',
        'Neuroticism': 'neuroticism',
        'Open Mindedness': 'openness',
        'Openness': 'openness',
        'partition': 'split'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in subject_metadata.columns:
            subject_metadata = subject_metadata.rename(columns={old_col: new_col})
            
    # Add default deception and cultural group if missing
    if 'deception' not in subject_metadata.columns:
        # Subjects 194-233 are from deception subset based on observation
        subject_metadata['deception'] = subject_metadata['subject_id'].apply(
            lambda x: 1 if 194 <= int(x) <= 233 else 0
        )
    if 'cultural_group' not in subject_metadata.columns:
        subject_metadata['cultural_group'] = 'zh'  # Chinese dataset
    
    # Look for extracted media files
    deception_dir = data_path / "deception"
    audio_dir = deception_dir / "raw_audio"
    video_dir = deception_dir / "raw_video"
    
    # Also check emotion directory (for Truthful samples)
    emotion_dir = data_path / "emotion"
    emotion_video_dir = emotion_dir / "raw_video"
    # Emotion directory has no raw_audio folder, will use video files for audio
    
    print(f"\nScanning directories:")
    print(f"  Deception Video: {video_dir} (Exists: {video_dir.exists()})")
    print(f"  Emotion Video:   {emotion_video_dir} (Exists: {emotion_video_dir.exists()})")
    
    # ---------------------------------------------------------
    # EXPAND TO CLIP LEVEL
    # ---------------------------------------------------------
    clip_samples = []
    
    for _, subject_row in subject_metadata.iterrows():
        subject_id = str(subject_row['subject_id'])
        
        # Find video clips for this subject
        # First check deception folder
        subject_video_dir = video_dir / subject_id
        is_emotion = False
        
        if not subject_video_dir.exists():
            # Check emotion folder
            subject_video_dir = emotion_video_dir / subject_id
            if subject_video_dir.exists():
                is_emotion = True
            else:
                continue
            
        video_files = list(subject_video_dir.glob("*.mp4"))
        
        for video_file in video_files:
            # Assume audio file has same name but .wav extension in audio dir
            # Only for deception dataset
            if not is_emotion:
                audio_file = audio_dir / subject_id / f"{video_file.stem}.wav"
                audio_path_str = str(audio_file) if audio_file.exists() else None
            else:
                # For emotion dataset, no separate audio files; use video as audio source
                # Or leave None and let processor handle it? 
                # extract_features.py needs to know to use video.
                # We'll mark audio_path as None here, but loader/extractor should handle it.
                # Actually, providing video_path as audio_path is a good trick?
                # No, better leave None and handle logic in extractor.
                audio_path_str = None
            
            # Use video filename as sample ID (e.g., "194-1-1")
            sample_id = video_file.stem
            
            sample = subject_row.to_dict().copy()
            sample['sample_id'] = sample_id
            sample['video_path'] = str(video_file)
            sample['audio_path'] = audio_path_str
            sample['transcript_path'] = None
            
            # Keep original subject_id
            sample['subject_id'] = subject_id
            
            # Force deception label based on source folder if needed
            # But subjects 194+ are deception anyway.
            # Just relying on subject_metadata['deception'] is consistent.
            
            clip_samples.append(sample)
            
    # Create final metadata DataFrame
    metadata = pd.DataFrame(clip_samples)
    
    # Statistics
    print(f"\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total subjects: {len(subject_metadata)}")
    print(f"Total clips found: {len(metadata)}")
    
    if len(metadata) > 0:
        video_count = metadata['video_path'].notna().sum()
        audio_count = metadata['audio_path'].notna().sum()
        print(f"\nMedia availability:")
        print(f"  With video: {video_count}")
        print(f"  With audio: {audio_count}")
        
        print(f"\nSplit distribution (clips):")
        print(metadata['split'].value_counts())
        
        # Save filtered metadata (training ready)
        output_file = output_path / "metadata.csv"
        metadata.to_csv(output_file, index=False)
        print(f"\nTraining metadata saved to: {output_file}")
        
        # Show sample
        print(f"\nSample entries (first 3):")
        print(metadata[['sample_id', 'subject_id', 'split', 'deception']].head(3))
    else:
        print("\n❌ No clips found! Check directory structure.")


def main():
    parser = argparse.ArgumentParser(description='Prepare MDPE dataset metadata')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to MDPE-Dataset folder')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for metadata.csv (optional)')
    
    args = parser.parse_args()
    
    prepare_mdpe_dataset(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
