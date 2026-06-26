import pandas as pd
import os
from pathlib import Path

def generate_metadata(data_dir):
    print(f"Generating full metadata from {data_dir}")
    
    # 1. Load existing metadata (Deception samples)
    meta_path = os.path.join(data_dir, 'metadata.csv')
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found")
        return

    df_deception = pd.read_csv(meta_path)
    print(f"Loaded {len(df_deception)} deception samples")
    df_deception['dataset_source'] = 'deception'
    
    # Ensure deception column exists and is 1
    if 'deception' not in df_deception.columns:
        df_deception['deception'] = 1
    else:
        df_deception['deception'] = 1 # Force ensure
        
    # Create mapping from subject_id to personality traits
    personality_cols = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    
    # Check for missing cols
    for col in personality_cols:
        if col not in df_deception.columns:
            print(f"Warning: Missing {col}, filling with 50.0")
            df_deception[col] = 50.0
            
    # Create simple dict mapping: subject_id -> {trait: value}
    # We take the first occurrence for each subject
    subject_traits = {}
    for _, row in df_deception.iterrows():
        subj = row['subject_id']
        if subj not in subject_traits:
            subject_traits[subj] = {col: row[col] for col in personality_cols}
            # Also keep cultural group if exists
            if 'cultural_group' in row:
                subject_traits[subj]['cultural_group'] = row['cultural_group']

    print(f"Mapped traits for {len(subject_traits)} subjects")

    # 2. Scan Emotion folder (Truthful/Control samples)
    emotion_dir = os.path.join(data_dir, 'emotion', 'raw_video')
    if not os.path.exists(emotion_dir):
        print(f"Error: {emotion_dir} not found")
        return

    emotion_samples = []
    
    # Walk through subject folders in emotion/raw_video
    for subject_folder in os.listdir(emotion_dir):
        # Folder name is usually subject_id (e.g. "194")
        if not subject_folder.isdigit():
            continue
            
        subject_id = int(subject_folder)
        subject_path = os.path.join(emotion_dir, subject_folder)
        
        if not os.path.isdir(subject_path):
            continue
            
        # Find video files
        for vid_file in os.listdir(subject_path):
            if not vid_file.endswith('.mp4'):
                continue
                
            sample_id = os.path.splitext(vid_file)[0]
            
            # Get traits for this subject
            # Default if subject not in deception set (unlikely)
            traits = subject_traits.get(subject_id, {col: 50.0 for col in personality_cols}) 
            
            # Create sample entry
            sample = {
                'sample_id': sample_id,
                'subject_id': subject_id,
                'deception': 0, # Truthful/Control
                'dataset_source': 'emotion',
                'video_path': os.path.join('emotion', 'raw_video', subject_folder, vid_file)
            }
            
            # Add traits
            sample.update(traits)
            emotion_samples.append(sample)

    print(f"Found {len(emotion_samples)} emotion samples")
    
    if not emotion_samples:
        print("No emotion samples found! Check directory structure.")
        return

    df_emotion = pd.DataFrame(emotion_samples)
    
    # 3. Combine and Save
    # Concatenate - pandas aligns columns automatically
    df_full = pd.concat([df_deception, df_emotion], ignore_index=True)
    
    # Fill N/A
    df_full['dataset_source'] = df_full['dataset_source'].fillna('deception')
    
    output_path = os.path.join(data_dir, 'metadata_full.csv')
    df_full.to_csv(output_path, index=False)
    
    print(f"Saved full metadata to {output_path}")
    print(f"Total samples: {len(df_full)}")
    print(f"Deception distribution: {df_full['deception'].value_counts().to_dict()}")

if __name__ == "__main__":
    generate_metadata(r'D:\Grad\Datasets\MDPE-Dataset')
