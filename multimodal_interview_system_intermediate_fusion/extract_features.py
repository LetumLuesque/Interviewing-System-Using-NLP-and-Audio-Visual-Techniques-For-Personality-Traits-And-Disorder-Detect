"""
Feature Pre-Extraction Script
Processes videos once and saves features to disk for faster training.
"""
import argparse
import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

from data.processors.video_processor import VideoProcessor
from data.processors.audio_processor import AudioProcessor
from data.processors.text_processor import TextProcessor


def extract_mdpe_features(data_dir: str, output_dir: str, limit: int = None, args=None):
    """
    Extract features from MDPE dataset videos.
    
    Args:
        data_dir: Path to MDPE dataset root
        output_dir: Where to save extracted features
        limit: Optional limit on number of samples to process
        args: argparse namespace with shard_id and num_shards
    """
    print("=" * 60)
    print("MDPE Feature Extraction")
    print("=" * 60)
    
    # Initialize processors
    video_processor = VideoProcessor(
        fps=1,
        frame_size=224,
        use_face_detection=True,
        max_frames=10
    )
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    
    # Load metadata - try multiple possible locations
    metadata_paths = [
        os.path.join(data_dir, 'metadata_full.csv'),
        os.path.join(data_dir, 'metadata.csv'),
        os.path.join(data_dir, 'deception', 'metadata.csv'),
    ]
    
    metadata_path = None
    for path in metadata_paths:
        if os.path.exists(path):
            metadata_path = path
            break
    
    if metadata_path is None:
        print(f"Error: Metadata not found in any of: {metadata_paths}")
        return
    
    metadata = pd.read_csv(metadata_path)
    print(f"Found {len(metadata)} samples in metadata")
    
    if limit:
        metadata = metadata.head(limit)
        print(f"Processing only first {limit} samples")
        
    # Sharding for parallel processing
    if args.num_shards > 1:
        # Calculate shard size
        shard_size = len(metadata) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size if args.shard_id < args.num_shards - 1 else len(metadata)
        
        metadata = metadata.iloc[start_idx:end_idx]
        print(f"Shard {args.shard_id}/{args.num_shards}: Processing samples {start_idx} to {end_idx} ({len(metadata)} samples)")
    
    # Create output directories
    video_out = os.path.join(output_dir, 'mdpe', 'video')
    audio_out = os.path.join(output_dir, 'mdpe', 'audio')
    text_out = os.path.join(output_dir, 'mdpe', 'text')
    
    os.makedirs(video_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(text_out, exist_ok=True)
    
    # Process each sample
    success_count = 0
    error_count = 0
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting MDPE"):
        # Use sample_id from metadata, replace dashes with underscores for filename
        raw_sample_id = str(row.get('sample_id', f"{row['subject_id']}_{idx}"))
        feature_id = raw_sample_id.replace('-', '_')
        
        # Parse sample_id to get subject_id for video path lookup (format: "194-1-1")
        parts = raw_sample_id.split('-')
        subject_id = parts[0] if parts else raw_sample_id
        
        try:
            # Video - look for matching file in subject folder
            # First check deception folder
            subject_video_dir = os.path.join(data_dir, 'deception', 'raw_video', subject_id)
            if not os.path.exists(subject_video_dir):
                # Check emotion folder
                 subject_video_dir = os.path.join(data_dir, 'emotion', 'raw_video', subject_id)
            
            video_path = None
            if os.path.exists(subject_video_dir):
                # Find video matching this sample_id pattern
                for vf in Path(subject_video_dir).glob('*.mp4'):
                    if raw_sample_id in vf.stem or vf.stem == raw_sample_id:
                        video_path = str(vf)
                        break
                # If no exact match, try to find by index
                if video_path is None:
                    video_files = sorted(Path(subject_video_dir).glob('*.mp4'))
                    if idx < len(video_files):
                        video_path = str(video_files[idx % len(video_files)])
                    elif video_files:
                        video_path = str(video_files[0])
            
            video_feat_path = os.path.join(video_out, f"{feature_id}.pt")
            
            if os.path.exists(video_feat_path):
                # Skip if already exists
                success_count += 1
                continue

            if video_path and os.path.exists(video_path):
                video_features = video_processor.process_video(video_path)
                torch.save(video_features, video_feat_path)
            
                # Audio extraction
                # 1. Try raw_audio (Deception folder)
                audio_path = video_path.replace('.mp4', '.wav').replace('raw_video', 'raw_audio')
                if not os.path.exists(audio_path):
                    # 2. Try raw_audio in emotion folder? (Usually doesn't exist)
                    audio_path = video_path.replace('.mp4', '.wav').replace('raw_video', 'raw_audio').replace('deception', 'emotion')
                
                # 3. Fallback to video file if wav missing (Librosa handles mp4)
                if not os.path.exists(audio_path):
                    audio_path = video_path

                if os.path.exists(audio_path):
                    try:
                        audio_features = audio_processor.process_audio(audio_path)
                        torch.save(audio_features, os.path.join(audio_out, f"{feature_id}.pt"))
                    except Exception as ae:
                        print(f"Warning: Audio extraction failed for {feature_id}: {ae}")
            
            # Text (if transcript exists) - check multiple possible locations
            text_out_path = os.path.join(text_out, f"{feature_id}.pt")
            if not os.path.exists(text_out_path):
                transcript_paths = [
                    os.path.join(data_dir, 'deception', 'deception_features', 'transcriptions', subject_id, f"{raw_sample_id}.txt"),
                    os.path.join(data_dir, 'deception', 'transcripts', subject_id, f"{raw_sample_id}.txt"),
                ]
                for transcript_path in transcript_paths:
                    if os.path.exists(transcript_path):
                        try:
                            with open(transcript_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            text_features = text_processor.process_transcript(text)
                            torch.save(text_features, text_out_path)
                            break
                        except Exception as e:
                            pass
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Only print first 5 errors
                print(f"Error processing {feature_id}: {e}")
    
    print(f"\nExtraction complete: {success_count} success, {error_count} errors")
    print(f"Features saved to: {output_dir}/mdpe/")


def extract_seumld_features(data_dir: str, output_dir: str, limit: int = None, fine_grained: bool = True):
    """
    Extract features from SEUMLD dataset videos.
    
    Args:
        data_dir: Path to SEUMLD dataset root
        output_dir: Where to save extracted features
        limit: Optional limit on number of samples to process
        fine_grained: If True, extract from Preprocess folder (question-level clips).
                     If False, extract from Original folder (subject-level videos).
    """
    print("=" * 60)
    print("SEUMLD Feature Extraction")
    print(f"Mode: {'Fine-grained (question-level)' if fine_grained else 'Coarse-grained (subject-level)'}")
    print("=" * 60)
    
    # Initialize processors
    video_processor = VideoProcessor(
        fps=1,
        frame_size=224,
        use_face_detection=True,
        max_frames=10
    )
    audio_processor = AudioProcessor()
    
    # Choose source directory based on mode
    if fine_grained:
        # Fine-grained: Use Preprocess folder (question-level clips)
        video_dir = os.path.join(data_dir, 'Preprocess', 'Video')
        audio_dir = os.path.join(data_dir, 'Preprocess', 'Audio')
        print(f"Extracting from Preprocess folder (fine-grained question-level clips)")
    else:
        # Coarse-grained: Use Original folder (subject-level videos)
        video_dir = os.path.join(data_dir, 'Original', 'Video')
        audio_dir = os.path.join(data_dir, 'Original', 'Audio')
        print(f"Extracting from Original folder (subject-level videos)")
    
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found: {video_dir}")
        return
    
    # Find video files
    # For fine-grained, videos are in subdirectories named after sample IDs (e.g., "005_01/video.mp4")
    # For coarse-grained, videos are directly in the Video folder
    video_files = []
    
    if fine_grained:
        # Fine-grained: videos are in subdirectories
        sample_dirs = [d for d in Path(video_dir).iterdir() if d.is_dir()]
        print(f"Found {len(sample_dirs)} sample directories")
        
        for sample_dir in sample_dirs:
            # Look for video files OR frame directories in this sample directory
            video_file = None
            # First try video files
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                matches = list(sample_dir.glob(f'*{ext}'))
                if matches:
                    video_file = matches[0]
                    break
            # If no video file, check if it's a frame directory (contains images)
            if not video_file:
                frame_files = list(sample_dir.glob('*.jpg')) + list(sample_dir.glob('*.png'))
                if frame_files:
                    # Use directory path - video processor can handle frame directories
                    video_file = sample_dir
            if video_file:
                video_files.append((video_file, sample_dir.name))  # Store (path, sample_id)
    else:
        # Coarse-grained: videos are directly in Video folder
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            files = list(Path(video_dir).glob(f'*{ext}'))
            if files:
                video_files = [(f, f.stem) for f in files]
                break
    
    print(f"Found {len(video_files)} video files in {video_dir}")
    
    if limit:
        video_files = video_files[:limit]
        print(f"Processing only first {limit} files")
    
    # Create output directories
    video_out = os.path.join(output_dir, 'seumld', 'video')
    audio_out = os.path.join(output_dir, 'seumld', 'audio')
    
    os.makedirs(video_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)
    
    # Process each video
    success_count = 0
    error_count = 0
    audio_count = 0
    skipped_count = 0
    
    for video_item in tqdm(video_files, desc="Extracting SEUMLD"):
        # video_item is either (Path, sample_id) tuple or just Path
        if isinstance(video_item, tuple):
            video_path, sample_id = video_item
            feature_id = sample_id  # Use sample directory name (e.g., "005_01")
        else:
            video_path = video_item
            # For coarse-grained, use subject ID (e.g., "005" -> "5")
            feature_id = str(int(video_path.stem)) if video_path.stem.isdigit() else video_path.stem
        
        try:
            # Video - skip if exists
            video_out_path = os.path.join(video_out, f"{feature_id}.pt")
            if os.path.exists(video_out_path):
                skipped_count += 1
                continue
            
            # Check if video_path is a directory (frame directory) or file
            if isinstance(video_path, Path) and video_path.is_dir():
                # Process frame directory
                frame_files = sorted(list(video_path.glob('*.jpg')) + list(video_path.glob('*.png')))
                if frame_files:
                    # Load frames and process them
                    import cv2
                    import numpy as np
                    frames = []
                    # Sample frames evenly (up to max_frames)
                    max_frames = 10
                    step = max(1, len(frame_files) // max_frames)
                    selected_frames = frame_files[::step][:max_frames]
                    
                    for frame_file in selected_frames:
                        frame = cv2.imread(str(frame_file))
                        if frame is not None:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = cv2.resize(frame, (224, 224))
                            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                            # Normalize
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                            frame_tensor = (frame_tensor - mean) / std
                            frames.append(frame_tensor)
                    
                    if frames:
                        video_features = torch.stack(frames, dim=0)
                    else:
                        raise ValueError(f"No valid frames found in {video_path}")
                else:
                    raise ValueError(f"No frame files found in {video_path}")
            else:
                # Process video file normally
                video_features = video_processor.process_video(str(video_path))
            
            torch.save(video_features, video_out_path)
            
            # Audio - look for matching audio file
            audio_out_path = os.path.join(audio_out, f"{feature_id}.pt")
            if not os.path.exists(audio_out_path):
                # Try to find matching audio file
                if fine_grained:
                    # Fine-grained: audio is in matching subdirectory (e.g., "005_01/audio.wav")
                    audio_subdir = os.path.join(audio_dir, feature_id)
                    audio_candidates = [
                        os.path.join(audio_subdir, f"{feature_id}.wav"),
                        os.path.join(audio_subdir, "audio.wav"),
                        os.path.join(audio_subdir, "*.wav")
                    ]
                    audio_path = None
                    for candidate in audio_candidates:
                        if '*' in candidate:
                            # Try glob pattern
                            matches = list(Path(audio_subdir).glob('*.wav'))
                            if matches:
                                audio_path = str(matches[0])
                                break
                        elif os.path.exists(candidate):
                            audio_path = candidate
                            break
                else:
                    # Coarse-grained: try different naming conventions
                    audio_candidates = [
                        os.path.join(audio_dir, f"{video_path.stem}.wav"),
                        os.path.join(audio_dir, f"{int(video_path.stem):03d}.wav"),
                        os.path.join(audio_dir, f"{int(video_path.stem):d}.wav")
                    ]
                    audio_path = None
                    for candidate in audio_candidates:
                        if os.path.exists(candidate):
                            audio_path = candidate
                            break
                
                if audio_path and os.path.exists(audio_path):
                    try:
                        audio_features = audio_processor.process_audio(audio_path)
                        torch.save(audio_features, audio_out_path)
                        audio_count += 1
                    except Exception as e:
                        if error_count <= 3:
                            print(f"Audio error for {feature_id}: {e}")
                elif fine_grained:
                    # For fine-grained, try extracting audio from video
                    try:
                        audio_features = audio_processor.process_audio(str(video_path))
                        torch.save(audio_features, audio_out_path)
                        audio_count += 1
                    except Exception as e:
                        if error_count <= 3:
                            print(f"Audio extraction from video failed for {feature_id}: {e}")
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"Error processing {feature_id}: {e}")
    
    print(f"\nExtraction complete:")
    print(f"  Success: {success_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Audio extracted: {audio_count}")
    print(f"  Features saved to: {output_dir}/seumld/")


def extract_reallife_features(data_dir: str, output_dir: str, limit: int = None):
    """
    Extract features from Real-life Deception Detection dataset videos.
    
    Args:
        data_dir: Path to Real-life Deception Detection dataset root
        output_dir: Where to save extracted features
        limit: Optional limit on number of samples to process
    """
    print("=" * 60)
    print("Real-life Deception Detection Feature Extraction")
    print("=" * 60)
    
    # Initialize processors
    video_processor = VideoProcessor(
        fps=1,
        frame_size=224,
        use_face_detection=True,
        max_frames=10
    )
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    
    # Find video files - check multiple possible locations
    video_files = []
    for subdir in ['Clips', 'clips', 'Videos', 'videos', 'Deceptive', 'Truthful', '']:
        search_dir = os.path.join(data_dir, subdir) if subdir else data_dir
        if os.path.exists(search_dir):
            video_files.extend(list(Path(search_dir).rglob('*.mp4')))
            video_files.extend(list(Path(search_dir).rglob('*.avi')))
    
    # Remove duplicates
    video_files = list(set(video_files))
    
    print(f"Found {len(video_files)} video files")
    
    if limit:
        video_files = video_files[:limit]
        print(f"Processing only first {limit} files")
    
    # Create output directories
    video_out = os.path.join(output_dir, 'reallife', 'video')
    audio_out = os.path.join(output_dir, 'reallife', 'audio')
    text_out = os.path.join(output_dir, 'reallife', 'text')
    
    os.makedirs(video_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(text_out, exist_ok=True)
    
    # Process each video
    success_count = 0
    error_count = 0
    audio_count = 0
    text_count = 0
    
    for video_path in tqdm(video_files, desc="Extracting Real-life"):
        feature_id = video_path.stem  # filename without extension
        
        try:
            # Skip if already exists
            video_out_path = os.path.join(video_out, f"{feature_id}.pt")
            if not os.path.exists(video_out_path):
                video_features = video_processor.process_video(str(video_path))
                torch.save(video_features, video_out_path)
            
            # Audio - extract directly from video file
            audio_out_path = os.path.join(audio_out, f"{feature_id}.pt")
            if not os.path.exists(audio_out_path):
                try:
                    audio_features = audio_processor.process_audio(str(video_path))
                    torch.save(audio_features, audio_out_path)
                    audio_count += 1
                except Exception as e:
                    pass  # Audio extraction may fail for some files
            
            # Transcript - check Transcription/Deceptive/ and Transcription/Truthful/ folders
            text_out_path = os.path.join(text_out, f"{feature_id}.pt")
            if not os.path.exists(text_out_path):
                transcript_path = None
                # Try different possible transcript locations
                for subdir in ['Transcription/Deceptive', 'Transcription/Truthful', 'Transcription']:
                    txt_path = os.path.join(data_dir, subdir, f"{feature_id}.txt")
                    if os.path.exists(txt_path):
                        transcript_path = txt_path
                        break
                
                if transcript_path:
                    try:
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        text_features = text_processor.process_transcript(text)
                        torch.save(text_features, text_out_path)
                        text_count += 1
                    except Exception as e:
                        pass  # Text processing may fail
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"Error processing {feature_id}: {e}")
    
    print(f"\nExtraction complete: {success_count} success, {error_count} errors")
    print(f"Audio extracted: {audio_count}, Text extracted: {text_count}")
    print(f"Features saved to: {output_dir}/reallife/")


def main():
    parser = argparse.ArgumentParser(description='Extract features from video datasets')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='features',
                        help='Output directory for extracted features')
    parser.add_argument('--dataset', type=str, default='mdpe',
                        choices=['mdpe', 'seumld', 'reallife', 'all'],
                        help='Which dataset to extract (mdpe, seumld, reallife, or all)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process (for testing)')
    parser.add_argument('--seumld_fine_grained', action='store_true', default=True,
                        help='Extract fine-grained (question-level) features for SEUMLD (default: True)')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Number of shards for parallel processing')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='Shard ID (0-indexed) for this process')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == 'mdpe' or args.dataset == 'all':
        extract_mdpe_features(args.data_dir, args.output_dir, args.limit, args=args)
    
    if args.dataset == 'seumld' or args.dataset == 'all':
        extract_seumld_features(args.data_dir, args.output_dir, args.limit, fine_grained=args.seumld_fine_grained)
    
    if args.dataset == 'reallife' or args.dataset == 'all':
        extract_reallife_features(args.data_dir, args.output_dir, args.limit)
    
    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
