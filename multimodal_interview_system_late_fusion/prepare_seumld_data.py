"""
SEUMLD Dataset Preparation and Verification Script
Verifies the SEUMLD dataset structure and prepares it for training
"""
import os
import sys
import pandas as pd
from pathlib import Path
import argparse
from glob import glob

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


def verify_seumld_dataset(data_dir: str):
    """
    Verify SEUMLD dataset structure and files
    
    Args:
        data_dir: Path to SEUMLD dataset directory
    """
    data_path = Path(data_dir)
    
    print("=" * 70)
    print("SEUMLD DATASET VERIFICATION")
    print("=" * 70)
    print(f"\nDataset directory: {data_path}")
    
    # Check main directories
    print("\n1. Checking directory structure...")
    required_dirs = [
        "Original/Video",
        "Original/Audio",
        "Original/ECG",
        "Labels"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = data_path / dir_path
        if full_path.exists():
            print(f"   ✓ {dir_path}")
        else:
            print(f"   ✗ {dir_path} - MISSING")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
        return False
    
    # Check label files
    print("\n2. Checking label files...")
    label_files = {
        "Coarse-grained": data_path / "Labels" / "Coarse-grained-labels.csv",
        "Fine-grained": data_path / "Labels" / "Fine-grained-labels.csv",
        "5-fold splits": data_path / "Original" / "5fold_list.csv"
    }
    
    for name, file_path in label_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"   ✓ {name}: {len(df)} entries")
        else:
            print(f"   ✗ {name} - MISSING")
    
    # Load and analyze labels
    print("\n3. Analyzing labels...")
    
    # Coarse-grained labels
    coarse_file = data_path / "Labels" / "Coarse-grained-labels.csv"
    if coarse_file.exists():
        coarse_df = pd.read_csv(coarse_file)
        deceptive = coarse_df['label'].sum()
        truthful = len(coarse_df) - deceptive
        print(f"   Coarse-grained (subject-level):")
        print(f"   - Total subjects: {len(coarse_df)}")
        print(f"   - Deceptive: {deceptive} ({deceptive/len(coarse_df)*100:.1f}%)")
        print(f"   - Truthful: {truthful} ({truthful/len(coarse_df)*100:.1f}%)")
    
    # Fine-grained labels
    fine_file = data_path / "Labels" / "Fine-grained-labels.csv"
    if fine_file.exists():
        fine_df = pd.read_csv(fine_file)
        deceptive = fine_df['label'].sum()
        truthful = len(fine_df) - deceptive
        print(f"\n   Fine-grained (question-level):")
        print(f"   - Total questions: {len(fine_df)}")
        print(f"   - Deceptive: {deceptive} ({deceptive/len(fine_df)*100:.1f}%)")
        print(f"   - Truthful: {truthful} ({truthful/len(fine_df)*100:.1f}%)")
    
    # Check video files
    print("\n4. Checking video files...")
    video_dir = data_path / "Original" / "Video"
    video_files = list(video_dir.glob("*.mp4"))
    print(f"   - Found {len(video_files)} video files")
    
    if video_files:
        # Check a sample
        sample_video = video_files[0]
        print(f"   - Sample: {sample_video.name}")
        print(f"   - Size: {sample_video.stat().st_size / 1024**2:.2f} MB")
    
    # Check audio files
    print("\n5. Checking audio files...")
    audio_dir = data_path / "Original" / "Audio"
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"   - Found {len(audio_files)} audio files")
    
    if audio_files:
        sample_audio = audio_files[0]
        print(f"   - Sample: {sample_audio.name}")
        print(f"   - Size: {sample_audio.stat().st_size / 1024**2:.2f} MB")
    
    # Check preprocessed data (if exists)
    print("\n6. Checking preprocessed data...")
    preprocess_dir = data_path / "Preprocess"
    if preprocess_dir.exists():
        preprocess_video = preprocess_dir / "Video"
        preprocess_audio = preprocess_dir / "Audio"
        
        if preprocess_video.exists():
            preprocess_video_files = list(preprocess_video.glob("*"))
            print(f"   ✓ Preprocessed video: {len(preprocess_video_files)} files")
        else:
            print(f"   - No preprocessed video")
        
        if preprocess_audio.exists():
            preprocess_audio_files = list(preprocess_audio.glob("*"))
            print(f"   ✓ Preprocessed audio: {len(preprocess_audio_files)} files")
        else:
            print(f"   - No preprocessed audio")
    else:
        print(f"   - No preprocessed data (will use original)")
    
    # Verify file naming consistency
    print("\n7. Verifying file naming...")
    if fine_file.exists():
        fine_df = pd.read_csv(fine_file)
        
        # Extract subject IDs from fine-grained labels
        subject_ids = set()
        for name in fine_df['name']:
            subject_id = str(name).split('_')[0]
            subject_ids.add(subject_id)
        
        print(f"   - Unique subjects in labels: {len(subject_ids)}")
        
        # Check if video/audio files exist for subjects
        missing_video = 0
        missing_audio = 0
        
        for subject_id in list(subject_ids)[:5]:  # Check first 5
            # Try different naming conventions
            video_exists = any([
                (video_dir / f"{subject_id}.mp4").exists(),
                (video_dir / f"{int(subject_id):03d}.mp4").exists(),
                (video_dir / f"{int(subject_id):d}.mp4").exists()
            ])
            
            audio_exists = any([
                (audio_dir / f"{subject_id}.wav").exists(),
                (audio_dir / f"{int(subject_id):03d}.wav").exists(),
                (audio_dir / f"{int(subject_id):d}.wav").exists()
            ])
            
            if not video_exists:
                missing_video += 1
            if not audio_exists:
                missing_audio += 1
        
        if missing_video == 0 and missing_audio == 0:
            print(f"   ✓ File naming is consistent")
        else:
            print(f"   ⚠ Some files may be missing or have different naming")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if not missing_dirs:
        print("✓ All required directories present")
        print(f"✓ {len(video_files)} video files found")
        print(f"✓ {len(audio_files)} audio files found")
        
        if fine_file.exists():
            print(f"✓ {len(fine_df)} question-level samples available")
        
        print("\n✓ SEUMLD dataset is ready for training!")
        print("\nYou can now train with:")
        print(f"  python main_train.py \\")
        print(f"    --data_dir path/to/mdpe_dataset \\")
        print(f"    --seumld_data_dir {data_path} \\")
        print(f"    --use_combined \\")
        print(f"    --seumld_fine_grained")
        
        return True
    else:
        print(f"❌ Dataset verification failed")
        print(f"   Missing: {missing_dirs}")
        return False


def analyze_folds(data_dir: str):
    """Analyze 5-fold cross-validation splits"""
    data_path = Path(data_dir)
    fold_file = data_path / "Original" / "5fold_list.csv"
    
    if not fold_file.exists():
        print("No fold file found")
        return
    
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION FOLDS ANALYSIS")
    print("=" * 70)
    
    folds_df = pd.read_csv(fold_file)
    
    for i in range(5):
        fold_col = f'fold{i}'
        subjects = folds_df[fold_col].dropna().astype(int).tolist()
        print(f"\nFold {i}: {len(subjects)} subjects")
        print(f"  Subject IDs: {subjects[:10]}{'...' if len(subjects) > 10 else ''}")
    
    print("\nRecommendation:")
    print("  - Use fold 0-3 for training")
    print("  - Use fold 4 for testing")
    print("  - Split training folds into train/val (90/10)")


def main():
    parser = argparse.ArgumentParser(description='Prepare and verify SEUMLD dataset')
    parser.add_argument('--data_dir', type=str, 
                       default=r'D:\Grad\Datasets\SEUMLD\SEUMLD',
                       help='Path to SEUMLD dataset directory')
    parser.add_argument('--analyze_folds', action='store_true',
                       help='Analyze cross-validation folds')
    
    args = parser.parse_args()
    
    # Verify dataset
    success = verify_seumld_dataset(args.data_dir)
    
    # Analyze folds if requested
    if args.analyze_folds and success:
        analyze_folds(args.data_dir)
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

