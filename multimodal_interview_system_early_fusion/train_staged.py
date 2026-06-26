"""
Staged Training Script for Multimodal Interview Analysis System

This script automates the staged training approach:
- Stage 1: MDPE only (baseline)
- Stage 2: MDPE + SEUMLD (expand deception data + multilingual generalization check)
"""
import argparse
import subprocess
import sys
from pathlib import Path
import json


def run_stage(stage_num, config_path, model_config_path, data_dir, 
              seumld_dir=None, resume_checkpoint=None,
              seed=42, seumld_fold=0, seumld_fine_grained=True, features_dir=None, no_preprocessed=False,
              freeze_backbones=False, fusion_method=None):
    """Run a training stage"""
    
    cmd = [
        sys.executable, "main_train.py",
        "--config", config_path,
        "--model_config", model_config_path,
        "--data_dir", data_dir,
        "--seed", str(seed),
        "--stage", str(stage_num)
    ]
    
    if features_dir:
        cmd.extend(["--features_dir", features_dir])
    
    if no_preprocessed:
        cmd.append("--no_preprocessed")
    
    if freeze_backbones:
        cmd.append("--freeze_backbones")
        
    if fusion_method:
        cmd.extend(["--fusion_method", fusion_method])
    
    if stage_num == 1:
        print("\n" + "=" * 80)
        print("STAGE 1: MDPE Only (Baseline)")
        print("=" * 80)
        print("Training with MDPE dataset only to establish baseline.")
        print("This stage learns all modalities (V+A+T) with complete labels.")
        print("-" * 80)
        
    elif stage_num == 2:
        print("\n" + "=" * 80)
        print("STAGE 2: MDPE + SEUMLD")
        print("=" * 80)
        print("Adding SEUMLD dataset to expand deception detection data.")
        print("This stage tests missing modality handling (SEUMLD has no text).")
        print("-" * 80)
        
        if not seumld_dir:
            raise ValueError("SEUMLD directory required for Stage 2")
            
        cmd.extend(["--use_combined", "--seumld_data_dir", seumld_dir])
        cmd.extend(["--seumld_fold", str(seumld_fold)])
        if seumld_fine_grained:
            cmd.append("--seumld_fine_grained")
        
        if resume_checkpoint:
            cmd.extend(["--resume", resume_checkpoint])

    elif stage_num == 3:
        print("\n" + "=" * 80)
        print("STAGE 3: Joint Balance (Personality + Deception)")
        print("=" * 80)
        print("Balancing personality and deception heads for final unified model.")
        print("-" * 80)
        
        if not seumld_dir:
            raise ValueError("SEUMLD directory required for Stage 3")
            
        cmd.extend(["--use_combined", "--seumld_data_dir", seumld_dir])
        cmd.extend(["--seumld_fold", str(seumld_fold)])
        if seumld_fine_grained:
            cmd.append("--seumld_fine_grained")
        
        if resume_checkpoint:
            cmd.extend(["--resume", resume_checkpoint])
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print("\n" + "-" * 80)
    
    # Run training
    result = subprocess.run(cmd, check=False)
    
    return result.returncode == 0


def find_latest_checkpoint(checkpoint_dir="checkpoints", stage_num=None, fusion_method=None):
    """Find the latest checkpoint file"""
    candidate_dirs = []
    root = Path(checkpoint_dir)
    
    # Use 'weighted' as default fusion name if none provided (matching main_train.py)
    fusion_name = fusion_method if fusion_method else "weighted"
    
    if stage_num is not None:
        # Check fusion-specific stage folder (new structure)
        candidate_dirs.append(root / fusion_name / f"stage{stage_num}")
        # Fallback to old structure for backward compatibility
        candidate_dirs.append(root / f"stage{stage_num}")
    
    candidate_dirs.append(root / fusion_name)
    candidate_dirs.append(root)

    for checkpoint_path in candidate_dirs:
        if not checkpoint_path.exists():
            continue

        # Look for best model checkpoints first
        checkpoints = list(checkpoint_path.glob("best_model_epoch_*.pt"))
        if not checkpoints:
            # Fallback to periodic checkpoints
            checkpoints = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))

        if not checkpoints:
            continue

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Staged Training for Multimodal Interview Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 3 stages automatically (MDPE → MDPE+SEUMLD → All 3 datasets)
  python train_staged.py --all --data_dir path/to/mdpe_dataset --seumld_dir path/to/seumld --reallife_dir path/to/reallife
  
  # Run Stage 1 only (MDPE baseline)
  python train_staged.py --stage 1 --data_dir path/to/mdpe_dataset
  
  # Run Stage 2 (MDPE + SEUMLD)
  python train_staged.py --stage 2 --data_dir path/to/mdpe_dataset --seumld_dir path/to/seumld
  
  # Resume from Stage 1 (auto-detects latest checkpoint)
  python train_staged.py --resume_from_stage 1 --data_dir path/to/mdpe_dataset --seumld_dir path/to/seumld
        """
    )
    
    parser.add_argument('--stage', type=int, choices=[1, 2, 3],
                       help='Training stage to run (1, 2, or 3)')
    parser.add_argument('--all', action='store_true',
                       help='Run all stages sequentially')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to MDPE dataset directory')
    parser.add_argument('--seumld_dir', type=str, default=r'D:\Grad\Datasets\SEUMLD\SEUMLD',
                       help='Path to SEUMLD dataset')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training config file')
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml',
                       help='Path to model config file')
    parser.add_argument('--stage1_config', type=str, default='config/training_config_stage1.yaml',
                       help='Path to Stage 1 specific config (optional)')
    parser.add_argument('--stage2_config', type=str, default='config/training_config_stage2.yaml',
                       help='Path to Stage 2 specific config (optional)')
    parser.add_argument('--stage3_config', type=str, default='config/training_config_stage3.yaml',
                       help='Path to Stage 3 specific config (optional)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (auto-detected if not specified)')
    parser.add_argument('--resume_from_stage', type=int, default=None, choices=[1, 2, 3],
                       help='Resume training from a specific stage using latest checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--seumld_fold', type=int, default=0,
                       help='SEUMLD cross-validation fold (0-4)')
    parser.add_argument('--seumld_fine_grained', action='store_true', default=True,
                       help='Use fine-grained (question-level) labels for SEUMLD')
    parser.add_argument('--skip_stage', type=int, nargs='+', default=[],
                       help='Skip specific stages when using --all (e.g., --skip_stage 3)')
    parser.add_argument('--features_dir', type=str, default=None,
                       help='Path to pre-extracted features directory')
    parser.add_argument('--no_preprocessed', action='store_true',
                       help='Do not use preprocessed data (force on-the-fly processing from original files)')
    parser.add_argument('--freeze_backbones', action='store_true',
                       help='Freeze feature extractor backbones')
    parser.add_argument('--fusion_method', type=str, default=None, choices=['weighted', 'mean', 'concat'],
                       help='Early fusion method to use (weighted, mean, concat)')
    
    args = parser.parse_args()
    
    if not args.stage and not args.all and not args.resume_from_stage:
        parser.error("Either --stage, --all, or --resume_from_stage must be specified")
    
    print("\n" + "=" * 80)
    print("STAGED TRAINING SYSTEM - MDPE + SEUMLD (Multilingual)")
    print("=" * 80)
    print(f"MDPE Directory: {args.data_dir}")
    if args.seumld_dir:
        print(f"SEUMLD Directory: {args.seumld_dir}")
    print("=" * 80)
    
    # Handle resume from stage
    if args.resume_from_stage:
        latest_checkpoint = find_latest_checkpoint(stage_num=args.resume_from_stage, fusion_method=args.fusion_method)
        if not latest_checkpoint:
            print(f"\n[ERROR] No checkpoint found to resume from stage {args.resume_from_stage}")
            print("Please run previous stages first or specify --resume with a checkpoint path.")
            sys.exit(1)
        
        print(f"\n[RESUME] Resuming from Stage {args.resume_from_stage}")
        print(f"Using checkpoint: {latest_checkpoint}")
        
        if args.resume_from_stage == 1:
            stages_to_run = [1, 2, 3]
        elif args.resume_from_stage == 2:
            stages_to_run = [2, 3]
        else:
            stages_to_run = [3]
        
        # Remove skipped stages
        stages_to_run = [s for s in stages_to_run if s not in args.skip_stage]
        current_checkpoint = latest_checkpoint
    else:
        stages_to_run = []
        if args.all:
            stages_to_run = [1, 2, 3]
            # Remove skipped stages
            stages_to_run = [s for s in stages_to_run if s not in args.skip_stage]
        else:
            stages_to_run = [args.stage]
        
        # Track checkpoints between stages
        current_checkpoint = args.resume
    
    for stage_num in stages_to_run:
        # Use stage-specific config if available, otherwise use default
        if stage_num == 1:
            config_path = args.stage1_config if Path(args.stage1_config).exists() else args.config
        elif stage_num == 2:
            config_path = args.stage2_config if Path(args.stage2_config).exists() else args.config
        elif stage_num == 3:
            config_path = args.stage3_config if Path(args.stage3_config).exists() else args.config
        
        # Auto-detect checkpoint if not provided and not stage 1
        # Also allow resuming stage 1 if checkpoint is provided
        if stage_num > 1 and not current_checkpoint:
            # For stage N, prefer checkpoint from stage N-1, then fallback to generic.
            current_checkpoint = find_latest_checkpoint(stage_num=stage_num - 1, fusion_method=args.fusion_method)
            if current_checkpoint:
                print(f"\n[OK] Auto-detected checkpoint: {current_checkpoint}")
        elif stage_num == 1 and current_checkpoint:
            print(f"\n[RESUME] Resuming Stage 1 from checkpoint: {current_checkpoint}")
        
        # Run stage
        success = run_stage(
            stage_num=stage_num,
            config_path=config_path,
            model_config_path=args.model_config,
            data_dir=args.data_dir,
            seumld_dir=args.seumld_dir,
            resume_checkpoint=current_checkpoint,
            seed=args.seed,
            seumld_fold=args.seumld_fold,
            seumld_fine_grained=args.seumld_fine_grained,
            features_dir=args.features_dir,
            no_preprocessed=args.no_preprocessed,
            freeze_backbones=args.freeze_backbones,
            fusion_method=args.fusion_method
        )
        
        if not success:
            print(f"\n[ERROR] Stage {stage_num} failed!")
            print(f"\n[TIP] To resume from this point, use:")
            latest_checkpoint = find_latest_checkpoint(stage_num=stage_num, fusion_method=args.fusion_method)
            if latest_checkpoint:
                print(f"   python train_staged.py --resume_from_stage {stage_num} \\")
                print(f"       --data_dir {args.data_dir} \\")
                if args.seumld_dir:
                    print(f"       --seumld_dir {args.seumld_dir} \\")
                print(f"       --resume {latest_checkpoint}")
            print("\nStopping staged training.")
            sys.exit(1)
        
        # Find latest checkpoint for next stage
        current_checkpoint = find_latest_checkpoint(stage_num=stage_num, fusion_method=args.fusion_method)
        if current_checkpoint:
            print(f"\n[OK] Stage {stage_num} completed successfully!")
            print(f"Latest checkpoint: {current_checkpoint}")
            print(f"\n[TIP] To resume training later, use:")
            if stage_num < 2:
                print(f"   python train_staged.py --resume_from_stage {stage_num + 1} \\")
            else:
                print(f"   python train_staged.py --stage 2 --resume {current_checkpoint} \\")
            print(f"       --data_dir {args.data_dir} \\")
            if args.seumld_dir:
                print(f"       --seumld_dir {args.seumld_dir}")
        else:
            print(f"\n[WARNING] Stage {stage_num} completed but no checkpoint found.")
        
        # Keep current_checkpoint for next stage if running all stages
        if args.all and stage_num < max(stages_to_run):
            # current_checkpoint already updated above
            pass
    
    print("\n" + "=" * 80)
    print("ALL STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    if current_checkpoint:
        print(f"Final model checkpoint: {current_checkpoint}")
    print("=" * 80)


if __name__ == '__main__':
    main()

