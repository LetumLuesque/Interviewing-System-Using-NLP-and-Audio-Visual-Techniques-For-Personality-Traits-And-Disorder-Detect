"""
Main training script for Multimodal Interview Analysis System
"""
import argparse
import yaml
import torch
from pathlib import Path

from utils.helpers import load_config, set_seed, get_device, create_dir_if_not_exists
from utils.logger import setup_logger, save_checkpoint, log_training_metrics, load_checkpoint
from data.loaders.mdpe_loader import create_dataloader
from data.loaders.combined_loader import create_combined_dataloader
from data.processors.video_processor import VideoProcessor
from data.processors.audio_processor import AudioProcessor
from data.processors.text_processor import TextProcessor
from data.augmentation.multimodal_aug import MultimodalAugmentation
from models.multimodal_model import MultimodalInterviewModel
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Interview Analysis Model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training config file')
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml',
                       help='Path to model config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to MDPE dataset directory')
    parser.add_argument('--seumld_data_dir', type=str, default=r'D:\Grad\Datasets\SEUMLD\SEUMLD',
                       help='Path to SEUMLD dataset')
    parser.add_argument('--use_combined', action='store_true',
                       help='Use combined dataset (MDPE + SEUMLD)')
    parser.add_argument('--seumld_fold', type=int, default=0,
                       help='SEUMLD cross-validation fold (0-4)')
    parser.add_argument('--seumld_fine_grained', action='store_true', default=True,
                       help='Use fine-grained (question-level) labels for SEUMLD')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--stage', type=int, default=None,
                       help='Training stage number (1 or 2) for logging purposes')
    parser.add_argument('--fusion_method', type=str, default=None,
                       help='Override late fusion method (mean, concat, weighted, average, max, learned)')
    parser.add_argument('--features_dir', type=str, default='features',
                       help='Path to pre-extracted features directory')
    parser.add_argument('--no_preprocessed', action='store_true',
                       help='Do not use preprocessed data (force on-the-fly processing from original files)')
    
    parser.add_argument('--freeze_backbones', action='store_true',
                       help='Freeze feature extractor backbones')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configurations
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    # Override fusion method if provided
    if args.fusion_method:
        fusion_val = args.fusion_method
        # Map early fusion terminology to late fusion terminology
        if fusion_val == 'mean': fusion_val = 'average'
        elif fusion_val == 'concat': fusion_val = 'learned'
        
        if 'fusion' not in model_config: model_config['fusion'] = {}
        if 'late_fusion' not in model_config['fusion']: model_config['fusion']['late_fusion'] = {}
        model_config['fusion']['late_fusion']['method'] = fusion_val
        
        # Update checkpoint directory structure
        if 'checkpoint' not in train_config: train_config['checkpoint'] = {}
        base_dir = train_config['checkpoint'].get('save_dir', 'checkpoints')
        if args.stage:
            train_config['checkpoint']['save_dir'] = f"{base_dir}/{args.fusion_method}/stage{args.stage}"
        else:
            train_config['checkpoint']['save_dir'] = f"{base_dir}/{args.fusion_method}"
    
    # Setup logging
    log_dir = train_config.get('logging', {}).get('log_dir', 'logs')
    create_dir_if_not_exists(log_dir)
    logger = setup_logger('training', log_dir)
    
    logger.info("=" * 80)
    logger.info("Multimodal Interview Analysis System - Training")
    if args.stage:
        logger.info(f"STAGE {args.stage} TRAINING")
    logger.info("=" * 80)
    logger.info(f"Training config: {args.config}")
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Data directory: {args.data_dir}")
    
    # Get device (respect config, fallback to auto-detect)
    device_config = train_config.get('device', 'auto')
    if device_config == 'auto' or device_config is None:
        device = get_device()
    elif device_config.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize processors
    data_config = model_config.get('data', {})
    video_config = data_config.get('video', {})
    audio_config = data_config.get('audio', {})
    text_config = data_config.get('text', {})
    
    video_processor = VideoProcessor(
        fps=video_config.get('fps', 1),
        frame_size=video_config.get('frame_size', 224),
        use_face_detection=True
    )
    
    audio_processor = AudioProcessor(
        sample_rate=audio_config.get('sample_rate', 16000),
        mfcc_coefficients=audio_config.get('mfcc_coefficients', 40),
        hop_length=audio_config.get('hop_length', 160)
    )
    
    text_processor = TextProcessor(
        tokenizer_name=text_config.get('tokenizer', 'bert-base-uncased'),
        max_length=text_config.get('max_length', 512)
    )
    
    # Data augmentation
    aug_config = train_config.get('augmentation', {})
    augmentation = MultimodalAugmentation(
        video_aug=aug_config.get('video', {}).get('enabled', True),
        audio_aug=aug_config.get('audio', {}).get('enabled', True),
        text_aug=aug_config.get('text', {}).get('enabled', True)
    )
    
    # Create data loaders
    dataset_config = train_config.get('dataset', {})
    
    # Use combined dataset if requested
    if args.use_combined and args.seumld_data_dir:
        datasets_used = ["MDPE"]
        if args.seumld_data_dir:
            datasets_used.append(f"SEUMLD (fold {args.seumld_fold})")
        
        if args.stage:
            logger.info(f"STAGE {args.stage}: Using combined dataset: {' + '.join(datasets_used)}")
        else:
            logger.info(f"Using combined dataset: {' + '.join(datasets_used)}")
        
        # Pass None for processors if using multiprocessing (they will be created in each worker)
        # Otherwise, pass the main process instances
        use_multiprocessing = train_config.get('training', {}).get('num_workers', 0) > 0
        
        training_cfg = train_config.get('training', {})
        persistent_workers = training_cfg.get('persistent_workers', False) and training_cfg.get('num_workers', 0) > 0
        
        train_loader = create_combined_dataloader(
            mdpe_data_dir=args.data_dir,
            seumld_data_dir=args.seumld_data_dir,
            split='train',
            batch_size=training_cfg.get('batch_size', 16),
            shuffle=True,
            num_workers=training_cfg.get('num_workers', 0),
            video_processor=None if use_multiprocessing else video_processor,
            audio_processor=None if use_multiprocessing else audio_processor,
            text_processor=None if use_multiprocessing else text_processor,
            transform=augmentation,
            seumld_fold=args.seumld_fold,
            seumld_fine_grained=args.seumld_fine_grained,
            persistent_workers=persistent_workers,
            balance_classes=True,
            features_dir=args.features_dir,
            use_preprocessed=not args.no_preprocessed
        )
        
        val_loader = create_combined_dataloader(
            mdpe_data_dir=args.data_dir,
            seumld_data_dir=args.seumld_data_dir,
            split='val',
            batch_size=training_cfg.get('batch_size', 16),
            shuffle=False,
            num_workers=training_cfg.get('num_workers', 0),
            video_processor=None if use_multiprocessing else video_processor,
            audio_processor=None if use_multiprocessing else audio_processor,
            text_processor=None if use_multiprocessing else text_processor,
            transform=None,
            seumld_fold=args.seumld_fold,
            seumld_fine_grained=args.seumld_fine_grained,
            persistent_workers=persistent_workers,
            balance_classes=False,  # No balancing for validation - want real metrics
            features_dir=args.features_dir,
            use_preprocessed=not args.no_preprocessed
        )
    else:
        logger.info("Using MDPE dataset only")
        training_cfg = train_config.get('training', {})
        
        # Use multiprocessing logic: pass None for processors if using workers
        # This forces the Dataset to create fresh pickle-safe processors in each worker
        use_multiprocessing = training_cfg.get('num_workers', 0) > 0
        
        train_loader = create_dataloader(
            data_dir=args.data_dir,
            split='train',
            batch_size=training_cfg.get('batch_size', 16),
            shuffle=True,
            num_workers=training_cfg.get('num_workers', 0),
            video_processor=None if use_multiprocessing else video_processor,
            audio_processor=None if use_multiprocessing else audio_processor,
            text_processor=None if use_multiprocessing else text_processor,
            transform=augmentation,
            features_dir=args.features_dir
        )
        
        val_loader = create_dataloader(
            data_dir=args.data_dir,
            split='val',
            batch_size=training_cfg.get('batch_size', 16),
            shuffle=False,
            num_workers=training_cfg.get('num_workers', 0),
            video_processor=None if use_multiprocessing else video_processor,
            audio_processor=None if use_multiprocessing else audio_processor,
            text_processor=None if use_multiprocessing else text_processor,
            transform=None,
            features_dir=args.features_dir
        )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = MultimodalInterviewModel(model_config)
    
    # FREEZE BACKBONES STRATEGY
    if args.freeze_backbones:
        logger.info("STRATEGY: Freezing Backbones for Fine-Tuning. Only Heads/Fusion trainable.")
        modules_to_freeze = [
            model.expr3dnet, model.vit_encoder,
            model.acoustic_net, model.wavlm_encoder,
            model.text_trait_net
        ]
        count = 0
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
                count += 1
        logger.info(f"Froze {count} parameters.")
    model.to(device)
    
    # Compile model for faster training (PyTorch 2.0+)
    compile_model = train_config.get('training', {}).get('compile_model', False)
    if compile_model and hasattr(torch, 'compile'):
        try:
            logger.info("Compiling model for faster training (PyTorch 2.0+)...")
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compiled successfully!")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer - merge model_config training settings into train_config for early stopping
    # Early stopping patience is in model_config, but trainer expects it in config['training']
    if 'early_stopping_patience' not in train_config.get('training', {}):
        model_training_config = model_config.get('training', {})
        if 'early_stopping_patience' in model_training_config:
            if 'training' not in train_config:
                train_config['training'] = {}
            train_config['training']['early_stopping_patience'] = model_training_config['early_stopping_patience']
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        logger=logger
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        from utils.logger import load_checkpoint
        resumed_epoch, resumed_loss = load_checkpoint(args.resume, model, trainer.optimizer, trainer.scheduler)
        logger.info(f"Resumed from checkpoint: {args.resume} (epoch {resumed_epoch}, loss {resumed_loss})")
        # Restore best model tracking for early stopping
        trainer.best_val_loss = resumed_loss
        trainer.best_epoch = resumed_epoch
        trainer.early_stopping_counter = 0  # Reset counter when resuming
        logger.info(f"Restored best model state: epoch {resumed_epoch}, val_loss {resumed_loss:.4f}")
    
    # Train
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()




















