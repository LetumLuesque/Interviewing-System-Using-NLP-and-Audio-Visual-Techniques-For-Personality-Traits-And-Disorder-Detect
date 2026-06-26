"""
Training module for multimodal interview analysis system
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR, ReduceLROnPlateau
from tqdm import tqdm
import os
import sys
from datetime import datetime
from typing import Dict, Optional

from utils.logger import setup_logger, save_checkpoint, log_training_metrics
from utils.helpers import get_device
from training.loss_functions import CombinedLoss
from training.metrics import MetricsCalculator



# Custom wrapper to force flush and handle Windows encoding
class TqdmFileWrapper:
    def __init__(self, file):
        self.file = file
    def write(self, x):
        if len(x.strip()) > 0:
            try:
                self.file.write(x)
                self.file.flush()
            except (OSError, UnicodeEncodeError):
                pass
    def flush(self):
        try:
            self.file.flush()
        except OSError:
            pass

class Trainer:
    """
    Trainer for multimodal interview analysis model
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: Optional[torch.device] = None,
        logger=None
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            logger: Logger instance
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or get_device()
        self.logger = logger or setup_logger('trainer')
        log_config = config.get('logging', {})
        self.batch_log_interval = max(1, int(log_config.get('batch_log_interval', 5)))
        
        # Move model to device
        self.model.to(self.device)

        # Freeze personality head if weight is 0
        if config.get('loss_weights', {}).get('personality_mse', 1.0) < 1e-6:
            self.logger.info("Personality MSE weight is 0.0 -> Freezing personality prediction heads")
            if hasattr(self.model, 'prediction_head'):
                # Freeze trait heads
                if hasattr(self.model.prediction_head, 'trait_heads'):
                    for param in self.model.prediction_head.trait_heads.parameters():
                        param.requires_grad = False
                    self.logger.info("Frozen trait_heads")
                # Freeze confidence head
                if hasattr(self.model.prediction_head, 'confidence_head'):
                    for param in self.model.prediction_head.confidence_head.parameters():
                        param.requires_grad = False
                    self.logger.info("Frozen confidence_head")

        # Freeze Deception head if weight is 0
        if config.get('loss_weights', {}).get('deception_ce', 1.0) < 1e-6:
            self.logger.info("Deception CE weight is 0.0 -> Freezing deception prediction head")
            freeze_shared_with_deception = bool(
                config.get('freeze_shared_layers_with_deception', True)
            )
            if hasattr(self.model, 'prediction_head'):
                # Freeze specific deception head
                if hasattr(self.model.prediction_head, 'deception_head'):
                    for param in self.model.prediction_head.deception_head.parameters():
                        param.requires_grad = False
                    self.logger.info("Frozen deception_head")
                
                # Optional shared-freeze mode for deception head freeze.
                # For corr-recovery runs, keep shared layers trainable by setting
                # freeze_shared_layers_with_deception: false in training config.
                if freeze_shared_with_deception:
                    if hasattr(self.model.prediction_head, 'mlp'):
                        for param in self.model.prediction_head.mlp.parameters():
                            param.requires_grad = False
                        self.logger.info("Frozen shared MLP in prediction_head")
                    
                    if hasattr(self.model.prediction_head, 'rnn'):
                        for param in self.model.prediction_head.rnn.parameters():
                            param.requires_grad = False
                        self.logger.info("Frozen shared RNN in prediction_head")
                    
                    for middleware in ['visual_fusion', 'audio_fusion', 'text_fusion', 'early_fusion']:
                        if hasattr(self.model, middleware):
                            module = getattr(self.model, middleware)
                            for param in module.parameters():
                                param.requires_grad = False
                            self.logger.info(f"Frozen middleware: {middleware}")
                else:
                    self.logger.info(
                        "Keeping shared prediction/fusion layers trainable "
                        "(freeze_shared_layers_with_deception=false)"
                    )
        
        # Loss function - use MAE for better regression performance
        loss_config = config.get('loss_weights', {})
        loss_behavior_config = config.get('loss', {})
        use_focal_loss = bool(loss_behavior_config.get('use_focal_loss', False))
        deception_class_weights = None
        if not use_focal_loss:
            deception_class_weights = self._compute_deception_class_weights()
        self.criterion = CombinedLoss(
            personality_weight=loss_config.get('personality_mse', 1.0),
            deception_weight=loss_config.get('deception_ce', 0.5),
            personality_ccc_weight=loss_config.get('personality_ccc', 0.0),
            cultural_bias_weight=loss_config.get('cultural_bias', 0.3),
            use_mae=True,  # Use MAE instead of MSE for personality prediction
            deception_class_weights=deception_class_weights,
            use_focal_loss=use_focal_loss
        )
        if deception_class_weights is not None:
            self.logger.info(
                f"Deception loss: weighted CE (class weights={deception_class_weights.tolist()})"
            )
        else:
            self.logger.info("Deception loss: focal loss")
        
        # Optimizer
        opt_config = config.get('optimizer', {})
        base_lr = float(opt_config.get('lr', 1e-4))
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=base_lr,
            weight_decay=float(opt_config.get('weight_decay', 1e-5)),
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=float(opt_config.get('eps', 1e-8))
        )
        
        # Scheduler with warmup
        sched_config = config.get('scheduler', {})
        warmup_epochs = config.get('training', {}).get('warmup_epochs', 5)
        
        if sched_config.get('type') == 'reduce_on_plateau':
            # ReduceLROnPlateau: reduces LR when val_loss stops improving
            # Better for fine-tuning than cosine restarts (no LR overshoot)
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(sched_config.get('factor', 0.5)),
                patience=int(sched_config.get('patience', 5)),
                min_lr=float(sched_config.get('min_lr', 1e-7))
            )
            self.warmup_epochs = 0
            self.scheduler_type = 'reduce_on_plateau'
        elif sched_config.get('type') == 'cosine_annealing_warm_restarts':
            # Create warmup scheduler
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,  # Start at 10% of base LR
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            
            # Create cosine annealing scheduler
            cosine_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.get('T_0', 10),
                T_mult=sched_config.get('T_mult', 2),
                eta_min=float(sched_config.get('eta_min', 1e-6))
            )
            
            # Combine warmup + cosine annealing
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            self.warmup_epochs = warmup_epochs
            self.scheduler_type = 'cosine'
        else:
            self.scheduler = None
            self.warmup_epochs = 0
            self.scheduler_type = 'none'
        
        # Training config
        self.num_epochs = config.get('training', {}).get('num_epochs', 100)
        self.gradient_clip = config.get('training', {}).get('gradient_clip', 1.0)
        self.mixed_precision = config.get('mixed_precision', False)
        
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Metrics
        self.metrics_calculator = MetricsCalculator()
        
        # Best model tracking
        validation_config = config.get('validation', {})
        metric_name = validation_config.get('metric', 'val_loss')
        self.selection_metric = str(metric_name).replace('val_', '')
        self.maximize_metric = self.selection_metric in {
            'deception_accuracy',
            'deception_precision',
            'deception_recall',
            'deception_f1',
            'deception_balanced_accuracy',
            'corr_mean',
            'joint_score'
        }
        self.best_metric_value = float('-inf') if self.maximize_metric else float('inf')
        self.best_val_loss = float('inf')  # Legacy tracker (kept for compatibility)
        self.best_epoch = 0
        self.start_epoch = 1
        
        # Early stopping
        training_config = config.get('training', {})
        self.early_stopping_patience = training_config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0
        self.early_stopping_enabled = self.early_stopping_patience > 0
        
        if self.early_stopping_enabled:
            self.logger.info(f"Early stopping enabled with patience: {self.early_stopping_patience} epochs")
        else:
            self.logger.info("Early stopping disabled")
        
        # Checkpoint directory
        checkpoint_dir = config.get('checkpoint', {}).get('save_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

    def _collect_deception_labels(self, dataset) -> list:
        """Collect deception labels from Dataset/ConcatDataset metadata."""
        labels = []
        if isinstance(dataset, ConcatDataset):
            for ds in dataset.datasets:
                labels.extend(self._collect_deception_labels(ds))
            return labels

        if not hasattr(dataset, 'samples'):
            return labels

        samples = dataset.samples
        if hasattr(samples, 'iloc'):
            # DataFrame-backed datasets
            if 'label' in samples.columns:
                labels.extend(samples['label'].astype(int).tolist())
            elif 'deception' in samples.columns:
                labels.extend(samples['deception'].astype(int).tolist())
            return labels

        if isinstance(samples, list):
            for sample in samples:
                if 'labels' in sample and 'deception' in sample['labels']:
                    value = sample['labels']['deception']
                    if torch.is_tensor(value):
                        value = value.item()
                    labels.append(int(value))
                elif 'label' in sample:
                    labels.append(int(sample['label']))
        return labels

    def _compute_deception_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights from the raw training dataset."""
        labels = self._collect_deception_labels(self.train_loader.dataset)
        if not labels:
            return torch.tensor([1.0, 1.0], dtype=torch.float32)

        counts = np.bincount(np.array(labels, dtype=np.int64), minlength=2).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        total = float(counts.sum())
        # Balanced CE formulation: w_c = N / (K * n_c)
        weights = total / (2.0 * counts)

        # Guardrail against extreme ratios that can destabilize training
        max_ratio = float(self.config.get('loss', {}).get('max_class_weight_ratio', 5.0))
        min_w = float(weights.min())
        max_w = float(weights.max())
        if min_w > 0 and (max_w / min_w) > max_ratio:
            scale = max_ratio / (max_w / min_w)
            weights = np.array([weights[0], weights[1] * scale], dtype=np.float32) \
                if weights[1] > weights[0] else np.array([weights[0] * scale, weights[1]], dtype=np.float32)

        return torch.tensor(weights, dtype=torch.float32)

    def _get_selection_metric_value(self, val_metrics: Dict[str, float]) -> float:
        """Get the scalar metric value used for checkpoint selection."""
        key = self.selection_metric
        if key in val_metrics:
            return float(val_metrics[key])
        if key == 'loss':
            return float(val_metrics.get('loss', float('inf')))
        return float(val_metrics.get('loss', float('inf')))
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # CRITICAL FIX: Force frozen modules to eval mode to prevent BatchNorm statistics update (drift)
        # 1. Freeze Deception Head if weight is 0
        if self.config.get('loss_weights', {}).get('deception_ce', 1.0) < 1e-6:
             freeze_shared_with_deception = bool(
                 self.config.get('freeze_shared_layers_with_deception', True)
             )
             if hasattr(self.model, 'prediction_head'):
                 if hasattr(self.model.prediction_head, 'deception_head'):
                     self.model.prediction_head.deception_head.eval()
                 if freeze_shared_with_deception:
                     if hasattr(self.model.prediction_head, 'mlp'):
                         self.model.prediction_head.mlp.eval()
                     if hasattr(self.model.prediction_head, 'rnn'):
                         self.model.prediction_head.rnn.eval()
                     
                     for middleware in ['visual_fusion', 'audio_fusion', 'text_fusion', 'early_fusion']:
                         if hasattr(self.model, middleware):
                             module = getattr(self.model, middleware)
                             module.eval()
        
        # 2. Freeze Backbones if requested
        if self.config.get('freeze_backbones', False):
            if hasattr(self.model, 'vit_encoder'): self.model.vit_encoder.eval()
            if hasattr(self.model, 'expr3dnet'): self.model.expr3dnet.eval()
            if hasattr(self.model, 'acoustic_net'): self.model.acoustic_net.eval()
            if hasattr(self.model, 'wavlm_encoder'): self.model.wavlm_encoder.eval()
            if hasattr(self.model, 'text_trait_net'): self.model.text_trait_net.eval()

        total_loss = 0.0
        num_batches = 0
        
        # Windows-compatible progress bar
        # Disable tqdm progress bar on Windows to avoid console encoding issues
        # Use logger instead for progress updates
        # Enable tqdm progress bar
        # disable_tqdm = sys.platform == 'win32' 
        disable_tqdm = False # ENABLED for real-time visibility
        try:
            # Force stdout and flush using wrapper
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]', 
                       file=TqdmFileWrapper(sys.stdout), disable=disable_tqdm, 
                       dynamic_ncols=False, leave=True,
                       ncols=80, # Safer width for Windows cmd
                       mininterval=0.5, # Slower updates to prevent flickering
                       ascii=True) # ASCII characters are safer on Windows
        except (OSError, UnicodeEncodeError, ValueError):
            # Fallback: disable tqdm completely and use logger
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]', 
                       disable=True)
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device with non_blocking for faster transfer
            non_blocking = self.device.type == 'cuda'
            video = batch['video'].to(self.device, non_blocking=non_blocking) if batch['video'] is not None else None
            audio = batch['audio'].to(self.device, non_blocking=non_blocking) if batch['audio'] is not None else None
            text = {k: v.to(self.device, non_blocking=non_blocking) for k, v in batch['text'].items()} if batch['text'] is not None else None
            
            personality_target = batch['personality'].to(self.device, non_blocking=non_blocking)
            deception_target = batch['deception'].to(self.device, non_blocking=non_blocking)
            cultural_groups = batch['cultural_groups']
            
            # Create personality mask: True for samples with real personality labels
            # SEUMLD (cultural_group=5) has dummy personality labels, exclude them
            if isinstance(cultural_groups, torch.Tensor):
                personality_mask = (cultural_groups != 5).to(self.device)
            else:
                personality_mask = torch.tensor([cg != 5 for cg in cultural_groups], dtype=torch.bool, device=self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            

            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    predictions = self.model(video, audio, text)
                    
                    # Check for NaN in predictions with more details
                    if torch.any(torch.isnan(predictions['traits'])) or torch.any(torch.isnan(predictions['deception_logits'])):
                        self.logger.warning(f"NaN detected in predictions at batch {batch_idx}")
                        # Debug: Check input ranges
                        if video is not None:
                            self.logger.warning(f"  Video: min={video.min().item():.4f}, max={video.max().item():.4f}, has_nan={torch.any(torch.isnan(video))}")
                        if audio is not None:
                            self.logger.warning(f"  Audio: min={audio.min().item():.4f}, max={audio.max().item():.4f}, has_nan={torch.any(torch.isnan(audio))}")
                        if text is not None:
                            self.logger.warning(f"  Text input_ids: min={text['input_ids'].min().item()}, max={text['input_ids'].max().item()}")
                        self.logger.warning(f"  Traits: min={predictions['traits'].min().item():.4f}, max={predictions['traits'].max().item():.4f}")
                        continue
                    
                    loss_dict = self.criterion(
                        predictions,
                        {'personality': personality_target, 'deception': deception_target},
                        personality_mask=personality_mask
                    )
                    loss = loss_dict['total_loss']
                    
                    # Check for NaN in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                        continue
                
                if loss.requires_grad:
                    self.scaler.scale(loss).backward()
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.logger.debug(f"Batch {batch_idx} has no gradient (skipping optimization)")
            else:
                predictions = self.model(video, audio, text)
                
                # Check for NaN in predictions
                if torch.any(torch.isnan(predictions['traits'])) or torch.any(torch.isnan(predictions['deception_logits'])):
                    self.logger.warning(f"NaN detected in predictions at batch {batch_idx}")
                    continue
                
                loss_dict = self.criterion(
                    predictions,
                    {'personality': personality_target, 'deception': deception_target},
                    personality_mask=personality_mask
                )
                loss = loss_dict['total_loss']
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                    continue
                
                if loss.requires_grad:
                    loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                else:
                    self.logger.debug(f"Batch {batch_idx} has no gradient (skipping optimization)")
            
            
            # Ensure targets match prediction size (in case of trimming)
            curr_batch_size = predictions['traits'].shape[0]
            if personality_target.shape[0] > curr_batch_size:
                personality_target = personality_target[:curr_batch_size]
                deception_target = deception_target[:curr_batch_size]
                if cultural_groups is not None:
                    cultural_groups = cultural_groups[:curr_batch_size]

            # Update metrics - only pass samples that have valid personality labels
            # (MDPE samples). Deception metrics use all samples.
            # Use the personality_mask to filter appropriately for personality traits.
            # Also filter cultural_groups to match personality samples for fairness metrics.
            p_pred = predictions['traits'][personality_mask]
            p_target = personality_target[personality_mask]
            
            # Filter cultural groups if they exist
            if cultural_groups is not None:
                if isinstance(cultural_groups, torch.Tensor):
                    # Ensure indexing happens on the same device
                    p_groups = cultural_groups[personality_mask.to(cultural_groups.device)].tolist()
                else:
                    p_groups = [g for i, g in enumerate(cultural_groups) if personality_mask[i].item()]
            else:
                p_groups = None
            
            self.metrics_calculator.update(
                p_pred,
                p_target,
                predictions['deception_logits'],
                deception_target,
                p_groups
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar (with error handling for Windows)
            if not disable_tqdm:
                try:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                except (OSError, UnicodeEncodeError, ValueError):
                    # Skip progress bar update on Windows console errors
                    pass
            
            # Always log to file/console handler at intervals
            if (batch_idx + 1) % self.batch_log_interval == 0 or batch_idx == 0:
                msg = f"Epoch {epoch} [Train] Batch {batch_idx + 1}/{len(self.train_loader)} | Loss: {loss.item():.4f}"
                # Only log to file if tqdm is enabled to avoid double printing in console
                if disable_tqdm:
                    self.logger.info(msg)
                    # Force flush for Windows background processes
                    for handler in self.logger.handlers:
                        handler.flush()
        
        # Compute metrics
        metrics = self.metrics_calculator.compute()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            # Windows-compatible progress bar
            # Enable tqdm progress bar
            # disable_tqdm = sys.platform == 'win32'
            disable_tqdm = False # ENABLED for real-time visibility
            try:
                # Force stdout and flush
                pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]', 
                           file=TqdmFileWrapper(sys.stdout), disable=disable_tqdm, 
                           dynamic_ncols=False, leave=True,
                           ncols=80, # Safer width
                           mininterval=0.5,
                           ascii=True)
            except (OSError, UnicodeEncodeError, ValueError):
                pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]', disable=True)
            
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device with non_blocking for faster transfer
                non_blocking = self.device.type == 'cuda'
                video = batch['video'].to(self.device, non_blocking=non_blocking) if batch['video'] is not None else None
                audio = batch['audio'].to(self.device, non_blocking=non_blocking) if batch['audio'] is not None else None
                text = {k: v.to(self.device, non_blocking=non_blocking) for k, v in batch['text'].items()} if batch['text'] is not None else None
                
                personality_target = batch['personality'].to(self.device, non_blocking=non_blocking)
                deception_target = batch['deception'].to(self.device, non_blocking=non_blocking)
                cultural_groups = batch['cultural_groups']
                
                # Create personality mask: exclude SEUMLD (cultural_group=5) from personality loss
                if isinstance(cultural_groups, torch.Tensor):
                    personality_mask = (cultural_groups != 5).to(self.device)
                else:
                    personality_mask = torch.tensor([cg != 5 for cg in cultural_groups], dtype=torch.bool, device=self.device)
                
                # Forward pass
                if self.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        predictions = self.model(video, audio, text)
                        loss_dict = self.criterion(
                            predictions,
                            {'personality': personality_target, 'deception': deception_target},
                            personality_mask=personality_mask
                        )
                        loss = loss_dict['total_loss']
                else:
                    predictions = self.model(video, audio, text)
                    loss_dict = self.criterion(
                        predictions,
                        {'personality': personality_target, 'deception': deception_target},
                        personality_mask=personality_mask
                    )
                    loss = loss_dict['total_loss']
                
                # Ensure targets match prediction size (in case of trimming)
                curr_batch_size = predictions['traits'].shape[0]
                if personality_target.shape[0] > curr_batch_size:
                    personality_target = personality_target[:curr_batch_size]
                    deception_target = deception_target[:curr_batch_size]
                    if cultural_groups is not None:
                        cultural_groups = cultural_groups[:curr_batch_size]
                
                # Update metrics - only pass samples that have valid personality labels
                # (MDPE samples). Deception metrics use all samples.
                # Use the personality_mask to filter appropriately.
                # Update metrics - only pass samples that have valid personality labels
                # (MDPE samples). Deception metrics use all samples.
                # Use the personality_mask to filter appropriately for personality traits.
                # Also filter cultural_groups to match personality samples for fairness metrics.
                p_pred = predictions['traits'][personality_mask]
                p_target = personality_target[personality_mask]
                
                # Filter cultural groups if they exist
                if cultural_groups is not None:
                    if isinstance(cultural_groups, torch.Tensor):
                        # Ensure indexing happens on the same device
                        p_groups = cultural_groups[personality_mask.to(cultural_groups.device)].tolist()
                    else:
                        p_groups = [g for i, g in enumerate(cultural_groups) if personality_mask[i].item()]
                else:
                    p_groups = None
                
                self.metrics_calculator.update(
                    p_pred,
                    p_target,
                    predictions['deception_logits'],
                    deception_target,
                    p_groups
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar (with error handling for Windows)
                if not disable_tqdm:
                    try:
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    except (OSError, UnicodeEncodeError, ValueError):
                        pass
                
                # Log progress every batch (live updates for Windows)
                if disable_tqdm:
                    if (batch_idx + 1) % self.batch_log_interval == 0 or batch_idx == 0:
                        msg = f"Epoch {epoch} [Val] Batch {batch_idx + 1}/{len(self.val_loader)} | Loss: {loss.item():.4f}"
                        try:
                            print(msg, flush=True)
                        except OSError:
                            pass  # stdout may be gone on Windows
                        self.logger.info(msg)
                        # Force flush for Windows background processes
                        for handler in self.logger.handlers:
                            handler.flush()
        
        # Compute metrics
        metrics = self.metrics_calculator.compute()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def train(self):
        """
        Main training loop
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.num_epochs}")
        self.logger.info(
            f"Checkpoint selection metric: {self.selection_metric} "
            f"({'max' if self.maximize_metric else 'min'})"
        )
        if self.best_epoch > 0:
            self.logger.info(
                f"Best model so far: epoch {self.best_epoch}, "
                f"{self.selection_metric} {self.best_metric_value:.4f}"
            )
        
        try:
            start_epoch = max(1, int(self.start_epoch))
            for epoch in range(start_epoch, self.num_epochs + 1):
                self.logger.info(f"{'='*80}")
                self.logger.info(f"Starting Epoch {epoch}/{self.num_epochs}")
                self.logger.info(f"{'='*80}")
                
                try:
                    # Train
                    self.logger.info(f"Training epoch {epoch}...")
                    train_metrics = self.train_epoch(epoch)
                    log_training_metrics(epoch, train_metrics, self.logger)
                    self.logger.info(f"Completed training for epoch {epoch}")
                    
                    # Validate
                    self.logger.info(f"Validating epoch {epoch}...")
                    val_metrics = self.validate(epoch)
                    log_training_metrics(epoch, {f'val_{k}': v for k, v in val_metrics.items()}, self.logger)
                    self.logger.info(f"Completed validation for epoch {epoch}")
                    
                    # Extract val_loss first (needed by scheduler and checkpoint logic)
                    val_loss = val_metrics['loss']
                    selection_metric_value = self._get_selection_metric_value(val_metrics)
                    
                    # Update scheduler
                    if self.scheduler:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        if self.scheduler_type == 'reduce_on_plateau':
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()
                        new_lr = self.optimizer.param_groups[0]['lr']
                        self.logger.info(f"Learning rate ({self.scheduler_type}): {current_lr:.6f} -> {new_lr:.6f}")
                    
                    # Save checkpoint
                    # Handle NaN/Inf metric - don't update best if metric is invalid
                    metric_tensor = torch.tensor(selection_metric_value)
                    if torch.isnan(metric_tensor) or torch.isinf(metric_tensor):
                        self.logger.warning(
                            f"Invalid validation metric ({self.selection_metric}) at epoch {epoch}, "
                            "skipping best model update"
                        )
                        if self.early_stopping_enabled:
                            self.early_stopping_counter += 1
                    else:
                        improved = (
                            selection_metric_value > self.best_metric_value
                            if self.maximize_metric
                            else selection_metric_value < self.best_metric_value
                        )

                        if improved:
                        # Improvement found
                            improvement = (
                                selection_metric_value - self.best_metric_value
                                if self.maximize_metric
                                else self.best_metric_value - selection_metric_value
                            )
                            self.best_metric_value = selection_metric_value
                            self.best_val_loss = val_loss
                            self.best_epoch = epoch
                            self.early_stopping_counter = 0  # Reset counter
                            
                            checkpoint_path = os.path.join(
                                self.checkpoint_dir,
                                f'best_model_epoch_{epoch}.pt'
                            )
                            save_checkpoint(
                                self.model,
                                self.optimizer,
                                self.scheduler,
                                epoch,
                                val_loss,
                                checkpoint_path
                            )
                            self.logger.info(
                                f"Saved best model at epoch {epoch} "
                                f"({self.selection_metric}: {selection_metric_value:.4f}, "
                                f"improvement: {improvement:.4f})"
                            )
                            
                            # Save summary to text file for easy inspection
                            summary_path = os.path.join(self.checkpoint_dir, 'best_model_summary.txt')
                            with open(summary_path, 'w') as f:
                                f.write(f"Best Model Summary\n")
                                f.write(f"==================\n")
                                f.write(f"Epoch: {epoch}\n")
                                f.write(f"{self.selection_metric}: {selection_metric_value:.4f}\n")
                                f.write(f"Validation Loss: {val_loss:.4f}\n")
                                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"\nMetrics:\n")
                                for k, v in val_metrics.items():
                                    if isinstance(v, torch.Tensor):
                                        v = v.item()
                                    f.write(f"  {k}: {v:.4f}\n")
                        else:
                            # No improvement
                            if self.early_stopping_enabled:
                                self.early_stopping_counter += 1
                                self.logger.info(
                                    f"No improvement for {self.early_stopping_counter}/"
                                    f"{self.early_stopping_patience} epochs "
                                    f"(best: {self.best_metric_value:.4f} at epoch {self.best_epoch})"
                                )
                    
                    # Periodic checkpoint
                    checkpoint_config = self.config.get('checkpoint', {})
                    if epoch % checkpoint_config.get('save_frequency', 5) == 0:
                        checkpoint_path = os.path.join(
                            self.checkpoint_dir,
                            f'checkpoint_epoch_{epoch}.pt'
                        )
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            val_loss,
                            checkpoint_path
                        )
                        self.logger.info(f"Saved periodic checkpoint at epoch {epoch}")
                    
                    self.logger.info(f"Epoch {epoch} completed successfully")
                    
                    # Check early stopping
                    if self.early_stopping_enabled and self.early_stopping_counter >= self.early_stopping_patience:
                        self.logger.info(f"{'='*80}")
                        self.logger.info(f"Early stopping triggered!")
                        self.logger.info(
                            f"No improvement in {self.selection_metric} "
                            f"for {self.early_stopping_patience} epochs"
                        )
                        self.logger.info(
                            f"Best model was at epoch {self.best_epoch} "
                            f"with {self.selection_metric}: {self.best_metric_value:.4f}"
                        )
                        self.logger.info(f"Stopping training at epoch {epoch}")
                        self.logger.info(f"{'='*80}")
                        break
                    
                except Exception as e:
                    self.logger.error(f"Error during epoch {epoch}: {str(e)}", exc_info=True)
                    self.logger.error(f"Training stopped at epoch {epoch}")
                    raise
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error during training: {str(e)}", exc_info=True)
            raise
        finally:
            if self.best_epoch > 0:
                self.logger.info(
                    f"Training completed. Best model at epoch {self.best_epoch} "
                    f"({self.selection_metric}: {self.best_metric_value:.4f}, "
                    f"val_loss: {self.best_val_loss:.4f})"
                )
            else:
                self.logger.info("Training completed. No best checkpoint was selected.")




















