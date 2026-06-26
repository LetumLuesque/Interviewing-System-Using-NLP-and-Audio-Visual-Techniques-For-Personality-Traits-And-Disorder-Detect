"""
Training module for multimodal interview analysis system
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
import os
import sys
import glob
from typing import Dict, Optional

from utils.logger import setup_logger, save_checkpoint, log_training_metrics
from utils.helpers import get_device
from training.loss_functions import CombinedLoss
from training.metrics import MetricsCalculator


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
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function - use MAE for better regression performance
        loss_config = config.get('loss_weights', {})
        loss_behavior_config = config.get('loss', {})
        use_focal_loss = bool(loss_behavior_config.get('use_focal_loss', False))
        deception_class_weights = None
        if not use_focal_loss:
            deception_class_weights = self._compute_deception_class_weights_from_dataset()
        self.criterion = CombinedLoss(
            personality_weight=loss_config.get('personality_mse', 1.0),
            deception_weight=loss_config.get('deception_ce', 0.5),
            cultural_bias_weight=loss_config.get('cultural_bias', 0.3),
            use_mae=True,  # Use MAE instead of MSE for personality prediction
            deception_class_weights=deception_class_weights,
            use_focal_loss=use_focal_loss
        )
        
        # Optimizer
        opt_config = config.get('optimizer', {})
        base_lr = float(opt_config.get('lr', 1e-4))
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=float(opt_config.get('weight_decay', 1e-5)),
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=float(opt_config.get('eps', 1e-8))
        )
        
        # Scheduler with warmup
        sched_config = config.get('scheduler', {})
        warmup_epochs = config.get('training', {}).get('warmup_epochs', 5)
        
        if sched_config.get('type') == 'cosine_annealing_warm_restarts':
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
        else:
            self.scheduler = None
            self.warmup_epochs = 0
        
        # Training config
        self.num_epochs = config.get('training', {}).get('num_epochs', 100)
        self.gradient_clip = config.get('training', {}).get('gradient_clip', 1.0)
        self.mixed_precision = config.get('mixed_precision', False)
        
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Metrics (supports configurable deception decision threshold)
        validation_config = config.get('validation', {})
        self.deception_threshold = float(validation_config.get('deception_threshold', 0.5))
        self.metrics_calculator = MetricsCalculator(deception_threshold=self.deception_threshold)
        
        # Checkpoint/selection strategy
        checkpoint_config = config.get('checkpoint', {})
        self.checkpoint_monitor = checkpoint_config.get('monitor', 'val_loss')
        self.checkpoint_mode = checkpoint_config.get('mode', 'min')
        self.metric_tie_epsilon = float(checkpoint_config.get('metric_tie_epsilon', 1e-8))
        self.keep_last_n = int(checkpoint_config.get('keep_last_n', 1))
        self.save_frequency = int(checkpoint_config.get('save_frequency', 0))

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_corr_mean = float('-inf')
        self.best_balanced_accuracy = float('-inf')
        self.best_epoch = 0
        
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
        checkpoint_dir = checkpoint_config.get('save_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self.logger.info(
            f"Checkpoint monitor strategy: monitor={self.checkpoint_monitor}, mode={self.checkpoint_mode}"
        )
        self.logger.info(f"Deception decision threshold: {self.deception_threshold:.3f}")
        if deception_class_weights is not None:
            self.logger.info(f"Deception class weights: {deception_class_weights.tolist()}")
        self.logger.info(f"Deception focal loss enabled: {use_focal_loss}")

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

    def _compute_deception_class_weights_from_dataset(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights from the raw training dataset.
        Returns [weight_truthful, weight_deceptive].
        """
        labels = self._collect_deception_labels(self.train_loader.dataset)
        if not labels:
            return torch.tensor([1.0, 1.0], dtype=torch.float32)

        counts = np.bincount(np.array(labels, dtype=np.int64), minlength=2).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        total = float(counts.sum())
        # Balanced CE formulation: w_c = N / (K * n_c)
        weights = total / (2.0 * counts)

        # Guardrail against extreme ratios that can destabilize training.
        max_ratio = float(self.config.get('loss', {}).get('max_class_weight_ratio', 5.0))
        min_w = float(weights.min())
        max_w = float(weights.max())
        if min_w > 0 and (max_w / min_w) > max_ratio:
            scale = max_ratio / (max_w / min_w)
            if weights[1] > weights[0]:
                weights = np.array([weights[0], weights[1] * scale], dtype=np.float32)
            else:
                weights = np.array([weights[0] * scale, weights[1]], dtype=np.float32)

        return torch.tensor(weights, dtype=torch.float32)

    def _prune_best_checkpoints(self):
        """Keep only the latest N best_model checkpoints."""
        if self.keep_last_n <= 0:
            return
        pattern = os.path.join(self.checkpoint_dir, "best_model_epoch_*.pt")
        checkpoints = sorted(
            [p for p in glob.glob(pattern)],
            key=lambda p: os.path.getmtime(p),
            reverse=True
        )
        for old_ckpt in checkpoints[self.keep_last_n:]:
            try:
                os.remove(old_ckpt)
            except OSError:
                pass

    @staticmethod
    def _is_invalid_metric(value: float) -> bool:
        metric_t = torch.tensor(value)
        return bool(torch.isnan(metric_t) or torch.isinf(metric_t))

    def _evaluate_improvement(self, val_metrics: Dict[str, float]):
        """
        Evaluate whether current epoch is better than the previously saved best model.

        Returns:
            (is_improved: bool, reason: str)
        """
        val_loss = float(val_metrics.get('loss', float('inf')))
        corr_mean = float(val_metrics.get('corr_mean', float('-inf')))
        bal_acc = float(val_metrics.get('deception_balanced_accuracy', float('-inf')))

        monitor = self.checkpoint_monitor
        eps = self.metric_tie_epsilon

        # Stage 1 strategy: maximize personality corr_mean
        if monitor == 'corr_mean':
            if self._is_invalid_metric(corr_mean):
                return False, "Invalid corr_mean (NaN/Inf)"
            if corr_mean > self.best_corr_mean:
                return True, f"corr_mean improved: {self.best_corr_mean:.4f} -> {corr_mean:.4f}"
            return False, f"corr_mean did not improve (best={self.best_corr_mean:.4f}, current={corr_mean:.4f})"

        # Stage 2 strategy: minimize val_loss, tie-break by maximizing balanced accuracy
        if monitor in ('val_loss_balanced_accuracy', 'val_loss_then_bal_acc'):
            if self._is_invalid_metric(val_loss):
                return False, "Invalid val_loss (NaN/Inf)"

            if val_loss < (self.best_val_loss - eps):
                return True, f"val_loss improved: {self.best_val_loss:.4f} -> {val_loss:.4f}"

            if abs(val_loss - self.best_val_loss) <= eps and not self._is_invalid_metric(bal_acc):
                if bal_acc > self.best_balanced_accuracy:
                    return True, (
                        f"val_loss tied ({val_loss:.4f}); balanced_accuracy improved: "
                        f"{self.best_balanced_accuracy:.4f} -> {bal_acc:.4f}"
                    )
            return False, (
                f"val_loss/bal_acc did not improve "
                f"(best_loss={self.best_val_loss:.4f}, curr_loss={val_loss:.4f}, "
                f"best_bal_acc={self.best_balanced_accuracy:.4f}, curr_bal_acc={bal_acc:.4f})"
            )

        # Default strategy: minimize val_loss
        if self._is_invalid_metric(val_loss):
            return False, "Invalid val_loss (NaN/Inf)"
        if val_loss < self.best_val_loss:
            return True, f"val_loss improved: {self.best_val_loss:.4f} -> {val_loss:.4f}"
        return False, f"val_loss did not improve (best={self.best_val_loss:.4f}, current={val_loss:.4f})"

    def _update_best_trackers(self, epoch: int, val_metrics: Dict[str, float]):
        """
        Update all best trackers whenever the selected checkpoint criterion improves.
        """
        self.best_epoch = epoch
        if 'loss' in val_metrics:
            self.best_val_loss = float(val_metrics['loss'])
        if 'corr_mean' in val_metrics and not self._is_invalid_metric(float(val_metrics['corr_mean'])):
            self.best_corr_mean = float(val_metrics['corr_mean'])
        if 'deception_balanced_accuracy' in val_metrics and not self._is_invalid_metric(float(val_metrics['deception_balanced_accuracy'])):
            self.best_balanced_accuracy = float(val_metrics['deception_balanced_accuracy'])
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Robust progress bar for Windows
        import sys
        
        # User requested live loading bar, but Windows console can be flaky
        # We'll use a wrapper that suppresses errors
        disable_tqdm = False 
        
        try:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]', 
                       file=sys.stdout, disable=disable_tqdm, 
                       mininterval=1.0, maxinterval=5.0,
                       dynamic_ncols=True)
        except Exception:
            pbar = tqdm(self.train_loader, disable=True)
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device with non_blocking for faster transfer
            non_blocking = self.device.type == 'cuda'
            video = batch['video'].to(self.device, non_blocking=non_blocking) if batch['video'] is not None else None
            audio = batch['audio'].to(self.device, non_blocking=non_blocking) if batch['audio'] is not None else None
            text = {k: v.to(self.device, non_blocking=non_blocking) for k, v in batch['text'].items()} if batch['text'] is not None else None
            
            personality_target = batch['personality'].to(self.device, non_blocking=non_blocking)
            deception_target = batch['deception'].to(self.device, non_blocking=non_blocking)
            cultural_groups = batch['cultural_groups']
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                try:
                    with torch.amp.autocast('cuda'):
                        predictions = self.model(video, audio, text)
                except ValueError as e:
                    if "At least one modality must be successfully processed" in str(e):
                        self.logger.warning(f"Skipping invalid training batch {batch_idx + 1}: {e}")
                        continue
                    raise

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
                    {'personality': personality_target, 'deception': deception_target}
                )
                loss = loss_dict['total_loss']

                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                    continue
                
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                try:
                    predictions = self.model(video, audio, text)
                except ValueError as e:
                    if "At least one modality must be successfully processed" in str(e):
                        self.logger.warning(f"Skipping invalid training batch {batch_idx + 1}: {e}")
                        continue
                    raise
                
                # Check for NaN in predictions
                if torch.any(torch.isnan(predictions['traits'])) or torch.any(torch.isnan(predictions['deception_logits'])):
                    self.logger.warning(f"NaN detected in predictions at batch {batch_idx}")
                    continue
                
                loss_dict = self.criterion(
                    predictions,
                    {'personality': personality_target, 'deception': deception_target}
                )
                loss = loss_dict['total_loss']
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                    continue
                
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            
            # Ensure targets match prediction size (in case of trimming)
            curr_batch_size = predictions['traits'].shape[0]
            if personality_target.shape[0] > curr_batch_size:
                personality_target = personality_target[:curr_batch_size]
                deception_target = deception_target[:curr_batch_size]
                if cultural_groups is not None:
                    cultural_groups = cultural_groups[:curr_batch_size]

            # Update metrics
            self.metrics_calculator.update(
                predictions['traits'],
                personality_target,
                predictions['deception_logits'],
                deception_target,
                cultural_groups
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar safely
            try:
                if not getattr(pbar, 'disable', True):
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            except Exception:
                pass
            
            # Log progress to file periodically
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
            
            # Print to logger if tqdm is disabled
            if disable_tqdm:
                if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                    msg = f"Epoch {epoch} [Train] Batch {batch_idx + 1}/{len(self.train_loader)} | Loss: {loss.item():.4f}"
                    try:
                        print(f"\r{msg}", end='', flush=True)
                    except OSError:
                        pass
                    self.logger.info(msg)
                    for handler in self.logger.handlers:
                        handler.flush()
        
        # Compute metrics
        metrics = self.metrics_calculator.compute()
        if num_batches == 0:
            raise RuntimeError("No valid training batches were processed in this epoch")
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
            disable_tqdm = False
            try:
                pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]', 
                           file=sys.stdout, disable=False, 
                           mininterval=1.0, maxinterval=5.0,
                           dynamic_ncols=True)
            except Exception:
                pbar = tqdm(self.val_loader, disable=True)
            
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device with non_blocking for faster transfer
                non_blocking = self.device.type == 'cuda'
                video = batch['video'].to(self.device, non_blocking=non_blocking) if batch['video'] is not None else None
                audio = batch['audio'].to(self.device, non_blocking=non_blocking) if batch['audio'] is not None else None
                text = {k: v.to(self.device, non_blocking=non_blocking) for k, v in batch['text'].items()} if batch['text'] is not None else None
                
                personality_target = batch['personality'].to(self.device, non_blocking=non_blocking)
                deception_target = batch['deception'].to(self.device, non_blocking=non_blocking)
                cultural_groups = batch['cultural_groups']
                
                # Forward pass
                if self.mixed_precision:
                    try:
                        with torch.amp.autocast('cuda'):
                            predictions = self.model(video, audio, text)
                    except ValueError as e:
                        if "At least one modality must be successfully processed" in str(e):
                            self.logger.warning(f"Skipping invalid validation batch {batch_idx + 1}: {e}")
                            continue
                        raise
                    loss_dict = self.criterion(
                        predictions,
                        {'personality': personality_target, 'deception': deception_target}
                    )
                    loss = loss_dict['total_loss']
                else:
                    try:
                        predictions = self.model(video, audio, text)
                    except ValueError as e:
                        if "At least one modality must be successfully processed" in str(e):
                            self.logger.warning(f"Skipping invalid validation batch {batch_idx + 1}: {e}")
                            continue
                        raise
                    loss_dict = self.criterion(
                        predictions,
                        {'personality': personality_target, 'deception': deception_target}
                    )
                    loss = loss_dict['total_loss']
                
                # Ensure targets match prediction size (in case of trimming)
                curr_batch_size = predictions['traits'].shape[0]
                if personality_target.shape[0] > curr_batch_size:
                    personality_target = personality_target[:curr_batch_size]
                    deception_target = deception_target[:curr_batch_size]
                    if cultural_groups is not None:
                        cultural_groups = cultural_groups[:curr_batch_size]
                
                # Update metrics
                self.metrics_calculator.update(
                    predictions['traits'],
                    personality_target,
                    predictions['deception_logits'],
                    deception_target,
                    cultural_groups
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar safely
                try:
                    if not getattr(pbar, 'disable', True):
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                except Exception:
                    pass
                
                # Log progress every batch (live updates for Windows)
                if disable_tqdm:
                    if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                        msg = f"Epoch {epoch} [Val] Batch {batch_idx + 1}/{len(self.val_loader)} | Loss: {loss.item():.4f}"
                        try:
                            print(f"\r{msg}", end='', flush=True)
                        except OSError:
                            pass
                        self.logger.info(msg)
                        for handler in self.logger.handlers:
                            handler.flush()
        
        # Compute metrics
        metrics = self.metrics_calculator.compute()
        if num_batches == 0:
            raise RuntimeError("No valid validation batches were processed in this epoch")
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def train(self):
        """
        Main training loop
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.num_epochs}")
        self.logger.info(
            f"Best model so far: epoch {self.best_epoch}, "
            f"val_loss {self.best_val_loss:.4f}, corr_mean {self.best_corr_mean:.4f}, "
            f"bal_acc {self.best_balanced_accuracy:.4f}"
        )
        
        try:
            start_epoch = self.best_epoch + 1 if self.best_epoch > 0 else 1
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
                    
                    # Update scheduler (handles warmup automatically)
                    if self.scheduler:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        self.scheduler.step()
                        new_lr = self.optimizer.param_groups[0]['lr']
                        phase = "warmup" if epoch < self.warmup_epochs else "cosine"
                        self.logger.info(f"Learning rate ({phase}): {current_lr:.6f} -> {new_lr:.6f}")
                    
                    # Save checkpoint based on configured monitor strategy
                    val_loss = float(val_metrics.get('loss', float('inf')))
                    is_improved, reason = self._evaluate_improvement(val_metrics)
                    if is_improved:
                        self._update_best_trackers(epoch, val_metrics)
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
                        self._prune_best_checkpoints()
                        self.logger.info(f"Saved best model at epoch {epoch} ({reason})")
                    else:
                        if self.early_stopping_enabled:
                            self.early_stopping_counter += 1
                            self.logger.info(
                                f"No improvement for {self.early_stopping_counter}/{self.early_stopping_patience} epochs "
                                f"(monitor={self.checkpoint_monitor}). {reason}"
                            )
                    
                    # Periodic checkpoint
                    if self.save_frequency > 0 and epoch % self.save_frequency == 0:
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
                        self.logger.info(f"No improvement in validation loss for {self.early_stopping_patience} epochs")
                        self.logger.info(f"Best model was at epoch {self.best_epoch} with validation loss: {self.best_val_loss:.4f}")
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
            self.logger.info(f"Training completed. Best model at epoch {self.best_epoch} (val_loss: {self.best_val_loss:.4f})")




















