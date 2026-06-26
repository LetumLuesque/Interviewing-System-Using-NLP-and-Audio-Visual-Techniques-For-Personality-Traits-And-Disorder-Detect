"""
Logging utilities for training and evaluation
"""
import logging
import os
from datetime import datetime
from pathlib import Path
import torch


def setup_logger(name, log_dir="logs", level=logging.INFO):
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    log_file = os.path.join(
        log_dir,
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_model_summary(model, input_shapes, logger):
    """
    Log model architecture summary
    
    Args:
        model: PyTorch model
        input_shapes: Dictionary of input shapes
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("Model Architecture Summary")
    logger.info("=" * 80)
    logger.info(f"Model: {model.__class__.__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    logger.info("\nInput shapes:")
    for key, shape in input_shapes.items():
        logger.info(f"  {key}: {shape}")
    
    logger.info("=" * 80)


def log_training_metrics(epoch, metrics, logger):
    """
    Log training metrics
    
    Args:
        epoch: Current epoch
        metrics: Dictionary of metrics
        logger: Logger instance
    """
    logger.info(f"\nEpoch {epoch} Metrics:")
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        logger.info(f"  {key}: {value:.4f}")


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """
    Save training checkpoint
    
    Args:
        model: Model state dict
        optimizer: Optimizer state dict
        scheduler: Scheduler state dict
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load training checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        
    Returns:
        Epoch number and loss
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    strict_load_success = True
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Warning: Strict loading failed, falling back to shape-matched loading. Reason: {e}")
        strict_load_success = False
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state, strict=False)
    
    if optimizer and 'optimizer_state_dict' in checkpoint and strict_load_success:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print(f"Warning: Failed to load optimizer state (likely due to parameter mismatch): {e}")
            print("Starting with fresh optimizer state.")
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    return epoch, loss
































