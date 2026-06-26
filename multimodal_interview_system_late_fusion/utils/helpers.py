"""
Helper utilities
"""
import torch
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get available device (CUDA or CPU)
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(model):
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_trait_scores(scores, min_val=0, max_val=100):
    """
    Normalize Big Five trait scores to [0, 100] range
    
    Args:
        scores: Trait scores tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Normalized scores
    """
    # Assuming scores are in [-1, 1] or similar range
    scores = (scores + 1) / 2  # Normalize to [0, 1]
    scores = scores * (max_val - min_val) + min_val
    return scores


def denormalize_trait_scores(scores, min_val=0, max_val=100):
    """
    Denormalize Big Five trait scores from [0, 100] to [-1, 1]
    
    Args:
        scores: Normalized trait scores
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Denormalized scores
    """
    scores = (scores - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    scores = scores * 2 - 1  # Normalize to [-1, 1]
    return scores


def create_dir_if_not_exists(path):
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)
































