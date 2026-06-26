"""
Evaluation metrics for multimodal interview analysis
"""
import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from scipy.stats import pearsonr


class MetricsCalculator:
    """
    Calculate evaluation metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.personality_predictions = []
        self.personality_targets = []
        self.deception_predictions = []
        self.deception_targets = []
        self.cultural_groups = []
    
    def update(
        self,
        personality_pred: torch.Tensor,
        personality_target: torch.Tensor,
        deception_pred: torch.Tensor,
        deception_target: torch.Tensor,
        cultural_groups: Optional[List] = None
    ):
        """
        Update metrics with batch predictions
        
        Args:
            personality_pred: Personality predictions [B, 5]
            personality_target: Personality targets [B, 5]
            deception_pred: Deception predictions [B] or [B, 2]
            deception_target: Deception targets [B]
            cultural_groups: Cultural group labels
        """
        # Convert to numpy
        personality_pred_np = personality_pred.detach().cpu().numpy()
        personality_target_np = personality_target.detach().cpu().numpy()
        
        # Handle deception predictions (logits or probabilities)
        if deception_pred.dim() > 1:
            deception_pred_np = deception_pred.argmax(dim=1).detach().cpu().numpy()
        else:
            deception_pred_np = (deception_pred > 0.5).long().detach().cpu().numpy()
        
        deception_target_np = deception_target.detach().cpu().numpy()
        
        # Accumulate
        self.personality_predictions.append(personality_pred_np)
        self.personality_targets.append(personality_target_np)
        self.deception_predictions.append(deception_pred_np)
        self.deception_targets.append(deception_target_np)
        
        if cultural_groups is not None:
            self.cultural_groups.extend(cultural_groups)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metrics
        """
        # Handle empty predictions (all batches skipped due to NaN)
        if len(self.personality_predictions) == 0:
            # Return default metrics if no valid predictions
            trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            metrics = {}
            for name in trait_names:
                metrics[f'mae_{name}'] = 0.0
            metrics['mae_mean'] = 0.0
            metrics['mae_std'] = 0.0
            metrics['deception_accuracy'] = 0.0
            metrics['deception_precision'] = 0.0
            metrics['deception_recall'] = 0.0
            metrics['deception_f1'] = 0.0
            metrics['deception_tn'] = 0.0
            metrics['deception_fp'] = 0.0
            metrics['deception_fn'] = 0.0
            metrics['deception_tp'] = 0.0
            metrics['loss'] = float('inf')  # Indicate no valid batches
            return metrics
        
        # Concatenate all predictions
        personality_pred = np.concatenate(self.personality_predictions, axis=0)
        personality_target = np.concatenate(self.personality_targets, axis=0)
        deception_pred = np.concatenate(self.deception_predictions, axis=0)
        deception_target = np.concatenate(self.deception_targets, axis=0)
        
        metrics = {}
        
        # Personality prediction metrics (MAE per trait)
        trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        # Check for NaN values - replace with mean to avoid breaking metrics
        has_nan_pred = np.any(np.isnan(personality_pred))
        has_nan_target = np.any(np.isnan(personality_target))
        
        if has_nan_pred or has_nan_target:
            # Replace NaN with 0 (or mean) to compute metrics
            if has_nan_pred:
                personality_pred = np.nan_to_num(personality_pred, nan=0.0)
            if has_nan_target:
                personality_target = np.nan_to_num(personality_target, nan=0.0)
        
        mae_per_trait = np.abs(personality_pred - personality_target).mean(axis=0)
        
        for i, name in enumerate(trait_names):
            val = float(mae_per_trait[i]) if not np.isnan(mae_per_trait[i]) else 0.0
            metrics[f'mae_{name}'] = val
        
        metrics['mae_mean'] = float(mae_per_trait.mean()) if not np.isnan(mae_per_trait.mean()) else 0.0
        metrics['mae_std'] = float(mae_per_trait.std()) if not np.isnan(mae_per_trait.std()) else 0.0
        
        # Correlation coefficients (Pearson's r) per trait
        for i, name in enumerate(trait_names):
            pred_trait = personality_pred[:, i]
            target_trait = personality_target[:, i]
            # Remove NaN values for correlation calculation
            valid_mask = ~(np.isnan(pred_trait) | np.isnan(target_trait))
            if valid_mask.sum() > 1:
                try:
                    corr, _ = pearsonr(pred_trait[valid_mask], target_trait[valid_mask])
                    metrics[f'corr_{name}'] = float(corr) if not np.isnan(corr) else 0.0
                except:
                    metrics[f'corr_{name}'] = 0.0
            else:
                metrics[f'corr_{name}'] = 0.0
        
        # Mean correlation across all traits
        corr_values = [metrics.get(f'corr_{name}', 0.0) for name in trait_names]
        metrics['corr_mean'] = float(np.mean(corr_values)) if len(corr_values) > 0 else 0.0
        
        # Deception detection metrics
        metrics['deception_accuracy'] = accuracy_score(deception_target, deception_pred)
        try:
            metrics['deception_balanced_accuracy'] = balanced_accuracy_score(deception_target, deception_pred)
        except ValueError:
            metrics['deception_balanced_accuracy'] = 0.0
        metrics['deception_precision'] = precision_score(deception_target, deception_pred, zero_division=0)
        metrics['deception_recall'] = recall_score(deception_target, deception_pred, zero_division=0)
        metrics['deception_f1'] = f1_score(deception_target, deception_pred, zero_division=0)
        
        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(deception_target, deception_pred).ravel()
            metrics['deception_tn'] = float(tn)
            metrics['deception_fp'] = float(fp)
            metrics['deception_fn'] = float(fn)
            metrics['deception_tp'] = float(tp)
        except ValueError:
            # Handle case where one class is missing (e.g., all 0 or all 1)
            # This happens if batch has only one class
            metrics['deception_tn'] = 0.0
            metrics['deception_fp'] = 0.0
            metrics['deception_fn'] = 0.0
            metrics['deception_tp'] = 0.0
        
        # Cultural fairness metrics
        if len(self.cultural_groups) > 0:
            metrics.update(self._compute_cultural_fairness(
                personality_pred, personality_target, self.cultural_groups
            ))
            
        # Joint Score for Stage 3 (Joint Balance)
        # Combines Personality Correlation and Deception F1 Score
        metrics['joint_score'] = (0.5 * metrics.get('corr_mean', 0.0)) + (0.5 * metrics.get('deception_f1', 0.0))
        
        return metrics
    
    def _compute_cultural_fairness(
        self,
        personality_pred: np.ndarray,
        personality_target: np.ndarray,
        cultural_groups: List
    ) -> Dict[str, float]:
        """
        Compute cultural fairness metrics
        
        Args:
            personality_pred: Personality predictions
            personality_target: Personality targets
            cultural_groups: Cultural group labels
            
        Returns:
            Dictionary of fairness metrics
        """
        metrics = {}
        
        # Group predictions by cultural group
        unique_groups = list(set(cultural_groups))
        
        if len(unique_groups) < 2:
            return metrics
        
        # Compute MAE per group
        mae_per_group = {}
        for group in unique_groups:
            group_mask = [i for i, g in enumerate(cultural_groups) if g == group]
            if len(group_mask) > 0:
                group_mae = np.abs(
                    personality_pred[group_mask] - personality_target[group_mask]
                ).mean()
                mae_per_group[group] = group_mae
        
        # Statistical parity difference (difference in mean predictions)
        mean_pred_per_group = {}
        for group in unique_groups:
            group_mask = [i for i, g in enumerate(cultural_groups) if g == group]
            if len(group_mask) > 0:
                mean_pred_per_group[group] = personality_pred[group_mask].mean()
        
        if len(mean_pred_per_group) >= 2:
            mean_preds = list(mean_pred_per_group.values())
            metrics['statistical_parity_difference'] = float(np.max(mean_preds) - np.min(mean_preds))
            metrics['mae_variance_across_groups'] = float(np.var(list(mae_per_group.values())))
        
        return metrics


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error
    
    Args:
        predictions: Predictions [B, ...]
        targets: Targets [B, ...]
        
    Returns:
        MAE value
    """
    return torch.abs(predictions - targets).mean()


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error
    
    Args:
        predictions: Predictions [B, ...]
        targets: Targets [B, ...]
        
    Returns:
        RMSE value
    """
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse)


def compute_correlation(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient
    
    Args:
        predictions: Predictions [B, ...]
        targets: Targets [B, ...]
        
    Returns:
        Correlation coefficient
    """
    # Flatten if needed
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Compute correlation
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()
    
    numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
    pred_var = ((pred_flat - pred_mean) ** 2).sum()
    target_var = ((target_flat - target_mean) ** 2).sum()
    
    denominator = torch.sqrt(pred_var * target_var)
    
    if denominator == 0:
        return torch.tensor(0.0)
    
    correlation = numerator / denominator
    return correlation
























