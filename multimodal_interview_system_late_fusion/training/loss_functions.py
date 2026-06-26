"""
Loss functions for multimodal interview analysis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CombinedLoss(nn.Module):
    """
    Combined loss: Personality MAE + Deception CE + Cultural Bias Penalty
    Uses MAE instead of MSE for better regression performance
    """
    
    def __init__(
        self,
        personality_weight: float = 1.0,
        deception_weight: float = 2.0,  # Increased from 0.5 to prioritize deception learning
        cultural_bias_weight: float = 0.3,
        use_mae: bool = True,
        deception_class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False  # Disabled for Stage 2: too aggressive with opposing dataset imbalances
    ):
        """
        Initialize combined loss
        
        Args:
            personality_weight: Weight for personality prediction loss
            deception_weight: Weight for deception detection loss
            cultural_bias_weight: Weight for cultural bias penalty
            use_mae: Use MAE (L1) instead of MSE (L2) for personality loss
            deception_class_weights: Weights for deception classes [truthful, deceptive]
            use_focal_loss: Use focal loss for deception (handles class imbalance better)
        """
        super(CombinedLoss, self).__init__()
        
        self.personality_weight = personality_weight
        self.deception_weight = deception_weight
        self.cultural_bias_weight = cultural_bias_weight
        self.use_mae = use_mae
        self.use_focal_loss = use_focal_loss
        
        # Use MAE (L1Loss) for personality - better for regression tasks
        self.personality_loss_fn = nn.L1Loss() if use_mae else nn.MSELoss()
        
        # Use class-weighted CE or focal loss for deception (handles imbalance)
        if deception_class_weights is None:
            # Neutral fallback; trainer should pass data-driven class weights.
            deception_class_weights = torch.tensor([1.0, 1.0])  # [truthful, deceptive]
        
        self.register_buffer('deception_class_weights', deception_class_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=deception_class_weights)
        # Higher alpha (0.75) to focus more on minority class (truthful)
        # Higher gamma (3.0) to focus even more on hard examples
        self.focal_loss = FocalLoss(alpha=0.55, gamma=3.0)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        cultural_bias: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            predictions: Dictionary with 'traits' and 'deception_logits'
            targets: Dictionary with 'personality' and 'deception'
            cultural_bias: Optional cultural bias penalty
            
        Returns:
            Dictionary with individual and total losses
        """
        # Ensure targets (batch size 24) match predictions (batch size 23) if trimming occurred
        pred_batch_size = predictions['traits'].shape[0]
        target_batch_size = targets['personality'].shape[0]
        
        if pred_batch_size < target_batch_size:
            # Trim targets to match predictions
            targets['personality'] = targets['personality'][:pred_batch_size]
            targets['deception'] = targets['deception'][:pred_batch_size]
            if cultural_bias is not None:
                cultural_bias = cultural_bias[:pred_batch_size]
        
        # Personality prediction loss (MAE is better for regression)
        personality_loss = self.personality_loss_fn(
            predictions['traits'],
            targets['personality']
        )
        
        # Deception detection loss (Focal Loss or Weighted CE for class imbalance)
        if self.use_focal_loss:
            deception_loss = self.focal_loss(
                predictions['deception_logits'],
                targets['deception']
            )
        else:
            # Move class weights to correct device and dtype if needed
            if self.deception_class_weights.device != predictions['deception_logits'].device:
                self.deception_class_weights = self.deception_class_weights.to(predictions['deception_logits'].device)
            if self.deception_class_weights.dtype != predictions['deception_logits'].dtype:
                self.deception_class_weights = self.deception_class_weights.to(predictions['deception_logits'].dtype)
            
            # Use functional cross entropy to ensure weights are on correct device
            deception_loss = F.cross_entropy(
                predictions['deception_logits'],
                targets['deception'],
                weight=self.deception_class_weights
            )
        
        # Total loss
        total_loss = (
            self.personality_weight * personality_loss +
            self.deception_weight * deception_loss
        )
        
        # Add cultural bias penalty if provided
        if cultural_bias is not None:
            total_loss = total_loss + self.cultural_bias_weight * cultural_bias
        
        return {
            'total_loss': total_loss,
            'personality_loss': personality_loss,
            'deception_loss': deception_loss,
            'cultural_bias': cultural_bias if cultural_bias is not None else torch.tensor(0.0)
        }


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in deception detection
    With per-class alpha weights: higher weight for minority class (truthful=0)
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 3.0):
        """
        Initialize focal loss
        
        Args:
            alpha: Weighting factor for class 0 (truthful/minority). Class 1 gets (1-alpha).
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        # alpha for class 0 (truthful), (1-alpha) for class 1 (deceptive)
        self.alpha = alpha  # 0.75 = 3x weight for truthful vs deceptive
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss with per-class weighting
        
        Args:
            inputs: Logits [B, num_classes]
            targets: Target labels [B] (0=truthful, 1=deceptive)
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Per-class alpha: alpha for class 0 (truthful), (1-alpha) for class 1 (deceptive)
        alpha_t = torch.where(targets == 0, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class TraitMAELoss(nn.Module):
    """
    Mean Absolute Error loss for individual traits
    """
    
    def __init__(self):
        super(TraitMAELoss, self).__init__()
        self.mae_loss = nn.L1Loss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trait_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MAE loss per trait
        
        Args:
            predictions: Trait predictions [B, num_traits]
            targets: Target traits [B, num_traits]
            trait_weights: Optional per-trait weights [num_traits]
            
        Returns:
            Dictionary with per-trait and total losses
        """
        # Compute per-trait MAE
        per_trait_loss = torch.abs(predictions - targets).mean(dim=0)  # [num_traits]
        
        # Weighted total loss
        if trait_weights is not None:
            total_loss = (per_trait_loss * trait_weights).sum()
        else:
            total_loss = per_trait_loss.mean()
        
        trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        return {
            'total_loss': total_loss,
            **{name: loss for name, loss in zip(trait_names, per_trait_loss)}
        }


class CulturalBiasPenalty(nn.Module):
    """
    Penalty term to reduce cultural bias in predictions
    """
    
    def __init__(self):
        super(CulturalBiasPenalty, self).__init__()
    
    def forward(
        self,
        predictions: torch.Tensor,
        cultural_groups: torch.Tensor,
        num_groups: int
    ) -> torch.Tensor:
        """
        Compute cultural bias penalty
        
        Args:
            predictions: Trait predictions [B, num_traits]
            cultural_groups: Cultural group indices [B]
            num_groups: Number of cultural groups
            
        Returns:
            Bias penalty (higher = more bias)
        """
        # Compute mean predictions per cultural group
        group_means = []
        for group_id in range(num_groups):
            group_mask = (cultural_groups == group_id)
            if group_mask.sum() > 0:
                group_mean = predictions[group_mask].mean(dim=0)  # [num_traits]
                group_means.append(group_mean)
        
        if len(group_means) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute variance across groups (higher variance = more bias)
        group_means_tensor = torch.stack(group_means)  # [num_groups, num_traits]
        bias_penalty = group_means_tensor.var(dim=0).mean()
        
        return bias_penalty

























