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
        deception_weight: float = 2.0,
        personality_ccc_weight: float = 0.0,
        cultural_bias_weight: float = 0.3,
        use_mae: bool = True,
        deception_class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False
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
            # Correct weights: neutral [1.0, 1.0] to let data speak
            # Previous [1.0, 1.3] caused model to predict ALL deceptive (TN=0)
            deception_class_weights = torch.tensor([1.0, 1.0])  # [truthful, deceptive]
        
        self.register_buffer('deception_class_weights', deception_class_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=deception_class_weights)
        # Focal Loss: alpha=0.5 (neutral class balance), gamma=2.0 (focus on hard examples)
        # Prevents mode collapse by down-weighting easy/correctly-classified examples
        self.focal_loss = FocalLoss(alpha=0.5, gamma=2.0)
    
        self.personality_ccc_weight = personality_ccc_weight
        self.ccc_loss_fn = CCCLoss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        cultural_bias: Optional[torch.Tensor] = None,
        personality_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            predictions: Dictionary with 'traits' and 'deception_logits'
            targets: Dictionary with 'personality' and 'deception'
            cultural_bias: Optional cultural bias penalty
            personality_mask: Optional boolean mask [B] - True for samples that should
                            contribute to personality loss (e.g., MDPE samples with real 
                            OCEAN labels). False for samples to exclude (e.g., SEUMLD with
                            dummy personality labels). If None, all samples contribute.
            
        Returns:
            Dictionary with individual and total losses
        """
        # Ensure targets match predictions size
        pred_batch_size = predictions['traits'].shape[0]
        target_batch_size = targets['personality'].shape[0]
        
        if pred_batch_size < target_batch_size:
            # Trim targets to match predictions
            targets['personality'] = targets['personality'][:pred_batch_size]
            targets['deception'] = targets['deception'][:pred_batch_size]
            if cultural_bias is not None:
                cultural_bias = cultural_bias[:pred_batch_size]
            if personality_mask is not None:
                personality_mask = personality_mask[:pred_batch_size]
        
        # Personality prediction loss (MAE + CCC)
        # If personality_mask is provided, only compute for masked (True) samples
        if personality_mask is not None and personality_mask.any() and not personality_mask.all():
            # Mixed batch: some samples have real personality labels, some don't
            masked_traits_pred = predictions['traits'][personality_mask]
            masked_traits_target = targets['personality'][personality_mask]
            
            mae_loss = self.personality_loss_fn(masked_traits_pred, masked_traits_target)
            ccc_loss = self.ccc_loss_fn(masked_traits_pred, masked_traits_target)
            
        elif personality_mask is not None and not personality_mask.any():
            # No samples have personality labels (pure SEUMLD batch)
            mae_loss = torch.tensor(0.0, device=predictions['traits'].device)
            ccc_loss = torch.tensor(0.0, device=predictions['traits'].device)
        else:
            # All samples have personality labels (normal case, or no mask)
            mae_loss = self.personality_loss_fn(predictions['traits'], targets['personality'])
            ccc_loss = self.ccc_loss_fn(predictions['traits'], targets['personality'])
        
        personality_loss = mae_loss  # Base metric
        
        # Deception detection loss (Focal Loss or Weighted CE for class imbalance)
        if self.use_focal_loss:
            deception_loss = self.focal_loss(predictions['deception_logits'], targets['deception'])
        else:
            if self.deception_class_weights.device != predictions['deception_logits'].device:
                self.deception_class_weights = self.deception_class_weights.to(predictions['deception_logits'].device)
            deception_loss = F.cross_entropy(
                predictions['deception_logits'],
                targets['deception'],
                weight=self.deception_class_weights
            )
        
        # Total loss
        total_loss = (
            self.personality_weight * personality_loss +
            self.deception_weight * deception_loss +
            self.personality_ccc_weight * ccc_loss
        )
        
        # Add cultural bias penalty if provided
        if cultural_bias is not None:
            total_loss = total_loss + self.cultural_bias_weight * cultural_bias
        
        return {
            'total_loss': total_loss,
            'personality_loss': personality_loss,
            'ccc_loss': ccc_loss,
            'deception_loss': deception_loss,
            'cultural_bias': cultural_bias if cultural_bias is not None else torch.tensor(0.0)
        }


class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient Loss (1 - CCC)
    Optimizes for high correlation between predictions and targets.
    """
    def __init__(self, eps: float = 1e-8):
        super(CCCLoss, self).__init__()
        self.eps = eps
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute CCC loss (mean across traits)
        """
        # x: [B, 5], y: [B, 5]
        vx = x - torch.mean(x, dim=0)
        vy = y - torch.mean(y, dim=0)
        
        rho = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0)) + self.eps)
        
        mx = torch.mean(x, dim=0)
        my = torch.mean(y, dim=0)
        sx2 = torch.var(x, dim=0)
        sy2 = torch.var(y, dim=0)
        
        ccc = (2 * rho * torch.sqrt(sx2) * torch.sqrt(sy2)) / (sx2 + sy2 + (mx - my) ** 2 + self.eps)
        
        # Loss is 1 - CCC (maximize CCC)
        return 1.0 - torch.mean(ccc)


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

























