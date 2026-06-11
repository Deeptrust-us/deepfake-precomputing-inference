"""Focal Loss implementation for handling class imbalance."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the predicted probability for the true class
    - alpha_t is the weighting factor for class t
    - gamma is the focusing parameter (gamma=0 is equivalent to CE loss)
    """
    
    def __init__(self, alpha: float = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for classes. If None, uses inverse frequency.
                   Can be a float (applied to positive class) or tuple (alpha_real, alpha_fake)
            gamma: Focusing parameter. Higher gamma down-weights easy examples more.
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted probabilities (B,) in [0, 1]
            targets: True labels (B,) in {0, 1}
            
        Returns:
            Focal loss value
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t (probability of true class)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Single alpha value: apply to positive class
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # Tuple: (alpha_real, alpha_fake)
                alpha_t = self.alpha[0] * (1 - targets) + self.alpha[1] * targets
            focal_weight = alpha_t * focal_weight
        
        # Compute focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



