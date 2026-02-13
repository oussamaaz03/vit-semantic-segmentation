"""
Custom Loss Functions for Semantic Segmentation
Author: Your improvements to the Segmenter project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    Better for handling class imbalance and small objects.
    """
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - model predictions (logits)
            target: (B, H, W) - ground truth labels
        Returns:
            dice_loss: scalar tensor
        """
        # Create mask for valid pixels (exclude ignore_index)
        valid_mask = (target != self.ignore_index)
        
        # Get number of classes
        n_classes = pred.shape[1]
        
        # Convert target to one-hot encoding
        # Clamp target to valid range [0, n_classes-1]
        target_clamped = target.clamp(0, n_classes - 1)
        target_one_hot = F.one_hot(target_clamped, n_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Apply valid mask
        valid_mask_expanded = valid_mask.unsqueeze(1).float()
        pred_masked = pred * valid_mask_expanded
        target_masked = target_one_hot * valid_mask_expanded
        
        # Apply softmax to predictions
        pred_soft = F.softmax(pred_masked, dim=1)
        
        # Calculate Dice coefficient
        # intersection: sum over spatial dimensions (H, W)
        intersection = (pred_soft * target_masked).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_masked.sum(dim=(2, 3))
        
        # Dice coefficient per class, then average
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class HybridLoss(nn.Module):
    """
    Hybrid Loss: Combination of Cross Entropy and Dice Loss
    
    This combines the advantages of both losses:
    - CrossEntropy: Good for pixel-wise classification
    - Dice: Good for segmentation quality and boundaries
    
    Args:
        ce_weight: Weight for CrossEntropy loss (default: 0.6)
        dice_weight: Weight for Dice loss (default: 0.4)
        ignore_index: Label index to ignore (default: 255)
        label_smoothing: Label smoothing factor (default: 0.1)
    """
    def __init__(self, ce_weight=0.6, dice_weight=0.4, ignore_index=255, label_smoothing=0.1):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # Cross Entropy Loss with label smoothing for regularization
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        
        # Dice Loss for better segmentation boundaries
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - model predictions (logits)
            target: (B, H, W) - ground truth labels
        Returns:
            hybrid_loss: weighted combination of CE and Dice losses
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # Weighted combination
        hybrid_loss = self.ce_weight * ce + self.dice_weight * dice
        
        return hybrid_loss
