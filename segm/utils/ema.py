"""
Exponential Moving Average (EMA) for model weights
Author: Your improvements to the Segmenter project

EMA provides smoother model weights and better generalization without extra training cost.
"""

import torch
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a moving average of model weights during training.
    The EMA model often generalizes better than the training model.
    
    Args:
        model: PyTorch model to track
        decay: EMA decay rate (default: 0.9999)
        device: Device to store EMA model on
    
    Usage:
        # Create EMA model
        ema = ModelEMA(model, decay=0.9999)
        
        # During training loop, after optimizer.step()
        ema.update(model)
        
        # For evaluation, use EMA model
        with torch.no_grad():
            output = ema(input)
    """
    def __init__(self, model, decay=0.9999, device=None):
        # Create a copy of the model for EMA
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.device = device
        
        # Move EMA model to specified device
        if self.device is not None:
            self.ema.to(device=device)
        
        # Freeze EMA parameters (no gradient computation needed)
        for param in self.ema.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        """
        Update EMA parameters using current model parameters.
        
        Formula: ema_param = decay * ema_param + (1 - decay) * model_param
        
        Args:
            model: Current training model
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                if self.device is not None:
                    model_param = model_param.to(device=self.device)
                
                # Update EMA parameter
                ema_param.copy_(
                    self.decay * ema_param + (1.0 - self.decay) * model_param
                )
    
    def forward(self, *args, **kwargs):
        """Forward pass using EMA model"""
        return self.ema(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Allow using ema_model(input) directly"""
        return self.ema(*args, **kwargs)
    
    @property
    def module(self):
        """Access the underlying EMA model"""
        return self.ema
