"""Model utilities for the osu! AI replay maker.

This module provides utility functions and classes for model operations,
including parameter counting, model initialization, and optimization helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np


class ModelUtils:
    """Utility class for model operations and helpers."""
    
    @staticmethod
    def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
        """Count the number of parameters in a model.
        
        Args:
            model: PyTorch model
            trainable_only: If True, only count trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """Get the size of a model in megabytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def initialize_weights(model: nn.Module, init_type: str = 'xavier_uniform') -> None:
        """Initialize model weights.
        
        Args:
            model: PyTorch model
            init_type: Type of initialization ('xavier_uniform', 'xavier_normal', 
                      'kaiming_uniform', 'kaiming_normal', 'normal', 'uniform')
        """
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif init_type == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif init_type == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif init_type == 'uniform':
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def get_learning_rate_schedule(optimizer_name: str, 
                                 base_lr: float,
                                 warmup_steps: int,
                                 total_steps: int,
                                 min_lr_ratio: float = 0.1) -> Dict[str, Any]:
        """Get learning rate schedule configuration.
        
        Args:
            optimizer_name: Name of the optimizer
            base_lr: Base learning rate
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum learning rate as ratio of base_lr
            
        Returns:
            Learning rate schedule configuration
        """
        if optimizer_name.lower() == 'adamw':
            return {
                'scheduler': 'cosine_with_warmup',
                'base_lr': base_lr,
                'warmup_steps': warmup_steps,
                'total_steps': total_steps,
                'min_lr': base_lr * min_lr_ratio
            }
        elif optimizer_name.lower() == 'sgd':
            return {
                'scheduler': 'step',
                'base_lr': base_lr,
                'step_size': total_steps // 3,
                'gamma': 0.1
            }
        else:
            return {
                'scheduler': 'constant',
                'base_lr': base_lr
            }
    
    @staticmethod
    def calculate_receptive_field(layers: List[Dict[str, Any]]) -> int:
        """Calculate the receptive field of a sequence of layers.
        
        Args:
            layers: List of layer configurations with 'kernel_size' and 'stride'
            
        Returns:
            Receptive field size
        """
        receptive_field = 1
        jump = 1
        
        for layer in layers:
            kernel_size = layer.get('kernel_size', 1)
            stride = layer.get('stride', 1)
            
            receptive_field += (kernel_size - 1) * jump
            jump *= stride
        
        return receptive_field
    
    @staticmethod
    def create_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        """Create padding mask for variable length sequences.
        
        Args:
            lengths: Tensor of sequence lengths [batch_size]
            max_len: Maximum sequence length (if None, use max of lengths)
            
        Returns:
            Boolean mask tensor [batch_size, max_len] where True indicates padding
        """
        if max_len is None:
            max_len = lengths.max().item()
        
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        
        return mask
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal (lower triangular) mask for autoregressive models.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Boolean mask tensor [seq_len, seq_len] where True indicates masked positions
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    @staticmethod
    def apply_gradient_clipping(model: nn.Module, max_norm: float) -> float:
        """Apply gradient clipping to model parameters.
        
        Args:
            model: PyTorch model
            max_norm: Maximum gradient norm
            
        Returns:
            Total gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def get_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, Any]:
        """Get a summary of the model architecture.
        
        Args:
            model: PyTorch model
            input_size: Input tensor size (without batch dimension)
            
        Returns:
            Dictionary containing model summary information
        """
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                
                m_key = f"{class_name}-{module_idx+1}"
                summary[m_key] = {}
                summary[m_key]["input_shape"] = list(input[0].size()) if input else []
                summary[m_key]["output_shape"] = list(output.size()) if hasattr(output, 'size') else []
                
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params
            
            if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
                hooks.append(module.register_forward_hook(hook))
        
        # Create summary dict
        summary = {}
        hooks = []
        
        # Register hooks
        model.apply(register_hook)
        
        # Create dummy input
        device = next(model.parameters()).device
        x = torch.randn(1, *input_size).to(device)
        
        # Forward pass
        model(x)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Calculate total parameters
        total_params = ModelUtils.count_parameters(model, trainable_only=False)
        trainable_params = ModelUtils.count_parameters(model, trainable_only=True)
        
        return {
            "summary": summary,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": ModelUtils.get_model_size_mb(model)
        }
    
    @staticmethod
    def save_checkpoint(model: nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       filepath: str,
                       **kwargs) -> None:
        """Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            loss: Current loss
            filepath: Path to save checkpoint
            **kwargs: Additional items to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            **kwargs
        }
        torch.save(checkpoint, filepath)
    
    @staticmethod
    def load_checkpoint(filepath: str, 
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: PyTorch model
            optimizer: Optimizer (optional)
            
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint