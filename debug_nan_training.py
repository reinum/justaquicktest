#!/usr/bin/env python3
"""
Debugging script to monitor training for NaN/Inf issues.
Run this alongside training to get detailed monitoring.
"""

import torch
import numpy as np
import logging
from pathlib import Path
import yaml
from src.config.model_config import ModelConfig, TrainingConfig
from src.models.transformer import OsuTransformer
from src.training.loss import ReplayLoss


def check_model_weights(model: torch.nn.Module, logger: logging.Logger):
    """Check model weights for NaN/Inf values."""
    nan_count = 0
    inf_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.data is not None:
            nan_mask = torch.isnan(param.data)
            inf_mask = torch.isinf(param.data)
            
            param_nan_count = nan_mask.sum().item()
            param_inf_count = inf_mask.sum().item()
            param_total = param.data.numel()
            
            if param_nan_count > 0:
                logger.warning(f"NaN detected in {name}: {param_nan_count}/{param_total} values")
                nan_count += param_nan_count
            
            if param_inf_count > 0:
                logger.warning(f"Inf detected in {name}: {param_inf_count}/{param_total} values")
                inf_count += param_inf_count
            
            total_params += param_total
    
    logger.info(f"Weight check: {nan_count} NaN, {inf_count} Inf out of {total_params} total parameters")
    return nan_count == 0 and inf_count == 0


def check_gradients(model: torch.nn.Module, logger: logging.Logger):
    """Check gradients for NaN/Inf values."""
    nan_count = 0
    inf_count = 0
    total_grads = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            nan_mask = torch.isnan(param.grad)
            inf_mask = torch.isinf(param.grad)
            
            grad_nan_count = nan_mask.sum().item()
            grad_inf_count = inf_mask.sum().item()
            grad_total = param.grad.numel()
            
            if grad_nan_count > 0:
                logger.warning(f"NaN gradient in {name}: {grad_nan_count}/{grad_total} values")
                nan_count += grad_nan_count
            
            if grad_inf_count > 0:
                logger.warning(f"Inf gradient in {name}: {grad_inf_count}/{grad_total} values")
                inf_count += grad_inf_count
            
            total_grads += grad_total
    
    logger.info(f"Gradient check: {nan_count} NaN, {inf_count} Inf out of {total_grads} total gradients")
    return nan_count == 0 and inf_count == 0


def test_model_forward_pass(config_path: str = "config/default.yaml"):
    """Test a single forward pass to check for NaN issues."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_config = ModelConfig(**config_dict['model'])
    training_config = TrainingConfig(**config_dict['training'])
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OsuTransformer(model_config).to(device)
    
    # Create loss function
    criterion = ReplayLoss(
        cursor_weight=training_config.cursor_loss_weight,
        key_weight=training_config.key_loss_weight
    )
    
    logger.info(f"Testing model on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy batch
    batch_size = training_config.batch_size
    seq_len = model_config.max_seq_length
    
    dummy_batch = {
        'cursor_data': torch.randn(batch_size, seq_len, 2, device=device),
        'beatmap_data': torch.randn(batch_size, seq_len, 10, device=device),
        'timing_data': torch.randn(batch_size, seq_len, 3, device=device),
        'key_data': torch.randn(batch_size, seq_len, 4, device=device),
        'accuracy_target': torch.randn(batch_size, seq_len, 1, device=device)
    }
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(
                cursor_data=dummy_batch['cursor_data'],
                beatmap_data=dummy_batch['beatmap_data'],
                timing_data=dummy_batch['timing_data'],
                key_data=dummy_batch['key_data'],
                accuracy_target=dummy_batch['accuracy_target']
            )
            
            logger.info("Forward pass successful")
            
            # Check outputs
            for key, value in outputs.items():
                if torch.isnan(value).any():
                    logger.error(f"NaN detected in output {key}")
                elif torch.isinf(value).any():
                    logger.error(f"Inf detected in output {key}")
                else:
                    logger.info(f"Output {key}: shape={value.shape}, range=[{value.min():.4f}, {value.max():.4f}]")
            
            # Test loss computation
            loss_dict = criterion(outputs, dummy_batch)
            logger.info("Loss computation successful")
            
            for key, value in loss_dict.items():
                if torch.isnan(value):
                    logger.error(f"NaN detected in loss {key}: {value.item()}")
                elif torch.isinf(value):
                    logger.error(f"Inf detected in loss {key}: {value.item()}")
                else:
                    logger.info(f"Loss {key}: {value.item():.6f}")
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise
    
    # Test training step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    
    try:
        outputs = model(
            cursor_data=dummy_batch['cursor_data'],
            beatmap_data=dummy_batch['beatmap_data'],
            timing_data=dummy_batch['timing_data'],
            key_data=dummy_batch['key_data'],
            accuracy_target=dummy_batch['accuracy_target']
        )
        
        loss_dict = criterion(outputs, dummy_batch)
        total_loss = loss_dict['total_loss']
        
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients
        check_gradients(model, logger)
        
        optimizer.step()
        
        # Check weights after update
        check_model_weights(model, logger)
        
        logger.info("Training step successful")
        
    except Exception as e:
        logger.error(f"Training step failed: {e}")
        raise


if __name__ == "__main__":
    test_model_forward_pass()