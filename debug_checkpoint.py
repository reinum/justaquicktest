#!/usr/bin/env python3
"""
Debug script to analyze the 30epochs_trained.pt checkpoint.

This script loads the trained model checkpoint and analyzes:
1. Model state and parameters
2. Training progress metrics
3. Loss values and convergence
4. Model output quality
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_checkpoint(checkpoint_path):
    """Analyze the checkpoint file to understand training progress."""
    print(f"=== Analyzing Checkpoint: {checkpoint_path} ===")
    print()
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return
    
    # Analyze checkpoint structure
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    # Check if it's a training checkpoint or just model weights
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        print("✓ Found model_state_dict")
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
        print("✓ Found state_dict")
    else:
        # Assume the whole checkpoint is the model state
        model_state = checkpoint
        print("✓ Using entire checkpoint as model state")
    
    # Analyze model parameters
    print(f"\n=== Model Parameters Analysis ===")
    total_params = 0
    layer_info = defaultdict(list)
    
    for name, param in model_state.items():
        param_count = param.numel()
        total_params += param_count
        
        # Group by layer type
        if 'embedding' in name:
            layer_info['embeddings'].append((name, param.shape, param_count))
        elif 'transformer' in name or 'attention' in name:
            layer_info['transformer'].append((name, param.shape, param_count))
        elif 'output' in name or 'head' in name:
            layer_info['output'].append((name, param.shape, param_count))
        else:
            layer_info['other'].append((name, param.shape, param_count))
    
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # Show parameter distribution
    for layer_type, params in layer_info.items():
        if params:
            total_layer_params = sum(p[2] for p in params)
            print(f"\n{layer_type.capitalize()} layers: {total_layer_params:,} params ({total_layer_params/total_params*100:.1f}%)")
            for name, shape, count in params[:3]:  # Show first 3
                print(f"  {name}: {shape} ({count:,} params)")
            if len(params) > 3:
                print(f"  ... and {len(params)-3} more layers")
    
    # Analyze parameter statistics
    print(f"\n=== Parameter Statistics ===")
    all_weights = []
    for name, param in model_state.items():
        if param.dtype == torch.float32 or param.dtype == torch.float16:
            weights = param.flatten().float()
            all_weights.append(weights)
            
            # Show stats for key layers
            if any(key in name for key in ['embedding', 'attention', 'output']):
                print(f"{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Mean: {weights.mean().item():.6f}")
                print(f"  Std: {weights.std().item():.6f}")
                print(f"  Min: {weights.min().item():.6f}")
                print(f"  Max: {weights.max().item():.6f}")
                print(f"  Zeros: {(weights == 0).sum().item()}/{weights.numel()} ({(weights == 0).float().mean().item()*100:.1f}%)")
                print()
    
    # Overall weight statistics
    if all_weights:
        all_weights = torch.cat(all_weights)
        print(f"Overall model statistics:")
        print(f"  Total weights: {all_weights.numel():,}")
        print(f"  Mean: {all_weights.mean().item():.6f}")
        print(f"  Std: {all_weights.std().item():.6f}")
        print(f"  Min: {all_weights.min().item():.6f}")
        print(f"  Max: {all_weights.max().item():.6f}")
        print(f"  Zeros: {(all_weights == 0).sum().item()}/{all_weights.numel()} ({(all_weights == 0).float().mean().item()*100:.1f}%)")
        
        # Check for potential issues
        if all_weights.std().item() < 0.001:
            print("⚠️  WARNING: Very low weight variance - model might not be training")
        if (all_weights == 0).float().mean().item() > 0.5:
            print("⚠️  WARNING: More than 50% weights are zero")
        if torch.isnan(all_weights).any():
            print("⚠️  WARNING: NaN values detected in weights")
        if torch.isinf(all_weights).any():
            print("⚠️  WARNING: Infinite values detected in weights")
    
    # Check training metadata if available
    if isinstance(checkpoint, dict):
        print(f"\n=== Training Metadata ===")
        
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            print(f"Step: {checkpoint['step']}")
        if 'best_loss' in checkpoint:
            print(f"Best loss: {checkpoint['best_loss']:.6f}")
        if 'train_loss' in checkpoint:
            print(f"Train loss: {checkpoint['train_loss']:.6f}")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.6f}")
        if 'learning_rate' in checkpoint:
            print(f"Learning rate: {checkpoint['learning_rate']:.2e}")
        
        # Check optimizer state
        if 'optimizer_state_dict' in checkpoint:
            print("✓ Optimizer state found")
            opt_state = checkpoint['optimizer_state_dict']
            if 'state' in opt_state and opt_state['state']:
                # Check if optimizer has momentum/running averages
                first_param_state = next(iter(opt_state['state'].values()))
                if 'exp_avg' in first_param_state:
                    print("✓ Adam momentum found (exp_avg)")
                if 'exp_avg_sq' in first_param_state:
                    print("✓ Adam second moment found (exp_avg_sq)")
        
        if 'scheduler_state_dict' in checkpoint:
            print("✓ Scheduler state found")
    
    print(f"\n=== Training Assessment ===")
    
    # Basic health checks
    health_score = 0
    max_score = 5
    
    # 1. Model has reasonable parameter distribution
    if all_weights is not None:
        if 0.001 < all_weights.std().item() < 1.0:
            print("✓ Parameter variance is reasonable")
            health_score += 1
        else:
            print("✗ Parameter variance is concerning")
    
    # 2. No NaN or infinite values
    if all_weights is not None and not torch.isnan(all_weights).any() and not torch.isinf(all_weights).any():
        print("✓ No NaN or infinite values")
        health_score += 1
    else:
        print("✗ Found NaN or infinite values")
    
    # 3. Not too many zero weights
    if all_weights is not None and (all_weights == 0).float().mean().item() < 0.3:
        print("✓ Reasonable number of non-zero weights")
        health_score += 1
    else:
        print("✗ Too many zero weights")
    
    # 4. Has training metadata
    if isinstance(checkpoint, dict) and any(key in checkpoint for key in ['epoch', 'step', 'train_loss']):
        print("✓ Contains training metadata")
        health_score += 1
    else:
        print("✗ Missing training metadata")
    
    # 5. Has optimizer state (indicates proper training)
    if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        print("✓ Contains optimizer state")
        health_score += 1
    else:
        print("✗ Missing optimizer state")
    
    print(f"\nModel Health Score: {health_score}/{max_score}")
    
    if health_score >= 4:
        print("🟢 Model appears to be training well")
    elif health_score >= 2:
        print("🟡 Model training has some issues but may be recoverable")
    else:
        print("🔴 Model training appears to have serious problems")
    
    return checkpoint

def compare_checkpoints():
    """Compare multiple checkpoints to see training progress."""
    print(f"\n=== Comparing Available Checkpoints ===")
    
    # Look for checkpoints
    checkpoint_files = []
    
    # Check root directory
    root_path = Path(".")
    for pt_file in root_path.glob("*.pt"):
        checkpoint_files.append(pt_file)
    
    # Check checkpoints directory
    checkpoints_dir = root_path / "checkpoints"
    if checkpoints_dir.exists():
        for pt_file in checkpoints_dir.glob("*.pt"):
            checkpoint_files.append(pt_file)
    
    if not checkpoint_files:
        print("No checkpoint files found")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for f in sorted(checkpoint_files):
        print(f"  {f}")
    
    # Load and compare key metrics
    checkpoint_data = []
    for cp_file in sorted(checkpoint_files):
        try:
            cp = torch.load(cp_file, map_location='cpu', weights_only=False)
            data = {'file': cp_file.name, 'path': cp_file}
            
            if isinstance(cp, dict):
                data['epoch'] = cp.get('epoch', 'Unknown')
                data['train_loss'] = cp.get('train_loss', 'Unknown')
                data['val_loss'] = cp.get('val_loss', 'Unknown')
                data['best_loss'] = cp.get('best_loss', 'Unknown')
            else:
                data['epoch'] = 'Unknown'
                data['train_loss'] = 'Unknown'
                data['val_loss'] = 'Unknown'
                data['best_loss'] = 'Unknown'
            
            checkpoint_data.append(data)
        except Exception as e:
            print(f"Failed to load {cp_file}: {e}")
    
    if checkpoint_data:
        print(f"\nCheckpoint comparison:")
        print(f"{'File':<25} {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Best Loss':<12}")
        print("-" * 75)
        
        for data in checkpoint_data:
            epoch = str(data['epoch']) if data['epoch'] != 'Unknown' else 'N/A'
            train_loss = f"{data['train_loss']:.6f}" if isinstance(data['train_loss'], (int, float)) else 'N/A'
            val_loss = f"{data['val_loss']:.6f}" if isinstance(data['val_loss'], (int, float)) else 'N/A'
            best_loss = f"{data['best_loss']:.6f}" if isinstance(data['best_loss'], (int, float)) else 'N/A'
            
            print(f"{data['file']:<25} {epoch:<8} {train_loss:<12} {val_loss:<12} {best_loss:<12}")

if __name__ == "__main__":
    # Analyze the 30epochs_trained.pt checkpoint
    checkpoint_path = "30epochs_trained.pt"
    
    if Path(checkpoint_path).exists():
        checkpoint = analyze_checkpoint(checkpoint_path)
        compare_checkpoints()
    else:
        print(f"Checkpoint file {checkpoint_path} not found")
        print("Available .pt files:")
        for pt_file in Path(".").glob("*.pt"):
            print(f"  {pt_file}")