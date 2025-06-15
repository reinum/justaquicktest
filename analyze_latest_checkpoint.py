#!/usr/bin/env python3
"""
Analyze the latest working checkpoint to assess training progress.
"""

import torch
import numpy as np
from pathlib import Path

def analyze_working_checkpoint():
    """Analyze the latest working checkpoint."""
    
    # Try to find the best working checkpoint
    checkpoint_paths = [
        "model_epoch_0033.pt",
        "checkpoints/latest_model.pt",
        "checkpoints/best_model.pt", 
        "checkpoints/model_epoch_0010.pt",
        "final_model.pt"
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            break
    
    if not checkpoint_path:
        print("No working checkpoint found")
        return
    
    print(f"=== Analyzing: {checkpoint_path} ===")
    print()
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return
    
    # Determine checkpoint structure
    print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Direct model state'}")
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
    else:
        model_state = checkpoint
    
    # Analyze model parameters
    print(f"\n=== Model Analysis ===")
    
    total_params = 0
    param_stats = []
    
    for name, param in model_state.items():
        if isinstance(param, torch.Tensor):
            param_count = param.numel()
            total_params += param_count
            
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                weights = param.flatten().float()
                param_stats.append({
                    'name': name,
                    'shape': param.shape,
                    'count': param_count,
                    'mean': weights.mean().item(),
                    'std': weights.std().item(),
                    'min': weights.min().item(),
                    'max': weights.max().item(),
                    'zeros': (weights == 0).sum().item(),
                    'nans': torch.isnan(weights).sum().item(),
                    'infs': torch.isinf(weights).sum().item()
                })
    
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Show key layer statistics
    print(f"\n=== Key Layer Statistics ===")
    
    key_layers = []
    for stat in param_stats:
        if any(keyword in stat['name'].lower() for keyword in ['embedding', 'attention', 'output', 'head']):
            key_layers.append(stat)
    
    for stat in key_layers[:10]:  # Show first 10 key layers
        print(f"{stat['name']}:")
        print(f"  Shape: {stat['shape']}")
        print(f"  Mean: {stat['mean']:.6f}, Std: {stat['std']:.6f}")
        print(f"  Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
        print(f"  Zeros: {stat['zeros']}/{stat['count']} ({stat['zeros']/stat['count']*100:.1f}%)")
        if stat['nans'] > 0:
            print(f"  ⚠️  NaNs: {stat['nans']}")
        if stat['infs'] > 0:
            print(f"  ⚠️  Infs: {stat['infs']}")
        print()
    
    # Overall statistics
    if param_stats:
        all_means = [s['mean'] for s in param_stats]
        all_stds = [s['std'] for s in param_stats]
        total_zeros = sum(s['zeros'] for s in param_stats)
        total_nans = sum(s['nans'] for s in param_stats)
        total_infs = sum(s['infs'] for s in param_stats)
        
        print(f"=== Overall Statistics ===")
        print(f"Mean of means: {np.mean(all_means):.6f}")
        print(f"Mean of stds: {np.mean(all_stds):.6f}")
        print(f"Total zeros: {total_zeros:,} ({total_zeros/total_params*100:.1f}%)")
        print(f"Total NaNs: {total_nans:,}")
        print(f"Total Infs: {total_infs:,}")
    
    # Training metadata
    if isinstance(checkpoint, dict):
        print(f"\n=== Training Metadata ===")
        
        metadata_keys = ['epoch', 'step', 'train_loss', 'val_loss', 'best_loss', 'learning_rate']
        for key in metadata_keys:
            if key in checkpoint:
                value = checkpoint[key]
                if isinstance(value, (int, float)):
                    if 'loss' in key:
                        print(f"{key.replace('_', ' ').title()}: {value:.6f}")
                    elif 'rate' in key:
                        print(f"{key.replace('_', ' ').title()}: {value:.2e}")
                    else:
                        print(f"{key.replace('_', ' ').title()}: {value}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Check for optimizer and scheduler
        if 'optimizer_state_dict' in checkpoint:
            print("✓ Has optimizer state")
        if 'scheduler_state_dict' in checkpoint:
            print("✓ Has scheduler state")
    
    # Health assessment
    print(f"\n=== Training Health Assessment ===")
    
    issues = []
    good_signs = []
    
    # Check parameter health
    if param_stats:
        avg_std = np.mean([s['std'] for s in param_stats])
        if avg_std < 0.001:
            issues.append("Very low parameter variance - model may not be learning")
        elif avg_std > 0.01 and avg_std < 1.0:
            good_signs.append("Parameter variance in reasonable range")
        
        zero_ratio = total_zeros / total_params
        if zero_ratio > 0.5:
            issues.append(f"High zero ratio ({zero_ratio*100:.1f}%) - possible dead neurons")
        elif zero_ratio < 0.3:
            good_signs.append("Reasonable number of active parameters")
        
        if total_nans > 0:
            issues.append(f"Found {total_nans} NaN values - training instability")
        else:
            good_signs.append("No NaN values detected")
        
        if total_infs > 0:
            issues.append(f"Found {total_infs} infinite values - gradient explosion")
        else:
            good_signs.append("No infinite values detected")
    
    # Check training progress
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint and checkpoint['epoch'] > 0:
            good_signs.append(f"Training progressed to epoch {checkpoint['epoch']}")
        
        if 'train_loss' in checkpoint and isinstance(checkpoint['train_loss'], (int, float)):
            if checkpoint['train_loss'] < 10.0:  # Reasonable loss range
                good_signs.append(f"Training loss is reasonable ({checkpoint['train_loss']:.4f})")
            else:
                issues.append(f"Training loss is high ({checkpoint['train_loss']:.4f})")
        
        if 'optimizer_state_dict' in checkpoint:
            good_signs.append("Optimizer state preserved")
        else:
            issues.append("Missing optimizer state - may affect training resumption")
        
        # Additional checks for training history
        if 'training_history' in checkpoint and checkpoint['training_history']:
            good_signs.append("Training history available")
            history = checkpoint['training_history']
            if isinstance(history, list) and len(history) > 5:
                recent_losses = [h.get('train_loss', float('inf')) for h in history[-5:] if isinstance(h, dict)]
                if recent_losses and all(isinstance(loss, (int, float)) and not np.isnan(loss) and not np.isinf(loss) for loss in recent_losses):
                    loss_trend = recent_losses[-1] - recent_losses[0]
                    if loss_trend < 0:
                        good_signs.append("Loss is decreasing over recent epochs")
        
        if 'best_val_loss' in checkpoint and isinstance(checkpoint['best_val_loss'], (int, float)):
            if not np.isnan(checkpoint['best_val_loss']) and not np.isinf(checkpoint['best_val_loss']):
                good_signs.append(f"Valid best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    print("\n🟢 Good signs:")
    for sign in good_signs:
        print(f"  ✓ {sign}")
    
    if issues:
        print("\n🔴 Issues found:")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("\n🟢 No major issues detected!")
    
    # Overall assessment
    health_score = len(good_signs) / (len(good_signs) + len(issues)) if (good_signs or issues) else 0.5
    
    print(f"\n=== Overall Assessment ===")
    print(f"Health Score: {health_score:.2f}")
    
    if health_score >= 0.8:
        print("🟢 Model appears to be training well")
        print("Recommendation: Continue training or use for generation")
    elif health_score >= 0.6:
        print("🟡 Model training shows mixed results")
        print("Recommendation: Monitor closely, consider adjusting hyperparameters")
    elif health_score >= 0.4:
        print("🟠 Model training has concerning issues")
        print("Recommendation: Review training setup, check data quality")
    else:
        print("🔴 Model training appears problematic")
        print("Recommendation: Restart training with different configuration")
    
    # Detailed training history analysis
    if isinstance(checkpoint, dict) and 'training_history' in checkpoint and checkpoint['training_history']:
        print(f"\n=== Training History Analysis ===")
        history = checkpoint['training_history']
        if isinstance(history, list):
            print(f"Total epochs recorded: {len(history)}")
            
            if len(history) > 0:
                first_epoch = history[0] if isinstance(history[0], dict) else {}
                last_epoch = history[-1] if isinstance(history[-1], dict) else {}
                first_loss = first_epoch.get('train_loss', 'N/A')
                last_loss = last_epoch.get('train_loss', 'N/A')
                print(f"First epoch loss: {first_loss}")
                print(f"Latest epoch loss: {last_loss}")
                
                if isinstance(first_loss, (int, float)) and isinstance(last_loss, (int, float)):
                    improvement = first_loss - last_loss
                    improvement_pct = (improvement / first_loss) * 100 if first_loss != 0 else 0
                    print(f"Total improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
                
                # Show recent training metrics
                print(f"\nRecent training metrics (last 5 epochs):")
                recent_history = history[-5:] if len(history) >= 5 else history
                for i, epoch_data in enumerate(recent_history):
                    if isinstance(epoch_data, dict):
                        epoch_num = len(history) - len(recent_history) + i
                        train_loss = epoch_data.get('train_loss', 'N/A')
                        val_loss = epoch_data.get('val_loss', 'N/A')
                        print(f"  Epoch {epoch_num}: Train={train_loss}, Val={val_loss}")
        else:
            print(f"Training history format: {type(history)}")
            print(f"Training history content: {history}")

if __name__ == "__main__":
    analyze_working_checkpoint()