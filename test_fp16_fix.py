#!/usr/bin/env python3
"""
Test script for FP16 overflow fixes in the osu! AI model.
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from files to avoid relative import issues
from src.models.embeddings import AccuracyConditioning
from src.models.attention import MultiHeadAttention

def test_accuracy_conditioning():
    """Test AccuracyConditioning with various input values."""
    print("Testing AccuracyConditioning...")
    
    # Test with different precisions
    for dtype in [torch.float32, torch.float16]:
        print(f"  Testing with {dtype}")
        
        model = AccuracyConditioning(d_model=512).to(dtype)
        
        # Test normal values
        normal_input = torch.tensor([[0.95]], dtype=dtype)
        try:
            output = model(normal_input)
            print(f"    Normal input (0.95): Output shape {output.shape}, range [{output.min():.2f}, {output.max():.2f}]")
        except Exception as e:
            print(f"    Normal input failed: {e}")
        
        # Test extreme values
        extreme_input = torch.tensor([[1.5]], dtype=dtype)  # Should be clamped to 1.0
        try:
            output = model(extreme_input)
            print(f"    Extreme input (1.5): Output shape {output.shape}, range [{output.min():.2f}, {output.max():.2f}]")
        except Exception as e:
            print(f"    Extreme input failed: {e}")
        
        # Test negative values
        negative_input = torch.tensor([[-0.5]], dtype=dtype)  # Should be clamped to 0.0
        try:
            output = model(negative_input)
            print(f"    Negative input (-0.5): Output shape {output.shape}, range [{output.min():.2f}, {output.max():.2f}]")
        except Exception as e:
            print(f"    Negative input failed: {e}")

def test_attention_mechanism():
    """Test MultiHeadAttention with FP16."""
    print("\nTesting MultiHeadAttention...")
    
    # Test with different precisions
    for dtype in [torch.float32, torch.float16]:
        print(f"  Testing with {dtype}")
        
        model = MultiHeadAttention(d_model=512, n_heads=8).to(dtype)
        
        # Create test input
        seq_len, batch_size = 64, 4  # Smaller sequence for testing
        x = torch.randn(seq_len, batch_size, 512, dtype=dtype)
        
        try:
            output, attn_weights = model(x, x, x)
            print(f"    Attention output shape: {output.shape}, range [{output.min():.2f}, {output.max():.2f}]")
            print(f"    Attention weights shape: {attn_weights.shape}, range [{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
        except Exception as e:
            print(f"    Attention failed: {e}")

def test_gradscaler():
    """Test GradScaler with updated configuration."""
    print("\nTesting GradScaler...")
    
    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GradScaler test")
        return
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.GELU(),
        nn.Linear(50, 1)
    ).cuda().to(torch.float32)  # Keep model in FP32, use autocast for FP16
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize GradScaler with conservative settings
    scaler = GradScaler('cuda',
        init_scale=2.**8,  # Lower initial scale
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=1000,
        enabled=True
    )
    
    # Test data
    x = torch.randn(32, 10, device='cuda', dtype=torch.float32)
    y = torch.randn(32, 1, device='cuda', dtype=torch.float32)
    
    try:
        # Forward pass with autocast
        with autocast('cuda'):
            output = model(x)
            loss = nn.MSELoss()(output, y)
        
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Initial scale: {scaler.get_scale()}")
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"  Final scale: {scaler.get_scale()}")
        print("  GradScaler test passed!")
        
    except Exception as e:
        print(f"  GradScaler test failed: {e}")

def main():
    print("FP16 Overflow Fix Test")
    print("=" * 50)
    
    # Test individual components
    test_accuracy_conditioning()
    test_attention_mechanism()
    test_gradscaler()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()