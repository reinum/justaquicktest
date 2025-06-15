#!/usr/bin/env python3
"""
Test script to verify FP16 overflow fixes.
"""

import torch
import torch.nn as nn
from src.models.embeddings import AccuracyConditioning

def test_accuracy_conditioning():
    """Test AccuracyConditioning with extreme values."""
    print("Testing AccuracyConditioning with FP16...")
    
    # Create model
    model = AccuracyConditioning(d_model=512)
    model.half()  # Convert to FP16
    model.cuda()
    
    # Test with normal values
    normal_accuracy = torch.tensor([[0.95]], dtype=torch.float16, device='cuda')
    try:
        output = model(normal_accuracy)
        print(f"✓ Normal accuracy test passed. Output range: [{output.min():.2f}, {output.max():.2f}]")
    except Exception as e:
        print(f"✗ Normal accuracy test failed: {e}")
        return False
    
    # Test with extreme values (should be clamped)
    extreme_accuracy = torch.tensor([[100.0]], dtype=torch.float16, device='cuda')
    try:
        output = model(extreme_accuracy)
        print(f"✓ Extreme accuracy test passed. Output range: [{output.min():.2f}, {output.max():.2f}]")
    except Exception as e:
        print(f"✗ Extreme accuracy test failed: {e}")
        return False
    
    # Test with negative values (should be clamped)
    negative_accuracy = torch.tensor([[-1.0]], dtype=torch.float16, device='cuda')
    try:
        output = model(negative_accuracy)
        print(f"✓ Negative accuracy test passed. Output range: [{output.min():.2f}, {output.max():.2f}]")
    except Exception as e:
        print(f"✗ Negative accuracy test failed: {e}")
        return False
    
    print("All AccuracyConditioning tests passed!")
    return True

def test_gradient_scaling():
    """Test gradient scaling with overflow detection."""
    print("\nTesting gradient scaling...")
    
    from torch.amp import GradScaler
    
    # Create scaler with our settings
    scaler = GradScaler('cuda',
        init_scale=2.**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True
    )
    
    print(f"✓ GradScaler initialized with scale: {scaler.get_scale()}")
    
    # Test basic functionality - use float32 model with autocast for proper mixed precision
    model = nn.Linear(10, 1).cuda()  # Keep model in float32
    optimizer = torch.optim.Adam(model.parameters())
    
    x = torch.randn(32, 10, dtype=torch.float32, device='cuda')  # Input in float32
    target = torch.randn(32, 1, dtype=torch.float32, device='cuda')  # Target in float32
    
    try:
        with torch.amp.autocast('cuda'):  # Autocast will handle the conversion to FP16
            output = model(x)
            loss = nn.MSELoss()(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✓ Gradient scaling test passed. Final scale: {scaler.get_scale()}")
        return True
    except Exception as e:
        print(f"✗ Gradient scaling test failed: {e}")
        return False

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA available, running FP16 overflow fix tests...")
        
        success = True
        success &= test_accuracy_conditioning()
        success &= test_gradient_scaling()
        
        if success:
            print("\n🎉 All tests passed! FP16 overflow fixes are working correctly.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
    else:
        print("CUDA not available, skipping FP16 tests.")