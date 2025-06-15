#!/usr/bin/env python3
"""
Debug script to test the exact cursor logits being generated during replay generation.

From the logs, we see logits like [1.3512, 1.0334] with range min: 0.0095, max: 1.3753, mean: 1.1884
Let's test if these values should produce movement.
"""

import torch
import numpy as np
from src.generation.sampling import CursorSampling

def test_actual_logits():
    """Test cursor sampling with the actual logits from replay generation."""
    print("=== Testing Actual Logits from Replay Generation ===")
    print()
    
    # Create cursor sampler (using current settings)
    cursor_sampler = CursorSampling()
    
    # Test with actual logits from the logs
    actual_logits = torch.tensor([[1.3512, 1.0334]])
    prev_pos = torch.tensor([[256.0, 192.0]])  # Center of screen
    
    print(f"Test logits: {actual_logits[0].tolist()}")
    print(f"Previous position: {prev_pos[0].tolist()}")
    print()
    
    # Sample multiple times to see variation
    movements = []
    for i in range(10):
        sampled_pos = cursor_sampler.sample(actual_logits, prev_pos)
        movement = torch.norm(sampled_pos - prev_pos).item()
        movements.append(movement)
        print(f"Sample {i+1}: pos=[{sampled_pos[0, 0].item():.1f}, {sampled_pos[0, 1].item():.1f}], movement={movement:.1f}px")
    
    avg_movement = np.mean(movements)
    print(f"\nAverage movement: {avg_movement:.1f} pixels")
    print(f"Movement range: {min(movements):.1f} - {max(movements):.1f} pixels")
    
    # Test with range of logits from the logs
    print("\n=== Testing Range of Logits ===")
    test_cases = [
        [0.0095, 0.0095],  # Min values
        [1.1884, 1.1884],  # Mean values
        [1.3753, 1.3753],  # Max values
        [1.3512, 1.0334],  # Actual example
        [0.5, -0.5],       # Mixed values
    ]
    
    for logits_vals in test_cases:
        test_logits = torch.tensor([logits_vals])
        sampled_pos = cursor_sampler.sample(test_logits, prev_pos)
        movement = torch.norm(sampled_pos - prev_pos).item()
        print(f"Logits {logits_vals}: movement={movement:.1f}px, pos=[{sampled_pos[0, 0].item():.1f}, {sampled_pos[0, 1].item():.1f}]")
    
    print("\n=== Analysis ===")
    print("If these logits produce reasonable movement here but not in replay generation,")
    print("the issue might be:")
    print("1. The cursor positions are being overwritten somewhere")
    print("2. The sampling is being called with wrong parameters")
    print("3. There's an issue in the sequence updating logic")
    print("4. The movement is happening but being reset each frame")

if __name__ == "__main__":
    test_actual_logits()