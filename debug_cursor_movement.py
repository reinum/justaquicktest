#!/usr/bin/env python3
"""
Debug script to analyze and fix cursor movement issues.

This script identifies the problem with cursor movement being too restricted
and provides a solution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class OriginalCursorSampling:
    """Original cursor sampling with movement restrictions."""
    
    def __init__(self, smoothness_weight: float = 0.1, boundary_penalty: float = 0.2):
        self.smoothness_weight = smoothness_weight
        self.boundary_penalty = boundary_penalty
    
    def sample(self, cursor_logits: torch.Tensor, 
               previous_pos: Optional[torch.Tensor] = None,
               screen_bounds: Tuple[int, int] = (512, 384)) -> torch.Tensor:
        """Original sampling with severe movement restriction."""
        cursor_pos = torch.tanh(cursor_logits)
        cursor_pos[:, 0] = (cursor_pos[:, 0] + 1) * screen_bounds[0] / 2
        cursor_pos[:, 1] = (cursor_pos[:, 1] + 1) * screen_bounds[1] / 2
        
        if previous_pos is not None and self.smoothness_weight > 0:
            if previous_pos.dim() == 3:
                previous_pos = previous_pos.squeeze(1)
            
            # PROBLEM: This limits movement to only 10% of screen!
            max_movement = min(screen_bounds) * 0.1  # Only ~38 pixels!
            
            movement = cursor_pos - previous_pos
            movement_magnitude = torch.norm(movement, dim=-1, keepdim=True)
            
            scale_factor = torch.clamp(max_movement / (movement_magnitude + 1e-8), max=1.0)
            cursor_pos = previous_pos + movement * scale_factor
        
        cursor_pos[:, 0] = torch.clamp(cursor_pos[:, 0], 0, screen_bounds[0])
        cursor_pos[:, 1] = torch.clamp(cursor_pos[:, 1], 0, screen_bounds[1])
        
        return cursor_pos

class ImprovedCursorSampling:
    """Improved cursor sampling with realistic movement."""
    
    def __init__(self, smoothness_weight: float = 0.1, boundary_penalty: float = 0.2):
        self.smoothness_weight = smoothness_weight
        self.boundary_penalty = boundary_penalty
    
    def sample(self, cursor_logits: torch.Tensor, 
               previous_pos: Optional[torch.Tensor] = None,
               screen_bounds: Tuple[int, int] = (512, 384)) -> torch.Tensor:
        """Improved sampling with realistic movement limits."""
        cursor_pos = torch.tanh(cursor_logits)
        cursor_pos[:, 0] = (cursor_pos[:, 0] + 1) * screen_bounds[0] / 2
        cursor_pos[:, 1] = (cursor_pos[:, 1] + 1) * screen_bounds[1] / 2
        
        if previous_pos is not None and self.smoothness_weight > 0:
            if previous_pos.dim() == 3:
                previous_pos = previous_pos.squeeze(1)
            
            # SOLUTION: Allow much larger movement (50% of screen per frame)
            # This is still smooth but allows realistic osu! cursor movement
            max_movement = min(screen_bounds) * 0.5  # ~192 pixels - much more realistic!
            
            movement = cursor_pos - previous_pos
            movement_magnitude = torch.norm(movement, dim=-1, keepdim=True)
            
            # Only limit extremely large movements
            scale_factor = torch.clamp(max_movement / (movement_magnitude + 1e-8), max=1.0)
            cursor_pos = previous_pos + movement * scale_factor
        
        cursor_pos[:, 0] = torch.clamp(cursor_pos[:, 0], 0, screen_bounds[0])
        cursor_pos[:, 1] = torch.clamp(cursor_pos[:, 1], 0, screen_bounds[1])
        
        return cursor_pos

def demonstrate_movement_difference():
    """Demonstrate the difference between original and improved sampling."""
    print("=== Cursor Movement Analysis ===")
    print()
    
    # Screen bounds
    screen_bounds = (512, 384)
    print(f"Screen bounds: {screen_bounds}")
    print(f"Smaller dimension: {min(screen_bounds)}")
    print()
    
    # Original limits
    original_max = min(screen_bounds) * 0.1
    print(f"Original max movement per frame: {original_max:.1f} pixels")
    print(f"This means cursor can only move {original_max:.1f} pixels per frame!")
    print("For a 60fps replay, that's extremely slow movement.")
    print()
    
    # Improved limits
    improved_max = min(screen_bounds) * 0.5
    print(f"Improved max movement per frame: {improved_max:.1f} pixels")
    print(f"This allows much more realistic osu! cursor movement.")
    print()
    
    # Simulate movement
    print("=== Movement Simulation ===")
    
    # Create test logits that would move cursor across screen
    torch.manual_seed(42)
    cursor_logits = torch.randn(1, 2) * 2  # Strong movement signal
    previous_pos = torch.tensor([[256.0, 192.0]])  # Center of screen
    
    print(f"Starting position: ({previous_pos[0, 0]:.1f}, {previous_pos[0, 1]:.1f})")
    print(f"Cursor logits: {cursor_logits[0].tolist()}")
    print()
    
    # Test original sampling
    original_sampler = OriginalCursorSampling()
    original_pos = original_sampler.sample(cursor_logits, previous_pos, screen_bounds)
    original_movement = torch.norm(original_pos - previous_pos).item()
    
    print(f"Original sampling result: ({original_pos[0, 0]:.1f}, {original_pos[0, 1]:.1f})")
    print(f"Movement distance: {original_movement:.1f} pixels")
    print()
    
    # Test improved sampling
    improved_sampler = ImprovedCursorSampling()
    improved_pos = improved_sampler.sample(cursor_logits, previous_pos, screen_bounds)
    improved_movement = torch.norm(improved_pos - previous_pos).item()
    
    print(f"Improved sampling result: ({improved_pos[0, 0]:.1f}, {improved_pos[0, 1]:.1f})")
    print(f"Movement distance: {improved_movement:.1f} pixels")
    print()
    
    print(f"Movement increase: {improved_movement / original_movement:.1f}x")
    print()
    
    print("=== Conclusion ===")
    print("The original cursor sampling severely limits movement, causing the cursor")
    print("to barely move and stay in roughly the same position throughout the replay.")
    print("The improved version allows realistic osu! cursor movement while still")
    print("maintaining smoothness constraints.")

if __name__ == "__main__":
    demonstrate_movement_difference()