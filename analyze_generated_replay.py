#!/usr/bin/env python3
"""
Analyze the generated replay data to check if cursor movement is working.
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_replay():
    """Analyze the generated replay data."""
    try:
        # Load the generated replay data
        data = np.load('generated_replay.npz', allow_pickle=True)
        
        print("Available keys in replay data:")
        for key in data.keys():
            print(f"  {key}: {data[key].shape}")
        
        # Get cursor positions
        cursor_positions = data['cursor_positions']
        print(f"\nCursor positions shape: {cursor_positions.shape}")
        print(f"Total cursor positions: {len(cursor_positions)}")
        
        # Analyze cursor movement
        if len(cursor_positions) > 1:
            # Calculate movement between consecutive positions
            movements = np.diff(cursor_positions, axis=0)
            distances = np.sqrt(np.sum(movements**2, axis=1))
            
            print(f"\nCursor Movement Analysis:")
            print(f"  First 10 positions: {cursor_positions[:10]}")
            print(f"  Last 10 positions: {cursor_positions[-10:]}")
            print(f"  Movement distances - min: {distances.min():.4f}, max: {distances.max():.4f}, mean: {distances.mean():.4f}")
            print(f"  Total distance traveled: {distances.sum():.2f} pixels")
            print(f"  Number of stationary frames (distance < 0.1): {np.sum(distances < 0.1)}")
            print(f"  Number of moving frames (distance >= 0.1): {np.sum(distances >= 0.1)}")
            
            # Check if cursor is completely stationary
            if distances.max() < 0.001:
                print("\n❌ ISSUE: Cursor appears to be completely stationary!")
            elif distances.mean() < 1.0:
                print("\n⚠️  WARNING: Cursor movement is very minimal (mean < 1 pixel per frame)")
            else:
                print("\n✅ SUCCESS: Cursor is moving with reasonable distances")
            
            # Plot cursor trajectory
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Full trajectory
            plt.subplot(2, 2, 1)
            plt.plot(cursor_positions[:, 0], cursor_positions[:, 1], 'b-', alpha=0.7, linewidth=0.5)
            plt.scatter(cursor_positions[0, 0], cursor_positions[0, 1], color='green', s=50, label='Start', zorder=5)
            plt.scatter(cursor_positions[-1, 0], cursor_positions[-1, 1], color='red', s=50, label='End', zorder=5)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Full Cursor Trajectory')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: X position over time
            plt.subplot(2, 2, 2)
            plt.plot(cursor_positions[:, 0], 'b-', linewidth=1)
            plt.xlabel('Frame')
            plt.ylabel('X Position')
            plt.title('X Position Over Time')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Y position over time
            plt.subplot(2, 2, 3)
            plt.plot(cursor_positions[:, 1], 'r-', linewidth=1)
            plt.xlabel('Frame')
            plt.ylabel('Y Position')
            plt.title('Y Position Over Time')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Movement distances
            plt.subplot(2, 2, 4)
            plt.plot(distances, 'g-', linewidth=1)
            plt.xlabel('Frame')
            plt.ylabel('Movement Distance (pixels)')
            plt.title('Frame-to-Frame Movement Distance')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('debug_output/cursor_analysis.png', dpi=150, bbox_inches='tight')
            print(f"\nCursor analysis plot saved to debug_output/cursor_analysis.png")
            
        else:
            print("\n❌ ERROR: Not enough cursor positions to analyze movement")
            
    except FileNotFoundError:
        print("❌ ERROR: generated_replay.npz not found. Run generate_replay.py first.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    analyze_replay()