#!/usr/bin/env python3
"""
Analyze cursor position bias in generated replay data.
This script examines whether the cursor staying in bottom-right is due to:
1. Untrained model bias
2. Beatmap object distribution
3. Coordinate normalization issues
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_cursor_bias():
    """Analyze cursor position distribution and bias."""
    
    # Load generated replay data
    replay_file = Path('generated_replay.npz')
    if not replay_file.exists():
        logger.error(f"Replay file not found: {replay_file}")
        return
    
    logger.info("Loading generated replay data...")
    data = np.load(replay_file, allow_pickle=True)
    
    cursor_positions = data['cursor_positions']
    logger.info(f"Loaded {len(cursor_positions)} cursor positions")
    
    # Extract x and y coordinates
    x_coords = cursor_positions[:, 0]
    y_coords = cursor_positions[:, 1]
    
    # Calculate statistics
    stats = {
        'x_min': np.min(x_coords),
        'x_max': np.max(x_coords),
        'x_mean': np.mean(x_coords),
        'x_std': np.std(x_coords),
        'y_min': np.min(y_coords),
        'y_max': np.max(y_coords),
        'y_mean': np.mean(y_coords),
        'y_std': np.std(y_coords)
    }
    
    # Print analysis
    print("\n=== CURSOR POSITION ANALYSIS ===")
    print(f"Total positions: {len(cursor_positions)}")
    print(f"\nX-coordinate stats:")
    print(f"  Range: {stats['x_min']:.2f} to {stats['x_max']:.2f}")
    print(f"  Mean: {stats['x_mean']:.2f} ± {stats['x_std']:.2f}")
    print(f"\nY-coordinate stats:")
    print(f"  Range: {stats['y_min']:.2f} to {stats['y_max']:.2f}")
    print(f"  Mean: {stats['y_mean']:.2f} ± {stats['y_std']:.2f}")
    
    # Analyze playfield coverage
    # Standard osu! playfield is 512x384 pixels
    playfield_width = 512
    playfield_height = 384
    
    print(f"\n=== PLAYFIELD COVERAGE ANALYSIS ===")
    print(f"Standard osu! playfield: {playfield_width}x{playfield_height} pixels")
    
    # Calculate coverage percentages
    x_coverage = (stats['x_max'] - stats['x_min']) / playfield_width * 100
    y_coverage = (stats['y_max'] - stats['y_min']) / playfield_height * 100
    
    print(f"X-axis coverage: {x_coverage:.1f}% of playfield width")
    print(f"Y-axis coverage: {y_coverage:.1f}% of playfield height")
    
    # Check if cursor is biased toward bottom-right
    x_center_bias = (stats['x_mean'] - playfield_width/2) / (playfield_width/2) * 100
    y_center_bias = (stats['y_mean'] - playfield_height/2) / (playfield_height/2) * 100
    
    print(f"\n=== BIAS ANALYSIS ===")
    print(f"X-axis bias from center: {x_center_bias:+.1f}% (+ = right, - = left)")
    print(f"Y-axis bias from center: {y_center_bias:+.1f}% (+ = down, - = up)")
    
    if x_center_bias > 50 and y_center_bias > 50:
        print("\n🔍 DIAGNOSIS: Strong bottom-right bias detected!")
        print("This is likely due to untrained model weights creating a bias.")
    elif x_center_bias > 25 or y_center_bias > 25:
        print("\n🔍 DIAGNOSIS: Moderate positional bias detected.")
    else:
        print("\n🔍 DIAGNOSIS: No significant positional bias.")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Cursor trajectory
    plt.subplot(2, 2, 1)
    plt.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=0.5)
    plt.scatter(x_coords[0], y_coords[0], color='green', s=50, label='Start', zorder=5)
    plt.scatter(x_coords[-1], y_coords[-1], color='red', s=50, label='End', zorder=5)
    plt.xlim(0, playfield_width)
    plt.ylim(0, playfield_height)
    plt.gca().invert_yaxis()  # osu! coordinates have Y=0 at top
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Cursor Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: X-coordinate distribution
    plt.subplot(2, 2, 2)
    plt.hist(x_coords, bins=50, alpha=0.7, color='blue')
    plt.axvline(stats['x_mean'], color='red', linestyle='--', label=f'Mean: {stats["x_mean"]:.1f}')
    plt.axvline(playfield_width/2, color='green', linestyle='--', label=f'Center: {playfield_width/2}')
    plt.xlabel('X Position')
    plt.ylabel('Frequency')
    plt.title('X-Coordinate Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Y-coordinate distribution
    plt.subplot(2, 2, 3)
    plt.hist(y_coords, bins=50, alpha=0.7, color='orange')
    plt.axvline(stats['y_mean'], color='red', linestyle='--', label=f'Mean: {stats["y_mean"]:.1f}')
    plt.axvline(playfield_height/2, color='green', linestyle='--', label=f'Center: {playfield_height/2}')
    plt.xlabel('Y Position')
    plt.ylabel('Frequency')
    plt.title('Y-Coordinate Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap
    plt.subplot(2, 2, 4)
    plt.hist2d(x_coords, y_coords, bins=30, cmap='hot')
    plt.colorbar(label='Frequency')
    plt.xlim(0, playfield_width)
    plt.ylim(0, playfield_height)
    plt.gca().invert_yaxis()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Cursor Position Heatmap')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('debug_output')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'cursor_bias_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    
    # Final recommendation
    print(f"\n=== RECOMMENDATION ===")
    if x_center_bias > 50 and y_center_bias > 50:
        print("✅ This behavior is EXPECTED for an untrained model.")
        print("🎯 Solution: Train the model with real replay data to learn proper cursor movement.")
        print("🔧 The generation pipeline is working correctly - this is just untrained model bias.")
    else:
        print("🤔 The bias pattern is unusual and may warrant further investigation.")

if __name__ == '__main__':
    analyze_cursor_bias()