#!/usr/bin/env python3
"""
Debug script to analyze generated replay data and identify issues with cursor movement.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_replay_data(replay_path: str) -> Dict[str, np.ndarray]:
    """Load replay data from .npz file."""
    try:
        data = np.load(replay_path, allow_pickle=True)
        return {
            'cursor_positions': data['cursor_positions'],
            'key_presses': data['key_presses'],
            'timestamps': data['timestamps'],
            'metadata': data['metadata'].item() if 'metadata' in data else {}
        }
    except Exception as e:
        logger.error(f"Failed to load replay data: {e}")
        return {}

def analyze_cursor_movement(cursor_positions: np.ndarray) -> Dict[str, Any]:
    """Analyze cursor movement patterns."""
    analysis = {}
    
    # Basic statistics
    analysis['total_frames'] = len(cursor_positions)
    analysis['x_positions'] = cursor_positions[:, 0].tolist()
    analysis['y_positions'] = cursor_positions[:, 1].tolist()
    
    # Movement analysis
    if len(cursor_positions) > 1:
        # Calculate movement deltas
        deltas = np.diff(cursor_positions, axis=0)
        distances = np.sqrt(np.sum(deltas**2, axis=1))
        
        analysis['movement_stats'] = {
            'total_distance': float(np.sum(distances)),
            'average_distance_per_frame': float(np.mean(distances)),
            'max_distance_per_frame': float(np.max(distances)),
            'min_distance_per_frame': float(np.min(distances)),
            'stationary_frames': int(np.sum(distances < 0.1)),  # Frames with minimal movement
            'stationary_percentage': float(np.sum(distances < 0.1) / len(distances) * 100)
        }
        
        # Position statistics
        analysis['position_stats'] = {
            'x_range': [float(np.min(cursor_positions[:, 0])), float(np.max(cursor_positions[:, 0]))],
            'y_range': [float(np.min(cursor_positions[:, 1])), float(np.max(cursor_positions[:, 1]))],
            'x_std': float(np.std(cursor_positions[:, 0])),
            'y_std': float(np.std(cursor_positions[:, 1])),
            'center_of_mass': [float(np.mean(cursor_positions[:, 0])), float(np.mean(cursor_positions[:, 1]))]
        }
        
        # Check for stuck cursor (same position for extended periods)
        unique_positions = np.unique(cursor_positions, axis=0)
        analysis['unique_positions'] = len(unique_positions)
        analysis['position_diversity'] = len(unique_positions) / len(cursor_positions)
        
        # Find longest stationary period
        stationary_mask = distances < 0.1
        stationary_runs = []
        current_run = 0
        for is_stationary in stationary_mask:
            if is_stationary:
                current_run += 1
            else:
                if current_run > 0:
                    stationary_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            stationary_runs.append(current_run)
        
        analysis['longest_stationary_period'] = max(stationary_runs) if stationary_runs else 0
        analysis['stationary_runs'] = stationary_runs
    
    return analysis

def analyze_key_presses(key_presses: np.ndarray) -> Dict[str, Any]:
    """Analyze key press patterns."""
    analysis = {}
    
    # Key press statistics
    analysis['total_frames'] = len(key_presses)
    
    # Count presses for each key (assuming 4 keys: M1, M2, K1, K2)
    key_names = ['M1', 'M2', 'K1', 'K2']
    for i, key_name in enumerate(key_names):
        key_column = key_presses[:, i]
        analysis[f'{key_name}_presses'] = int(np.sum(key_column > 0.5))
        analysis[f'{key_name}_press_percentage'] = float(np.sum(key_column > 0.5) / len(key_column) * 100)
    
    # Total key activity
    any_key_pressed = np.any(key_presses > 0.5, axis=1)
    analysis['total_active_frames'] = int(np.sum(any_key_pressed))
    analysis['activity_percentage'] = float(np.sum(any_key_pressed) / len(key_presses) * 100)
    
    # Key press patterns
    analysis['key_press_data'] = key_presses.tolist()
    
    return analysis

def create_visualization(cursor_positions: np.ndarray, key_presses: np.ndarray, 
                        timestamps: np.ndarray, output_dir: str = "debug_output"):
    """Create visualizations of the replay data."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Cursor movement plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(cursor_positions[:, 0], cursor_positions[:, 1], 'b-', alpha=0.7, linewidth=1)
    plt.scatter(cursor_positions[0, 0], cursor_positions[0, 1], color='green', s=100, label='Start', zorder=5)
    plt.scatter(cursor_positions[-1, 0], cursor_positions[-1, 1], color='red', s=100, label='End', zorder=5)
    plt.xlim(0, 512)
    plt.ylim(0, 384)
    plt.gca().invert_yaxis()  # osu! coordinate system
    plt.title('Cursor Movement Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # X position over time
    plt.subplot(2, 2, 2)
    plt.plot(timestamps, cursor_positions[:, 0], 'r-', linewidth=1)
    plt.title('X Position Over Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('X Position')
    plt.grid(True, alpha=0.3)
    
    # Y position over time
    plt.subplot(2, 2, 3)
    plt.plot(timestamps, cursor_positions[:, 1], 'g-', linewidth=1)
    plt.title('Y Position Over Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)
    
    # Key presses over time
    plt.subplot(2, 2, 4)
    key_names = ['M1', 'M2', 'K1', 'K2']
    colors = ['red', 'blue', 'orange', 'purple']
    for i, (key_name, color) in enumerate(zip(key_names, colors)):
        key_data = key_presses[:, i]
        # Show key presses as vertical lines
        press_times = timestamps[key_data > 0.5]
        for press_time in press_times:
            plt.axvline(x=press_time, color=color, alpha=0.7, linewidth=1, label=key_name if press_time == press_times[0] else "")
    
    plt.title('Key Presses Over Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('Key')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/replay_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {output_dir}/replay_analysis.png")

def main():
    """Main analysis function."""
    replay_file = "generated_replay.npz"
    
    if not Path(replay_file).exists():
        logger.error(f"Replay file not found: {replay_file}")
        logger.info("Please run generate_replay.py first to create the replay data.")
        return
    
    logger.info(f"Loading replay data from {replay_file}...")
    replay_data = load_replay_data(replay_file)
    
    if not replay_data:
        logger.error("Failed to load replay data")
        return
    
    cursor_positions = replay_data['cursor_positions']
    key_presses = replay_data['key_presses']
    timestamps = replay_data['timestamps']
    metadata = replay_data['metadata']
    
    logger.info(f"Loaded replay with {len(cursor_positions)} frames")
    
    # Analyze cursor movement
    logger.info("Analyzing cursor movement...")
    cursor_analysis = analyze_cursor_movement(cursor_positions)
    
    # Analyze key presses
    logger.info("Analyzing key presses...")
    key_analysis = analyze_key_presses(key_presses)
    
    # Combine all analysis
    full_analysis = {
        'metadata': metadata,
        'cursor_analysis': cursor_analysis,
        'key_analysis': key_analysis,
        'file_info': {
            'file_size_bytes': Path(replay_file).stat().st_size,
            'total_frames': len(cursor_positions)
        }
    }
    
    # Save analysis to JSON
    output_file = "replay_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    
    logger.info(f"Analysis saved to {output_file}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualization(cursor_positions, key_presses, timestamps)
    
    # Print summary
    print("\n" + "="*60)
    print("REPLAY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total frames: {len(cursor_positions)}")
    print(f"File size: {Path(replay_file).stat().st_size} bytes")
    
    if 'movement_stats' in cursor_analysis:
        print(f"\nCURSOR MOVEMENT:")
        print(f"  Total distance traveled: {cursor_analysis['movement_stats']['total_distance']:.2f} pixels")
        print(f"  Average movement per frame: {cursor_analysis['movement_stats']['average_distance_per_frame']:.2f} pixels")
        print(f"  Stationary frames: {cursor_analysis['movement_stats']['stationary_frames']} ({cursor_analysis['movement_stats']['stationary_percentage']:.1f}%)")
        print(f"  Longest stationary period: {cursor_analysis['longest_stationary_period']} frames")
        print(f"  Unique positions: {cursor_analysis['unique_positions']} ({cursor_analysis['position_diversity']*100:.1f}% diversity)")
        print(f"  Position range: X[{cursor_analysis['position_stats']['x_range'][0]:.1f}, {cursor_analysis['position_stats']['x_range'][1]:.1f}], Y[{cursor_analysis['position_stats']['y_range'][0]:.1f}, {cursor_analysis['position_stats']['y_range'][1]:.1f}]")
    
    print(f"\nKEY ACTIVITY:")
    print(f"  Total active frames: {key_analysis['total_active_frames']} ({key_analysis['activity_percentage']:.1f}%)")
    for key in ['M1', 'M2', 'K1', 'K2']:
        print(f"  {key} presses: {key_analysis[f'{key}_presses']} ({key_analysis[f'{key}_press_percentage']:.1f}%)")
    
    # Identify potential issues
    print(f"\nPOTENTIAL ISSUES:")
    issues = []
    
    if 'movement_stats' in cursor_analysis:
        if cursor_analysis['movement_stats']['stationary_percentage'] > 80:
            issues.append(f"Cursor is stationary {cursor_analysis['movement_stats']['stationary_percentage']:.1f}% of the time")
        
        if cursor_analysis['longest_stationary_period'] > 100:
            issues.append(f"Cursor stays in same position for {cursor_analysis['longest_stationary_period']} consecutive frames")
        
        if cursor_analysis['position_diversity'] < 0.1:
            issues.append(f"Very low position diversity ({cursor_analysis['position_diversity']*100:.1f}%)")
        
        if cursor_analysis['movement_stats']['total_distance'] < 100:
            issues.append(f"Very low total movement distance ({cursor_analysis['movement_stats']['total_distance']:.1f} pixels)")
    
    if key_analysis['activity_percentage'] > 50:
        issues.append(f"High key activity ({key_analysis['activity_percentage']:.1f}%) - might be random tapping")
    
    if Path(replay_file).stat().st_size < 10000:  # Less than 10KB
        issues.append(f"Very small file size ({Path(replay_file).stat().st_size} bytes) suggests minimal data")
    
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"  ✅ No obvious issues detected")
    
    print("\n" + "="*60)
    print(f"Full analysis saved to: {output_file}")
    print(f"Visualization saved to: debug_output/replay_analysis.png")
    print("="*60)

if __name__ == "__main__":
    main()