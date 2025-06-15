#!/usr/bin/env python3
"""
Analyze generated replay data and create visualizations including cursor heatmap
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime
import os

def load_replay_data(npz_path):
    """Load replay data from .npz file"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        print(f"Available keys in replay data: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"Error loading replay data: {e}")
        return None

def analyze_cursor_movement(cursor_data):
    """Analyze cursor movement patterns"""
    if len(cursor_data.shape) == 1:
        # Reshape if needed
        cursor_data = cursor_data.reshape(-1, 2)
    
    x_coords = cursor_data[:, 0]
    y_coords = cursor_data[:, 1]
    
    # Calculate movement statistics
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    velocities = distances  # Assuming 1 unit time between points
    
    analysis = {
        'total_points': len(cursor_data),
        'x_range': [float(np.min(x_coords)), float(np.max(x_coords))],
        'y_range': [float(np.min(y_coords)), float(np.max(y_coords))],
        'total_distance': float(np.sum(distances)),
        'average_velocity': float(np.mean(velocities)),
        'max_velocity': float(np.max(velocities)),
        'min_velocity': float(np.min(velocities)),
        'velocity_std': float(np.std(velocities)),
        'center_of_mass': [float(np.mean(x_coords)), float(np.mean(y_coords))]
    }
    
    return analysis, cursor_data

def analyze_key_presses(key_data):
    """Analyze key press patterns"""
    if key_data is None or len(key_data) == 0:
        return {'total_presses': 0, 'press_pattern': []}
    
    # Count key presses
    total_presses = np.sum(key_data > 0) if key_data.dtype != bool else np.sum(key_data)
    
    # Analyze press patterns
    press_changes = np.diff(key_data.astype(int))
    press_starts = np.where(press_changes > 0)[0]
    press_ends = np.where(press_changes < 0)[0]
    
    # Calculate average press duration safely
    avg_duration = 0
    if len(press_starts) > 0 and len(press_ends) > 0:
        # Match press starts with press ends
        min_len = min(len(press_starts), len(press_ends))
        if min_len > 0:
            # Take only matching pairs
            matched_starts = press_starts[:min_len]
            matched_ends = press_ends[:min_len]
            # Ensure ends come after starts
            valid_pairs = matched_ends > matched_starts
            if np.any(valid_pairs):
                durations = matched_ends[valid_pairs] - matched_starts[valid_pairs]
                avg_duration = float(np.mean(durations))
    
    analysis = {
        'total_presses': int(total_presses),
        'press_starts': len(press_starts),
        'press_ends': len(press_ends),
        'average_press_duration': avg_duration
    }
    
    return analysis

def create_cursor_heatmap(cursor_data, output_path='cursor_heatmap.png'):
    """Create a heatmap visualization of cursor positions"""
    plt.figure(figsize=(12, 9))
    
    # OSU! standard playfield dimensions (512x384 pixels)
    playfield_width = 512
    playfield_height = 384
    
    x_coords = cursor_data[:, 0]
    y_coords = cursor_data[:, 1]
    
    # Create 2D histogram for heatmap
    bins_x = 50
    bins_y = 38  # Maintain aspect ratio
    
    # Create heatmap
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=[bins_x, bins_y])
    
    # Create custom colormap
    colors = ['#000033', '#000055', '#000077', '#0000BB', '#0000FF', 
              '#3333FF', '#6666FF', '#9999FF', '#CCCCFF', '#FFFFFF']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('cursor_heat', colors, N=n_bins)
    
    # Plot heatmap
    plt.imshow(heatmap.T, origin='lower', cmap=cmap, 
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               aspect='auto', interpolation='bilinear')
    
    # Add playfield boundary
    rect = patches.Rectangle((0, 0), playfield_width, playfield_height, 
                           linewidth=2, edgecolor='white', facecolor='none', alpha=0.7)
    plt.gca().add_patch(rect)
    
    # Customize plot
    plt.colorbar(label='Cursor Density')
    plt.title('Cursor Movement Heatmap\nGenerated Replay Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('X Position (pixels)', fontsize=12)
    plt.ylabel('Y Position (pixels)', fontsize=12)
    
    # Add statistics text
    stats_text = f"Total Points: {len(cursor_data)}\n"
    stats_text += f"X Range: {np.min(x_coords):.1f} - {np.max(x_coords):.1f}\n"
    stats_text += f"Y Range: {np.min(y_coords):.1f} - {np.max(y_coords):.1f}\n"
    stats_text += f"Center: ({np.mean(x_coords):.1f}, {np.mean(y_coords):.1f})"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cursor heatmap saved to: {output_path}")
    plt.show()

def create_movement_trajectory(cursor_data, output_path='cursor_trajectory.png'):
    """Create a trajectory plot showing cursor movement over time"""
    plt.figure(figsize=(12, 9))
    
    x_coords = cursor_data[:, 0]
    y_coords = cursor_data[:, 1]
    
    # Create trajectory plot with color gradient
    points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from matplotlib.collections import LineCollection
    
    # Create line collection with color gradient
    lc = LineCollection(segments, cmap='viridis', alpha=0.7)
    lc.set_array(np.arange(len(segments)))
    lc.set_linewidth(1.5)
    
    plt.gca().add_collection(lc)
    
    # Mark start and end points
    plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', 
                label='Start', zorder=5, edgecolors='black')
    plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='s', 
                label='End', zorder=5, edgecolors='black')
    
    # Add playfield boundary
    rect = patches.Rectangle((0, 0), 512, 384, 
                           linewidth=2, edgecolor='black', facecolor='none', alpha=0.5)
    plt.gca().add_patch(rect)
    
    plt.xlim(min(0, np.min(x_coords) - 20), max(512, np.max(x_coords) + 20))
    plt.ylim(min(0, np.min(y_coords) - 20), max(384, np.max(y_coords) + 20))
    
    plt.colorbar(lc, label='Time Progression')
    plt.title('Cursor Movement Trajectory\nGenerated Replay Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('X Position (pixels)', fontsize=12)
    plt.ylabel('Y Position (pixels)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cursor trajectory saved to: {output_path}")
    plt.show()

def create_velocity_analysis(cursor_data, output_path='velocity_analysis.png'):
    """Create velocity analysis plots"""
    x_coords = cursor_data[:, 0]
    y_coords = cursor_data[:, 1]
    
    # Calculate velocities
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Velocity over time
    ax1.plot(distances, alpha=0.7, linewidth=1)
    ax1.set_title('Cursor Velocity Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Velocity (pixels/step)')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    ax1.axhline(np.mean(distances), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(distances):.2f}')
    ax1.axhline(np.median(distances), color='green', linestyle='--', alpha=0.7, label=f'Median: {np.median(distances):.2f}')
    ax1.legend()
    
    # Velocity distribution
    ax2.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_title('Cursor Velocity Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Velocity (pixels/step)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Velocity analysis saved to: {output_path}")
    plt.show()

def export_to_json(replay_data, cursor_analysis, key_analysis, output_path='replay_analysis.json'):
    """Export analysis results to JSON for self-analysis"""
    
    # Prepare data for JSON export
    json_data = {
        'metadata': {
            'analysis_timestamp': datetime.now().isoformat(),
            'source_file': 'generated_replay.npz',
            'beatmap': 'testmap.osu'
        },
        'cursor_analysis': cursor_analysis,
        'key_analysis': key_analysis,
        'raw_data_summary': {
            'available_keys': list(replay_data.keys()),
            'data_shapes': {key: list(replay_data[key].shape) for key in replay_data.keys()}
        }
    }
    
    # Add cursor positions (sample for analysis)
    if 'cursor_pos' in replay_data:
        cursor_data = replay_data['cursor_pos']
        if len(cursor_data.shape) == 1:
            cursor_data = cursor_data.reshape(-1, 2)
        
        # Sample every 10th point to reduce file size
        sample_indices = np.arange(0, len(cursor_data), 10)
        sampled_cursor = cursor_data[sample_indices]
        
        json_data['sampled_cursor_positions'] = {
            'sample_rate': 10,
            'total_original_points': len(cursor_data),
            'sampled_points': len(sampled_cursor),
            'positions': sampled_cursor.tolist()
        }
    
    # Add key press data (sample)
    if 'key_pressed' in replay_data:
        key_data = replay_data['key_pressed']
        sample_indices = np.arange(0, len(key_data), 10)
        sampled_keys = key_data[sample_indices]
        
        json_data['sampled_key_presses'] = {
            'sample_rate': 10,
            'total_original_points': len(key_data),
            'sampled_points': len(sampled_keys),
            'key_states': sampled_keys.tolist()
        }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Analysis data exported to JSON: {output_path}")
    return json_data

def main():
    """Main analysis function"""
    print("=== Generated Replay Analysis ===")
    print("Loading replay data...")
    
    # Load replay data
    replay_data = load_replay_data('generated_replay.npz')
    if replay_data is None:
        print("Failed to load replay data")
        return
    
    print(f"\nLoaded replay data with keys: {list(replay_data.keys())}")
    
    # Analyze cursor movement
    cursor_key = None
    for key in ['cursor_pos', 'cursor_positions', 'positions', 'cursor']:
        if key in replay_data:
            cursor_key = key
            break
    
    if cursor_key is None:
        print("No cursor position data found in replay")
        return
    
    print(f"\nAnalyzing cursor movement from key: {cursor_key}")
    cursor_analysis, cursor_data = analyze_cursor_movement(replay_data[cursor_key])
    
    # Analyze key presses
    key_key = None
    for key in ['key_pressed', 'keys', 'key_presses', 'input']:
        if key in replay_data:
            key_key = key
            break
    
    if key_key:
        print(f"Analyzing key presses from key: {key_key}")
        key_analysis = analyze_key_presses(replay_data[key_key])
    else:
        print("No key press data found")
        key_analysis = {'total_presses': 0, 'note': 'No key data available'}
    
    # Print analysis results
    print("\n=== CURSOR ANALYSIS ===")
    for key, value in cursor_analysis.items():
        print(f"{key}: {value}")
    
    print("\n=== KEY PRESS ANALYSIS ===")
    for key, value in key_analysis.items():
        print(f"{key}: {value}")
    
    # Export to JSON
    print("\n=== EXPORTING TO JSON ===")
    json_data = export_to_json(replay_data, cursor_analysis, key_analysis)
    
    # Create visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    
    print("Creating cursor heatmap...")
    create_cursor_heatmap(cursor_data)
    
    print("Creating movement trajectory...")
    create_movement_trajectory(cursor_data)
    
    print("Creating velocity analysis...")
    create_velocity_analysis(cursor_data)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Generated files:")
    print("- replay_analysis.json (JSON data for self-analysis)")
    print("- cursor_heatmap.png (Cursor density heatmap)")
    print("- cursor_trajectory.png (Movement trajectory)")
    print("- velocity_analysis.png (Velocity analysis)")

if __name__ == '__main__':
    main()