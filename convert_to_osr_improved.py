#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import numpy as np
import logging
import argparse

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

import osrparse
from src.data.beatmap_parser import BeatmapParser
from src.generation.generator import ReplayGenerator
from src.config.model_config import ModelConfig

def load_config(config_path: str = "config.yaml") -> ModelConfig:
    """Load configuration from YAML file."""
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return ModelConfig(**config_dict)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return ModelConfig()
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return ModelConfig()

def load_npz_replay(npz_path: str):
    """Load replay data from NPZ file."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract data
        timestamps = data['timestamps']
        cursor_positions = data['cursor_positions']
        key_presses = data['key_presses']
        
        # Try to load metadata if available
        metadata = {}
        try:
            if 'metadata' in data:
                metadata = data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
        except Exception as e:
            logging.warning(f"Could not load metadata: {e}")
        
        return {
            'timestamps': timestamps,
            'cursor_positions': cursor_positions,
            'key_presses': key_presses,
            'metadata': metadata
        }
    except Exception as e:
        logging.error(f"Error loading NPZ file: {e}")
        return None

def convert_key_state_to_osu(key_state: np.ndarray) -> int:
    """Convert key state array to osu! key format.
    
    Args:
        key_state: Array of shape [4] representing [K1, K2, M1, M2]
        
    Returns:
        Integer representing key state in osu! format
    """
    keys = 0
    
    # osu! key mapping:
    # 1 = M1 (left mouse)
    # 2 = M2 (right mouse) 
    # 4 = K1 (keyboard key 1)
    # 8 = K2 (keyboard key 2)
    
    if key_state[2] > 0.5:  # M1
        keys |= 1
    if key_state[3] > 0.5:  # M2
        keys |= 2
    if key_state[0] > 0.5:  # K1
        keys |= 4
    if key_state[1] > 0.5:  # K2
        keys |= 8
    
    return keys

def create_replay_events(timestamps, cursor_positions, key_presses):
    """Create osrparse ReplayEvent objects from our data."""
    events = []
    
    for i in range(len(timestamps)):
        # Calculate time delta
        if i == 0:
            time_delta = int(timestamps[i])
        else:
            time_delta = int(timestamps[i] - timestamps[i-1])
        
        # Get cursor position
        x = float(cursor_positions[i, 0])
        y = float(cursor_positions[i, 1])
        
        # Convert key state
        keys = convert_key_state_to_osu(key_presses[i])
        
        # Create ReplayEvent
        event = osrparse.ReplayEventOsu(time_delta, x, y, keys)
        events.append(event)
    
    return events

def convert_to_osr_with_original_beatmap(npz_path: str, original_osr_path: str, output_path: str = "generated_replay.osr"):
    """Convert NPZ replay to OSR format using the beatmap info from an original OSR file."""
    
    # Load the original replay to get beatmap information
    try:
        original_replay = osrparse.Replay.from_path(original_osr_path)
        print(f"Loaded original replay: {original_osr_path}")
        print(f"Original beatmap hash: {original_replay.beatmap_hash}")
        print(f"Original username: {original_replay.username}")
        print(f"Original score: {original_replay.score}")
    except Exception as e:
        print(f"Error loading original replay: {e}")
        return False
    
    # Load our generated replay data
    replay_data = load_npz_replay(npz_path)
    if replay_data is None:
        print("Failed to load NPZ replay data")
        return False
    
    print(f"Loaded NPZ replay with {len(replay_data['timestamps'])} frames")
    
    # Create replay events
    replay_events = create_replay_events(
        replay_data['timestamps'],
        replay_data['cursor_positions'],
        replay_data['key_presses']
    )
    
    print(f"Created {len(replay_events)} replay events")
    
    # Calculate some basic statistics
    total_frames = len(replay_events)
    
    # Estimate hit counts (simplified)
    # In a real implementation, you'd analyze the actual hits
    estimated_300s = int(total_frames * 0.8)  # Assume 80% perfect hits
    estimated_100s = int(total_frames * 0.15)  # 15% good hits
    estimated_50s = int(total_frames * 0.05)   # 5% okay hits
    
    # Create new replay object using osrparse
    new_replay = osrparse.Replay(
        mode=0,  # osu! standard
        game_version=20210520,  # Recent game version
        beatmap_hash=original_replay.beatmap_hash,  # Use original beatmap hash
        username="AI Player",
        replay_hash="",  # Will be calculated automatically
        count_300=estimated_300s,
        count_100=estimated_100s,
        count_50=estimated_50s,
        count_geki=0,
        count_katu=0,
        count_miss=0,
        score=1000000,  # High score
        max_combo=total_frames,  # Assume full combo
        perfect=True,
        mods=0,  # No mods
        life_bar_graph=[],  # Empty life bar for now
        timestamp=original_replay.timestamp,  # Use original timestamp
        replay_data=replay_events,
        replay_id=0,  # Offline replay
        rng_seed=12345  # Random seed
    )
    
    # Write the replay to file
    try:
        new_replay.write_path(output_path)
        print(f"Successfully created {output_path}")
        
        # Verify the created file
        verify_replay = osrparse.Replay.from_path(output_path)
        print(f"Verification: Created replay has {len(verify_replay.replay_data)} frames")
        print(f"Verification: Beatmap hash matches: {verify_replay.beatmap_hash == original_replay.beatmap_hash}")
        print(f"Verification: File size: {Path(output_path).stat().st_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"Error writing replay: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert NPZ replay to OSR format')
    parser.add_argument('npz_file', help='Path to NPZ replay file')
    parser.add_argument('original_osr', help='Path to original OSR file for beatmap info')
    parser.add_argument('-o', '--output', default='generated_replay.osr', help='Output OSR file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    print(f"Converting {args.npz_file} to {args.output}")
    print(f"Using beatmap info from {args.original_osr}")
    
    success = convert_to_osr_with_original_beatmap(args.npz_file, args.original_osr, args.output)
    
    if success:
        print("\nConversion completed successfully!")
        print(f"Generated replay: {args.output}")
        print("\nTo use the replay:")
        print("1. Copy the .osr file to your osu!/Replays/ folder")
        print("2. Open osu! and go to the beatmap")
        print("3. Press F2 to open local rankings and find your replay")
    else:
        print("\nConversion failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())