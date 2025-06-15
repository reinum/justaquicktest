#!/usr/bin/env python3

import numpy as np
import osrparse
from pathlib import Path
import sys

def load_npz_replay(npz_path):
    """Load replay data from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'cursor_data': data['cursor_positions'],
        'key_data': data['key_presses'],
        'time_data': data['timestamps'],
        'metadata': data['metadata']
    }

def convert_key_state_to_osrparse(key_state):
    """Convert key state array to osrparse Key enum."""
    # key_state is [K1, K2, M1, M2] where 1 means pressed, 0 means not pressed
    key_value = 0
    if len(key_state) >= 4:
        if key_state[0]:  # K1
            key_value |= osrparse.Key.K1.value
        if key_state[1]:  # K2
            key_value |= osrparse.Key.K2.value
        if key_state[2]:  # M1
            key_value |= osrparse.Key.M1.value
        if key_state[3]:  # M2
            key_value |= osrparse.Key.M2.value
    
    # Return the combined key value as osrparse.Key
    return osrparse.Key(key_value)

def estimate_hit_counts(replay_data):
    """Estimate hit counts from replay data."""
    # Simple estimation based on key presses
    total_frames = len(replay_data['key_data'])
    
    # Count key press events
    key_presses = 0
    prev_keys = np.zeros(4)
    
    for keys in replay_data['key_data']:
        # Count new key presses (transition from 0 to 1)
        new_presses = np.sum((keys > 0) & (prev_keys == 0))
        key_presses += new_presses
        prev_keys = keys.copy()
    
    # Distribute hits (rough estimation)
    count_300 = int(key_presses * 0.8)  # 80% perfect hits
    count_100 = int(key_presses * 0.15)  # 15% good hits
    count_50 = int(key_presses * 0.04)   # 4% ok hits
    count_miss = key_presses - count_300 - count_100 - count_50  # remaining misses
    
    return count_300, count_100, count_50, count_miss

def convert_npz_to_osr(npz_path, original_osr_path, output_path):
    """Convert NPZ replay to OSR format using original replay metadata."""
    
    # Load original replay for metadata
    print(f"Loading original replay: {original_osr_path}")
    original = osrparse.Replay.from_path(original_osr_path)
    
    # Load NPZ replay data
    print(f"Loading NPZ replay data: {npz_path}")
    replay_data = load_npz_replay(npz_path)
    
    # Convert replay data to osrparse format
    print("Converting replay events...")
    events = []
    
    cursor_data = replay_data['cursor_data']
    key_data = replay_data['key_data']
    time_data = replay_data['time_data']
    
    for i in range(len(cursor_data)):
        time_delta = int(time_data[i]) if i == 0 else int(time_data[i] - time_data[i-1])
        x = float(cursor_data[i][0])
        y = float(cursor_data[i][1])
        keys = convert_key_state_to_osrparse(key_data[i])
        
        event = osrparse.ReplayEventOsu(time_delta, x, y, keys)
        events.append(event)
    
    # Estimate hit counts
    count_300, count_100, count_50, count_miss = estimate_hit_counts(replay_data)
    
    # Calculate score (simple estimation)
    score = count_300 * 300 + count_100 * 100 + count_50 * 50
    max_combo = count_300 + count_100 + count_50  # Assuming no combo breaks
    
    print(f"Estimated stats: 300s={count_300}, 100s={count_100}, 50s={count_50}, misses={count_miss}")
    print(f"Estimated score: {score}, max combo: {max_combo}")
    
    # Create new replay using original metadata
    new_replay = osrparse.Replay(
        mode=original.mode,
        game_version=original.game_version,
        beatmap_hash=original.beatmap_hash,  # Use original beatmap hash
        username="AI Player",
        replay_hash="",  # Will be calculated automatically
        count_300=count_300,
        count_100=count_100,
        count_50=count_50,
        count_geki=0,  # For osu!standard, geki = count_300 in some cases
        count_katu=0,  # For osu!standard, katu = count_100 in some cases
        count_miss=count_miss,
        score=score,
        max_combo=max_combo,
        perfect=1 if count_miss == 0 else 0,
        mods=original.mods,  # Use original mods
        life_bar_graph=[],  # Empty life bar for now
        timestamp=original.timestamp,  # Use original timestamp
        replay_data=events,
        replay_id=0,
        rng_seed=12345
    )
    
    # Write the new replay
    print(f"Writing OSR file: {output_path}")
    new_replay.write_path(output_path)
    
    # Verify the written file
    file_size = Path(output_path).stat().st_size
    print(f"Successfully created {output_path} ({file_size} bytes)")
    
    # Verify by reading it back
    try:
        verify = osrparse.Replay.from_path(output_path)
        print(f"Verification successful: {len(verify.replay_data)} events")
        print(f"Beatmap hash: {verify.beatmap_hash}")
        print(f"Username: {verify.username}")
        print(f"Score: {verify.score}")
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    npz_file = "generated_replay.npz"
    original_osr = "ogreplay.osr"
    output_osr = "ai_replay_final.osr"
    
    if not Path(npz_file).exists():
        print(f"Error: {npz_file} not found")
        sys.exit(1)
        
    if not Path(original_osr).exists():
        print(f"Error: {original_osr} not found")
        sys.exit(1)
    
    success = convert_npz_to_osr(npz_file, original_osr, output_osr)
    
    if success:
        print("\nConversion completed successfully!")
        print(f"You can now use {output_osr} in osu!")
    else:
        print("\nConversion failed!")
        sys.exit(1)