#!/usr/bin/env python3

import osrparse
import sys
from pathlib import Path

# Test osrparse functionality
print("Testing osrparse...")

# Load the original replay
try:
    original = osrparse.Replay.from_path("ogreplay.osr")
    print(f"Successfully loaded original replay")
    print(f"Mode: {original.mode}")
    print(f"Game version: {original.game_version}")
    print(f"Beatmap hash: {original.beatmap_hash}")
    print(f"Username: {original.username}")
    print(f"Score: {original.score}")
    print(f"Max combo: {original.max_combo}")
    print(f"Perfect: {original.perfect}")
    print(f"Mods: {original.mods}")
    print(f"Timestamp: {original.timestamp}")
    print(f"Replay data length: {len(original.replay_data)}")
    
    # Check the type of mods and mode
    print(f"\nType analysis:")
    print(f"Mode type: {type(original.mode)}")
    print(f"Mods type: {type(original.mods)}")
    print(f"Perfect type: {type(original.perfect)}")
    
    # Try to create a simple replay event
    print(f"\nTesting ReplayEvent creation:")
    # Use Key enum instead of integer
    keys = osrparse.Key.M1  # Left mouse button
    event = osrparse.ReplayEventOsu(100, 256.0, 192.0, keys)
    print(f"Created event: time_delta={event.time_delta}, x={event.x}, y={event.y}, keys={event.keys}")
    
    # Try to create a minimal replay
    print(f"\nTesting Replay creation:")
    test_events = [osrparse.ReplayEventOsu(100, 256.0, 192.0, keys)]
    
    # Use the same values as the original replay but with minimal data
    test_replay = osrparse.Replay(
        mode=original.mode,
        game_version=original.game_version,
        beatmap_hash=original.beatmap_hash,
        username="Test Player",
        replay_hash="",
        count_300=1,
        count_100=0,
        count_50=0,
        count_geki=0,
        count_katu=0,
        count_miss=0,
        score=100000,
        max_combo=1,
        perfect=original.perfect,
        mods=original.mods,
        life_bar_graph=[],
        timestamp=original.timestamp,
        replay_data=test_events,
        replay_id=0,
        rng_seed=12345
    )
    
    print(f"Created test replay successfully")
    
    # Try to write it
    test_replay.write_path("test_minimal.osr")
    print(f"Successfully wrote test_minimal.osr")
    
    # Verify by reading it back
    verify = osrparse.Replay.from_path("test_minimal.osr")
    print(f"Verification: {len(verify.replay_data)} events")
    print(f"File size: {Path('test_minimal.osr').stat().st_size} bytes")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()