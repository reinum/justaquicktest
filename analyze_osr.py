#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import struct

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

import osrparse
import numpy as np
from src.data.beatmap_parser import BeatmapParser

def analyze_osr_file(osr_path):
    """Analyze an OSR file using osrparse."""
    print(f"Analyzing {osr_path}...")
    print(f"File size: {Path(osr_path).stat().st_size} bytes")
    
    try:
        # Parse the replay
        replay = osrparse.Replay.from_path(osr_path)
        
        print("\n=== Replay Information ===")
        print(f"Mode: {replay.mode}")
        print(f"Game Version: {replay.game_version}")
        print(f"Beatmap Hash: {replay.beatmap_hash}")
        print(f"Username: {replay.username}")
        print(f"Replay Hash: {replay.replay_hash}")
        print(f"Score: {replay.score}")
        print(f"Max Combo: {replay.max_combo}")
        print(f"Perfect: {replay.perfect}")
        print(f"Mods: {replay.mods}")
        print(f"Replay ID: {replay.replay_id}")
        print(f"RNG Seed: {replay.rng_seed}")
        
        print("\n=== Hit Statistics ===")
        print(f"300s: {replay.count_300}")
        print(f"100s: {replay.count_100}")
        print(f"50s: {replay.count_50}")
        print(f"Gekis: {replay.count_geki}")
        print(f"Katus: {replay.count_katu}")
        print(f"Misses: {replay.count_miss}")
        
        print("\n=== Replay Data ===")
        print(f"Number of frames: {len(replay.replay_data)}")
        
        if len(replay.replay_data) > 0:
            print("\nFirst 5 frames:")
            for i, frame in enumerate(replay.replay_data[:5]):
                print(f"  Frame {i}: time_delta={frame.time_delta}, x={frame.x}, y={frame.y}, keys={frame.keys}")
            
            print("\nLast 5 frames:")
            for i, frame in enumerate(replay.replay_data[-5:]):
                print(f"  Frame {len(replay.replay_data)-5+i}: time_delta={frame.time_delta}, x={frame.x}, y={frame.y}, keys={frame.keys}")
        
        print("\n=== Life Bar ===")
        if replay.life_bar_graph:
            print(f"Life bar states: {len(replay.life_bar_graph)}")
            if len(replay.life_bar_graph) > 0:
                print("First 3 life bar states:")
                for i, state in enumerate(replay.life_bar_graph[:3]):
                    print(f"  State {i}: time={state.time}, life={state.life}")
        else:
            print("No life bar data")
        
        return replay
        
    except Exception as e:
        print(f"Error analyzing replay with osrparse: {e}")
        print("Attempting raw binary analysis...")
        return analyze_raw_osr(osr_path)

def analyze_raw_osr(osr_path):
    """Analyze OSR file at binary level."""
    try:
        with open(osr_path, 'rb') as f:
            data = f.read()
        
        print(f"\nRaw binary analysis of {osr_path}:")
        print(f"Total file size: {len(data)} bytes")
        
        # Try to read the header manually
        offset = 0
        
        # Game mode (1 byte)
        mode = struct.unpack('<B', data[offset:offset+1])[0]
        offset += 1
        print(f"Mode: {mode}")
        
        # Game version (4 bytes)
        version = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        print(f"Game version: {version}")
        
        # Beatmap MD5 hash (string)
        hash_len = data[offset]
        offset += 1
        if hash_len == 11:  # 0x0b prefix for strings
            hash_len = data[offset]
            offset += 1
        beatmap_hash = data[offset:offset+hash_len].decode('utf-8')
        offset += hash_len
        print(f"Beatmap hash: {beatmap_hash}")
        
        print(f"Processed {offset} bytes of header so far")
        print(f"Remaining bytes: {len(data) - offset}")
        
        # Show hex dump of first 100 bytes
        print("\nFirst 100 bytes (hex):")
        for i in range(0, min(100, len(data)), 16):
            hex_part = ' '.join(f'{b:02x}' for b in data[i:i+16])
            ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data[i:i+16])
            print(f"{i:04x}: {hex_part:<48} {ascii_part}")
        
        return None
        
    except Exception as e:
        print(f"Error in raw analysis: {e}")
        return None

def compare_replays():
    """Compare the original ogreplay.osr with our generated replay."""
    print("\n" + "="*60)
    print("COMPARING REPLAYS")
    print("="*60)
    
    # Analyze original replay
    print("\n### ORIGINAL REPLAY (ogreplay.osr) ###")
    original = analyze_osr_file("ogreplay.osr")
    
    # Analyze our generated replay
    print("\n### GENERATED REPLAY (ai_replay_final.osr) ###")
    generated = analyze_osr_file("ai_replay_final.osr")
    
    if original and generated:
        print("\n=== COMPARISON ===")
        print(f"Original frames: {len(original.replay_data)}")
        print(f"Generated frames: {len(generated.replay_data)}")
        print(f"Frame count ratio: {len(generated.replay_data) / len(original.replay_data):.2f}")
        
        print(f"\nOriginal beatmap hash: {original.beatmap_hash}")
        print(f"Generated beatmap hash: {generated.beatmap_hash}")
        print(f"Beatmap hash match: {original.beatmap_hash == generated.beatmap_hash}")
        
        print(f"\nOriginal RNG seed: {original.rng_seed}")
        print(f"Generated RNG seed: {generated.rng_seed}")
        
        # Check if life bar data exists
        orig_life = len(original.life_bar_graph) if original.life_bar_graph else 0
        gen_life = len(generated.life_bar_graph) if generated.life_bar_graph else 0
        print(f"\nOriginal life bar states: {orig_life}")
        print(f"Generated life bar states: {gen_life}")
    elif original:
        print("\n=== ANALYSIS ===")
        print("Original replay parsed successfully, but generated replay failed to parse.")
        print("This suggests our OSR generation has structural issues.")

def find_beatmap_for_replay(replay):
    """Find the beatmap file for a given replay."""
    if not replay:
        return None
        
    beatmap_hash = replay.beatmap_hash
    print(f"\nLooking for beatmap with hash: {beatmap_hash}")
    
    # Check in reduced datasets
    for dataset_dir in ["reduced_dataset_5", "reduced_dataset_1000"]:
        beatmap_dir = Path(dataset_dir) / "beatmaps"
        if beatmap_dir.exists():
            for beatmap_file in beatmap_dir.glob("*.osu"):
                try:
                    parser = BeatmapParser()
                    beatmap_data = parser.parse_beatmap(str(beatmap_file))
                    if beatmap_data.get('md5_hash') == beatmap_hash:
                        print(f"Found matching beatmap: {beatmap_file}")
                        return str(beatmap_file)
                except Exception as e:
                    continue
    
    print("No matching beatmap found in datasets")
    return None

if __name__ == "__main__":
    original_file = "ogreplay.osr"
    generated_file = "ai_replay_final.osr"
    
    compare_replays()
    
    # Try to find the beatmap for the original replay
    print("\n" + "="*60)
    print("BEATMAP SEARCH")
    print("="*60)
    
    try:
        original = osrparse.Replay.from_path("ogreplay.osr")
        beatmap_path = find_beatmap_for_replay(original)
        if beatmap_path:
            print(f"\nBeatmap found: {beatmap_path}")
            print("This beatmap can be used for generating a proper replay.")
    except Exception as e:
        print(f"Error finding beatmap: {e}")