#!/usr/bin/env python3

import osrparse
import numpy as np
from pathlib import Path

def test_replay_file(osr_path):
    """Test that an OSR file is valid and can be loaded properly."""
    print(f"Testing replay file: {osr_path}")
    
    if not Path(osr_path).exists():
        print(f"❌ File does not exist: {osr_path}")
        return False
    
    file_size = Path(osr_path).stat().st_size
    print(f"📁 File size: {file_size} bytes")
    
    try:
        # Load the replay
        replay = osrparse.Replay.from_path(osr_path)
        print(f"✅ Successfully loaded replay")
        
        # Basic validation
        print(f"🎮 Game mode: {replay.mode}")
        print(f"🗺️  Beatmap hash: {replay.beatmap_hash}")
        print(f"👤 Player: {replay.username}")
        print(f"🎯 Score: {replay.score:,}")
        print(f"🔥 Max combo: {replay.max_combo}")
        print(f"📊 Hit counts: {replay.count_300}/{replay.count_100}/{replay.count_50}/{replay.count_miss}")
        print(f"🎬 Replay frames: {len(replay.replay_data)}")
        
        # Validate replay data
        if len(replay.replay_data) == 0:
            print(f"⚠️  Warning: No replay data found")
            return False
        
        # Check first and last frames
        first_frame = replay.replay_data[0]
        last_frame = replay.replay_data[-1]
        
        print(f"🎬 First frame: t={first_frame.time_delta}, pos=({first_frame.x:.1f}, {first_frame.y:.1f}), keys={first_frame.keys}")
        print(f"🎬 Last frame: t={last_frame.time_delta}, pos=({last_frame.x:.1f}, {last_frame.y:.1f}), keys={last_frame.keys}")
        
        # Calculate total replay duration
        total_time = sum(frame.time_delta for frame in replay.replay_data)
        print(f"⏱️  Total duration: {total_time/1000:.2f} seconds")
        
        # Validate cursor positions are reasonable
        positions = [(frame.x, frame.y) for frame in replay.replay_data]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_range = (min(x_coords), max(x_coords))
        y_range = (min(y_coords), max(y_coords))
        
        print(f"📍 Cursor range: X={x_range[0]:.1f}-{x_range[1]:.1f}, Y={y_range[0]:.1f}-{y_range[1]:.1f}")
        
        # Check if positions are within osu! playfield (roughly 0-512 x 0-384)
        valid_positions = all(0 <= x <= 512 and 0 <= y <= 384 for x, y in positions)
        if valid_positions:
            print(f"✅ All cursor positions are within valid playfield")
        else:
            print(f"⚠️  Some cursor positions are outside normal playfield")
        
        # Count key press events
        key_events = sum(1 for frame in replay.replay_data if frame.keys.value > 0)
        print(f"🔘 Frames with key presses: {key_events}/{len(replay.replay_data)} ({key_events/len(replay.replay_data)*100:.1f}%)")
        
        print(f"✅ Replay file appears to be valid and ready for osu!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load replay: {e}")
        return False

def compare_with_original(original_path, generated_path):
    """Compare generated replay with original."""
    print(f"\n🔍 Comparing replays...")
    
    try:
        original = osrparse.Replay.from_path(original_path)
        generated = osrparse.Replay.from_path(generated_path)
        
        print(f"📊 Frame count comparison:")
        print(f"   Original: {len(original.replay_data)} frames")
        print(f"   Generated: {len(generated.replay_data)} frames")
        print(f"   Ratio: {len(generated.replay_data)/len(original.replay_data):.2f}")
        
        print(f"🎯 Score comparison:")
        print(f"   Original: {original.score:,}")
        print(f"   Generated: {generated.score:,}")
        print(f"   Ratio: {generated.score/original.score:.2f}")
        
        print(f"🔥 Combo comparison:")
        print(f"   Original: {original.max_combo}")
        print(f"   Generated: {generated.max_combo}")
        
        print(f"🎮 Metadata match:")
        print(f"   Same beatmap: {original.beatmap_hash == generated.beatmap_hash}")
        print(f"   Same mode: {original.mode == generated.mode}")
        print(f"   Same mods: {original.mods == generated.mods}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

if __name__ == "__main__":
    print("🎮 OSU! REPLAY VALIDATION TEST")
    print("=" * 50)
    
    # Test our generated replay
    print("\n🤖 Testing AI-generated replay...")
    ai_valid = test_replay_file("ai_replay_final.osr")
    
    # Test original replay for comparison
    print("\n👤 Testing original replay...")
    original_valid = test_replay_file("ogreplay.osr")
    
    # Compare if both are valid
    if ai_valid and original_valid:
        compare_with_original("ogreplay.osr", "ai_replay_final.osr")
    
    print("\n" + "=" * 50)
    if ai_valid:
        print("🎉 SUCCESS: AI replay is valid and ready for osu!")
        print("📁 File: ai_replay_final.osr")
        print("💡 You can now import this replay into osu! and watch the AI play!")
    else:
        print("❌ FAILED: AI replay has issues")