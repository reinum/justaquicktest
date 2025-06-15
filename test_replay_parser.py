#!/usr/bin/env python3
"""
Test script for the enhanced replay parser using osrparse.

This script tests the ReplayDataLoader with both osrparse integration
and the existing C# beatmap parser for slider calculations.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import ReplayDataLoader, ProcessedReplay


def test_single_replay(replay_path: str, beatmap_dir: str = None):
    """Test parsing a single replay file."""
    print(f"Testing single replay: {replay_path}")
    print("-" * 50)
    
    if not os.path.exists(replay_path):
        print(f"Error: Replay file not found: {replay_path}")
        return False
    
    loader = ReplayDataLoader(beatmap_dir=beatmap_dir)
    replay = loader.parse_replay_file(replay_path)
    
    if not replay:
        print("Failed to parse replay file")
        return False
    
    print(f"Player: {replay.player_name}")
    print(f"Beatmap Hash: {replay.beatmap_hash}")
    print(f"Accuracy: {replay.accuracy:.2f}%")
    print(f"Score: {replay.total_score:,}")
    print(f"Max Combo: {replay.max_combo}")
    print(f"Hit Counts: {replay.count_300}/{replay.count_100}/{replay.count_50}/{replay.count_miss}")
    print(f"Mods: {replay.mods}")
    print(f"Timestamp: {replay.timestamp}")
    print(f"Total Frames: {len(replay.cursor_positions)}")
    print(f"Duration: {replay.duration_ms}ms")
    
    if len(replay.cursor_positions) > 0:
        print(f"Cursor Range: X={replay.cursor_positions[:, 0].min():.1f}-{replay.cursor_positions[:, 0].max():.1f}, "
              f"Y={replay.cursor_positions[:, 1].min():.1f}-{replay.cursor_positions[:, 1].max():.1f}")
    
    if len(replay.speeds) > 0:
        print(f"Speed Stats: Avg={replay.speeds.mean():.1f}, Max={replay.speeds.max():.1f}, Std={replay.speeds.std():.1f}")
    
    if replay.beatmap_objects:
        print(f"Beatmap Objects: {len(replay.beatmap_objects)} objects loaded")
    else:
        print("Beatmap Objects: Not available")
    
    # Test feature extraction
    features = replay.get_feature_dict()
    print("\nExtracted Features:")
    for key, value in sorted(features.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    return True


def test_csv_loading(csv_path: str, replay_dir: str, beatmap_dir: str = None, max_replays: int = 5):
    """Test loading replays from CSV index."""
    print(f"\nTesting CSV loading: {csv_path}")
    print("-" * 50)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return False
    
    if not os.path.exists(replay_dir):
        print(f"Error: Replay directory not found: {replay_dir}")
        return False
    
    loader = ReplayDataLoader(beatmap_dir=beatmap_dir)
    replays = loader.load_dataset_from_csv(csv_path, replay_dir, max_replays=max_replays)
    
    if not replays:
        print("No replays loaded successfully")
        return False
    
    print(f"Successfully loaded {len(replays)} replays")
    
    # Test feature extraction to DataFrame
    df = loader.extract_features_dataframe(replays)
    print(f"Features DataFrame shape: {df.shape}")
    print(f"Feature columns: {list(df.columns)}")
    
    # Show summary statistics
    print("\nSummary Statistics:")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(df[numeric_cols].describe())
    
    return True


def main():
    """Main test function."""
    print("Osu! AI Replay Maker - Enhanced Replay Parser Test")
    print("=" * 60)
    
    # Test configuration
    base_dir = Path(__file__).parent
    csv_path = base_dir / "index.csv"
    replay_dir = base_dir / "replays"  # Adjust this path as needed
    beatmap_dir = base_dir / "beatmaps"  # Adjust this path as needed
    osuparse_dir = base_dir / "osuparse"
    
    print(f"Base directory: {base_dir}")
    print(f"CSV path: {csv_path}")
    print(f"Replay directory: {replay_dir}")
    print(f"Beatmap directory: {beatmap_dir}")
    print(f"OsuParse directory: {osuparse_dir}")
    
    # Check if osrparse is available
    try:
        import osrparse
        print(f"✓ osrparse version: {osrparse.__version__}")
    except ImportError:
        print("✗ osrparse not available - install with: pip install osrparse")
        return
    
    # Check if C# parser is available
    if osuparse_dir.exists():
        print("✓ C# beatmap parser directory found")
    else:
        print("✗ C# beatmap parser directory not found")
    
    print()
    
    # Test 1: Single replay file (if available)
    if len(sys.argv) > 1:
        replay_file = sys.argv[1]
        beatmap_dir_arg = sys.argv[2] if len(sys.argv) > 2 else str(beatmap_dir) if beatmap_dir.exists() else None
        success = test_single_replay(replay_file, beatmap_dir_arg)
        if not success:
            return
    
    # Test 2: CSV loading (if CSV exists)
    if csv_path.exists():
        replay_dir_str = str(replay_dir) if replay_dir.exists() else str(base_dir)
        beatmap_dir_str = str(beatmap_dir) if beatmap_dir.exists() else None
        success = test_csv_loading(str(csv_path), replay_dir_str, beatmap_dir_str, max_replays=3)
        if not success:
            print("CSV loading test failed, but this might be expected if replay files are not available")
    else:
        print(f"CSV file not found at {csv_path}, skipping CSV loading test")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("\nNext steps:")
    print("1. Install osrparse: pip install osrparse")
    print("2. Place some .osr files in the replays directory")
    print("3. Optionally place .osu beatmap files in the beatmaps directory")
    print("4. Run: python test_replay_parser.py <replay_file.osr> [beatmap_directory]")


if __name__ == "__main__":
    main()