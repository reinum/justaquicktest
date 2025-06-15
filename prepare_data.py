#!/usr/bin/env python3
"""
Data preparation script for osu! AI Replay Maker

This script helps prepare your dataset by:
1. Scanning for .osu and .osr files
2. Creating the index.csv file
3. Converting replays to numpy format for faster loading
4. Validating the dataset structure

Usage:
    python prepare_data.py                    # Scan and prepare dataset
    python prepare_data.py --validate-only    # Only validate existing dataset
    python prepare_data.py --force-rebuild    # Force rebuild index and numpy files
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.beatmap_parser import BeatmapParser
from src.data.replay_parser import ReplayParser
from src.utils.dataset_validator import DatasetValidator


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_preparation.log')
        ]
    )


def scan_beatmaps(beatmap_dir: str) -> List[Dict]:
    """Scan beatmap directory and extract metadata."""
    beatmap_files = list(Path(beatmap_dir).glob('*.osu'))
    beatmaps = []
    
    print(f"🗺️  Scanning {len(beatmap_files)} beatmap files...")
    
    parser = BeatmapParser()
    if not parser.is_available():
        print("❌ Beatmap parser not available. Please check OsuParsers.dll")
        return []
    
    for beatmap_file in tqdm(beatmap_files, desc="Processing beatmaps"):
        try:
            # Parse beatmap
            if parser.validate_beatmap(str(beatmap_file)):
                hit_objects = parser.get_hit_objects(str(beatmap_file))
                timing_points = parser.get_timing_points(str(beatmap_file))
                
                # Extract basic metadata from filename or file content
                beatmap_id = beatmap_file.stem
                
                beatmap_info = {
                    'beatmap_id': beatmap_id,
                    'beatmap_path': str(beatmap_file.relative_to(Path(beatmap_dir).parent)),
                    'hit_objects_count': len(hit_objects) if hit_objects else 0,
                    'timing_points_count': len(timing_points) if timing_points else 0,
                    'file_size': beatmap_file.stat().st_size,
                    'valid': True
                }
                
                beatmaps.append(beatmap_info)
            else:
                print(f"⚠️  Invalid beatmap: {beatmap_file.name}")
                
        except Exception as e:
            print(f"❌ Error processing {beatmap_file.name}: {e}")
            continue
    
    print(f"✅ Successfully processed {len(beatmaps)} beatmaps")
    return beatmaps


def scan_replays(replay_dir: str) -> List[Dict]:
    """Scan replay directory and extract metadata."""
    replay_files = list(Path(replay_dir).glob('*.osr'))
    replays = []
    
    print(f"🎮 Scanning {len(replay_files)} replay files...")
    
    parser = ReplayParser()
    
    for replay_file in tqdm(replay_files, desc="Processing replays"):
        try:
            # Parse replay metadata
            replay_data = parser.parse_replay(str(replay_file))
            
            if replay_data:
                replay_info = {
                    'replay_id': replay_file.stem,
                    'replay_path': str(replay_file.relative_to(Path(replay_dir).parent)),
                    'beatmap_hash': getattr(replay_data, 'beatmap_hash', ''),
                    'player_name': getattr(replay_data, 'player_name', ''),
                    'score': getattr(replay_data, 'score', 0),
                    'accuracy': getattr(replay_data, 'accuracy', 0.0),
                    'max_combo': getattr(replay_data, 'max_combo', 0),
                    'count_300': getattr(replay_data, 'count_300', 0),
                    'count_100': getattr(replay_data, 'count_100', 0),
                    'count_50': getattr(replay_data, 'count_50', 0),
                    'count_miss': getattr(replay_data, 'count_miss', 0),
                    'mods': getattr(replay_data, 'mods', 0),
                    'file_size': replay_file.stat().st_size,
                    'valid': True
                }
                
                replays.append(replay_info)
            else:
                print(f"⚠️  Could not parse replay: {replay_file.name}")
                
        except Exception as e:
            print(f"❌ Error processing {replay_file.name}: {e}")
            continue
    
    print(f"✅ Successfully processed {len(replays)} replays")
    return replays


def create_index_csv(beatmaps: List[Dict], replays: List[Dict], output_path: str):
    """Create index CSV file matching replays to beatmaps."""
    print("📊 Creating index CSV...")
    
    # Create beatmap hash lookup
    beatmap_lookup = {}
    for beatmap in beatmaps:
        # You might need to calculate hash from beatmap content
        # For now, use beatmap_id as a simple lookup
        beatmap_lookup[beatmap['beatmap_id']] = beatmap
    
    # Match replays to beatmaps
    matched_data = []
    unmatched_count = 0
    
    for replay in replays:
        # Try to find matching beatmap
        beatmap_hash = replay.get('beatmap_hash', '')
        
        # Simple matching by hash or ID
        matched_beatmap = None
        for beatmap in beatmaps:
            if beatmap['beatmap_id'] == beatmap_hash or beatmap_hash in beatmap['beatmap_id']:
                matched_beatmap = beatmap
                break
        
        if matched_beatmap:
            combined_data = {
                **replay,
                'beatmap_id': matched_beatmap['beatmap_id'],
                'beatmap_path': matched_beatmap['beatmap_path'],
                'hit_objects_count': matched_beatmap['hit_objects_count'],
                'star_rating': 0.0,  # You might want to calculate this
                'difficulty': 'Unknown',
                'length_seconds': 0,  # You might want to calculate this
            }
            matched_data.append(combined_data)
        else:
            unmatched_count += 1
    
    # Create DataFrame and save
    df = pd.DataFrame(matched_data)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created index with {len(matched_data)} entries")
    print(f"⚠️  {unmatched_count} replays could not be matched to beatmaps")
    
    return df


def convert_replays_to_numpy(replays_df: pd.DataFrame, output_dir: str):
    """Convert replay files to numpy format for faster loading."""
    print("🔄 Converting replays to numpy format...")
    
    os.makedirs(output_dir, exist_ok=True)
    parser = ReplayParser()
    
    converted_count = 0
    
    for _, row in tqdm(replays_df.iterrows(), total=len(replays_df), desc="Converting replays"):
        try:
            replay_path = Path('dataset') / row['replay_path']
            output_path = Path(output_dir) / f"{row['replay_id']}.npy"
            
            # Skip if already exists
            if output_path.exists():
                continue
            
            # Parse and convert replay
            replay_data = parser.parse_replay(str(replay_path))
            if replay_data and hasattr(replay_data, 'replay_data'):
                # Convert to numpy array
                # This is a simplified conversion - you might need to adjust based on your data structure
                numpy_data = np.array(replay_data.replay_data, dtype=np.float32)
                np.save(output_path, numpy_data)
                converted_count += 1
                
        except Exception as e:
            print(f"❌ Error converting {row['replay_id']}: {e}")
            continue
    
    print(f"✅ Converted {converted_count} replays to numpy format")


def validate_dataset(dataset_dir: str) -> bool:
    """Validate the dataset structure and integrity."""
    print("🔍 Validating dataset...")
    
    validator = DatasetValidator()
    
    # Check directory structure
    required_dirs = ['beatmaps', 'replays']
    required_files = ['index.csv']
    
    issues = []
    
    for dir_name in required_dirs:
        dir_path = Path(dataset_dir) / dir_name
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_path}")
        elif not any(dir_path.iterdir()):
            issues.append(f"Empty directory: {dir_path}")
    
    for file_name in required_files:
        file_path = Path(dataset_dir) / file_name
        if not file_path.exists():
            issues.append(f"Missing file: {file_path}")
    
    if issues:
        print("❌ Dataset validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    # Validate index CSV
    try:
        df = pd.read_csv(Path(dataset_dir) / 'index.csv')
        print(f"📊 Index contains {len(df)} entries")
        
        # Check for required columns
        required_columns = ['replay_id', 'beatmap_id', 'replay_path', 'beatmap_path']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns in index.csv: {missing_columns}")
            return False
        
        print("✅ Dataset validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating index.csv: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Prepare osu! AI dataset')
    parser.add_argument('--dataset-dir', default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing dataset')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild index and numpy files')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("🎮 osu! AI Replay Maker - Data Preparation")
    print("=" * 45)
    
    dataset_dir = Path(args.dataset_dir)
    
    # Validate only mode
    if args.validate_only:
        success = validate_dataset(str(dataset_dir))
        return 0 if success else 1
    
    # Check if dataset directory exists
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("Please create the dataset directory and add your .osu and .osr files")
        return 1
    
    beatmap_dir = dataset_dir / 'beatmaps'
    replay_dir = dataset_dir / 'replays'
    index_path = dataset_dir / 'index.csv'
    numpy_dir = dataset_dir / 'replays' / 'npy'
    
    # Check if we need to rebuild
    if index_path.exists() and not args.force_rebuild:
        print("📋 Index file already exists. Use --force-rebuild to recreate.")
        df = pd.read_csv(index_path)
    else:
        # Scan beatmaps and replays
        beatmaps = scan_beatmaps(str(beatmap_dir))
        replays = scan_replays(str(replay_dir))
        
        if not beatmaps:
            print("❌ No valid beatmaps found")
            return 1
        
        if not replays:
            print("❌ No valid replays found")
            return 1
        
        # Create index
        df = create_index_csv(beatmaps, replays, str(index_path))
    
    # Convert replays to numpy format
    if not numpy_dir.exists() or args.force_rebuild:
        convert_replays_to_numpy(df, str(numpy_dir))
    else:
        print("🔄 Numpy replay files already exist. Use --force-rebuild to recreate.")
    
    # Final validation
    success = validate_dataset(str(dataset_dir))
    
    if success:
        print("\n🎉 Dataset preparation completed successfully!")
        print(f"📊 Ready to train with {len(df)} replay-beatmap pairs")
        print("\n💡 Next steps:")
        print("  1. Review the generated index.csv file")
        print("  2. Adjust config/default.yaml if needed")
        print("  3. Run: python train.py")
    else:
        print("\n❌ Dataset preparation completed with issues")
        print("Please fix the issues before training")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())