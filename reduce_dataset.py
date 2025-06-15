#!/usr/bin/env python3
"""
Dataset Reduction Script

This script reduces the size of the osu! replay dataset by copying only a specified
number of replays and their corresponding files to a new directory.

Usage:
    python reduce_dataset.py --replays 2500
"""

import argparse
import os
import shutil
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reduce dataset size by copying specified number of replays"
    )
    parser.add_argument(
        "--replays",
        type=int,
        required=True,
        help="Number of replays to keep in the reduced dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="og_dataset",
        help="Input dataset directory (default: og_dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory name (default: reduced_dataset_N where N is replay count)"
    )
    return parser.parse_args()

def create_output_directory(output_dir):
    """Create output directory structure."""
    output_path = Path(output_dir)
    
    # Create main directories
    output_path.mkdir(exist_ok=True)
    (output_path / "beatmaps").mkdir(exist_ok=True)
    (output_path / "replays").mkdir(exist_ok=True)
    (output_path / "replays" / "npy").mkdir(exist_ok=True)
    (output_path / "replays" / "osr").mkdir(exist_ok=True)
    
    return output_path

def load_index_csv(input_dir):
    """Load and return the index.csv file."""
    index_path = Path(input_dir) / "index.csv"
    
    if not index_path.exists():
        print(f"Error: index.csv not found in {input_dir}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(index_path)
        print(f"Loaded index.csv with {len(df)} entries")
        return df
    except Exception as e:
        print(f"Error reading index.csv: {e}")
        sys.exit(1)

def copy_file_if_exists(src_path, dst_path, file_type="file"):
    """Copy a file if it exists, with error handling."""
    try:
        if src_path.exists():
            # Create parent directory if it doesn't exist
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return True
        else:
            # Only show warning for missing files if we're in verbose mode
            # Comment out to reduce noise: print(f"Warning: {file_type} file not found: {src_path.name}")
            return False
    except Exception as e:
        print(f"Error copying {file_type}: {e}")
        return False

def reduce_dataset(input_dir, output_dir, num_replays):
    """Reduce the dataset to the specified number of replays."""
    input_path = Path(input_dir)
    output_path = create_output_directory(output_dir)
    
    # Load index.csv
    df = load_index_csv(input_dir)
    
    # Handle the requested number of replays
    if num_replays > len(df):
        print(f"Warning: Requested {num_replays} replays, but only {len(df)} available. Using all {len(df)} replays.")
        num_replays = len(df)
    
    # Take the first N replays
    reduced_df = df.head(num_replays)
    
    # Initialize counters
    copied_beatmaps = 0
    copied_npy_replays = 0
    copied_osr_replays = 0
    
    print(f"Copying {num_replays} replays...")
    
    for idx, row in reduced_df.iterrows():
        # Get replay hash and beatmap hash from the CSV
        replay_hash = None
        beatmap_hash = None
        
        # Get replay hash (this is the filename for replay files)
        if 'replayHash' in df.columns:
            replay_hash = str(row['replayHash'])
        elif 'replay_hash' in df.columns:
            replay_hash = str(row['replay_hash'])
        
        # Get beatmap hash (this is the filename for beatmap files)
        if 'beatmapHash' in df.columns:
            beatmap_hash = str(row['beatmapHash'])
        elif 'beatmap_hash' in df.columns:
            beatmap_hash = str(row['beatmap_hash'])
        elif 'beatmap-Id' in df.columns:
            beatmap_hash = str(row['beatmap-Id'])
        
        # Skip this entry if we can't find the required hashes
        if not replay_hash or not beatmap_hash:
            print(f"Warning: Missing hash values for entry {idx}, skipping...")
            continue
        
        # Copy beatmap file using beatmap hash
        beatmap_src = input_path / "beatmaps" / f"{beatmap_hash}.osu"
        beatmap_dst = output_path / "beatmaps" / f"{beatmap_hash}.osu"
        if copy_file_if_exists(beatmap_src, beatmap_dst, "beatmap"):
            copied_beatmaps += 1
        
        # Copy replay files using replay hash
        npy_src = input_path / "replays" / "npy" / f"{replay_hash}.npy"
        npy_dst = output_path / "replays" / "npy" / f"{replay_hash}.npy"
        if copy_file_if_exists(npy_src, npy_dst, "npy replay"):
            copied_npy_replays += 1
        
        osr_src = input_path / "replays" / "osr" / f"{replay_hash}.osr"
        osr_dst = output_path / "replays" / "osr" / f"{replay_hash}.osr"
        if copy_file_if_exists(osr_src, osr_dst, "osr replay"):
            copied_osr_replays += 1
    
    # Save reduced index.csv
    output_index_path = output_path / "index.csv"
    reduced_df.to_csv(output_index_path, index=False)
    print(f"Saved reduced index.csv with {len(reduced_df)} entries")
    
    # Print summary
    print("\n=== Copy Summary ===")
    print(f"Beatmaps copied: {copied_beatmaps}/{num_replays}")
    print(f"NPY replays copied: {copied_npy_replays}/{num_replays}")
    print(f"OSR replays copied: {copied_osr_replays}/{num_replays}")
    print(f"\nReduced dataset saved to: {output_path}")
    
    return output_path

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set default output directory name
    if args.output_dir is None:
        args.output_dir = f"reduced_dataset_{args.replays}"
    
    # Check if input directory exists
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    # Check if output directory already exists
    if Path(args.output_dir).exists():
        response = input(f"Output directory '{args.output_dir}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled")
            sys.exit(0)
        shutil.rmtree(args.output_dir)
    
    print(f"Reducing dataset from '{args.input_dir}' to '{args.output_dir}'")
    print(f"Target replay count: {args.replays}")
    
    try:
        output_path = reduce_dataset(args.input_dir, args.output_dir, args.replays)
        print(f"\nDataset reduction completed successfully!")
        print(f"Reduced dataset location: {output_path.absolute()}")
    except Exception as e:
        print(f"Error during dataset reduction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()