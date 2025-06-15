#!/usr/bin/env python3
"""
Dataset Analysis Script

Analyzes the structure of the osu! dataset to understand file naming patterns
and identify issues with missing files.
"""

import pandas as pd
import os
from pathlib import Path
import sys

def analyze_index_csv(dataset_dir="og_dataset"):
    """Analyze the index.csv file structure."""
    index_path = Path(dataset_dir) / "index.csv"
    
    if not index_path.exists():
        print(f"Error: index.csv not found in {dataset_dir}")
        return None
    
    try:
        df = pd.read_csv(index_path)
        print(f"Index CSV loaded successfully with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head(3).to_string())
        
        # Look for hash columns
        hash_columns = []
        for col in df.columns:
            if 'hash' in col.lower():
                hash_columns.append(col)
        
        print(f"\nHash columns found: {hash_columns}")
        
        if hash_columns:
            for col in hash_columns:
                print(f"\nSample values from {col}:")
                print(df[col].head(5).tolist())
        
        return df
    except Exception as e:
        print(f"Error reading index.csv: {e}")
        return None

def check_file_existence(dataset_dir="og_dataset", sample_size=10):
    """Check if files actually exist for the first few entries."""
    df = analyze_index_csv(dataset_dir)
    if df is None:
        return
    
    dataset_path = Path(dataset_dir)
    
    print(f"\n=== Checking file existence for first {sample_size} entries ===")
    
    for idx in range(min(sample_size, len(df))):
        row = df.iloc[idx]
        print(f"\nEntry {idx}:")
        
        # Check for beatmap hash
        beatmap_hash = None
        if 'beatmapHash' in df.columns:
            beatmap_hash = str(row['beatmapHash'])
        elif 'beatmap-Id' in df.columns:
            beatmap_hash = str(row['beatmap-Id'])
        
        if beatmap_hash:
            beatmap_file = dataset_path / "beatmaps" / f"{beatmap_hash}.osu"
            print(f"  Beatmap ({beatmap_hash}.osu): {'EXISTS' if beatmap_file.exists() else 'MISSING'}")
        
        # Check for replay hash
        replay_hash = None
        if 'replayHash' in df.columns:
            replay_hash = str(row['replayHash'])
        
        if replay_hash:
            npy_file = dataset_path / "replays" / "npy" / f"{replay_hash}.npy"
            osr_file = dataset_path / "replays" / "osr" / f"{replay_hash}.osr"
            print(f"  NPY replay ({replay_hash}.npy): {'EXISTS' if npy_file.exists() else 'MISSING'}")
            print(f"  OSR replay ({replay_hash}.osr): {'EXISTS' if osr_file.exists() else 'MISSING'}")
        
        # Try index-based naming
        idx_npy = dataset_path / "replays" / "npy" / f"{idx}.npy"
        idx_osr = dataset_path / "replays" / "osr" / f"{idx}.osr"
        print(f"  NPY replay ({idx}.npy): {'EXISTS' if idx_npy.exists() else 'MISSING'}")
        print(f"  OSR replay ({idx}.osr): {'EXISTS' if idx_osr.exists() else 'MISSING'}")
        
        # Try replay_ prefix
        replay_npy = dataset_path / "replays" / "npy" / f"replay_{idx}.npy"
        replay_osr = dataset_path / "replays" / "osr" / f"replay_{idx}.osr"
        print(f"  NPY replay (replay_{idx}.npy): {'EXISTS' if replay_npy.exists() else 'MISSING'}")
        print(f"  OSR replay (replay_{idx}.osr): {'EXISTS' if replay_osr.exists() else 'MISSING'}")

def list_actual_files(dataset_dir="og_dataset", limit=10):
    """List actual files in the dataset directories."""
    dataset_path = Path(dataset_dir)
    
    print(f"\n=== Actual files in {dataset_dir} ===")
    
    # List beatmap files
    beatmap_dir = dataset_path / "beatmaps"
    if beatmap_dir.exists():
        beatmap_files = list(beatmap_dir.glob("*.osu"))[:limit]
        print(f"\nBeatmap files (first {limit}):")
        for f in beatmap_files:
            print(f"  {f.name}")
    
    # List NPY replay files
    npy_dir = dataset_path / "replays" / "npy"
    if npy_dir.exists():
        npy_files = list(npy_dir.glob("*.npy"))[:limit]
        print(f"\nNPY replay files (first {limit}):")
        for f in npy_files:
            print(f"  {f.name}")
    
    # List OSR replay files
    osr_dir = dataset_path / "replays" / "osr"
    if osr_dir.exists():
        osr_files = list(osr_dir.glob("*.osr"))[:limit]
        print(f"\nOSR replay files (first {limit}):")
        for f in osr_files:
            print(f"  {f.name}")

def main():
    """Main function."""
    dataset_dir = "og_dataset"
    
    if not Path(dataset_dir).exists():
        print(f"Error: Dataset directory '{dataset_dir}' does not exist")
        sys.exit(1)
    
    print(f"Analyzing dataset: {dataset_dir}")
    print("=" * 50)
    
    # Analyze index.csv
    analyze_index_csv(dataset_dir)
    
    # Check file existence
    check_file_existence(dataset_dir, sample_size=5)
    
    # List actual files
    list_actual_files(dataset_dir, limit=10)

if __name__ == "__main__":
    main()