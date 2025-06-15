"""Data loader for preprocessed numpy replay files.

This module provides tools for loading preprocessed replay data stored as .npy files.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NumpyReplay:
    """Preprocessed replay data loaded from numpy files."""
    replay_hash: str
    data: np.ndarray  # The loaded numpy array
    metadata: Optional[Dict[str, Any]] = None  # Metadata from CSV if available
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the replay data array."""
        return self.data.shape
    
    @property
    def duration_frames(self) -> int:
        """Number of frames in the replay."""
        return self.data.shape[0] if len(self.data.shape) > 0 else 0
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the replay data."""
        if self.data.size == 0:
            return {}
        
        stats = {
            'shape': self.shape,
            'duration_frames': self.duration_frames,
            'data_type': str(self.data.dtype),
            'memory_size_mb': self.data.nbytes / (1024 * 1024),
            'min_value': float(np.min(self.data)),
            'max_value': float(np.max(self.data)),
            'mean_value': float(np.mean(self.data)),
            'std_value': float(np.std(self.data)),
        }
        
        # Add metadata if available
        if self.metadata:
            stats.update({
                'player_name': self.metadata.get('playerName', 'Unknown'),
                'accuracy': self.metadata.get('performance-Accuracy', 0),
                'score': self.metadata.get('performance-Score', 0),
                'beatmap_title': self.metadata.get('beatmap-Title', 'Unknown'),
                'beatmap_artist': self.metadata.get('beatmap-Artist', 'Unknown'),
                'mods': self.metadata.get('mods', 0),
            })
        
        return stats


class NumpyReplayLoader:
    """Loader for preprocessed numpy replay files."""
    
    def __init__(self, replay_dir: str, csv_path: Optional[str] = None):
        """
        Initialize the numpy replay loader.
        
        Args:
            replay_dir: Directory containing .npy replay files
            csv_path: Optional path to CSV file with metadata
        """
        self.replay_dir = Path(replay_dir)
        self.csv_path = Path(csv_path) if csv_path else None
        self.metadata_df = None
        
        # Load metadata if CSV is provided
        if self.csv_path and self.csv_path.exists():
            try:
                self.metadata_df = pd.read_csv(self.csv_path)
                print(f"Loaded metadata for {len(self.metadata_df)} replays")
            except Exception as e:
                print(f"Warning: Could not load CSV metadata: {e}")
    
    def load_replay(self, replay_hash: str) -> Optional[NumpyReplay]:
        """
        Load a single replay by hash.
        
        Args:
            replay_hash: Hash identifier for the replay
            
        Returns:
            NumpyReplay object or None if not found
        """
        npy_path = self.replay_dir / f"{replay_hash}.npy"
        
        if not npy_path.exists():
            return None
        
        try:
            # Load numpy data
            data = np.load(npy_path)
            
            # Get metadata if available
            metadata = None
            if self.metadata_df is not None:
                matching_rows = self.metadata_df[self.metadata_df['replayHash'] == replay_hash]
                if not matching_rows.empty:
                    metadata = matching_rows.iloc[0].to_dict()
            
            return NumpyReplay(
                replay_hash=replay_hash,
                data=data,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error loading replay {replay_hash}: {e}")
            return None
    
    def list_available_replays(self) -> List[str]:
        """
        List all available replay hashes.
        
        Returns:
            List of replay hash strings
        """
        if not self.replay_dir.exists():
            return []
        
        npy_files = list(self.replay_dir.glob("*.npy"))
        return [f.stem for f in npy_files]
    
    def load_batch(self, replay_hashes: List[str]) -> List[NumpyReplay]:
        """
        Load multiple replays in batch.
        
        Args:
            replay_hashes: List of replay hash identifiers
            
        Returns:
            List of successfully loaded NumpyReplay objects
        """
        replays = []
        
        for i, hash_id in enumerate(replay_hashes):
            if i % 100 == 0 and i > 0:
                print(f"Loaded {i}/{len(replay_hashes)} replays")
            
            replay = self.load_replay(hash_id)
            if replay:
                replays.append(replay)
        
        print(f"Successfully loaded {len(replays)}/{len(replay_hashes)} replays")
        return replays
    
    def load_random_sample(self, n_samples: int = 10) -> List[NumpyReplay]:
        """
        Load a random sample of replays.
        
        Args:
            n_samples: Number of replays to sample
            
        Returns:
            List of randomly sampled NumpyReplay objects
        """
        available_hashes = self.list_available_replays()
        
        if not available_hashes:
            return []
        
        # Sample random hashes
        n_samples = min(n_samples, len(available_hashes))
        sampled_hashes = np.random.choice(available_hashes, size=n_samples, replace=False)
        
        return self.load_batch(sampled_hashes.tolist())
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        available_replays = self.list_available_replays()
        
        info = {
            'total_replays': len(available_replays),
            'replay_dir': str(self.replay_dir),
            'csv_path': str(self.csv_path) if self.csv_path else None,
            'has_metadata': self.metadata_df is not None,
        }
        
        if self.metadata_df is not None:
            info.update({
                'metadata_rows': len(self.metadata_df),
                'metadata_columns': list(self.metadata_df.columns),
            })
        
        # Sample a few replays to get data shape info
        if available_replays:
            sample_replays = self.load_random_sample(min(5, len(available_replays)))
            if sample_replays:
                shapes = [replay.shape for replay in sample_replays]
                dtypes = [str(replay.data.dtype) for replay in sample_replays]
                
                info.update({
                    'sample_shapes': shapes,
                    'sample_dtypes': dtypes,
                    'typical_shape': shapes[0] if shapes else None,
                    'typical_dtype': dtypes[0] if dtypes else None,
                })
        
        return info
    
    def create_features_dataframe(self, replays: List[NumpyReplay]) -> pd.DataFrame:
        """
        Create a features DataFrame from loaded replays.
        
        Args:
            replays: List of NumpyReplay objects
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        for replay in replays:
            stats = replay.get_basic_stats()
            stats['replay_hash'] = replay.replay_hash
            features_list.append(stats)
        
        return pd.DataFrame(features_list)


if __name__ == "__main__":
    # Test the numpy loader
    import sys
    
    if len(sys.argv) > 1:
        replay_dir = sys.argv[1]
        csv_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        replay_dir = "dataset/replays/npy"
        csv_path = "dataset/index.csv"
    
    print(f"Testing NumpyReplayLoader with:")
    print(f"  Replay dir: {replay_dir}")
    print(f"  CSV path: {csv_path}")
    print()
    
    loader = NumpyReplayLoader(replay_dir, csv_path)
    
    # Get dataset info
    info = loader.get_dataset_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Load a random sample
    print("Loading random sample...")
    sample_replays = loader.load_random_sample(3)
    
    if sample_replays:
        print(f"Loaded {len(sample_replays)} sample replays:")
        for replay in sample_replays:
            stats = replay.get_basic_stats()
            print(f"\nReplay {replay.replay_hash}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        # Create features DataFrame
        df = loader.create_features_dataframe(sample_replays)
        print(f"\nFeatures DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    else:
        print("No replays could be loaded")