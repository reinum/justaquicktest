import os
import sys
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Import osrparse for replay parsing
try:
    from osrparse import Replay, ReplayEventOsu
except ImportError:
    print("osrparse not installed. Install with: pip install osrparse")
    sys.exit(1)

# Import the existing C# beatmap parser for accurate slider calculations
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "osuparse"))
    from osuparser import findHitObject
except ImportError:
    print("Warning: C# beatmap parser not available. Slider calculations may be inaccurate.")
    findHitObject = None


@dataclass
class ProcessedReplay:
    """Processed replay data with extracted features for ML training."""
    # Basic replay info
    beatmap_hash: str
    player_name: str
    accuracy: float
    total_score: int
    max_combo: int
    
    # Hit counts
    count_300: int
    count_100: int
    count_50: int
    count_miss: int
    
    # Mods and timing
    mods: int
    timestamp: datetime
    
    # Processed movement data
    cursor_positions: np.ndarray  # Shape: (n_frames, 2)
    timestamps: np.ndarray        # Shape: (n_frames,)
    key_states: np.ndarray        # Shape: (n_frames, 4) - [M1, M2, K1, K2]
    
    # Movement features
    velocities: np.ndarray        # Shape: (n_frames-1, 2)
    speeds: np.ndarray           # Shape: (n_frames-1,)
    accelerations: np.ndarray    # Shape: (n_frames-2, 2)
    
    # Beatmap info (if available)
    beatmap_objects: Optional[List[Dict]] = None
    
    @property
    def total_hits(self) -> int:
        return self.count_300 + self.count_100 + self.count_50 + self.count_miss
    
    @property
    def duration_ms(self) -> int:
        """Total replay duration in milliseconds."""
        if len(self.timestamps) == 0:
            return 0
        return int(self.timestamps[-1] - self.timestamps[0])
    
    def get_feature_dict(self) -> Dict[str, Any]:
        """Extract features as a dictionary for ML training."""
        features = {
            # Basic replay info
            'accuracy': self.accuracy,
            'total_score': self.total_score,
            'max_combo': self.max_combo,
            'count_300': self.count_300,
            'count_100': self.count_100,
            'count_50': self.count_50,
            'count_miss': self.count_miss,
            'mods': self.mods,
            'duration_ms': self.duration_ms,
            'total_hits': self.total_hits,
            
            # Movement statistics
            'avg_speed': np.mean(self.speeds) if len(self.speeds) > 0 else 0,
            'max_speed': np.max(self.speeds) if len(self.speeds) > 0 else 0,
            'speed_std': np.std(self.speeds) if len(self.speeds) > 0 else 0,
            'speed_95th': np.percentile(self.speeds, 95) if len(self.speeds) > 0 else 0,
            
            # Acceleration statistics
            'avg_acceleration': np.mean(np.linalg.norm(self.accelerations, axis=1)) if len(self.accelerations) > 0 else 0,
            'max_acceleration': np.max(np.linalg.norm(self.accelerations, axis=1)) if len(self.accelerations) > 0 else 0,
            
            # Cursor position statistics
            'cursor_range_x': np.ptp(self.cursor_positions[:, 0]) if len(self.cursor_positions) > 0 else 0,
            'cursor_range_y': np.ptp(self.cursor_positions[:, 1]) if len(self.cursor_positions) > 0 else 0,
            'avg_cursor_x': np.mean(self.cursor_positions[:, 0]) if len(self.cursor_positions) > 0 else 0,
            'avg_cursor_y': np.mean(self.cursor_positions[:, 1]) if len(self.cursor_positions) > 0 else 0,
            
            # Key press statistics
            'key_press_rate': np.mean(np.any(self.key_states, axis=1)) if len(self.key_states) > 0 else 0,
            'left_click_rate': np.mean(self.key_states[:, 0]) if len(self.key_states) > 0 else 0,
            'right_click_rate': np.mean(self.key_states[:, 1]) if len(self.key_states) > 0 else 0,
            'keyboard_rate': np.mean(np.any(self.key_states[:, 2:], axis=1)) if len(self.key_states) > 0 else 0,
            
            # Frame statistics
            'total_frames': len(self.cursor_positions),
            'avg_frame_time': np.mean(np.diff(self.timestamps)) if len(self.timestamps) > 1 else 0,
        }
        
        return features


class ReplayDataLoader:
    """Enhanced replay data loader using osrparse and optional C# beatmap parser."""
    
    def __init__(self, beatmap_dir: Optional[str] = None):
        """
        Initialize the replay data loader.
        
        Args:
            beatmap_dir: Directory containing .osu beatmap files for slider calculations
        """
        self.beatmap_dir = Path(beatmap_dir) if beatmap_dir else None
        self.beatmap_cache = {}  # Cache for parsed beatmaps
    
    def parse_replay_file(self, replay_path: str) -> Optional[ProcessedReplay]:
        """
        Parse a single .osr replay file using osrparse.
        
        Args:
            replay_path: Path to the .osr file
            
        Returns:
            ProcessedReplay object or None if parsing failed
        """
        try:
            # Parse replay using osrparse
            replay = Replay.from_path(replay_path)
            
            # Extract basic replay information
            accuracy = self._calculate_accuracy(replay)
            
            # Process replay events (frames)
            cursor_positions, timestamps, key_states = self._process_replay_events(replay.replay_data)
            
            # Calculate movement features
            velocities, speeds, accelerations = self._calculate_movement_features(
                cursor_positions, timestamps
            )
            
            # Get beatmap objects if beatmap directory is available
            beatmap_objects = None
            if self.beatmap_dir and replay.beatmap_hash:
                beatmap_objects = self._get_beatmap_objects(replay.beatmap_hash)
            
            return ProcessedReplay(
                beatmap_hash=replay.beatmap_hash,
                player_name=replay.username,
                accuracy=accuracy,
                total_score=replay.score,
                max_combo=replay.max_combo,
                count_300=replay.count_300,
                count_100=replay.count_100,
                count_50=replay.count_50,
                count_miss=replay.count_miss,
                mods=replay.mods.value if hasattr(replay.mods, 'value') else int(replay.mods),
                timestamp=replay.timestamp,
                cursor_positions=cursor_positions,
                timestamps=timestamps,
                key_states=key_states,
                velocities=velocities,
                speeds=speeds,
                accelerations=accelerations,
                beatmap_objects=beatmap_objects
            )
            
        except Exception as e:
            print(f"Error parsing replay {replay_path}: {e}")
            return None
    
    def _calculate_accuracy(self, replay) -> float:
        """Calculate accuracy percentage from hit counts."""
        total_hits = replay.count_300 + replay.count_100 + replay.count_50 + replay.count_miss
        if total_hits == 0:
            return 0.0
        
        weighted_hits = (replay.count_300 * 300 + 
                        replay.count_100 * 100 + 
                        replay.count_50 * 50)
        max_score = total_hits * 300
        return (weighted_hits / max_score) * 100.0
    
    def _process_replay_events(self, replay_data: List[ReplayEventOsu]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process replay events into numpy arrays.
        
        Returns:
            cursor_positions: (n_frames, 2) array of [x, y] positions
            timestamps: (n_frames,) array of absolute timestamps
            key_states: (n_frames, 4) array of [M1, M2, K1, K2] states
        """
        if not replay_data:
            return np.array([]), np.array([]), np.array([])
        
        positions = []
        timestamps = []
        key_states = []
        
        current_time = 0
        
        for event in replay_data:
            current_time += event.time_delta
            
            positions.append([event.x, event.y])
            timestamps.append(current_time)
            
            # Extract key states (assuming osrparse Key enum)
            keys = event.keys
            if hasattr(keys, 'value'):
                key_value = keys.value
            else:
                key_value = int(keys)
            
            # Parse key states: M1, M2, K1, K2
            m1 = bool(key_value & 1)    # Left mouse
            m2 = bool(key_value & 2)    # Right mouse
            k1 = bool(key_value & 4)    # Keyboard key 1
            k2 = bool(key_value & 8)    # Keyboard key 2
            
            key_states.append([m1, m2, k1, k2])
        
        return (
            np.array(positions, dtype=np.float32),
            np.array(timestamps, dtype=np.int32),
            np.array(key_states, dtype=bool)
        )
    
    def _calculate_movement_features(self, positions: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate movement features from cursor positions and timestamps.
        
        Returns:
            velocities: (n_frames-1, 2) array of velocity vectors
            speeds: (n_frames-1,) array of speed magnitudes
            accelerations: (n_frames-2, 2) array of acceleration vectors
        """
        if len(positions) < 2:
            return np.array([]), np.array([]), np.array([])
        
        # Calculate time differences (convert to seconds)
        time_diffs = np.diff(timestamps) / 1000.0
        time_diffs = np.maximum(time_diffs, 1e-6)  # Avoid division by zero
        
        # Calculate velocities (pixels per second)
        position_diffs = np.diff(positions, axis=0)
        velocities = position_diffs / time_diffs.reshape(-1, 1)
        
        # Calculate speeds (magnitude of velocity)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Calculate accelerations
        if len(velocities) < 2:
            accelerations = np.array([])
        else:
            velocity_diffs = np.diff(velocities, axis=0)
            time_diffs_acc = time_diffs[1:]  # One less element due to diff
            accelerations = velocity_diffs / time_diffs_acc.reshape(-1, 1)
        
        return velocities, speeds, accelerations
    
    def _get_beatmap_objects(self, beatmap_hash: str) -> Optional[List[Dict]]:
        """
        Get beatmap objects using the C# parser for accurate slider calculations.
        
        Args:
            beatmap_hash: Hash of the beatmap
            
        Returns:
            List of beatmap objects or None if not found
        """
        if not self.beatmap_dir or not findHitObject:
            return None
        
        # Check cache first
        if beatmap_hash in self.beatmap_cache:
            return self.beatmap_cache[beatmap_hash]
        
        # Find beatmap file
        beatmap_file = self.beatmap_dir / f"{beatmap_hash}.osu"
        if not beatmap_file.exists():
            return None
        
        try:
            # Use C# parser for accurate slider calculations
            beatmap_objects = findHitObject(str(beatmap_file))
            
            # Cache the result
            self.beatmap_cache[beatmap_hash] = beatmap_objects
            
            return beatmap_objects
            
        except Exception as e:
            print(f"Error parsing beatmap {beatmap_hash}: {e}")
            return None
    
    def load_dataset_from_csv(self, csv_path: str, replay_dir: str, 
                             max_replays: Optional[int] = None) -> List[ProcessedReplay]:
        """
        Load replays from a CSV index file.
        
        Args:
            csv_path: Path to the CSV index file
            replay_dir: Directory containing .osr replay files
            max_replays: Maximum number of replays to load (None for all)
            
        Returns:
            List of ProcessedReplay objects
        """
        try:
            # Read CSV index
            df = pd.read_csv(csv_path)
            print(f"Found {len(df)} replays in index")
            
            if max_replays:
                df = df.head(max_replays)
                print(f"Loading first {len(df)} replays")
            
            replays = []
            replay_dir = Path(replay_dir)
            
            for idx, row in df.iterrows():
                if idx % 1000 == 0:
                    print(f"Processing replay {idx + 1}/{len(df)}")
                
                # Construct replay file path (assuming hash-based naming)
                replay_hash = row.get('replay_hash', row.get('hash', ''))
                if not replay_hash:
                    continue
                
                replay_path = replay_dir / f"{replay_hash}.osr"
                if not replay_path.exists():
                    continue
                
                processed_replay = self.parse_replay_file(str(replay_path))
                if processed_replay:
                    replays.append(processed_replay)
            
            print(f"Successfully loaded {len(replays)} replays")
            return replays
            
        except Exception as e:
            print(f"Error loading dataset from CSV: {e}")
            return []
    
    def extract_features_dataframe(self, replays: List[ProcessedReplay]) -> pd.DataFrame:
        """
        Extract features from processed replays into a pandas DataFrame.
        
        Args:
            replays: List of ProcessedReplay objects
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        for replay in replays:
            features = replay.get_feature_dict()
            features['beatmap_hash'] = replay.beatmap_hash
            features['player_name'] = replay.player_name
            features_list.append(features)
        
        return pd.DataFrame(features_list)


if __name__ == "__main__":
    # Test the enhanced replay parser
    import sys
    
    if len(sys.argv) > 1:
        replay_path = sys.argv[1]
        beatmap_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        loader = ReplayDataLoader(beatmap_dir=beatmap_dir)
        replay = loader.parse_replay_file(replay_path)
        
        if replay:
            print(f"Successfully parsed replay: {replay.player_name}")
            print(f"Accuracy: {replay.accuracy:.2f}%")
            print(f"Score: {replay.total_score}")
            print(f"Frames: {len(replay.cursor_positions)}")
            print(f"Duration: {replay.duration_ms}ms")
            
            features = replay.get_feature_dict()
            print("\nExtracted features:")
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("Failed to parse replay file")
    else:
        print("Usage: python replay_parser.py <replay_file.osr> [beatmap_directory]")