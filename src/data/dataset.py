"""Dataset classes for training the osu! replay transformer."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
import random
import math
from pathlib import Path
import logging
from dataclasses import dataclass

from .npy_loader import NumpyReplay, NumpyReplayLoader as ReplayLoader
from ..config.model_config import DataConfig
from .slider_features import SliderFeatureExtractor, create_slider_info_from_beatmap


class OsuReplayDataset(Dataset):
    """Dataset for osu! replay training with sliding window approach."""
    
    def __init__(self, data_config: DataConfig, split: str = 'train'):
        """
        Initialize the dataset.
        
        Args:
            data_config: Data configuration
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = data_config
        self.split = split
        self.window_size = data_config.sequence_length
        self.stride = data_config.stride
        
        # Load data
        self.loader = ReplayLoader(replay_dir=data_config.replay_dir, csv_path=data_config.csv_path)
        self.replays = self._load_replays()
        
        # Initialize slider feature extractor
        self.slider_extractor = SliderFeatureExtractor()
        
        # Create sliding windows
        self.windows = self._create_windows()
        
        print(f"Loaded {len(self.replays)} replays with {len(self.windows)} windows for {split} split")
    
    def _load_replays(self) -> List[NumpyReplay]:
        """Load and filter replays based on configuration."""
        # Get all available replay hashes
        available_hashes = self.loader.list_available_replays()
        # Load all replays
        all_replays = self.loader.load_batch(available_hashes)
        
        # Filter by split
        if self.split == 'train':
            replays = all_replays[:int(len(all_replays) * self.config.train_split)]
        elif self.split == 'val':
            start_idx = int(len(all_replays) * self.config.train_split)
            end_idx = start_idx + int(len(all_replays) * self.config.val_split)
            replays = all_replays[start_idx:end_idx]
        else:  # test
            start_idx = int(len(all_replays) * (self.config.train_split + self.config.val_split))
            replays = all_replays[start_idx:]
        
        # Filter by quality criteria
        filtered_replays = []
        for replay in replays:
            if self._is_valid_replay(replay):
                filtered_replays.append(replay)
        
        return filtered_replays
    
    def _is_valid_replay(self, replay: NumpyReplay) -> bool:
        """Check if replay meets quality criteria."""
        # Check minimum length using data shape
        if replay.duration_frames < self.config.min_sequence_length:
            return False
        
        # Check accuracy range if metadata is available
        if replay.metadata and 'performance-Accuracy' in replay.metadata:
            accuracy = replay.metadata['performance-Accuracy']
            if accuracy < self.config.min_accuracy or accuracy > self.config.max_accuracy:
                return False
        
        # Check star rating if metadata is available
        if replay.metadata and 'beatmap-StarRating' in replay.metadata:
            star_rating = replay.metadata['beatmap-StarRating']
            if star_rating < self.config.min_star_rating or star_rating > self.config.max_star_rating:
                return False
        
        return True
    
    def _create_windows(self) -> List[Tuple[int, int, int]]:
        """Create sliding windows from replays.
        
        Returns:
            List of (replay_idx, start_frame, end_frame) tuples
        """
        windows = []
        
        for replay_idx, replay in enumerate(self.replays):
            replay_length = replay.duration_frames
            
            # Create sliding windows
            for start_frame in range(0, replay_length - self.window_size + 1, self.stride):
                end_frame = start_frame + self.window_size
                windows.append((replay_idx, start_frame, end_frame))
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.
        
        Returns:
            Dictionary containing:
            - cursor_data: Cursor positions (seq_len, 2)
            - beatmap_data: Beatmap features (seq_len, beatmap_features)
            - slider_data: Slider features (seq_len, slider_features)
            - timing_data: Timing information (seq_len, 1)
            - key_data: Key states (seq_len, 4)
            - accuracy_target: Target accuracy (1,)
            - mask: Attention mask (seq_len,)
        """
        replay_idx, start_frame, end_frame = self.windows[idx]
        replay = self.replays[replay_idx]
        
        # Extract window data
        window_data = replay.data[start_frame:end_frame]
        
        # Extract features
        cursor_data = self._extract_cursor_data(window_data)
        beatmap_data = self._extract_beatmap_data(window_data)
        slider_data = self._extract_slider_data(window_data, replay)
        timing_data = self._extract_timing_data(window_data)
        key_data = self._extract_key_data(window_data)
        accuracy_target = self._extract_accuracy_target(replay)
        
        # Create attention mask (all ones for now, can be modified for padding)
        mask = torch.ones(self.window_size, dtype=torch.bool)
        
        return {
            'cursor_data': cursor_data,
            'beatmap_data': beatmap_data,
            'slider_data': slider_data,
            'timing_data': timing_data,
            'key_data': key_data,
            'accuracy_target': accuracy_target,
            'mask': mask
        }
    
    def _extract_cursor_data(self, window_data) -> torch.Tensor:
        """Extract cursor position data."""
        # Assuming columns: [time, cursor_x, cursor_y, feature1, feature2]
        cursor_x = window_data[:, 1]  # Column 1: cursor_x
        cursor_y = window_data[:, 2]  # Column 2: cursor_y
        
        # Normalize to [0, 1] range (assuming osu! coordinates)
        cursor_x = cursor_x / 512.0  # osu! width
        cursor_y = cursor_y / 384.0  # osu! height
        
        cursor_data = np.stack([cursor_x, cursor_y], axis=-1)
        return torch.tensor(cursor_data, dtype=torch.float32)
    
    def _extract_beatmap_data(self, window_data) -> torch.Tensor:
        """Extract beatmap-related features."""
        # Extract available beatmap features from numpy array
        # Assuming window_data is a numpy array with shape (seq_len, features)
        beatmap_features = []
        
        # For now, create placeholder beatmap features since we don't have
        # the exact column mapping for the numpy array
        seq_len = window_data.shape[0]
        
        # Add placeholder features (can be updated when column mapping is known)
        # Hit object positions (normalized)
        if window_data.shape[1] > 3:  # Check if we have enough columns
            hit_x = window_data[:, 3] / 512.0 if window_data.shape[1] > 3 else np.zeros(seq_len)
            beatmap_features.append(hit_x)
        else:
            beatmap_features.append(np.zeros(seq_len))
            
        if window_data.shape[1] > 4:  # Check if we have enough columns
            hit_y = window_data[:, 4] / 384.0 if window_data.shape[1] > 4 else np.zeros(seq_len)
            beatmap_features.append(hit_y)
        else:
            beatmap_features.append(np.zeros(seq_len))
        
        # Hit object type
        if window_data.shape[1] > 5:  # Check if we have enough columns
            hit_type = window_data[:, 5] if window_data.shape[1] > 5 else np.zeros(seq_len)
            beatmap_features.append(hit_type)
        else:
            beatmap_features.append(np.zeros(seq_len))
        
        # Approach rate, overall difficulty, etc. (placeholder features)
        # Add more placeholder features for beatmap characteristics
        for i in range(4):  # 4 additional features
            if window_data.shape[1] > 6 + i:  # Check if we have enough columns
                values = window_data[:, 6 + i]
                beatmap_features.append(values)
            else:
                beatmap_features.append(np.zeros(seq_len))
        
        # Distance to hit object (placeholder)
        if window_data.shape[1] > 10:  # Check if we have enough columns
            distance = window_data[:, 10] / 512.0  # Normalize
            beatmap_features.append(distance)
        else:
            beatmap_features.append(np.zeros(seq_len))
        
        # Stack features
        beatmap_data = np.stack(beatmap_features, axis=-1)
        return torch.tensor(beatmap_data, dtype=torch.float32)
    
    def _extract_timing_data(self, window_data) -> torch.Tensor:
        """Extract timing information."""
        # Assume timing is in the first column (time) or create synthetic timing
        if window_data.shape[1] > 0:
            timing = window_data[:, 0] / 1000.0  # Convert to seconds if it's time in ms
        else:
            timing = np.arange(window_data.shape[0]) * 0.016  # Assume 60 FPS
        
        return torch.tensor(timing, dtype=torch.float32).unsqueeze(-1)
    
    def _extract_key_data(self, window_data) -> torch.Tensor:
        """Extract key press data."""
        keys = []
        seq_len = window_data.shape[0]
        
        # Assume key data is in specific columns or create placeholder
        # For now, create 4 key channels with placeholder data
        for i in range(4):  # 4 keys: m1, m2, k1, k2
            if window_data.shape[1] > 11 + i:  # Check if we have enough columns
                key_data = window_data[:, 11 + i].astype(np.float32)
            else:
                key_data = np.zeros(seq_len, dtype=np.float32)
            keys.append(key_data)
        
        key_data = np.stack(keys, axis=-1)
        return torch.tensor(key_data, dtype=torch.float32)
    
    def _extract_slider_data(self, window_data, replay: NumpyReplay) -> torch.Tensor:
        """Extract slider-specific features."""
        seq_len = window_data.shape[0]
        slider_features = []
        
        # Get beatmap metadata for slider calculations
        beatmap_metadata = getattr(replay, 'beatmap_metadata', {})
        bpm = beatmap_metadata.get('bpm', 120.0)
        beat_length = 60000.0 / bpm  # ms per beat
        slider_multiplier = beatmap_metadata.get('slider_multiplier', 1.4)
        
        for i in range(seq_len):
            current_time = window_data[i, 0] if window_data.shape[1] > 0 else i * 16.67  # Assume 60 FPS
            cursor_pos = (window_data[i, 1], window_data[i, 2]) if window_data.shape[1] > 2 else (0, 0)
            
            # Calculate cursor velocity (simple finite difference)
            if i > 0 and window_data.shape[1] > 2:
                dt = max(1.0, window_data[i, 0] - window_data[i-1, 0])  # Time difference
                dx = window_data[i, 1] - window_data[i-1, 1]
                dy = window_data[i, 2] - window_data[i-1, 2]
                cursor_velocity = (dx / dt, dy / dt)
            else:
                cursor_velocity = (0.0, 0.0)
            
            # Check if there's an active slider at this time
            # For now, create placeholder slider info - this would need actual beatmap data
            slider_info = self._get_active_slider(current_time, beatmap_metadata)
            
            if slider_info:
                # Extract slider features using the slider extractor
                features = self.slider_extractor.extract_slider_features(
                    slider_info, current_time, cursor_pos, cursor_velocity
                )
                
                # Convert to list in consistent order
                feature_values = [
                    features.get('slider_progress', 0.0),
                    features.get('target_slider_x', 0.0),
                    features.get('target_slider_y', 0.0),
                    features.get('slider_active', 0.0),
                    features.get('target_velocity', 0.0),
                    features.get('current_velocity', 0.0),
                    features.get('velocity_error', 0.0),
                    features.get('time_remaining', 0.0),
                    features.get('time_elapsed', 0.0),
                    features.get('urgency_factor', 0.0),
                    features.get('curve_complexity', 0.0),
                    features.get('direction_change', 0.0),
                    features.get('current_bpm', bpm / 200.0)  # Normalized
                ]
            else:
                # No active slider - all features are zero
                feature_values = [0.0] * 13
            
            slider_features.append(feature_values)
        
        slider_data = np.array(slider_features, dtype=np.float32)
        return torch.tensor(slider_data, dtype=torch.float32)
    
    def _get_active_slider(self, current_time: float, beatmap_metadata: dict):
        """Get active slider at current time."""
        # Check if we have hit objects in the beatmap metadata
        hit_objects = beatmap_metadata.get('hit_objects', [])
        
        for hit_object in hit_objects:
            # Check if this is a slider (type 2 in osu! format)
            if hit_object.get('type', 0) & 2:  # Slider flag
                start_time = hit_object.get('time', 0)
                duration = hit_object.get('duration', 0)
                end_time = start_time + duration
                
                # Check if current time is within slider duration
                if start_time <= current_time <= end_time:
                    # Create a basic slider info object
                    slider_info = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'start_x': hit_object.get('x', 256),
                        'start_y': hit_object.get('y', 192),
                        'curve_type': hit_object.get('curve_type', 'L'),  # Linear by default
                        'curve_points': hit_object.get('curve_points', []),
                        'slides': hit_object.get('slides', 1),
                        'length': hit_object.get('length', 100.0)
                    }
                    return slider_info
        
        return None
    
    def _extract_accuracy_target(self, replay: NumpyReplay) -> torch.Tensor:
        """Extract target accuracy for conditioning."""
        # Use metadata if available, otherwise default
        if replay.metadata and 'accuracy' in replay.metadata:
            accuracy = replay.metadata['accuracy']
        else:
            accuracy = 0.95  # Default target
        
        return torch.tensor([accuracy], dtype=torch.float32)


class ReplayDataLoader:
    """Data loader with custom collation for variable-length sequences."""
    
    def __init__(self, dataset: OsuReplayDataset, batch_size: int, 
                 shuffle: bool = True, num_workers: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collation function for batching."""
        # Stack all tensors
        collated = {}
        
        for key in batch[0].keys():
            if key == 'accuracy_target':
                # Stack accuracy targets
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                # Stack sequence data and transpose to (seq_len, batch_size, ...)
                stacked = torch.stack([item[key] for item in batch])
                collated[key] = stacked.transpose(0, 1)
        
        return collated
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


class ReplayAugmentation:
    """Data augmentation for replay sequences."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.noise_std = config.noise_std
        self.time_stretch_range = config.time_stretch_range
        self.cursor_offset_range = config.cursor_offset_range
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation to a sample."""
        if not self.config.use_augmentation:
            return sample
        
        # Copy sample to avoid modifying original
        augmented = {k: v.clone() for k, v in sample.items()}
        
        # Add cursor noise
        if self.noise_std > 0:
            noise = torch.randn_like(augmented['cursor_data']) * self.noise_std
            augmented['cursor_data'] = augmented['cursor_data'] + noise
            # Clamp to valid range
            augmented['cursor_data'] = torch.clamp(augmented['cursor_data'], 0, 1)
        
        # Time stretching (modify timing data)
        if self.time_stretch_range[0] < 1.0 or self.time_stretch_range[1] > 1.0:
            stretch_factor = random.uniform(*self.time_stretch_range)
            augmented['timing_data'] = augmented['timing_data'] * stretch_factor
        
        # Cursor offset
        if self.cursor_offset_range > 0:
            offset_x = random.uniform(-self.cursor_offset_range, self.cursor_offset_range)
            offset_y = random.uniform(-self.cursor_offset_range, self.cursor_offset_range)
            offset = torch.tensor([offset_x, offset_y])
            augmented['cursor_data'] = augmented['cursor_data'] + offset
            # Clamp to valid range
            augmented['cursor_data'] = torch.clamp(augmented['cursor_data'], 0, 1)
        
        return augmented


def create_dataloaders(data_config: DataConfig) -> Tuple[ReplayDataLoader, ReplayDataLoader, ReplayDataLoader]:
    """Create train, validation, and test data loaders."""
    # Create datasets
    train_dataset = OsuReplayDataset(data_config, 'train')
    val_dataset = OsuReplayDataset(data_config, 'val')
    test_dataset = OsuReplayDataset(data_config, 'test')
    
    # Apply augmentation to training dataset
    if data_config.use_augmentation:
        augmentation = ReplayAugmentation(data_config)
        train_dataset.augmentation = augmentation
    
    # Create data loaders
    train_loader = ReplayDataLoader(
        train_dataset, 
        data_config.batch_size, 
        shuffle=True, 
        num_workers=data_config.num_workers
    )
    
    val_loader = ReplayDataLoader(
        val_dataset, 
        data_config.batch_size, 
        shuffle=False, 
        num_workers=data_config.num_workers
    )
    
    test_loader = ReplayDataLoader(
        test_dataset, 
        data_config.batch_size, 
        shuffle=False, 
        num_workers=data_config.num_workers
    )
    
    return train_loader, val_loader, test_loader