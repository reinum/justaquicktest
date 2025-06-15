"""Post-processing utilities for generated replays."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import interpolate, signal
from scipy.spatial.distance import euclidean
import logging
from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import GenerationResult


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""
    smooth_cursor: bool = True
    smooth_window: int = 5
    remove_jitter: bool = True
    jitter_threshold: float = 10.0
    interpolate_gaps: bool = True
    max_gap_ms: float = 50.0
    enforce_timing: bool = True
    timing_tolerance: float = 5.0
    fix_impossible_movements: bool = True
    max_speed_px_ms: float = 2.0
    optimize_key_presses: bool = True
    min_key_duration: float = 10.0


class ReplayPostProcessor:
    """Post-processor for cleaning and optimizing generated replays."""
    
    def __init__(self, config: Optional[PostProcessConfig] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config or PostProcessConfig()
        self.logger = logger or logging.getLogger(__name__)
    
    def process(self, result: 'GenerationResult', 
               beatmap_data: Dict[str, Any]) -> 'GenerationResult':
        """Apply all post-processing steps to a generation result.
        
        Args:
            result: Raw generation result
            beatmap_data: Beatmap information for context
            
        Returns:
            Processed generation result
        """
        self.logger.info("Starting post-processing...")
        
        # Copy the result to avoid modifying the original
        from .generator import GenerationResult
        processed_result = GenerationResult(
            cursor_positions=result.cursor_positions.copy(),
            key_presses=result.key_presses.copy(),
            timestamps=result.timestamps.copy(),
            metadata=result.metadata.copy(),
            generation_time=result.generation_time,
            confidence_scores=result.confidence_scores.copy() if result.confidence_scores is not None else None
        )
        
        # Apply processing steps
        if self.config.smooth_cursor:
            processed_result.cursor_positions = self._smooth_cursor_trajectory(
                processed_result.cursor_positions,
                processed_result.timestamps
            )
        
        if self.config.remove_jitter:
            processed_result.cursor_positions = self._remove_cursor_jitter(
                processed_result.cursor_positions,
                processed_result.timestamps
            )
        
        if self.config.interpolate_gaps:
            processed_result = self._interpolate_timing_gaps(
                processed_result,
                beatmap_data
            )
        
        if self.config.fix_impossible_movements:
            processed_result.cursor_positions = self._fix_impossible_movements(
                processed_result.cursor_positions,
                processed_result.timestamps
            )
        
        if self.config.optimize_key_presses:
            processed_result.key_presses = self._optimize_key_presses(
                processed_result.key_presses,
                processed_result.timestamps,
                beatmap_data
            )
        
        if self.config.enforce_timing:
            processed_result = self._enforce_timing_constraints(
                processed_result,
                beatmap_data
            )
        
        # Update metadata
        processed_result.metadata['post_processed'] = True
        processed_result.metadata['post_process_config'] = self.config.__dict__
        
        self.logger.info("Post-processing completed")
        return processed_result
    
    def _smooth_cursor_trajectory(self, cursor_positions: np.ndarray, 
                                 timestamps: np.ndarray) -> np.ndarray:
        """Smooth cursor trajectory to remove noise.
        
        Args:
            cursor_positions: Array of shape [seq_len, 2]
            timestamps: Array of shape [seq_len]
            
        Returns:
            Smoothed cursor positions
        """
        if len(cursor_positions) < self.config.smooth_window:
            return cursor_positions
        
        smoothed = cursor_positions.copy()
        
        # Apply moving average filter
        window = self.config.smooth_window
        
        for dim in range(2):  # x and y coordinates
            # Pad the signal to handle edges
            padded = np.pad(cursor_positions[:, dim], 
                          (window//2, window//2), 
                          mode='edge')
            
            # Apply moving average
            smoothed[:, dim] = np.convolve(
                padded, 
                np.ones(window) / window, 
                mode='valid'
            )
        
        return smoothed
    
    def _remove_cursor_jitter(self, cursor_positions: np.ndarray,
                             timestamps: np.ndarray) -> np.ndarray:
        """Remove small jittery movements that don't contribute to gameplay.
        
        Args:
            cursor_positions: Array of shape [seq_len, 2]
            timestamps: Array of shape [seq_len]
            
        Returns:
            De-jittered cursor positions
        """
        if len(cursor_positions) < 3:
            return cursor_positions
        
        cleaned = cursor_positions.copy()
        threshold = self.config.jitter_threshold
        
        for i in range(1, len(cursor_positions) - 1):
            # Calculate movement vectors
            prev_movement = cursor_positions[i] - cursor_positions[i-1]
            next_movement = cursor_positions[i+1] - cursor_positions[i]
            
            # Check if this is a jitter (small movement followed by reversal)
            prev_dist = np.linalg.norm(prev_movement)
            next_dist = np.linalg.norm(next_movement)
            
            if prev_dist < threshold and next_dist < threshold:
                # Check if movements are in opposite directions
                dot_product = np.dot(prev_movement, next_movement)
                if dot_product < 0:  # Opposite directions
                    # Replace with interpolated position
                    cleaned[i] = (cursor_positions[i-1] + cursor_positions[i+1]) / 2
        
        return cleaned
    
    def _interpolate_timing_gaps(self, result: 'GenerationResult',
                                beatmap_data: Dict[str, Any]) -> 'GenerationResult':
        """Interpolate missing frames in timing gaps.
        
        Args:
            result: Generation result
            beatmap_data: Beatmap information
            
        Returns:
            Result with interpolated frames
        """
        timestamps = result.timestamps
        cursor_positions = result.cursor_positions
        key_presses = result.key_presses
        
        # Find gaps larger than threshold
        time_diffs = np.diff(timestamps)
        gap_indices = np.where(time_diffs > self.config.max_gap_ms)[0]
        
        if len(gap_indices) == 0:
            return result
        
        # Interpolate each gap
        new_timestamps = []
        new_cursor_positions = []
        new_key_presses = []
        
        last_idx = 0
        
        for gap_idx in gap_indices:
            # Add data up to the gap
            new_timestamps.extend(timestamps[last_idx:gap_idx+1])
            new_cursor_positions.extend(cursor_positions[last_idx:gap_idx+1])
            new_key_presses.extend(key_presses[last_idx:gap_idx+1])
            
            # Interpolate the gap
            start_time = timestamps[gap_idx]
            end_time = timestamps[gap_idx + 1]
            gap_duration = end_time - start_time
            
            # Number of frames to interpolate
            num_frames = int(gap_duration / 16.67)  # Assuming ~60 FPS
            
            if num_frames > 1:
                # Time interpolation
                interp_times = np.linspace(start_time, end_time, num_frames + 2)[1:-1]
                
                # Cursor interpolation (linear)
                start_pos = cursor_positions[gap_idx]
                end_pos = cursor_positions[gap_idx + 1]
                
                for i, t in enumerate(interp_times):
                    alpha = (i + 1) / (num_frames + 1)
                    interp_pos = start_pos * (1 - alpha) + end_pos * alpha
                    
                    new_timestamps.append(t)
                    new_cursor_positions.append(interp_pos)
                    new_key_presses.append(key_presses[gap_idx])  # Keep same key state
            
            last_idx = gap_idx + 1
        
        # Add remaining data
        new_timestamps.extend(timestamps[last_idx:])
        new_cursor_positions.extend(cursor_positions[last_idx:])
        new_key_presses.extend(key_presses[last_idx:])
        
        # Create new result
        from .generator import GenerationResult
        return GenerationResult(
            cursor_positions=np.array(new_cursor_positions),
            key_presses=np.array(new_key_presses),
            timestamps=np.array(new_timestamps),
            metadata=result.metadata,
            generation_time=result.generation_time,
            confidence_scores=result.confidence_scores
        )
    
    def _fix_impossible_movements(self, cursor_positions: np.ndarray,
                                timestamps: np.ndarray) -> np.ndarray:
        """Fix cursor movements that are physically impossible.
        
        Args:
            cursor_positions: Array of shape [seq_len, 2]
            timestamps: Array of shape [seq_len]
            
        Returns:
            Fixed cursor positions
        """
        if len(cursor_positions) < 2:
            return cursor_positions
        
        fixed = cursor_positions.copy()
        max_speed = self.config.max_speed_px_ms
        
        for i in range(1, len(cursor_positions)):
            # Calculate movement
            movement = cursor_positions[i] - cursor_positions[i-1]
            distance = np.linalg.norm(movement)
            time_diff = timestamps[i] - timestamps[i-1]
            
            if time_diff > 0:
                speed = distance / time_diff
                
                if speed > max_speed:
                    # Scale down the movement
                    scale_factor = max_speed / speed
                    scaled_movement = movement * scale_factor
                    fixed[i] = fixed[i-1] + scaled_movement
        
        return fixed
    
    def _optimize_key_presses(self, key_presses: np.ndarray,
                            timestamps: np.ndarray,
                            beatmap_data: Dict[str, Any]) -> np.ndarray:
        """Optimize key press timing and patterns.
        
        Args:
            key_presses: Array of shape [seq_len, 4]
            timestamps: Array of shape [seq_len]
            beatmap_data: Beatmap information
            
        Returns:
            Optimized key presses
        """
        optimized = key_presses.copy()
        hit_objects = beatmap_data.get('hit_objects', [])
        
        # Remove very short key presses
        min_duration = self.config.min_key_duration
        
        for key_idx in range(4):  # For each key
            key_states = optimized[:, key_idx]
            
            # Find press and release events
            press_starts = []
            press_ends = []
            
            in_press = False
            press_start = 0
            
            for i in range(len(key_states)):
                if key_states[i] > 0.5 and not in_press:
                    # Key press started
                    in_press = True
                    press_start = i
                elif key_states[i] <= 0.5 and in_press:
                    # Key press ended
                    in_press = False
                    press_duration = timestamps[i] - timestamps[press_start]
                    
                    if press_duration < min_duration:
                        # Remove short press
                        optimized[press_start:i, key_idx] = 0
            
            # Handle case where key is still pressed at the end
            if in_press:
                press_duration = timestamps[-1] - timestamps[press_start]
                if press_duration < min_duration:
                    optimized[press_start:, key_idx] = 0
        
        # Ensure key presses align with hit objects
        self._align_keys_with_hits(optimized, timestamps, hit_objects)
        
        return optimized
    
    def _align_keys_with_hits(self, key_presses: np.ndarray,
                            timestamps: np.ndarray,
                            hit_objects: List[Dict[str, Any]]):
        """Align key presses with hit object timing.
        
        Args:
            key_presses: Array of shape [seq_len, 4] (modified in-place)
            timestamps: Array of shape [seq_len]
            hit_objects: List of hit objects
        """
        tolerance = self.config.timing_tolerance
        
        for hit_obj in hit_objects:
            hit_time = hit_obj['Time']
            
            # Find closest timestamp
            time_diffs = np.abs(timestamps - hit_time)
            closest_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_idx] <= tolerance:
                # Ensure at least one key is pressed at hit time
                if np.sum(key_presses[closest_idx, :2]) == 0:  # No keyboard keys pressed
                    # Press the appropriate key based on pattern
                    key_to_press = self._determine_key_for_hit(hit_obj, closest_idx, key_presses)
                    key_presses[closest_idx, key_to_press] = 1
    
    def _determine_key_for_hit(self, hit_obj: Dict[str, Any], 
                             timestamp_idx: int,
                             key_presses: np.ndarray) -> int:
        """Determine which key should be pressed for a hit object.
        
        Args:
            hit_obj: Hit object information
            timestamp_idx: Index in the timestamp array
            key_presses: Current key press states
            
        Returns:
            Key index (0 or 1 for K1/K2)
        """
        # Simple alternating pattern
        # In practice, this could be more sophisticated
        
        # Look at recent key presses to determine pattern
        recent_window = max(0, timestamp_idx - 5)
        recent_presses = key_presses[recent_window:timestamp_idx, :2]
        
        # Count recent presses for each key
        k1_count = np.sum(recent_presses[:, 0])
        k2_count = np.sum(recent_presses[:, 1])
        
        # Alternate or use less-used key
        if k1_count <= k2_count:
            return 0  # K1
        else:
            return 1  # K2
    
    def _enforce_timing_constraints(self, result: 'GenerationResult',
                                   beatmap_data: Dict[str, Any]) -> 'GenerationResult':
        """Enforce timing constraints based on beatmap.
        
        Args:
            result: Generation result
            beatmap_data: Beatmap information
            
        Returns:
            Result with enforced timing
        """
        # This is a placeholder for more sophisticated timing enforcement
        # In practice, you might adjust timestamps to better align with
        # the beatmap's timing points and hit objects
        
        return result
    
    def validate_replay(self, result: 'GenerationResult',
                       beatmap_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated replay for common issues.
        
        Args:
            result: Generation result to validate
            beatmap_data: Beatmap information
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        # Check cursor bounds
        cursor_pos = result.cursor_positions
        if np.any(cursor_pos[:, 0] < 0) or np.any(cursor_pos[:, 0] > 512):
            issues.append("Cursor X position out of bounds")
        if np.any(cursor_pos[:, 1] < 0) or np.any(cursor_pos[:, 1] > 384):
            issues.append("Cursor Y position out of bounds")
        
        # Check for impossible movements
        if len(cursor_pos) > 1:
            movements = np.diff(cursor_pos, axis=0)
            distances = np.linalg.norm(movements, axis=1)
            time_diffs = np.diff(result.timestamps)
            
            valid_time_diffs = time_diffs[time_diffs > 0]
            if len(valid_time_diffs) > 0:
                speeds = distances[:len(valid_time_diffs)] / valid_time_diffs
                max_speed = np.max(speeds)
                
                if max_speed > self.config.max_speed_px_ms:
                    warnings.append(f"High cursor speed detected: {max_speed:.2f} px/ms")
        
        # Check key press patterns
        key_presses = result.key_presses
        total_presses = np.sum(key_presses)
        if total_presses == 0:
            issues.append("No key presses detected")
        
        # Check timing consistency
        if len(result.timestamps) > 1:
            time_diffs = np.diff(result.timestamps)
            if np.any(time_diffs < 0):
                issues.append("Non-monotonic timestamps")
            
            avg_interval = np.mean(time_diffs[time_diffs > 0])
            if avg_interval > 100:  # More than 100ms between frames
                warnings.append(f"Large average frame interval: {avg_interval:.2f}ms")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'stats': {
                'total_frames': len(result.timestamps),
                'total_key_presses': int(total_presses),
                'duration_ms': result.timestamps[-1] - result.timestamps[0] if len(result.timestamps) > 1 else 0,
                'avg_cursor_speed': np.mean(speeds) if 'speeds' in locals() and len(speeds) > 0 else 0
            }
        }