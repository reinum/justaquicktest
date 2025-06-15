"""Slider feature extraction module for enhanced AI training.

This module implements comprehensive slider feature extraction including:
- Position features (progress, target position, path points)
- Velocity features (target velocity, current velocity, velocity error)
- Temporal features (time remaining, elapsed, urgency factor)
- Geometric features (curve complexity, direction changes)
- Context features (BPM, slider velocity, multipliers)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import math
from dataclasses import dataclass
from enum import Enum


class SliderCurveType(Enum):
    """Slider curve types in osu!."""
    LINEAR = 0
    PERFECT_CIRCLE = 1
    BEZIER = 2
    CATMULL = 3


@dataclass
class SliderInfo:
    """Information about a slider object."""
    start_time: float  # ms
    end_time: float    # ms
    start_pos: Tuple[float, float]  # (x, y)
    curve_type: SliderCurveType
    curve_points: List[Tuple[float, float]]  # Control points
    repeat_count: int
    pixel_length: float
    slider_multiplier: float
    slider_velocity: float  # SV multiplier
    bpm: float
    beat_length: float  # ms per beat


class SliderPathCalculator:
    """Calculate slider paths using osu! algorithms."""
    
    def __init__(self):
        self.path_cache = {}  # Cache calculated paths
    
    def calculate_path(self, slider_info: SliderInfo, num_points: int = 100) -> np.ndarray:
        """Calculate slider path points.
        
        Args:
            slider_info: Slider information
            num_points: Number of points to calculate along the path
            
        Returns:
            Array of shape (num_points, 2) with path coordinates
        """
        cache_key = self._get_cache_key(slider_info, num_points)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if slider_info.curve_type == SliderCurveType.LINEAR:
            path = self._calculate_linear_path(slider_info, num_points)
        elif slider_info.curve_type == SliderCurveType.PERFECT_CIRCLE:
            path = self._calculate_circle_path(slider_info, num_points)
        elif slider_info.curve_type == SliderCurveType.BEZIER:
            path = self._calculate_bezier_path(slider_info, num_points)
        elif slider_info.curve_type == SliderCurveType.CATMULL:
            path = self._calculate_catmull_path(slider_info, num_points)
        else:
            # Fallback to linear
            path = self._calculate_linear_path(slider_info, num_points)
        
        self.path_cache[cache_key] = path
        return path
    
    def _get_cache_key(self, slider_info: SliderInfo, num_points: int) -> str:
        """Generate cache key for slider path."""
        points_str = '_'.join([f"{p[0]:.1f},{p[1]:.1f}" for p in slider_info.curve_points])
        return f"{slider_info.curve_type.value}_{points_str}_{num_points}_{slider_info.pixel_length}"
    
    def _calculate_linear_path(self, slider_info: SliderInfo, num_points: int) -> np.ndarray:
        """Calculate linear slider path."""
        start = np.array(slider_info.start_pos)
        if len(slider_info.curve_points) > 1:
            end = np.array(slider_info.curve_points[1])
        else:
            end = start + np.array([slider_info.pixel_length, 0])
        
        t_values = np.linspace(0, 1, num_points)
        path = start[None, :] + t_values[:, None] * (end - start)[None, :]
        return path
    
    def _calculate_circle_path(self, slider_info: SliderInfo, num_points: int) -> np.ndarray:
        """Calculate perfect circle slider path."""
        if len(slider_info.curve_points) < 3:
            return self._calculate_linear_path(slider_info, num_points)
        
        p1 = np.array(slider_info.start_pos)
        p2 = np.array(slider_info.curve_points[1])
        p3 = np.array(slider_info.curve_points[2])
        
        # Calculate circle center and radius
        center, radius = self._circle_from_three_points(p1, p2, p3)
        
        # Calculate angles
        start_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
        end_angle = math.atan2(p3[1] - center[1], p3[0] - center[0])
        
        # Determine arc direction and length
        angle_diff = end_angle - start_angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Scale to match pixel length
        arc_length = abs(angle_diff) * radius
        if arc_length > 0:
            scale_factor = slider_info.pixel_length / arc_length
            angle_diff *= scale_factor
        
        # Generate path points
        angles = np.linspace(start_angle, start_angle + angle_diff, num_points)
        path = center[None, :] + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        return path
    
    def _calculate_bezier_path(self, slider_info: SliderInfo, num_points: int) -> np.ndarray:
        """Calculate Bezier curve slider path using De Casteljau's algorithm."""
        control_points = [slider_info.start_pos] + slider_info.curve_points[1:]
        control_points = np.array(control_points)
        
        if len(control_points) < 2:
            return self._calculate_linear_path(slider_info, num_points)
        
        t_values = np.linspace(0, 1, num_points)
        path = np.zeros((num_points, 2))
        
        for i, t in enumerate(t_values):
            path[i] = self._de_casteljau(control_points, t)
        
        # Scale to match pixel length
        path = self._scale_path_to_length(path, slider_info.pixel_length)
        return path
    
    def _calculate_catmull_path(self, slider_info: SliderInfo, num_points: int) -> np.ndarray:
        """Calculate Catmull-Rom spline slider path."""
        control_points = [slider_info.start_pos] + slider_info.curve_points[1:]
        control_points = np.array(control_points)
        
        if len(control_points) < 4:
            return self._calculate_bezier_path(slider_info, num_points)
        
        t_values = np.linspace(0, 1, num_points)
        path = np.zeros((num_points, 2))
        
        for i, t in enumerate(t_values):
            path[i] = self._catmull_rom_interpolate(control_points, t)
        
        # Scale to match pixel length
        path = self._scale_path_to_length(path, slider_info.pixel_length)
        return path
    
    def _circle_from_three_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate circle center and radius from three points."""
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            # Points are collinear, return large radius
            center = (p1 + p3) / 2
            radius = np.linalg.norm(p3 - p1) / 2
            return center, radius
        
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        
        center = np.array([ux, uy])
        radius = np.linalg.norm(p1 - center)
        return center, radius
    
    def _de_casteljau(self, control_points: np.ndarray, t: float) -> np.ndarray:
        """De Casteljau's algorithm for Bezier curves."""
        points = control_points.copy()
        n = len(points)
        
        for i in range(1, n):
            for j in range(n - i):
                points[j] = (1 - t) * points[j] + t * points[j + 1]
        
        return points[0]
    
    def _catmull_rom_interpolate(self, control_points: np.ndarray, t: float) -> np.ndarray:
        """Catmull-Rom spline interpolation."""
        n = len(control_points)
        if n < 4:
            return control_points[0]
        
        # Scale t to segment index
        segment_t = t * (n - 3)
        segment_idx = int(segment_t)
        local_t = segment_t - segment_idx
        
        # Clamp to valid range
        segment_idx = max(0, min(segment_idx, n - 4))
        
        p0 = control_points[segment_idx]
        p1 = control_points[segment_idx + 1]
        p2 = control_points[segment_idx + 2]
        p3 = control_points[segment_idx + 3]
        
        # Catmull-Rom formula
        t2 = local_t * local_t
        t3 = t2 * local_t
        
        return (0.5 * ((2 * p1) +
                      (-p0 + p2) * local_t +
                      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                      (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    
    def _scale_path_to_length(self, path: np.ndarray, target_length: float) -> np.ndarray:
        """Scale path to match target pixel length."""
        if len(path) < 2:
            return path
        
        # Calculate cumulative distances
        diffs = np.diff(path, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative_distances[-1]
        
        if total_length == 0:
            return path
        
        # Interpolate to match target length
        scale_factor = target_length / total_length
        target_distances = cumulative_distances * scale_factor
        
        # Interpolate path points
        scaled_path = np.zeros_like(path)
        for i in range(len(path)):
            if i == 0:
                scaled_path[i] = path[0]
            else:
                # Find interpolation ratio
                ratio = target_distances[i] / cumulative_distances[-1] if cumulative_distances[-1] > 0 else 0
                ratio = min(1.0, ratio)
                
                # Linear interpolation along original path
                interp_distance = ratio * total_length
                idx = np.searchsorted(cumulative_distances, interp_distance)
                idx = min(idx, len(path) - 1)
                
                if idx == 0:
                    scaled_path[i] = path[0]
                else:
                    t = (interp_distance - cumulative_distances[idx-1]) / distances[idx-1] if distances[idx-1] > 0 else 0
                    scaled_path[i] = path[idx-1] + t * (path[idx] - path[idx-1])
        
        return scaled_path


class SliderFeatureExtractor:
    """Extract comprehensive slider features for AI training."""
    
    def __init__(self):
        self.path_calculator = SliderPathCalculator()
    
    def extract_slider_features(self, slider_info: SliderInfo, current_time: float, 
                              cursor_pos: Tuple[float, float], cursor_velocity: Tuple[float, float]) -> Dict[str, float]:
        """Extract all slider features for the current time.
        
        Args:
            slider_info: Slider information
            current_time: Current time in ms
            cursor_pos: Current cursor position (x, y)
            cursor_velocity: Current cursor velocity (dx/dt, dy/dt) in pixels/ms
            
        Returns:
            Dictionary of slider features
        """
        features = {}
        
        # Position features
        position_features = self._extract_position_features(slider_info, current_time)
        features.update(position_features)
        
        # Velocity features
        velocity_features = self._extract_velocity_features(slider_info, current_time, cursor_velocity)
        features.update(velocity_features)
        
        # Temporal features
        temporal_features = self._extract_temporal_features(slider_info, current_time)
        features.update(temporal_features)
        
        # Geometric features
        geometric_features = self._extract_geometric_features(slider_info, current_time)
        features.update(geometric_features)
        
        # Context features
        context_features = self._extract_context_features(slider_info)
        features.update(context_features)
        
        return features
    
    def _extract_position_features(self, slider_info: SliderInfo, current_time: float) -> Dict[str, float]:
        """Extract position-related features."""
        # Calculate slider progress (0-1)
        duration = slider_info.end_time - slider_info.start_time
        if duration <= 0:
            progress = 1.0
        else:
            progress = max(0.0, min(1.0, (current_time - slider_info.start_time) / duration))
        
        # Get path points
        path = self.path_calculator.calculate_path(slider_info, num_points=100)
        
        # Calculate target position
        path_idx = int(progress * (len(path) - 1))
        path_idx = max(0, min(path_idx, len(path) - 1))
        target_pos = path[path_idx]
        
        return {
            'slider_progress': progress,
            'target_slider_x': target_pos[0] / 512.0,  # Normalize to [0, 1]
            'target_slider_y': target_pos[1] / 384.0,  # Normalize to [0, 1]
            'slider_active': 1.0 if slider_info.start_time <= current_time <= slider_info.end_time else 0.0
        }
    
    def _extract_velocity_features(self, slider_info: SliderInfo, current_time: float, 
                                 cursor_velocity: Tuple[float, float]) -> Dict[str, float]:
        """Extract velocity-related features."""
        # Calculate target velocity using osu! formula
        # velocity = pixel_length / (slider_multiplier * 100 * SV) * beat_length
        duration_ms = slider_info.pixel_length / (slider_info.slider_multiplier * 100 * slider_info.slider_velocity) * slider_info.beat_length
        
        if duration_ms <= 0:
            target_velocity = 0.0
        else:
            target_velocity = slider_info.pixel_length / duration_ms  # pixels/ms
        
        # Current velocity magnitude
        current_velocity_mag = math.sqrt(cursor_velocity[0]**2 + cursor_velocity[1]**2)
        
        # Velocity error
        velocity_error = target_velocity - current_velocity_mag
        
        return {
            'target_velocity': target_velocity / 10.0,  # Normalize (typical max ~10 pixels/ms)
            'current_velocity': current_velocity_mag / 10.0,  # Normalize
            'velocity_error': velocity_error / 10.0,  # Normalize
        }
    
    def _extract_temporal_features(self, slider_info: SliderInfo, current_time: float) -> Dict[str, float]:
        """Extract temporal features."""
        duration = slider_info.end_time - slider_info.start_time
        time_elapsed = max(0.0, current_time - slider_info.start_time)
        time_remaining = max(0.0, slider_info.end_time - current_time)
        
        # Urgency factor (higher when time is running out)
        if duration > 0:
            urgency_factor = 1.0 - (time_remaining / duration)
        else:
            urgency_factor = 1.0
        
        return {
            'time_remaining': time_remaining / 1000.0,  # Normalize to seconds
            'time_elapsed': time_elapsed / 1000.0,  # Normalize to seconds
            'urgency_factor': urgency_factor
        }
    
    def _extract_geometric_features(self, slider_info: SliderInfo, current_time: float) -> Dict[str, float]:
        """Extract geometric complexity features."""
        # Calculate curve complexity based on control points
        if slider_info.curve_type == SliderCurveType.LINEAR:
            curve_complexity = 0.0
        elif slider_info.curve_type == SliderCurveType.PERFECT_CIRCLE:
            curve_complexity = 0.3
        elif slider_info.curve_type == SliderCurveType.BEZIER:
            # Complexity based on number of control points
            num_points = len(slider_info.curve_points)
            curve_complexity = min(1.0, num_points / 10.0)
        else:  # CATMULL
            curve_complexity = 0.8
        
        # Calculate direction change (upcoming angle change)
        path = self.path_calculator.calculate_path(slider_info, num_points=50)
        direction_change = self._calculate_direction_change(path, current_time, slider_info)
        
        return {
            'curve_complexity': curve_complexity,
            'direction_change': direction_change,
            'path_segment_type': float(slider_info.curve_type.value) / 3.0  # Normalize
        }
    
    def _extract_context_features(self, slider_info: SliderInfo) -> Dict[str, float]:
        """Extract context features."""
        return {
            'current_bpm': slider_info.bpm / 200.0,  # Normalize (typical max ~200 BPM)
            'slider_velocity': slider_info.slider_velocity,  # Already normalized (typically 0.5-2.0)
            'slider_multiplier': slider_info.slider_multiplier / 3.0  # Normalize (typical max ~3.0)
        }
    
    def _calculate_direction_change(self, path: np.ndarray, current_time: float, 
                                  slider_info: SliderInfo) -> float:
        """Calculate upcoming direction change in the slider path."""
        if len(path) < 3:
            return 0.0
        
        # Find current position in path
        duration = slider_info.end_time - slider_info.start_time
        if duration <= 0:
            return 0.0
        
        progress = (current_time - slider_info.start_time) / duration
        progress = max(0.0, min(1.0, progress))
        
        current_idx = int(progress * (len(path) - 1))
        
        # Look ahead a few points
        lookahead = min(5, len(path) - current_idx - 1)
        if lookahead < 2:
            return 0.0
        
        # Calculate angle change
        p1 = path[current_idx]
        p2 = path[current_idx + lookahead // 2]
        p3 = path[current_idx + lookahead]
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle between vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
        
        angle_change = math.acos(cos_angle)
        return angle_change / math.pi  # Normalize to [0, 1]


def create_slider_info_from_beatmap(hit_object: Dict, bpm: float, beat_length: float, 
                                   slider_multiplier: float = 1.4, slider_velocity: float = 1.0) -> Optional[SliderInfo]:
    """Create SliderInfo from beatmap hit object data.
    
    Args:
        hit_object: Dictionary containing hit object data
        bpm: Current BPM
        beat_length: Beat length in ms
        slider_multiplier: Base slider multiplier
        slider_velocity: Slider velocity multiplier
        
    Returns:
        SliderInfo object or None if not a slider
    """
    if hit_object.get('type') != 'slider':
        return None
    
    # Parse curve type
    curve_type_str = hit_object.get('curve_type', 'L')
    if curve_type_str == 'L':
        curve_type = SliderCurveType.LINEAR
    elif curve_type_str == 'P':
        curve_type = SliderCurveType.PERFECT_CIRCLE
    elif curve_type_str == 'B':
        curve_type = SliderCurveType.BEZIER
    elif curve_type_str == 'C':
        curve_type = SliderCurveType.CATMULL
    else:
        curve_type = SliderCurveType.BEZIER  # Default
    
    # Calculate end time
    pixel_length = hit_object.get('pixel_length', 100.0)
    repeat_count = hit_object.get('repeat_count', 1)
    
    # Use osu! formula: duration = length / (SliderMultiplier * 100 * SV) * beatLength
    duration_ms = pixel_length / (slider_multiplier * 100 * slider_velocity) * beat_length
    total_duration = duration_ms * repeat_count
    
    return SliderInfo(
        start_time=hit_object['time'],
        end_time=hit_object['time'] + total_duration,
        start_pos=(hit_object['x'], hit_object['y']),
        curve_type=curve_type,
        curve_points=hit_object.get('curve_points', [(hit_object['x'], hit_object['y'])]),
        repeat_count=repeat_count,
        pixel_length=pixel_length,
        slider_multiplier=slider_multiplier,
        slider_velocity=slider_velocity,
        bpm=bpm,
        beat_length=beat_length
    )