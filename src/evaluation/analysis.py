"""Analysis utilities for replay evaluation and comparison."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


@dataclass
class StatisticalAnalysis:
    """Results of statistical analysis."""
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float
    distribution_test: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'median': self.median,
            'q25': self.q25,
            'q75': self.q75,
            'min': self.min_val,
            'max': self.max_val,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'distribution_test': self.distribution_test
        }


@dataclass
class VisualAnalysis:
    """Results of visual analysis."""
    plots_generated: List[str]
    summary_stats: Dict[str, Any]
    anomalies_detected: List[Dict[str, Any]]
    patterns_identified: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'plots_generated': self.plots_generated,
            'summary_stats': self.summary_stats,
            'anomalies_detected': self.anomalies_detected,
            'patterns_identified': self.patterns_identified
        }


class ReplayAnalyzer:
    """Comprehensive replay analysis tool."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = Path(output_dir) if output_dir else Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_replay_data(
        self,
        replay_data: List[Dict[str, Any]],
        analysis_type: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """Analyze replay data comprehensively.
        
        Args:
            replay_data: List of replay events
            analysis_type: Type of analysis ('basic', 'statistical', 'visual', 'comprehensive')
            
        Returns:
            Analysis results dictionary
        """
        results = {}
        
        if analysis_type in ['basic', 'comprehensive']:
            results['basic'] = self._basic_analysis(replay_data)
            
        if analysis_type in ['statistical', 'comprehensive']:
            results['statistical'] = self._statistical_analysis(replay_data)
            
        if analysis_type in ['visual', 'comprehensive']:
            results['visual'] = self._visual_analysis(replay_data)
            
        return results
    
    def compare_replays(
        self,
        original_replay: List[Dict[str, Any]],
        generated_replay: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare two replays and analyze differences.
        
        Args:
            original_replay: Original replay data
            generated_replay: Generated replay data
            
        Returns:
            Comparison analysis results
        """
        results = {
            'similarity_metrics': self._calculate_similarity_metrics(original_replay, generated_replay),
            'movement_analysis': self._analyze_movement_patterns(original_replay, generated_replay),
            'timing_analysis': self._analyze_timing_patterns(original_replay, generated_replay),
            'accuracy_analysis': self._analyze_accuracy_patterns(original_replay, generated_replay)
        }
        
        return results
    
    def _basic_analysis(self, replay_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform basic analysis of replay data."""
        if not replay_data:
            return {'error': 'No replay data provided'}
            
        # Extract coordinates and times
        x_coords = [event.get('x', 0) for event in replay_data]
        y_coords = [event.get('y', 0) for event in replay_data]
        times = [event.get('time', i) for i, event in enumerate(replay_data)]
        
        # Calculate basic metrics
        total_distance = self._calculate_total_distance(x_coords, y_coords)
        avg_speed = self._calculate_average_speed(x_coords, y_coords, times)
        max_speed = self._calculate_max_speed(x_coords, y_coords, times)
        
        # Calculate cursor coverage
        coverage = self._calculate_cursor_coverage(x_coords, y_coords)
        
        return {
            'total_events': len(replay_data),
            'duration_ms': max(times) - min(times) if times else 0,
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'max_speed': max_speed,
            'cursor_coverage': coverage,
            'x_range': (min(x_coords), max(x_coords)) if x_coords else (0, 0),
            'y_range': (min(y_coords), max(y_coords)) if y_coords else (0, 0)
        }
    
    def _statistical_analysis(self, replay_data: List[Dict[str, Any]]) -> StatisticalAnalysis:
        """Perform statistical analysis of replay data."""
        # Extract movement speeds
        x_coords = [event.get('x', 0) for event in replay_data]
        y_coords = [event.get('y', 0) for event in replay_data]
        times = [event.get('time', i) for i, event in enumerate(replay_data)]
        
        speeds = self._calculate_speeds(x_coords, y_coords, times)
        
        if not speeds:
            # Return default values if no speeds calculated
            return StatisticalAnalysis(
                mean=0, std=0, median=0, q25=0, q75=0,
                min_val=0, max_val=0, skewness=0, kurtosis=0,
                distribution_test={'test': 'none', 'p_value': 1.0}
            )
        
        speeds_array = np.array(speeds)
        
        # Calculate statistics
        mean_speed = np.mean(speeds_array)
        std_speed = np.std(speeds_array)
        median_speed = np.median(speeds_array)
        q25 = np.percentile(speeds_array, 25)
        q75 = np.percentile(speeds_array, 75)
        min_speed = np.min(speeds_array)
        max_speed = np.max(speeds_array)
        skewness = stats.skew(speeds_array)
        kurtosis_val = stats.kurtosis(speeds_array)
        
        # Normality test
        if len(speeds_array) > 3:
            stat, p_value = stats.shapiro(speeds_array[:5000])  # Limit for performance
            distribution_test = {'test': 'shapiro', 'statistic': stat, 'p_value': p_value}
        else:
            distribution_test = {'test': 'insufficient_data', 'p_value': 1.0}
        
        return StatisticalAnalysis(
            mean=mean_speed,
            std=std_speed,
            median=median_speed,
            q25=q25,
            q75=q75,
            min_val=min_speed,
            max_val=max_speed,
            skewness=skewness,
            kurtosis=kurtosis_val,
            distribution_test=distribution_test
        )
    
    def _visual_analysis(self, replay_data: List[Dict[str, Any]]) -> VisualAnalysis:
        """Perform visual analysis of replay data."""
        plots_generated = []
        anomalies = []
        patterns = []
        
        # Extract data
        x_coords = [event.get('x', 0) for event in replay_data]
        y_coords = [event.get('y', 0) for event in replay_data]
        times = [event.get('time', i) for i, event in enumerate(replay_data)]
        
        # Generate trajectory plot
        try:
            self._plot_trajectory(x_coords, y_coords, save_path=self.output_dir / 'trajectory.png')
            plots_generated.append('trajectory.png')
        except Exception as e:
            print(f"Failed to generate trajectory plot: {e}")
        
        # Generate speed distribution plot
        try:
            speeds = self._calculate_speeds(x_coords, y_coords, times)
            if speeds:
                self._plot_speed_distribution(speeds, save_path=self.output_dir / 'speed_distribution.png')
                plots_generated.append('speed_distribution.png')
        except Exception as e:
            print(f"Failed to generate speed distribution plot: {e}")
        
        # Detect anomalies
        anomalies = self._detect_anomalies(x_coords, y_coords, times)
        
        # Identify patterns
        patterns = self._identify_patterns(x_coords, y_coords, times)
        
        summary_stats = {
            'total_plots': len(plots_generated),
            'anomalies_found': len(anomalies),
            'patterns_found': len(patterns)
        }
        
        return VisualAnalysis(
            plots_generated=plots_generated,
            summary_stats=summary_stats,
            anomalies_detected=anomalies,
            patterns_identified=patterns
        )
    
    def _calculate_similarity_metrics(
        self,
        original: List[Dict[str, Any]],
        generated: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate similarity metrics between two replays."""
        # Extract coordinates
        orig_x = [event.get('x', 0) for event in original]
        orig_y = [event.get('y', 0) for event in original]
        gen_x = [event.get('x', 0) for event in generated]
        gen_y = [event.get('y', 0) for event in generated]
        
        # Align sequences (simple approach - pad shorter one)
        max_len = max(len(orig_x), len(gen_x))
        orig_x.extend([orig_x[-1]] * (max_len - len(orig_x)))
        orig_y.extend([orig_y[-1]] * (max_len - len(orig_y)))
        gen_x.extend([gen_x[-1]] * (max_len - len(gen_x)))
        gen_y.extend([gen_y[-1]] * (max_len - len(gen_y)))
        
        # Calculate metrics
        mse_x = mean_squared_error(orig_x, gen_x)
        mse_y = mean_squared_error(orig_y, gen_y)
        mae_x = mean_absolute_error(orig_x, gen_x)
        mae_y = mean_absolute_error(orig_y, gen_y)
        
        # Calculate correlation
        corr_x = np.corrcoef(orig_x, gen_x)[0, 1] if len(set(orig_x)) > 1 and len(set(gen_x)) > 1 else 0
        corr_y = np.corrcoef(orig_y, gen_y)[0, 1] if len(set(orig_y)) > 1 and len(set(gen_y)) > 1 else 0
        
        return {
            'mse_x': mse_x,
            'mse_y': mse_y,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'correlation_x': corr_x,
            'correlation_y': corr_y,
            'overall_similarity': (corr_x + corr_y) / 2
        }
    
    def _analyze_movement_patterns(
        self,
        original: List[Dict[str, Any]],
        generated: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze movement patterns between replays."""
        # Extract movement data
        orig_movements = self._extract_movement_features(original)
        gen_movements = self._extract_movement_features(generated)
        
        return {
            'original_patterns': orig_movements,
            'generated_patterns': gen_movements,
            'pattern_similarity': self._compare_movement_patterns(orig_movements, gen_movements)
        }
    
    def _analyze_timing_patterns(
        self,
        original: List[Dict[str, Any]],
        generated: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze timing patterns between replays."""
        orig_times = [event.get('time', i) for i, event in enumerate(original)]
        gen_times = [event.get('time', i) for i, event in enumerate(generated)]
        
        orig_intervals = np.diff(orig_times) if len(orig_times) > 1 else [0]
        gen_intervals = np.diff(gen_times) if len(gen_times) > 1 else [0]
        
        return {
            'original_timing': {
                'mean_interval': np.mean(orig_intervals),
                'std_interval': np.std(orig_intervals),
                'total_duration': max(orig_times) - min(orig_times) if orig_times else 0
            },
            'generated_timing': {
                'mean_interval': np.mean(gen_intervals),
                'std_interval': np.std(gen_intervals),
                'total_duration': max(gen_times) - min(gen_times) if gen_times else 0
            }
        }
    
    def _analyze_accuracy_patterns(
        self,
        original: List[Dict[str, Any]],
        generated: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze accuracy patterns between replays."""
        # This would require hit object data to be meaningful
        # For now, return basic accuracy metrics
        return {
            'position_accuracy': self._calculate_position_accuracy(original, generated),
            'timing_accuracy': self._calculate_timing_accuracy(original, generated)
        }
    
    def _calculate_total_distance(self, x_coords: List[float], y_coords: List[float]) -> float:
        """Calculate total cursor distance traveled."""
        if len(x_coords) < 2:
            return 0.0
            
        total = 0.0
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            total += np.sqrt(dx**2 + dy**2)
        
        return total
    
    def _calculate_average_speed(self, x_coords: List[float], y_coords: List[float], times: List[float]) -> float:
        """Calculate average cursor speed."""
        speeds = self._calculate_speeds(x_coords, y_coords, times)
        return np.mean(speeds) if speeds else 0.0
    
    def _calculate_max_speed(self, x_coords: List[float], y_coords: List[float], times: List[float]) -> float:
        """Calculate maximum cursor speed."""
        speeds = self._calculate_speeds(x_coords, y_coords, times)
        return np.max(speeds) if speeds else 0.0
    
    def _calculate_speeds(self, x_coords: List[float], y_coords: List[float], times: List[float]) -> List[float]:
        """Calculate cursor speeds between consecutive points."""
        if len(x_coords) < 2:
            return []
            
        speeds = []
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            dt = times[i] - times[i-1] if i < len(times) else 1
            
            if dt > 0:
                distance = np.sqrt(dx**2 + dy**2)
                speed = distance / dt
                speeds.append(speed)
            else:
                speeds.append(0.0)
        
        return speeds
    
    def _calculate_cursor_coverage(self, x_coords: List[float], y_coords: List[float]) -> Dict[str, float]:
        """Calculate cursor coverage statistics."""
        if not x_coords or not y_coords:
            return {'coverage_area': 0.0, 'coverage_ratio': 0.0}
            
        # Calculate bounding box
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        coverage_area = (max_x - min_x) * (max_y - min_y)
        total_area = 512 * 384  # osu! playfield size
        coverage_ratio = coverage_area / total_area
        
        return {
            'coverage_area': coverage_area,
            'coverage_ratio': coverage_ratio
        }
    
    def _plot_trajectory(self, x_coords: List[float], y_coords: List[float], save_path: Path) -> None:
        """Plot cursor trajectory."""
        plt.figure(figsize=(10, 8))
        plt.plot(x_coords, y_coords, alpha=0.7, linewidth=1)
        plt.scatter(x_coords[0], y_coords[0], color='green', s=50, label='Start')
        plt.scatter(x_coords[-1], y_coords[-1], color='red', s=50, label='End')
        plt.xlim(0, 512)
        plt.ylim(0, 384)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Cursor Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_speed_distribution(self, speeds: List[float], save_path: Path) -> None:
        """Plot speed distribution."""
        plt.figure(figsize=(10, 6))
        plt.hist(speeds, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Speed (pixels/ms)')
        plt.ylabel('Frequency')
        plt.title('Cursor Speed Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _detect_anomalies(self, x_coords: List[float], y_coords: List[float], times: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in replay data."""
        anomalies = []
        
        # Detect sudden position jumps
        for i in range(1, len(x_coords)):
            dx = abs(x_coords[i] - x_coords[i-1])
            dy = abs(y_coords[i] - y_coords[i-1])
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 200:  # Threshold for sudden jump
                anomalies.append({
                    'type': 'position_jump',
                    'index': i,
                    'distance': distance,
                    'position': (x_coords[i], y_coords[i])
                })
        
        return anomalies
    
    def _identify_patterns(self, x_coords: List[float], y_coords: List[float], times: List[float]) -> List[str]:
        """Identify movement patterns."""
        patterns = []
        
        # Check for circular movements
        if self._detect_circular_pattern(x_coords, y_coords):
            patterns.append('circular_movement')
        
        # Check for linear movements
        if self._detect_linear_pattern(x_coords, y_coords):
            patterns.append('linear_movement')
        
        # Check for repetitive patterns
        if self._detect_repetitive_pattern(x_coords, y_coords):
            patterns.append('repetitive_movement')
        
        return patterns
    
    def _detect_circular_pattern(self, x_coords: List[float], y_coords: List[float]) -> bool:
        """Detect circular movement patterns."""
        # Simple heuristic: check if points form roughly circular shape
        if len(x_coords) < 10:
            return False
            
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in zip(x_coords, y_coords)]
        std_distance = np.std(distances)
        mean_distance = np.mean(distances)
        
        # If standard deviation is small relative to mean, it might be circular
        return std_distance / mean_distance < 0.3 if mean_distance > 0 else False
    
    def _detect_linear_pattern(self, x_coords: List[float], y_coords: List[float]) -> bool:
        """Detect linear movement patterns."""
        if len(x_coords) < 3:
            return False
            
        # Calculate correlation coefficient
        correlation = np.corrcoef(x_coords, y_coords)[0, 1]
        return abs(correlation) > 0.8
    
    def _detect_repetitive_pattern(self, x_coords: List[float], y_coords: List[float]) -> bool:
        """Detect repetitive movement patterns."""
        # Simple check for repeated sequences
        if len(x_coords) < 20:
            return False
            
        # Check for repeated subsequences
        window_size = 5
        for i in range(len(x_coords) - 2 * window_size):
            seq1_x = x_coords[i:i + window_size]
            seq1_y = y_coords[i:i + window_size]
            
            for j in range(i + window_size, len(x_coords) - window_size):
                seq2_x = x_coords[j:j + window_size]
                seq2_y = y_coords[j:j + window_size]
                
                # Check similarity
                diff_x = np.mean([abs(a - b) for a, b in zip(seq1_x, seq2_x)])
                diff_y = np.mean([abs(a - b) for a, b in zip(seq1_y, seq2_y)])
                
                if diff_x < 10 and diff_y < 10:  # Threshold for similarity
                    return True
        
        return False
    
    def _extract_movement_features(self, replay_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract movement features from replay data."""
        x_coords = [event.get('x', 0) for event in replay_data]
        y_coords = [event.get('y', 0) for event in replay_data]
        times = [event.get('time', i) for i, event in enumerate(replay_data)]
        
        speeds = self._calculate_speeds(x_coords, y_coords, times)
        
        return {
            'mean_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'speed_variance': np.var(speeds) if speeds else 0,
            'total_distance': self._calculate_total_distance(x_coords, y_coords),
            'movement_efficiency': self._calculate_movement_efficiency(x_coords, y_coords)
        }
    
    def _compare_movement_patterns(self, orig_patterns: Dict[str, Any], gen_patterns: Dict[str, Any]) -> float:
        """Compare movement patterns between original and generated replays."""
        # Simple similarity score based on feature differences
        features = ['mean_speed', 'max_speed', 'speed_variance', 'total_distance', 'movement_efficiency']
        
        similarities = []
        for feature in features:
            orig_val = orig_patterns.get(feature, 0)
            gen_val = gen_patterns.get(feature, 0)
            
            if orig_val == 0 and gen_val == 0:
                similarity = 1.0
            elif orig_val == 0 or gen_val == 0:
                similarity = 0.0
            else:
                similarity = 1.0 - abs(orig_val - gen_val) / max(abs(orig_val), abs(gen_val))
            
            similarities.append(max(0, similarity))
        
        return np.mean(similarities)
    
    def _calculate_movement_efficiency(self, x_coords: List[float], y_coords: List[float]) -> float:
        """Calculate movement efficiency (straight line distance / actual distance)."""
        if len(x_coords) < 2:
            return 1.0
            
        # Straight line distance from start to end
        straight_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
        
        # Actual distance traveled
        actual_distance = self._calculate_total_distance(x_coords, y_coords)
        
        if actual_distance == 0:
            return 1.0
            
        return straight_distance / actual_distance
    
    def _calculate_position_accuracy(self, original: List[Dict[str, Any]], generated: List[Dict[str, Any]]) -> float:
        """Calculate position accuracy between replays."""
        # Simple position-based accuracy
        orig_x = [event.get('x', 0) for event in original]
        orig_y = [event.get('y', 0) for event in original]
        gen_x = [event.get('x', 0) for event in generated]
        gen_y = [event.get('y', 0) for event in generated]
        
        # Align sequences
        min_len = min(len(orig_x), len(gen_x))
        if min_len == 0:
            return 0.0
            
        orig_x = orig_x[:min_len]
        orig_y = orig_y[:min_len]
        gen_x = gen_x[:min_len]
        gen_y = gen_y[:min_len]
        
        # Calculate average distance
        distances = [np.sqrt((ox - gx)**2 + (oy - gy)**2) for ox, oy, gx, gy in zip(orig_x, orig_y, gen_x, gen_y)]
        avg_distance = np.mean(distances)
        
        # Convert to accuracy score (lower distance = higher accuracy)
        max_distance = 100  # Maximum reasonable distance for "accurate" movement
        accuracy = max(0, 1 - avg_distance / max_distance)
        
        return accuracy
    
    def _calculate_timing_accuracy(self, original: List[Dict[str, Any]], generated: List[Dict[str, Any]]) -> float:
        """Calculate timing accuracy between replays."""
        orig_times = [event.get('time', i) for i, event in enumerate(original)]
        gen_times = [event.get('time', i) for i, event in enumerate(generated)]
        
        if not orig_times or not gen_times:
            return 0.0
            
        # Calculate timing intervals
        orig_intervals = np.diff(orig_times) if len(orig_times) > 1 else [0]
        gen_intervals = np.diff(gen_times) if len(gen_times) > 1 else [0]
        
        # Align intervals
        min_len = min(len(orig_intervals), len(gen_intervals))
        if min_len == 0:
            return 0.0
            
        orig_intervals = orig_intervals[:min_len]
        gen_intervals = gen_intervals[:min_len]
        
        # Calculate timing differences
        timing_diffs = [abs(oi - gi) for oi, gi in zip(orig_intervals, gen_intervals)]
        avg_timing_diff = np.mean(timing_diffs)
        
        # Convert to accuracy score
        max_timing_diff = 50  # Maximum reasonable timing difference (ms)
        accuracy = max(0, 1 - avg_timing_diff / max_timing_diff)
        
        return accuracy