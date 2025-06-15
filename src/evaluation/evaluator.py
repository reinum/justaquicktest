"""Main evaluation module for assessing replay quality and model performance."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

from ..generation.generator import GenerationResult
from ..data.dataset import OsuReplayDataset


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Cursor movement metrics
    cursor_mse: float = 0.0
    cursor_mae: float = 0.0
    cursor_smoothness: float = 0.0
    trajectory_similarity: float = 0.0
    velocity_correlation: float = 0.0
    
    # Key press metrics
    key_accuracy: float = 0.0
    key_precision: float = 0.0
    key_recall: float = 0.0
    key_f1: float = 0.0
    timing_accuracy: float = 0.0
    
    # Gameplay metrics
    hit_accuracy: float = 0.0
    rhythm_consistency: float = 0.0
    flow_quality: float = 0.0
    difficulty_consistency: float = 0.0
    
    # Performance metrics
    generation_time: float = 0.0
    memory_usage: float = 0.0
    model_confidence: float = 0.0
    
    # Overall scores
    technical_score: float = 0.0
    gameplay_score: float = 0.0
    overall_score: float = 0.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    metrics: EvaluationMetrics
    detailed_analysis: Dict[str, Any]
    comparison_data: Optional[Dict[str, Any]] = None
    visualizations: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'metrics': self.metrics.__dict__,
            'detailed_analysis': self.detailed_analysis,
            'comparison_data': self.comparison_data,
            'recommendations': self.recommendations
        }
    
    def save(self, path: Union[str, Path]):
        """Save evaluation result to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ReplayEvaluator:
    """Comprehensive evaluator for generated replays."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Evaluation weights
        self.weights = {
            'cursor_movement': 0.3,
            'key_timing': 0.25,
            'gameplay_quality': 0.25,
            'technical_performance': 0.2
        }
    
    def evaluate(self, 
                generated_replay: GenerationResult,
                reference_replay: Optional[GenerationResult] = None,
                beatmap_data: Optional[Dict[str, Any]] = None,
                detailed: bool = True) -> EvaluationResult:
        """Evaluate a generated replay comprehensively.
        
        Args:
            generated_replay: The generated replay to evaluate
            reference_replay: Optional reference replay for comparison
            beatmap_data: Beatmap information for context
            detailed: Whether to perform detailed analysis
            
        Returns:
            Comprehensive evaluation result
        """
        self.logger.info("Starting replay evaluation")
        
        metrics = EvaluationMetrics()
        detailed_analysis = {}
        comparison_data = None
        
        # Evaluate cursor movement
        cursor_metrics = self._evaluate_cursor_movement(
            generated_replay, reference_replay, beatmap_data
        )
        metrics.cursor_mse = cursor_metrics['mse']
        metrics.cursor_mae = cursor_metrics['mae']
        metrics.cursor_smoothness = cursor_metrics['smoothness']
        metrics.trajectory_similarity = cursor_metrics['trajectory_similarity']
        metrics.velocity_correlation = cursor_metrics['velocity_correlation']
        
        if detailed:
            detailed_analysis['cursor_movement'] = cursor_metrics
        
        # Evaluate key presses
        key_metrics = self._evaluate_key_presses(
            generated_replay, reference_replay, beatmap_data
        )
        metrics.key_accuracy = key_metrics['accuracy']
        metrics.key_precision = key_metrics['precision']
        metrics.key_recall = key_metrics['recall']
        metrics.key_f1 = key_metrics['f1']
        metrics.timing_accuracy = key_metrics['timing_accuracy']
        
        if detailed:
            detailed_analysis['key_presses'] = key_metrics
        
        # Evaluate gameplay quality
        gameplay_metrics = self._evaluate_gameplay_quality(
            generated_replay, beatmap_data
        )
        metrics.hit_accuracy = gameplay_metrics['hit_accuracy']
        metrics.rhythm_consistency = gameplay_metrics['rhythm_consistency']
        metrics.flow_quality = gameplay_metrics['flow_quality']
        metrics.difficulty_consistency = gameplay_metrics['difficulty_consistency']
        
        if detailed:
            detailed_analysis['gameplay_quality'] = gameplay_metrics
        
        # Evaluate technical performance
        tech_metrics = self._evaluate_technical_performance(generated_replay)
        metrics.generation_time = tech_metrics['generation_time']
        metrics.memory_usage = tech_metrics['memory_usage']
        metrics.model_confidence = tech_metrics['model_confidence']
        
        if detailed:
            detailed_analysis['technical_performance'] = tech_metrics
        
        # Calculate overall scores
        metrics.technical_score = self._calculate_technical_score(metrics)
        metrics.gameplay_score = self._calculate_gameplay_score(metrics)
        metrics.overall_score = self._calculate_overall_score(metrics)
        
        # Generate comparison data if reference is available
        if reference_replay is not None:
            comparison_data = self._generate_comparison_data(
                generated_replay, reference_replay, beatmap_data
            )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, detailed_analysis)
        
        self.logger.info(f"Evaluation completed. Overall score: {metrics.overall_score:.3f}")
        
        return EvaluationResult(
            metrics=metrics,
            detailed_analysis=detailed_analysis,
            comparison_data=comparison_data,
            recommendations=recommendations
        )
    
    def _evaluate_cursor_movement(self, 
                                 generated: GenerationResult,
                                 reference: Optional[GenerationResult],
                                 beatmap_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate cursor movement quality."""
        cursor_pos = generated.cursor_positions
        
        metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'smoothness': 0.0,
            'trajectory_similarity': 0.0,
            'velocity_correlation': 0.0,
            'acceleration_consistency': 0.0,
            'path_efficiency': 0.0
        }
        
        # Calculate smoothness (lower is better)
        if len(cursor_pos) > 2:
            # Second derivative (acceleration)
            acceleration = np.diff(cursor_pos, n=2, axis=0)
            acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
            metrics['smoothness'] = 1.0 / (1.0 + np.std(acceleration_magnitude))
            metrics['acceleration_consistency'] = 1.0 - np.std(acceleration_magnitude) / (np.mean(acceleration_magnitude) + 1e-8)
        
        # Calculate velocity statistics
        if len(cursor_pos) > 1:
            velocities = np.diff(cursor_pos, axis=0)
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            
            # Path efficiency (straight line distance vs actual path)
            if len(cursor_pos) > 1:
                straight_distance = euclidean(cursor_pos[0], cursor_pos[-1])
                actual_distance = np.sum(velocity_magnitudes)
                metrics['path_efficiency'] = straight_distance / (actual_distance + 1e-8)
        
        # Compare with reference if available
        if reference is not None:
            ref_cursor = reference.cursor_positions
            
            # Align sequences if different lengths
            min_len = min(len(cursor_pos), len(ref_cursor))
            gen_aligned = cursor_pos[:min_len]
            ref_aligned = ref_cursor[:min_len]
            
            # Calculate MSE and MAE
            metrics['mse'] = mean_squared_error(ref_aligned.flatten(), gen_aligned.flatten())
            metrics['mae'] = mean_absolute_error(ref_aligned.flatten(), gen_aligned.flatten())
            
            # Trajectory similarity using DTW or correlation
            if len(gen_aligned) > 1 and len(ref_aligned) > 1:
                gen_vel = np.diff(gen_aligned, axis=0)
                ref_vel = np.diff(ref_aligned, axis=0)
                
                gen_vel_mag = np.linalg.norm(gen_vel, axis=1)
                ref_vel_mag = np.linalg.norm(ref_vel, axis=1)
                
                if len(gen_vel_mag) > 0 and len(ref_vel_mag) > 0:
                    correlation, _ = stats.pearsonr(gen_vel_mag, ref_vel_mag)
                    metrics['velocity_correlation'] = max(0, correlation)
                    
                    # Trajectory similarity using cosine similarity
                    trajectory_sim = np.mean([
                        np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-8)
                        for g, r in zip(gen_vel, ref_vel)
                        if np.linalg.norm(g) > 1e-8 and np.linalg.norm(r) > 1e-8
                    ])
                    metrics['trajectory_similarity'] = max(0, trajectory_sim)
        
        # Evaluate against beatmap if available
        if beatmap_data is not None:
            hit_objects = beatmap_data.get('hit_objects', [])
            if hit_objects:
                # Check if cursor follows expected patterns
                pattern_score = self._evaluate_cursor_patterns(cursor_pos, hit_objects)
                metrics['pattern_adherence'] = pattern_score
        
        return metrics
    
    def _evaluate_key_presses(self,
                             generated: GenerationResult,
                             reference: Optional[GenerationResult],
                             beatmap_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate key press accuracy and timing."""
        key_presses = generated.key_presses
        timestamps = generated.timestamps
        
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'timing_accuracy': 0.0,
            'rhythm_consistency': 0.0,
            'alternation_quality': 0.0
        }
        
        # Analyze key press patterns
        if len(key_presses) > 0:
            # Calculate key press statistics
            total_presses = np.sum(key_presses[:, :2])  # Only keyboard keys
            
            if total_presses > 0:
                # Alternation analysis
                k1_presses = key_presses[:, 0] > 0.5
                k2_presses = key_presses[:, 1] > 0.5
                
                alternation_score = self._calculate_alternation_score(k1_presses, k2_presses)
                metrics['alternation_quality'] = alternation_score
                
                # Rhythm consistency
                press_intervals = self._get_key_press_intervals(key_presses, timestamps)
                if len(press_intervals) > 1:
                    rhythm_score = 1.0 - (np.std(press_intervals) / (np.mean(press_intervals) + 1e-8))
                    metrics['rhythm_consistency'] = max(0, rhythm_score)
        
        # Compare with reference
        if reference is not None:
            ref_keys = reference.key_presses
            
            # Align sequences
            min_len = min(len(key_presses), len(ref_keys))
            gen_aligned = key_presses[:min_len]
            ref_aligned = ref_keys[:min_len]
            
            # Calculate classification metrics
            for key_idx in range(min(gen_aligned.shape[1], ref_aligned.shape[1])):
                gen_key = gen_aligned[:, key_idx] > 0.5
                ref_key = ref_aligned[:, key_idx] > 0.5
                
                # Calculate precision, recall, F1
                tp = np.sum(gen_key & ref_key)
                fp = np.sum(gen_key & ~ref_key)
                fn = np.sum(~gen_key & ref_key)
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                metrics['precision'] += precision
                metrics['recall'] += recall
                metrics['f1'] += f1
            
            # Average across keys
            num_keys = min(gen_aligned.shape[1], ref_aligned.shape[1])
            if num_keys > 0:
                metrics['precision'] /= num_keys
                metrics['recall'] /= num_keys
                metrics['f1'] /= num_keys
            
            # Overall accuracy
            accuracy = np.mean(gen_aligned == ref_aligned)
            metrics['accuracy'] = accuracy
        
        # Evaluate timing against beatmap
        if beatmap_data is not None:
            hit_objects = beatmap_data.get('hit_objects', [])
            if hit_objects:
                timing_score = self._evaluate_key_timing(key_presses, timestamps, hit_objects)
                metrics['timing_accuracy'] = timing_score
        
        return metrics
    
    def _evaluate_gameplay_quality(self,
                                  generated: GenerationResult,
                                  beatmap_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate overall gameplay quality."""
        metrics = {
            'hit_accuracy': 0.0,
            'rhythm_consistency': 0.0,
            'flow_quality': 0.0,
            'difficulty_consistency': 0.0,
            'style_consistency': 0.0,
            'humanness': 0.0
        }
        
        if beatmap_data is None:
            return metrics
        
        cursor_pos = generated.cursor_positions
        key_presses = generated.key_presses
        timestamps = generated.timestamps
        hit_objects = beatmap_data.get('hit_objects', [])
        
        if not hit_objects:
            return metrics
        
        # Evaluate hit accuracy
        hit_accuracy = self._calculate_hit_accuracy(cursor_pos, timestamps, hit_objects)
        metrics['hit_accuracy'] = hit_accuracy
        
        # Evaluate flow quality (smooth transitions between objects)
        flow_score = self._calculate_flow_quality(cursor_pos, timestamps, hit_objects)
        metrics['flow_quality'] = flow_score
        
        # Evaluate difficulty consistency
        difficulty_score = self._evaluate_difficulty_consistency(
            cursor_pos, key_presses, timestamps, beatmap_data
        )
        metrics['difficulty_consistency'] = difficulty_score
        
        # Evaluate humanness (how human-like the play is)
        humanness_score = self._calculate_humanness_score(
            cursor_pos, key_presses, timestamps
        )
        metrics['humanness'] = humanness_score
        
        return metrics
    
    def _evaluate_technical_performance(self, generated: GenerationResult) -> Dict[str, float]:
        """Evaluate technical aspects of generation."""
        metrics = {
            'generation_time': generated.generation_time,
            'memory_usage': 0.0,  # Would need to be tracked during generation
            'model_confidence': 0.0,
            'sequence_length': len(generated.timestamps),
            'data_quality': 0.0
        }
        
        # Calculate average confidence if available
        if generated.confidence_scores is not None:
            metrics['model_confidence'] = np.mean(generated.confidence_scores)
        
        # Evaluate data quality
        data_quality = self._evaluate_data_quality(generated)
        metrics['data_quality'] = data_quality
        
        return metrics
    
    def _calculate_technical_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate technical performance score."""
        # Normalize and combine technical metrics
        smoothness_score = metrics.cursor_smoothness
        confidence_score = metrics.model_confidence
        timing_score = metrics.timing_accuracy
        
        # Weight the components
        technical_score = (
            smoothness_score * 0.4 +
            confidence_score * 0.3 +
            timing_score * 0.3
        )
        
        return max(0, min(1, technical_score))
    
    def _calculate_gameplay_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate gameplay quality score."""
        # Combine gameplay-related metrics
        hit_score = metrics.hit_accuracy
        rhythm_score = metrics.rhythm_consistency
        flow_score = metrics.flow_quality
        key_score = (metrics.key_precision + metrics.key_recall) / 2
        
        gameplay_score = (
            hit_score * 0.3 +
            rhythm_score * 0.25 +
            flow_score * 0.25 +
            key_score * 0.2
        )
        
        return max(0, min(1, gameplay_score))
    
    def _calculate_overall_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall evaluation score."""
        # Weighted combination of all aspects
        cursor_score = (metrics.cursor_smoothness + metrics.trajectory_similarity) / 2
        key_score = metrics.key_f1
        gameplay_score = metrics.gameplay_score
        technical_score = metrics.technical_score
        
        overall_score = (
            cursor_score * self.weights['cursor_movement'] +
            key_score * self.weights['key_timing'] +
            gameplay_score * self.weights['gameplay_quality'] +
            technical_score * self.weights['technical_performance']
        )
        
        return max(0, min(1, overall_score))
    
    def _generate_comparison_data(self,
                                 generated: GenerationResult,
                                 reference: GenerationResult,
                                 beatmap_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed comparison data."""
        comparison = {
            'sequence_lengths': {
                'generated': len(generated.timestamps),
                'reference': len(reference.timestamps)
            },
            'timing_comparison': {},
            'spatial_comparison': {},
            'key_comparison': {}
        }
        
        # Add detailed comparisons here
        # This would include statistical tests, visualizations, etc.
        
        return comparison
    
    def _generate_recommendations(self,
                                 metrics: EvaluationMetrics,
                                 detailed_analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on evaluation."""
        recommendations = []
        
        # Cursor movement recommendations
        if metrics.cursor_smoothness < 0.7:
            recommendations.append("Consider improving cursor smoothness with better trajectory planning")
        
        if metrics.trajectory_similarity < 0.6:
            recommendations.append("Work on trajectory similarity - the generated path differs significantly from reference")
        
        # Key press recommendations
        if metrics.key_f1 < 0.8:
            recommendations.append("Improve key press prediction accuracy")
        
        if metrics.timing_accuracy < 0.7:
            recommendations.append("Focus on timing accuracy - key presses are not well-aligned with beat")
        
        # Gameplay recommendations
        if metrics.hit_accuracy < 0.8:
            recommendations.append("Improve hit accuracy by better cursor positioning")
        
        if metrics.rhythm_consistency < 0.6:
            recommendations.append("Work on rhythm consistency for more natural gameplay")
        
        # Technical recommendations
        if metrics.generation_time > 10.0:
            recommendations.append("Consider optimizing model for faster generation")
        
        if metrics.model_confidence < 0.7:
            recommendations.append("Model confidence is low - consider more training or better architecture")
        
        return recommendations
    
    # Helper methods for specific calculations
    def _calculate_alternation_score(self, k1_presses: np.ndarray, k2_presses: np.ndarray) -> float:
        """Calculate how well keys alternate."""
        if len(k1_presses) < 2:
            return 0.0
        
        # Find press events
        k1_events = np.where(np.diff(k1_presses.astype(int)) > 0)[0]
        k2_events = np.where(np.diff(k2_presses.astype(int)) > 0)[0]
        
        if len(k1_events) == 0 or len(k2_events) == 0:
            return 0.0
        
        # Calculate alternation ratio
        total_events = len(k1_events) + len(k2_events)
        alternations = 0
        
        all_events = sorted([(t, 0) for t in k1_events] + [(t, 1) for t in k2_events])
        
        for i in range(1, len(all_events)):
            if all_events[i][1] != all_events[i-1][1]:
                alternations += 1
        
        return alternations / max(1, total_events - 1)
    
    def _get_key_press_intervals(self, key_presses: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Get intervals between key presses."""
        press_times = []
        
        for i in range(len(key_presses) - 1):
            current_pressed = np.any(key_presses[i, :2] > 0.5)
            next_pressed = np.any(key_presses[i+1, :2] > 0.5)
            
            if not current_pressed and next_pressed:
                press_times.append(timestamps[i+1])
        
        if len(press_times) > 1:
            return np.diff(press_times)
        else:
            return np.array([])
    
    def _calculate_hit_accuracy(self, cursor_pos: np.ndarray, 
                               timestamps: np.ndarray,
                               hit_objects: List[Dict[str, Any]]) -> float:
        """Calculate how accurately hits are performed."""
        if len(hit_objects) == 0:
            return 0.0
        
        hit_scores = []
        
        for hit_obj in hit_objects:
            hit_time = hit_obj['time']
            hit_pos = np.array([hit_obj['x'], hit_obj['y']])
            
            # Find closest timestamp
            time_diffs = np.abs(timestamps - hit_time)
            closest_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_idx] <= 50:  # Within 50ms
                cursor_at_hit = cursor_pos[closest_idx]
                distance = euclidean(cursor_at_hit, hit_pos)
                
                # Score based on distance (closer is better)
                hit_score = max(0, 1 - distance / 100)  # 100px = 0 score
                hit_scores.append(hit_score)
        
        return np.mean(hit_scores) if hit_scores else 0.0
    
    def _calculate_flow_quality(self, cursor_pos: np.ndarray,
                               timestamps: np.ndarray,
                               hit_objects: List[Dict[str, Any]]) -> float:
        """Calculate flow quality between hit objects."""
        if len(hit_objects) < 2:
            return 0.0
        
        flow_scores = []
        
        for i in range(len(hit_objects) - 1):
            obj1 = hit_objects[i]
            obj2 = hit_objects[i + 1]
            
            # Get cursor path between objects
            start_time = obj1['time']
            end_time = obj2['time']
            
            mask = (timestamps >= start_time) & (timestamps <= end_time)
            path_segment = cursor_pos[mask]
            
            if len(path_segment) > 1:
                # Calculate path efficiency
                start_pos = np.array([obj1['x'], obj1['y']])
                end_pos = np.array([obj2['x'], obj2['y']])
                
                straight_distance = euclidean(start_pos, end_pos)
                actual_path = np.sum([euclidean(path_segment[j], path_segment[j+1]) 
                                    for j in range(len(path_segment)-1)])
                
                if actual_path > 0:
                    efficiency = straight_distance / actual_path
                    flow_scores.append(min(1.0, efficiency))
        
        return np.mean(flow_scores) if flow_scores else 0.0
    
    def _evaluate_difficulty_consistency(self, cursor_pos: np.ndarray,
                                       key_presses: np.ndarray,
                                       timestamps: np.ndarray,
                                       beatmap_data: Dict[str, Any]) -> float:
        """Evaluate if performance matches beatmap difficulty."""
        # This is a placeholder for difficulty-based evaluation
        # In practice, you'd analyze speed, accuracy, and complexity
        # relative to the beatmap's star rating and patterns
        
        difficulty = beatmap_data.get('difficulty_rating', 5.0)
        
        # Calculate performance metrics
        if len(cursor_pos) > 1:
            velocities = np.diff(cursor_pos, axis=0)
            avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
            
            # Expected speed based on difficulty
            expected_speed = difficulty * 10  # Rough approximation
            
            speed_consistency = 1.0 - abs(avg_speed - expected_speed) / expected_speed
            return max(0, speed_consistency)
        
        return 0.5  # Neutral score
    
    def _calculate_humanness_score(self, cursor_pos: np.ndarray,
                                  key_presses: np.ndarray,
                                  timestamps: np.ndarray) -> float:
        """Calculate how human-like the replay appears."""
        humanness_factors = []
        
        # Cursor movement humanness
        if len(cursor_pos) > 2:
            # Check for natural acceleration/deceleration
            velocities = np.diff(cursor_pos, axis=0)
            accelerations = np.diff(velocities, axis=0)
            
            # Human movement typically has smooth acceleration curves
            acc_smoothness = 1.0 / (1.0 + np.std(np.linalg.norm(accelerations, axis=1)))
            humanness_factors.append(acc_smoothness)
            
            # Check for micro-corrections (small adjustments)
            small_movements = np.sum(np.linalg.norm(velocities, axis=1) < 5) / len(velocities)
            humanness_factors.append(min(1.0, small_movements * 2))  # Some micro-movements expected
        
        # Key press humanness
        if len(key_presses) > 0:
            # Check for natural key press patterns
            press_intervals = self._get_key_press_intervals(key_presses, timestamps)
            if len(press_intervals) > 1:
                # Human timing has some natural variation
                timing_variation = np.std(press_intervals) / (np.mean(press_intervals) + 1e-8)
                natural_variation = min(1.0, timing_variation * 10)  # Some variation is good
                humanness_factors.append(natural_variation)
        
        return np.mean(humanness_factors) if humanness_factors else 0.5
    
    def _evaluate_data_quality(self, generated: GenerationResult) -> float:
        """Evaluate the quality of generated data."""
        quality_factors = []
        
        # Check for valid ranges
        cursor_pos = generated.cursor_positions
        if len(cursor_pos) > 0:
            # Cursor should be within screen bounds
            x_valid = np.all((cursor_pos[:, 0] >= 0) & (cursor_pos[:, 0] <= 512))
            y_valid = np.all((cursor_pos[:, 1] >= 0) & (cursor_pos[:, 1] <= 384))
            quality_factors.append(float(x_valid and y_valid))
        
        # Check timestamp monotonicity
        timestamps = generated.timestamps
        if len(timestamps) > 1:
            monotonic = np.all(np.diff(timestamps) >= 0)
            quality_factors.append(float(monotonic))
        
        # Check for reasonable frame rates
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            reasonable_intervals = np.all((intervals >= 1) & (intervals <= 100))  # 1-100ms
            quality_factors.append(float(reasonable_intervals))
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _evaluate_cursor_patterns(self, cursor_pos: np.ndarray, 
                                 hit_objects: List[Dict[str, Any]]) -> float:
        """Evaluate how well cursor follows expected patterns."""
        if len(hit_objects) < 2:
            return 0.5
        
        pattern_scores = []
        
        # Check if cursor moves towards hit objects
        for i, hit_obj in enumerate(hit_objects):
            hit_time = hit_obj['time']
            hit_pos = np.array([hit_obj['x'], hit_obj['y']])
            
            # Find cursor positions around hit time
            # This is a simplified check - in practice you'd want more sophisticated analysis
            
            pattern_scores.append(0.7)  # Placeholder score
        
        return np.mean(pattern_scores) if pattern_scores else 0.5
    
    def _evaluate_key_timing(self, key_presses: np.ndarray,
                           timestamps: np.ndarray,
                           hit_objects: List[Dict[str, Any]]) -> float:
        """Evaluate key press timing against hit objects."""
        if len(hit_objects) == 0:
            return 0.0
        
        timing_scores = []
        
        for hit_obj in hit_objects:
            hit_time = hit_obj['time']
            
            # Find key presses around hit time
            time_window = 50  # 50ms window
            mask = np.abs(timestamps - hit_time) <= time_window
            
            if np.any(mask):
                keys_in_window = key_presses[mask]
                
                # Check if any key was pressed
                if np.any(keys_in_window[:, :2] > 0.5):  # Keyboard keys
                    # Calculate timing accuracy
                    press_times = timestamps[mask][np.any(keys_in_window[:, :2] > 0.5, axis=1)]
                    if len(press_times) > 0:
                        closest_press = press_times[np.argmin(np.abs(press_times - hit_time))]
                        timing_error = abs(closest_press - hit_time)
                        timing_score = max(0, 1 - timing_error / time_window)
                        timing_scores.append(timing_score)
        
        return np.mean(timing_scores) if timing_scores else 0.0