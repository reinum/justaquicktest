"""Metrics for evaluating osu! replay generation performance."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class ReplayMetrics:
    """Comprehensive metrics for replay generation evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.cursor_errors = []
        self.key_accuracies = []
        self.timing_errors = []
        self.smoothness_scores = []
        self.trajectory_similarities = []
        self.velocity_correlations = []
    
    def update(self, predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor],
               mask: Optional[torch.Tensor] = None):
        """Update metrics with new predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            mask: Optional mask for valid positions
        """
        # Convert to numpy for easier computation
        pred_cursor = predictions['cursor_pred'].detach().cpu().numpy()
        target_cursor = targets['cursor_data'].transpose(0, 1).detach().cpu().numpy()  # (batch_size, seq_len) -> (seq_len, batch_size)
        pred_keys = predictions['key_pred'].detach().cpu().numpy()
        target_keys = targets['key_data'].transpose(0, 1).detach().cpu().numpy()  # (batch_size, seq_len) -> (seq_len, batch_size)
        
        if mask is not None:
            mask_np = mask.transpose(0, 1).detach().cpu().numpy()  # (batch_size, seq_len) -> (seq_len, batch_size)
        else:
            mask_np = np.ones(pred_cursor.shape[:2], dtype=bool)
        
        # Compute metrics for each sequence in batch
        batch_size = pred_cursor.shape[1]
        for i in range(batch_size):
            seq_mask = mask_np[:, i]
            
            # Cursor metrics
            cursor_error = self._compute_cursor_error(
                pred_cursor[:, i], target_cursor[:, i], seq_mask
            )
            self.cursor_errors.append(cursor_error)
            
            # Key metrics
            key_accuracy = self._compute_key_accuracy(
                pred_keys[:, i], target_keys[:, i], seq_mask
            )
            self.key_accuracies.append(key_accuracy)
            
            # Trajectory metrics
            smoothness = self._compute_smoothness(
                pred_cursor[:, i], seq_mask
            )
            self.smoothness_scores.append(smoothness)
            
            similarity = self._compute_trajectory_similarity(
                pred_cursor[:, i], target_cursor[:, i], seq_mask
            )
            self.trajectory_similarities.append(similarity)
            
            velocity_corr = self._compute_velocity_correlation(
                pred_cursor[:, i], target_cursor[:, i], seq_mask
            )
            self.velocity_correlations.append(velocity_corr)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics.
        
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        if self.cursor_errors:
            metrics['cursor_mse'] = np.mean(self.cursor_errors)
            metrics['cursor_rmse'] = np.sqrt(metrics['cursor_mse'])
            metrics['cursor_mae'] = np.mean([np.sqrt(err) for err in self.cursor_errors])
        
        if self.key_accuracies:
            metrics['key_accuracy'] = np.mean(self.key_accuracies)
        
        if self.smoothness_scores:
            metrics['smoothness'] = np.mean(self.smoothness_scores)
        
        if self.trajectory_similarities:
            metrics['trajectory_similarity'] = np.mean(self.trajectory_similarities)
        
        if self.velocity_correlations:
            valid_corrs = [c for c in self.velocity_correlations if not np.isnan(c)]
            if valid_corrs:
                metrics['velocity_correlation'] = np.mean(valid_corrs)
        
        return metrics
    
    def _compute_cursor_error(self, pred: np.ndarray, target: np.ndarray, 
                             mask: np.ndarray) -> float:
        """Compute cursor position error (MSE)."""
        valid_pred = pred[mask]
        valid_target = target[mask]
        
        if len(valid_pred) == 0:
            return 0.0
        
        error = np.mean((valid_pred - valid_target) ** 2)
        return error
    
    def _compute_key_accuracy(self, pred: np.ndarray, target: np.ndarray,
                             mask: np.ndarray) -> float:
        """Compute key press accuracy."""
        valid_pred = pred[mask]
        valid_target = target[mask]
        
        if len(valid_pred) == 0:
            return 0.0
        
        # Convert logits to binary predictions
        pred_binary = (valid_pred > 0).astype(int)
        target_binary = valid_target.astype(int)
        
        # Compute accuracy for each key separately and average
        accuracies = []
        for key_idx in range(pred_binary.shape[1]):
            acc = accuracy_score(target_binary[:, key_idx], pred_binary[:, key_idx])
            accuracies.append(acc)
        
        return np.mean(accuracies)
    
    def _compute_smoothness(self, cursor: np.ndarray, mask: np.ndarray) -> float:
        """Compute cursor movement smoothness (lower is smoother)."""
        valid_cursor = cursor[mask]
        
        if len(valid_cursor) < 3:
            return 0.0
        
        # Compute second derivative (acceleration)
        velocity = np.diff(valid_cursor, axis=0)
        acceleration = np.diff(velocity, axis=0)
        
        # Smoothness is inverse of acceleration magnitude
        smoothness = np.mean(np.linalg.norm(acceleration, axis=1))
        return smoothness
    
    def _compute_trajectory_similarity(self, pred: np.ndarray, target: np.ndarray,
                                     mask: np.ndarray) -> float:
        """Compute trajectory similarity using DTW-like measure."""
        valid_pred = pred[mask]
        valid_target = target[mask]
        
        if len(valid_pred) < 2:
            return 1.0
        
        # Simple similarity based on point-wise distances
        distances = np.linalg.norm(valid_pred - valid_target, axis=1)
        similarity = np.exp(-np.mean(distances))  # Exponential decay
        
        return similarity
    
    def _compute_velocity_correlation(self, pred: np.ndarray, target: np.ndarray,
                                    mask: np.ndarray) -> float:
        """Compute correlation between velocity profiles."""
        valid_pred = pred[mask]
        valid_target = target[mask]
        
        if len(valid_pred) < 2:
            return 0.0
        
        # Compute velocities
        pred_velocity = np.diff(valid_pred, axis=0)
        target_velocity = np.diff(valid_target, axis=0)
        
        # Compute velocity magnitudes
        pred_speed = np.linalg.norm(pred_velocity, axis=1)
        target_speed = np.linalg.norm(target_velocity, axis=1)
        
        # Compute correlation
        if len(pred_speed) < 2 or np.std(pred_speed) == 0 or np.std(target_speed) == 0:
            return 0.0
        
        correlation, _ = pearsonr(pred_speed, target_speed)
        return correlation if not np.isnan(correlation) else 0.0


class AccuracyMetrics:
    """Metrics specifically for accuracy-conditioned generation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.target_accuracies = []
        self.achieved_accuracies = []
        self.accuracy_errors = []
    
    def update(self, target_accuracy: torch.Tensor, 
               generated_replay: Dict[str, torch.Tensor],
               beatmap_data: torch.Tensor):
        """Update accuracy metrics.
        
        Args:
            target_accuracy: Target accuracy values
            generated_replay: Generated replay data
            beatmap_data: Beatmap information for accuracy calculation
        """
        # Simulate accuracy calculation (simplified)
        # In practice, this would involve hit detection logic
        cursor_pred = generated_replay['cursor_pred']
        key_pred = generated_replay['key_pred']
        
        batch_size = target_accuracy.shape[0]
        for i in range(batch_size):
            target_acc = target_accuracy[i].item()
            
            # Simplified accuracy estimation
            # This should be replaced with proper hit detection
            achieved_acc = self._estimate_accuracy(
                cursor_pred[:, i], key_pred[:, i], beatmap_data[:, i]
            )
            
            self.target_accuracies.append(target_acc)
            self.achieved_accuracies.append(achieved_acc)
            self.accuracy_errors.append(abs(target_acc - achieved_acc))
    
    def compute(self) -> Dict[str, float]:
        """Compute accuracy metrics."""
        if not self.accuracy_errors:
            return {}
        
        return {
            'accuracy_mae': np.mean(self.accuracy_errors),
            'accuracy_correlation': pearsonr(self.target_accuracies, self.achieved_accuracies)[0]
            if len(self.target_accuracies) > 1 else 0.0
        }
    
    def _estimate_accuracy(self, cursor: torch.Tensor, keys: torch.Tensor,
                          beatmap: torch.Tensor) -> float:
        """Estimate achieved accuracy (simplified)."""
        # This is a placeholder - real implementation would need
        # proper hit detection logic based on osu! mechanics
        return 0.95  # Placeholder value


class TimingMetrics:
    """Metrics for timing accuracy."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.timing_errors = []
        self.rhythm_correlations = []
    
    def update(self, pred_timing: torch.Tensor, target_timing: torch.Tensor,
               mask: Optional[torch.Tensor] = None):
        """Update timing metrics."""
        pred_np = pred_timing.detach().cpu().numpy()
        target_np = target_timing.detach().cpu().numpy()
        
        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.ones(pred_np.shape[:2], dtype=bool)
        
        batch_size = pred_np.shape[1]
        for i in range(batch_size):
            seq_mask = mask_np[:, i]
            valid_pred = pred_np[seq_mask, i]
            valid_target = target_np[seq_mask, i]
            
            if len(valid_pred) > 1:
                # Timing error
                timing_error = np.mean(np.abs(valid_pred - valid_target))
                self.timing_errors.append(timing_error)
                
                # Rhythm correlation
                if len(valid_pred) > 2:
                    pred_intervals = np.diff(valid_pred.flatten())
                    target_intervals = np.diff(valid_target.flatten())
                    
                    if np.std(pred_intervals) > 0 and np.std(target_intervals) > 0:
                        corr, _ = pearsonr(pred_intervals, target_intervals)
                        if not np.isnan(corr):
                            self.rhythm_correlations.append(corr)
    
    def compute(self) -> Dict[str, float]:
        """Compute timing metrics."""
        metrics = {}
        
        if self.timing_errors:
            metrics['timing_mae'] = np.mean(self.timing_errors)
        
        if self.rhythm_correlations:
            metrics['rhythm_correlation'] = np.mean(self.rhythm_correlations)
        
        return metrics


class PerformanceMetrics:
    """Metrics for model performance and efficiency."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.inference_times = []
        self.memory_usage = []
        self.throughput = []
    
    def update(self, inference_time: float, memory_mb: float, 
               batch_size: int, sequence_length: int):
        """Update performance metrics."""
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_mb)
        
        # Throughput: sequences per second
        throughput = batch_size / inference_time if inference_time > 0 else 0
        self.throughput.append(throughput)
    
    def compute(self) -> Dict[str, float]:
        """Compute performance metrics."""
        metrics = {}
        
        if self.inference_times:
            metrics['avg_inference_time'] = np.mean(self.inference_times)
            metrics['inference_time_std'] = np.std(self.inference_times)
        
        if self.memory_usage:
            metrics['avg_memory_mb'] = np.mean(self.memory_usage)
            metrics['peak_memory_mb'] = np.max(self.memory_usage)
        
        if self.throughput:
            metrics['avg_throughput'] = np.mean(self.throughput)
        
        return metrics


class CombinedMetrics:
    """Combined metrics tracker for comprehensive evaluation."""
    
    def __init__(self):
        self.replay_metrics = ReplayMetrics()
        self.accuracy_metrics = AccuracyMetrics()
        self.timing_metrics = TimingMetrics()
        self.performance_metrics = PerformanceMetrics()
    
    def reset(self):
        """Reset all metrics."""
        self.replay_metrics.reset()
        self.accuracy_metrics.reset()
        self.timing_metrics.reset()
        self.performance_metrics.reset()
    
    def update(self, predictions: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor],
               mask: Optional[torch.Tensor] = None,
               **kwargs):
        """Update all metrics."""
        self.replay_metrics.update(predictions, targets, mask)
        
        if 'timing_data' in targets:
            self.timing_metrics.update(
                predictions.get('timing_pred', targets['timing_data']),
                targets['timing_data'],
                mask
            )
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        all_metrics = {}
        
        all_metrics.update(self.replay_metrics.compute())
        all_metrics.update(self.accuracy_metrics.compute())
        all_metrics.update(self.timing_metrics.compute())
        all_metrics.update(self.performance_metrics.compute())
        
        return all_metrics