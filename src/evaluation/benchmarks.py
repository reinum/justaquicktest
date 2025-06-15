"""Benchmarking utilities for standardized model evaluation."""

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import logging
from abc import ABC, abstractmethod

from ..generation.generator import ReplayGenerator, GenerationResult
from ..config.model_config import ModelConfig, GenerationConfig
from .evaluator import ReplayEvaluator, EvaluationResult


@dataclass
class BenchmarkResult:
    """Result from a benchmark test."""
    
    benchmark_name: str
    test_name: str
    score: float
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'benchmark_name': self.benchmark_name,
            'test_name': self.test_name,
            'score': self.score,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'execution_time': self.execution_time,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    overall_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
        self._update_overall_score()
    
    def _update_overall_score(self):
        """Update overall score based on results."""
        if self.results:
            successful_results = [r for r in self.results if r.success]
            if successful_results:
                self.overall_score = np.mean([r.score for r in successful_results])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        successful = len([r for r in self.results if r.success])
        total = len(self.results)
        
        return {
            'name': self.name,
            'overall_score': self.overall_score,
            'tests_passed': successful,
            'total_tests': total,
            'success_rate': successful / total if total > 0 else 0,
            'total_execution_time': sum(r.execution_time for r in self.results),
            'metadata': self.metadata
        }
    
    def save(self, path: Union[str, Path]):
        """Save benchmark suite to file."""
        data = {
            'summary': self.get_summary(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class BaseBenchmark(ABC):
    """Base class for benchmarks."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def run(self, generator: ReplayGenerator, 
           evaluator: ReplayEvaluator,
           **kwargs) -> List[BenchmarkResult]:
        """Run the benchmark."""
        pass
    
    def _time_execution(self, func: Callable, *args, **kwargs) -> tuple:
        """Time function execution."""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time, True, None
        except Exception as e:
            execution_time = time.time() - start_time
            return None, execution_time, False, str(e)


class PerformanceBenchmark(BaseBenchmark):
    """Benchmark for testing generation performance."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__("Performance", logger)
    
    def run(self, generator: ReplayGenerator,
           evaluator: ReplayEvaluator,
           test_beatmaps: List[Dict[str, Any]],
           **kwargs) -> List[BenchmarkResult]:
        """Run performance benchmarks."""
        results = []
        
        # Test generation speed
        results.extend(self._test_generation_speed(generator, test_beatmaps))
        
        # Test memory usage
        results.extend(self._test_memory_usage(generator, test_beatmaps))
        
        # Test scalability
        results.extend(self._test_scalability(generator, test_beatmaps))
        
        return results
    
    def _test_generation_speed(self, generator: ReplayGenerator,
                              test_beatmaps: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Test replay generation speed."""
        results = []
        
        for i, beatmap in enumerate(test_beatmaps[:5]):  # Test on first 5 beatmaps
            self.logger.info(f"Testing generation speed on beatmap {i+1}")
            
            # Generate replay and time it
            replay_result, exec_time, success, error = self._time_execution(
                generator.generate_replay, beatmap
            )
            
            if success and replay_result:
                # Calculate speed metrics
                replay_duration = replay_result.timestamps[-1] - replay_result.timestamps[0]
                generation_ratio = exec_time / (replay_duration / 1000)  # seconds
                
                score = min(1.0, 1.0 / generation_ratio)  # Better if faster than real-time
                
                metrics = {
                    'generation_time': exec_time,
                    'replay_duration': replay_duration,
                    'generation_ratio': generation_ratio,
                    'frames_per_second': len(replay_result.timestamps) / exec_time
                }
            else:
                score = 0.0
                metrics = {'generation_time': exec_time}
            
            result = BenchmarkResult(
                benchmark_name=self.name,
                test_name=f"generation_speed_beatmap_{i+1}",
                score=score,
                metrics=metrics,
                execution_time=exec_time,
                success=success,
                error_message=error,
                metadata={'beatmap_id': beatmap.get('id', f'test_{i+1}')}
            )
            
            results.append(result)
        
        return results
    
    def _test_memory_usage(self, generator: ReplayGenerator,
                          test_beatmaps: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Test memory usage during generation."""
        results = []
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            for i, beatmap in enumerate(test_beatmaps[:3]):  # Test on first 3 beatmaps
                self.logger.info(f"Testing memory usage on beatmap {i+1}")
                
                # Measure memory before
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Generate replay
                replay_result, exec_time, success, error = self._time_execution(
                    generator.generate_replay, beatmap
                )
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                # Score based on memory efficiency (lower is better)
                score = max(0, 1.0 - memory_used / 1000)  # Penalize if >1GB used
                
                metrics = {
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after,
                    'memory_used_mb': memory_used,
                    'memory_efficiency': memory_used / exec_time if exec_time > 0 else 0
                }
                
                result = BenchmarkResult(
                    benchmark_name=self.name,
                    test_name=f"memory_usage_beatmap_{i+1}",
                    score=score,
                    metrics=metrics,
                    execution_time=exec_time,
                    success=success,
                    error_message=error,
                    metadata={'beatmap_id': beatmap.get('id', f'test_{i+1}')}
                )
                
                results.append(result)
                
        except ImportError:
            self.logger.warning("psutil not available, skipping memory tests")
        
        return results
    
    def _test_scalability(self, generator: ReplayGenerator,
                         test_beatmaps: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Test scalability with different sequence lengths."""
        results = []
        
        # Test with different generation lengths
        test_lengths = [30, 60, 120, 300]  # seconds
        
        for length in test_lengths:
            if not test_beatmaps:
                continue
                
            beatmap = test_beatmaps[0].copy()  # Use first beatmap
            
            # Modify beatmap to have specific length
            beatmap['duration'] = length * 1000  # Convert to ms
            
            self.logger.info(f"Testing scalability with {length}s duration")
            
            replay_result, exec_time, success, error = self._time_execution(
                generator.generate_replay, beatmap
            )
            
            if success and replay_result:
                # Calculate scalability metrics
                frames_generated = len(replay_result.timestamps)
                time_per_frame = exec_time / frames_generated if frames_generated > 0 else float('inf')
                
                # Score based on linear scalability
                expected_time = length * 0.1  # Expect 0.1s per second of replay
                scalability_ratio = expected_time / exec_time if exec_time > 0 else 0
                score = min(1.0, scalability_ratio)
                
                metrics = {
                    'target_duration': length,
                    'frames_generated': frames_generated,
                    'time_per_frame': time_per_frame,
                    'scalability_ratio': scalability_ratio
                }
            else:
                score = 0.0
                metrics = {'target_duration': length}
            
            result = BenchmarkResult(
                benchmark_name=self.name,
                test_name=f"scalability_{length}s",
                score=score,
                metrics=metrics,
                execution_time=exec_time,
                success=success,
                error_message=error
            )
            
            results.append(result)
        
        return results


class AccuracyBenchmark(BaseBenchmark):
    """Benchmark for testing generation accuracy."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__("Accuracy", logger)
    
    def run(self, generator: ReplayGenerator,
           evaluator: ReplayEvaluator,
           test_data: List[Dict[str, Any]],
           **kwargs) -> List[BenchmarkResult]:
        """Run accuracy benchmarks."""
        results = []
        
        # Test cursor accuracy
        results.extend(self._test_cursor_accuracy(generator, evaluator, test_data))
        
        # Test key timing accuracy
        results.extend(self._test_key_accuracy(generator, evaluator, test_data))
        
        # Test overall gameplay accuracy
        results.extend(self._test_gameplay_accuracy(generator, evaluator, test_data))
        
        return results
    
    def _test_cursor_accuracy(self, generator: ReplayGenerator,
                             evaluator: ReplayEvaluator,
                             test_data: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Test cursor movement accuracy."""
        results = []
        
        for i, test_case in enumerate(test_data[:10]):  # Test on first 10 cases
            beatmap = test_case['beatmap']
            reference = test_case.get('reference_replay')
            
            self.logger.info(f"Testing cursor accuracy on case {i+1}")
            
            # Generate replay
            replay_result, exec_time, success, error = self._time_execution(
                generator.generate_replay, beatmap
            )
            
            if success and replay_result:
                # Evaluate against reference if available
                eval_result = evaluator.evaluate(
                    replay_result, reference, beatmap, detailed=False
                )
                
                score = eval_result.metrics.cursor_smoothness * 0.5 + \
                       eval_result.metrics.trajectory_similarity * 0.5
                
                metrics = {
                    'cursor_mse': eval_result.metrics.cursor_mse,
                    'cursor_mae': eval_result.metrics.cursor_mae,
                    'cursor_smoothness': eval_result.metrics.cursor_smoothness,
                    'trajectory_similarity': eval_result.metrics.trajectory_similarity
                }
            else:
                score = 0.0
                metrics = {}
            
            result = BenchmarkResult(
                benchmark_name=self.name,
                test_name=f"cursor_accuracy_case_{i+1}",
                score=score,
                metrics=metrics,
                execution_time=exec_time,
                success=success,
                error_message=error,
                metadata={'test_case_id': test_case.get('id', f'case_{i+1}')}
            )
            
            results.append(result)
        
        return results
    
    def _test_key_accuracy(self, generator: ReplayGenerator,
                          evaluator: ReplayEvaluator,
                          test_data: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Test key press accuracy."""
        results = []
        
        for i, test_case in enumerate(test_data[:10]):
            beatmap = test_case['beatmap']
            reference = test_case.get('reference_replay')
            
            self.logger.info(f"Testing key accuracy on case {i+1}")
            
            replay_result, exec_time, success, error = self._time_execution(
                generator.generate_replay, beatmap
            )
            
            if success and replay_result:
                eval_result = evaluator.evaluate(
                    replay_result, reference, beatmap, detailed=False
                )
                
                score = eval_result.metrics.key_f1
                
                metrics = {
                    'key_accuracy': eval_result.metrics.key_accuracy,
                    'key_precision': eval_result.metrics.key_precision,
                    'key_recall': eval_result.metrics.key_recall,
                    'key_f1': eval_result.metrics.key_f1,
                    'timing_accuracy': eval_result.metrics.timing_accuracy
                }
            else:
                score = 0.0
                metrics = {}
            
            result = BenchmarkResult(
                benchmark_name=self.name,
                test_name=f"key_accuracy_case_{i+1}",
                score=score,
                metrics=metrics,
                execution_time=exec_time,
                success=success,
                error_message=error,
                metadata={'test_case_id': test_case.get('id', f'case_{i+1}')}
            )
            
            results.append(result)
        
        return results
    
    def _test_gameplay_accuracy(self, generator: ReplayGenerator,
                               evaluator: ReplayEvaluator,
                               test_data: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Test overall gameplay accuracy."""
        results = []
        
        for i, test_case in enumerate(test_data[:5]):  # Test on first 5 cases
            beatmap = test_case['beatmap']
            reference = test_case.get('reference_replay')
            
            self.logger.info(f"Testing gameplay accuracy on case {i+1}")
            
            replay_result, exec_time, success, error = self._time_execution(
                generator.generate_replay, beatmap
            )
            
            if success and replay_result:
                eval_result = evaluator.evaluate(
                    replay_result, reference, beatmap, detailed=True
                )
                
                score = eval_result.metrics.gameplay_score
                
                metrics = {
                    'hit_accuracy': eval_result.metrics.hit_accuracy,
                    'rhythm_consistency': eval_result.metrics.rhythm_consistency,
                    'flow_quality': eval_result.metrics.flow_quality,
                    'overall_score': eval_result.metrics.overall_score
                }
            else:
                score = 0.0
                metrics = {}
            
            result = BenchmarkResult(
                benchmark_name=self.name,
                test_name=f"gameplay_accuracy_case_{i+1}",
                score=score,
                metrics=metrics,
                execution_time=exec_time,
                success=success,
                error_message=error,
                metadata={'test_case_id': test_case.get('id', f'case_{i+1}')}
            )
            
            results.append(result)
        
        return results


class BeatmapBenchmark(BaseBenchmark):
    """Benchmark for testing on different beatmap types."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__("Beatmap_Diversity", logger)
    
    def run(self, generator: ReplayGenerator,
           evaluator: ReplayEvaluator,
           beatmap_categories: Dict[str, List[Dict[str, Any]]],
           **kwargs) -> List[BenchmarkResult]:
        """Run beatmap diversity benchmarks."""
        results = []
        
        for category, beatmaps in beatmap_categories.items():
            self.logger.info(f"Testing on {category} beatmaps")
            
            category_scores = []
            category_metrics = []
            
            for i, beatmap in enumerate(beatmaps[:3]):  # Test 3 per category
                replay_result, exec_time, success, error = self._time_execution(
                    generator.generate_replay, beatmap
                )
                
                if success and replay_result:
                    eval_result = evaluator.evaluate(
                        replay_result, None, beatmap, detailed=False
                    )
                    
                    category_scores.append(eval_result.metrics.overall_score)
                    category_metrics.append(eval_result.metrics.__dict__)
            
            # Calculate category score
            if category_scores:
                score = np.mean(category_scores)
                avg_metrics = {}
                
                # Average metrics across beatmaps in category
                if category_metrics:
                    for key in category_metrics[0].keys():
                        if isinstance(category_metrics[0][key], (int, float)):
                            avg_metrics[key] = np.mean([m[key] for m in category_metrics])
            else:
                score = 0.0
                avg_metrics = {}
            
            result = BenchmarkResult(
                benchmark_name=self.name,
                test_name=f"category_{category}",
                score=score,
                metrics=avg_metrics,
                execution_time=sum(exec_time for exec_time in [0] * len(beatmaps[:3])),
                success=len(category_scores) > 0,
                metadata={
                    'category': category,
                    'beatmaps_tested': len(beatmaps[:3]),
                    'successful_generations': len(category_scores)
                }
            )
            
            results.append(result)
        
        return results


def create_benchmark_suite(name: str = "Model_Evaluation") -> BenchmarkSuite:
    """Create a comprehensive benchmark suite."""
    return BenchmarkSuite(name=name)


def run_full_benchmark(generator: ReplayGenerator,
                      evaluator: ReplayEvaluator,
                      test_data: Dict[str, Any],
                      output_dir: Optional[Union[str, Path]] = None,
                      logger: Optional[logging.Logger] = None) -> BenchmarkSuite:
    """Run a full benchmark suite.
    
    Args:
        generator: Replay generator to test
        evaluator: Evaluator for assessment
        test_data: Test data containing beatmaps and reference replays
        output_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        Complete benchmark suite with results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Starting full benchmark suite")
    
    suite = create_benchmark_suite()
    
    # Performance benchmarks
    perf_benchmark = PerformanceBenchmark(logger)
    perf_results = perf_benchmark.run(
        generator, evaluator, 
        test_data.get('beatmaps', [])
    )
    
    for result in perf_results:
        suite.add_result(result)
    
    # Accuracy benchmarks
    acc_benchmark = AccuracyBenchmark(logger)
    acc_results = acc_benchmark.run(
        generator, evaluator,
        test_data.get('test_cases', [])
    )
    
    for result in acc_results:
        suite.add_result(result)
    
    # Beatmap diversity benchmarks
    if 'beatmap_categories' in test_data:
        beatmap_benchmark = BeatmapBenchmark(logger)
        beatmap_results = beatmap_benchmark.run(
            generator, evaluator,
            test_data['beatmap_categories']
        )
        
        for result in beatmap_results:
            suite.add_result(result)
    
    # Save results if output directory specified
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"benchmark_results_{timestamp}.json"
        
        suite.save(results_file)
        logger.info(f"Benchmark results saved to {results_file}")
    
    logger.info(f"Benchmark suite completed. Overall score: {suite.overall_score:.3f}")
    
    return suite


def create_test_data(beatmap_dir: Union[str, Path],
                    replay_dir: Optional[Union[str, Path]] = None,
                    max_beatmaps: int = 50) -> Dict[str, Any]:
    """Create test data for benchmarking.
    
    Args:
        beatmap_dir: Directory containing beatmap files
        replay_dir: Directory containing reference replays
        max_beatmaps: Maximum number of beatmaps to include
        
    Returns:
        Test data dictionary
    """
    test_data = {
        'beatmaps': [],
        'test_cases': [],
        'beatmap_categories': {
            'easy': [],
            'normal': [],
            'hard': [],
            'insane': [],
            'expert': []
        }
    }
    
    # This would be implemented to load actual beatmap and replay data
    # For now, return empty structure
    
    return test_data