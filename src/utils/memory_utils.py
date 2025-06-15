"""Memory management utilities for the osu! AI replay maker."""

import gc
import logging
import psutil
import torch
from typing import Dict, Any, Optional
from functools import wraps
import time


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage information
    """
    # System memory
    memory = psutil.virtual_memory()
    
    # Process memory
    process = psutil.Process()
    process_memory = process.memory_info()
    
    usage = {
        'system': {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        },
        'process': {
            'rss_gb': process_memory.rss / (1024**3),  # Resident Set Size
            'vms_gb': process_memory.vms / (1024**3),  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    }
    
    # GPU memory (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        
        usage['gpu'] = {
            'total_gb': gpu_memory / (1024**3),
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'free_gb': (gpu_memory - reserved) / (1024**3),
            'utilization_percent': (allocated / gpu_memory) * 100
        }
    
    return usage


def clear_cache() -> None:
    """Clear various caches to free memory."""
    # Python garbage collection
    gc.collect()
    
    # PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_memory_usage(logger: Optional[logging.Logger] = None) -> None:
    """Log current memory usage.
    
    Args:
        logger: Logger instance to use (creates one if None)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    usage = get_memory_usage()
    
    logger.info(f"Memory Usage:")
    logger.info(f"  System: {usage['system']['used_gb']:.1f}GB / {usage['system']['total_gb']:.1f}GB ({usage['system']['percent']:.1f}%)")
    logger.info(f"  Process: {usage['process']['rss_gb']:.1f}GB ({usage['process']['percent']:.1f}%)")
    
    if 'gpu' in usage:
        logger.info(f"  GPU: {usage['gpu']['allocated_gb']:.1f}GB / {usage['gpu']['total_gb']:.1f}GB ({usage['gpu']['utilization_percent']:.1f}%)")


def memory_monitor(func):
    """Decorator to monitor memory usage of a function.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function that logs memory usage
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Log memory before
        usage_before = get_memory_usage()
        logger.debug(f"Memory before {func.__name__}: {usage_before['process']['rss_gb']:.1f}GB")
        
        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Log memory after
            end_time = time.time()
            usage_after = get_memory_usage()
            
            memory_diff = usage_after['process']['rss_gb'] - usage_before['process']['rss_gb']
            execution_time = end_time - start_time
            
            logger.debug(
                f"Memory after {func.__name__}: {usage_after['process']['rss_gb']:.1f}GB "
                f"(+{memory_diff:+.1f}GB) in {execution_time:.2f}s"
            )
    
    return wrapper


class MemoryTracker:
    """Track memory usage over time."""
    
    def __init__(self, name: str = "memory_tracker"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.snapshots = []
        self.start_time = time.time()
    
    def snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a memory snapshot.
        
        Args:
            label: Optional label for the snapshot
            
        Returns:
            Memory usage data
        """
        usage = get_memory_usage()
        timestamp = time.time() - self.start_time
        
        snapshot = {
            'timestamp': timestamp,
            'label': label,
            'usage': usage
        }
        
        self.snapshots.append(snapshot)
        
        if label:
            self.logger.debug(f"Memory snapshot '{label}': {usage['process']['rss_gb']:.1f}GB")
        
        return snapshot
    
    def get_peak_usage(self) -> Dict[str, Any]:
        """Get peak memory usage from all snapshots.
        
        Returns:
            Peak usage data
        """
        if not self.snapshots:
            return {}
        
        peak_system = max(self.snapshots, key=lambda x: x['usage']['system']['used_gb'])
        peak_process = max(self.snapshots, key=lambda x: x['usage']['process']['rss_gb'])
        
        result = {
            'peak_system': peak_system,
            'peak_process': peak_process
        }
        
        if 'gpu' in self.snapshots[0]['usage']:
            peak_gpu = max(self.snapshots, key=lambda x: x['usage']['gpu']['allocated_gb'])
            result['peak_gpu'] = peak_gpu
        
        return result
    
    def get_memory_growth(self) -> Dict[str, float]:
        """Calculate memory growth from first to last snapshot.
        
        Returns:
            Memory growth in GB
        """
        if len(self.snapshots) < 2:
            return {}
        
        first = self.snapshots[0]['usage']
        last = self.snapshots[-1]['usage']
        
        growth = {
            'system_gb': last['system']['used_gb'] - first['system']['used_gb'],
            'process_gb': last['process']['rss_gb'] - first['process']['rss_gb']
        }
        
        if 'gpu' in first and 'gpu' in last:
            growth['gpu_gb'] = last['gpu']['allocated_gb'] - first['gpu']['allocated_gb']
        
        return growth
    
    def report(self) -> str:
        """Generate a memory usage report.
        
        Returns:
            Formatted report string
        """
        if not self.snapshots:
            return "No memory snapshots recorded"
        
        peak_usage = self.get_peak_usage()
        memory_growth = self.get_memory_growth()
        
        report = f"Memory Report for {self.name}:\n"
        report += f"  Duration: {self.snapshots[-1]['timestamp']:.2f}s\n"
        report += f"  Snapshots: {len(self.snapshots)}\n"
        
        if peak_usage:
            report += f"  Peak System Memory: {peak_usage['peak_system']['usage']['system']['used_gb']:.1f}GB\n"
            report += f"  Peak Process Memory: {peak_usage['peak_process']['usage']['process']['rss_gb']:.1f}GB\n"
            
            if 'peak_gpu' in peak_usage:
                report += f"  Peak GPU Memory: {peak_usage['peak_gpu']['usage']['gpu']['allocated_gb']:.1f}GB\n"
        
        if memory_growth:
            report += f"  Memory Growth:\n"
            report += f"    System: {memory_growth['system_gb']:+.1f}GB\n"
            report += f"    Process: {memory_growth['process_gb']:+.1f}GB\n"
            
            if 'gpu_gb' in memory_growth:
                report += f"    GPU: {memory_growth['gpu_gb']:+.1f}GB\n"
        
        return report
    
    def save_report(self, output_path: str) -> None:
        """Save memory report to file.
        
        Args:
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write(self.report())
            f.write("\n\nDetailed Snapshots:\n")
            
            for i, snapshot in enumerate(self.snapshots):
                f.write(f"\nSnapshot {i+1}:\n")
                f.write(f"  Time: {snapshot['timestamp']:.2f}s\n")
                f.write(f"  Label: {snapshot['label']}\n")
                f.write(f"  System: {snapshot['usage']['system']['used_gb']:.1f}GB\n")
                f.write(f"  Process: {snapshot['usage']['process']['rss_gb']:.1f}GB\n")
                
                if 'gpu' in snapshot['usage']:
                    f.write(f"  GPU: {snapshot['usage']['gpu']['allocated_gb']:.1f}GB\n")


def check_memory_requirements(required_gb: float, buffer_gb: float = 2.0) -> bool:
    """Check if there's enough memory available.
    
    Args:
        required_gb: Required memory in GB
        buffer_gb: Additional buffer memory in GB
        
    Returns:
        True if enough memory is available
    """
    usage = get_memory_usage()
    available_gb = usage['system']['available_gb']
    
    return available_gb >= (required_gb + buffer_gb)


def estimate_model_memory(model_params: int, batch_size: int = 1, sequence_length: int = 1024) -> Dict[str, float]:
    """Estimate memory requirements for a model.
    
    Args:
        model_params: Number of model parameters
        batch_size: Batch size
        sequence_length: Sequence length
        
    Returns:
        Estimated memory requirements in GB
    """
    # Model parameters (float32 = 4 bytes)
    model_memory = model_params * 4 / (1024**3)
    
    # Gradients (same size as parameters)
    gradient_memory = model_memory
    
    # Optimizer states (Adam uses ~2x parameter memory)
    optimizer_memory = model_memory * 2
    
    # Activations (rough estimate)
    # Assume each layer stores activations of size batch_size * sequence_length * hidden_size
    # Rough estimate: hidden_size ≈ sqrt(model_params / num_layers)
    estimated_hidden_size = int((model_params / 12) ** 0.5)  # Assume 12 layers
    activation_memory = batch_size * sequence_length * estimated_hidden_size * 4 / (1024**3)
    
    return {
        'model_gb': model_memory,
        'gradients_gb': gradient_memory,
        'optimizer_gb': optimizer_memory,
        'activations_gb': activation_memory,
        'total_gb': model_memory + gradient_memory + optimizer_memory + activation_memory
    }


def optimize_memory_settings() -> Dict[str, Any]:
    """Get optimized memory settings based on available hardware.
    
    Returns:
        Dictionary with recommended settings
    """
    usage = get_memory_usage()
    
    # Base recommendations
    settings = {
        'batch_size': 8,
        'gradient_accumulation_steps': 1,
        'max_sequence_length': 1024,
        'use_gradient_checkpointing': False,
        'use_mixed_precision': False
    }
    
    # Adjust based on available memory
    available_gb = usage['system']['available_gb']
    
    if available_gb < 8:
        # Low memory system
        settings.update({
            'batch_size': 2,
            'gradient_accumulation_steps': 4,
            'max_sequence_length': 512,
            'use_gradient_checkpointing': True
        })
    elif available_gb < 16:
        # Medium memory system
        settings.update({
            'batch_size': 4,
            'gradient_accumulation_steps': 2,
            'max_sequence_length': 768,
            'use_gradient_checkpointing': True
        })
    elif available_gb >= 32:
        # High memory system
        settings.update({
            'batch_size': 16,
            'max_sequence_length': 2048
        })
    
    # GPU-specific settings
    if 'gpu' in usage:
        gpu_gb = usage['gpu']['total_gb']
        
        if gpu_gb >= 8:
            settings['use_mixed_precision'] = True
        
        if gpu_gb < 6:
            # Reduce batch size for smaller GPUs
            settings['batch_size'] = min(settings['batch_size'], 4)
            settings['use_gradient_checkpointing'] = True
    
    return settings