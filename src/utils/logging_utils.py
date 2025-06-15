"""Logging utilities for the osu! AI replay maker."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional file path to write logs to
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log messages
    """
    # Default format string
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_training_logging(
    experiment_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> str:
    """Setup logging for training experiments.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        level: Logging level
        
    Returns:
        Path to the log file
    """
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{experiment_name}_{timestamp}.log"
    
    # Setup logging
    setup_logging(
        level=level,
        log_file=str(log_file),
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = get_logger(__name__)
    logger.info(f"Training logging initialized: {log_file}")
    logger.info(f"Experiment: {experiment_name}")
    
    return str(log_file)


class TrainingLogger:
    """Logger specifically for training progress and metrics."""
    
    def __init__(self, name: str = "training"):
        self.logger = get_logger(name)
        self.step = 0
        self.epoch = 0
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log the start of an epoch."""
        self.epoch = epoch
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")
    
    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[dict] = None
    ) -> None:
        """Log the end of an epoch with metrics."""
        msg = f"Epoch {epoch} completed - Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            msg += f", Val Loss: {val_loss:.4f}"
        
        if metrics:
            metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            msg += f", {', '.join(metric_strs)}"
        
        self.logger.info(msg)
    
    def log_step(
        self,
        step: int,
        loss: float,
        lr: Optional[float] = None,
        metrics: Optional[dict] = None
    ) -> None:
        """Log training step information."""
        self.step = step
        msg = f"Step {step} - Loss: {loss:.4f}"
        
        if lr is not None:
            msg += f", LR: {lr:.6f}"
        
        if metrics:
            metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            msg += f", {', '.join(metric_strs)}"
        
        self.logger.info(msg)
    
    def log_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Log checkpoint saving."""
        msg = f"Checkpoint saved: {path}"
        if is_best:
            msg += " (Best model)"
        self.logger.info(msg)
    
    def log_validation(self, metrics: dict) -> None:
        """Log validation metrics."""
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        self.logger.info(f"Validation - {', '.join(metric_strs)}")
    
    def log_generation_stats(self, stats: dict) -> None:
        """Log generation statistics."""
        self.logger.info(f"Generation completed - {stats}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error with context."""
        if context:
            self.logger.error(f"{context}: {error}")
        else:
            self.logger.error(f"Error: {error}")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, total: int, name: str = "progress", log_interval: int = 100):
        self.logger = get_logger(name)
        self.total = total
        self.current = 0
        self.log_interval = log_interval
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """Update progress and log if needed."""
        self.current += increment
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            self._log_progress()
    
    def _log_progress(self) -> None:
        """Log current progress."""
        percentage = (self.current / self.total) * 100
        elapsed = datetime.now() - self.start_time
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = str(eta).split('.')[0]  # Remove microseconds
        else:
            eta_str = "Unknown"
        
        self.logger.info(
            f"Progress: {self.current}/{self.total} ({percentage:.1f}%) - "
            f"Elapsed: {str(elapsed).split('.')[0]} - ETA: {eta_str}"
        )
    
    def finish(self) -> None:
        """Log completion."""
        elapsed = datetime.now() - self.start_time
        self.logger.info(
            f"Completed {self.total} items in {str(elapsed).split('.')[0]}"
        )


def log_system_info() -> None:
    """Log system information for debugging."""
    import platform
    import psutil
    import torch
    
    logger = get_logger(__name__)
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    memory = psutil.virtual_memory()
    logger.info(f"RAM: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
    
    logger.info(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    else:
        logger.info("CUDA: Not available")
    
    logger.info("=== End System Information ===")