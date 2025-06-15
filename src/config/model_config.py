"""Configuration classes for the osu! AI replay maker."""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch


@dataclass
class ModelConfig:
    """Configuration for the transformer model architecture."""
    
    # Model Architecture
    d_model: int = 512  # Model dimension
    n_heads: int = 8    # Number of attention heads
    n_layers: int = 6   # Number of transformer layers
    d_ff: int = 2048    # Feed-forward dimension
    dropout: float = 0.1
    
    # Sequence Parameters
    max_seq_length: int = 2048  # Maximum sequence length
    cursor_history_length: int = 64  # Previous cursor positions to consider
    beatmap_context_length: int = 128  # Beatmap objects to consider
    
    # Input/Output Dimensions
    cursor_dim: int = 2  # x, y coordinates
    key_dim: int = 4     # K1, K2, M1, M2 key states
    beatmap_feature_dim: int = 8  # Hit object features (time, x, y, type, etc.)
    slider_feature_dim: int = 13  # Slider-specific features (progress, velocity, etc.)
    accuracy_condition_dim: int = 4  # 300s, 100s, 50s, misses percentages
    
    # Positional Encoding
    use_positional_encoding: bool = True
    max_position_embeddings: int = 4096
    
    # Attention Configuration
    use_relative_attention: bool = False
    causal: bool = True  # For autoregressive generation
    max_relative_position: int = 128
    use_cross_attention: bool = False
    n_cross_attention_layers: int = 2
    
    # Memory Optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True  # If available
    
    # Output Generation
    output_cursor_dim: int = 2  # x, y coordinates
    output_timing_dim: int = 1  # Hit timing offset
    output_key_dim: int = 4     # Key press probabilities
    

@dataclass
class TrainingConfig:
    """Configuration for training the model."""
    
    # Training Parameters
    batch_size: int = 16  # Optimized for 4GB VRAM
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    gradient_clip_norm: float = 1.0
    
    # Mixed Precision
    use_mixed_precision: bool = True
    fp16: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine_with_warmup"
    gradient_accumulation_steps: int = 4
    
    # Curriculum Learning
    use_curriculum_learning: bool = True
    curriculum_start_difficulty: float = 1.0  # Star rating
    curriculum_end_difficulty: float = 8.0
    curriculum_steps: int = 10000
    
    # Multi-task Learning Weights
    cursor_loss_weight: float = 1.0
    timing_loss_weight: float = 0.5
    key_loss_weight: float = 0.3
    accuracy_loss_weight: float = 0.2
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    
    # Validation
    val_check_interval: int = 1000
    early_stopping_patience: int = 10
    
    # Checkpointing
    save_interval: int = 1  # Save every epoch
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = True
    project_name: str = "osu-ai-replay-maker"
    

@dataclass
class DataConfig:
    """Configuration for data processing and loading."""
    
    # Dataset Paths
    data_path: str = "dataset"  # Root path to dataset
    replay_dir: str = "dataset/replays/npy"
    beatmap_dir: str = "dataset/beatmaps"
    csv_path: str = "dataset/index.csv"
    
    # Data Splits
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Sequence Processing
    sequence_length: int = 1024  # Length of training sequences
    stride: int = 512            # Stride for sliding window (sequence_length // 2)
    overlap_ratio: float = 0.5   # Overlap between consecutive sequences
    min_sequence_length: int = 256  # Minimum sequence length to keep
    
    # Data Filtering
    min_accuracy: float = 0.8    # Minimum accuracy to include replay
    max_accuracy: float = 1.0    # Maximum accuracy to include replay
    min_star_rating: float = 1.0  # Minimum difficulty
    max_star_rating: float = 10.0 # Maximum difficulty
    
    # Normalization
    normalize_coordinates: bool = True
    coordinate_range: Tuple[int, int] = (0, 512)  # osu! playfield size
    normalize_timing: bool = True
    
    # Data Augmentation
    use_augmentation: bool = True
    rotation_range: float = 15.0  # degrees
    scale_range: Tuple[float, float] = (0.9, 1.1)
    timing_jitter: float = 0.02   # seconds
    
    # Loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Memory Management
    max_replays_in_memory: int = 1000  # For large datasets
    cache_preprocessed: bool = True
    

@dataclass
class GenerationConfig:
    """Configuration for replay generation."""
    
    # Generation Parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Sampling Strategy
    sampling_strategy: str = 'nucleus'  # 'nucleus', 'top_k', 'temperature'
    smoothness_weight: float = 0.1
    
    # Sampling
    use_beam_search: bool = False
    beam_size: int = 5
    length_penalty: float = 1.0
    max_length: int = 4096
    
    # Output Control
    max_generation_length: int = 4096
    min_generation_length: int = 100
    
    # Accuracy Conditioning
    target_300_ratio: float = 0.95
    target_100_ratio: float = 0.04
    target_50_ratio: float = 0.01
    target_miss_ratio: float = 0.0
    
    # Post-processing
    smooth_cursor_movement: bool = True
    smoothing_window: int = 5
    enforce_hit_windows: bool = True
    

def get_device() -> torch.device:
    """Get the appropriate device for training/inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_default_config() -> Tuple[ModelConfig, TrainingConfig, DataConfig, GenerationConfig]:
    """Get default configuration for all components."""
    return (
        ModelConfig(),
        TrainingConfig(),
        DataConfig(),
        GenerationConfig()
    )