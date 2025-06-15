#!/usr/bin/env python3
"""
Simple training script for osu! AI Replay Maker

Usage:
    python train.py                    # Use default config
    python train.py --config custom.yaml  # Use custom config
    python train.py --resume checkpoint.pt  # Resume from checkpoint
    python train.py --dataset my_dataset   # Use custom dataset folder
    python train.py -d reduced_dataset_100 # Use reduced dataset
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.config.model_config import ModelConfig, TrainingConfig, DataConfig, get_device
from src.models.transformer import OsuTransformer
from src.data.dataset import OsuReplayDataset
from src.training.trainer import OsuTrainer
from src.utils.logging_utils import setup_logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Creating default config...")
        return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def create_configs_from_yaml(config_dict: dict):
    """Create config objects from YAML dictionary."""
    # Model config
    model_config = ModelConfig()
    if 'model' in config_dict:
        for key, value in config_dict['model'].items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
    
    # Training config
    training_config = TrainingConfig()
    if 'training' in config_dict:
        for key, value in config_dict['training'].items():
            if hasattr(training_config, key):
                # Convert numeric values to proper types
                if key == 'learning_rate' and isinstance(value, str):
                    value = float(value)
                elif key in ['weight_decay', 'gradient_clip_norm', 'curriculum_start_difficulty', 
                           'curriculum_end_difficulty', 'cursor_loss_weight', 'timing_loss_weight',
                           'key_loss_weight', 'accuracy_loss_weight', 'dropout_rate', 'label_smoothing'] and isinstance(value, str):
                    value = float(value)
                setattr(training_config, key, value)
    
    # Data config
    data_config = DataConfig()
    if 'data' in config_dict:
        for key, value in config_dict['data'].items():
            if hasattr(data_config, key):
                setattr(data_config, key, value)
    
    return model_config, training_config, data_config


def check_dataset(data_config: DataConfig) -> bool:
    """Check if dataset exists and is properly structured."""
    issues = []
    
    # Check directories
    if not os.path.exists(data_config.beatmap_dir):
        issues.append(f"Beatmap directory not found: {data_config.beatmap_dir}")
    
    if not os.path.exists(data_config.replay_dir):
        issues.append(f"Replay directory not found: {data_config.replay_dir}")
    
    if not os.path.exists(data_config.csv_path):
        issues.append(f"Index CSV not found: {data_config.csv_path}")
    
    if issues:
        print("\n❌ Dataset issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n💡 Make sure you have:")
        print("  1. Beatmap files (.osu) in dataset/beatmaps/")
        print("  2. Replay files (.osr) in dataset/replays/")
        print("  3. Index CSV file at dataset/index.csv")
        print("  4. Run data preprocessing if needed")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Train osu! AI Replay Maker')
    parser.add_argument('--config', '-c', default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', '-r', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--dataset', '-d', type=str, default='dataset',
                       help='Path to dataset folder (default: dataset)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    print("🎮 osu! AI Replay Maker - Training")
    print("=" * 40)
    
    # Load configuration
    print(f"📋 Loading config from: {args.config}")
    config_dict = load_config(args.config)
    model_config, training_config, data_config = create_configs_from_yaml(config_dict)
    
    # Override dataset paths if custom dataset folder is specified
    if args.dataset != 'dataset':
        print(f"📁 Using custom dataset folder: {args.dataset}")
        data_config.data_path = args.dataset
        data_config.replay_dir = f"{args.dataset}/replays/npy"
        data_config.beatmap_dir = f"{args.dataset}/beatmaps"
        data_config.csv_path = f"{args.dataset}/index.csv"
    
    # Check dataset
    print("🔍 Checking dataset...")
    if not check_dataset(data_config):
        return 1
    
    # Get device
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Create model
        print("🧠 Creating model...")
        model = OsuTransformer(model_config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Create datasets
        print("📊 Loading datasets...")
        train_dataset = OsuReplayDataset(
            data_config=data_config,
            split='train'
        )
        
        val_dataset = OsuReplayDataset(
            data_config=data_config,
            split='val'
        )
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        # For small datasets, don't drop last batch to avoid empty loaders
        drop_last_train = len(train_dataset) >= training_config.batch_size
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            drop_last=drop_last_train
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            drop_last=False
        )
        
        # Create trainer
        print("🏃 Initializing trainer...")
        trainer = OsuTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            model_config=model_config,
            device=device,
            logger=logger
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"🔄 Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        print("🚀 Starting training...")
        print(f"   Epochs: {training_config.max_epochs}")
        print(f"   Batch size: {training_config.batch_size}")
        print(f"   Learning rate: {training_config.learning_rate}")
        print(f"   Mixed precision: {training_config.use_mixed_precision}")
        print()
        
        trainer.train()
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Checkpoints saved in: checkpoints/")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())