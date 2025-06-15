#!/usr/bin/env python3
"""
Osu! AI Replay Maker - Main Entry Point

This script provides a command-line interface for training models and generating replays.

Usage:
    python main.py train --config config/default.yaml
    python main.py generate --model checkpoints/best_model.pt --beatmap path/to/map.osu
    python main.py evaluate --model checkpoints/best_model.pt --dataset dataset/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.config.model_config import ModelConfig
from src.training.trainer import OsuTrainer
from src.generation.generator import ReplayGenerator
from src.evaluation.evaluator import ReplayEvaluator
from src.data.dataset import OsuDataset
from src.utils.logging_utils import setup_logging


def setup_args() -> argparse.ArgumentParser:
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="Osu! AI Replay Maker",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    train_parser.add_argument(
        '--resume', '-r',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    train_parser.add_argument(
        '--output', '-o',
        type=str,
        default='checkpoints/',
        help='Output directory for checkpoints'
    )
    train_parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    # Generation command
    gen_parser = subparsers.add_parser('generate', help='Generate replays')
    gen_parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    gen_parser.add_argument(
        '--beatmap', '-b',
        type=str,
        required=True,
        help='Path to beatmap (.osu file)'
    )
    gen_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output path for generated replay (.osr file)'
    )
    gen_parser.add_argument(
        '--accuracy',
        type=float,
        default=0.95,
        help='Target accuracy (0.0-1.0)'
    )
    gen_parser.add_argument(
        '--target-300s',
        type=float,
        default=0.85,
        help='Target ratio of 300s (0.0-1.0)'
    )
    gen_parser.add_argument(
        '--target-100s',
        type=float,
        default=0.10,
        help='Target ratio of 100s (0.0-1.0)'
    )
    gen_parser.add_argument(
        '--target-50s',
        type=float,
        default=0.04,
        help='Target ratio of 50s (0.0-1.0)'
    )
    gen_parser.add_argument(
        '--target-misses',
        type=float,
        default=0.01,
        help='Target ratio of misses (0.0-1.0)'
    )
    gen_parser.add_argument(
        '--mods',
        type=str,
        nargs='*',
        default=[],
        help='Mods to apply (e.g., HD DT HR)'
    )
    gen_parser.add_argument(
        '--sampling-strategy',
        type=str,
        default='temperature',
        choices=['temperature', 'top_k', 'top_p', 'beam_search', 'nucleus'],
        help='Sampling strategy for generation'
    )
    gen_parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for sampling (higher = more random)'
    )
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    eval_parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to evaluation dataset'
    )
    eval_parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results/',
        help='Output directory for evaluation results'
    )
    eval_parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to evaluate'
    )
    eval_parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run comprehensive benchmark suite'
    )
    
    # Data processing command
    data_parser = subparsers.add_parser('process', help='Process dataset')
    data_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input dataset directory'
    )
    data_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output processed dataset directory'
    )
    data_parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    data_parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of worker processes'
    )
    
    # Utility commands
    utils_parser = subparsers.add_parser('utils', help='Utility commands')
    utils_subparsers = utils_parser.add_subparsers(dest='util_command')
    
    # Validate dataset
    validate_parser = utils_subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    
    # Convert replay format
    convert_parser = utils_subparsers.add_parser('convert', help='Convert replay formats')
    convert_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input replay file'
    )
    convert_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output replay file'
    )
    convert_parser.add_argument(
        '--format',
        type=str,
        choices=['osr', 'json', 'csv'],
        default='osr',
        help='Output format'
    )
    
    return parser


def train_command(args) -> None:
    """Execute training command."""
    logging.info(f"Starting training with config: {args.config}")
    
    # Load configuration
    config = ModelConfig.from_file(args.config)
    
    # Override config with command line arguments
    if args.wandb:
        config.logging.use_wandb = True
    
    # Initialize trainer
    trainer = OsuTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logging.info(f"Resuming training from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
        logging.info("Training completed successfully!")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        trainer.save_checkpoint(Path(args.output) / "interrupted_checkpoint.pt")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


def generate_command(args) -> None:
    """Execute generation command."""
    logging.info(f"Generating replay for beatmap: {args.beatmap}")
    
    # Load model
    generator = ReplayGenerator.from_checkpoint(args.model)
    
    # Set generation parameters
    generation_config = {
        'target_accuracy': args.accuracy,
        'target_300s': args.target_300s,
        'target_100s': args.target_100s,
        'target_50s': args.target_50s,
        'target_misses': args.target_misses,
        'mods': args.mods,
        'sampling_strategy': args.sampling_strategy,
        'temperature': args.temperature
    }
    
    # Generate replay
    try:
        result = generator.generate(
            beatmap_path=args.beatmap,
            **generation_config
        )
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            beatmap_name = Path(args.beatmap).stem
            output_path = f"generated_{beatmap_name}.osr"
        
        # Export replay
        result.export(output_path)
        
        logging.info(f"Replay generated successfully: {output_path}")
        logging.info(f"Generation stats: {result.stats}")
        
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise


def evaluate_command(args) -> None:
    """Execute evaluation command."""
    logging.info(f"Evaluating model: {args.model}")
    
    # Load model and evaluator
    generator = ReplayGenerator.from_checkpoint(args.model)
    evaluator = ReplayEvaluator()
    
    # Load evaluation dataset
    dataset = OsuDataset(args.dataset, split='test')
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.benchmark:
            # Run comprehensive benchmark
            from src.evaluation.benchmarks import create_benchmark_suite
            
            benchmark_suite = create_benchmark_suite()
            results = benchmark_suite.run_all(generator, dataset)
            
            # Save benchmark results
            results.save(output_dir / "benchmark_results.json")
            logging.info(f"Benchmark completed: {output_dir / 'benchmark_results.json'}")
        else:
            # Run standard evaluation
            results = evaluator.evaluate_dataset(
                generator=generator,
                dataset=dataset,
                num_samples=args.num_samples
            )
            
            # Save evaluation results
            results.save(output_dir / "evaluation_results.json")
            logging.info(f"Evaluation completed: {output_dir / 'evaluation_results.json'}")
            
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


def process_command(args) -> None:
    """Execute data processing command."""
    logging.info(f"Processing dataset: {args.input} -> {args.output}")
    
    # Load configuration
    config = ModelConfig.from_file(args.config)
    
    # Process dataset
    from src.data.preprocessing import DataProcessor
    
    processor = DataProcessor(config)
    processor.process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        num_workers=args.workers
    )
    
    logging.info("Dataset processing completed!")


def utils_command(args) -> None:
    """Execute utility commands."""
    if args.util_command == 'validate':
        from src.utils.dataset_validator import validate_dataset
        
        logging.info(f"Validating dataset: {args.dataset}")
        is_valid, issues = validate_dataset(args.dataset)
        
        if is_valid:
            logging.info("Dataset validation passed!")
        else:
            logging.warning(f"Dataset validation failed: {issues}")
            
    elif args.util_command == 'convert':
        from src.utils.replay_converter import convert_replay
        
        logging.info(f"Converting replay: {args.input} -> {args.output}")
        convert_replay(args.input, args.output, args.format)
        logging.info("Conversion completed!")


def main() -> None:
    """Main entry point."""
    # Setup argument parser
    parser = setup_args()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        logging.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        logging.warning("CUDA not available, using CPU")
    
    # Execute command
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'generate':
            generate_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'process':
            process_command(args)
        elif args.command == 'utils':
            utils_command(args)
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()