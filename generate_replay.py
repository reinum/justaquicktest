from src.generation.generator import ReplayGenerator

import torch
from pathlib import Path
import logging
import yaml
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.transformer import OsuTransformer
from src.generation.generator import ReplayGenerator
from src.config.model_config import ModelConfig, GenerationConfig

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config('config/default.yaml')

# Create model config
model_config = ModelConfig()
if 'model' in config:
    for key, value in config['model'].items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)

# Create generation config
generation_config = GenerationConfig()
if 'generation' in config:
    for key, value in config['generation'].items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize model
model = OsuTransformer(model_config)

# Create generator
generator = ReplayGenerator(
    model=model,
    config=generation_config,
    model_config=model_config,
    device=device,
    logger=logger
)

# Load checkpoint
checkpoint_path = Path('final_model.pt')
if checkpoint_path.exists():
    generator.load_checkpoint(checkpoint_path)
    logger.info("Model loaded successfully")
else:
    logger.warning(f"Checkpoint not found: {checkpoint_path}")
    logger.warning("Using untrained model - this will generate random outputs")

# Generate a replay
beatmap_path = Path('testmap.osu')
if beatmap_path.exists():
    try:
        result = generator.generate_replay(
            beatmap_path=beatmap_path,
            target_accuracy=0.95
        )
        
        logger.info(f"Replay generated successfully in {result.generation_time:.2f}s")
        logger.info(f"Generated {len(result.cursor_positions)} cursor positions")
        logger.info(f"Generated {len(result.key_presses)} key presses")
        
        # Save the result (you would need to implement export functionality)
        # For now, just save the raw data
        import numpy as np
        np.savez('generated_replay.npz',
                cursor_positions=result.cursor_positions,
                key_presses=result.key_presses,
                timestamps=result.timestamps,
                metadata=result.metadata)
        
        logger.info("Replay data saved to generated_replay.npz")
        
    except Exception as e:
        logger.error(f"Failed to generate replay: {e}")
        exit(1)
else:
    logger.error(f"Beatmap file not found: {beatmap_path}")
    exit(1)