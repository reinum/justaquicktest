#!/usr/bin/env python3
"""
Convert generated .npz replay file to .osr format for osu!
"""

import numpy as np
import logging
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.generation.export import OSRExporter
from src.generation.generator import GenerationResult
from src.data.beatmap_parser import BeatmapParser

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def load_npz_replay(npz_path: str) -> GenerationResult:
    """Load replay data from .npz file.
    
    Args:
        npz_path: Path to the .npz file
        
    Returns:
        GenerationResult object
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Handle metadata safely
    metadata = {}
    if 'metadata' in data:
        try:
            metadata_item = data['metadata'].item()
            if isinstance(metadata_item, dict):
                metadata = metadata_item
        except:
            metadata = {}
    
    return GenerationResult(
        cursor_positions=data['cursor_positions'],
        key_presses=data['key_presses'],
        timestamps=data['timestamps'],
        metadata=metadata,
        generation_time=0.0,
        confidence_scores=data.get('confidence_scores', None)
    )

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config('config/default.yaml')
    
    # Paths
    npz_path = 'generated_replay.npz'
    osr_path = 'generated_replay.osr'
    beatmap_path = 'testmap.osu'
    
    try:
        # Check if files exist
        if not Path(npz_path).exists():
            logger.error(f"NPZ file not found: {npz_path}")
            logger.info("Please run generate_replay.py first to create the .npz file")
            return
            
        if not Path(beatmap_path).exists():
            logger.error(f"Beatmap file not found: {beatmap_path}")
            return
        
        # Load replay data from .npz
        logger.info(f"Loading replay data from {npz_path}")
        result = load_npz_replay(npz_path)
        
        # Parse beatmap
        logger.info(f"Parsing beatmap: {beatmap_path}")
        beatmap_parser = BeatmapParser()
        beatmap_data = beatmap_parser.parse_beatmap(beatmap_path)
        
        # Convert to dictionary format expected by exporter
        beatmap_dict = {
            'hit_objects': beatmap_data,
            'path': beatmap_path,
            'beatmap_id': 0,  # Default for local beatmaps
            'beatmap_set_id': 0,
            'title': 'Test Map',
            'artist': 'Unknown',
            'creator': 'Unknown',
            'version': 'AI Generated',
            'md5_hash': '00000000000000000000000000000000'  # Placeholder
        }
        
        # Export to .osr format
        logger.info(f"Exporting to {osr_path}")
        exporter = OSRExporter(logger)
        
        success = exporter.export(
            result=result,
            beatmap_data=beatmap_dict,
            output_path=osr_path,
            player_name="AI Player",
            mods=0,  # No mods
            score=1000000,  # Default score
            max_combo=len(beatmap_data) if beatmap_data else 0,  # Use total hit objects as combo
            perfect=True,
            accuracy=100.0
        )
        
        if success:
            logger.info(f"Successfully converted {npz_path} to {osr_path}")
            logger.info(f"You can now load {osr_path} into osu!")
            logger.info("\nTo use the replay:")
            logger.info("1. Copy the .osr file to your osu!/Replays/ folder")
            logger.info("2. Open osu! and go to the beatmap")
            logger.info("3. Press F2 to open local rankings and find your replay")
        else:
            logger.error("Failed to convert replay")
            
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()