"""Beatmap parser module that wraps the C# OsuParsers library."""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add the osuparse directory to the path
project_root = Path(__file__).parent.parent.parent
osuparse_path = project_root / "osuparse"
sys.path.insert(0, str(osuparse_path))

try:
    from osuparser import findHitObject
except ImportError as e:
    logging.warning(f"Could not import C# beatmap parser: {e}")
    findHitObject = None


class BeatmapParser:
    """Parser for osu! beatmap files using C# OsuParsers library."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the beatmap parser.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        if findHitObject is None:
            self.logger.warning("C# beatmap parser not available. Some features may be limited.")
    
    def parse_beatmap(self, beatmap_path: str) -> Optional[List[Dict[str, Any]]]:
        """Parse a beatmap file and extract hit objects.
        
        Args:
            beatmap_path: Path to the .osu beatmap file
            
        Returns:
            List of hit object dictionaries, or None if parsing fails
        """
        if findHitObject is None:
            self.logger.error("C# beatmap parser not available")
            return None
            
        try:
            hit_objects = findHitObject(beatmap_path)
            self.logger.debug(f"Parsed {len(hit_objects)} hit objects from {beatmap_path}")
            return hit_objects
        except Exception as e:
            self.logger.error(f"Failed to parse beatmap {beatmap_path}: {e}")
            return None
    
    def get_hit_objects(self, beatmap_path: str) -> List[Dict[str, Any]]:
        """Get hit objects from a beatmap file.
        
        Args:
            beatmap_path: Path to the .osu beatmap file
            
        Returns:
            List of hit object dictionaries (empty list if parsing fails)
        """
        hit_objects = self.parse_beatmap(beatmap_path)
        return hit_objects if hit_objects is not None else []
    
    def is_available(self) -> bool:
        """Check if the C# beatmap parser is available.
        
        Returns:
            True if the parser is available, False otherwise
        """
        return findHitObject is not None
    
    def get_slider_info(self, beatmap_path: str) -> List[Dict[str, Any]]:
        """Get detailed slider information from a beatmap.
        
        Args:
            beatmap_path: Path to the .osu beatmap file
            
        Returns:
            List of slider dictionaries with detailed curve information
        """
        hit_objects = self.get_hit_objects(beatmap_path)
        sliders = [obj for obj in hit_objects if obj.get('Type') == 'slider']
        
        self.logger.debug(f"Found {len(sliders)} sliders in {beatmap_path}")
        return sliders
    
    def get_timing_points(self, beatmap_path: str) -> List[Dict[str, Any]]:
        """Get timing points from a beatmap (placeholder for future implementation).
        
        Args:
            beatmap_path: Path to the .osu beatmap file
            
        Returns:
            List of timing point dictionaries (currently empty)
        """
        # This would require extending the C# parser to extract timing points
        self.logger.warning("Timing point extraction not yet implemented")
        return []
    
    def validate_beatmap(self, beatmap_path: str) -> bool:
        """Validate that a beatmap file can be parsed successfully.
        
        Args:
            beatmap_path: Path to the .osu beatmap file
            
        Returns:
            True if the beatmap is valid and parseable, False otherwise
        """
        if not os.path.exists(beatmap_path):
            self.logger.error(f"Beatmap file not found: {beatmap_path}")
            return False
            
        hit_objects = self.parse_beatmap(beatmap_path)
        return hit_objects is not None and len(hit_objects) > 0