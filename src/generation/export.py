"""Export utilities for saving replays in osu! format."""

import struct
import lzma
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np
from pathlib import Path
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import GenerationResult


class OSRExporter:
    """Exporter for saving replays in osu! .osr format."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def export(self, result: 'GenerationResult', 
              beatmap_data: Dict[str, Any],
              output_path: Union[str, Path],
              player_name: str = "AI Player",
              mods: int = 0,
              score: int = 1000000,
              max_combo: int = 0,
              perfect: bool = True,
              accuracy: float = 100.0) -> bool:
        """Export a replay to .osr format.
        
        Args:
            result: Generated replay data
            beatmap_data: Beatmap information
            output_path: Path to save the .osr file
            player_name: Name of the player
            mods: Mod combination (bitwise)
            score: Total score
            max_combo: Maximum combo achieved
            perfect: Whether the play was perfect
            accuracy: Accuracy percentage
            
        Returns:
            True if export was successful
        """
        try:
            self.logger.info(f"Exporting replay to {output_path}")
            
            # Prepare replay data
            replay_data = self._prepare_replay_data(result)
            
            # Create .osr file
            with open(output_path, 'wb') as f:
                # Write header
                self._write_header(f, beatmap_data, player_name, mods, 
                                 score, max_combo, perfect, accuracy)
                
                # Write compressed replay data
                self._write_replay_data(f, replay_data)
                
                # Write online score ID (0 for offline)
                f.write(struct.pack('<Q', 0))
            
            self.logger.info("Replay exported successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export replay: {e}")
            return False
    
    def _prepare_replay_data(self, result: 'GenerationResult') -> str:
        """Convert generation result to osu! replay data format.
        
        Args:
            result: Generated replay data
            
        Returns:
            Replay data string in osu! format
        """
        replay_frames = []
        
        # Convert to osu! replay format
        for i in range(len(result.timestamps)):
            if i == 0:
                time_delta = int(result.timestamps[i])
            else:
                time_delta = int(result.timestamps[i] - result.timestamps[i-1])
            
            # Cursor position
            x = float(result.cursor_positions[i, 0])
            y = float(result.cursor_positions[i, 1])
            
            # Key state (convert to osu! format)
            keys = self._convert_key_state(result.key_presses[i])
            
            # Format: time_delta|x|y|keys
            frame = f"{time_delta}|{x}|{y}|{keys}"
            replay_frames.append(frame)
        
        # Join frames with commas
        replay_data = ','.join(replay_frames)
        
        # Add seed (random number for RNG)
        replay_data += f",{self._generate_seed()}"
        
        return replay_data
    
    def _convert_key_state(self, key_state: np.ndarray) -> int:
        """Convert key state array to osu! key format.
        
        Args:
            key_state: Array of shape [4] representing [K1, K2, M1, M2]
            
        Returns:
            Integer representing key state in osu! format
        """
        keys = 0
        
        # osu! key mapping:
        # 1 = M1 (left mouse)
        # 2 = M2 (right mouse) 
        # 4 = K1 (keyboard key 1)
        # 8 = K2 (keyboard key 2)
        # 16 = Smoke
        
        if key_state[2] > 0.5:  # M1
            keys |= 1
        if key_state[3] > 0.5:  # M2
            keys |= 2
        if key_state[0] > 0.5:  # K1
            keys |= 4
        if key_state[1] > 0.5:  # K2
            keys |= 8
        
        return keys
    
    def _generate_seed(self) -> int:
        """Generate a random seed for the replay.
        
        Returns:
            Random seed value
        """
        import random
        return random.randint(0, 2**31 - 1)
    
    def _write_header(self, f, beatmap_data: Dict[str, Any], 
                     player_name: str, mods: int, score: int,
                     max_combo: int, perfect: bool, accuracy: float):
        """Write the .osr file header.
        
        Args:
            f: File object to write to
            beatmap_data: Beatmap information
            player_name: Player name
            mods: Mod combination
            score: Total score
            max_combo: Maximum combo
            perfect: Whether play was perfect
            accuracy: Accuracy percentage
        """
        # Game mode (0 = osu!)
        f.write(struct.pack('<B', 0))
        
        # Game version (20210520 as example)
        f.write(struct.pack('<I', 20210520))
        
        # Beatmap MD5 hash
        beatmap_hash = beatmap_data.get('md5_hash', '0' * 32)
        self._write_string(f, beatmap_hash)
        
        # Player name
        self._write_string(f, player_name)
        
        # Replay MD5 hash (will be calculated later)
        replay_hash = '0' * 32  # Placeholder
        self._write_string(f, replay_hash)
        
        # Hit statistics
        # For simplicity, we'll calculate basic stats
        total_objects = beatmap_data.get('hit_object_count', 100)
        hit_300 = int(total_objects * (accuracy / 100))
        hit_100 = total_objects - hit_300
        hit_50 = 0
        hit_geki = 0  # Perfect hits in Taiko/Mania
        hit_katu = 0  # Good hits in Taiko/Mania
        hit_miss = 0
        
        f.write(struct.pack('<H', hit_300))  # 300s
        f.write(struct.pack('<H', hit_100))  # 100s
        f.write(struct.pack('<H', hit_50))   # 50s
        f.write(struct.pack('<H', hit_geki)) # Gekis
        f.write(struct.pack('<H', hit_katu)) # Katus
        f.write(struct.pack('<H', hit_miss)) # Misses
        
        # Total score
        f.write(struct.pack('<I', score))
        
        # Max combo
        if max_combo == 0:
            max_combo = total_objects  # Assume FC if not specified
        f.write(struct.pack('<H', max_combo))
        
        # Perfect combo flag
        f.write(struct.pack('<B', 1 if perfect else 0))
        
        # Mods used
        f.write(struct.pack('<I', mods))
        
        # Life bar graph (empty for now)
        self._write_string(f, "")
        
        # Timestamp (Windows ticks)
        timestamp = self._get_windows_ticks()
        f.write(struct.pack('<Q', timestamp))
    
    def _write_replay_data(self, f, replay_data: str):
        """Write compressed replay data to file.
        
        Args:
            f: File object to write to
            replay_data: Replay data string
        """
        # Compress replay data using LZMA
        compressed_data = lzma.compress(
            replay_data.encode('utf-8'),
            format=lzma.FORMAT_ALONE,
            preset=6
        )
        
        # Write length of compressed data
        f.write(struct.pack('<I', len(compressed_data)))
        
        # Write compressed data
        f.write(compressed_data)
    
    def _write_string(self, f, string: str):
        """Write a string in osu! format.
        
        Args:
            f: File object to write to
            string: String to write
        """
        if not string:
            # Empty string
            f.write(struct.pack('<B', 0x00))
        else:
            # Non-empty string
            f.write(struct.pack('<B', 0x0b))
            
            # Encode string as UTF-8
            encoded = string.encode('utf-8')
            
            # Write length as ULEB128
            self._write_uleb128(f, len(encoded))
            
            # Write string data
            f.write(encoded)
    
    def _write_uleb128(self, f, value: int):
        """Write an unsigned integer in ULEB128 format.
        
        Args:
            f: File object to write to
            value: Integer value to write
        """
        while value >= 0x80:
            f.write(struct.pack('<B', (value & 0x7F) | 0x80))
            value >>= 7
        f.write(struct.pack('<B', value & 0x7F))
    
    def _get_windows_ticks(self) -> int:
        """Get current time as Windows ticks.
        
        Returns:
            Current time in Windows ticks format
        """
        # Windows ticks = 100-nanosecond intervals since January 1, 1601
        # Unix timestamp = seconds since January 1, 1970
        # Difference = 11644473600 seconds
        
        import time
        unix_timestamp = time.time()
        windows_ticks = int((unix_timestamp + 11644473600) * 10000000)
        return windows_ticks
    
    def calculate_replay_hash(self, replay_data: str, 
                            beatmap_hash: str, 
                            player_name: str,
                            score: int) -> str:
        """Calculate MD5 hash for the replay.
        
        Args:
            replay_data: Replay data string
            beatmap_hash: Beatmap MD5 hash
            player_name: Player name
            score: Total score
            
        Returns:
            MD5 hash string
        """
        # Create hash input string
        hash_input = f"{beatmap_hash}{player_name}{replay_data}{score}"
        
        # Calculate MD5 hash
        md5_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        
        return md5_hash
    
    def export_with_metadata(self, result: 'GenerationResult',
                           beatmap_data: Dict[str, Any],
                           output_path: Union[str, Path],
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Export replay with custom metadata.
        
        Args:
            result: Generated replay data
            beatmap_data: Beatmap information
            output_path: Path to save the .osr file
            metadata: Custom metadata for the replay
            
        Returns:
            True if export was successful
        """
        if metadata is None:
            metadata = {}
        
        # Extract metadata with defaults
        player_name = metadata.get('player_name', 'AI Player')
        mods = metadata.get('mods', 0)
        score = metadata.get('score', 1000000)
        max_combo = metadata.get('max_combo', 0)
        perfect = metadata.get('perfect', True)
        accuracy = metadata.get('accuracy', 100.0)
        
        return self.export(
            result=result,
            beatmap_data=beatmap_data,
            output_path=output_path,
            player_name=player_name,
            mods=mods,
            score=score,
            max_combo=max_combo,
            perfect=perfect,
            accuracy=accuracy
        )
    
    def validate_export(self, osr_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate an exported .osr file.
        
        Args:
            osr_path: Path to the .osr file
            
        Returns:
            Validation results
        """
        try:
            with open(osr_path, 'rb') as f:
                # Read and validate header
                game_mode = struct.unpack('<B', f.read(1))[0]
                game_version = struct.unpack('<I', f.read(4))[0]
                
                # Read strings
                beatmap_hash = self._read_string(f)
                player_name = self._read_string(f)
                replay_hash = self._read_string(f)
                
                # Read hit statistics
                hit_300 = struct.unpack('<H', f.read(2))[0]
                hit_100 = struct.unpack('<H', f.read(2))[0]
                hit_50 = struct.unpack('<H', f.read(2))[0]
                hit_geki = struct.unpack('<H', f.read(2))[0]
                hit_katu = struct.unpack('<H', f.read(2))[0]
                hit_miss = struct.unpack('<H', f.read(2))[0]
                
                score = struct.unpack('<I', f.read(4))[0]
                max_combo = struct.unpack('<H', f.read(2))[0]
                perfect = struct.unpack('<B', f.read(1))[0]
                mods = struct.unpack('<I', f.read(4))[0]
                
                life_bar = self._read_string(f)
                timestamp = struct.unpack('<Q', f.read(8))[0]
                
                # Read replay data length
                replay_data_length = struct.unpack('<I', f.read(4))[0]
                
                return {
                    'valid': True,
                    'game_mode': game_mode,
                    'game_version': game_version,
                    'beatmap_hash': beatmap_hash,
                    'player_name': player_name,
                    'replay_hash': replay_hash,
                    'score': score,
                    'max_combo': max_combo,
                    'perfect': bool(perfect),
                    'mods': mods,
                    'replay_data_length': replay_data_length,
                    'file_size': Path(osr_path).stat().st_size
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _read_string(self, f) -> str:
        """Read a string from osu! format.
        
        Args:
            f: File object to read from
            
        Returns:
            Decoded string
        """
        indicator = struct.unpack('<B', f.read(1))[0]
        
        if indicator == 0x00:
            return ""
        elif indicator == 0x0b:
            # Read ULEB128 length
            length = self._read_uleb128(f)
            
            # Read string data
            string_data = f.read(length)
            return string_data.decode('utf-8')
        else:
            raise ValueError(f"Invalid string indicator: {indicator}")
    
    def _read_uleb128(self, f) -> int:
        """Read an unsigned integer in ULEB128 format.
        
        Args:
            f: File object to read from
            
        Returns:
            Decoded integer value
        """
        result = 0
        shift = 0
        
        while True:
            byte = struct.unpack('<B', f.read(1))[0]
            result |= (byte & 0x7F) << shift
            
            if (byte & 0x80) == 0:
                break
                
            shift += 7
        
        return result


def create_replay_metadata(beatmap_data: Dict[str, Any],
                         generation_config: Dict[str, Any],
                         target_accuracy: float = 95.0,
                         target_score: Optional[int] = None) -> Dict[str, Any]:
    """Create metadata for replay export.
    
    Args:
        beatmap_data: Beatmap information
        generation_config: Generation configuration
        target_accuracy: Target accuracy percentage
        target_score: Target score (calculated if None)
        
    Returns:
        Metadata dictionary
    """
    # Calculate estimated score if not provided
    if target_score is None:
        hit_objects = beatmap_data.get('hit_object_count', 100)
        base_score = hit_objects * 300  # Assume all 300s
        target_score = int(base_score * (target_accuracy / 100))
    
    # Calculate combo
    max_combo = beatmap_data.get('max_combo', beatmap_data.get('hit_object_count', 100))
    
    # Determine if perfect (FC)
    perfect = target_accuracy >= 99.0
    
    return {
        'player_name': generation_config.get('player_name', 'AI Player'),
        'mods': generation_config.get('mods', 0),
        'score': target_score,
        'max_combo': max_combo if perfect else int(max_combo * 0.9),
        'perfect': perfect,
        'accuracy': target_accuracy
    }