"""Replay format conversion utilities for the osu! AI replay maker."""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from ..data.replay_parser import ReplayDataLoader
from ..generation.export import OSRExporter


class ReplayConverter:
    """Converts between different replay formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.replay_parser = ReplayDataLoader()
        self.osr_exporter = OSRExporter()
    
    def convert_to_json(self, input_path: str, output_path: str) -> None:
        """Convert replay to JSON format.
        
        Args:
            input_path: Path to input replay file (.osr)
            output_path: Path to output JSON file
        """
        self.logger.info(f"Converting {input_path} to JSON: {output_path}")
        
        try:
            # Parse the replay
            replay_data = self.replay_parser.parse_replay(input_path)
            
            # Convert to JSON-serializable format
            json_data = self._prepare_for_json(replay_data)
            
            # Write to JSON file
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            self.logger.info(f"Successfully converted to JSON: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to convert to JSON: {e}")
            raise
    
    def convert_to_csv(self, input_path: str, output_path: str) -> None:
        """Convert replay to CSV format.
        
        Args:
            input_path: Path to input replay file (.osr)
            output_path: Path to output CSV file
        """
        self.logger.info(f"Converting {input_path} to CSV: {output_path}")
        
        try:
            # Parse the replay
            replay_data = self.replay_parser.parse_replay(input_path)
            
            # Extract replay events
            replay_events = replay_data.get('replay_data', [])
            
            if not replay_events:
                raise ValueError("No replay events found")
            
            # Write to CSV file
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                if replay_events:
                    header = list(replay_events[0].keys())
                    writer.writerow(header)
                    
                    # Write data rows
                    for event in replay_events:
                        writer.writerow([event.get(key, '') for key in header])
            
            self.logger.info(f"Successfully converted to CSV: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to convert to CSV: {e}")
            raise
    
    def convert_from_json(self, input_path: str, output_path: str) -> None:
        """Convert JSON format back to .osr replay.
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output replay file (.osr)
        """
        self.logger.info(f"Converting JSON {input_path} to OSR: {output_path}")
        
        try:
            # Load JSON data
            with open(input_path, 'r') as f:
                json_data = json.load(f)
            
            # Convert back to replay format
            replay_data = self._prepare_from_json(json_data)
            
            # Export as .osr file
            self.osr_exporter.export_replay(
                replay_data=replay_data,
                output_path=output_path
            )
            
            self.logger.info(f"Successfully converted from JSON: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to convert from JSON: {e}")
            raise
    
    def convert_from_csv(self, input_path: str, output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Convert CSV format back to .osr replay.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output replay file (.osr)
            metadata: Optional metadata for the replay
        """
        self.logger.info(f"Converting CSV {input_path} to OSR: {output_path}")
        
        try:
            # Load CSV data
            replay_events = []
            with open(input_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string values back to appropriate types
                    event = {}
                    for key, value in row.items():
                        if key in ['time_delta', 'x', 'y']:
                            event[key] = float(value) if value else 0.0
                        elif key in ['keys']:
                            event[key] = int(value) if value else 0
                        else:
                            event[key] = value
                    replay_events.append(event)
            
            # Create replay data structure
            replay_data = {
                'replay_data': replay_events,
                'game_mode': metadata.get('game_mode', 0) if metadata else 0,
                'game_version': metadata.get('game_version', 20210520) if metadata else 20210520,
                'beatmap_hash': metadata.get('beatmap_hash', '') if metadata else '',
                'player_name': metadata.get('player_name', 'AI') if metadata else 'AI',
                'replay_hash': metadata.get('replay_hash', '') if metadata else '',
                'count_300': metadata.get('count_300', 0) if metadata else 0,
                'count_100': metadata.get('count_100', 0) if metadata else 0,
                'count_50': metadata.get('count_50', 0) if metadata else 0,
                'count_geki': metadata.get('count_geki', 0) if metadata else 0,
                'count_katu': metadata.get('count_katu', 0) if metadata else 0,
                'count_miss': metadata.get('count_miss', 0) if metadata else 0,
                'total_score': metadata.get('total_score', 0) if metadata else 0,
                'max_combo': metadata.get('max_combo', 0) if metadata else 0,
                'perfect': metadata.get('perfect', False) if metadata else False,
                'mods': metadata.get('mods', 0) if metadata else 0,
                'life_bar': metadata.get('life_bar', []) if metadata else [],
                'timestamp': metadata.get('timestamp', 0) if metadata else 0
            }
            
            # Export as .osr file
            self.osr_exporter.export_replay(
                replay_data=replay_data,
                output_path=output_path
            )
            
            self.logger.info(f"Successfully converted from CSV: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to convert from CSV: {e}")
            raise
    
    def _prepare_for_json(self, replay_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare replay data for JSON serialization."""
        json_data = {}
        
        for key, value in replay_data.items():
            if key == 'replay_data' and isinstance(value, list):
                # Convert replay events to JSON-friendly format
                json_data[key] = [
                    {k: v for k, v in event.items() if v is not None}
                    for event in value
                ]
            elif hasattr(value, '__dict__'):
                # Convert dataclass or object to dict
                json_data[key] = asdict(value) if hasattr(value, '__dataclass_fields__') else value.__dict__
            else:
                json_data[key] = value
        
        return json_data
    
    def _prepare_from_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare JSON data for replay format."""
        replay_data = {}
        
        for key, value in json_data.items():
            if key == 'replay_data' and isinstance(value, list):
                # Ensure replay events have correct format
                replay_events = []
                for event in value:
                    if isinstance(event, dict):
                        replay_events.append(event)
                    else:
                        # Handle other formats if needed
                        replay_events.append({'raw_data': event})
                replay_data[key] = replay_events
            else:
                replay_data[key] = value
        
        return replay_data
    
    def batch_convert(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str,
        input_pattern: str = "*.osr"
    ) -> None:
        """Convert multiple replay files in batch.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output files
            output_format: Target format ('json', 'csv', 'osr')
            input_pattern: File pattern to match
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find input files
        input_files = list(input_path.glob(input_pattern))
        
        self.logger.info(f"Converting {len(input_files)} files to {output_format}")
        
        for input_file in input_files:
            try:
                # Determine output file path
                output_file = output_path / f"{input_file.stem}.{output_format}"
                
                # Convert based on format
                if output_format == 'json':
                    self.convert_to_json(str(input_file), str(output_file))
                elif output_format == 'csv':
                    self.convert_to_csv(str(input_file), str(output_file))
                elif output_format == 'osr':
                    if input_file.suffix == '.json':
                        self.convert_from_json(str(input_file), str(output_file))
                    elif input_file.suffix == '.csv':
                        self.convert_from_csv(str(input_file), str(output_file))
                    else:
                        self.logger.warning(f"Cannot convert {input_file} to OSR format")
                        continue
                else:
                    self.logger.error(f"Unsupported output format: {output_format}")
                    continue
                
                self.logger.info(f"Converted: {input_file.name} -> {output_file.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to convert {input_file}: {e}")
                continue
        
        self.logger.info("Batch conversion completed")
    
    def get_replay_info(self, replay_path: str) -> Dict[str, Any]:
        """Get basic information about a replay file.
        
        Args:
            replay_path: Path to replay file
            
        Returns:
            Dictionary with replay information
        """
        try:
            replay_data = self.replay_parser.parse_replay(replay_path)
            
            info = {
                'file_path': replay_path,
                'file_size': Path(replay_path).stat().st_size,
                'player_name': replay_data.get('player_name', 'Unknown'),
                'beatmap_hash': replay_data.get('beatmap_hash', ''),
                'game_mode': replay_data.get('game_mode', 0),
                'total_score': replay_data.get('total_score', 0),
                'max_combo': replay_data.get('max_combo', 0),
                'count_300': replay_data.get('count_300', 0),
                'count_100': replay_data.get('count_100', 0),
                'count_50': replay_data.get('count_50', 0),
                'count_miss': replay_data.get('count_miss', 0),
                'mods': replay_data.get('mods', 0),
                'replay_length': len(replay_data.get('replay_data', [])),
                'timestamp': replay_data.get('timestamp', 0)
            }
            
            # Calculate accuracy
            total_hits = info['count_300'] + info['count_100'] + info['count_50'] + info['count_miss']
            if total_hits > 0:
                accuracy = (info['count_300'] * 300 + info['count_100'] * 100 + info['count_50'] * 50) / (total_hits * 300)
                info['accuracy'] = accuracy
            else:
                info['accuracy'] = 0.0
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get replay info: {e}")
            return {'file_path': replay_path, 'error': str(e)}


def convert_replay(
    input_path: str,
    output_path: str,
    output_format: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to convert a single replay file.
    
    Args:
        input_path: Path to input replay file
        output_path: Path to output file
        output_format: Target format ('json', 'csv', 'osr')
        metadata: Optional metadata for conversion
    """
    converter = ReplayConverter()
    
    if output_format == 'json':
        converter.convert_to_json(input_path, output_path)
    elif output_format == 'csv':
        converter.convert_to_csv(input_path, output_path)
    elif output_format == 'osr':
        input_ext = Path(input_path).suffix.lower()
        if input_ext == '.json':
            converter.convert_from_json(input_path, output_path)
        elif input_ext == '.csv':
            converter.convert_from_csv(input_path, output_path, metadata)
        else:
            raise ValueError(f"Cannot convert from {input_ext} to OSR")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def batch_convert_replays(
    input_dir: str,
    output_dir: str,
    output_format: str,
    input_pattern: str = "*.osr"
) -> None:
    """Convenience function for batch conversion.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output files
        output_format: Target format ('json', 'csv', 'osr')
        input_pattern: File pattern to match
    """
    converter = ReplayConverter()
    converter.batch_convert(input_dir, output_dir, output_format, input_pattern)