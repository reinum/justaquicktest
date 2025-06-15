"""Dataset validation utilities for the osu! AI replay maker."""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from ..data.replay_parser import ReplayDataLoader
from ..data.beatmap_parser import BeatmapParser


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the dataset."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'file', 'data', 'format', 'consistency'
    message: str
    file_path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    is_valid: bool
    total_files: int
    valid_files: int
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    
    def save(self, output_path: str) -> None:
        """Save validation report to JSON file."""
        report_data = {
            'is_valid': self.is_valid,
            'total_files': self.total_files,
            'valid_files': self.valid_files,
            'issues': [
                {
                    'severity': issue.severity,
                    'category': issue.category,
                    'message': issue.message,
                    'file_path': issue.file_path,
                    'details': issue.details
                }
                for issue in self.issues
            ],
            'statistics': self.statistics
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)


class DatasetValidator:
    """Validates osu! replay datasets for training."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.replay_parser = ReplayDataLoader()
        self.beatmap_parser = BeatmapParser()
        self.issues: List[ValidationIssue] = []
    
    def validate_dataset(self, dataset_path: str) -> ValidationReport:
        """Validate an entire dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            ValidationReport with results
        """
        self.logger.info(f"Validating dataset: {dataset_path}")
        self.issues = []
        
        dataset_dir = Path(dataset_path)
        
        # Check if dataset directory exists
        if not dataset_dir.exists():
            self.issues.append(ValidationIssue(
                severity='error',
                category='file',
                message=f"Dataset directory does not exist: {dataset_path}"
            ))
            return ValidationReport(
                is_valid=False,
                total_files=0,
                valid_files=0,
                issues=self.issues,
                statistics={}
            )
        
        # Validate directory structure
        self._validate_directory_structure(dataset_dir)
        
        # Find and validate files
        replay_files = list(dataset_dir.glob('**/*.osr'))
        beatmap_files = list(dataset_dir.glob('**/*.osu'))
        
        self.logger.info(f"Found {len(replay_files)} replay files and {len(beatmap_files)} beatmap files")
        
        # Validate index file if it exists
        index_file = dataset_dir / 'index.csv'
        if index_file.exists():
            self._validate_index_file(index_file, replay_files, beatmap_files)
        else:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='file',
                message="No index.csv file found"
            ))
        
        # Validate individual files
        valid_replays = self._validate_replay_files(replay_files)
        valid_beatmaps = self._validate_beatmap_files(beatmap_files)
        
        # Validate data consistency
        self._validate_data_consistency(replay_files, beatmap_files)
        
        # Generate statistics
        statistics = self._generate_statistics(replay_files, beatmap_files)
        
        # Determine overall validity
        error_count = sum(1 for issue in self.issues if issue.severity == 'error')
        is_valid = error_count == 0 and len(replay_files) > 0
        
        return ValidationReport(
            is_valid=is_valid,
            total_files=len(replay_files) + len(beatmap_files),
            valid_files=valid_replays + valid_beatmaps,
            issues=self.issues,
            statistics=statistics
        )
    
    def _validate_directory_structure(self, dataset_dir: Path) -> None:
        """Validate expected directory structure."""
        expected_dirs = ['replays', 'beatmaps']
        
        for dir_name in expected_dirs:
            dir_path = dataset_dir / dir_name
            if not dir_path.exists():
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='file',
                    message=f"Expected directory not found: {dir_name}"
                ))
    
    def _validate_index_file(self, index_file: Path, replay_files: List[Path], beatmap_files: List[Path]) -> None:
        """Validate the index.csv file."""
        try:
            df = pd.read_csv(index_file)
            
            # Check required columns
            required_columns = ['replay_path', 'beatmap_path', 'player_name', 'mods']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.issues.append(ValidationIssue(
                    severity='error',
                    category='format',
                    message=f"Missing required columns in index.csv: {missing_columns}"
                ))
            
            # Check file references
            for idx, row in df.iterrows():
                replay_path = Path(row['replay_path'])
                beatmap_path = Path(row['beatmap_path'])
                
                if not replay_path.exists():
                    self.issues.append(ValidationIssue(
                        severity='error',
                        category='file',
                        message=f"Replay file referenced in index but not found: {replay_path}",
                        details={'row': idx}
                    ))
                
                if not beatmap_path.exists():
                    self.issues.append(ValidationIssue(
                        severity='error',
                        category='file',
                        message=f"Beatmap file referenced in index but not found: {beatmap_path}",
                        details={'row': idx}
                    ))
        
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity='error',
                category='format',
                message=f"Failed to read index.csv: {e}"
            ))
    
    def _validate_replay_files(self, replay_files: List[Path]) -> int:
        """Validate replay files."""
        valid_count = 0
        
        for replay_file in replay_files:
            try:
                # Try to parse the replay
                replay_data = self.replay_parser.parse_replay(str(replay_file))
                
                # Basic validation checks
                if not replay_data.get('replay_data'):
                    self.issues.append(ValidationIssue(
                        severity='error',
                        category='data',
                        message="Replay contains no replay data",
                        file_path=str(replay_file)
                    ))
                    continue
                
                # Check replay data format
                replay_events = replay_data['replay_data']
                if len(replay_events) < 10:  # Minimum reasonable length
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='data',
                        message=f"Replay is very short ({len(replay_events)} events)",
                        file_path=str(replay_file)
                    ))
                
                # Check for reasonable coordinate ranges
                x_coords = [event.get('x', 0) for event in replay_events if 'x' in event]
                y_coords = [event.get('y', 0) for event in replay_events if 'y' in event]
                
                if x_coords and (min(x_coords) < -100 or max(x_coords) > 700):
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='data',
                        message="Replay contains unusual X coordinates",
                        file_path=str(replay_file),
                        details={'x_range': [min(x_coords), max(x_coords)]}
                    ))
                
                if y_coords and (min(y_coords) < -100 or max(y_coords) > 500):
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='data',
                        message="Replay contains unusual Y coordinates",
                        file_path=str(replay_file),
                        details={'y_range': [min(y_coords), max(y_coords)]}
                    ))
                
                valid_count += 1
                
            except Exception as e:
                self.issues.append(ValidationIssue(
                    severity='error',
                    category='format',
                    message=f"Failed to parse replay: {e}",
                    file_path=str(replay_file)
                ))
        
        return valid_count
    
    def _validate_beatmap_files(self, beatmap_files: List[Path]) -> int:
        """Validate beatmap files."""
        valid_count = 0
        
        for beatmap_file in beatmap_files:
            try:
                # Try to parse the beatmap
                beatmap_data = self.beatmap_parser.parse_beatmap(str(beatmap_file))
                
                # Basic validation checks
                if not beatmap_data.get('hit_objects'):
                    self.issues.append(ValidationIssue(
                        severity='error',
                        category='data',
                        message="Beatmap contains no hit objects",
                        file_path=str(beatmap_file)
                    ))
                    continue
                
                # Check for reasonable number of hit objects
                hit_objects = beatmap_data['hit_objects']
                if len(hit_objects) < 10:
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='data',
                        message=f"Beatmap has very few hit objects ({len(hit_objects)})",
                        file_path=str(beatmap_file)
                    ))
                
                # Check timing points
                if not beatmap_data.get('timing_points'):
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='data',
                        message="Beatmap has no timing points",
                        file_path=str(beatmap_file)
                    ))
                
                valid_count += 1
                
            except Exception as e:
                self.issues.append(ValidationIssue(
                    severity='error',
                    category='format',
                    message=f"Failed to parse beatmap: {e}",
                    file_path=str(beatmap_file)
                ))
        
        return valid_count
    
    def _validate_data_consistency(self, replay_files: List[Path], beatmap_files: List[Path]) -> None:
        """Validate consistency between replays and beatmaps."""
        # Check if we have a reasonable ratio of replays to beatmaps
        if len(replay_files) == 0:
            self.issues.append(ValidationIssue(
                severity='error',
                category='consistency',
                message="No replay files found in dataset"
            ))
        
        if len(beatmap_files) == 0:
            self.issues.append(ValidationIssue(
                severity='error',
                category='consistency',
                message="No beatmap files found in dataset"
            ))
        
        # Warn if ratio seems off
        if len(replay_files) > 0 and len(beatmap_files) > 0:
            ratio = len(replay_files) / len(beatmap_files)
            if ratio < 0.1:
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='consistency',
                    message=f"Very few replays per beatmap (ratio: {ratio:.2f})"
                ))
            elif ratio > 100:
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='consistency',
                    message=f"Very many replays per beatmap (ratio: {ratio:.2f})"
                ))
    
    def _generate_statistics(self, replay_files: List[Path], beatmap_files: List[Path]) -> Dict[str, Any]:
        """Generate dataset statistics."""
        stats = {
            'total_replays': len(replay_files),
            'total_beatmaps': len(beatmap_files),
            'replay_to_beatmap_ratio': len(replay_files) / max(len(beatmap_files), 1),
            'total_size_mb': sum(f.stat().st_size for f in replay_files + beatmap_files) / (1024 * 1024),
            'avg_replay_size_kb': np.mean([f.stat().st_size for f in replay_files]) / 1024 if replay_files else 0,
            'avg_beatmap_size_kb': np.mean([f.stat().st_size for f in beatmap_files]) / 1024 if beatmap_files else 0,
            'issues_by_severity': {
                'error': sum(1 for issue in self.issues if issue.severity == 'error'),
                'warning': sum(1 for issue in self.issues if issue.severity == 'warning'),
                'info': sum(1 for issue in self.issues if issue.severity == 'info')
            }
        }
        
        return stats


def validate_dataset(dataset_path: str, output_path: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Convenience function to validate a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        output_path: Optional path to save validation report
        
    Returns:
        Tuple of (is_valid, list_of_issue_messages)
    """
    validator = DatasetValidator()
    report = validator.validate_dataset(dataset_path)
    
    if output_path:
        report.save(output_path)
    
    issue_messages = [issue.message for issue in report.issues]
    return report.is_valid, issue_messages


def quick_validate(dataset_path: str) -> bool:
    """Quick validation check for a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        True if dataset appears valid, False otherwise
    """
    dataset_dir = Path(dataset_path)
    
    # Basic checks
    if not dataset_dir.exists():
        return False
    
    # Check for some replay and beatmap files
    replay_files = list(dataset_dir.glob('**/*.osr'))
    beatmap_files = list(dataset_dir.glob('**/*.osu'))
    
    return len(replay_files) > 0 and len(beatmap_files) > 0