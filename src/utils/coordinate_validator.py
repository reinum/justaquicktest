"""Coordinate validation utilities for osu! replay generation."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CoordinateValidator:
    """Validates coordinate ranges and spatial properties."""
    
    def __init__(self, screen_width: int = 512, screen_height: int = 384):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
    def validate_normalized_coordinates(self, coords: torch.Tensor) -> Dict[str, bool]:
        """Validate that coordinates are in [0,1] range.
        
        Args:
            coords: Tensor of shape (..., 2) with x,y coordinates
            
        Returns:
            Dictionary with validation results
        """
        if coords.dim() < 2 or coords.size(-1) != 2:
            raise ValueError(f"Expected coordinates with shape (..., 2), got {coords.shape}")
            
        x_coords = coords[..., 0]
        y_coords = coords[..., 1]
        
        results = {
            'x_in_range': torch.all((x_coords >= 0) & (x_coords <= 1)).item(),
            'y_in_range': torch.all((y_coords >= 0) & (y_coords <= 1)).item(),
            'no_nan': torch.all(torch.isfinite(coords)).item(),
            'no_inf': torch.all(~torch.isinf(coords)).item()
        }
        
        results['all_valid'] = all(results.values())
        return results
        
    def validate_screen_coordinates(self, coords: torch.Tensor) -> Dict[str, bool]:
        """Validate that coordinates are in screen pixel range.
        
        Args:
            coords: Tensor of shape (..., 2) with x,y coordinates in pixels
            
        Returns:
            Dictionary with validation results
        """
        if coords.dim() < 2 or coords.size(-1) != 2:
            raise ValueError(f"Expected coordinates with shape (..., 2), got {coords.shape}")
            
        x_coords = coords[..., 0]
        y_coords = coords[..., 1]
        
        results = {
            'x_in_range': torch.all((x_coords >= 0) & (x_coords <= self.screen_width)).item(),
            'y_in_range': torch.all((y_coords >= 0) & (y_coords <= self.screen_height)).item(),
            'no_nan': torch.all(torch.isfinite(coords)).item(),
            'no_inf': torch.all(~torch.isinf(coords)).item()
        }
        
        results['all_valid'] = all(results.values())
        return results
        
    def compute_spatial_diversity_metrics(self, coords: torch.Tensor) -> Dict[str, float]:
        """Compute spatial diversity metrics for coordinate sequences.
        
        Args:
            coords: Tensor of shape (seq_len, 2) or (batch, seq_len, 2)
            
        Returns:
            Dictionary with diversity metrics
        """
        if coords.dim() == 3:
            # Batch dimension present, compute metrics per batch and average
            batch_metrics = []
            for i in range(coords.size(0)):
                batch_metrics.append(self.compute_spatial_diversity_metrics(coords[i]))
            
            # Average metrics across batch
            avg_metrics = {}
            for key in batch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in batch_metrics])
            return avg_metrics
            
        if coords.dim() != 2 or coords.size(-1) != 2:
            raise ValueError(f"Expected coordinates with shape (seq_len, 2), got {coords.shape}")
            
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # Variance metrics
        x_var = torch.var(x_coords).item()
        y_var = torch.var(y_coords).item()
        total_var = x_var + y_var
        
        # Range metrics
        x_range = (torch.max(x_coords) - torch.min(x_coords)).item()
        y_range = (torch.max(y_coords) - torch.min(y_coords)).item()
        
        # Boundary exploration (distance to edges)
        if coords.max() <= 1.0:  # Normalized coordinates
            # Distance to boundaries in normalized space
            boundary_dists = torch.stack([
                x_coords,  # distance to left
                1 - x_coords,  # distance to right
                y_coords,  # distance to top
                1 - y_coords   # distance to bottom
            ], dim=0)
            min_dist_to_boundary = torch.min(boundary_dists).item()
        else:  # Screen coordinates
            boundary_dists = torch.stack([
                x_coords,  # distance to left
                self.screen_width - x_coords,  # distance to right
                y_coords,  # distance to top
                self.screen_height - y_coords   # distance to bottom
            ], dim=0)
            min_dist_to_boundary = torch.min(boundary_dists).item()
            
        # Movement metrics
        if coords.size(0) > 1:
            movements = coords[1:] - coords[:-1]
            movement_distances = torch.norm(movements, dim=1)
            avg_movement = movement_distances.mean().item()
            max_movement = movement_distances.max().item()
        else:
            avg_movement = 0.0
            max_movement = 0.0
            
        return {
            'x_variance': x_var,
            'y_variance': y_var,
            'total_variance': total_var,
            'x_range': x_range,
            'y_range': y_range,
            'min_boundary_distance': min_dist_to_boundary,
            'avg_movement_distance': avg_movement,
            'max_movement_distance': max_movement
        }
        
    def log_validation_results(self, coords: torch.Tensor, coord_type: str = "coordinates"):
        """Log comprehensive validation results.
        
        Args:
            coords: Coordinate tensor to validate
            coord_type: Description of coordinate type for logging
        """
        try:
            # Determine if normalized or screen coordinates
            if coords.max() <= 1.0:
                validation = self.validate_normalized_coordinates(coords)
                coord_space = "normalized"
            else:
                validation = self.validate_screen_coordinates(coords)
                coord_space = "screen"
                
            diversity = self.compute_spatial_diversity_metrics(coords)
            
            logger.info(f"Validation results for {coord_type} ({coord_space} space):")
            logger.info(f"  Valid: {validation['all_valid']}")
            logger.info(f"  X in range: {validation['x_in_range']}")
            logger.info(f"  Y in range: {validation['y_in_range']}")
            logger.info(f"  No NaN/Inf: {validation['no_nan'] and validation['no_inf']}")
            
            logger.info(f"Spatial diversity metrics:")
            logger.info(f"  Total variance: {diversity['total_variance']:.6f}")
            logger.info(f"  X range: {diversity['x_range']:.6f}")
            logger.info(f"  Y range: {diversity['y_range']:.6f}")
            logger.info(f"  Min boundary distance: {diversity['min_boundary_distance']:.6f}")
            logger.info(f"  Avg movement: {diversity['avg_movement_distance']:.6f}")
            
        except Exception as e:
            logger.error(f"Error validating {coord_type}: {e}")
            
    def check_coordinate_consistency(self, 
                                   training_coords: torch.Tensor,
                                   generated_coords: torch.Tensor) -> Dict[str, float]:
        """Check consistency between training and generated coordinate distributions.
        
        Args:
            training_coords: Training coordinate samples
            generated_coords: Generated coordinate samples
            
        Returns:
            Dictionary with consistency metrics
        """
        train_metrics = self.compute_spatial_diversity_metrics(training_coords)
        gen_metrics = self.compute_spatial_diversity_metrics(generated_coords)
        
        consistency = {}
        for key in train_metrics:
            if train_metrics[key] != 0:
                consistency[f"{key}_ratio"] = gen_metrics[key] / train_metrics[key]
            else:
                consistency[f"{key}_ratio"] = float('inf') if gen_metrics[key] != 0 else 1.0
                
        return consistency