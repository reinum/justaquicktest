"""Loss functions for osu! replay training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class CursorLoss(nn.Module):
    """Loss function for cursor position prediction."""
    
    def __init__(self, loss_type: str = 'mse', smoothness_weight: float = 0.1):
        super().__init__()
        self.loss_type = loss_type
        self.smoothness_weight = smoothness_weight
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none', delta=0.1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute cursor position loss.
        
        Args:
            pred: Predicted cursor positions (seq_len, batch_size, 2)
            target: Target cursor positions (seq_len, batch_size, 2)
            mask: Optional mask for valid positions
            
        Returns:
            Scalar loss value
        """
        # Base position loss
        position_loss = self.base_loss(pred, target)  # (seq_len, batch_size, 2)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(position_loss)
            position_loss = position_loss * mask
        
        # Average over coordinates and sequence
        position_loss = position_loss.mean()
        
        # Smoothness regularization
        smoothness_loss = self._compute_smoothness_loss(pred, mask)
        
        total_loss = position_loss + self.smoothness_weight * smoothness_loss
        
        return total_loss
    
    def _compute_smoothness_loss(self, pred: torch.Tensor, 
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute smoothness regularization loss."""
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute velocity (first derivative)
        velocity = pred[1:] - pred[:-1]  # (seq_len-1, batch_size, 2)
        
        # Compute acceleration (second derivative)
        if pred.shape[0] >= 3:
            acceleration = velocity[1:] - velocity[:-1]  # (seq_len-2, batch_size, 2)
            smoothness = torch.mean(acceleration ** 2)
        else:
            smoothness = torch.mean(velocity ** 2)
        
        return smoothness


class KeyLoss(nn.Module):
    """Loss function for key press prediction."""
    
    def __init__(self, loss_type: str = 'bce', class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss_type = loss_type
        self.class_weights = class_weights
        
        if loss_type == 'bce':
            self.base_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=class_weights)
        elif loss_type == 'focal':
            self.alpha = 0.25
            self.gamma = 2.0
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute key press loss.
        
        Args:
            pred: Predicted key states (seq_len, batch_size, 4)
            target: Target key states (seq_len, batch_size, 4)
            mask: Optional mask for valid positions
            
        Returns:
            Scalar loss value
        """
        if self.loss_type == 'bce':
            loss = self.base_loss(pred, target)  # (seq_len, batch_size, 4)
        elif self.loss_type == 'focal':
            loss = self._focal_loss(pred, target)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask
        
        return loss.mean()
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for imbalanced key press data."""
        # Convert logits to probabilities
        prob = torch.sigmoid(pred)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = prob * target + (1 - prob) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        return focal_loss


class SliderLoss(nn.Module):
    """Loss function for slider-specific predictions."""
    
    def __init__(self, path_weight: float = 1.0, timing_weight: float = 0.5, 
                 smoothness_weight: float = 0.3):
        super().__init__()
        self.path_weight = path_weight
        self.timing_weight = timing_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, pred_cursor: torch.Tensor, target_cursor: torch.Tensor,
                slider_features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute slider-specific loss.
        
        Args:
            pred_cursor: Predicted cursor positions (seq_len, batch_size, 2)
            target_cursor: Target cursor positions (seq_len, batch_size, 2)
            slider_features: Slider feature data (seq_len, batch_size, 13)
            mask: Optional mask for valid positions
            
        Returns:
            Scalar slider loss value
        """
        # Extract slider activity (assuming it's encoded in the features)
        slider_active = slider_features[:, :, 0] > 0  # position_on_path > 0 indicates active slider
        
        if not slider_active.any():
            return torch.tensor(0.0, device=pred_cursor.device)
        
        # Path adherence loss - cursor should follow the slider path
        path_loss = self._compute_path_adherence_loss(
            pred_cursor, target_cursor, slider_features, slider_active, mask
        )
        
        # Timing consistency loss - cursor should move at appropriate speed
        timing_loss = self._compute_timing_consistency_loss(
            pred_cursor, slider_features, slider_active, mask
        )
        
        # Smoothness loss - cursor movement should be smooth during sliders
        smoothness_loss = self._compute_slider_smoothness_loss(
            pred_cursor, slider_active, mask
        )
        
        total_loss = (
            self.path_weight * path_loss +
            self.timing_weight * timing_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        return total_loss
    
    def _compute_path_adherence_loss(self, pred_cursor: torch.Tensor, 
                                   target_cursor: torch.Tensor,
                                   slider_features: torch.Tensor,
                                   slider_active: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss for how well cursor follows slider path."""
        # Distance to target position (feature index 1)
        distance_to_target = slider_features[:, :, 1]  # (seq_len, batch_size)
        
        # Weight the position loss by slider activity
        position_diff = torch.norm(pred_cursor - target_cursor, dim=-1)  # (seq_len, batch_size)
        
        # Apply slider activity mask
        slider_mask = slider_active.float()
        if mask is not None:
            slider_mask = slider_mask * mask
        
        # Weighted position loss
        weighted_loss = position_diff * slider_mask
        
        # Additional penalty for deviating from slider path
        path_penalty = distance_to_target * slider_mask
        
        total_path_loss = weighted_loss + 0.5 * path_penalty
        
        return total_path_loss.sum() / (slider_mask.sum() + 1e-8)
    
    def _compute_timing_consistency_loss(self, pred_cursor: torch.Tensor,
                                       slider_features: torch.Tensor,
                                       slider_active: torch.Tensor,
                                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss for timing consistency during sliders."""
        if pred_cursor.shape[0] < 2:
            return torch.tensor(0.0, device=pred_cursor.device)
        
        # Expected velocity based on slider progress (feature index 4)
        slider_progress = slider_features[:, :, 4]  # (seq_len, batch_size)
        progress_velocity = torch.diff(slider_progress, dim=0)  # (seq_len-1, batch_size)
        
        # Actual cursor velocity
        cursor_velocity = torch.norm(
            torch.diff(pred_cursor, dim=0), dim=-1
        )  # (seq_len-1, batch_size)
        
        # Slider activity for velocity frames
        velocity_slider_active = slider_active[1:].float()
        if mask is not None:
            velocity_slider_active = velocity_slider_active * mask[1:]
        
        # Timing consistency loss
        timing_diff = torch.abs(cursor_velocity - torch.abs(progress_velocity) * 100)  # Scale factor
        timing_loss = timing_diff * velocity_slider_active
        
        return timing_loss.sum() / (velocity_slider_active.sum() + 1e-8)
    
    def _compute_slider_smoothness_loss(self, pred_cursor: torch.Tensor,
                                      slider_active: torch.Tensor,
                                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute smoothness loss specifically for slider segments."""
        if pred_cursor.shape[0] < 3:
            return torch.tensor(0.0, device=pred_cursor.device)
        
        # Compute acceleration during slider segments
        velocity = torch.diff(pred_cursor, dim=0)  # (seq_len-1, batch_size, 2)
        acceleration = torch.diff(velocity, dim=0)  # (seq_len-2, batch_size, 2)
        acceleration_magnitude = torch.norm(acceleration, dim=-1)  # (seq_len-2, batch_size)
        
        # Slider activity for acceleration frames
        accel_slider_active = slider_active[2:].float()
        if mask is not None:
            accel_slider_active = accel_slider_active * mask[2:]
        
        # Smoothness loss - penalize high acceleration during sliders
        smoothness_loss = acceleration_magnitude * accel_slider_active
        
        return smoothness_loss.sum() / (accel_slider_active.sum() + 1e-8)


class ReplayLoss(nn.Module):
    """Combined loss function for replay generation."""
    
    def __init__(self, cursor_weight: float = 1.0, key_weight: float = 1.0,
                 slider_weight: float = 0.5, cursor_loss_type: str = 'mse', 
                 key_loss_type: str = 'bce', smoothness_weight: float = 0.1, 
                 temporal_weight: float = 0.1):
        super().__init__()
        
        self.cursor_weight = cursor_weight
        self.key_weight = key_weight
        self.slider_weight = slider_weight
        self.temporal_weight = temporal_weight
        
        # Individual loss components
        self.cursor_loss = CursorLoss(cursor_loss_type, smoothness_weight)
        self.key_loss = KeyLoss(key_loss_type)
        self.slider_loss = SliderLoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute combined replay loss.
        
        Args:
            predictions: Dictionary with 'cursor_pred' and 'key_pred'
            targets: Dictionary with 'cursor_data' and 'key_data'
            mask: Optional mask for valid positions
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Cursor position loss
        # Model outputs: (seq_len, batch_size, 2)
        # Targets: (batch_size, seq_len, 2) -> transpose to (seq_len, batch_size, 2)
        cursor_target = targets['cursor_data'].transpose(0, 1)  # (seq_len, batch_size, 2)
        
        # Transpose mask if provided to match (seq_len, batch_size) format
        transposed_mask = mask.transpose(0, 1) if mask is not None else None
        
        cursor_loss = self.cursor_loss(
            predictions['cursor_pred'], 
            cursor_target, 
            transposed_mask
        )
        losses['cursor_loss'] = cursor_loss
        
        # Key press loss
        # Model outputs: (seq_len, batch_size, 4)
        # Targets: (batch_size, seq_len, 4) -> transpose to (seq_len, batch_size, 4)
        key_target = targets['key_data'].transpose(0, 1)  # (seq_len, batch_size, 4)
        key_loss = self.key_loss(
            predictions['key_pred'], 
            key_target, 
            transposed_mask
        )
        losses['key_loss'] = key_loss
        
        # Temporal consistency loss
        temporal_loss = self._compute_temporal_loss(
            predictions['cursor_pred'], 
            cursor_target,  # Use transposed target
            transposed_mask
        )
        losses['temporal_loss'] = temporal_loss
        
        # Slider-specific loss (if slider data is available)
        slider_loss = torch.tensor(0.0, device=cursor_loss.device)
        if 'slider_data' in targets:
            slider_target = targets['slider_data'].transpose(0, 1)  # (seq_len, batch_size, 13)
            slider_loss = self.slider_loss(
                predictions['cursor_pred'],
                cursor_target,
                slider_target,
                transposed_mask
            )
        losses['slider_loss'] = slider_loss
        
        # Combined loss
        total_loss = (
            self.cursor_weight * cursor_loss +
            self.key_weight * key_loss +
            self.temporal_weight * temporal_loss +
            self.slider_weight * slider_loss
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_temporal_loss(self, pred: torch.Tensor, target: torch.Tensor,
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute temporal consistency loss."""
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute velocity differences
        pred_velocity = pred[1:] - pred[:-1]
        target_velocity = target[1:] - target[:-1]
        
        velocity_loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            velocity_mask = mask[1:].unsqueeze(-1).expand_as(velocity_loss)
            velocity_loss = velocity_loss * velocity_mask
        
        return velocity_loss.mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning better representations."""
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            embeddings: Feature embeddings (batch_size, embedding_dim)
            labels: Binary labels indicating positive/negative pairs
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Positive pairs (same class)
        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos_distances = distances * pos_mask.float()
        
        # Negative pairs (different class)
        neg_mask = ~pos_mask
        neg_distances = distances * neg_mask.float()
        
        # Contrastive loss
        pos_loss = pos_distances.pow(2)
        neg_loss = F.relu(self.margin - neg_distances).pow(2)
        
        # Average over valid pairs
        pos_loss = pos_loss.sum() / (pos_mask.sum() + 1e-8)
        neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-8)
        
        return pos_loss + neg_loss


class PerceptualLoss(nn.Module):
    """Perceptual loss for cursor trajectory similarity."""
    
    def __init__(self, feature_layers: list = [1, 2, 3]):
        super().__init__()
        self.feature_layers = feature_layers
        
        # Simple CNN for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.
        
        Args:
            pred: Predicted cursor trajectory (seq_len, batch_size, 2)
            target: Target cursor trajectory (seq_len, batch_size, 2)
            
        Returns:
            Perceptual loss value
        """
        # Reshape for conv1d: (batch_size, channels, seq_len)
        pred = pred.transpose(0, 1).transpose(1, 2)
        target = target.transpose(0, 1).transpose(1, 2)
        
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # Compute feature loss
        feature_loss = F.mse_loss(pred_features, target_features)
        
        return feature_loss


class AdversarialLoss(nn.Module):
    """Adversarial loss for realistic replay generation."""
    
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator
    
    def forward(self, generated_replays: torch.Tensor, 
                real_replays: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute adversarial loss.
        
        Args:
            generated_replays: Generated replay sequences
            real_replays: Real replay sequences
            
        Returns:
            Dictionary with generator and discriminator losses
        """
        # Discriminator predictions
        real_pred = self.discriminator(real_replays)
        fake_pred = self.discriminator(generated_replays.detach())
        
        # Discriminator loss
        real_loss = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )
        discriminator_loss = (real_loss + fake_loss) / 2
        
        # Generator loss
        fake_pred_for_gen = self.discriminator(generated_replays)
        generator_loss = F.binary_cross_entropy_with_logits(
            fake_pred_for_gen, torch.ones_like(fake_pred_for_gen)
        )
        
        return {
            'discriminator_loss': discriminator_loss,
            'generator_loss': generator_loss
        }