"""Embedding and encoding modules for the osu! replay transformer."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(0)
        # Use positional encoding without extra unsqueeze
        pe = self.pe[:seq_len, :]  # Shape: (seq_len, d_model)
        x = x + pe  # Broadcasting: (seq_len, batch_size, d_model) + (seq_len, d_model)
        return self.dropout(x)


class TimingEncoding(nn.Module):
    """Encoding for timing information in osu! replays."""
    
    def __init__(self, d_model: int, max_time_ms: int = 600000):  # 10 minutes max
        super().__init__()
        self.d_model = d_model
        self.max_time_ms = max_time_ms
        
        # Linear projection for timing
        self.time_projection = nn.Linear(1, d_model)
        
        # Learnable timing embeddings for different time scales
        self.beat_embedding = nn.Embedding(1000, d_model // 4)  # Beat positions
        self.measure_embedding = nn.Embedding(100, d_model // 4)  # Measure positions
        
    def forward(self, timestamps: torch.Tensor, bpm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode timing information.
        
        Args:
            timestamps: Tensor of shape (seq_len, batch_size, 1) with timestamps in ms
            bpm: Optional BPM information for beat-aware encoding
            
        Returns:
            Timing encodings of shape (seq_len, batch_size, d_model)
        """
        # Normalize timestamps
        normalized_time = timestamps / self.max_time_ms
        time_encoding = self.time_projection(normalized_time)
        
        if bpm is not None:
            # Calculate beat and measure positions
            beat_duration_ms = 60000 / bpm  # ms per beat
            beat_positions = (timestamps / beat_duration_ms) % 1000
            measure_positions = (timestamps / (beat_duration_ms * 4)) % 100
            
            beat_emb = self.beat_embedding(beat_positions.long())
            measure_emb = self.measure_embedding(measure_positions.long())
            
            # Combine encodings
            time_encoding = torch.cat([time_encoding[:, :, :self.d_model//2], 
                                     beat_emb, measure_emb], dim=-1)
        
        return time_encoding


class AccuracyConditioning(nn.Module):
    """Conditioning module for target accuracy parameters."""
    
    def __init__(self, d_model: int, accuracy_dim: int = 1):
        super().__init__()
        self.d_model = d_model
        
        # Project accuracy parameters to model dimension
        self.accuracy_projection = nn.Sequential(
            nn.Linear(accuracy_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Learnable embeddings for different accuracy ranges
        self.accuracy_embeddings = nn.Embedding(21, d_model)  # 0-100% in 5% increments
        
    def forward(self, accuracy_params: torch.Tensor) -> torch.Tensor:
        """Generate conditioning vectors from accuracy parameters.
        
        Args:
            accuracy_params: Tensor of shape (batch_size, 1) with target accuracy
            
        Returns:
            Conditioning vectors of shape (batch_size, d_model)
        """
        # Project raw accuracy parameters
        accuracy_encoding = self.accuracy_projection(accuracy_params)
        
        # Add categorical accuracy embedding based on overall accuracy
        overall_accuracy = accuracy_params[:, 0]  # Single accuracy value
        accuracy_category = torch.clamp((overall_accuracy * 20).long(), 0, 20)
        category_embedding = self.accuracy_embeddings(accuracy_category)
        
        return accuracy_encoding + category_embedding


class BeatmapEncoder(nn.Module):
    """Encoder for beatmap features and hit objects."""
    
    def __init__(self, d_model: int, beatmap_feature_dim: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # Hit object type embeddings
        self.hit_type_embedding = nn.Embedding(4, d_model // 4)  # Circle, slider, spinner, etc.
        
        # Position encoding for hit objects
        self.position_projection = nn.Linear(2, d_model // 4)  # x, y coordinates
        
        # Timing and difficulty features
        self.feature_projection = nn.Linear(beatmap_feature_dim - 4, d_model // 2)  # Exclude time, x, y, type
        
        # Final projection
        self.output_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, beatmap_features: torch.Tensor) -> torch.Tensor:
        """Encode beatmap hit objects.
        
        Args:
            beatmap_features: Tensor of shape (seq_len, batch_size, feature_dim)
                             Features: [time, x, y, type, ...other features]
            
        Returns:
            Encoded beatmap features of shape (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, _ = beatmap_features.shape
        
        # Extract components
        positions = beatmap_features[:, :, 1:3]  # x, y
        hit_types = beatmap_features[:, :, 3].long()  # type
        other_features = beatmap_features[:, :, 4:]  # remaining features
        
        # Encode each component
        pos_encoding = self.position_projection(positions)
        type_encoding = self.hit_type_embedding(hit_types)
        feature_encoding = self.feature_projection(other_features)
        
        # Combine encodings
        combined = torch.cat([pos_encoding, type_encoding, feature_encoding], dim=-1)
        
        # Final projection and normalization
        output = self.output_projection(combined)
        return self.layer_norm(output)


class SliderEncoder(nn.Module):
    """Encoder for slider-specific features."""
    
    def __init__(self, d_model: int, slider_feature_dim: int = 13):
        super().__init__()
        self.d_model = d_model
        
        # Position features (position on slider path, distance to target)
        self.position_projection = nn.Linear(2, d_model // 4)
        
        # Velocity features (velocity magnitude, direction alignment)
        self.velocity_projection = nn.Linear(2, d_model // 4)
        
        # Temporal features (progress, time remaining)
        self.temporal_projection = nn.Linear(2, d_model // 4)
        
        # Geometric features (curvature, path length, etc.)
        self.geometric_projection = nn.Linear(4, d_model // 4)
        
        # Context features (slider type, repeat count, difficulty)
        self.context_projection = nn.Linear(3, d_model // 4)
        
        # Final projection to combine all features
        self.output_projection = nn.Linear(d_model + d_model // 4, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, slider_features: torch.Tensor) -> torch.Tensor:
        """Encode slider features.
        
        Args:
            slider_features: Tensor of shape (seq_len, batch_size, 13)
                           Features: [position_on_path, distance_to_target, 
                                    velocity_magnitude, direction_alignment,
                                    progress, time_remaining,
                                    curvature, path_length, segment_count, direction_changes,
                                    slider_type, repeat_count, difficulty_rating]
            
        Returns:
            Encoded slider features of shape (seq_len, batch_size, d_model)
        """
        # Extract feature groups
        position_features = slider_features[:, :, 0:2]  # position_on_path, distance_to_target
        velocity_features = slider_features[:, :, 2:4]  # velocity_magnitude, direction_alignment
        temporal_features = slider_features[:, :, 4:6]  # progress, time_remaining
        geometric_features = slider_features[:, :, 6:10]  # curvature, path_length, segment_count, direction_changes
        context_features = slider_features[:, :, 10:13]  # slider_type, repeat_count, difficulty_rating
        
        # Encode each feature group
        position_encoding = self.position_projection(position_features)
        velocity_encoding = self.velocity_projection(velocity_features)
        temporal_encoding = self.temporal_projection(temporal_features)
        geometric_encoding = self.geometric_projection(geometric_features)
        context_encoding = self.context_projection(context_features)
        
        # Combine all encodings
        combined = torch.cat([
            position_encoding, velocity_encoding, temporal_encoding,
            geometric_encoding, context_encoding
        ], dim=-1)
        
        # Final projection and normalization
        output = self.output_projection(combined)
        return self.layer_norm(output)


class CursorEncoder(nn.Module):
    """Encoder for cursor position and key state history."""
    
    def __init__(self, d_model: int, cursor_dim: int = 2, key_dim: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # Cursor position encoding
        self.cursor_projection = nn.Linear(cursor_dim, d_model // 2)
        
        # Key state encoding
        self.key_projection = nn.Linear(key_dim, d_model // 4)
        
        # Velocity and acceleration encoding
        self.velocity_projection = nn.Linear(cursor_dim, d_model // 4)
        
        # Final projection
        self.output_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, cursor_positions: torch.Tensor, key_states: torch.Tensor, 
                velocities: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode cursor history.
        
        Args:
            cursor_positions: Tensor of shape (seq_len, batch_size, 2)
            key_states: Tensor of shape (seq_len, batch_size, 4)
            velocities: Optional velocity tensor of shape (seq_len, batch_size, 2)
            
        Returns:
            Encoded cursor features of shape (seq_len, batch_size, d_model)
        """
        # Encode components
        cursor_encoding = self.cursor_projection(cursor_positions)
        key_encoding = self.key_projection(key_states)
        
        if velocities is not None:
            velocity_encoding = self.velocity_projection(velocities)
        else:
            # Calculate velocities from positions
            velocities = torch.diff(cursor_positions, dim=0, prepend=cursor_positions[:1])
            velocity_encoding = self.velocity_projection(velocities)
        
        # Combine encodings
        combined = torch.cat([cursor_encoding, key_encoding, velocity_encoding], dim=-1)
        
        # Final projection and normalization
        output = self.output_projection(combined)
        return self.layer_norm(output)