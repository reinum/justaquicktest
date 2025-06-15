"""Core Transformer model for osu! replay generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .embeddings import PositionalEncoding, TimingEncoding, AccuracyConditioning, BeatmapEncoder, CursorEncoder, SliderEncoder
from .attention import (
    SelfAttention, CausalSelfAttention, CrossAttention,
    RelativePositionalAttention
)
from ..config.model_config import ModelConfig


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Choose attention type
        if config.use_relative_attention:
            self.self_attention = RelativePositionalAttention(
                config.d_model, config.n_heads, 
                config.max_relative_position, config.dropout
            )
        elif config.causal:
            self.self_attention = CausalSelfAttention(
                config.d_model, config.n_heads, config.max_seq_length,
                config.dropout, config.use_flash_attention
            )
        else:
            self.self_attention = SelfAttention(
                config.d_model, config.n_heads, config.dropout,
                config.use_flash_attention
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            mask: Attention mask
            key_padding_mask: Key padding mask
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        if isinstance(self.self_attention, RelativePositionalAttention):
            attn_output = self.self_attention(x, mask)
        elif hasattr(self.self_attention, 'causal_mask'):  # CausalSelfAttention
            attn_output = self.self_attention(x, key_padding_mask)
        else:
            attn_output = self.self_attention(x, mask, key_padding_mask)
        
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class OsuTransformer(nn.Module):
    """Main transformer model for osu! replay generation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.cursor_encoding = CursorEncoder(config.d_model)
        self.beatmap_encoding = BeatmapEncoder(config.d_model, beatmap_feature_dim=8)
        self.slider_encoding = SliderEncoder(config.d_model, slider_feature_dim=getattr(config, 'slider_feature_dim', 13))
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        self.timing_encoding = TimingEncoding(config.d_model)
        self.accuracy_conditioning = AccuracyConditioning(config.d_model)
        
        # Input projection
        self.input_projection = nn.Linear(config.d_model * 3, config.d_model)  # Cursor + beatmap + slider
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Cross-attention for beatmap context
        if config.use_cross_attention:
            self.cross_attention_layers = nn.ModuleList([
                CrossAttention(config.d_model, config.n_heads, config.dropout)
                for _ in range(config.n_cross_attention_layers)
            ])
        
        # Output heads
        self.cursor_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 2),  # x, y coordinates
            nn.Sigmoid()  # Output [0,1] to match training data normalization
        )
        
        self.key_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 4, 4)  # M1, M2, K1, K2
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, cursor_data: torch.Tensor, beatmap_data: torch.Tensor, slider_data: torch.Tensor,
                timing_data: torch.Tensor, key_data: torch.Tensor, accuracy_target: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the transformer.
        
        Args:
            cursor_data: Cursor positions of shape (seq_len, batch_size, 2)
            beatmap_data: Beatmap features of shape (seq_len, batch_size, beatmap_features)
            slider_data: Slider features of shape (seq_len, batch_size, slider_features)
            timing_data: Timing information of shape (seq_len, batch_size, 1)
            key_data: Key states of shape (seq_len, batch_size, 4)
            accuracy_target: Target accuracy of shape (batch_size, 1)
            mask: Attention mask
            key_padding_mask: Key padding mask
            
        Returns:
            Dictionary containing cursor predictions and key predictions
        """
        seq_len, batch_size = cursor_data.shape[:2]
        # The actual batch_size should match accuracy_target.shape[0]
        actual_batch_size = accuracy_target.shape[0]
        actual_seq_len = cursor_data.shape[0]
        print(f"DEBUG: cursor_data.shape: {cursor_data.shape}")
        print(f"DEBUG: accuracy_target.shape: {accuracy_target.shape}")
        print(f"DEBUG: Extracted seq_len={seq_len}, batch_size={batch_size}")
        print(f"DEBUG: Corrected seq_len={actual_seq_len}, batch_size={actual_batch_size}")
        # Use the corrected values
        seq_len, batch_size = actual_seq_len, actual_batch_size
        
        # Transpose input tensors to correct shape: [seq_len, batch_size, feature_dim]
        cursor_data = cursor_data.transpose(0, 1)  # [batch_size, seq_len, 2] -> [seq_len, batch_size, 2]
        key_data = key_data.transpose(0, 1)  # [batch_size, seq_len, 4] -> [seq_len, batch_size, 4]
        beatmap_data = beatmap_data.transpose(0, 1)  # [batch_size, seq_len, 6] -> [seq_len, batch_size, 6]
        slider_data = slider_data.transpose(0, 1)  # [batch_size, seq_len, 13] -> [seq_len, batch_size, 13]
        timing_data = timing_data.transpose(0, 1)  # [batch_size, seq_len, 1] -> [seq_len, batch_size, 1]
        print(f"DEBUG: cursor_data.shape after transpose: {cursor_data.shape}")
        print(f"DEBUG: beatmap_data.shape after transpose: {beatmap_data.shape}")
        print(f"DEBUG: slider_data.shape after transpose: {slider_data.shape}")
        print(f"DEBUG: timing_data.shape after transpose: {timing_data.shape}")
        
        # Update seq_len and batch_size after transposition
        seq_len, batch_size = cursor_data.shape[0], cursor_data.shape[1]
        print(f"DEBUG: Final seq_len={seq_len}, batch_size={batch_size}")
        
        # Encode inputs
        cursor_emb = self.cursor_encoding(cursor_data, key_data)  # (seq_len, batch_size, d_model)
        print(f"DEBUG: cursor_emb shape after encoding: {cursor_emb.shape}")
        beatmap_emb = self.beatmap_encoding(beatmap_data)  # (seq_len, batch_size, d_model)
        print(f"DEBUG: beatmap_emb shape after encoding: {beatmap_emb.shape}")
        slider_emb = self.slider_encoding(slider_data)  # (seq_len, batch_size, d_model)
        print(f"DEBUG: slider_emb shape after encoding: {slider_emb.shape}")
        timing_emb = self.timing_encoding(timing_data)  # (seq_len, batch_size, d_model)
        print(f"DEBUG: timing_emb shape after encoding: {timing_emb.shape}")
        
        # Add positional encoding
        cursor_emb = self.positional_encoding(cursor_emb)
        print(f"DEBUG: cursor_emb shape after positional encoding: {cursor_emb.shape}")
        beatmap_emb = self.positional_encoding(beatmap_emb)
        print(f"DEBUG: beatmap_emb shape after positional encoding: {beatmap_emb.shape}")
        slider_emb = self.positional_encoding(slider_emb)
        print(f"DEBUG: slider_emb shape after positional encoding: {slider_emb.shape}")
        
        # Add timing information
        cursor_emb = cursor_emb + timing_emb
        print(f"DEBUG: cursor_emb shape after adding timing: {cursor_emb.shape}")
        beatmap_emb = beatmap_emb + timing_emb
        print(f"DEBUG: beatmap_emb shape after adding timing: {beatmap_emb.shape}")
        slider_emb = slider_emb + timing_emb
        print(f"DEBUG: slider_emb shape after adding timing: {slider_emb.shape}")
        
        # Combine cursor, beatmap, and slider embeddings
        combined_emb = torch.cat([cursor_emb, beatmap_emb, slider_emb], dim=-1)  # (seq_len, batch_size, 3*d_model)
        print(f"DEBUG: combined_emb shape: {combined_emb.shape}")
        x = self.input_projection(combined_emb)  # (seq_len, batch_size, d_model)
        print(f"DEBUG: x after projection shape: {x.shape}")
        
        # Add accuracy conditioning
        accuracy_emb = self.accuracy_conditioning(accuracy_target)  # (batch_size, d_model)
        print(f"DEBUG: accuracy_emb shape: {accuracy_emb.shape}")
        print(f"DEBUG: seq_len: {seq_len}, batch_size: {batch_size}")
        # Reshape to match x: (seq_len, batch_size, d_model)
        accuracy_emb = accuracy_emb.unsqueeze(0).expand(seq_len, -1, -1)  # (seq_len, batch_size, d_model)
        print(f"DEBUG: accuracy_emb expanded shape: {accuracy_emb.shape}")
        print(f"DEBUG: x shape: {x.shape}")
        x = x + accuracy_emb
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        
        # Apply cross-attention if enabled
        if self.config.use_cross_attention:
            for cross_layer in self.cross_attention_layers:
                x = cross_layer(x, beatmap_emb, mask)
        
        # Generate predictions
        cursor_pred = self.cursor_head(x)  # (seq_len, batch_size, 2)
        key_pred = self.key_head(x)  # (seq_len, batch_size, 4)
        
        return {
            'cursor_pred': cursor_pred,
            'key_pred': key_pred,
            'hidden_states': x
        }
    
    def generate(self, beatmap_data: torch.Tensor, timing_data: torch.Tensor,
                 accuracy_target: torch.Tensor, max_length: int = 1000,
                 temperature: float = 1.0, top_k: int = 50,
                 initial_cursor: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate replay sequence autoregressively.
        
        Args:
            beatmap_data: Beatmap features of shape (seq_len, batch_size, beatmap_features)
            timing_data: Timing information of shape (seq_len, batch_size, 1)
            accuracy_target: Target accuracy of shape (batch_size, 1)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            initial_cursor: Initial cursor position of shape (1, batch_size, 2)
            
        Returns:
            Generated cursor and key sequences
        """
        self.eval()
        batch_size = beatmap_data.shape[1]
        device = beatmap_data.device
        
        # Initialize sequences
        if initial_cursor is None:
            cursor_seq = torch.zeros(1, batch_size, 2, device=device)
        else:
            cursor_seq = initial_cursor
        
        key_seq = torch.zeros(1, batch_size, 4, device=device)
        
        with torch.no_grad():
            for step in range(max_length):
                # Get current sequence length
                current_len = cursor_seq.shape[0]
                
                # Prepare inputs for current step
                current_beatmap = beatmap_data[:current_len]
                current_timing = timing_data[:current_len]
                
                # Forward pass
                outputs = self.forward(
                    cursor_seq, current_beatmap, current_timing, accuracy_target
                )
                
                # Get predictions for next step
                next_cursor_logits = outputs['cursor_pred'][-1]  # (batch_size, 2)
                next_key_logits = outputs['key_pred'][-1]  # (batch_size, 4)
                
                # Sample next cursor position
                if temperature > 0:
                    next_cursor = self._sample_cursor(next_cursor_logits, temperature)
                else:
                    next_cursor = next_cursor_logits
                
                # Sample next key states
                next_keys = self._sample_keys(next_key_logits, temperature, top_k)
                
                # Append to sequences
                cursor_seq = torch.cat([cursor_seq, next_cursor.unsqueeze(0)], dim=0)
                key_seq = torch.cat([key_seq, next_keys.unsqueeze(0)], dim=0)
                
                # Check if we've reached the end of the beatmap
                if current_len >= beatmap_data.shape[0] - 1:
                    break
        
        return {
            'cursor_sequence': cursor_seq,
            'key_sequence': key_seq
        }
    
    def _sample_cursor(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample cursor position with optional noise."""
        if temperature > 0:
            noise = torch.randn_like(logits) * temperature
            return logits + noise
        return logits
    
    def _sample_keys(self, logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        """Sample key states using top-k sampling."""
        if temperature == 0:
            return torch.sigmoid(logits)
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, _ = torch.topk(logits, top_k, dim=-1)
            min_top_k = top_k_logits[..., -1:]
            logits = torch.where(logits < min_top_k, torch.full_like(logits, -float('inf')), logits)
        
        # Sample from categorical distribution
        probs = torch.sigmoid(logits)
        keys = torch.bernoulli(probs)
        
        return keys
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
        buffer_memory = sum(b.numel() * b.element_size() for b in self.buffers()) / 1024**2
        
        return {
            'parameters_mb': param_memory,
            'buffers_mb': buffer_memory,
            'total_mb': param_memory + buffer_memory
        }


class OsuTransformerLM(OsuTransformer):
    """Language model variant for autoregressive generation."""
    
    def __init__(self, config: ModelConfig):
        # Force causal attention for language modeling
        config.causal = True
        super().__init__(config)
        
        # Additional language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for language modeling.
        
        Args:
            input_ids: Token IDs of shape (seq_len, batch_size)
            attention_mask: Attention mask
            labels: Target labels for loss computation
            
        Returns:
            Dictionary containing logits and optional loss
        """
        # Token embeddings
        x = self.token_embedding(input_ids)  # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Language modeling head
        logits = self.lm_head(x)  # (seq_len, batch_size, vocab_size)
        
        outputs = {'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[:-1].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            outputs['loss'] = loss
        
        return outputs


def create_model(config: ModelConfig) -> OsuTransformer:
    """Factory function to create model based on configuration."""
    if config.model_type == 'transformer':
        return OsuTransformer(config)
    elif config.model_type == 'transformer_lm':
        return OsuTransformerLM(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")