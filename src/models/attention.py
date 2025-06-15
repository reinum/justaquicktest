"""Attention mechanisms for the osu! replay transformer."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional flash attention optimization."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 use_flash_attention: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, 
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (seq_len, batch_size, d_model)
            key: Key tensor of shape (seq_len, batch_size, d_model)
            value: Value tensor of shape (seq_len, batch_size, d_model)
            mask: Attention mask of shape (seq_len, seq_len)
            key_padding_mask: Key padding mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        seq_len, batch_size, d_model = query.shape
        
        # Linear projections
        Q = self.w_q(query)  # (seq_len, batch_size, d_model)
        K = self.w_k(key)    # (seq_len, batch_size, d_model)
        V = self.w_v(value)  # (seq_len, batch_size, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(seq_len, batch_size, self.n_heads, self.d_k).transpose(1, 2)  # (seq_len, n_heads, batch_size, d_k)
        K = K.view(seq_len, batch_size, self.n_heads, self.d_k).transpose(1, 2)  # (seq_len, n_heads, batch_size, d_k)
        V = V.view(seq_len, batch_size, self.n_heads, self.d_k).transpose(1, 2)  # (seq_len, n_heads, batch_size, d_k)
        
        if self.use_flash_attention:
            # Use flash attention if available
            output, attention_weights = self._flash_attention(Q, K, V, mask, key_padding_mask)
        else:
            # Standard attention computation
            output, attention_weights = self._standard_attention(Q, K, V, mask, key_padding_mask)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(seq_len, batch_size, d_model)
        
        # Final linear projection
        output = self.w_o(output)
        
        return output, attention_weights
    
    def _standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention."""
        # Q, K, V have shape [seq_len, n_heads, batch_size, d_k]
        # We need to transpose to [batch_size, n_heads, seq_len, d_k] for proper attention computation
        Q = Q.transpose(0, 2)  # [batch_size, n_heads, seq_len, d_k]
        K = K.transpose(0, 2)  # [batch_size, n_heads, seq_len, d_k]
        V = V.transpose(0, 2)  # [batch_size, n_heads, seq_len, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, n_heads, seq_len, seq_len]
        
        # Apply masks
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        if key_padding_mask is not None:
            # key_padding_mask has shape [batch_size, seq_len]
            # scores has shape [batch_size, n_heads, seq_len, seq_len]
            # Expand key_padding_mask to [batch_size, 1, 1, seq_len] for broadcasting
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(key_padding_mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch_size, n_heads, seq_len, d_k]
        
        # Transpose back to [seq_len, n_heads, batch_size, d_k] to match expected output format
        output = output.transpose(0, 2)  # [seq_len, n_heads, batch_size, d_k]
        
        return output, attention_weights
    
    def _flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flash attention implementation (if available)."""
        # Reshape for flash attention: (batch_size, seq_len, n_heads, d_k)
        Q = Q.permute(2, 0, 1, 3)
        K = K.permute(2, 0, 1, 3)
        V = V.permute(2, 0, 1, 3)
        
        # Apply flash attention
        output = flash_attn_func(Q, K, V, dropout_p=self.dropout.p if self.training else 0.0)
        
        # Reshape back: (seq_len, n_heads, batch_size, d_k)
        output = output.permute(1, 2, 0, 3)
        
        # For compatibility, return dummy attention weights
        attention_weights = torch.zeros(self.n_heads, Q.shape[0], Q.shape[1], Q.shape[1], 
                                      device=Q.device, dtype=Q.dtype)
        
        return output, attention_weights


class CrossAttention(nn.Module):
    """Cross-attention between cursor sequence and beatmap sequence."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, cursor_seq: torch.Tensor, beatmap_seq: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-attention between cursor and beatmap sequences.
        
        Args:
            cursor_seq: Cursor sequence of shape (seq_len, batch_size, d_model)
            beatmap_seq: Beatmap sequence of shape (seq_len, batch_size, d_model)
            mask: Optional attention mask
            
        Returns:
            Enhanced cursor sequence with beatmap context
        """
        # Cross-attention: cursor attends to beatmap
        if mask is not None:
            attended, _ = self.attention(cursor_seq, beatmap_seq, beatmap_seq, mask)
        else:
            attended, _ = self.attention(cursor_seq, beatmap_seq, beatmap_seq)
        cursor_seq = self.norm1(cursor_seq + attended)
        
        # Feed-forward
        ffn_output = self.ffn(cursor_seq)
        cursor_seq = self.norm2(cursor_seq + ffn_output)
        
        return cursor_seq


class SelfAttention(nn.Module):
    """Self-attention layer with residual connections and layer normalization."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_flash_attention: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, use_flash_attention)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Self-attention forward pass.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            mask: Attention mask
            key_padding_mask: Key padding mask
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention
        attended, _ = self.attention(x, x, x, mask, key_padding_mask)
        x = self.norm1(x + attended)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class CausalSelfAttention(SelfAttention):
    """Causal self-attention for autoregressive generation."""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 4096,
                 dropout: float = 0.1, use_flash_attention: bool = False):
        super().__init__(d_model, n_heads, dropout, use_flash_attention)
        
        # Register causal mask
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask)
        
    def forward(self, x: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Causal self-attention forward pass.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            key_padding_mask: Key padding mask
            
        Returns:
            Output tensor of same shape as input
        """
        seq_len, batch_size = x.size(0), x.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # Expand causal mask to match attention scores shape [batch_size, n_heads, seq_len, seq_len]
        # causal_mask should be [1, 1, seq_len, seq_len] to broadcast properly
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        return super().forward(x, causal_mask, key_padding_mask)


class RelativePositionalAttention(nn.Module):
    """Attention with relative positional encoding for better temporal modeling."""
    
    def __init__(self, d_model: int, n_heads: int, max_relative_position: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_relative_position = max_relative_position
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        self.relative_position_k = nn.Parameter(torch.randn(2 * max_relative_position + 1, self.d_k))
        self.relative_position_v = nn.Parameter(torch.randn(2 * max_relative_position + 1, self.d_k))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with relative positional attention.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        seq_len, batch_size, _ = x.shape
        
        # Linear projections
        Q = self.w_q(x).view(seq_len, batch_size, self.n_heads, self.d_k)
        K = self.w_k(x).view(seq_len, batch_size, self.n_heads, self.d_k)
        V = self.w_v(x).view(seq_len, batch_size, self.n_heads, self.d_k)
        
        # Compute relative positions
        relative_positions = self._get_relative_positions(seq_len)
        
        # Compute attention with relative positions
        output = self._relative_attention(Q, K, V, relative_positions, mask)
        
        # Reshape and project
        output = output.view(seq_len, batch_size, self.d_model)
        output = self.w_o(output)
        
        return output
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Get relative position indices."""
        positions = torch.arange(seq_len, device=self.relative_position_k.device)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions = torch.clamp(relative_positions, 
                                       -self.max_relative_position, 
                                       self.max_relative_position)
        relative_positions += self.max_relative_position
        return relative_positions
    
    def _relative_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          relative_positions: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention with relative positional encoding."""
        seq_len, batch_size, n_heads, d_k = Q.shape
        
        # Standard attention scores
        scores = torch.einsum('sbhd,tbhd->sbth', Q, K) * self.scale
        
        # Add relative position bias
        rel_k = self.relative_position_k[relative_positions]  # (seq_len, seq_len, d_k)
        rel_scores = torch.einsum('sbhd,std->sbth', Q, rel_k) * self.scale
        scores = scores + rel_scores
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.einsum('sbth,tbhd->sbhd', attention_weights, V)
        
        # Add relative position bias to values
        rel_v = self.relative_position_v[relative_positions]  # (seq_len, seq_len, d_k)
        rel_output = torch.einsum('sbth,std->sbhd', attention_weights, rel_v)
        output = output + rel_output
        
        return output