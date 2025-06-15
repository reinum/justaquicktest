"""Sampling strategies for replay generation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from abc import ABC, abstractmethod
import math


class SamplingStrategy(ABC):
    """Base class for sampling strategies."""
    
    @abstractmethod
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from logits.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            
        Returns:
            Sampled indices [batch_size]
        """
        pass


class TemperatureSampling(SamplingStrategy):
    """Temperature-based sampling."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample using temperature scaling."""
        if self.temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TopKSampling(SamplingStrategy):
    """Top-k sampling."""
    
    def __init__(self, k: int = 50, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from top-k most likely tokens."""
        if self.k <= 0:
            return TemperatureSampling(self.temperature).sample(logits)
        
        # Get top-k values and indices
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        
        # Apply temperature
        if self.temperature != 1.0:
            top_k_logits = top_k_logits / self.temperature
        
        # Sample from top-k
        probs = F.softmax(top_k_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Map back to original vocabulary
        return torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)


class TopPSampling(SamplingStrategy):
    """Top-p (nucleus) sampling."""
    
    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        self.p = p
        self.temperature = temperature
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from nucleus of tokens with cumulative probability p."""
        if self.p >= 1.0:
            return TemperatureSampling(self.temperature).sample(logits)
        
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        
        # Create mask for tokens to keep
        mask = cumulative_probs <= self.p
        
        # Ensure at least one token is kept
        mask[..., 0] = True
        
        # Zero out probabilities for tokens outside nucleus
        filtered_probs = probs * mask.float()
        
        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        # Sample from filtered distribution
        sampled_indices = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        
        # Map back to original vocabulary
        return torch.gather(sorted_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)


class NucleusSampling(TopPSampling):
    """Alias for TopPSampling."""
    pass


class BeamSearch:
    """Beam search for sequence generation."""
    
    def __init__(self, beam_size: int = 5, max_length: int = 1000, 
                 length_penalty: float = 1.0, early_stopping: bool = True):
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
    
    def search(self, model, initial_input: Dict[str, torch.Tensor], 
               eos_token_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform beam search.
        
        Args:
            model: The model to use for generation
            initial_input: Initial input dictionary
            eos_token_id: End-of-sequence token ID
            
        Returns:
            List of beam results with scores and sequences
        """
        device = next(model.parameters()).device
        batch_size = initial_input['cursor_pos'].size(0)
        
        # Initialize beams
        beams = []
        for b in range(batch_size):
            beam_input = {k: v[b:b+1] for k, v in initial_input.items()}
            beams.append({
                'input': beam_input,
                'score': 0.0,
                'finished': False,
                'length': beam_input['cursor_pos'].size(1)
            })
        
        finished_beams = [[] for _ in range(batch_size)]
        
        for step in range(self.max_length):
            if all(len(finished) >= self.beam_size for finished in finished_beams):
                break
            
            # Collect active beams
            active_beams = []
            for b in range(batch_size):
                active_beams.extend([
                    (b, beam) for beam in beams 
                    if not beam['finished'] and len(finished_beams[b]) < self.beam_size
                ])
            
            if not active_beams:
                break
            
            # Generate next tokens for all active beams
            all_candidates = []
            
            for batch_idx, beam in active_beams:
                with torch.no_grad():
                    outputs = model(beam['input'])
                    
                    # Get logits for next token
                    cursor_logits = outputs['cursor_pos'][:, -1, :]  # [1, 2]
                    key_logits = outputs['key_press'][:, -1, :]     # [1, 4]
                    
                    # For simplicity, we'll focus on cursor prediction
                    # In practice, you'd want to handle both cursor and keys
                    
                    # Sample top candidates
                    top_k = min(self.beam_size * 2, cursor_logits.size(-1))
                    top_scores, top_indices = torch.topk(cursor_logits, top_k, dim=-1)
                    
                    for i in range(top_k):
                        score = beam['score'] + top_scores[0, i].item()
                        
                        # Apply length penalty
                        length_penalty = ((beam['length'] + 1) / 6) ** self.length_penalty
                        normalized_score = score / length_penalty
                        
                        candidate = {
                            'batch_idx': batch_idx,
                            'parent_beam': beam,
                            'score': score,
                            'normalized_score': normalized_score,
                            'next_token': top_indices[0, i].item(),
                            'length': beam['length'] + 1
                        }
                        
                        all_candidates.append(candidate)
            
            # Select best candidates for each batch
            new_beams = []
            for b in range(batch_size):
                batch_candidates = [c for c in all_candidates if c['batch_idx'] == b]
                batch_candidates.sort(key=lambda x: x['normalized_score'], reverse=True)
                
                for candidate in batch_candidates[:self.beam_size]:
                    # Create new beam
                    parent = candidate['parent_beam']
                    
                    # Extend sequence (simplified - in practice you'd properly extend the input)
                    new_input = {k: v.clone() for k, v in parent['input'].items()}
                    
                    new_beam = {
                        'input': new_input,
                        'score': candidate['score'],
                        'finished': False,
                        'length': candidate['length']
                    }
                    
                    # Check if finished
                    if (eos_token_id is not None and 
                        candidate['next_token'] == eos_token_id) or \
                       candidate['length'] >= self.max_length:
                        new_beam['finished'] = True
                        finished_beams[b].append(new_beam)
                    else:
                        new_beams.append(new_beam)
            
            beams = new_beams
        
        # Return best beams for each batch
        results = []
        for b in range(batch_size):
            batch_beams = finished_beams[b]
            if not batch_beams:
                # No finished beams, take best active beam
                active = [beam for beam in beams if beam.get('batch_idx') == b]
                if active:
                    batch_beams = [max(active, key=lambda x: x['score'])]
            
            if batch_beams:
                best_beam = max(batch_beams, key=lambda x: x['score'])
                results.append(best_beam)
            else:
                results.append(None)
        
        return results


class AdaptiveSampling:
    """Adaptive sampling that adjusts strategy based on context."""
    
    def __init__(self, strategies: Dict[str, SamplingStrategy], 
                 adaptation_fn: Optional[callable] = None):
        self.strategies = strategies
        self.adaptation_fn = adaptation_fn or self._default_adaptation
        self.current_strategy = list(strategies.keys())[0]
    
    def _default_adaptation(self, step: int, context: Dict[str, Any]) -> str:
        """Default adaptation function."""
        # Simple rule: use temperature for early steps, top-p for later
        if step < 100:
            return 'temperature'
        else:
            return 'top_p'
    
    def sample(self, logits: torch.Tensor, step: int = 0, 
               context: Optional[Dict[str, Any]] = None, **kwargs) -> torch.Tensor:
        """Sample using adaptive strategy."""
        context = context or {}
        strategy_name = self.adaptation_fn(step, context)
        
        if strategy_name in self.strategies:
            self.current_strategy = strategy_name
            return self.strategies[strategy_name].sample(logits, **kwargs)
        else:
            return self.strategies[self.current_strategy].sample(logits, **kwargs)


class CursorSampling:
    """Specialized sampling for cursor positions."""
    
    def __init__(self, smoothness_weight: float = 0.1, 
                 boundary_penalty: float = 0.2):
        self.smoothness_weight = smoothness_weight
        self.boundary_penalty = boundary_penalty
    
    def sample(self, cursor_logits: torch.Tensor, 
               previous_pos: Optional[torch.Tensor] = None,
               screen_bounds: Tuple[int, int] = (512, 384)) -> torch.Tensor:
        """Sample cursor position with smoothness and boundary constraints.
        
        Args:
            cursor_logits: Logits for cursor position [batch_size, 2]
            previous_pos: Previous cursor position [batch_size, 2]
            screen_bounds: Screen dimensions (width, height)
            
        Returns:
            Sampled cursor position [batch_size, 2]
        """
        # Convert logits to coordinates
        cursor_pos = torch.tanh(cursor_logits)  # [-1, 1]
        
        # Scale to screen coordinates
        cursor_pos[:, 0] = (cursor_pos[:, 0] + 1) * screen_bounds[0] / 2
        cursor_pos[:, 1] = (cursor_pos[:, 1] + 1) * screen_bounds[1] / 2
        
        # Apply smoothness constraint
        if previous_pos is not None and self.smoothness_weight > 0:
            # Reshape previous_pos to match cursor_pos dimensions
            if previous_pos.dim() == 3:
                previous_pos = previous_pos.squeeze(1)  # [batch, 1, 2] -> [batch, 2]
            
            # Limit maximum movement - increased from 0.1 to 0.5 for realistic osu! movement
            max_movement = min(screen_bounds) * 0.5  # 50% of smaller dimension (~192 pixels)
            
            movement = cursor_pos - previous_pos
            movement_magnitude = torch.norm(movement, dim=-1, keepdim=True)
            
            # Scale down large movements
            scale_factor = torch.clamp(max_movement / (movement_magnitude + 1e-8), max=1.0)
            cursor_pos = previous_pos + movement * scale_factor
        
        # Apply boundary constraints
        cursor_pos[:, 0] = torch.clamp(cursor_pos[:, 0], 0, screen_bounds[0])
        cursor_pos[:, 1] = torch.clamp(cursor_pos[:, 1], 0, screen_bounds[1])
        
        return cursor_pos


class KeySampling:
    """Specialized sampling for key presses."""
    
    def __init__(self, temperature: float = 1.0, 
                 key_constraints: Optional[Dict[str, float]] = None):
        self.temperature = temperature
        self.key_constraints = key_constraints or {}
    
    def sample(self, key_logits: torch.Tensor, 
               beatmap_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Sample key presses with game-specific constraints.
        
        Args:
            key_logits: Logits for key presses [batch_size, 4] (K1, K2, M1, M2)
            beatmap_context: Context about current beatmap state
            
        Returns:
            Sampled key states [batch_size, 4]
        """
        # Apply temperature
        if self.temperature != 1.0:
            key_logits = key_logits / self.temperature
        
        # Apply constraints based on beatmap context
        if beatmap_context:
            # Example: reduce mouse key probability during streams
            if beatmap_context.get('is_stream', False):
                key_logits[:, 2:] *= 0.1  # Reduce mouse key probability
            
            # Example: enforce alternating for certain patterns
            if beatmap_context.get('force_alternating', False):
                last_key = beatmap_context.get('last_key')
                if last_key == 0:  # Last was K1
                    key_logits[:, 0] *= 0.1  # Reduce K1 probability
                elif last_key == 1:  # Last was K2
                    key_logits[:, 1] *= 0.1  # Reduce K2 probability
        
        # Sample binary key states
        key_probs = torch.sigmoid(key_logits)
        key_states = torch.bernoulli(key_probs)
        
        return key_states


def create_sampling_strategy(config: Dict[str, Any]) -> SamplingStrategy:
    """Factory function to create sampling strategy from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Sampling strategy instance
    """
    strategy_type = config.get('type', 'temperature')
    
    if strategy_type == 'temperature':
        return TemperatureSampling(
            temperature=config.get('temperature', 1.0)
        )
    
    elif strategy_type == 'top_k':
        return TopKSampling(
            k=config.get('k', 50),
            temperature=config.get('temperature', 1.0)
        )
    
    elif strategy_type == 'top_p' or strategy_type == 'nucleus':
        return TopPSampling(
            p=config.get('p', 0.9),
            temperature=config.get('temperature', 1.0)
        )
    
    elif strategy_type == 'adaptive':
        strategies = {}
        for name, strategy_config in config.get('strategies', {}).items():
            strategies[name] = create_sampling_strategy(strategy_config)
        
        return AdaptiveSampling(strategies)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy_type}")