"""Main replay generator for osu! AI model."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import time
import logging
from dataclasses import dataclass

from ..models.transformer import OsuTransformer
from ..config.model_config import GenerationConfig, ModelConfig
from ..data.beatmap_parser import BeatmapParser
from .sampling import (
    SamplingStrategy, CursorSampling, KeySampling, 
    create_sampling_strategy, BeamSearch
)
from .postprocess import ReplayPostProcessor


@dataclass
class GenerationResult:
    """Result of replay generation."""
    cursor_positions: np.ndarray  # [seq_len, 2]
    key_presses: np.ndarray      # [seq_len, 4]
    timestamps: np.ndarray       # [seq_len]
    metadata: Dict[str, Any]
    generation_time: float
    confidence_scores: Optional[np.ndarray] = None


class ReplayGenerator:
    """Main class for generating osu! replays using the trained model."""
    
    def __init__(self, 
                 model: OsuTransformer,
                 config: GenerationConfig,
                 model_config: ModelConfig,
                 device: torch.device,
                 logger: Optional[logging.Logger] = None):
        
        self.model = model.to(device)
        self.config = config
        self.model_config = model_config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize sampling strategies
        self._setup_sampling_strategies()
        
        # Initialize post-processor
        self.post_processor = ReplayPostProcessor()
        
        # Initialize beatmap parser
        self.beatmap_parser = BeatmapParser()
        
        self.logger.info("ReplayGenerator initialized")
    
    def _setup_sampling_strategies(self):
        """Setup sampling strategies for generation."""
        # Main sampling strategy
        self.sampling_strategy = create_sampling_strategy({
            'type': self.config.sampling_strategy,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p
        })
        
        # Specialized samplers
        self.cursor_sampler = CursorSampling(
            smoothness_weight=self.config.smoothness_weight,
            boundary_penalty=0.2
        )
        
        self.key_sampler = KeySampling(
            temperature=self.config.temperature * 0.8,  # Slightly lower for keys
            key_constraints={}
        )
        
        # Beam search (if enabled)
        if self.config.use_beam_search:
            self.beam_search = BeamSearch(
                beam_size=self.config.beam_size,
                max_length=self.config.max_length,
                length_penalty=self.config.length_penalty
            )
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Model loaded from {checkpoint_path}")
    
    def generate_replay(self, 
                       beatmap_path: Path,
                       target_accuracy: float = 0.95,
                       mods: Optional[List[str]] = None,
                       seed: Optional[int] = None) -> GenerationResult:
        """Generate a complete replay for a beatmap.
        
        Args:
            beatmap_path: Path to .osu beatmap file
            target_accuracy: Target accuracy (0.0 to 1.0)
            mods: List of mod strings (e.g., ['HD', 'DT'])
            seed: Random seed for reproducibility
            
        Returns:
            GenerationResult containing the generated replay
        """
        start_time = time.time()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Parse beatmap
        beatmap_data = self.beatmap_parser.parse_beatmap(str(beatmap_path))
        
        # Prepare input data
        input_data = self._prepare_input(beatmap_data, target_accuracy, mods)
        
        # Generate replay
        if self.config.use_beam_search:
            result = self._generate_with_beam_search(input_data, beatmap_data)
        else:
            result = self._generate_autoregressive(input_data, beatmap_data)
        
        # Post-process
        # Convert beatmap_data list to dict format expected by post-processor
        beatmap_dict = {
            'hit_objects': beatmap_data,
            'path': str(beatmap_path)
        }
        result = self.post_processor.process(result, beatmap_dict)
        
        generation_time = time.time() - start_time
        result.generation_time = generation_time
        
        self.logger.info(f"Replay generated in {generation_time:.2f}s")
        
        return result
    
    def _prepare_input(self, 
                      beatmap_data: List[Dict[str, Any]],
                      target_accuracy: float,
                      mods: Optional[List[str]]) -> Dict[str, torch.Tensor]:
        """Prepare input tensors for the model."""
        # Extract beatmap features
        hit_objects = beatmap_data if beatmap_data else []
        timing_points = []  # Timing points not available from current parser
        
        # Create sequences
        seq_len = len(hit_objects)
        
        # Beatmap encoding
        beatmap_features = self._encode_beatmap(hit_objects, timing_points)
        
        # Timing information
        timestamps = torch.tensor([obj['Time'] for obj in hit_objects], 
                                dtype=torch.float32).unsqueeze(0)
        
        # Accuracy conditioning
        accuracy_tensor = torch.full((1, seq_len), target_accuracy, dtype=torch.float32)
        
        # Mod encoding (simplified)
        mod_encoding = self._encode_mods(mods or [])
        mod_tensor = mod_encoding.unsqueeze(0).repeat(1, seq_len, 1)
        
        # Initialize cursor and key sequences
        cursor_sequence = torch.zeros((1, seq_len, 2), dtype=torch.float32)
        cursor_sequence[0, 0] = torch.tensor([256.0, 192.0])  # Start at center
        
        key_sequence = torch.zeros((1, seq_len, 4), dtype=torch.float32)
        
        input_data = {
            'beatmap_features': beatmap_features.unsqueeze(0),
            'timestamps': timestamps,
            'accuracy_target': accuracy_tensor,
            'mod_encoding': mod_tensor,
            'cursor_pos': cursor_sequence,
            'key_press': key_sequence,
            'attention_mask': torch.ones((1, seq_len), dtype=torch.bool)
        }
        
        # Move to device
        input_data = {k: v.to(self.device) for k, v in input_data.items()}
        
        return input_data
    
    def _encode_beatmap(self, hit_objects: List[Dict], 
                       timing_points: List[Dict]) -> torch.Tensor:
        """Encode beatmap features."""
        features = []
        
        for obj in hit_objects:
            # Basic features: Time, X, Y, Type (to match BeatmapEncoder expectation)
            obj_features = [
                obj['Time'] / 1000.0,  # Convert to seconds (time)
                obj['X'] / 512.0,  # Normalize x
                obj['Y'] / 384.0,  # Normalize y
                1.0 if obj['Type'] == 'hitcircle' else 2.0,  # Object type encoding
                0.0,  # Placeholder feature 1
                0.0,  # Placeholder feature 2
                0.0,  # Placeholder feature 3
                0.0,  # Placeholder feature 4
            ]
            
            features.append(obj_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_mods(self, mods: List[str]) -> torch.Tensor:
        """Encode mod information."""
        # Simple one-hot encoding for common mods
        mod_dict = {
            'NM': 0, 'HD': 1, 'HR': 2, 'DT': 3, 'NC': 4, 'FL': 5,
            'EZ': 6, 'HT': 7, 'SO': 8, 'NF': 9
        }
        
        encoding = torch.zeros(len(mod_dict))
        for mod in mods:
            if mod in mod_dict:
                encoding[mod_dict[mod]] = 1.0
        
        return encoding
    
    def _generate_autoregressive(self, 
                               input_data: Dict[str, torch.Tensor],
                               beatmap_data: List[Dict[str, Any]]) -> GenerationResult:
        """Generate replay using autoregressive sampling."""
        hit_objects = beatmap_data
        seq_len = len(hit_objects)
        
        # Initialize output sequences
        cursor_positions = []
        key_presses = []
        timestamps = []
        confidence_scores = []
        
        # Get full sequences for updating
        cursor_sequence = input_data['cursor_pos'].clone()
        key_sequence = input_data['key_press'].clone()
        
        with torch.no_grad():
            for step in range(seq_len):
                # Prepare input for current step
                step_input = {
                    'beatmap_features': input_data['beatmap_features'][:, :step+1],
                    'timestamps': input_data['timestamps'][:, :step+1],
                    'accuracy_target': input_data['accuracy_target'][:, :step+1],
                    'mod_encoding': input_data['mod_encoding'][:, :step+1],
                    'cursor_pos': cursor_sequence[:, :step+1],
                    'key_press': key_sequence[:, :step+1],
                    'attention_mask': input_data['attention_mask'][:, :step+1]
                }
                
                # Forward pass - extract individual tensors for model
                cursor_data = step_input['cursor_pos'].transpose(0, 1)  # [1, 1, 2] -> [1, 1, 2]
                beatmap_data = step_input['beatmap_features'].transpose(0, 1)  # [1, seq, features] -> [seq, 1, features]
                timing_data = step_input['timestamps'].unsqueeze(-1).transpose(0, 1)  # [1, seq] -> [seq, 1, 1]
                key_data = step_input['key_press'].transpose(0, 1)  # [1, 1, 4] -> [1, 1, 4]
                accuracy_data = step_input['accuracy_target'].transpose(0, 1)  # [1, seq] -> [seq, 1]
                
                outputs = self.model(
                    cursor_data=cursor_data,
                    beatmap_data=beatmap_data,
                    timing_data=timing_data,
                    key_data=key_data,
                    accuracy_target=accuracy_data,
                    mask=step_input['attention_mask']
                )
                
                # Extract predictions for current step
                # Model outputs are (seq_len, batch_size, features)
                print(f"DEBUG: outputs['cursor_pred'].shape: {outputs['cursor_pred'].shape}")
                print(f"DEBUG: outputs['key_pred'].shape: {outputs['key_pred'].shape}")
                cursor_logits = outputs['cursor_pred'][-1, :, :]  # [1, 2]
                key_logits = outputs['key_pred'][-1, :, :]      # [1, 4]
                print(f"DEBUG: cursor_logits.shape: {cursor_logits.shape}")
                print(f"DEBUG: key_logits.shape: {key_logits.shape}")
                print(f"DEBUG: cursor_sequence.shape: {cursor_sequence.shape}")
                print(f"DEBUG: step: {step}")
                
                # Sample cursor position
                previous_pos = cursor_sequence[:, step-1:step] if step > 0 else cursor_sequence[:, :1]
                print(f"DEBUG: previous_pos.shape: {previous_pos.shape}")
                cursor_pos = self.cursor_sampler.sample(
                    cursor_logits, 
                    previous_pos=previous_pos,
                    screen_bounds=(512, 384)
                )
                print(f"DEBUG: cursor_pos.shape: {cursor_pos.shape}")
                
                # Sample key presses
                beatmap_context = {
                    'current_object': hit_objects[step],
                    'is_stream': self._is_stream_section(hit_objects, step),
                    'last_key': self._get_last_pressed_key(key_sequence[:, step-1:step] if step > 0 else key_sequence[:, :1])
                }
                
                key_states = self.key_sampler.sample(
                    key_logits,
                    beatmap_context=beatmap_context
                )
                
                # Calculate confidence (simplified)
                cursor_confidence = torch.softmax(cursor_logits, dim=-1).max().item()
                key_confidence = torch.sigmoid(key_logits).mean().item()
                confidence = (cursor_confidence + key_confidence) / 2
                
                # Store results
                cursor_positions.append(cursor_pos.cpu().numpy()[0])
                key_presses.append(key_states.cpu().numpy()[0])
                timestamps.append(hit_objects[step]['Time'])
                confidence_scores.append(confidence)
                
                # Update sequences
                # Ensure cursor_pos and key_states have correct batch dimension
                if cursor_pos.shape[0] != 1:
                    cursor_pos = cursor_pos[:1]  # Take only first batch element
                if key_states.shape[0] != 1:
                    key_states = key_states[:1]  # Take only first batch element
                    
                cursor_sequence[:, step] = cursor_pos.squeeze(0)
                key_sequence[:, step] = key_states.squeeze(0)
                
                # Progress logging
                if step % 100 == 0:
                    self.logger.debug(f"Generated step {step}/{seq_len}")
        
        return GenerationResult(
            cursor_positions=np.array(cursor_positions),
            key_presses=np.array(key_presses),
            timestamps=np.array(timestamps),
            confidence_scores=np.array(confidence_scores),
            metadata={
                'beatmap_path': '',  # Not available in current beatmap_data format
                'target_accuracy': input_data['accuracy_target'][0, 0].item(),
                'generation_method': 'autoregressive',
                'model_config': self.model_config.__dict__
            },
            generation_time=0.0  # Will be set by caller
        )
    
    def _generate_with_beam_search(self, 
                                 input_data: Dict[str, torch.Tensor],
                                 beatmap_data: Dict[str, Any]) -> GenerationResult:
        """Generate replay using beam search."""
        # Simplified beam search implementation
        # In practice, this would be more sophisticated
        
        with torch.no_grad():
            beam_results = self.beam_search.search(
                self.model, 
                input_data, 
                eos_token_id=None
            )
        
        # Extract best result
        best_result = beam_results[0] if beam_results[0] is not None else None
        
        if best_result is None:
            # Fallback to autoregressive
            return self._generate_autoregressive(input_data, beatmap_data)
        
        # Convert beam result to GenerationResult
        # This is simplified - in practice you'd extract the full sequence
        return self._generate_autoregressive(input_data, beatmap_data)
    
    def _is_stream_section(self, hit_objects: List[Dict], current_idx: int) -> bool:
        """Check if current section is a stream (rapid consecutive hits)."""
        if current_idx < 2 or current_idx >= len(hit_objects) - 2:
            return False
        
        # Check timing between consecutive objects
        window = hit_objects[current_idx-2:current_idx+3]
        intervals = []
        
        for i in range(len(window) - 1):
            interval = window[i+1]['Time'] - window[i]['Time']
            intervals.append(interval)
        
        # Consider it a stream if intervals are consistently short
        avg_interval = np.mean(intervals)
        return avg_interval < 150  # Less than 150ms between hits
    
    def _get_last_pressed_key(self, key_states: torch.Tensor) -> Optional[int]:
        """Get the index of the last pressed key."""
        # Handle different tensor shapes: (batch, seq, features) or (batch, features)
        if key_states.dim() == 3:
            # Shape: (batch, seq, features) - take the last sequence element
            last_keys = key_states[0, -1, :]
        else:
            # Shape: (batch, features)
            last_keys = key_states[0, :]
        
        pressed_keys = torch.where(last_keys > 0.5)[0]
        return pressed_keys[-1].item() if len(pressed_keys) > 0 else None
    
    def generate_batch(self, 
                      beatmap_paths: List[Path],
                      target_accuracies: List[float],
                      mods_list: Optional[List[List[str]]] = None,
                      batch_size: int = 4) -> List[GenerationResult]:
        """Generate replays for multiple beatmaps in batches."""
        results = []
        
        for i in range(0, len(beatmap_paths), batch_size):
            batch_paths = beatmap_paths[i:i+batch_size]
            batch_accuracies = target_accuracies[i:i+batch_size]
            batch_mods = mods_list[i:i+batch_size] if mods_list else [None] * len(batch_paths)
            
            batch_results = []
            for path, accuracy, mods in zip(batch_paths, batch_accuracies, batch_mods):
                try:
                    result = self.generate_replay(path, accuracy, mods)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to generate replay for {path}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        return results
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the generation process."""
        return {
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'sampling_strategy': self.config.sampling_strategy,
            'temperature': self.config.temperature,
            'use_beam_search': self.config.use_beam_search,
            'beam_size': self.config.beam_size if self.config.use_beam_search else None
        }