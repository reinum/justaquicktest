import sys
import os
sys.path.append('src')

from src.data.dataset import OsuReplayDataset, ReplayDataLoader
from src.config.model_config import DataConfig, ModelConfig
from src.models.transformer import OsuTransformer
import torch

# Setup configs
data_config = DataConfig()
data_config.replay_dir = 'reduced_dataset_5/replays/npy'
data_config.csv_path = 'reduced_dataset_5/index.csv'
data_config.beatmap_dir = 'reduced_dataset_5/beatmaps'

model_config = ModelConfig()

try:
    dataset = OsuReplayDataset(data_config=data_config, split='train')
    dataloader = ReplayDataLoader(dataset, batch_size=2, shuffle=False)
    
    model = OsuTransformer(model_config)
    
    for batch in dataloader:
        print("Testing forward pass step by step...")
        
        cursor_data = batch['cursor_data']
        beatmap_data = batch['beatmap_data']
        timing_data = batch['timing_data']
        key_data = batch['key_data']
        accuracy_target = batch['accuracy_target']
        
        print(f"cursor_data: {cursor_data.shape}")
        print(f"beatmap_data: {beatmap_data.shape}")
        print(f"timing_data: {timing_data.shape}")
        print(f"key_data: {key_data.shape}")
        print(f"accuracy_target: {accuracy_target.shape}")
        
        seq_len, batch_size = cursor_data.shape[:2]
        print(f"seq_len: {seq_len}, batch_size: {batch_size}")
        
        # Encode inputs
        cursor_emb = model.cursor_encoding(cursor_data, key_data)
        print(f"cursor_emb: {cursor_emb.shape}")
        
        beatmap_emb = model.beatmap_encoding(beatmap_data)
        print(f"beatmap_emb: {beatmap_emb.shape}")
        
        timing_emb = model.timing_encoding(timing_data)
        print(f"timing_emb: {timing_emb.shape}")
        
        # Add positional encoding
        cursor_emb = model.positional_encoding(cursor_emb)
        beatmap_emb = model.positional_encoding(beatmap_emb)
        print(f"cursor_emb after pos: {cursor_emb.shape}")
        print(f"beatmap_emb after pos: {beatmap_emb.shape}")
        
        # Add timing information
        cursor_emb = cursor_emb + timing_emb
        beatmap_emb = beatmap_emb + timing_emb
        print(f"cursor_emb after timing: {cursor_emb.shape}")
        print(f"beatmap_emb after timing: {beatmap_emb.shape}")
        
        # Combine cursor and beatmap embeddings
        combined_emb = torch.cat([cursor_emb, beatmap_emb], dim=-1)
        print(f"combined_emb: {combined_emb.shape}")
        
        x = model.input_projection(combined_emb)
        print(f"x after projection: {x.shape}")
        
        # Add accuracy conditioning
        accuracy_emb = model.accuracy_conditioning(accuracy_target)
        print(f"accuracy_emb: {accuracy_emb.shape}")
        
        accuracy_emb = accuracy_emb.unsqueeze(0).expand(seq_len, -1, -1)
        print(f"accuracy_emb expanded: {accuracy_emb.shape}")
        
        print(f"About to add x {x.shape} + accuracy_emb {accuracy_emb.shape}")
        
        break
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()