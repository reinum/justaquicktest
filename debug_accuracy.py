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
        print("Input shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
        
        # Test accuracy conditioning
        accuracy_emb = model.accuracy_conditioning(batch['accuracy_target'])
        print(f"\naccuracy_emb shape: {accuracy_emb.shape}")
        
        seq_len = batch['cursor_data'].shape[0]
        batch_size = batch['cursor_data'].shape[1]
        print(f"seq_len: {seq_len}, batch_size: {batch_size}")
        
        # Test expansion
        accuracy_emb_expanded = accuracy_emb.unsqueeze(0).expand(seq_len, -1, -1)
        print(f"accuracy_emb_expanded shape: {accuracy_emb_expanded.shape}")
        
        break
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()