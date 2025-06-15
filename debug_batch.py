import sys
import os
sys.path.append('src')

from src.data.dataset import OsuReplayDataset, ReplayDataLoader
from src.config.model_config import DataConfig
import torch

# Setup data config
data_config = DataConfig()
data_config.replay_dir = 'reduced_dataset_5/replays/npy'
data_config.csv_path = 'reduced_dataset_5/index.csv'
data_config.beatmap_dir = 'reduced_dataset_5/beatmaps'

try:
    dataset = OsuReplayDataset(data_config=data_config, split='train')
    dataloader = ReplayDataLoader(dataset, batch_size=2, shuffle=False)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataloader length: {len(dataloader)}")
    
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"{key}: {type(value)} - {value}")
        break  # Only check first batch
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()