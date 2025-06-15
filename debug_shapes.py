import sys
import os
sys.path.append('src')

from src.data.dataset import OsuReplayDataset
from src.config.model_config import DataConfig
import torch

# Setup data config
data_config = DataConfig()
data_config.replay_dir = 'reduced_dataset_5/replays/npy'
data_config.csv_path = 'reduced_dataset_5/index.csv'
data_config.beatmap_dir = 'reduced_dataset_5/beatmaps'

try:
    dataset = OsuReplayDataset(data_config=data_config, split='train')
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample keys:", list(sample.keys()))
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape {value.shape}, dtype {value.dtype}")
                if key == 'beatmap_data':
                    print(f"  First few values: {value[:3, :, :] if len(value.shape) == 3 else value[:3]}")
            else:
                print(f"{key}: {type(value)} - {value}")
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()