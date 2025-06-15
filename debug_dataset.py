import sys
import os
sys.path.append('src')

from src.data.dataset import OsuReplayDataset
from src.config.model_config import DataConfig

# Setup data config
data_config = DataConfig()
data_config.replay_dir = 'reduced_dataset_5/replays/npy'
data_config.csv_path = 'reduced_dataset_5/index.csv'
data_config.beatmap_dir = 'reduced_dataset_5/beatmaps'

print(f"Replay dir: {data_config.replay_dir}")
print(f"Replay dir exists: {os.path.exists(data_config.replay_dir)}")
if os.path.exists(data_config.replay_dir):
    files = os.listdir(data_config.replay_dir)
    print(f"Files in replay dir: {len(files)}")
    if files:
        print(f"First few files: {files[:5]}")

print(f"CSV path: {data_config.csv_path}")
print(f"CSV exists: {os.path.exists(data_config.csv_path)}")

try:
    dataset = OsuReplayDataset(data_config=data_config, split='train')
    print(f"Dataset length: {len(dataset)}")
except Exception as e:
    print(f"Error creating dataset: {e}")
    import traceback
    traceback.print_exc()