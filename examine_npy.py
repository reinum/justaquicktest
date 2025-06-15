import numpy as np
import os

# Load a sample numpy file
npy_file = 'reduced_dataset_5/replays/npy/6170131dcd9ad32d4c0287422b256423.npy'

if os.path.exists(npy_file):
    data = np.load(npy_file)
    print(f'Shape: {data.shape}')
    print(f'Data type: {data.dtype}')
    print(f'First few rows:')
    print(data[:5])
    print(f'\nLast few rows:')
    print(data[-5:])
    print(f'\nColumn statistics:')
    if len(data.shape) == 2:
        for i in range(min(10, data.shape[1])):
            col_data = data[:, i]
            print(f'Column {i}: min={col_data.min():.3f}, max={col_data.max():.3f}, mean={col_data.mean():.3f}')
else:
    print(f'File not found: {npy_file}')
    print('Available files:')
    npy_dir = 'reduced_dataset_5/replays/npy'
    if os.path.exists(npy_dir):
        files = os.listdir(npy_dir)
        for f in files[:5]:
            print(f'  {f}')