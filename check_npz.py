#!/usr/bin/env python3

import numpy as np

# Check what's in the NPZ file
data = np.load('generated_replay.npz', allow_pickle=True)
print('Keys in NPZ file:', list(data.keys()))
print('\nShapes:')
for k in data.keys():
    print(f'{k}: {data[k].shape}')
    
print('\nFirst few values:')
for k in data.keys():
    print(f'{k}:')
    if k == 'metadata':
        print(data[k])  # metadata is likely a single object
    else:
        print(data[k][:5] if len(data[k]) > 5 else data[k])
    print()