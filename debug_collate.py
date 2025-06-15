import torch
from torch.utils.data import DataLoader

# Create a small dataset to test collation
class MockDataset:
    def __init__(self):
        pass
    
    def __len__(self):
        return 4
    
    def __getitem__(self, idx):
        return {
            'cursor_data': torch.randn(1024, 2),  # (seq_len, features)
            'beatmap_data': torch.randn(1024, 4),
            'timing_data': torch.randn(1024, 3),
            'key_data': torch.randn(1024, 4),
            'accuracy_target': torch.randn(1)  # Single value per sample
        }

# Test the collate function
dataset = MockDataset()

# Create a mock collate function like in the actual code
def mock_collate_fn(batch):
    collated = {}
    
    for key in batch[0].keys():
        if key == 'accuracy_target':
            # Stack accuracy targets
            collated[key] = torch.stack([item[key] for item in batch])
            print(f"{key} (no transpose): {collated[key].shape}")
        else:
            # Stack sequence data and transpose to (seq_len, batch_size, ...)
            stacked = torch.stack([item[key] for item in batch])
            collated[key] = stacked.transpose(0, 1)
            print(f"{key} (transposed): {collated[key].shape}")
    
    return collated

# Test with batch size 2
batch = [dataset[i] for i in range(2)]
result = mock_collate_fn(batch)

print("\nFinal shapes:")
for key, tensor in result.items():
    print(f"{key}: {tensor.shape}")

print("\nExpected for transformer:")
print("cursor_data, beatmap_data, etc: (seq_len=1024, batch_size=2, features)")
print("accuracy_target: (batch_size=2, 1) for conditioning")