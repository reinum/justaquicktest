import torch

# Test mask expansion
mask = torch.ones(1024, 2)
loss = torch.ones(1024, 2, 2)

print('mask shape:', mask.shape)
print('loss shape:', loss.shape)

try:
    expanded = mask.unsqueeze(-1).expand_as(loss)
    print('expanded shape:', expanded.shape)
    print('Expansion successful')
except Exception as e:
    print('Expansion failed:', e)

# Test the actual operation
try:
    result = loss * expanded
    print('Multiplication successful, result shape:', result.shape)
except Exception as e:
    print('Multiplication failed:', e)