import torch

# Simulate the actual shapes from the debug output
seq_len = 2
batch_size = 1024
d_model = 512

# Create tensors with the actual shapes
x = torch.randn(2, 1024, 512)  # (seq_len, batch_size, d_model)
accuracy_target = torch.randn(2, 1)  # (batch_size, 1) - but this is wrong shape

print(f"x.shape: {x.shape}")
print(f"accuracy_target.shape: {accuracy_target.shape}")

# The issue: accuracy_target should be (batch_size, 1) = (1024, 1)
# But we're getting (2, 1) which suggests batch_size=2, not 1024

# Let's simulate the correct scenario
accuracy_target_correct = torch.randn(1024, 1)  # Correct shape
accuracy_emb = torch.randn(1024, 512)  # After processing through AccuracyConditioning

print(f"\nCorrect scenario:")
print(f"accuracy_target_correct.shape: {accuracy_target_correct.shape}")
print(f"accuracy_emb.shape: {accuracy_emb.shape}")

# Try different broadcasting methods
print(f"\nTesting broadcasting methods:")

# Method 1: unsqueeze(0) + repeat
accuracy_emb_broadcast1 = accuracy_emb.unsqueeze(0).repeat(seq_len, 1, 1)
print(f"Method 1 - unsqueeze(0) + repeat: {accuracy_emb_broadcast1.shape}")

# Method 2: unsqueeze(0) + expand
accuracy_emb_broadcast2 = accuracy_emb.unsqueeze(0).expand(seq_len, -1, -1)
print(f"Method 2 - unsqueeze(0) + expand: {accuracy_emb_broadcast2.shape}")

# Test addition
try:
    result1 = x + accuracy_emb_broadcast1
    print(f"Method 1 addition successful: {result1.shape}")
except Exception as e:
    print(f"Method 1 addition failed: {e}")

try:
    result2 = x + accuracy_emb_broadcast2
    print(f"Method 2 addition successful: {result2.shape}")
except Exception as e:
    print(f"Method 2 addition failed: {e}")

# Now test with the wrong shapes we're actually getting
print(f"\nTesting with actual wrong shapes:")
accuracy_emb_wrong = torch.randn(2, 512)  # What we're actually getting
print(f"accuracy_emb_wrong.shape: {accuracy_emb_wrong.shape}")

try:
    accuracy_emb_broadcast_wrong = accuracy_emb_wrong.unsqueeze(0).repeat(seq_len, 1, 1)
    print(f"Wrong broadcast shape: {accuracy_emb_broadcast_wrong.shape}")
    result_wrong = x + accuracy_emb_broadcast_wrong
    print(f"Wrong addition successful: {result_wrong.shape}")
except Exception as e:
    print(f"Wrong addition failed: {e}")