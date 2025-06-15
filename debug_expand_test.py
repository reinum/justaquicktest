import torch

# Test the exact scenario from the debug output
seq_len = 2
batch_size = 1024
d_model = 512

# Create accuracy_emb with the shape we're getting
accuracy_emb = torch.randn(batch_size, d_model)  # [1024, 512]
print(f"Original accuracy_emb shape: {accuracy_emb.shape}")

# Test the expand operation
try:
    # This is what the code is doing
    accuracy_emb_expanded = accuracy_emb.unsqueeze(0).expand(seq_len, -1, -1)
    print(f"Expanded shape: {accuracy_emb_expanded.shape}")
    print("Expand operation successful!")
except Exception as e:
    print(f"Expand operation failed: {e}")

# But wait, let me check what we're actually getting from debug
# The debug shows accuracy_emb has shape [2, 512], not [1024, 512]
accuracy_emb_wrong = torch.randn(2, 512)  # What debug shows
print(f"\nWrong accuracy_emb shape: {accuracy_emb_wrong.shape}")

try:
    accuracy_emb_wrong_expanded = accuracy_emb_wrong.unsqueeze(0).expand(seq_len, -1, -1)
    print(f"Wrong expanded shape: {accuracy_emb_wrong_expanded.shape}")
except Exception as e:
    print(f"Wrong expand operation failed: {e}")

# The issue is that accuracy_emb should have shape [batch_size, d_model] = [1024, 512]
# But it actually has shape [2, 512], which means accuracy_target has wrong shape
print(f"\nExpected: accuracy_target shape should be [{batch_size}, 1] = [1024, 1]")
print(f"Actual: accuracy_target shape is [2, 1]")
print("This means the batch_size is actually 2, not 1024!")

# Let me test with the correct understanding
actual_seq_len = 1024
actual_batch_size = 2
accuracy_emb_correct = torch.randn(actual_batch_size, d_model)  # [2, 512]
print(f"\nCorrect scenario:")
print(f"seq_len={actual_seq_len}, batch_size={actual_batch_size}")
print(f"accuracy_emb shape: {accuracy_emb_correct.shape}")

try:
    accuracy_emb_correct_expanded = accuracy_emb_correct.unsqueeze(0).expand(actual_seq_len, -1, -1)
    print(f"Correct expanded shape: {accuracy_emb_correct_expanded.shape}")
    print("This should work!")
except Exception as e:
    print(f"Still failed: {e}")