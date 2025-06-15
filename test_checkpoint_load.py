import torch
import traceback

print("Testing model_epoch_0033.pt loading...")

try:
    print("Attempting to load checkpoint...")
    checkpoint = torch.load('model_epoch_0033.pt', map_location='cpu', weights_only=False)
    print("✓ Successfully loaded checkpoint!")
    print(f"Keys in checkpoint: {list(checkpoint.keys())[:10]}")
    
    if 'model_state_dict' in checkpoint:
        print(f"Model state dict keys: {len(checkpoint['model_state_dict'])}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"Train loss: {checkpoint['train_loss']}")
        
except Exception as e:
    print(f"❌ Failed to load checkpoint: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    # Try to get more info about the file
    import os
    if os.path.exists('model_epoch_0033.pt'):
        size = os.path.getsize('model_epoch_0033.pt')
        print(f"\nFile exists, size: {size} bytes ({size/1024/1024:.1f} MB)")
        
        # Try to read first few bytes
        try:
            with open('model_epoch_0033.pt', 'rb') as f:
                first_bytes = f.read(100)
                print(f"First 20 bytes (hex): {first_bytes[:20].hex()}")
                print(f"First 20 bytes (ascii): {repr(first_bytes[:20])}")
        except Exception as read_e:
            print(f"Could not read file bytes: {read_e}")
    else:
        print("File does not exist!")