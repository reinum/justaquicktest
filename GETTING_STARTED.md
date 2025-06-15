# 🚀 Getting Started with osu! AI Replay Maker

This guide will help you set up and start training your own osu! AI replay generation model.

## 📋 Prerequisites

Before you begin, make sure you have:

- **Python 3.8+** installed
- **CUDA-compatible GPU** (4GB+ VRAM recommended)
- **16GB+ RAM** for data processing
- **SSD storage** for optimal performance
- **osu! replay files (.osr)** and **beatmap files (.osu)**

## 🛠️ Installation

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Test if everything imports correctly
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Dataset Preparation

### 1. Organize Your Data

Create the following directory structure:

```
dataset/
├── beatmaps/          # Place your .osu files here
│   ├── map1.osu
│   ├── map2.osu
│   └── ...
├── replays/           # Place your .osr files here
│   ├── replay1.osr
│   ├── replay2.osr
│   └── ...
└── index.csv          # Will be generated automatically
```

### 2. Prepare the Dataset

Run the data preparation script:

```bash
# Scan and prepare your dataset
python prepare_data.py

# Or force rebuild if you've added new files
python prepare_data.py --force-rebuild

# Validate existing dataset
python prepare_data.py --validate-only
```

This script will:
- ✅ Scan all .osu and .osr files
- ✅ Extract metadata and validate files
- ✅ Create an index.csv file matching replays to beatmaps
- ✅ Convert replays to numpy format for faster loading
- ✅ Validate the dataset structure

### 3. Dataset Requirements

For good results, you should have:
- **Minimum**: 1,000+ replay-beatmap pairs
- **Recommended**: 10,000+ pairs
- **Optimal**: 100,000+ pairs

The model works best with:
- Diverse difficulty levels (1★ to 8★)
- Various accuracy levels (80% to 100%)
- Different play styles and players

## ⚙️ Configuration

### 1. Review Default Configuration

The default configuration is in `config/default.yaml`. Key settings:

```yaml
# Model size (adjust based on your GPU)
model:
  d_model: 512          # Model dimension
  n_heads: 8            # Attention heads
  n_layers: 6           # Transformer layers

# Training settings
training:
  batch_size: 8         # Reduced for 4GB VRAM
  learning_rate: 1e-4   # Learning rate
  max_epochs: 50        # Training epochs
  use_mixed_precision: true  # Memory optimization

# Data settings
data:
  sequence_length: 1024 # Input sequence length
  min_accuracy: 0.8     # Filter low accuracy replays
  max_star_rating: 8.0  # Filter very hard maps
```

### 2. Customize for Your Hardware

**For 4GB VRAM:**
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8
```

**For 8GB+ VRAM:**
```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 2
```

**For CPU-only training:**
```yaml
training:
  use_mixed_precision: false
  batch_size: 2
```

## 🏃 Training

### 1. Start Training

```bash
# Start training with default config
python train.py

# Use custom config
python train.py --config my_config.yaml

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pt

# Enable debug logging
python train.py --debug
```

### 2. Monitor Training

During training, you'll see:

```
🎮 osu! AI Replay Maker - Training
========================================
📋 Loading config from: config/default.yaml
🔍 Checking dataset...
🖥️  Using device: cuda
   GPU: NVIDIA GeForce RTX 3050
   VRAM: 4.0 GB
🧠 Creating model...
   Total parameters: 25,165,824
   Trainable parameters: 25,165,824
📊 Loading datasets...
   Training samples: 8,500
   Validation samples: 2,125
🏃 Initializing trainer...
🚀 Starting training...
   Epochs: 50
   Batch size: 8
   Learning rate: 0.0001
   Mixed precision: True

Epoch 1/50:
  Train Loss: 2.456 | Val Loss: 2.123 | Accuracy: 0.234
  Time: 45.2s | ETA: 37m 30s
```

### 3. Training Tips

- **Start small**: Begin with a subset of your data to test
- **Monitor GPU usage**: Use `nvidia-smi` to check VRAM usage
- **Adjust batch size**: If you get OOM errors, reduce batch_size
- **Use curriculum learning**: The model starts with easier maps
- **Be patient**: Good results typically need 20-50 epochs

## 📊 Monitoring Progress

### 1. Checkpoints

Checkpoints are saved in `checkpoints/`:
- `best_model.pt` - Best validation loss
- `latest_model.pt` - Most recent checkpoint
- `checkpoint_epoch_X.pt` - Epoch-specific checkpoints

### 2. Logs

Training logs are saved to:
- Console output
- `training.log` file
- TensorBoard logs (if enabled)
- Weights & Biases (if configured)

### 3. Validation Metrics

Key metrics to watch:
- **Loss**: Should decrease over time
- **Accuracy**: How well the model predicts hit timing
- **Cursor Error**: Average pixel distance from target
- **Key Accuracy**: Correct key press prediction

## 🎮 Generation (After Training)

Once training is complete, you can generate replays:

```python
from src.generation.generator import ReplayGenerator

# Load trained model
generator = ReplayGenerator.from_checkpoint('checkpoints/best_model.pt')

# Generate replay
replay = generator.generate(
    beatmap_path='dataset/beatmaps/example.osu',
    target_accuracy=0.95,
    target_300s=0.85,
    target_100s=0.10,
    target_50s=0.04,
    target_misses=0.01
)

# Export to .osr format
replay.export('generated_replay.osr')
```

## 🔧 Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps`
- Enable `use_mixed_precision`

**"No module named 'src'"**
- Make sure you're running from the project root
- Check that `src/__init__.py` exists

**"Dataset validation failed"**
- Run `python prepare_data.py --validate-only`
- Check that .osu and .osr files are valid
- Ensure index.csv was created correctly

**"BeatmapParser not available"**
- Check that `osuparse/OsuParsers.dll` exists
- Ensure you're on Windows (required for C# parser)

### Performance Issues

**Slow training:**
- Use SSD storage for dataset
- Increase `num_workers` in data config
- Enable `pin_memory` and `prefetch_factor`

**High memory usage:**
- Reduce `sequence_length`
- Lower `max_replays_in_memory`
- Disable `cache_preprocessed`

## 📈 Expected Results

### Training Timeline

- **Epochs 1-10**: Model learns basic patterns
- **Epochs 10-30**: Accuracy improves significantly
- **Epochs 30-50**: Fine-tuning and convergence

### Performance Targets

- **Training Loss**: Should reach < 1.0
- **Validation Accuracy**: Target > 0.8
- **Cursor Error**: Target < 20 pixels
- **Generation Speed**: Real-time (1x playback)

## 🎯 Next Steps

After successful training:

1. **Evaluate your model** using the evaluation tools
2. **Generate test replays** on various beatmaps
3. **Fine-tune hyperparameters** for better results
4. **Experiment with different architectures**
5. **Collect more data** for improved performance

## 💡 Tips for Better Results

1. **Quality over quantity**: Clean, high-accuracy replays work better
2. **Diverse data**: Include various difficulties and play styles
3. **Balanced dataset**: Ensure good distribution of accuracy levels
4. **Regular validation**: Monitor overfitting with validation metrics
5. **Experiment**: Try different model sizes and hyperparameters

## 🆘 Getting Help

If you encounter issues:

1. Check this guide and the main README
2. Look at the logs for error messages
3. Try the troubleshooting section
4. Create an issue on GitHub with:
   - Your system specs
   - Error messages
   - Steps to reproduce

---

**Happy training! 🎮✨**