# Osu! AI Replay Maker - Development Plan

## Project Overview
Create an AI system that can learn from osu! replays and generate human-like replays with controllable accuracy parameters (300s, 100s, 50s, misses).

## Dataset Analysis
- **Size**: 382,000 replays (excellent for training)
- **Format**: CSV index with replay metadata + .osr replay files + .osu beatmap files
- **Key Features Available**:
  - Performance data: accuracy, score, hit counts (300s, 100s, 50s, misses)
  - Beatmap data: BPM, HP, OD, AR, CS, hit objects
  - Player data: mods, date, player name
  - Replay files: actual cursor movement and timing data

## Architecture Design

### Phase 1: Data Processing Pipeline
1. **Replay Parser**: Parse .osr files to extract cursor movements and key presses
2. **Beatmap Parser**: Parse .osu files to extract hit object timing and positions
3. **Feature Engineering**: Create training features from replay + beatmap data
4. **Data Preprocessing**: Normalize, filter, and prepare training data

### Phase 2: Model Architecture
**Primary Model: Transformer-based Sequence Model**
- **Input**: Beatmap features + accuracy parameters + temporal context
- **Output**: Cursor positions and key press timings
- **Architecture**: 
  - Encoder: Beatmap features + accuracy conditioning
  - Decoder: Autoregressive cursor movement generation
  - Attention: Multi-head attention for temporal dependencies

**Alternative Models to Experiment**:
- LSTM/GRU for sequential modeling
- VAE for latent space replay generation
- Diffusion models for smooth cursor trajectories

### Phase 3: Training Strategy
1. **Data Split**: 70% train, 20% validation, 10% test
2. **Curriculum Learning**: 
   - Start with simple beatmaps (low star rating)
   - Gradually increase complexity
3. **Multi-task Learning**:
   - Primary: Cursor position prediction
   - Auxiliary: Hit timing prediction, accuracy prediction
4. **Conditional Training**: Use accuracy parameters as conditioning inputs

### Phase 4: Evaluation & Refinement
1. **Metrics**:
   - Accuracy match (target vs generated)
   - Cursor smoothness (human-like movement)
   - Timing accuracy (hit windows)
   - Visual similarity to human replays
2. **Validation**: Generate replays and compare with human performance

## Implementation Plan

### Week 1: Foundation
- [x] Project setup and data exploration
- [ ] Replay file parser (.osr format)
- [ ] Beatmap file parser (.osu format)
- [ ] Basic data loading pipeline

### Week 2: Data Processing
- [ ] Feature extraction from replays
- [ ] Beatmap feature engineering
- [ ] Data normalization and preprocessing
- [ ] Training data generation

### Week 3: Model Development
- [ ] Transformer model implementation
- [ ] Training loop with accuracy conditioning
- [ ] Basic evaluation metrics
- [ ] Initial training experiments

### Week 4: Training & Optimization
- [ ] Hyperparameter tuning
- [ ] Advanced training strategies
- [ ] Model evaluation and validation
- [ ] Replay generation pipeline

## Technical Specifications

### Hardware Requirements
- **GPU**: RTX 3050 Mobile (4GB VRAM) - sufficient with optimization
- **Memory**: 16GB+ RAM recommended
- **Storage**: SSD for fast data loading

### Software Stack
- **Framework**: PyTorch with CUDA support
- **Data**: Pandas, NumPy for data processing
- **Visualization**: Matplotlib, Plotly for analysis
- **Utilities**: tqdm, wandb for training monitoring

### Memory Optimization for 4GB GPU
- Mixed precision training (FP16)
- Gradient checkpointing
- Batch size: 16-32 (optimized for VRAM)
- Model sharding if needed

## Expected Outcomes

### Success Metrics
1. **Accuracy Control**: Generate replays within ±0.5% of target accuracy
2. **Human-like Movement**: Smooth, realistic cursor trajectories
3. **Timing Precision**: Hit timing within human reaction windows
4. **Scalability**: Work across different beatmap difficulties

### Deliverables
1. Trained AI model capable of generating human-like replays
2. Replay generation interface with accuracy parameters
3. Evaluation tools and metrics
4. Documentation and usage examples

## Risk Mitigation

### Potential Challenges
1. **Overfitting**: Large dataset helps, use regularization
2. **VRAM Limitations**: Optimize batch size and use mixed precision
3. **Training Time**: Use efficient architectures and early stopping
4. **Data Quality**: Filter and validate replay data

### Contingency Plans
- If Transformer is too memory-intensive: Fall back to LSTM
- If training is too slow: Use smaller model or reduce sequence length
- If accuracy control is poor: Add more conditioning mechanisms

## Next Steps
1. Start with replay file parsing
2. Implement basic data loading
3. Create initial model prototype
4. Begin training experiments

This plan provides a comprehensive roadmap for developing a sophisticated osu! replay AI that can generate human-like gameplay with controllable accuracy parameters.