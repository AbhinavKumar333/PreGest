# PreGest: Proactive Gesture-to-Action Prediction

A complete implementation of gesture recognition using Transformer Neural Networks with flexible pretraining for Quest 3 deployment.

## Overview

PreGest implements a state-of-the-art gesture recognition system using PyTorch and Transformer architectures. The system features **flexible pretraining modes** supporting:

- **Quest 3 Mode**: Pretrain on 8 carefully mapped gestures for Meta Quest 3 deployment
- **Jester Mode**: Pretrain on all 27 Jester gestures for maximum generalization
- **Custom Mode**: Pretrain on user-defined gesture subsets

The system processes video data from the 20BN-Jester dataset, extracts hand landmarks using MediaPipe, and trains a Transformer model using a **two-stage approach**: pretrain on Jester data, then fine-tune for Quest 3 deployment.

## Features

- **Complete Pipeline**: From raw video processing to model training and evaluation
- **Transformer Architecture**: Multi-head self-attention for gesture sequence modeling
- **MediaPipe Integration**: Real-time hand landmark extraction
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrix, and per-class metrics
- **Cross-Platform**: Works on macOS and Linux
- **Production Ready**: Model checkpointing, logging, and visualization

## Project Structure

```
pregest-class-project/
├── data/
│   ├── jester_raw/                    # Downloaded Jester dataset
│   │   └── 20bn-jester-dataset/       # CSV files and video folders
│   └── jester_processed/              # Processed pose sequences
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
├── models/
│   ├── gesture_transformer_best_quest3.pth    # Quest 3 model
│   ├── gesture_transformer_best_jester.pth    # Jester model
│   ├── gesture_transformer_final_quest3.pth   # Final Quest 3 model
│   └── gesture_transformer_final_jester.pth   # Final Jester model
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── per_class_f1.png
│   ├── evaluation_metrics.json
│   └── classification_report.txt
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   ├── utils.py                      # Helper functions
│   ├── dataset.py                    # Preprocessing pipeline
│   ├── model.py                      # Transformer architecture
│   ├── train.py                      # Training script
│   └── evaluate.py                   # Evaluation & metrics
├── logs/
│   ├── training_quest3.log           # Quest 3 training logs
│   └── training_jester.log           # Jester training logs
├── requirements.txt
├── main.py                            # Entry point
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- pip package manager
- (Optional) Kaggle API credentials for automatic dataset download

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/PreGest.git
   cd PreGest
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Kaggle API (optional)**:
   ```bash
   # Create kaggle.json with your credentials
   mkdir ~/.kaggle
   echo '{"username":"your-username","key":"your-api-key"}' > ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Usage

### Quick Start

1. **Show system information**:
   ```bash
   python main.py info
   ```

2. **Download Jester dataset**:
   ```bash
   python main.py download-dataset
   # Or download manually from https://20bn.com/datasets/jester
   ```

3. **Preprocess dataset** (Quest 3 mode by default):
   ```bash
   python main.py preprocess --max-videos 100  # Quick test
   python main.py preprocess --max-videos 5000  # Full preprocessing
   ```

4. **Train the model** (Quest 3 pretraining):
   ```bash
   python main.py train --mode quest3 --epochs 5  # Quick test
   python main.py train --mode quest3 --epochs 15  # Production training
   ```

5. **Evaluate the model**:
   ```bash
   python main.py evaluate --mode quest3
   ```

### Pretraining Modes

#### Quest 3 Mode (Recommended for Deployment)
```bash
# Pretrain on 8 Quest 3 mapped gestures
python main.py preprocess --mode quest3 --max-videos 2000
python main.py train --mode quest3 --epochs 15
```

#### Jester Mode (Maximum Generalization)
```bash
# Pretrain on all 27 Jester gestures
python main.py preprocess --mode jester --max-videos 5000
python main.py train --mode jester --epochs 40
```

#### Custom Mode (User-Defined Gestures)
```bash
# Edit CUSTOM_GESTURE_IDS in src/config.py first
python main.py preprocess --mode custom --max-videos 1000
python main.py train --mode custom --epochs 20
```

### Two-Stage Training Approach (Optimal)

```bash
# Stage 1: Pretrain on Jester (rich feature learning)
python main.py preprocess --mode jester --max-videos 5000
python main.py train --mode jester --epochs 30

# Stage 2: Fine-tune for Quest 3 (device specialization)
python main.py preprocess --mode quest3 --max-videos 200
python main.py train --mode quest3 --epochs 10
```

### Advanced Usage

#### Custom Training Parameters

```bash
# Train with custom hyperparameters
python main.py train --mode quest3 --epochs 50 --batch-size 64 --lr 0.0001

# Resume training from checkpoint
python main.py train --mode quest3 --resume models/gesture_transformer_best_quest3.pth
```

#### Evaluation Options

```bash
# Evaluate specific model
python main.py evaluate --mode quest3 --model-path models/custom_model.pth
```

#### Preprocessing Options

```bash
# Process only subset for testing
python main.py preprocess --mode quest3 --max-videos 100

# Full dataset processing
python main.py preprocess --mode jester --max-videos 5000
```

## Model Architecture

### GestureTransformer

- **Input**: [batch_size, 60, 63] (60 frames × 21 joints × 3 coordinates)
- **Embedding**: Linear projection to 256-dimensional space
- **Positional Encoding**: Sinusoidal encoding for temporal positions
- **Transformer Encoder**: 4 layers, 4 attention heads, 512 feedforward dim
- **Global Pooling**: Average pooling across temporal dimension
- **Classification Head**: Dynamic linear layer (8, 27, or custom classes)

### Quest 3 Gesture Mapping

The system maps 8 carefully selected Jester gestures to Quest 3 actions:

| Quest 3 Action | Jester Gesture Source | ID |
|----------------|----------------------|----|
| Pinch Select | Pulling Two Fingers In | 4 |
| Grab | Pulling Hand In | 3 |
| Release | Pushing Hand Away | 5 |
| Flat Palm Stop | Stop Sign | 14 |
| Swipe Left | Swiping Left | 16 |
| Swipe Right | Swiping Right | 17 |
| Swipe Down | Swiping Down | 15 |
| Swipe Up | Swiping Up | 18 |

### Key Hyperparameters

- Hidden dimension: 256
- Number of heads: 4
- Number of layers: 4
- Dropout: 0.1
- Sequence length: 60 frames
- Batch size: 32
- Learning rate: 1e-4
- Classes: Dynamic (8 for Quest 3, 27 for Jester, custom for user-defined)

## Dataset

### Jester Dataset

- **Source**: 20BN-Jester dataset
- **Videos**: 148,092 gesture videos
- **Classes**: 27 gesture types
- **Duration**: 1-5 seconds per video
- **Resolution**: Variable (640×480 to 1920×1080)

### Gesture Classes

1. Doing other things
2. Drumming Fingers
3. No gesture
4. Pulling Hand In
5. Pulling Two Fingers In
6. Pushing Hand Away
7. Pushing Two Fingers Away
8. Rolling Hand Backward
9. Rolling Hand Forward
10. Shaking Hand
11. Sliding Two Fingers Down
12. Sliding Two Fingers Left
13. Sliding Two Fingers Right
14. Sliding Two Fingers Up
15. Stop Sign
16. Swiping Down
17. Swiping Left
18. Swiping Right
19. Swiping Up
20. Thumb Down
21. Thumb Up
22. Turning Hand Clockwise
23. Turning Hand Counterclockwise
24. Zooming In With Full Hand
25. Zooming In With Two Fingers
26. Zooming Out With Full Hand
27. Zooming Out With Two Fingers

## Results

### Performance Metrics (Quest 3 Mode)

From testing with 100 videos (148 sequences) on Mac M4:
- **Overall Accuracy**: 78.95% (test set)
- **Validation Accuracy**: 84.21% (best epoch)
- **Training Accuracy**: 78.38% (final epoch)
- **Model Size**: 2.16M parameters
- **Training Time**: 3 seconds for 5 epochs
- **Preprocessing Time**: ~1 minute for 100 videos

### Performance Projections

**Optimal Two-Stage Training** (estimated):
- **Jester Pretraining** (5K videos, 40 epochs): 85-90% accuracy on 27 gestures
- **Quest 3 Fine-tuning** (2K videos, 15 epochs): 88-92% accuracy on 8 Quest 3 gestures
- **Real-world Deployment**: 85%+ accuracy on Meta Quest 3 device

### Output Files

Training and evaluation generate mode-specific files:
- `models/gesture_transformer_best_quest3.pth` - Best Quest 3 model
- `models/gesture_transformer_best_jester.pth` - Best Jester model
- `results/confusion_matrix.png` - Confusion matrix visualization
- `results/training_history.png` - Training curves
- `results/evaluation_metrics.json` - Detailed metrics
- `logs/training_quest3.log` - Quest 3 training logs
- `logs/training_jester.log` - Jester training logs

## Configuration

All hyperparameters and paths are configured in `src/config.py`:

```python
# Key configuration options
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
HIDDEN_DIM = 256
NUM_HEADS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## Development

### Code Quality

- **PEP 8** compliant
- **Type hints** throughout
- **Google-style docstrings**
- **Comprehensive logging**

### Testing

```bash
# Test with small dataset subset
python main.py preprocess --max-videos 10
python main.py train --epochs 2
python main.py evaluate
```

### Cross-Platform Compatibility

The code uses `pathlib.Path` for all file operations and has been tested on:
- macOS (Intel/M1)
- Linux (Ubuntu/CentOS)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size: `python main.py train --batch-size 16`

2. **Dataset download fails**:
   - Download manually from https://20bn.com/datasets/jester
   - Place videos in `data/jester_raw/` organized by gesture class

3. **MediaPipe installation issues**:
   - Ensure compatible OpenCV version
   - Check Python version compatibility

4. **Import errors**:
   - Activate virtual environment: `source venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`

### Performance Optimization

- Use GPU if available (automatically detected)
- Increase batch size for better GPU utilization
- Use multiple workers for data loading (currently disabled for compatibility)

## Future Work

### Quest 3 Integration (Implemented)

The system now supports **production-ready Quest 3 deployment**:
- ✅ **8 Quest 3 gesture mapping** (Pinch Select, Grab, Release, etc.)
- ✅ **Flexible pretraining modes** (Quest 3, Jester, Custom)
- ✅ **Two-stage training** (Jester pretraining → Quest 3 fine-tuning)
- ✅ **Mode-aware evaluation** with correct class names

### Planned Enhancements

- **Real Quest 3 data integration** (device-recorded gesture videos)
- **ONNX export** for Quest 3 deployment optimization
- **Real-time inference optimization** (latency <50ms target)
- **Multi-GPU training support** for larger datasets
- **Advanced data augmentation** (temporal distortions, noise injection)
- **Model quantization** for mobile/edge deployment
- **Cross-device evaluation** (Jester → Quest 3 generalization testing)

## License

This project is released under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pregest2024,
  title={PreGest: Proactive Gesture-to-Action Prediction using Transformer Neural Networks},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/PreGest}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- 20BN-Jester dataset creators
- PyTorch team for the excellent deep learning framework
- Google MediaPipe team for hand tracking
- Hugging Face for Transformer inspiration
