"""Configuration management for PreGest project."""

import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QUEST3_RAW_DIR = DATA_DIR / "quest3" / "raw"
QUEST3_PROCESSED_DIR = DATA_DIR / "quest3" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [QUEST3_RAW_DIR, QUEST3_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# QUEST 3 GESTURES - Direct training on Quest 3 data
QUEST3_GESTURES = {
    0: "flat_palm_stop",
    1: "grab",
    2: "pinch_select",
    3: "release",
    4: "swipe_down",
    5: "swipe_left",
    6: "swipe_right",
    7: "swipe_up"
}

NUM_QUEST3_CLASSES = len(QUEST3_GESTURES)

# Dataset selection
ACTIVE_DATASET = "quest3"  # Quest 3 only

# Data configuration
DATA_CONFIG = {
    'seq_length': 60,              # 60-frame temporal windows for Quest 3
    'image_size': (224, 224),      # Resized for ResNet18 input
    'train_ratio': 0.8,            # 80% training for Quest 3
    'val_ratio': 0.2,              # 20% validation for Quest 3
    'test_ratio': 0.0,             # Test from separate directory for Quest 3
    'lead_frames': 20,             # 20 frames before gesture onset
    'gesture_frames': 40,          # 40 frames after gesture onset
}

# Model configuration - Multi-modal Transformer
MODEL_CONFIG = {
    'backbone': 'resnet18',         # Changed from squeezenet for Quest 3
    'rgb_pretrained': True,         # Use ImageNet pretrained for RGB encoder
    'mask_pretrained': False,       # Train mask encoder from scratch
    'feature_dim': 512,
    'fusion_dim': 256,              # Reduced from 64 for Quest 3
    'hidden_dim': 256,              # Reduced from 128 for Quest 3
    'num_heads': 4,                 # Increased from 2 for Quest 3
    'num_layers': 2,                # Same as before
    'feedforward_dim': 512,         # Increased from 128 for Quest 3
    'dropout': 0.5,                 # Increased from 0.4 for Quest 3
    'max_seq_len': 60,              # Maximum sequence length
    'num_classes': NUM_QUEST3_CLASSES,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 2,                # Optimized for MPS memory (M4 GPU training)
    'learning_rate': 1e-4,          # Learning rate for Quest 3
    'weight_decay': 1e-4,           # L2 regularization for Quest 3
    'num_epochs': 50,               # Maximum training epochs for Quest 3
    'patience': 7,                  # Early stopping patience for Quest 3
    'gradient_clip_norm': 1.0,      # Gradient clipping
    'scheduler_patience': 5,
    'warmup_epochs': 3,             # Learning rate warmup
    'seed': 42,                     # Random seed
    'class_weights': None,          # No class weights for Quest 3 (balanced)
}

# Sequence Parameters
SEQUENCE_LENGTH = 30  # frames per window (reduced for Quest 3 short videos)
FRAME_SIZE = (224, 224)
NUM_CHANNELS = 4  # RGB (3) + mask (1)

# Augmentation configuration - Domain adaptation for office environments
AUGMENTATION_CONFIG = {
    'enable': True,
    'color_jitter': {
        'brightness': 0.4,          # ±40% brightness change
        'contrast': 0.4,            # ±40% contrast change
        'saturation': 0.4,          # ±40% saturation change
        'hue': 0.2,                 # ±20° hue shift
    },
    'geometric': {
        'rotation_range': (-8, 8),  # ±8° rotation (realistic camera tilt)
        'scale_range': (0.97, 1.03), # ±3% scaling (slight zoom)
        'translate_range': (-0.02, 0.02),  # ±2% translation
    },
    'noise': {
        'gaussian_std': 0.01,       # 1% Gaussian noise
    },
    'temporal': {
        'dropout_prob': 0.05,       # 5% frame dropout
        'speed_range': (0.95, 1.05), # ±5% speed variation
    },
}

# File paths
BEST_MODEL_PATH = MODELS_DIR / "quest3_transformer_best.pth"
FINAL_MODEL_PATH = MODELS_DIR / "quest3_transformer_final.pth"

# Logging
LOG_FILE = LOG_DIR / "training_quest3.log"
SEED = TRAIN_CONFIG['seed']

# Device - Auto-detect optimal device for training
import torch

def get_optimal_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Check for MPS (Apple Silicon GPU)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_optimal_device()
print(f"Using device: {DEVICE}")
