"""Configuration management for PreGest project."""

import torch
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
JESTER_RAW_DIR = DATA_DIR / "jester" / "raw"
JESTER_PROCESSED_DIR = DATA_DIR / "jester" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [JESTER_RAW_DIR, JESTER_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# Dataset parameters - 25 Jester gesture classes
JESTER_CLASSES = [
    "Doing other things",
    "Drumming Fingers",
    "No gesture",
    "Pulling Hand In",
    "Pulling Two Fingers In",
    "Pushing Hand Away",
    "Pushing Two Fingers Away",
    "Rolling Hand Backward",
    "Rolling Hand Forward",
    "Shaking Hand",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Up",
    "Stop Sign",
    "Swiping Down",
    "Swiping Left",
    "Swiping Right",
    "Swiping Up",
    "Thumbs Down",
    "Thumbs Up",
    "Turning Hand Counterclockwise",
    "Turning Hand Clockwise",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers"
]

# Verify we have exactly 25 classes
assert len(JESTER_CLASSES) == 25, f"Expected 25 classes, got {len(JESTER_CLASSES)}"
NUM_CLASSES = len(JESTER_CLASSES)


# Data configuration
DATA_CONFIG = {
    'seq_length': 60,
    'feature_dim': 63,  # 21 joints * 3 coordinates
    'window_stride': 5,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
}

# Model configuration
MODEL_CONFIG = {
    'input_dim': 63,
    'hidden_dim': 256,
    'num_heads': 4,
    'num_layers': 4,
    'num_classes': NUM_CLASSES,
    'dropout': 0.1,
    'max_seq_len': 60,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_epochs': 30,
    'patience': 10,
    'gradient_clip_norm': 1.0,
    'seed': 42,
}

# File paths
TRAIN_DATA_PATH = JESTER_PROCESSED_DIR / "train.pt"
VAL_DATA_PATH = JESTER_PROCESSED_DIR / "val.pt"
TEST_DATA_PATH = JESTER_PROCESSED_DIR / "test.pt"

BEST_MODEL_PATH = MODELS_DIR / "gesture_transformer_best.pth"
FINAL_MODEL_PATH = MODELS_DIR / "gesture_transformer_final.pth"

# Logging
LOG_FILE = LOG_DIR / "training.log"
SEED = TRAIN_CONFIG['seed']

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
