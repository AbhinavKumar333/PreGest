"""Configuration management for PreGest project."""

import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
JESTER_RAW_DIR = DATA_DIR / "jester" / "raw" / "20bn-jester-dataset"
JESTER_PROCESSED_DIR = DATA_DIR / "jester" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [JESTER_RAW_DIR, JESTER_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# JESTER DATASET - All 27 gesture classes
JESTER_CLASSES = [
    "Doing other things",           # 0
    "Drumming Fingers",             # 1
    "No gesture",                   # 2
    "Pulling Hand In",              # 3
    "Pulling Two Fingers In",       # 4
    "Pushing Hand Away",            # 5
    "Pushing Two Fingers Away",     # 6
    "Rolling Hand Backward",        # 7
    "Rolling Hand Forward",         # 8
    "Shaking Hand",                 # 9
    "Sliding Two Fingers Down",     # 10
    "Sliding Two Fingers Left",     # 11
    "Sliding Two Fingers Right",    # 12
    "Sliding Two Fingers Up",       # 13
    "Stop Sign",                    # 14
    "Swiping Down",                 # 15
    "Swiping Left",                 # 16
    "Swiping Right",                # 17
    "Swiping Up",                   # 18
    "Thumb Down",                   # 19
    "Thumb Up",                     # 20
    "Turning Hand Clockwise",       # 21
    "Turning Hand Counterclockwise",# 22
    "Zooming In With Full Hand",    # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",   # 25
    "Zooming Out With Two Fingers"  # 26
]

NUM_JESTER_CLASSES = len(JESTER_CLASSES)
assert NUM_JESTER_CLASSES == 27, f"Expected 27 Jester classes, got {NUM_JESTER_CLASSES}"


# QUEST 3 MAPPING - Map 8 relevant Jester gestures to Quest 3 gestures
QUEST3_TARGET_GESTURES = {
    0: "Pinch Select",      # From Jester 4: Pulling Two Fingers In
    1: "Grab",              # From Jester 3: Pulling Hand In
    2: "Release",           # From Jester 5: Pushing Hand Away
    3: "Flat Palm Stop",    # From Jester 14: Stop Sign
    4: "Swipe Left",        # From Jester 16: Swiping Left
    5: "Swipe Right",       # From Jester 17: Swiping Right
    6: "Swipe Down",        # From Jester 15: Swiping Down
    7: "Swipe Up",          # From Jester 18: Swiping Up
}

# Mapping: Jester gesture ID → Quest 3 gesture ID
JESTER_TO_QUEST3_MAPPING = {
    4: 0,   # Pulling Two Fingers In → Pinch Select
    3: 1,   # Pulling Hand In → Grab
    5: 2,   # Pushing Hand Away → Release
    14: 3,  # Stop Sign → Flat Palm Stop
    16: 4,  # Swiping Left → Swipe Left
    17: 5,  # Swiping Right → Swipe Right
    15: 6,  # Swiping Down → Swipe Down
    18: 7,  # Swiping Up → Swipe Up
}

NUM_QUEST3_CLASSES = len(QUEST3_TARGET_GESTURES)


# PRETRAINING MODE CONFIGURATION - Choose which gestures to pretrain on
# Options:
# - "quest3": Pretrain on only 8 Quest3-mapped gestures
# - "jester": Pretrain on all 27 Jester gestures
# - "custom": Pretrain on custom subset (edit CUSTOM_GESTURE_IDS below)
PRETRAINING_MODE = "quest3"  # Change to "jester" or "custom" as needed

# Custom gesture IDs (if using PRETRAINING_MODE = "custom")
# Example: [4, 3, 5, 14, 16, 17, 15, 18] for 8 Quest3 gestures
# Example: [1, 4, 5, 9, 16, 17, 15, 18, 11, 12, 13, 10] for 12 gestures
CUSTOM_GESTURE_IDS = [4, 3, 5, 14, 16, 17, 15, 18]  # Quest3 by default

# Determine number of classes based on pretraining mode
if PRETRAINING_MODE == "quest3":
    NUM_CLASSES = NUM_QUEST3_CLASSES
    SELECTED_GESTURE_IDS = list(JESTER_TO_QUEST3_MAPPING.keys())
elif PRETRAINING_MODE == "jester":
    NUM_CLASSES = NUM_JESTER_CLASSES
    SELECTED_GESTURE_IDS = list(range(NUM_JESTER_CLASSES))
elif PRETRAINING_MODE == "custom":
    SELECTED_GESTURE_IDS = CUSTOM_GESTURE_IDS
    NUM_CLASSES = len(CUSTOM_GESTURE_IDS)
else:
    raise ValueError(f"Unknown PRETRAINING_MODE: {PRETRAINING_MODE}")

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

BEST_MODEL_PATH = MODELS_DIR / f"gesture_transformer_best_{PRETRAINING_MODE}.pth"
FINAL_MODEL_PATH = MODELS_DIR / f"gesture_transformer_final_{PRETRAINING_MODE}.pth"

# Logging
LOG_FILE = LOG_DIR / f"training_{PRETRAINING_MODE}.log"
SEED = TRAIN_CONFIG['seed']

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
