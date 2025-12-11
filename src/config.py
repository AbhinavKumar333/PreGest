"""Configuration management for PreGest project"""

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
ACTIVE_DATASET = "quest3"  

# Data configuration
DATA_CONFIG = {
    'seq_length': 30,  
    'image_size': (224, 224),  
    'train_ratio': 0.8,  
    'val_ratio': 0.2,  
    'test_ratio': 0.0,  
    'lead_frames': 20,  
    'gesture_frames': 40,  
}

# Model configuration - Multi-modal Transformer
MODEL_CONFIG = {
    'backbone': 'resnet18',  
    'rgb_pretrained': True,  
    'mask_pretrained': False,  
    'feature_dim': 512,
    'fusion_dim': 256,  
    'hidden_dim': 256,  
    'num_heads': 4,  
    'num_layers': 2,  
    'feedforward_dim': 512,  
    'dropout': 0.5,  
    'max_seq_len': 30,  
    'num_classes': NUM_QUEST3_CLASSES,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 2,  
    'learning_rate': 1e-4,  
    'weight_decay': 1e-4,  
    'num_epochs': 50,  
    'patience': 7,  
    'gradient_clip_norm': 1.0,  
    'scheduler_patience': 5,
    'warmup_epochs': 3,  
    'seed': 42,  
    'class_weights': None,  
}

# Sequence Parameters
SEQUENCE_LENGTH = 30  
FRAME_SIZE = (224, 224)
NUM_CHANNELS = 4  

# Augmentation configuration - Domain adaptation for office environments
AUGMENTATION_CONFIG = {
    'enable': True,
    'color_jitter': {
        'brightness': 0.4,  
        'contrast': 0.4,  
        'saturation': 0.4,  
        'hue': 0.2,  
    },
    'geometric': {
        'rotation_range': (-8, 8),  
        'scale_range': (0.97, 1.03),  
        'translate_range': (-0.02, 0.02),  
    },
    'noise': {
        'gaussian_std': 0.01,  
    },
    'temporal': {
        'dropout_prob': 0.05,  
        'speed_range': (0.95, 1.05),  
    },
}

# File paths
BEST_MODEL_PATH = MODELS_DIR / "quest3_transformer_best.pth"
FINAL_MODEL_PATH = MODELS_DIR / "quest3_transformer_final.pth"

# Logging
LOG_FILE = LOG_DIR / "training_quest3.log"
SEED = TRAIN_CONFIG['seed']

# Device - Auto-detect optimal device for training
def get_optimal_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_optimal_device()
print(f"Using device: {DEVICE}")
