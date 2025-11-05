"""Utility functions for PreGest project."""

import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union

from .config import LOG_FILE, SEED


def setup_logging(log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_file: Path to log file. If None, uses default from config.
        
    Returns:
        Configured logger instance.
    """
    if log_file is None:
        log_file = LOG_FILE
    
    # Create logger
    logger = logging.getLogger('pregest')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def set_random_seeds(seed: int = SEED) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get device (GPU if available, else CPU).
    
    Returns:
        torch.device instance.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger = setup_logging()
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger = setup_logging()
        logger.info("Using CPU")
    
    return device


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """Ensure parent directory of a file path exists.

    Args:
        path: File path whose parent directory should exist.

    Returns:
        Path object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string (e.g., "1h 23m 45s").
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Union[str, Path],
    device: torch.device
) -> int:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        checkpoint_path: Path to checkpoint file.
        device: Device to load to.
        
    Returns:
        Epoch number from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    
    logger = setup_logging()
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: Union[str, Path]
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        loss: Current loss.
        checkpoint_path: Path to save checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    logger = setup_logging()
    logger.info(f"Saved checkpoint to {checkpoint_path}")
