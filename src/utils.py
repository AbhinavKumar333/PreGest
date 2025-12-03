"""Utility functions for PreGest project."""

import logging
import torch
from pathlib import Path
from typing import Optional, Union

from .config import LOG_DIR

def setup_logging(log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """Set up logging configuration for Quest 3 gesture recognition.
    
    Args:
        log_file: Path to log file. If None, uses default from config.
    
    Returns:
        Configured logger instance.
    """
    if log_file is None:
        log_file = LOG_DIR / "training.log"
    
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

def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in Quest 3 model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """Ensure directory path exists.
    
    Args:
        path: Path to create.
    
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

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
