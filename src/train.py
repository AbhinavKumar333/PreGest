"""Training pipeline for gesture recognition transformer."""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging

from .config import (
    TRAIN_CONFIG, BEST_MODEL_PATH, FINAL_MODEL_PATH, DEVICE, NUM_CLASSES,
    PRETRAINING_MODE, JESTER_CLASSES, QUEST3_TARGET_GESTURES, SELECTED_GESTURE_IDS
)
from .model import create_model
from .utils import setup_logging, set_random_seeds, format_time, count_parameters
from .dataset import preprocess_jester_dataset, load_preprocessed_data


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip_norm: float = 1.0
) -> Tuple[float, float]:
    """Train model for one epoch.
    
    Args:
        model: PyTorch model.
        train_loader: Training dataloader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to run on.
        gradient_clip_norm: Gradient clipping norm.
        
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model on validation set.
    
    Args:
        model: PyTorch model.
        val_loader: Validation dataloader.
        criterion: Loss function.
        device: Device to run on.
        
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits = model(sequences)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_transformer(
    train_loader: Optional[torch.utils.data.DataLoader] = None,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Complete training pipeline with flexible pretraining options.
    
    Supports:
    - Pretraining on 8 Quest3-mapped gestures
    - Pretraining on all 27 Jester gestures
    - Pretraining on custom gesture subset
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        
    Returns:
        Dictionary with training results
    """
    logger = setup_logging()
    
    # Use defaults from config if not provided
    if num_epochs is None:
        num_epochs = TRAIN_CONFIG['num_epochs']
    if learning_rate is None:
        learning_rate = TRAIN_CONFIG['learning_rate']
    if weight_decay is None:
        weight_decay = TRAIN_CONFIG['weight_decay']
    if device is None:
        device = DEVICE
    
    # Set random seeds for reproducibility
    set_random_seeds(TRAIN_CONFIG['seed'])
    
    logger.info("="*70)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*70)
    logger.info(f"\nPretraining Mode: {PRETRAINING_MODE}")
    logger.info(f"Number of Classes: {NUM_CLASSES}")
    
    # Log pretraining configuration
    if PRETRAINING_MODE == "quest3":
        logger.info("Pretraining on 8 Quest3-mapped gestures:")
        for gesture_id in SELECTED_GESTURE_IDS:
            gesture_name = JESTER_CLASSES[gesture_id]
            logger.info(f"  {gesture_id}: {gesture_name}")
    elif PRETRAINING_MODE == "jester":
        logger.info(f"Pretraining on all {NUM_CLASSES} Jester gestures")
    else:
        logger.info(f"Pretraining on custom {NUM_CLASSES} gestures: {SELECTED_GESTURE_IDS}")
    
    # Load or preprocess data if not provided
    if train_loader is None or val_loader is None:
        logger.info("\nLoading preprocessed data...")
        from .dataset import JESTER_PROCESSED_DIR
        if not (JESTER_PROCESSED_DIR / 'train.pt').exists():
            logger.info("Preprocessed data not found. Running preprocessing...")
            train_loader, val_loader, test_loader = preprocess_jester_dataset()
        else:
            train_loader, val_loader, test_loader = load_preprocessed_data()
    
    # Create model
    logger.info("\nCreating model...")
    model = create_model(
        num_classes=NUM_CLASSES,
        device=device
    )
    num_params = count_parameters(model)
    logger.info(f"Model created with {num_params:,} parameters")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    logger.info("\n" + "="*70)
    logger.info("TRAINING")
    logger.info("="*70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip_norm=TRAIN_CONFIG['gradient_clip_norm']
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        logger.info(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"Time: {format_time(epoch_time)}"
        )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logger.info(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")
    logger.info(f"Total training time: {format_time(total_time)}")
    logger.info(f"Pretraining mode: {PRETRAINING_MODE}")
    logger.info(f"Number of classes: {NUM_CLASSES}")
    
    # Load best model
    logger.info(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    
    # Save final model
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    logger.info(f"Final model saved to {FINAL_MODEL_PATH}")
    
    # Test on test set if provided
    if test_loader is not None:
        logger.info("\nEvaluating on test set...")
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        training_history['test_loss'] = test_loss
        training_history['test_acc'] = test_acc
    
    return training_history
