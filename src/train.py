"""Training pipeline for Quest 3 gesture recognition"""

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import math

from .config import (
    TRAIN_CONFIG, BEST_MODEL_PATH, FINAL_MODEL_PATH, DEVICE, MODEL_CONFIG,
    QUEST3_GESTURES, NUM_QUEST3_CLASSES, RESULTS_DIR
)
from .model import create_model
from .quest3_dataset import get_quest3_dataloaders
from .utils import setup_logging, format_time, count_parameters


def log_per_class_accuracy(model: nn.Module,
                          val_loader: torch.utils.data.DataLoader,
                          device: torch.device,
                          gesture_classes: Dict[int, str],
                          logger) -> None:
    """Log per-class accuracy on Quest 3 validation set"""
    model.eval()
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for windows, labels in val_loader:
            # Quest 3 windows split into RGB and mask
            rgb = windows[:, :, :3].to(device)
            mask = windows[:, :, 3:].to(device)
            labels = labels.to(device)

            outputs = model(rgb, mask)
            # Average over sequence for classification (Quest 3)
            logits = outputs.mean(dim=1)  
            predictions = torch.argmax(logits, dim=1)

            # Count per-class accuracy
            for pred, true in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
                class_id = true
                if class_id not in class_correct:
                    class_correct[class_id] = 0
                    class_total[class_id] = 0
                class_total[class_id] += 1
                if pred == true:
                    class_correct[class_id] += 1

    # Log per-class accuracy
    logger.info("Per-class validation accuracy:")
    for class_id in sorted(gesture_classes.keys()):
        if class_id in class_correct:
            accuracy = class_correct[class_id] / class_total[class_id]
            class_name = gesture_classes[class_id]
            logger.info(f"  {class_name}: {accuracy:.4f} ({class_correct[class_id]}/{class_total[class_id]})")


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> Tuple[float, float]:
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (windows, labels) in enumerate(train_loader):
        # Split windows into RGB and mask channels 
        rgb = windows[:, :, :3].to(device)    
        mask = windows[:, :, 3:].to(device)   
        labels = labels.to(device)            

        # Forward pass correct format for Quest 3
        optimizer.zero_grad()
        outputs = model(rgb, mask)     

        # Average over sequence for classification (Quest 3)
        logits = outputs.mean(dim=1)   

        # For Quest 3 logits 
        logits_flat = logits  
        labels_flat = labels  

        # Compute loss
        loss = criterion(logits_flat, labels_flat)

        # Backward pass
        loss.backward()
        model.clip_gradients(TRAIN_CONFIG['gradient_clip_norm'])  
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits_flat, dim=1)
        correct += (predictions == labels_flat).sum().item()
        total += labels_flat.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model on Quest 3 validation set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for windows, labels in val_loader:
            # Quest 3 windows split into RGB and mask
            rgb = windows[:, :, :3].to(device)
            mask = windows[:, :, 3:].to(device)
            labels = labels.to(device)

            outputs = model(rgb, mask)
            # Average over sequence for classification (Quest 3)
            logits = outputs.mean(dim=1)  

            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(
    num_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Complete training pipeline for Quest 3 gesture recognition"""
    logger = setup_logging()

    if device is None:
        device = DEVICE

    # Create Quest 3 dataloaders
    train_loader, val_loader, test_loader = get_quest3_dataloaders(batch_size=batch_size, num_workers=0)
    num_classes = NUM_QUEST3_CLASSES
    dataset_name = "Quest 3"
    gesture_classes = QUEST3_GESTURES

    logger.info("QUEST 3 GESTURE SPOTTING TRAINING")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Classes: {num_classes} (8 Quest 3 gestures)")
    logger.info(f"Model: Multi-modal Transformer")
    logger.info(f"Device: {device}")

    # Create model
    logger.info("Creating model...")
    model = create_model(
        num_classes=num_classes,
        backbone=MODEL_CONFIG.get('backbone', 'resnet18'),
        rgb_pretrained=MODEL_CONFIG['rgb_pretrained'],
        mask_pretrained=MODEL_CONFIG['mask_pretrained'],
        fusion_dim=MODEL_CONFIG['fusion_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_layers=MODEL_CONFIG['num_layers'],
        feedforward_dim=MODEL_CONFIG['feedforward_dim'],
        dropout=MODEL_CONFIG['dropout'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
        device=device
    )

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer 
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Create learning rate scheduler 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           
        factor=0.5,           
        patience=5,           
        min_lr=1e-6           
    )

    # Create loss function for Quest 3
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    logger.info("\nStarting training...")

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Train epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step learning rate scheduler
        scheduler.step(val_loss)

        # Calculate train-val gap
        train_val_gap = train_acc - val_acc

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_val_gap'] = history.get('train_val_gap', []) + [train_val_gap]

        epoch_time = time.time() - epoch_start_time

        logger.info(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"Gap: {train_val_gap:.3f} | "
            f"Time: {format_time(epoch_time)}"
        )

        # Enhanced early stopping for Quest 3
        should_stop = False
        stop_reason = ""

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logger.info(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Check for severe overfitting 
        if train_val_gap > 0.30:
            should_stop = True
            stop_reason = f"severe overfitting (gap: {train_val_gap:.3f})"

        # Check for potential overfitting
        elif val_acc >= 1.0:
            logger.warning("Validation accuracy reached 100% - possible overfitting")
            

        # Standard early stopping
        if patience_counter >= TRAIN_CONFIG['patience']:
            should_stop = True
            stop_reason = "patience exceeded"

        if should_stop:
            logger.info(f"Early stopping at epoch {epoch+1} ({stop_reason})")
            break

        # Log per-class accuracy every 5 epochs
        if (epoch + 1) % 5 == 0:
            log_per_class_accuracy(model, val_loader, device, gesture_classes, logger)

    total_time = time.time() - start_time

    logger.info("TRAINING COMPLETED")
    logger.info(f"Total training time: {format_time(total_time)}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {BEST_MODEL_PATH}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    logger.info("\nTest Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")

    # Save final model
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    logger.info(f"Final model saved to: {FINAL_MODEL_PATH}")

    # Add test results to history
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc

    # Save training history to JSON
    history_path = RESULTS_DIR / "quest3_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")

    return history
