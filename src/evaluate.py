"""Evaluation and metrics for gesture recognition model."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, top_k_accuracy_score
)
import json
import logging

from .config import (
    JESTER_CLASSES, QUEST3_TARGET_GESTURES, RESULTS_DIR, BEST_MODEL_PATH,
    TRAIN_CONFIG, MODEL_CONFIG, NUM_CLASSES, PRETRAINING_MODE
)
from .model import create_model
from .utils import setup_logging, ensure_directory_exists


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Evaluate model on test set with comprehensive metrics.
    
    Args:
        model: Trained PyTorch model.
        test_loader: Test dataloader.
        device: Device to run evaluation on.
        class_names: List of class names for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger = setup_logging()
    
    # Determine class names
    if class_names is None:
        if PRETRAINING_MODE == "quest3":
            class_names = [QUEST3_TARGET_GESTURES[i] for i in range(NUM_CLASSES)]
        else:
            class_names = JESTER_CLASSES[:NUM_CLASSES]
    
    logger.info("="*70)
    logger.info("MODEL EVALUATION")
    logger.info("="*70)
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    
    # Collect predictions
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits = model(sequences)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Calculate metrics
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    
    # Try top-3 accuracy
    try:
        top3_accuracy = top_k_accuracy_score(all_labels, all_logits, k=3)
    except:
        top3_accuracy = None
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Per-class detailed metrics
    per_class_metrics = {}
    precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision_per[i]),
            'recall': float(recall_per[i]),
            'f1': float(f1_per[i]),
        }
    
    # Results dictionary
    results = {
        'overall_accuracy': float(overall_accuracy),
        'top3_accuracy': float(top3_accuracy) if top3_accuracy is not None else None,
        'weighted_precision': float(precision),
        'weighted_recall': float(recall),
        'weighted_f1': float(f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics,
    }
    
    # Log results
    logger.info(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    if top3_accuracy is not None:
        logger.info(f"Top-3 Accuracy: {top3_accuracy:.4f}")
    logger.info(f"\nWeighted Metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    
    logger.info(f"\nMacro Metrics:")
    logger.info(f"  Precision: {macro_precision:.4f}")
    logger.info(f"  Recall: {macro_recall:.4f}")
    logger.info(f"  F1-Score: {macro_f1:.4f}")
    
    logger.info(f"\nPer-Class Metrics:")
    for class_name, metrics in per_class_metrics.items():
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall: {metrics['recall']:.4f}")
        logger.info(f"    F1-Score: {metrics['f1']:.4f}")
    
    # Visualizations
    plot_confusion_matrix(cm, class_names)
    plot_per_class_f1(per_class_metrics, class_names)
    
    # Save results
    save_evaluation_results(results, class_names)
    
    logger.info("\n" + "="*70)
    
    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    """Plot confusion matrix heatmap."""
    logger = setup_logging()
    ensure_directory_exists(RESULTS_DIR)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = RESULTS_DIR / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_per_class_f1(per_class_metrics: Dict[str, Dict], class_names: List[str]) -> None:
    """Plot per-class F1 scores."""
    logger = setup_logging()
    ensure_directory_exists(RESULTS_DIR)
    
    f1_scores = [per_class_metrics[name]['f1'] for name in class_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_names)), f1_scores)
    plt.xlabel('Gesture Class')
    plt.ylabel('F1-Score')
    plt.title('Per-Class F1 Scores')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = RESULTS_DIR / 'per_class_f1.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Per-class F1 plot saved to {save_path}")
    plt.close()


def save_evaluation_results(results: Dict[str, Any], class_names: List[str]) -> None:
    """Save evaluation results to JSON file."""
    logger = setup_logging()
    ensure_directory_exists(RESULTS_DIR)

    results_file = RESULTS_DIR / 'evaluation_results.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {results_file}")


def run_complete_evaluation(model_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Run complete evaluation pipeline on test set.

    Args:
        model_path: Path to model checkpoint. If None, uses BEST_MODEL_PATH.

    Returns:
        Evaluation results dictionary or None if failed.
    """
    logger = setup_logging()

    try:
        # Load model
        if model_path is None:
            model_path = BEST_MODEL_PATH

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None

        logger.info(f"Loading model from {model_path}")
        model = create_model()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # Load test data
        from .dataset import load_preprocessed_data
        _, _, test_loader = load_preprocessed_data()

        if test_loader is None:
            logger.error("Failed to load test data")
            return None

        # Run evaluation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        results = evaluate_model(model, test_loader, device)

        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None


def plot_training_history(training_history: Dict[str, List[float]]) -> None:
    """Plot training history (loss and accuracy curves).

    Args:
        training_history: Dictionary with training metrics
    """
    logger = setup_logging()
    ensure_directory_exists(RESULTS_DIR)

    # Extract data
    epochs = list(range(1, len(training_history.get('train_loss', [])) + 1))

    train_loss = training_history.get('train_loss', [])
    val_loss = training_history.get('val_loss', [])
    train_acc = training_history.get('train_acc', [])
    val_acc = training_history.get('val_acc', [])

    # Create subplots
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    if val_loss:
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    if val_acc:
        ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = RESULTS_DIR / 'training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training history plot saved to {save_path}")
    plt.close()
