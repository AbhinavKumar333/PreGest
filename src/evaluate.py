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
    JESTER_CLASSES, RESULTS_DIR, BEST_MODEL_PATH,
    TRAIN_CONFIG, MODEL_CONFIG
)
from .model import create_model
from .utils import setup_logging, ensure_directory_exists
from .dataset import preprocess_jester_dataset


def evaluate_model(model: torch.nn.Module,
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  class_names: List[str] = JESTER_CLASSES) -> Dict[str, Any]:
    """Evaluate model on test set with comprehensive metrics.

    Args:
        model: Trained PyTorch model.
        test_loader: Test dataloader.
        device: Device to run evaluation on.
        class_names: List of class names.

    Returns:
        Dictionary with evaluation metrics.
    """
    logger = setup_logging()
    logger.info("Starting model evaluation...")

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Overall metrics
    overall_accuracy = accuracy_score(all_labels, all_predictions)

    # Top-3 accuracy
    top3_accuracy = top_k_accuracy_score(all_labels, all_probabilities, k=3)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )

    # Macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )

    # Weighted-averaged metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Per-class metrics dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }

    # Results dictionary
    results = {
        'overall_accuracy': float(overall_accuracy),
        'top3_accuracy': float(top3_accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics,
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probabilities.tolist()
    }

    logger.info(f"Evaluation completed - Accuracy: {overall_accuracy:.4f}, "
               f"F1-Score: {macro_f1:.4f}, Top-3 Acc: {top3_accuracy:.4f}")

    return results


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str] = JESTER_CLASSES,
                         save_path: Optional[Path] = None) -> None:
    """Plot and save confusion matrix.

    Args:
        cm: Confusion matrix array.
        class_names: List of class names.
        save_path: Path to save plot. If None, uses default.
    """
    if save_path is None:
        save_path = RESULTS_DIR / "confusion_matrix.png"

    ensure_directory_exists(save_path)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of predictions'})

    plt.title('Confusion Matrix - Gesture Recognition', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(training_history: Dict[str, List[float]],
                         save_path: Optional[Path] = None) -> None:
    """Plot training history (loss and accuracy curves).

    Args:
        training_history: Dictionary with training metrics.
        save_path: Path to save plot. If None, uses default.
    """
    if save_path is None:
        save_path = RESULTS_DIR / "training_history.png"

    ensure_directory_exists(save_path)

    # Generate epochs if not provided
    num_epochs = len(training_history.get('train_loss', []))
    epochs = list(range(1, num_epochs + 1))

    train_loss = training_history.get('train_loss', [])
    val_loss = training_history.get('val_loss', [])
    train_acc = training_history.get('train_acc', [])
    val_acc = training_history.get('val_acc', [])

    # Convert accuracies to percentages if needed
    if train_acc and max(train_acc) <= 1.0:
        train_acc = [x * 100 for x in train_acc]
    if val_acc and max(val_acc) <= 1.0:
        val_acc = [x * 100 for x in val_acc]

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss plot
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training history plot saved to {save_path}")


def plot_per_class_metrics(per_class_metrics: Dict[str, Dict[str, float]],
                          save_path: Optional[Path] = None) -> None:
    """Plot per-class F1 scores as a bar chart.

    Args:
        per_class_metrics: Dictionary with per-class metrics.
        save_path: Path to save plot. If None, uses default.
    """
    if save_path is None:
        save_path = RESULTS_DIR / "per_class_f1.png"

    ensure_directory_exists(save_path)

    class_names = list(per_class_metrics.keys())
    f1_scores = [metrics['f1_score'] for metrics in per_class_metrics.values()]

    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(class_names)), f1_scores, color='skyblue', alpha=0.8)

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                '.3f', ha='center', va='bottom', fontsize=8)

    plt.title('Per-Class F1 Scores', fontsize=16, pad=20)
    plt.xlabel('Gesture Classes', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Per-class F1 scores plot saved to {save_path}")


def save_evaluation_results(results: Dict[str, Any],
                           filepath: Optional[Path] = None) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary.
        filepath: Path to save results. If None, uses default.
    """
    if filepath is None:
        filepath = RESULTS_DIR / "evaluation_metrics.json"

    ensure_directory_exists(filepath)

    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = value
        else:
            serializable_results[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Evaluation results saved to {filepath}")


def generate_classification_report(results: Dict[str, Any],
                                 class_names: List[str] = JESTER_CLASSES,
                                 save_path: Optional[Path] = None) -> str:
    """Generate detailed classification report.

    Args:
        results: Evaluation results dictionary.
        class_names: List of class names.
        save_path: Path to save report. If None, uses default.

    Returns:
        Classification report string.
    """
    if save_path is None:
        save_path = RESULTS_DIR / "classification_report.txt"

    ensure_directory_exists(save_path)

    y_true = results['labels']
    y_pred = results['predictions']

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    with open(save_path, 'w') as f:
        f.write("GESTURE RECOGNITION CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print(f"Classification report saved to {save_path}")
    return report


def run_complete_evaluation(model_path: Optional[Path] = None,
                          config: Dict = TRAIN_CONFIG) -> Dict[str, Any]:
    """Run complete evaluation pipeline.

    Args:
        model_path: Path to model checkpoint. If None, uses best model.
        config: Training configuration.

    Returns:
        Complete evaluation results.
    """
    logger = setup_logging()
    logger.info("Running complete evaluation pipeline...")

    # Set device
    device = config['device']

    # Load model
    if model_path is None:
        model_path = BEST_MODEL_PATH

    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return {}

    logger.info(f"Loading model from {model_path}")
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data
    dataloaders = preprocess_jester_dataset()
    if 'test' not in dataloaders:
        logger.error("Test dataloader not available")
        return {}

    test_loader = dataloaders['test']

    # Run evaluation
    results = evaluate_model(model, test_loader, device)

    # Generate visualizations and reports
    plot_confusion_matrix(np.array(results['confusion_matrix']))
    plot_per_class_metrics(results['per_class_metrics'])
    save_evaluation_results(results)
    generate_classification_report(results)

    logger.info("Complete evaluation pipeline finished")
    return results


def compare_models(model_paths: List[Path],
                  model_names: List[str],
                  config: Dict = TRAIN_CONFIG) -> Dict[str, Dict[str, Any]]:
    """Compare multiple models.

    Args:
        model_paths: List of paths to model checkpoints.
        model_names: List of model names for comparison.
        config: Training configuration.

    Returns:
        Dictionary with comparison results.
    """
    logger = setup_logging()
    logger.info(f"Comparing {len(model_paths)} models...")

    comparison_results = {}

    for model_path, model_name in zip(model_paths, model_names):
        logger.info(f"Evaluating {model_name}...")
        results = run_complete_evaluation(model_path, config)
        comparison_results[model_name] = results

    # Save comparison
    comparison_file = RESULTS_DIR / "model_comparison.json"
    ensure_directory_exists(comparison_file)

    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    logger.info(f"Model comparison saved to {comparison_file}")
    return comparison_results
