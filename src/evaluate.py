"""Evaluation and metrics for EgoCentricGesture spotting model."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

from .config import (
    QUEST3_GESTURES, RESULTS_DIR, BEST_MODEL_PATH, NUM_QUEST3_CLASSES, DEVICE, ACTIVE_DATASET
)
from .model import create_model
from .quest3_dataset import get_quest3_dataloaders
from .utils import setup_logging, ensure_directory_exists
import time


def accuracy_score_manual(y_true, y_pred):
    """Manual accuracy calculation."""
    return np.mean(y_true == y_pred)


def precision_recall_fscore_support_manual(y_true, y_pred, average=None):
    """Manual precision, recall, fscore calculation."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Initialize arrays
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    fscore = np.zeros(n_classes)
    support = np.zeros(n_classes, dtype=int)

    for i, cls in enumerate(classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        # Precision, recall, f-score
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fscore[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        support[i] = np.sum(y_true == cls)

    if average == 'macro':
        return np.mean(precision), np.mean(recall), np.mean(fscore), None
    elif average == 'weighted':
        weights = support / np.sum(support)
        return np.sum(precision * weights), np.sum(recall * weights), np.sum(fscore * weights), None
    elif average is None:
        return precision, recall, fscore, support
    else:
        raise ValueError(f"Unsupported average: {average}")


def confusion_matrix_manual(y_true, y_pred):
    """Manual confusion matrix calculation."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(y_true)):
        true_idx = np.where(classes == y_true[i])[0][0]
        pred_idx = np.where(classes == y_pred[i])[0][0]
        cm[true_idx, pred_idx] += 1

    return cm


def analyze_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> List[Tuple[str, int]]:
    """Analyze confusion matrix to find most confused pairs.

    Args:
        cm: Confusion matrix
        class_names: List of class names

    Returns:
        List of (pair_description, count) tuples, sorted by count descending
    """
    confused_pairs = []

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pair = f"{class_names[i]} → {class_names[j]}"
                confused_pairs.append((pair, int(cm[i, j])))

    # Sort by confusion count descending
    confused_pairs.sort(key=lambda x: x[1], reverse=True)

    return confused_pairs


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset: str = "auto"
) -> Dict[str, Any]:
    """Evaluate gesture spotting model with frame-level metrics.

    Args:
        model: Trained gesture spotting model
        test_loader: Test dataloader
        device: Device to run evaluation on
        dataset: Dataset type ("quest3", "egocentric", or "auto")

    Returns:
        Dictionary with evaluation metrics
    """
    logger = setup_logging()

    # Determine dataset and classes
    if dataset == "auto":
        dataset = ACTIVE_DATASET

    if dataset == "quest3":
        num_classes = NUM_QUEST3_CLASSES
        class_names = [QUEST3_GESTURES[i] for i in range(num_classes)]
        dataset_title = "Quest 3"
    else:
        # Unsupported dataset - should not happen in cleaned version
        raise ValueError(f"Unsupported dataset: {dataset}")

    logger.info("="*70)
    logger.info(f"{dataset_title.upper()} GESTURE SPOTTING EVALUATION")
    logger.info("="*70)

    model.eval()
    all_predictions = []
    all_labels = []

    # Performance measurement
    inference_times = []
    total_frames = 0

    with torch.no_grad():
        for batch_data, labels in test_loader:
            # Quest 3: batch_data is (B, 60, 4, 224, 224) - combined RGB and mask
            rgb = batch_data[:, :, :3].to(device)  # (B, 60, 3, 224, 224)
            mask = batch_data[:, :, 3:].to(device)  # (B, 60, 1, 224, 224)
            labels = labels.to(device)

            # Measure inference time
            start_time = time.time()
            logits = model(rgb, mask)  # (B, 60, C)
            inference_times.append(time.time() - start_time)

            # Quest 3 averaging (same as training)
            B = logits.shape[0]  # Get batch size
            logits = logits.mean(dim=1)  # (B, num_classes) - average over sequence
            logits_flat = logits  # (B, num_classes)
            labels_flat = labels  # (B,)
            total_frames += B  # Count sequences, not frames for Quest 3

            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Performance metrics
    avg_inference_time = np.mean(inference_times)
    throughput_fps = total_frames / sum(inference_times)

    # Calculate frame-level accuracy
    overall_accuracy = accuracy_score_manual(all_labels, all_predictions)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support_manual(
        all_labels, all_predictions, average='weighted'
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support_manual(
        all_labels, all_predictions, average='macro'
    )

    # Confusion matrix
    cm = confusion_matrix_manual(all_labels, all_predictions)

    # Per-class detailed metrics
    per_class_metrics = {}
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support_manual(
        all_labels, all_predictions, average=None
    )

    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision_per[i]),
            'recall': float(recall_per[i]),
            'f1': float(f1_per[i]),
            'support': int(support_per[i])
        }

    # Error analysis - top confused pairs
    confused_pairs = analyze_confusion_matrix(cm, class_names)

    # Results dictionary
    results = {
        'dataset': dataset,
        'overall_accuracy': float(overall_accuracy),
        'weighted_precision': float(precision),
        'weighted_recall': float(recall),
        'weighted_f1': float(f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics,
        'performance_metrics': {
            'avg_inference_time_ms': float(avg_inference_time * 1000),
            'throughput_fps': float(throughput_fps),
            'total_frames': int(total_frames)
        },
        'error_analysis': confused_pairs,
    }

    # Log results
    logger.info(f"Frame-Level Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Macro F1-Score: {macro_f1:.4f}")
    logger.info(f"Weighted F1-Score: {f1:.4f}")
    logger.info(f"Average Inference Time: {avg_inference_time*1000:.2f} ms")
    logger.info(f"Throughput: {throughput_fps:.1f} FPS")

    logger.info(f"\nPer-Class Metrics:")
    for class_name, metrics in per_class_metrics.items():
        logger.info(f"  {class_name}: Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    logger.info(f"\nTop 10 Most Confused Pairs:")
    for i, (pair, count) in enumerate(confused_pairs[:10]):
        logger.info(f"  {i+1}. {pair}: {count} confusions")

    # Visualizations
    plot_confusion_matrix(cm, class_names, dataset_title)
    plot_per_class_f1(per_class_metrics, class_names, dataset_title)

    # Save results
    save_evaluation_results(results, dataset)

    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], dataset_title: str = "Gesture") -> None:
    """Plot confusion matrix heatmap."""
    logger = setup_logging()
    ensure_directory_exists(RESULTS_DIR)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{dataset_title} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = RESULTS_DIR / f'{dataset_title.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_per_class_f1(per_class_metrics: Dict[str, Dict], class_names: List[str], dataset_title: str = "Gesture") -> None:
    """Plot per-class F1 scores."""
    logger = setup_logging()
    ensure_directory_exists(RESULTS_DIR)

    f1_scores = [per_class_metrics[name]['f1'] for name in class_names]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_names)), f1_scores)
    plt.xlabel('Gesture Class')
    plt.ylabel('F1-Score')
    plt.title(f'{dataset_title} Per-Class F1 Scores')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.tight_layout()

    save_path = RESULTS_DIR / f'{dataset_title.lower().replace(" ", "_")}_per_class_f1.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Per-class F1 plot saved to {save_path}")
    plt.close()


def save_evaluation_results(results: Dict[str, Any], dataset: str = "gesture") -> None:
    """Save evaluation results to JSON file."""
    logger = setup_logging()
    ensure_directory_exists(RESULTS_DIR)

    results_file = RESULTS_DIR / f'{dataset}_evaluation_results.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {results_file}")


def run_complete_evaluation(model_path: Optional[Path] = None, dataset: str = "auto") -> Optional[Dict[str, Any]]:
    """Run complete evaluation pipeline on test set.

    Args:
        model_path: Path to model checkpoint. If None, uses BEST_MODEL_PATH.
        dataset: Dataset type ("quest3", "egocentric", or "auto")

    Returns:
        Evaluation results dictionary or None if failed.
    """
    logger = setup_logging()

    try:
        # Determine dataset
        if dataset == "auto":
            dataset = ACTIVE_DATASET

        # Load model
        if model_path is None:
            model_path = BEST_MODEL_PATH

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None

        logger.info(f"Loading model from {model_path}")

        # Determine number of classes for model creation
        from .config import MODEL_CONFIG
        num_classes = NUM_QUEST3_CLASSES  # Only Quest 3 supported
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
            dropout=MODEL_CONFIG['dropout']
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # Load test data - use consistent batch size
        from .config import TRAIN_CONFIG
        if dataset == "quest3":
            _, _, test_loader = get_quest3_dataloaders(batch_size=TRAIN_CONFIG['batch_size'])
        else:
            # EgoCentric preprocessing not supported in cleaned version
            logger.error("EgoCentric evaluation not supported in cleaned version")
            return None

        if test_loader is None:
            logger.error("Failed to load test data")
            return None

        # Run evaluation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        results = evaluate_model(model, test_loader, device, dataset)

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
