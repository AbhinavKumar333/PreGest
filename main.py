#!/usr/bin/env python3

"""Main entry point for PreGest gesture recognition system."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import (
    TRAIN_CONFIG, BEST_MODEL_PATH, RESULTS_DIR, PRETRAINING_MODE,
    NUM_CLASSES, SELECTED_GESTURE_IDS, JESTER_CLASSES, QUEST3_TARGET_GESTURES
)
from src.utils import setup_logging, format_time
from src.dataset import preprocess_jester_dataset, download_and_extract_jester
from src.train import train_transformer
from src.evaluate import evaluate_model, plot_confusion_matrix
from src.model import create_model
from src.utils import count_parameters
import torch


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="PreGest: Proactive Gesture-to-Action Prediction using Transformer Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic workflow
  python main.py preprocess --max-videos 5000  # Preprocess dataset
  python main.py train                          # Train the model
  python main.py evaluate                       # Evaluate trained model
  
  # With different pretraining modes
  python main.py train --mode quest3             # Train on 8 Quest3 gestures
  python main.py train --mode jester             # Train on all 27 Jester gestures
  python main.py train --mode custom             # Train on custom gesture subset
  
  # Advanced options
  python main.py train --epochs 50 --lr 1e-4    # Custom training params
  python main.py evaluate --model-path models/custom_model.pth
  python main.py info                            # Show system info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # TRAIN COMMAND    
    train_parser = subparsers.add_parser('train', help='Train the gesture recognition model')
    
    train_parser.add_argument(
        '--mode',
        type=str,
        choices=['quest3', 'jester', 'custom'],
        default='quest3',
        help='Pretraining mode: quest3 (8 gestures), jester (27 gestures), or custom'
    )
    
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=TRAIN_CONFIG['num_epochs'],
        help=f'Number of training epochs (default: {TRAIN_CONFIG["num_epochs"]})'
    )
    
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=TRAIN_CONFIG['batch_size'],
        help=f'Batch size (default: {TRAIN_CONFIG["batch_size"]})'
    )
    
    train_parser.add_argument(
        '--lr',
        type=float,
        default=TRAIN_CONFIG['learning_rate'],
        help=f'Learning rate (default: {TRAIN_CONFIG["learning_rate"]})'
    )
    
    train_parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint path'
    )

    # EVALUATE COMMAND
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    
    eval_parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model checkpoint (default: best model)'
    )
    
    eval_parser.add_argument(
        '--mode',
        type=str,
        choices=['quest3', 'jester', 'custom'],
        default='quest3',
        help='Pretraining mode used for training (for correct label mapping)'
    )
    
    # DOWNLOAD DATASET COMMAND
    download_parser = subparsers.add_parser(
        'download-dataset',
        help='Download Jester dataset from Kaggle'
    )
    
    download_parser.add_argument(
        '--kaggle-username',
        type=str,
        help='Kaggle username for authentication'
    )
    
    download_parser.add_argument(
        '--kaggle-key',
        type=str,
        help='Kaggle API key for authentication'
    )
    
    # PREPROCESS COMMAND
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess Jester dataset'
    )
    
    preprocess_parser.add_argument(
        '--max-videos',
        type=int,
        help='Maximum number of videos to process (for testing)'
    )
    
    preprocess_parser.add_argument(
        '--mode',
        type=str,
        choices=['quest3', 'jester', 'custom'],
        default='quest3',
        help='Pretraining mode: quest3 (8 gestures), jester (27 gestures), or custom'
    )
    
    # INFO COMMAND
    info_parser = subparsers.add_parser('info', help='Show system and model information')
    
    info_parser.add_argument(
        '--mode',
        type=str,
        choices=['quest3', 'jester', 'custom'],
        default='quest3',
        help='Pretraining mode to show info for'
    )
    
    return parser


def update_config_from_args(mode: str) -> None:
    """Update config based on command line mode argument."""
    
    from src import config
    
    if mode not in ['quest3', 'jester', 'custom']:
        raise ValueError(f"Unknown mode: {mode}. Choose from: quest3, jester, custom")
    
    config.PRETRAINING_MODE = mode
    
    # Recalculate dependent values
    if mode == "quest3":
        config.NUM_CLASSES = len(config.QUEST3_TARGET_GESTURES)
        config.SELECTED_GESTURE_IDS = list(config.JESTER_TO_QUEST3_MAPPING.keys())
    elif mode == "jester":
        config.NUM_CLASSES = config.NUM_JESTER_CLASSES
        config.SELECTED_GESTURE_IDS = list(range(config.NUM_JESTER_CLASSES))
    elif mode == "custom":
        config.SELECTED_GESTURE_IDS = config.CUSTOM_GESTURE_IDS
        config.NUM_CLASSES = len(config.CUSTOM_GESTURE_IDS)
    
    # Update model paths to include mode suffix
    config.BEST_MODEL_PATH = config.MODELS_DIR / f"gesture_transformer_best_{mode}.pth"
    config.FINAL_MODEL_PATH = config.MODELS_DIR / f"gesture_transformer_final_{mode}.pth"
    config.LOG_FILE = config.LOG_DIR / f"training_{mode}.log"


def command_train(args: argparse.Namespace) -> None:
    """Handle train command."""
    
    logger = setup_logging()
    
    # Update config based on mode
    mode = getattr(args, 'mode', 'quest3')
    update_config_from_args(mode)
    
    logger.info("="*70)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*70)
    logger.info(f"Pretraining Mode: {PRETRAINING_MODE}")
    logger.info(f"Number of Classes: {NUM_CLASSES}")
    logger.info(f"Selected Gesture IDs: {SELECTED_GESTURE_IDS}")
    logger.info("="*70)
    
    # Train the model
    training_history = train_transformer(
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    if training_history:
        # Get final metrics
        final_train_acc = training_history['train_acc'][-1] * 100
        final_val_acc = training_history['val_acc'][-1] * 100
        best_val_loss = min(training_history['val_loss'])
        
        # Get model info
        model = create_model(num_classes=NUM_CLASSES)
        total_params = count_parameters(model)
        
        # Print final results
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Pretraining Mode: {PRETRAINING_MODE}")
        print(f"Number of Classes: {NUM_CLASSES}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        
        if 'test_acc' in training_history:
            test_acc = training_history['test_acc'] * 100
            print(f"Test Accuracy: {test_acc:.2f}%")
        
        print(f"Best Model Saved: {BEST_MODEL_PATH}")
        print("="*70)
    else:
        logger.error("Training failed - check logs for details")
        sys.exit(1)


def command_evaluate(args: argparse.Namespace) -> None:
    """Handle evaluate command."""
    
    logger = setup_logging()
    
    # Update config based on mode
    mode = getattr(args, 'mode', 'quest3')
    update_config_from_args(mode)
    
    logger.info("="*70)
    logger.info("STARTING EVALUATION")
    logger.info("="*70)
    logger.info(f"Pretraining Mode: {PRETRAINING_MODE}")
    logger.info(f"Number of Classes: {NUM_CLASSES}")
    logger.info("="*70)
    
    # Load model
    model_path = Path(args.model_path) if args.model_path else BEST_MODEL_PATH
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes=NUM_CLASSES, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load test data
    from src.dataset import load_preprocessed_data
    _, _, test_loader = load_preprocessed_data()
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    if results:
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        if results.get('top3_accuracy'):
            print(f"Top-3 Accuracy: {results['top3_accuracy']:.4f}")
        print(f"Macro F1-Score: {results['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
        print(f"Results saved to: {RESULTS_DIR}")
        print("="*70)
    else:
        logger.error("Evaluation failed - check logs for details")
        sys.exit(1)


def command_preprocess(args: argparse.Namespace) -> None:
    """Handle preprocess command."""
    
    logger = setup_logging()
    
    # Update config based on mode
    mode = getattr(args, 'mode', 'quest3')
    update_config_from_args(mode)
    
    logger.info("="*70)
    logger.info("STARTING DATASET PREPROCESSING")
    logger.info("="*70)
    logger.info(f"Pretraining Mode: {PRETRAINING_MODE}")
    logger.info(f"Number of Classes: {NUM_CLASSES}")
    logger.info(f"Selected Gesture IDs: {SELECTED_GESTURE_IDS}")
    logger.info("="*70)
    
    # Get max_videos parameter
    max_videos = getattr(args, 'max_videos', None)
    
    if max_videos:
        logger.info(f"Limiting processing to {max_videos} videos for faster testing")
    
    try:
        train_loader, val_loader, test_loader = preprocess_jester_dataset(
            max_videos=max_videos
        )
        
        print("\n" + "="*70)
        print("DATASET PREPROCESSING COMPLETED!")
        print("="*70)
        print(f"Pretraining Mode: {PRETRAINING_MODE}")
        print(f"Number of Classes: {NUM_CLASSES}")
        print("Next step: python main.py train --mode {}".format(PRETRAINING_MODE))
        print("="*70)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        print("\nDataset preprocessing failed.")
        print("Make sure you have downloaded the dataset first:")
        print("  python main.py download-dataset")
        sys.exit(1)


def command_download(args: argparse.Namespace) -> None:
    """Handle download-dataset command."""
    
    logger = setup_logging()
    logger.info("Attempting to download Jester dataset...")
    
    success = download_and_extract_jester()
    
    if success:
        print("\nDataset downloaded successfully!")
        print("Next step: python main.py preprocess --max-videos 5000")
    else:
        print("\nDataset download failed.")
        print("Please download manually from:")
        print("  https://www.qualcomm.com/developer/software/jester-dataset/downloads")
        sys.exit(1)


def command_info(args: argparse.Namespace) -> None:
    """Handle info command."""
    
    # Update config based on mode
    mode = getattr(args, 'mode', 'quest3')
    update_config_from_args(mode)
    
    print("\n" + "="*70)
    print("PreGest: Proactive Gesture-to-Action Prediction")
    print("="*70)
    
    # Model info
    model = create_model(num_classes=NUM_CLASSES)
    total_params = count_parameters(model)
    
    print("\n📊 Model Information:")
    print(f"  Model: GestureTransformer")
    print(f"  Total Parameters: {total_params:,}")
    
    # Dataset info
    print("\n📁 Dataset Information:")
    print(f"  Dataset: 20BN-Jester")
    print(f"  Pretraining Mode: {PRETRAINING_MODE}")
    print(f"  Number of Classes: {NUM_CLASSES}")
    
    if PRETRAINING_MODE == "quest3":
        print(f"  Gestures (Quest3-mapped):")
        for gesture_id, gesture_name in QUEST3_TARGET_GESTURES.items():
            jester_id = list(JESTER_TO_QUEST3_MAPPING.keys())[gesture_id]
            print(f"    {gesture_id}: {gesture_name} (Jester: {jester_id})")
    elif PRETRAINING_MODE == "jester":
        print(f"  Gestures (All Jester): {SELECTED_GESTURE_IDS}")
        print(f"  First 5 gestures:")
        for i in range(min(5, NUM_CLASSES)):
            print(f"    {i}: {JESTER_CLASSES[i]}")
    else:
        print(f"  Custom Gestures: {SELECTED_GESTURE_IDS}")
    
    # Training config
    print("\n⚙️  Training Configuration:")
    print(f"  Epochs: {TRAIN_CONFIG['num_epochs']}")
    print(f"  Batch Size: {TRAIN_CONFIG['batch_size']}")
    print(f"  Learning Rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # File paths
    print("\n📂 Important Paths:")
    print(f"  Models: models/")
    print(f"  Results: results/")
    print(f"  Data: data/")
    print(f"  Logs: logs/")
    
    print("\n" + "="*70 + "\n")


def main() -> None:
    """Main entry point."""
    
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    command_handlers = {
        'train': command_train,
        'evaluate': command_evaluate,
        'preprocess': command_preprocess,
        'download-dataset': command_download,
        'info': command_info,
    }
    
    handler = command_handlers.get(args.command)
    
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
