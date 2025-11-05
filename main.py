#!/usr/bin/env python3
"""Main entry point for PreGest gesture recognition system."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import TRAIN_CONFIG, BEST_MODEL_PATH, RESULTS_DIR
from src.utils import setup_logging, format_time
from src.dataset import preprocess_jester_dataset, download_and_extract_jester
from src.train import train_transformer
from src.evaluate import run_complete_evaluation, plot_training_history
from src.model import create_model
from src.utils import count_parameters


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="PreGest: Proactive Gesture-to-Action Prediction using Transformer Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train the model
  python main.py evaluate                 # Evaluate trained model
  python main.py download-dataset         # Download Jester dataset
  python main.py preprocess               # Preprocess dataset
  python main.py train --epochs 50        # Train with custom epochs
  python main.py evaluate --model-path models/custom_model.pth
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the gesture recognition model')
    train_parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'],
                             help=f'Number of training epochs (default: {TRAIN_CONFIG["num_epochs"]})')
    train_parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'],
                             help=f'Batch size (default: {TRAIN_CONFIG["batch_size"]})')
    train_parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'],
                             help=f'Learning rate (default: {TRAIN_CONFIG["learning_rate"]})')
    train_parser.add_argument('--resume', type=str,
                             help='Resume training from checkpoint path')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str,
                            help='Path to model checkpoint (default: best model)')

    # Download dataset command
    download_parser = subparsers.add_parser('download-dataset',
                                           help='Download Jester dataset from Kaggle')
    download_parser.add_argument('--kaggle-username', type=str,
                                help='Kaggle username for authentication')
    download_parser.add_argument('--kaggle-key', type=str,
                                help='Kaggle API key for authentication')

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess',
                                             help='Preprocess Jester dataset')
    preprocess_parser.add_argument('--max-videos', type=int,
                                  help='Maximum number of videos to process (for testing)')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system and model information')

    return parser


def command_train(args: argparse.Namespace) -> None:
    """Handle train command."""
    logger = setup_logging()
    logger.info("Starting training pipeline...")

    # Update config with command line arguments
    num_epochs = args.epochs if args.epochs != TRAIN_CONFIG['num_epochs'] else None
    learning_rate = args.lr if args.lr != TRAIN_CONFIG['learning_rate'] else None

    # Train the model
    training_history = train_transformer(
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    if training_history:
        # Plot training history
        plot_training_history(training_history)

        # Get final metrics
        final_train_acc = training_history['train_acc'][-1] * 100
        final_val_acc = training_history['val_acc'][-1] * 100
        best_val_loss = min(training_history['val_loss'])

        # Get model info
        model = create_model()
        total_params = count_parameters(model)

        # Print final results
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Total Parameters: {total_params:,}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        if 'test_acc' in training_history:
            test_acc = training_history['test_acc'] * 100
            print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Best Model Saved: {BEST_MODEL_PATH}")
        print("="*60)
    else:
        logger.error("Training failed - check logs for details")
        sys.exit(1)


def command_evaluate(args: argparse.Namespace) -> None:
    """Handle evaluate command."""
    logger = setup_logging()
    logger.info("Starting evaluation...")

    model_path = Path(args.model_path) if args.model_path else None

    results = run_complete_evaluation(model_path)

    if results:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Top-3 Accuracy: {results['top3_accuracy']:.4f}")
        print(f"Macro F1-Score: {results['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
        print(f"Results saved to: {RESULTS_DIR}")
        print("="*60)
    else:
        logger.error("Evaluation failed - check logs for details")
        sys.exit(1)


def command_preprocess(args: argparse.Namespace) -> None:
    """Handle preprocess command."""
    logger = setup_logging()
    logger.info("Starting dataset preprocessing...")

    # Get max_videos parameter
    max_videos = getattr(args, 'max_videos', None)
    if max_videos:
        logger.info(f"Limiting processing to {max_videos} videos for faster testing")

    try:
        train_loader, val_loader, test_loader = preprocess_jester_dataset(max_videos=max_videos)
        print("Dataset preprocessing completed!")
        print("Next step: python main.py train")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        print("Dataset preprocessing failed.")
        print("Make sure you have downloaded the dataset first:")
        print("python main.py download-dataset")
        sys.exit(1)


def command_info(args: argparse.Namespace) -> None:
    """Handle info command."""
    print("PreGest: Proactive Gesture-to-Action Prediction")
    print("="*50)

    # Model info
    model = create_model()
    total_params = count_parameters(model)
    trainable_params = count_parameters(model)

    print(f"Model: GestureTransformer")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print()

    # Dataset info
    from src.config import NUM_CLASSES, JESTER_CLASSES
    print(f"Dataset: 20BN-Jester")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Classes: {JESTER_CLASSES[:5]}...")  # Show first 5
    print()

    # Training config
    print("Training Configuration:")
    print(f"  Epochs: {TRAIN_CONFIG['num_epochs']}")
    print(f"  Batch Size: {TRAIN_CONFIG['batch_size']}")
    print(f"  Learning Rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Device: {TRAIN_CONFIG['device']}")
    print()

    # File paths
    print("Important Paths:")
    print(f"  Models: models/")
    print(f"  Results: results/")
    print(f"  Data: data/")
    print(f"  Logs: pregrest_training.log")


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
