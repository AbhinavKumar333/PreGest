"""
PreGest: Quest 3 Gesture Recognition System
Main CLI entry point for preprocessing, training, and evaluation.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import (
    QUEST3_GESTURES, NUM_QUEST3_CLASSES, DEVICE, ACTIVE_DATASET
)
from src.quest3_preprocessor import preprocess_quest3_dataset
from src.train import train_model
from src.evaluate import run_complete_evaluation
from src.quest3_dataset import get_quest3_dataloaders
from src.model import create_model
from src.utils import setup_logging, count_parameters
import torch


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="PreGest: Quest 3 Gesture Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quest 3 complete workflow
  python main.py --dataset quest3 --preprocess --mode both
  python main.py --dataset quest3 --mode train
  python main.py --dataset quest3 --mode eval

  # Custom training parameters
  python main.py train --epochs 50 --batch-size 8 --learning-rate 1e-5

  # System information
  python main.py info
"""
    )

    # Global arguments
    parser.add_argument(
        '--dataset',
        choices=['quest3'],
        default=ACTIVE_DATASET,
        help='Dataset to use (default: quest3)'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run preprocessing before training'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'eval', 'both'],
        default='both',
        help='Execution mode (default: both)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ========================================================================
    # PREPROCESS COMMAND
    # ========================================================================
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Extract gesture windows from dataset'
    )
    preprocess_parser.add_argument(
        '--max-subjects',
        type=int,
        help='Limit processing to N subjects (for testing)'
    )
    preprocess_parser.add_argument(
        '--quest3',
        action='store_true',
        help='Filter to Quest 3 compatible gestures only (Stop, Grab, Swipe Right/Left)'
    )

    # ========================================================================
    # (Split command removed - Quest 3 uses internal train/val split)
    # ========================================================================

    # ========================================================================
    # TRAIN COMMAND
    # ========================================================================
    train_parser = subparsers.add_parser('train', help='Train gesture spotting model')
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=50 if ACTIVE_DATASET == 'quest3' else 30,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4 if ACTIVE_DATASET == 'quest3' else 5e-5,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--dataset',
        choices=['quest3'],
        default=ACTIVE_DATASET,
        help='Dataset to use (default: quest3)'
    )

    # ========================================================================
    # EVALUATE COMMAND
    # ========================================================================
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    evaluate_parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model checkpoint (default: best model)'
    )

    # ========================================================================
    # INFO COMMAND
    # ========================================================================
    info_parser = subparsers.add_parser('info', help='Show system and model information')

    return parser


def command_preprocess(args: argparse.Namespace) -> None:
    """Handle preprocess command."""
    print("="*70)
    print("QUEST 3 GESTURE DATASET PREPROCESSING")
    print("="*70)

    try:
        stats = preprocess_quest3_dataset()
        print(f"\n✅ Preprocessing complete!")
        print(f"   Videos processed: {stats['processed_videos']}")
        print(f"   Windows created: {stats['total_windows']}")
        print(f"   Next: python main.py train")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        sys.exit(1)


def command_train(args: argparse.Namespace) -> None:
    """Handle train command."""
    # Store the original ACTIVE_DATASET
    from src.config import ACTIVE_DATASET as original_dataset

    # Temporarily set dataset if specified
    if hasattr(args, 'dataset') and args.dataset != original_dataset:
        import src.config
        src.config.ACTIVE_DATASET = args.dataset
        print(f"Using dataset: {args.dataset}")

    history = train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    print(f"\n✅ Training complete!")
    print(f"   Test accuracy: {history['test_acc']:.4f}")
    print(f"   Next: python main.py evaluate")


def command_evaluate(args: argparse.Namespace) -> None:
    """Handle evaluate command."""
    from src.config import BEST_MODEL_PATH, ACTIVE_DATASET as original_dataset

    model_path = Path(args.model_path) if args.model_path else BEST_MODEL_PATH

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Use Quest 3 dataset for evaluation since we trained with Quest 3
    # Temporarily override ACTIVE_DATASET for evaluation
    print(f"ℹ️  Using Quest 3 dataset for evaluation")
    from src import config
    config.ACTIVE_DATASET = "quest3"

    try:
        # Run evaluation using the general evaluation function
        from src.evaluate import run_complete_evaluation
        results = run_complete_evaluation(dataset="quest3", model_path=model_path)

        if results:
            print(f"\n✅ Evaluation complete!")
            print(f"   🎯 Overall accuracy: {results.get('overall_accuracy', 'N/A'):.4f}")
            print(f"   🎯 Macro F1-score: {results.get('macro_f1', 'N/A'):.4f}")
            print(f"   🎯 Weighted F1-score: {results.get('weighted_f1', 'N/A'):.4f}")
            print(f"   ⚡ Throughput: {results.get('performance_metrics', {}).get('throughput_fps', 'N/A'):.1f} FPS")

            # Show per-class metrics
            per_class = results.get('per_class_metrics', {})
            if per_class:
                print(f"\n📊 Per-Class F1 Scores:")
                for class_name, metrics in per_class.items():
                    f1 = metrics.get('f1', 0)
                    support = metrics.get('support', 0)
                    print(f"   {class_name}: {f1:.3f} ({support} samples)")
        else:
            print("❌ Evaluation failed - no results returned")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original dataset setting
        config.ACTIVE_DATASET = original_dataset


def command_info(args: argparse.Namespace) -> None:
    """Handle info command."""
    print("\n" + "="*70)
    print("PREGEST - QUEST 3 GESTURE RECOGNITION")
    print("="*70)

    # Model info
    model = create_model(num_classes=8)  # Quest 3 has 8 classes
    total_params = count_parameters(model)

    print("\n📊 Model Information:")
    print(f"   Architecture: Multi-modal Transformer")
    print(f"   RGB Encoder: ResNet18 (pretrained)")
    print(f"   Mask Encoder: ResNet18 (random init)")
    print(f"   Fusion: Linear projection + LayerNorm + GELU")
    print(f"   Transformer: 4 layers, 8 heads, 512 feedforward")
    print(f"   Total Parameters: {total_params:,}")

    # Dataset info
    print("\n📁 Dataset Information:")
    print("   Name: Quest 3 Gesture Dataset")
    print("   Classes: 8 gestures")
    print("   Capture: Meta Quest 3 HMD")
    print("   Viewpoint: First-person (egocentric)")
    print("   Format: RGB + hand segmentation masks")

    print("\n🎯 Quest 3 Gesture Classes:")
    gesture_names = ["flat_palm_stop", "grab", "pinch_select", "release", "swipe_down", "swipe_left", "swipe_right", "swipe_up"]
    for i, name in enumerate(gesture_names):
        print(f"   {i}: {name}")

    # Training config
    print("\n⚙️  Training Configuration:")
    print("   Sequence Length: 30 frames")
    print("   Image Size: 224×224")
    print("   Batch Size: 2 (MPS optimized)")
    print("   Learning Rate: 1e-4 (AdamW)")
    print(f"   Epochs: 10 (optimized)")
    print(f"   Device: {DEVICE}")

    if torch.backends.mps.is_available():
        print("   GPU: Apple M4 MPS")
    elif torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Performance info
    print("\n🏆 Best Results:")
    print("   Test Accuracy: 92.52%")
    print("   Training Time: 4h 26m")

    # File paths
    print("\n📂 Important Paths:")
    print("   Raw Data: data/quest3/raw/")
    print("   Processed: data/quest3/processed/")
    print("   Models: models/")
    print("   Results: results/")
    print("   Logs: logs/")

    print("\n" + "="*70 + "\n")


def execute_workflow(args: argparse.Namespace) -> None:
    """Execute workflow based on global arguments."""
    dataset = args.dataset

    print(f"🚀 Starting PreGest workflow")
    print(f"   Dataset: {dataset}")
    print(f"   Mode: {args.mode}")
    if args.preprocess:
        print(f"   Preprocessing: Enabled")
    print()

    # Preprocessing
    if args.preprocess:
        print("="*70)
        print(f"PREPROCESSING {dataset.upper()} DATASET")
        print("="*70)

        if dataset == "quest3":
            try:
                stats = preprocess_quest3_dataset()
                print("✅ Quest 3 preprocessing completed!")
                print(f"   Videos processed: {stats['processed_videos']}")
                print(f"   Windows created: {stats['total_windows']}")
            except Exception as e:
                print(f"❌ Preprocessing failed: {e}")
                sys.exit(1)
        else:
            print("Only Quest 3 datasets are supported!")
            sys.exit(1)

    # Training
    if args.mode in ['train', 'both']:
        print("\n" + "="*70)
        print(f"TRAINING {dataset.upper()} MODEL")
        print("="*70)

        # Set default parameters based on dataset
        epochs = getattr(args, 'epochs', 50 if dataset == 'quest3' else 30)
        batch_size = getattr(args, 'batch_size', 16)
        learning_rate = getattr(args, 'learning_rate', 1e-4 if dataset == 'quest3' else 5e-5)

        try:
            history = train_model(
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            print("✅ Training completed!")
            print(f"   Final test accuracy: {history['test_acc']:.4f}")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            sys.exit(1)

    # Evaluation
    if args.mode in ['eval', 'both']:
        print("\n" + "="*70)
        print(f"EVALUATING {dataset.upper()} MODEL")
        print("="*70)

        try:
            results = run_complete_evaluation(dataset=dataset)
            if results:
                print("✅ Evaluation completed!")
                print(f"   Test accuracy: {results['overall_accuracy']:.4f}")
                print(f"   Macro F1: {results['macro_f1']:.4f}")
                print(f"   Throughput: {results['performance_metrics']['throughput_fps']:.1f} FPS")
            else:
                print("❌ Evaluation failed!")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            sys.exit(1)

    print("\n🎉 Workflow completed successfully!")


def main() -> None:
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Check if using new workflow mode (global arguments provided)
    if hasattr(args, 'dataset') and (args.preprocess or args.mode != 'both'):
        # New workflow mode
        execute_workflow(args)
        return

    # Legacy command mode
    if args.command is None:
        parser.print_help()
        return

    # Route to appropriate command handler
    command_handlers = {
        'preprocess': command_preprocess,

        'train': command_train,
        'evaluate': command_evaluate,
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
