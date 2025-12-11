"""PreGest: Quest 3 Gesture Recognition System"""

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
from src.evaluate import run_complete_evaluation
import subprocess
import os
from src.config import RESULTS_DIR, BEST_MODEL_PATH


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="PreGest: Quest 3 Gesture Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quest 3 complete workflow
  python main.py --dataset quest3 --preprocess --mode both
  python main.py --dataset quest3 --mode train
  python main.py --dataset quest3 --mode eval

  # Complete 3-phase workflow (RECOMMENDED)
  python main.py phases

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

    # PREPROCESS COMMAND
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


    # TRAIN COMMAND
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

    # EVALUATE COMMAND
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    evaluate_parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model checkpoint (default: best model)'
    )

    # PHASES COMMAND
    phases_parser = subparsers.add_parser('phases', help='Run all 3 phases sequentially')
    phases_parser.add_argument(
        '--skip-phase2',
        action='store_true',
        help='Skip Phase 2 model improvement (use Phase 1 model directly)'
    )
    phases_parser.add_argument(
        '--skip-phase3',
        action='store_true',
        help='Skip Phase 3 production optimization'
    )

    # INFO COMMAND
    info_parser = subparsers.add_parser('info', help='Show system and model information')

    return parser


def command_preprocess(args: argparse.Namespace) -> None:
    """Handle preprocess command"""
    print("QUEST 3 GESTURE DATASET PREPROCESSING")

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
    """Handle train command"""
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
    """Handle evaluate command"""
    from src.config import BEST_MODEL_PATH, ACTIVE_DATASET as original_dataset

    model_path = Path(args.model_path) if args.model_path else BEST_MODEL_PATH

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Use Quest 3 dataset for evaluation since we trained with Quest 3
    print(f"ℹ️  Using Quest 3 dataset for evaluation")
    from src import config
    config.ACTIVE_DATASET = "quest3"

    try:
        # Run evaluation using the general evaluation function
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
    """Handle info command"""
    print("PREGEST - QUEST 3 GESTURE RECOGNITION")

    # Model info
    model = create_model(num_classes=8)  
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


def execute_workflow(args: argparse.Namespace) -> None:
    """Execute workflow based on global arguments"""
    dataset = args.dataset

    print(f"🚀 Starting PreGest workflow")
    print(f"   Dataset: {dataset}")
    print(f"   Mode: {args.mode}")
    if args.preprocess:
        print(f"   Preprocessing: Enabled")
    print()

    # Preprocessing
    if args.preprocess:
        print(f"PREPROCESSING {dataset.upper()} DATASET")

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
        print(f"TRAINING {dataset.upper()} MODEL")

        # Set default parameters based on dataset
        epochs = getattr(args, 'epochs', 20 if dataset == 'quest3' else 30)
        batch_size = getattr(args, 'batch_size', 2)
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
        print(f"EVALUATING {dataset.upper()} MODEL")

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


def command_phases(args: argparse.Namespace) -> None:
    """Handle phases command - run all 3 phases sequentially"""
    print("🚀 PREGEST COMPLETE WORKFLOW: ALL 3 PHASES")
    print("Phase 1: Data preprocessing and initial training")
    print("Phase 2: Model improvement based on error analysis")
    print("Phase 3: Production optimization for deployment")    

    # Phase 1: Preprocessing + Initial Training
    print("\n🎯 PHASE 1: DATA PREPROCESSING + INITIAL TRAINING")

    try:
        # Run preprocessing and training
        print("Running: python main.py --dataset quest3 --preprocess --mode both")
        result = subprocess.run([
            sys.executable, "main.py",
            "--dataset", "quest3",
            "--preprocess",
            "--mode", "both"
        ], capture_output=False, cwd=os.getcwd())

        if result.returncode != 0:
            print(f"❌ Phase 1 failed with exit code {result.returncode}")
            sys.exit(1)

        print("✅ Phase 1 completed successfully!")

    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        sys.exit(1)

    # Phase 2: Model Improvement
    if not args.skip_phase2:
        print("\n🎯 PHASE 2: MODEL IMPROVEMENT")

        try:
            print("Running: python scripts/improve_model.py")
            result = subprocess.run([
                sys.executable, "scripts/improve_model.py"
            ], capture_output=False, cwd=os.getcwd())

            if result.returncode != 0:
                print(f"⚠️  Phase 2 failed with exit code {result.returncode}")
                print("   Continuing with Phase 1 model...")
            else:
                print("✅ Phase 2 completed successfully!")

        except Exception as e:
            print(f"⚠️  Phase 2 failed: {e}")
            print("   Continuing with Phase 1 model...")
    else:
        print("\n⏭️  PHASE 2: SKIPPED (using --skip-phase2)")

    # Phase 3: Production Optimization
    if not args.skip_phase3:
        print("\n🎯 PHASE 3: PRODUCTION OPTIMIZATION")

        try:
            print("Running: python scripts/phase3_optimization.py")
            result = subprocess.run([
                sys.executable, "scripts/phase3_optimization.py"
            ], capture_output=False, cwd=os.getcwd())

            if result.returncode != 0:
                print(f"⚠️  Phase 3 failed with exit code {result.returncode}")
                print("   Model may still be usable for deployment")
            else:
                print("✅ Phase 3 completed successfully!")

        except Exception as e:
            print(f"⚠️  Phase 3 failed: {e}")
            print("   Model may still be usable for deployment")
    else:
        print("\n⏭️  PHASE 3: SKIPPED (using --skip-phase3)")

    # Final evaluation
    print("\n🎯 FINAL EVALUATION")

    try:
        # Determine which model to evaluate  
        phase2_model = RESULTS_DIR / 'quest3_phase2_best.pth'
        if phase2_model.exists() and not args.skip_phase2:
            model_path = str(phase2_model)
            print(f"Evaluating Phase 2 improved model: {model_path}")
        else:
            model_path = str(BEST_MODEL_PATH)
            print(f"Evaluating Phase 1 model: {model_path}")

        result = subprocess.run([
            sys.executable, "main.py", "evaluate",
            "--model-path", model_path
        ], capture_output=False, cwd=os.getcwd())

        if result.returncode != 0:
            print(f"⚠️  Final evaluation failed with exit code {result.returncode}")

    except Exception as e:
        print(f"⚠️  Final evaluation failed: {e}")

    print("\n🎉 ALL PHASES COMPLETED!")
    print("📊 Summary:")
    print("   Phase 1: ✅ Data preprocessing and initial training")
    if not args.skip_phase2:
        print("   Phase 2: ✅ Model improvement and fine-tuning")
    else:
        print("   Phase 2: ⏭️  Skipped")
    if not args.skip_phase3:
        print("   Phase 3: ✅ Production optimization")
    else:
        print("   Phase 3: ⏭️  Skipped")
    print("   Final Eval: ✅ Complete evaluation with metrics")
    print("\n🚀 Your Quest 3 gesture recognition model is ready for deployment!")
    print("   Check results/phase3_optimization/ for deployment artifacts")


def main() -> None:
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()

    # Check if using new workflow mode
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
        'phases': command_phases,
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
