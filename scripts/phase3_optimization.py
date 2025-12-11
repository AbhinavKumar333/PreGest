"""PreGest Phase 3: Production Optimization for Quest 3 Deployment"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import logging
import time
import numpy as np
import sys
from tqdm import tqdm
import onnxruntime as ort
import onnx
from onnxconverter_common import float16
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# PreGest imports
from src.model import create_model
from src.config import (
    QUEST3_GESTURES, NUM_QUEST3_CLASSES, MODEL_CONFIG,
    TRAIN_CONFIG, BEST_MODEL_PATH, RESULTS_DIR, DEVICE
)
from src.quest3_dataset import get_quest3_dataloaders
from src.utils import format_time

# New update
# ---- JSON SAFE CONVERSION ----
def to_json_safe(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, (np.int32, np.int64, np.longlong)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    return obj


class Phase3Optimizer:
    """Phase 3 optimization engine for Quest 3 deployment"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or RESULTS_DIR / 'quest3_phase2_best.pth'
        self.device = DEVICE
        self.results_dir = RESULTS_DIR / 'phase3_optimization'
        self.results_dir.mkdir(exist_ok=True)

        # Performance targets
        self.target_latency = 50  
        self.target_accuracy_drop = 0.05  

        # Initialize results tracking
        self.optimization_results = {
            'original': {},
            'quantized': {},
            'pruned': {},
            'onnx_fp32': {},
            'onnx_fp16': {},
            'tensorrt': {}
        }

    def load_baseline_model(self) -> torch.nn.Module:
        """Load the Phase 2 optimized model as baseline"""
        print("🔄 Loading Phase 2 optimized model...")

        # Load Phase 2 configuration
        phase2_config = MODEL_CONFIG.copy()
        phase2_config.update({
            'fusion_dim': 320,
            'hidden_dim': 288,
            'num_heads': 6,
            'feedforward_dim': 576,
            'dropout': 0.45,
            'num_classes': NUM_QUEST3_CLASSES
        })

        model = create_model(**{k: v for k, v in phase2_config.items()
                               if k in ['num_classes', 'backbone', 'rgb_pretrained',
                                       'mask_pretrained', 'fusion_dim', 'hidden_dim',
                                       'num_heads', 'num_layers', 'feedforward_dim', 'dropout']})

        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model from {self.model_path}")
        else:
            print(f"⚠️  Model not found at {self.model_path}, using randomly initialized weights")

        model.to(self.device)
        model.eval()
        return model

    def benchmark_model(self, model: torch.nn.Module, name: str,
                       num_runs: int = 100) -> Dict[str, float]:
        """Benchmark model performance (latency, memory, accuracy)"""
        print(f"📊 Benchmarking {name}...")

        # Check if model is quantized
        is_quantized = hasattr(model, '_modules') and any(
            isinstance(module, torch.nn.quantized.Linear) or isinstance(module, torch.nn.quantized.Conv2d)
            for module in model.modules()
        )

        if is_quantized and self.device.type == 'mps':
            print("   ⚠️  Quantized model detected - using CPU for benchmarking")
            benchmark_device = torch.device('cpu')
        else:
            benchmark_device = self.device

        # Get test data
        _, _, test_loader = get_quest3_dataloaders(batch_size=1, phase2_config=None)

        # Move model to appropriate device for benchmarking
        model = model.to(benchmark_device)

        # Warmup
        dummy_rgb = torch.randn(1, 30, 3, 224, 224).to(benchmark_device)
        dummy_mask = torch.randn(1, 30, 1, 224, 224).to(benchmark_device)

        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_rgb, dummy_mask)

        # Benchmark latency
        latencies = []
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        with torch.no_grad():
            for _ in tqdm(range(num_runs), desc=f"Benchmarking {name}"):
                start_time = time.time()
                _ = model(dummy_rgb, dummy_mask)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  

        # Benchmark accuracy
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for frames, labels in tqdm(test_loader, desc=f"Accuracy test {name}"):
                rgb_frames = frames[:, :, :3].to(benchmark_device)
                mask_frames = frames[:, :, 3:].to(benchmark_device)
                labels = labels.to(benchmark_device)

                outputs = model(rgb_frames, mask_frames)
                logits = outputs.mean(dim=1)
                predictions = logits.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total

        # Memory usage
        memory_usage = psutil.virtual_memory().percent
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  
        else:
            gpu_memory = 0

        results = {
            'latency_ms': np.mean(latencies),
            'latency_std': np.std(latencies),
            'accuracy': accuracy,
            'memory_percent': memory_usage,
            'gpu_memory_mb': gpu_memory,
            'model_size_mb': self.get_model_size(model)
        }

        print(f"   Latency: {results['latency_ms']:.2f}±{results['latency_std']:.2f}ms")
        print(f"   Accuracy: {results['accuracy']:.2f}%")
        print(f"   Memory: {results['memory_percent']:.1f}%, GPU: {results['gpu_memory_mb']:.1f}MB")

        self.optimization_results[name] = results
        return results

    def get_model_size(self, model: torch.nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2

    def apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply dynamic quantization to the model"""
        print("🔧 Applying dynamic quantization...")

        # Prepare model for quantization
        model.eval()

        # Handle MPS device limitation for quantization
        original_device = next(model.parameters()).device
        if original_device.type == 'mps':
            print("   ⚠️  MPS device detected - moving model to CPU for quantization")
            model = model.to('cpu')

        try:
            # Configure quantization backend
            torch.backends.quantized.engine = 'qnnpack'  

            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            print("✅ Model quantized successfully")

            # Keep quantized models on CPU as they don't support MPS operations
            print("   ✅ Quantized model kept on CPU (required for quantized operations)")

            return quantized_model

        except Exception as e:
            print(f"   ⚠️  Quantization failed ({type(e).__name__}): {str(e)[:100]}...")
            print("   💡 Skipping quantization - proceeding with other optimizations")
            return model.to(original_device)  

    def apply_pruning(self, model: torch.nn.Module, pruning_ratio: float = 0.3) -> torch.nn.Module:
        """Apply magnitude-based pruning to reduce model size"""
        print(f"🔧 Applying {pruning_ratio*100:.0f}% magnitude pruning...")

        try:
            # Use torch.nn.utils.prune if available 
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=pruning_ratio)

            # Remove pruning masks to make pruning permanent
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.nn.utils.prune.remove(module, 'weight')

            print("✅ Model pruned successfully")

        except AttributeError:
            # Fallback: simple weight zeroing for older PyTorch versions
            print("⚠️  Advanced pruning not available, using simple weight pruning...")

            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    # Get weight tensor
                    weight = module.weight.data
                    # Calculate threshold for pruning
                    threshold = torch.quantile(torch.abs(weight), pruning_ratio)
                    # Zero out small weights
                    mask = torch.abs(weight) < threshold
                    weight[mask] = 0.0

            print("✅ Simple weight pruning applied")

        return model

    def export_to_onnx(self, model: torch.nn.Module, precision: str = 'fp32') -> str:
        """Export model to ONNX format with specified precision"""
        print(f"🔧 Exporting model to ONNX ({precision})...")

        onnx_path = self.results_dir / f'quest3_model_{precision}.onnx'

        # Ensure model is on CPU for ONNX export
        model_device = next(model.parameters()).device
        if model_device.type != 'cpu':
            print("   ⚠️  Moving model to CPU for ONNX export")
            model = model.to('cpu')

        # Create dummy input
        dummy_rgb = torch.randn(1, 30, 3, 224, 224).to('cpu')
        dummy_mask = torch.randn(1, 30, 1, 224, 224).to('cpu')

        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_rgb, dummy_mask),
            onnx_path,
            input_names=['rgb_frames', 'mask_frames'],
            output_names=['logits'],
            dynamic_axes={
                'rgb_frames': {0: 'batch_size', 1: 'seq_len'},
                'mask_frames': {0: 'batch_size', 1: 'seq_len'},
                'logits': {0: 'batch_size', 1: 'seq_len'}
            },
            opset_version=14,  
            verbose=False
        )

        # Convert to FP16 if requested
        if precision == 'fp16':
            fp16_model = float16.convert_float_to_float16_model_path(str(onnx_path))
            onnx_path = onnx_path.with_suffix('.fp16.onnx')
            onnx.save(fp16_model, onnx_path)

        print(f"✅ Model exported to {onnx_path}")
        return str(onnx_path)

    def optimize_onnx_model(self, onnx_path: str) -> str:
        """Apply ONNX Runtime optimizations"""
        print("🔧 Optimizing ONNX model with ONNX Runtime...")

        # Create optimized ONNX model
        optimized_path = str(Path(onnx_path).with_suffix('.optimized.onnx'))

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = optimized_path

        # Create session to trigger optimization
        ort.InferenceSession(onnx_path, sess_options)

        print(f"✅ ONNX model optimized and saved to {optimized_path}")
        return optimized_path

    def benchmark_onnx_model(self, onnx_path: str, name: str,
                           num_runs: int = 100) -> Dict[str, float]:
        """Benchmark ONNX model performance"""
        print(f"📊 Benchmarking ONNX model {name}...")

        # Create ONNX session
        session = ort.InferenceSession(onnx_path)

        # Get test data
        _, _, test_loader = get_quest3_dataloaders(batch_size=1, phase2_config=None)

        # Benchmark latency
        latencies = []

        # Create dummy input for latency test
        dummy_rgb = np.random.randn(1, 30, 3, 224, 224).astype(np.float32)
        dummy_mask = np.random.randn(1, 30, 1, 224, 224).astype(np.float32)

        for _ in tqdm(range(num_runs), desc=f"ONNX latency {name}"):
            start_time = time.time()
            _ = session.run(None, {
                'rgb_frames': dummy_rgb,
                'mask_frames': dummy_mask
            })
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)

        # Benchmark accuracy
        correct = 0
        total = 0

        for frames, labels in tqdm(test_loader, desc=f"ONNX accuracy {name}"):
            rgb_frames = frames[:, :, :3].numpy()
            mask_frames = frames[:, :, 3:].numpy()

            outputs = session.run(None, {
                'rgb_frames': rgb_frames,
                'mask_frames': mask_frames
            })[0]

            logits = np.mean(outputs, axis=1)
            predictions = np.argmax(logits, axis=1)

            correct += np.sum(predictions == labels.numpy())
            total += labels.size(0)

        accuracy = correct / total

        results = {
            'latency_ms': np.mean(latencies),
            'latency_std': np.std(latencies),
            'accuracy': accuracy,
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory_mb': 0,  
            'model_size_mb': Path(onnx_path).stat().st_size / 1024**2
        }

        print(f"   Latency: {results['latency_ms']:.2f}±{results['latency_std']:.2f}ms")
        print(f"   Accuracy: {results['accuracy']:.2f}%")

        self.optimization_results[name] = results
        return results

    def run_phase3_optimization(self):
        """Execute complete Phase 3 optimization pipeline"""
        print("🚀 PHASE 3: PRODUCTION OPTIMIZATION FOR QUEST 3")

        # Step 1: Load and benchmark baseline model
        print("\n📊 STEP 1: BASELINE BENCHMARKING")
        baseline_model = self.load_baseline_model()
        baseline_results = self.benchmark_model(baseline_model, 'original')

        # Step 2: Skip quantization 
        print("\n🔧 STEP 2: QUANTIZATION OPTIMIZATION - SKIPPED")
        print("   ℹ️  Quantization has MPS compatibility issues on Apple Silicon")
        print("   💡 ONNX optimization provides better performance gains")
        quantized_results = {'latency_ms': float('inf'), 'accuracy': 0, 'memory_percent': 0, 'gpu_memory_mb': 0, 'model_size_mb': 0}

        # Step 3: Apply pruning
        print("\n🔧 STEP 3: PRUNING OPTIMIZATION")
        pruned_model = self.apply_pruning(baseline_model, 0.3)
        pruned_results = self.benchmark_model(pruned_model, 'pruned')

        # Step 4: Export to ONNX FP32
        print("\n🔧 STEP 4: ONNX EXPORT (FP32)")
        onnx_fp32_path = self.export_to_onnx(baseline_model, 'fp32')
        onnx_fp32_results = self.benchmark_onnx_model(onnx_fp32_path, 'onnx_fp32')

        # Step 5: Export ONNX FP16 
        print("\n🔧 STEP 5: ONNX EXPORT (FP16)")
        try:
            onnx_fp16_path = self.export_to_onnx(baseline_model, 'fp16')
            onnx_fp16_results = self.benchmark_onnx_model(onnx_fp16_path, 'onnx_fp16')
        except Exception as e:
            print(f"   ⚠️  FP16 ONNX export/optimization failed: {str(e)[:100]}...")
            print("   💡 Skipping FP16 optimization - using FP32 ONNX results instead")
            onnx_fp16_results = onnx_fp32_results.copy()
            onnx_fp16_results['latency_ms'] = float('inf')  

        # Step 6: Generate optimization report
        print("\n📊 STEP 6: OPTIMIZATION REPORT")
        self.generate_optimization_report()

        print("\n🎯 PHASE 3 OPTIMIZATION COMPLETE!")
        print(f"   Results saved to {self.results_dir}")

    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        report_path = self.results_dir / 'phase3_optimization_report.json'

        # Calculate improvements
        baseline = self.optimization_results.get('original', {})
        # Filter out results that don't have latency_ms or have invalid latency
        valid_results = [(name, results) for name, results in self.optimization_results.items()
                        if name != 'original' and 'latency_ms' in results and
                        results['latency_ms'] != float('inf')]

        if valid_results:
            best_optimized = min(valid_results, key=lambda x: x[1]['latency_ms'])
        else:
            # Fallback to original if no optimizations worked
            best_optimized = ('original', baseline)

        report = {
            'optimization_summary': {
                'baseline_latency_ms': baseline.get('latency_ms', 0),
                'baseline_accuracy': baseline.get('accuracy', 0),
                'best_optimized_method': best_optimized[0],
                'optimized_latency_ms': best_optimized[1]['latency_ms'],
                'optimized_accuracy': best_optimized[1]['accuracy'],
                'speedup_factor': baseline.get('latency_ms', 1) / best_optimized[1]['latency_ms'],
                'accuracy_drop': baseline.get('accuracy', 0) - best_optimized[1]['accuracy'],
                'meets_latency_target': best_optimized[1]['latency_ms'] < self.target_latency,
                'meets_accuracy_target': (baseline.get('accuracy', 0) - best_optimized[1]['accuracy']) < self.target_accuracy_drop
            },
            'detailed_results': self.optimization_results,
            'recommendations': self.get_optimization_recommendations(),
            'deployment_ready': self.check_deployment_readiness()
        }

        # New update
        # Convert everything to JSON-safe types
        safe_report = to_json_safe(report)

        with open(report_path, 'w') as f:
            json.dump(safe_report, f, indent=2)

        # Print summary
        print("\n🎯 OPTIMIZATION SUMMARY:")
        print(f"   Baseline latency: {report['optimization_summary']['baseline_latency_ms']:.1f}ms")
        print(f"   Optimized latency: {report['optimization_summary']['optimized_latency_ms']:.1f}ms")
        print(f"   Speedup factor: {report['optimization_summary']['speedup_factor']:.2f}x")
        print(f"   Accuracy drop: {report['optimization_summary']['accuracy_drop']:.2f}")
        print(f"   Meets latency target (<{self.target_latency}ms): {report['optimization_summary']['meets_latency_target']}")
        print(f"   Meets accuracy target (<{self.target_accuracy_drop*100:.0f}% drop): {report['optimization_summary']['meets_accuracy_target']}")
        print(f"   Deployment ready: {report['deployment_ready']}")

        return report

    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on results"""
        recommendations = []

        baseline = self.optimization_results.get('original', {})
        onnx_fp32 = self.optimization_results.get('onnx_fp32', {})
        onnx_fp16 = self.optimization_results.get('onnx_fp16', {})

        # Use FP16 if available, otherwise FP32
        best_onnx = onnx_fp16 if onnx_fp16.get('latency_ms', float('inf')) < float('inf') else onnx_fp32

        if best_onnx.get('latency_ms', float('inf')) < self.target_latency:
            precision = "FP16" if best_onnx == onnx_fp16 else "FP32"
            recommendations.append(f"✅ ONNX {precision} optimization achieves real-time performance target")
        else:
            recommendations.append("⚠️  Further optimization needed - consider TensorRT deployment or model distillation")

        accuracy_drop = baseline.get('accuracy', 0) - best_onnx.get('accuracy', 0)
        if accuracy_drop < self.target_accuracy_drop:
            recommendations.append("✅ Accuracy loss within acceptable range")
        else:
            recommendations.append("⚠️  Accuracy drop too high - consider fine-tuning quantized model")

        speedup = baseline.get('latency_ms', 1) / best_onnx.get('latency_ms', 1)
        if speedup > 5:
            recommendations.append(f"✅ Excellent speedup achieved ({speedup:.1f}x)")
        else:
            recommendations.append("🔧 Consider additional optimizations: pruning, distillation, or architecture changes")

        precision = "FP16" if best_onnx == onnx_fp16 else "FP32"
        recommendations.append(f"🚀 Recommended deployment: ONNX Runtime with {precision} precision on Quest 3")
        recommendations.append("☁️  Consider cloud-edge hybrid for complex gesture sequences")

        return recommendations

    def check_deployment_readiness(self) -> bool:
        """Check if optimized model is ready for deployment"""
        baseline = self.optimization_results.get('original', {})
        onnx_fp32 = self.optimization_results.get('onnx_fp32', {})
        onnx_fp16 = self.optimization_results.get('onnx_fp16', {})

        # Use FP16 if available and valid, otherwise FP32
        best_onnx = onnx_fp16 if onnx_fp16.get('latency_ms', float('inf')) < float('inf') else onnx_fp32

        latency_ok = best_onnx.get('latency_ms', float('inf')) < self.target_latency
        accuracy_ok = (baseline.get('accuracy', 0) - best_onnx.get('accuracy', 0)) < self.target_accuracy_drop

        return latency_ok and accuracy_ok

def main():
    """Execute Phase 3 optimizations"""
    print("🎯 PREGEST PHASE 3: QUEST 3 DEPLOYMENT OPTIMIZATION")

    optimizer = Phase3Optimizer()
    optimizer.run_phase3_optimization()

if __name__ == "__main__":
    main()
