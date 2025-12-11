"""PreGest Phase 3: Deployment Configuration for Quest 3"""

from pathlib import Path
import json
from typing import Dict, List, Optional

# Deployment targets
DEPLOYMENT_TARGETS = {
    'quest3_local': {
        'platform': 'quest3',
        'inference_engine': 'onnx_fp16',
        'compute_device': 'cpu',  
        'latency_target_ms': 50,
        'memory_limit_mb': 256,
        'model_format': 'onnx',
        'precision': 'fp16',
        'batch_size': 1,
        'warmup_runs': 5,
    },
    'quest3_cloud': {
        'platform': 'cloud',
        'inference_engine': 'tensorrt',
        'compute_device': 'gpu',
        'latency_target_ms': 10,
        'memory_limit_mb': 1024,
        'model_format': 'tensorrt',
        'precision': 'fp16',
        'batch_size': 8,
        'warmup_runs': 10,
    },
    'webxr_browser': {
        'platform': 'web',
        'inference_engine': 'onnx_webgl',
        'compute_device': 'webgl',
        'latency_target_ms': 100,
        'memory_limit_mb': 512,
        'model_format': 'onnx',
        'precision': 'fp16',
        'batch_size': 1,
        'warmup_runs': 3,
    }
}

# Cloud-Edge Hybrid Configuration
HYBRID_CONFIG = {
    'edge_processing': {
        'feature_extraction': True,
        'temporal_smoothing': True,
        'confidence_threshold': 0.7,
        'cache_size': 10,
        'local_model_size_mb': 50,
    },
    'cloud_processing': {
        'complex_gestures': ['grab', 'pinch_select'],
        'uncertain_predictions': True,
        'batch_processing': True,
        'cloud_model_size_mb': 200,
        'api_endpoint': 'https://api.pregest.cloud/inference',
        'timeout_ms': 200,
    },
    'network_optimization': {
        'compression': 'gzip',
        'protocol': 'websocket',
        'heartbeat_interval_ms': 1000,
        'reconnection_attempts': 3,
    }
}

# Model Optimization Profiles
OPTIMIZATION_PROFILES = {
    'latency_optimized': {
        'priority': 'speed',
        'precision': 'fp16',
        'pruning_ratio': 0.4,
        'quantization': 'dynamic_int8',
        'attention_optimization': 'linear',
        'target_latency_ms': 30,
    },
    'accuracy_optimized': {
        'priority': 'accuracy',
        'precision': 'fp32',
        'pruning_ratio': 0.1,
        'quantization': None,
        'attention_optimization': 'full',
        'target_accuracy_min': 0.90,
    },
    'balanced': {
        'priority': 'balanced',
        'precision': 'fp16',
        'pruning_ratio': 0.25,
        'quantization': 'dynamic_int8',
        'attention_optimization': 'sparse',
        'target_latency_ms': 50,
        'target_accuracy_min': 0.92,
    }
}

class DeploymentConfig:
    """Configuration manager for PreGest deployment"""

    def __init__(self, target_platform: str = 'quest3_local'):
        self.target_platform = target_platform
        self.config = DEPLOYMENT_TARGETS.get(target_platform, DEPLOYMENT_TARGETS['quest3_local'])
        self.hybrid_config = HYBRID_CONFIG
        self.optimization_profile = 'balanced'

    def get_inference_config(self) -> Dict:
        """Get inference configuration for current deployment target"""
        return {
            'engine': self.config['inference_engine'],
            'device': self.config['compute_device'],
            'precision': self.config['precision'],
            'batch_size': self.config['batch_size'],
            'warmup_runs': self.config['warmup_runs'],
            'memory_limit': self.config['memory_limit_mb'],
            'latency_target': self.config['latency_target_ms'],
        }

    def get_model_config(self) -> Dict:
        """Get model configuration for current optimization profile"""
        profile = OPTIMIZATION_PROFILES[self.optimization_profile]
        return {
            'precision': profile['precision'],
            'pruning_ratio': profile['pruning_ratio'],
            'quantization': profile['quantization'],
            'attention_optimization': profile['attention_optimization'],
        }

    def get_hybrid_config(self) -> Dict:
        """Get cloud-edge hybrid configuration"""
        return self.hybrid_config

    def update_for_performance(self, current_latency: float, current_accuracy: float):
        """Dynamically adjust configuration based on performance metrics"""

        # Auto-tune optimization profile
        if current_latency > self.config['latency_target_ms'] * 1.5:
            # Too slow, switch to latency optimization
            if self.optimization_profile != 'latency_optimized':
                print(f"⚡ Switching to latency optimization (current: {current_latency:.1f}ms)")
                self.optimization_profile = 'latency_optimized'

        elif current_accuracy < 0.85:
            # Too inaccurate, switch to accuracy optimization
            if self.optimization_profile != 'accuracy_optimized':
                print(f"🎯 Switching to accuracy optimization (current: {current_accuracy:.3f})")
                self.optimization_profile = 'accuracy_optimized'

        else:
            # Performance acceptable, use balanced approach
            if self.optimization_profile != 'balanced':
                print("⚖️  Using balanced optimization")
                self.optimization_profile = 'balanced'

    def save_config(self, filepath: str):
        """Save current configuration to file"""
        config_data = {
            'target_platform': self.target_platform,
            'deployment_config': self.config,
            'optimization_profile': self.optimization_profile,
            'hybrid_config': self.hybrid_config,
            'inference_config': self.get_inference_config(),
            'model_config': self.get_model_config(),
        }

        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"✅ Deployment configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: str) -> 'DeploymentConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)

        instance = cls(config_data['target_platform'])
        instance.optimization_profile = config_data.get('optimization_profile', 'balanced')
        instance.hybrid_config = config_data.get('hybrid_config', HYBRID_CONFIG)

        return instance

# Quest 3 Specific Optimizations
QUEST3_OPTIMIZATIONS = {
    'gesture_pipeline': {
        'frame_rate': 30,  
        'gesture_window_ms': 1000,  
        'confidence_smoothing': True,
        'temporal_filtering': 'exponential_moving_average',
        'spatial_stabilization': True,
    },
    'hand_tracking_integration': {
        'api_version': 'v1.1',
        'joint_count': 21,  
        'coordinate_system': 'local',  
        'occlusion_handling': 'interpolation',
        'gesture_trigger_distance': 0.1,  
    },
    'performance_monitoring': {
        'metrics_collection': True,
        'latency_tracking': True,
        'memory_monitoring': True,
        'gesture_success_rate': True,
        'report_interval_ms': 5000,  
    }
}

def create_production_deployment_package(model_path: str, config_path: str,
                                       output_dir: str = 'deployment_package'):
    """Create a production deployment package for Quest 3."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Copy optimized model
    import shutil
    shutil.copy(model_path, output_path / 'model.onnx')

    # Save deployment configuration
    config = DeploymentConfig()
    config.save_config(str(output_path / 'deployment_config.json'))

    # Create deployment manifest
    manifest = {
        'name': 'PreGest_Quest3',
        'version': '3.0.0',
        'platform': 'quest3',
        'model_format': 'onnx_fp16',
        'dependencies': ['onnxruntime'],
        'entry_point': 'gesture_recognition.py',
        'configuration': config.get_inference_config(),
        'optimizations': QUEST3_OPTIMIZATIONS,
    }

    with open(output_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)



    print(f"✅ Production deployment package created at {output_path}")
    return str(output_path)

if __name__ == "__main__":
    config = DeploymentConfig('quest3_local')
    print("🚀 Quest 3 Deployment Configuration:")
    print(json.dumps(config.get_inference_config(), indent=2))

    # Create sample deployment package
    create_production_deployment_package(
        model_path='results/phase3_optimization/quest3_model_fp16.optimized.onnx',
        config_path='src/deployment_config.py',
        output_dir='quest3_deployment'
    )
