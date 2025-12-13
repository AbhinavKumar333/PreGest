"""
PreGest Demo Web App - Gesture Recognition Interface
A Gradio-based web application for uploading Quest 3 gesture videos and getting predictions.
"""

import gradio as gr
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import BEST_MODEL_PATH, QUEST3_GESTURES, DEVICE, SEQUENCE_LENGTH, MODEL_CONFIG
from src.model import create_model
from src.quest3_preprocessor import extract_frames, generate_hand_mask


def load_model():
    """Load the trained gesture recognition model."""
    # Use the exact config that was used during training
    model = create_model(
        num_classes=MODEL_CONFIG['num_classes'],
        backbone=MODEL_CONFIG['backbone'],
        rgb_pretrained=MODEL_CONFIG['rgb_pretrained'],
        mask_pretrained=MODEL_CONFIG['mask_pretrained'],
        fusion_dim=MODEL_CONFIG['fusion_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_layers=MODEL_CONFIG['num_layers'],
        feedforward_dim=MODEL_CONFIG['feedforward_dim'],
        dropout=MODEL_CONFIG['dropout'],
        max_seq_len=60  # Override: checkpoint was trained with 60 frames
    )
    
    # Try to load the best model
    model_path = BEST_MODEL_PATH
    if not model_path.exists():
        # Fallback to phase2 model
        model_path = Path("results_new/phase3_optimization/quest3_model_fp32.onnx")
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print("Warning: No trained model found. Using random weights.")
    
    model.to(DEVICE)
    model.eval()
    return model



# Use 60 frames to match the trained model
MODEL_SEQ_LENGTH = 60

def preprocess_video(video_path: str) -> torch.Tensor:
    """Preprocess a video file into model input tensors."""
    # Extract frames
    frames = extract_frames(Path(video_path), fps=30)
    
    if len(frames) < MODEL_SEQ_LENGTH:
        print(f"Video too short: {len(frames)} frames. Need at least {MODEL_SEQ_LENGTH}.")
        # Pad with last frame if too short
        while len(frames) < MODEL_SEQ_LENGTH:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Take only the first MODEL_SEQ_LENGTH frames
    frames = frames[:MODEL_SEQ_LENGTH]
    
    rgb_tensors = []
    mask_tensors = []
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Generate hand mask
        mask = generate_hand_mask(frame)
        mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        # Normalize RGB
        rgb_normalized = resized.astype(np.float32) / 255.0
        rgb_normalized = (rgb_normalized - mean) / std
        rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).float()
        
        # Normalize Mask
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float() / 255.0
        
        rgb_tensors.append(rgb_tensor)
        mask_tensors.append(mask_tensor)
    
    # Stack into batch
    rgb_batch = torch.stack(rgb_tensors).unsqueeze(0)  # (1, T, 3, H, W)
    mask_batch = torch.stack(mask_tensors).unsqueeze(0)  # (1, T, 1, H, W)
    
    return rgb_batch, mask_batch


def predict_gesture(video_file):
    """Main prediction function for Gradio interface."""
    if video_file is None:
        return "Please upload a video file.", None
    
    try:
        # Load model (cached)
        global model
        if 'model' not in globals():
            model = load_model()
        
        # Preprocess video
        rgb_batch, mask_batch = preprocess_video(video_file)
        rgb_batch = rgb_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            logits = model(rgb_batch, mask_batch)
            
            # Handle different output shapes
            if logits.dim() == 3:  # (B, T, C) - frame-level predictions
                logits = logits.mean(dim=1)  # Average over time
            
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get gesture name
        gesture_name = QUEST3_GESTURES.get(predicted_class, f"Unknown ({predicted_class})")
        
        # Create probability dictionary for Gradio Label component
        prob_dict = {
            QUEST3_GESTURES[i]: float(probabilities[0, i].cpu())
            for i in range(len(QUEST3_GESTURES))
        }
        
        result_text = f"**Predicted Gesture**: {gesture_name}\n**Confidence**: {confidence:.1%}"
        
        return result_text, prob_dict
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        return error_msg, None


# Create Gradio Interface
with gr.Blocks(title="PreGest Demo") as demo:
    gr.Markdown(
        """
        # PreGest: Quest 3 Gesture Recognition Demo
        
        Upload a gesture video recorded from Meta Quest 3 and get real-time predictions.
        
        **Supported Gestures**: Flat Palm Stop, Grab, Pinch Select, Release, Swipe Down/Left/Right/Up
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Gesture Video")
            predict_btn = gr.Button("🔮 Predict Gesture", variant="primary")
        
        with gr.Column(scale=1):
            result_text = gr.Markdown(label="Prediction Result")
            prob_output = gr.Label(label="Class Probabilities", num_top_classes=8)
    
    # Example videos section (if available)
    gr.Markdown("---")
    gr.Markdown("### How to Use")
    gr.Markdown(
        """
        1. Record a gesture video using Meta Quest 3 (or use a sample video)
        2. Upload the MP4 file above
        3. Click **Predict Gesture** to see the classification result
        """
    )
    
    # Connect the button to the prediction function
    predict_btn.click(
        fn=predict_gesture,
        inputs=[video_input],
        outputs=[result_text, prob_output]
    )


if __name__ == "__main__":
    print("Starting PreGest Demo Web App...")
    print(f"Using device: {DEVICE}")
    
    # Pre-load model
    model = load_model()
    
    # Launch the app with public sharing
    demo.launch(
        share=True,  # Creates a temporary public link!
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
