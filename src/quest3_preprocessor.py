"""
Quest 3 Video Preprocessor for Gesture Recognition
Processes MP4 videos into model-ready tensors with hand segmentation.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

# Note: Using simplified skin color detection for now
# Full MMPose integration can be added later with proper model setup

from config import (
    QUEST3_RAW_DIR, QUEST3_PROCESSED_DIR, QUEST3_GESTURES,
    SEQUENCE_LENGTH, FRAME_SIZE, NUM_CHANNELS
)


def setup_logging():
    """Set up logging for preprocessing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def extract_frames(video_path: Path, fps: int = 30) -> List[np.ndarray]:
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to MP4 video file
        fps: Target frames per second

    Returns:
        List of RGB frames as numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame sampling interval
    if video_fps <= fps:
        # Video FPS is lower or equal, take all frames
        interval = 1
    else:
        # Sample frames to match target FPS
        interval = max(1, int(video_fps / fps))

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames at the calculated interval
        if frame_count % interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_count += 1

    cap.release()
    return frames


def resize_and_normalize_frames(frames: List[np.ndarray],
                               target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Resize frames and apply ImageNet normalization.

    Args:
        frames: List of RGB frames
        target_size: Target (height, width)

    Returns:
        Normalized tensor of shape (num_frames, 3, H, W)
    """
    if not frames:
        return torch.empty(0, 3, target_size[0], target_size[1])

    processed_frames = []

    # ImageNet normalization values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

        # Convert to float and normalize to [0, 1]
        resized = resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        normalized = (resized - mean) / std

        # Convert to tensor and transpose to (C, H, W)
        tensor_frame = torch.from_numpy(normalized).permute(2, 0, 1)
        processed_frames.append(tensor_frame)

    # Stack into tensor (N, 3, H, W)
    return torch.stack(processed_frames)


def generate_hand_mask(frame: np.ndarray) -> np.ndarray:
    """Generate binary hand mask using OpenMMLab MMPose.

    For now, this is a simplified implementation that creates masks
    based on skin color detection as a fallback until full MMPose
    integration is complete.

    Args:
        frame: RGB frame as numpy array

    Returns:
        Binary mask of shape (H, W) with hand pixels as 255, background as 0
    """
    try:
        # For now, use a simple skin color-based detection
        # This is a temporary implementation until MMPose is properly set up

        # Convert RGB to HSV for better skin color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask for skin-colored pixels
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        return skin_mask

    except Exception as e:
        print(f"Warning: Hand mask generation failed: {e}")
        # Fallback: return empty mask
        h, w = frame.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)


def process_video_frames(frames: List[np.ndarray],
                        logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Process frames to create RGB tensors and hand masks.

    Args:
        frames: List of RGB frames
        logger: Logger for warnings

    Returns:
        Tuple of (rgb_tensor, mask_tensor, valid_frames_count)
    """
    if not frames:
        return torch.empty(0, 3, 224, 224), torch.empty(0, 1, 224, 224), 0

    rgb_frames = []
    mask_frames = []
    valid_count = 0

    for i, frame in enumerate(frames):
        try:
            # Generate hand mask
            mask = generate_hand_mask(frame)

            # Resize mask to 224x224
            mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

            # Convert mask to tensor (1, H, W)
            mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float() / 255.0

            # Resize and normalize RGB frame
            rgb_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0

            # Apply ImageNet normalization to RGB
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_tensor = (rgb_tensor - mean) / std

            rgb_frames.append(rgb_tensor)
            mask_frames.append(mask_tensor)
            valid_count += 1

        except Exception as e:
            logger.warning(f"Failed to process frame {i}: {e}")
            continue

    if not rgb_frames:
        return torch.empty(0, 3, 224, 224), torch.empty(0, 1, 224, 224), 0

    # Stack frames
    rgb_tensor = torch.stack(rgb_frames)  # (N, 3, 224, 224)
    mask_tensor = torch.stack(mask_frames)  # (N, 1, 224, 224)

    return rgb_tensor, mask_tensor, valid_count


def create_temporal_windows(rgb_frames: torch.Tensor,
                           mask_frames: torch.Tensor,
                           window_size: int = 60,
                           stride: int = 15,
                           max_missing_ratio: float = 0.1) -> List[torch.Tensor]:
    """Create sliding temporal windows from frame sequences.

    Args:
        rgb_frames: RGB frames tensor (N, 3, H, W)
        mask_frames: Mask frames tensor (N, 1, H, W)
        window_size: Number of frames per window
        stride: Stride between windows
        max_missing_ratio: Maximum ratio of frames without hands

    Returns:
        List of window tensors (window_size, 4, H, W)
    """
    if len(rgb_frames) < window_size:
        return []

    windows = []
    num_frames = len(rgb_frames)

    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size

        # Extract window frames
        rgb_window = rgb_frames[start:end]    # (60, 3, 224, 224)
        mask_window = mask_frames[start:end]  # (60, 1, 224, 224)

        # Count frames with hand detection (mask sum > 0)
        hand_frames = (mask_window.sum(dim=[1, 2, 3]) > 0).sum().item()
        missing_ratio = 1.0 - (hand_frames / window_size)

        # Skip windows with too many missing hands
        if missing_ratio > max_missing_ratio:
            continue

        # Concatenate RGB and mask channels: (60, 4, 224, 224)
        window = torch.cat([rgb_window, mask_window], dim=1)
        windows.append(window)

    return windows


def process_single_video(video_path: Path,
                        gesture_id: int,
                        logger: logging.Logger) -> Tuple[List[torch.Tensor], int]:
    """Process a single video into temporal windows.

    Args:
        video_path: Path to video file
        gesture_id: Gesture class ID
        logger: Logger instance

    Returns:
        Tuple of (windows_list, valid_frames_count)
    """
    try:
        # Extract frames
        frames = extract_frames(video_path)
        if not frames:
            logger.warning(f"SKIPPED {video_path.name}: No frames extracted")
            return [], 0

        # Process frames into RGB and masks
        rgb_tensor, mask_tensor, valid_frames = process_video_frames(frames, logger)

        if valid_frames < SEQUENCE_LENGTH:
            logger.warning(f"SKIPPED {video_path.name}: Only {valid_frames} valid frames (need {SEQUENCE_LENGTH})")
            return [], valid_frames

        # Create temporal windows (use SEQUENCE_LENGTH as window size)
        windows = create_temporal_windows(rgb_tensor, mask_tensor, window_size=SEQUENCE_LENGTH, max_missing_ratio=1.0)

        if not windows:
            logger.warning(f"SKIPPED {video_path.name}: No valid windows created ({valid_frames} frames)")
            return [], valid_frames

        # Log success
        logger.info(f"PROCESSED {video_path.name}: {valid_frames} frames -> {len(windows)} windows")

        return windows, valid_frames

    except Exception as e:
        logger.error(f"SKIPPED {video_path.name}: Processing failed - {e}")
        return [], 0


def collect_video_files() -> Dict[str, Dict[int, List[Path]]]:
    """Collect all video files organized by split and gesture.

    Returns:
        Dict with structure: {split: {gesture_id: [video_paths]}}
    """
    video_files = {'Train': {}, 'Test': {}}

    # Process train and test directories
    for split in ['Train', 'Test']:
        split_dir = QUEST3_RAW_DIR / split
        # Convert to lowercase for internal use
        split_key = split.lower()

        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")

        for gesture_id, gesture_name in QUEST3_GESTURES.items():
            gesture_dir = split_dir / gesture_name

            if not gesture_dir.exists():
                print(f"Warning: Gesture directory not found: {gesture_dir}")
                video_files[split][gesture_id] = []
                continue

            # Find all MP4/mp4 files (case insensitive)
            mp4_files = list(gesture_dir.glob("*.MP4")) + list(gesture_dir.glob("*.mp4"))
            video_files[split][gesture_id] = mp4_files

    return video_files


def create_data_split(video_files: Dict[str, List[Path]],
                     train_ratio: float = 0.8,
                     seed: int = 42) -> Dict[str, List[str]]:
    """Create train/val/test splits from video files.

    Args:
        video_files: Dict of {gesture_id: [video_paths]}
        train_ratio: Ratio of videos to use for training
        seed: Random seed

    Returns:
        Dict with video IDs for each split
    """
    random.seed(seed)
    splits = {'train': [], 'val': [], 'test': []}

    # Process each gesture class separately
    for gesture_id, videos in video_files.items():
        if not videos:
            continue

        # Shuffle videos
        shuffled_videos = videos.copy()
        random.shuffle(shuffled_videos)

        # Split into train and val
        n_train = int(len(shuffled_videos) * train_ratio)
        train_videos = shuffled_videos[:n_train]
        val_videos = shuffled_videos[n_train:]

        # Add video IDs to splits
        for video_path in train_videos:
            splits['train'].append(video_path.stem)
        for video_path in val_videos:
            splits['val'].append(video_path.stem)

    # Test videos remain as-is
    test_videos = []
    for gesture_videos in video_files.values():
        for video_path in gesture_videos:
            test_videos.append(video_path.stem)
    splits['test'] = test_videos

    return splits


def save_splits_metadata(splits: Dict[str, List[str]], output_path: Path):
    """Save data splits to JSON file.

    Args:
        splits: Dict with video IDs for each split
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)


def preprocess_quest3_dataset() -> Dict[str, any]:
    """Main preprocessing function for Quest 3 dataset.

    Returns:
        Dict with preprocessing statistics
    """
    logger = setup_logging()
    logger.info("Starting Quest 3 dataset preprocessing")

    try:
        # Create output directory
        QUEST3_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Collect video files
        logger.info("Collecting video files...")
        all_videos = collect_video_files()

        # Count total videos
        total_videos = sum(len(videos) for split_videos in all_videos.values()
                          for videos in split_videos.values())
        logger.info(f"Found {total_videos} videos across all splits")

        # Create data splits
        logger.info("Creating train/val splits...")
        train_videos = {}
        for gesture_id, videos in all_videos['Train'].items():
            train_videos[gesture_id] = videos

        splits = create_data_split(train_videos)
        splits['test'] = [Path(video).stem for videos in all_videos['Test'].values() for video in videos]

        # Save splits metadata
        splits_path = QUEST3_PROCESSED_DIR / "splits.json"
        save_splits_metadata(splits, splits_path)
        logger.info(f"Saved splits metadata to {splits_path}")

        # Process videos with streaming chunked saving to avoid memory issues
        logger.info("Processing videos into temporal windows (streaming mode)...")

        # Debug: Log video structure
        logger.info("Video collection debug:")
        for split, gesture_dict in all_videos.items():
            gesture_count = {}
            for gesture_id, videos in gesture_dict.items():
                gesture_count[gesture_id] = len(videos)
            logger.info(f"  {split}: {gesture_count}")
        logger.info(f"Total: {total_videos} videos")

        # Create video ID to split mapping
        video_to_split = {}
        for split, video_ids in splits.items():
            for video_id in video_ids:
                video_to_split[video_id] = split

        # Streaming processing: process videos in batches and save immediately
        chunk_size = 50  # Process in small batches

        # Process each split separately with streaming
        for split in ['train', 'val', 'test']:
            logger.info(f"Processing {split.upper()} split...")
            chunk_idx = 0
            processed_videos = 0
            split_windows = 0

            # Process all videos, but only save those belonging to current split
            for split_name in ['Train', 'Test']:
                for gesture_id, videos in all_videos[split_name].items():
                    for video_path in videos:
                        # Determine which split this video belongs to
                        video_id = video_path.stem
                        actual_split = video_to_split.get(video_id, split_name.lower())

                        if actual_split != split:
                            continue  # Skip videos for other splits

                        # Process video immediately
                        windows, valid_frames = process_single_video(video_path, gesture_id, logger)

                        if windows:
                            # Save this single video's windows in chunks
                            for i in range(0, len(windows), chunk_size):
                                end_idx = min(i + chunk_size, len(windows))
                                chunk_windows = windows[i:end_idx]
                                chunk_labels = [gesture_id] * len(chunk_windows)

                                buffer_data = {'windows': chunk_windows, 'labels': chunk_labels}
                                _save_chunk(buffer_data, split, chunk_idx)
                                chunk_idx += 1

                            processed_videos += 1
                            split_windows += len(windows)

                            if processed_videos % 5 == 0:
                                logger.info(f"{split.upper()}: {processed_videos} videos → {split_windows} windows")

            logger.info(f"Completed {split.upper()}: {processed_videos} videos, {split_windows} windows in {chunk_idx} chunks")

        return {
            'total_videos': total_videos,
            'processed_videos': 'streaming_complete',
            'total_windows': 'chunked_storage',
            'chunks_saved': 'variable_per_split'
        }

    except Exception as e:
        logger.error(f"Preprocessing failed with error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback:\\n{traceback.format_exc()}")
        raise  # Re-raise for caller to handle

    finally:
        logger.info("Preprocessing completed.")

def _save_chunk(buffer_data, split, chunk_idx):
    """Save a chunk of windows and labels to disk."""
    windows_tensor = torch.stack(buffer_data['windows'])
    labels_tensor = torch.tensor(buffer_data['labels'], dtype=torch.long)

    windows_path = QUEST3_PROCESSED_DIR / f"{split}_windows_chunk_{chunk_idx}.pt"
    labels_path = QUEST3_PROCESSED_DIR / f"{split}_labels_chunk_{chunk_idx}.pt"

    torch.save(windows_tensor, windows_path)
    torch.save(labels_tensor, labels_path)

    # Update metadata file
    metadata_path = QUEST3_PROCESSED_DIR / f"{split}_chunks_metadata.json"
    existing_metadata = {'num_chunks': 0, 'chunk_size': len(buffer_data['windows']), 'total_windows': 0}

    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
        except:
            pass  # Use defaults if file corrupted

    existing_metadata['num_chunks'] = max(existing_metadata['num_chunks'], chunk_idx + 1)
    existing_metadata['total_windows'] += len(buffer_data['windows'])

    with open(metadata_path, 'w') as f:
        json.dump(existing_metadata, f)

    chunk_size_mb = (windows_tensor.numel() * 4) // (1024**2)  # Rough estimate in MB
    print(f"💾 Saved {split} chunk {chunk_idx}: {len(windows_tensor)} windows (~{chunk_size_mb}MB)")


if __name__ == "__main__":
    # Run preprocessing when script is executed directly
    preprocess_quest3_dataset()
