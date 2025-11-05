"""
Dataset preprocessing pipeline for Jester gesture recognition.
Processes static image files (NOT videos) to extract hand landmarks
and create temporal sequences.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
import logging

# Optional mediapipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

from .config import (
    JESTER_RAW_DIR, JESTER_PROCESSED_DIR, DATA_CONFIG,
    TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, NUM_CLASSES
)
from .utils import setup_logging, ensure_directory_exists


class GestureDataset(torch.utils.data.Dataset):
    """PyTorch dataset for gesture recognition."""
    
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor):
        """Initialize dataset.
        
        Args:
            sequences: Tensor of shape [N, seq_len, input_dim].
            labels: Tensor of shape [N] with gesture class labels.
        """
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.sequences[idx], self.labels[idx]


def setup_jester_dataset() -> bool:
    """Setup Jester dataset with clear instructions."""
    logger = setup_logging()
    
    if JESTER_RAW_DIR.exists() and len(list(JESTER_RAW_DIR.glob('*'))) > 0:
        logger.info("✓ Jester dataset already exists")
        return True
    
    logger.warning("\n" + "="*70)
    logger.warning("JESTER DATASET SETUP REQUIRED")
    logger.warning("="*70)
    logger.warning("\nThe Jester dataset contains IMAGE files organized by gesture class.")
    logger.warning("Expected structure:")
    logger.warning(f"  {JESTER_RAW_DIR}/")
    logger.warning("  ├── gesture_0/")
    logger.warning("  │   ├── image_001.jpg")
    logger.warning("  │   ├── image_002.jpg")
    logger.warning("  │   └── ...")
    logger.warning("  ├── gesture_1/")
    logger.warning("  │   └── ...")
    logger.warning("  └── ... (25 gesture classes total)\n")
    
    logger.warning("Option 1: Automatic Download (Requires Kaggle API)")
    logger.warning("  1. pip install kaggle")
    logger.warning("  2. Place your kaggle.json in ~/.kaggle/")
    logger.warning("  3. Run: dataset.download_and_extract_jester()")
    logger.warning("\nOption 2: Manual Download")
    logger.warning("  1. Visit: https://www.kaggle.com/datasets/toxicmender/20bn-jester")
    logger.warning("  2. Download and extract the dataset")
    logger.warning(f"  3. Move extracted images to: {JESTER_RAW_DIR}")
    logger.warning("="*70 + "\n")
    
    return False


def download_and_extract_jester(output_dir: Path = JESTER_RAW_DIR) -> bool:
    """Download Jester dataset using Kaggle API."""
    logger = setup_logging()
    
    try:
        import kaggle
        logger.info("Downloading Jester dataset from Kaggle...")
        
        kaggle.api.dataset_download_files(
            'toxicmender/20bn-jester',
            path=output_dir,
            unzip=True
        )
        logger.info(f"✓ Dataset downloaded to {output_dir}")
        return True
    except ImportError:
        logger.error("Kaggle API not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def extract_hand_landmarks_from_images(
    image_dir: Path = JESTER_RAW_DIR,
    max_videos: Optional[int] = None
) -> Dict[int, List[np.ndarray]]:
    """
    Extract hand landmarks from IMAGE files using MediaPipe.

    Args:
        image_dir: Directory containing video subdirectories with images

    Returns:
        Dict mapping gesture_id to list of [21, 3] landmark arrays
    """
    logger = setup_logging()
    logger.info("Extracting hand landmarks from images...")

    # Check if this is the standard Jester structure (20bn-jester-v1/)
    jester_dir = image_dir / "20bn-jester-v1"
    if jester_dir.exists():
        logger.info("Found Jester dataset structure, using video directories")
        video_dirs = sorted([d for d in jester_dir.iterdir() if d.is_dir() and d.name.isdigit()])

        # Use specified max_videos or default to all videos for full training
        if max_videos is None:
            max_videos = len(video_dirs)  # Default to all videos
        max_videos = min(max_videos, len(video_dirs))
        video_dirs = video_dirs[:max_videos]

        if max_videos == len(video_dirs):
            logger.info(f"Processing {len(video_dirs)} videos for full training")
        else:
            logger.info(f"Processing {len(video_dirs)} videos for fast testing (limited by --max-videos={max_videos})")
        logger.info("💡 Tip: Use --max-videos 50 for very fast testing, or omit for full dataset")
    else:
        # Fallback to gesture-based structure
        video_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir()])
        logger.info(f"Using gesture-based structure with {len(video_dirs)} directories")

    if not MEDIAPIPE_AVAILABLE:
        logger.warning("MediaPipe not available - generating mock landmark data for testing")
        return _generate_mock_landmarks_from_videos(video_dirs)

    # Initialize MediaPipe Hands for STATIC IMAGE mode
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,  # CRITICAL: Use True for images
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    landmarks_by_gesture = {}

    for video_id, video_dir in enumerate(video_dirs):
        # Assign mock gesture labels for testing (cycle through available gestures)
        gesture_id = video_id % NUM_CLASSES
        if gesture_id not in landmarks_by_gesture:
            landmarks_by_gesture[gesture_id] = []

        # Find all image files (sorted by frame number)
        image_files = sorted(video_dir.glob('*.jpg')) + sorted(video_dir.glob('*.png'))

        if len(image_files) == 0:
            logger.warning(f"No images found in {video_dir.name}")
            continue

        for image_path in image_files:
            try:
                # Read image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process image with MediaPipe
                results = hands.process(image_rgb)

                # Extract landmarks if hand detected
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Convert to numpy array [21, 3]
                    landmarks = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in hand_landmarks.landmark
                    ])  # Shape: [21, 3]

                    landmarks_by_gesture[gesture_id].append(landmarks)

            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue

    logger.info(f"✓ Extracted landmarks from {sum(len(v) for v in landmarks_by_gesture.values())} images")

    return landmarks_by_gesture


def _generate_mock_landmarks_from_videos(video_dirs: List[Path]) -> Dict[int, List[np.ndarray]]:
    """
    Generate mock landmark data for testing when MediaPipe is not available.

    Args:
        video_dirs: List of video directories containing image sequences

    Returns:
        Mock landmarks dictionary
    """
    logger = setup_logging()
    logger.info("Generating mock landmark data for testing...")

    landmarks_by_gesture = {}

    # Set random seed for reproducible mock data
    np.random.seed(42)

    for video_id, video_dir in enumerate(video_dirs):
        # Assign mock gesture labels for testing (cycle through available gestures)
        gesture_id = video_id % NUM_CLASSES
        if gesture_id not in landmarks_by_gesture:
            landmarks_by_gesture[gesture_id] = []

        # Find all image files (sorted by frame number)
        image_files = sorted(video_dir.glob('*.jpg')) + sorted(video_dir.glob('*.png'))
        num_images = len(image_files)

        if num_images == 0:
            continue

        # Generate mock landmarks for each image
        for _ in range(num_images):
            # Generate realistic-looking hand landmarks
            # Base positions around wrist (index 0)
            wrist_x, wrist_y = np.random.uniform(0.3, 0.7, 2)

            # Generate 21 landmarks with some realistic hand structure
            landmarks = []
            for i in range(21):
                if i == 0:  # Wrist
                    x, y, z = wrist_x, wrist_y, 0.0
                else:
                    # Add some variation around the wrist
                    x = wrist_x + np.random.normal(0, 0.1)
                    y = wrist_y + np.random.normal(0, 0.1)
                    z = np.random.normal(0, 0.05)

                    # Keep within [0, 1] bounds
                    x = np.clip(x, 0, 1)
                    y = np.clip(y, 0, 1)

                landmarks.append([x, y, z])

            landmarks_by_gesture[gesture_id].append(np.array(landmarks))

    total_landmarks = sum(len(v) for v in landmarks_by_gesture.values())
    logger.info(f"✓ Generated {total_landmarks} mock landmarks for {len(landmarks_by_gesture)} gestures")

    return landmarks_by_gesture


def normalize_pose_sequence(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize hand pose to canonical space.
    
    Args:
        landmarks: [21, 3] or [T, 21, 3] hand landmarks
        
    Returns:
        Normalized landmarks with same shape
    """
    # Handle both single frame [21, 3] and sequence [T, 21, 3]
    if landmarks.ndim == 2:  # Single frame
        # Center at wrist (index 0)
        centered = landmarks - landmarks[0]
        # Normalize by standard deviation
        std = np.std(centered) + 1e-6
        normalized = centered / std
        return normalized
    
    elif landmarks.ndim == 3:  # Sequence
        normalized = []
        for frame in landmarks:
            centered = frame - frame[0]
            std = np.std(centered) + 1e-6
            normalized.append(centered / std)
        return np.array(normalized)
    
    else:
        raise ValueError(f"Expected 2D or 3D array, got {landmarks.ndim}D")


def create_temporal_sequences_from_images(
    landmarks_by_gesture: Dict[int, List[np.ndarray]],
    window_size: int = 60,
    stride: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create temporal sequences by combining consecutive images.
    
    Since Jester contains static images, we treat consecutive images
    from the same gesture class as a temporal sequence.
    
    Args:
        landmarks_by_gesture: Dict mapping gesture_id to list of [21, 3] landmarks
        window_size: Number of frames per sequence (default: 60)
        stride: Step size for sliding window (default: 5)
        
    Returns:
        Tuple of (sequences, labels) where:
        - sequences: torch.Tensor of shape [N, 60, 63]
        - labels: torch.Tensor of shape [N] with gesture class IDs
    """
    logger = setup_logging()
    logger.info(f"Creating temporal sequences (window_size={window_size}, stride={stride})...")
    
    all_sequences = []
    all_labels = []
    
    for gesture_id, landmarks_list in landmarks_by_gesture.items():
        if len(landmarks_list) < window_size:
            logger.warning(f"Gesture {gesture_id} has only {len(landmarks_list)} images "
                         f"(need {window_size}), skipping")
            continue
        
        # Normalize all landmarks for this gesture
        normalized_list = []
        for landmarks in landmarks_list:
            normalized = normalize_pose_sequence(landmarks)
            normalized_list.append(normalized)
        
        # Create sliding window sequences
        num_sequences = 0
        for start_idx in range(0, len(normalized_list) - window_size, stride):
            # Extract window of consecutive landmarks
            window = normalized_list[start_idx:start_idx + window_size]
            
            # Stack: [60, 21, 3] → Reshape: [60, 63]
            sequence = np.stack(window).reshape(window_size, -1)
            all_sequences.append(sequence)
            all_labels.append(gesture_id)
            num_sequences += 1
        
        logger.info(f"  Gesture {gesture_id}: Created {num_sequences} sequences from "
                   f"{len(landmarks_list)} images")
    
    # Convert to PyTorch tensors
    sequences = torch.tensor(np.array(all_sequences), dtype=torch.float32)
    labels = torch.tensor(all_labels, dtype=torch.long)
    
    logger.info(f"✓ Created {len(sequences)} total sequences")
    logger.info(f"  Sequences shape: {sequences.shape}")
    logger.info(f"  Labels shape: {labels.shape}")
    
    return sequences, labels


def create_train_val_test_split(
    sequences: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader,
           torch.utils.data.DataLoader]:
    """
    Create stratified train/val/test split.

    Args:
        sequences: Tensor of all sequences
        labels: Tensor of all labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split

    logger = setup_logging()
    logger.info("Creating train/val/test split...")

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    indices = np.arange(len(sequences))
    labels_np = labels.cpu().numpy()

    # Check if we have enough samples for stratified split
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    min_samples = min(counts)

    if min_samples < 2:
        logger.warning(f"Some classes have very few samples (min: {min_samples}). Using random split instead of stratified.")
        # Fall back to random split
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=random_state
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=random_state
        )
    else:
        # Use stratified split
        try:
            # First split: train+val vs test
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_ratio,
                stratify=labels_np,
                random_state=random_state
            )

            # Second split: train vs val
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio / (train_ratio + val_ratio),
                stratify=labels_np[train_val_idx],
                random_state=random_state
            )
        except ValueError:
            # If stratified split still fails, use random split
            logger.warning("Stratified split failed, using random split.")
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_ratio, random_state=random_state
            )
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio / (train_ratio + val_ratio),
                random_state=random_state
            )

    # Create datasets
    train_dataset = GestureDataset(sequences[train_idx], labels[train_idx])
    val_dataset = GestureDataset(sequences[val_idx], labels[val_idx])
    test_dataset = GestureDataset(sequences[test_idx], labels[test_idx])

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    logger.info(f"✓ Split created:")
    logger.info(f"  Train: {len(train_dataset)} sequences")
    logger.info(f"  Val:   {len(val_dataset)} sequences")
    logger.info(f"  Test:  {len(test_dataset)} sequences")

    return train_loader, val_loader, test_loader


def save_processed_data(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    output_dir: Path = JESTER_PROCESSED_DIR
) -> None:
    """Save processed dataloaders to disk."""
    logger = setup_logging()
    logger.info(f"Saving processed data to {output_dir}...")
    
    ensure_directory_exists(output_dir / 'dummy_file')
    
    # Extract data from dataloaders
    train_sequences, train_labels = [], []
    for seq, label in train_loader:
        train_sequences.append(seq)
        train_labels.append(label)
    
    val_sequences, val_labels = [], []
    for seq, label in val_loader:
        val_sequences.append(seq)
        val_labels.append(label)
    
    test_sequences, test_labels = [], []
    for seq, label in test_loader:
        test_sequences.append(seq)
        test_labels.append(label)
    
    # Save as torch tensors
    torch.save({
        'sequences': torch.cat(train_sequences),
        'labels': torch.cat(train_labels)
    }, output_dir / 'train.pt')
    
    torch.save({
        'sequences': torch.cat(val_sequences),
        'labels': torch.cat(val_labels)
    }, output_dir / 'val.pt')
    
    torch.save({
        'sequences': torch.cat(test_sequences),
        'labels': torch.cat(test_labels)
    }, output_dir / 'test.pt')
    
    logger.info("✓ Data saved successfully")


def preprocess_jester_dataset(
    image_dir: Path = JESTER_RAW_DIR,
    output_dir: Path = JESTER_PROCESSED_DIR,
    max_videos: Optional[int] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader,
           torch.utils.data.DataLoader]:
    """
    Complete preprocessing pipeline for Jester images.
    
    Steps:
    1. Load images from directory
    2. Extract hand landmarks using MediaPipe
    3. Normalize poses
    4. Create temporal sequences
    5. Split into train/val/test
    6. Save processed data
    
    Args:
        image_dir: Directory containing gesture_X subdirectories with images
        output_dir: Where to save processed data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("STARTING JESTER PREPROCESSING PIPELINE")
    logger.info("="*70)
    
    # Check if dataset exists
    if not image_dir.exists() or len(list(image_dir.glob('*'))) == 0:
        logger.error(f"Image directory not found: {image_dir}")
        if not setup_jester_dataset():
            raise FileNotFoundError(f"Jester images not found in {image_dir}")
    
    # Step 1: Extract hand landmarks from images
    logger.info("\n[1/4] Extracting hand landmarks from images...")
    landmarks_by_gesture = extract_hand_landmarks_from_images(image_dir, max_videos)
    
    # Step 2: Create temporal sequences
    logger.info("\n[2/4] Creating temporal sequences...")
    # Use smaller window size for testing with short videos
    window_size = min(60, max(len(landmarks) for landmarks in landmarks_by_gesture.values()) - 5)
    if window_size < 10:
        window_size = 10  # Minimum window size
    logger.info(f"Using window_size={window_size} for testing")

    sequences, labels = create_temporal_sequences_from_images(
        landmarks_by_gesture,
        window_size=window_size,
        stride=2  # Smaller stride for more sequences
    )
    
    # Step 3: Create train/val/test split
    logger.info("\n[3/4] Creating train/val/test split...")
    train_loader, val_loader, test_loader = create_train_val_test_split(
        sequences, labels
    )
    
    # Step 4: Save processed data
    logger.info("\n[4/4] Saving processed data...")
    save_processed_data(train_loader, val_loader, test_loader, output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("✓ PREPROCESSING COMPLETE")
    logger.info("="*70 + "\n")
    
    return train_loader, val_loader, test_loader


def load_preprocessed_data(
    data_dir: Path = JESTER_PROCESSED_DIR,
    batch_size: int = 32
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader,
           torch.utils.data.DataLoader]:
    """
    Load preprocessed data from saved files.
    
    Args:
        data_dir: Directory containing train.pt, val.pt, test.pt
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = setup_logging()
    
    logger.info("Loading preprocessed data...")
    
    # Load data
    train_data = torch.load(data_dir / 'train.pt')
    val_data = torch.load(data_dir / 'val.pt')
    test_data = torch.load(data_dir / 'test.pt')
    
    # Create datasets
    train_dataset = GestureDataset(train_data['sequences'], train_data['labels'])
    val_dataset = GestureDataset(val_data['sequences'], val_data['labels'])
    test_dataset = GestureDataset(test_data['sequences'], test_data['labels'])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    logger.info(f"✓ Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
