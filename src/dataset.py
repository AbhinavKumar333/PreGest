"""
Dataset preprocessing pipeline for Jester gesture recognition.
Supports flexible pretraining on 8 Quest3 gestures, 27 Jester gestures, or custom subset.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
import logging
import random

# Optional mediapipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

from .config import (
    JESTER_RAW_DIR, JESTER_PROCESSED_DIR, DATA_CONFIG,
    TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, NUM_CLASSES,
    JESTER_CLASSES, QUEST3_TARGET_GESTURES, JESTER_TO_QUEST3_MAPPING,
    PRETRAINING_MODE, NUM_JESTER_CLASSES, NUM_QUEST3_CLASSES,
    SELECTED_GESTURE_IDS
)
from .utils import setup_logging, ensure_directory_exists


def load_jester_labels(csv_path: Path) -> Dict[str, int]:
    """
    Load video-to-label mapping from Jester CSV file.

    Args:
        csv_path: Path to CSV file (Train.csv, Validation.csv, or Test.csv)

    Returns:
        Dictionary mapping video_id (string) to label_id (int)
    """
    logger = setup_logging()
    logger.info(f"Loading labels from {csv_path}")

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Handle different column names (id vs video_id)
    video_id_col = 'id' if 'id' in df.columns else 'video_id'

    # Check if label_id column exists and has valid values
    if 'label_id' not in df.columns:
        logger.warning(f"No label_id column in {csv_path.name} - this may be a test set without labels")
        return {}

    # Count valid (non-NaN) labels
    valid_labels = df['label_id'].notna().sum()
    total_rows = len(df)

    if valid_labels == 0:
        logger.warning(f"No valid labels found in {csv_path.name} - skipping this file")
        return {}

    if valid_labels < total_rows:
        logger.warning(f"Only {valid_labels}/{total_rows} videos have labels in {csv_path.name}")

    # Create mapping from video_id to label_id
    label_mapping = {}
    for _, row in df.iterrows():
        if pd.isna(row['label_id']):
            continue  # Skip rows without labels

        video_id = str(int(row[video_id_col]))  # Ensure string format
        label_id = int(float(row['label_id']))  # Handle potential float values
        label_mapping[video_id] = label_id

    logger.info(f"Loaded {len(label_mapping)} video-label mappings from {csv_path.name}")
    return label_mapping


def _generate_mock_landmarks_for_selected_gestures(
    gesture_ids: List[int],
    max_videos: Optional[int] = None
) -> Dict[int, List[np.ndarray]]:
    """
    Generate mock landmark data for testing when MediaPipe is not available.
    Only generates data for selected gesture IDs.

    Args:
        gesture_ids: List of gesture IDs to generate data for
        max_videos: Maximum number of videos to simulate

    Returns:
        Mock landmarks dictionary organized by gesture_id
    """
    logger = setup_logging()
    logger.info("Generating mock landmark data for selected gestures...")

    landmarks_by_gesture = {i: [] for i in gesture_ids}

    # Set random seed for reproducible mock data
    np.random.seed(42)

    # Simulate processing videos for each gesture
    videos_per_gesture = max_videos // len(gesture_ids) if max_videos else 10

    for gesture_id in gesture_ids:
        for video_idx in range(videos_per_gesture):
            # Generate mock landmarks for each frame in video
            # Simulate 30-40 frames per video
            num_frames = np.random.randint(30, 41)

            for _ in range(num_frames):
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
    logger.info(f"✓ Generated {total_landmarks} mock landmarks for {len(gesture_ids)} gestures")

    return landmarks_by_gesture


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
    logger.warning("\nThe Jester dataset contains numbered video folders (1, 2, 3, ...)")
    logger.warning(f"with image files organized as: {JESTER_RAW_DIR}/")
    logger.warning("  ├── 1/")
    logger.warning("  │   ├── 00001.jpg ... 00035.jpg")
    logger.warning("  ├── 2/")
    logger.warning("  │   └── ...")
    logger.warning("  └── ...\n")
    
    logger.warning("Option 1: Automatic Download (Requires Kaggle API)")
    logger.warning("  1. pip install kaggle")
    logger.warning("  2. Place kaggle.json in ~/.kaggle/")
    logger.warning("  3. Run: dataset.download_and_extract_jester()\n")
    
    logger.warning("Option 2: Manual Download (from Qualcomm)")
    logger.warning("  1. Visit: https://www.qualcomm.com/developer/software/jester-dataset/downloads")
    logger.warning("  2. Download and extract the dataset")
    logger.warning(f"  3. Move to: {JESTER_RAW_DIR}\n")
    
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


def extract_hand_landmarks_from_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Extract hand landmarks from a single image file using MediaPipe.
    
    Args:
        image_path: Path to image file
        
    Returns:
        landmarks [21, 3] or None if no hand detected
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in hand_landmarks.landmark
            ])  # Shape: [21, 3]
            return landmarks
        
        return None
    
    except Exception as e:
        return None


def extract_hand_landmarks_from_videos(
    video_dir: Path = JESTER_RAW_DIR,
    max_videos: Optional[int] = None,
    gesture_ids: Optional[List[int]] = None
) -> Dict[int, List[np.ndarray]]:
    """
    Extract hand landmarks from Jester dataset video folders using MediaPipe.

    Args:
        video_dir: Directory containing Train/, Validation/, Test/ subdirectories
        max_videos: Limit number of videos to process
        gesture_ids: List of gesture IDs to include (from SELECTED_GESTURE_IDS)

    Returns:
        Dict mapping gesture_id → list of [21, 3] landmarks
    """
    if not MEDIAPIPE_AVAILABLE:
        logger = setup_logging()
        logger.warning("MediaPipe not available - generating mock landmark data for testing")
        return _generate_mock_landmarks_for_selected_gestures(gesture_ids or SELECTED_GESTURE_IDS, max_videos)

    logger = setup_logging()
    logger.info("Extracting hand landmarks from Jester dataset...")

    # Load label mappings from CSV files
    train_labels = load_jester_labels(video_dir / "Train.csv")
    val_labels = load_jester_labels(video_dir / "Validation.csv")
    test_labels = load_jester_labels(video_dir / "Test.csv")

    # Combine all label mappings
    all_labels = {**train_labels, **val_labels, **test_labels}
    logger.info(f"Total videos with labels: {len(all_labels)}")

    # Determine which gestures to include
    if gesture_ids is None:
        gesture_ids = SELECTED_GESTURE_IDS

    logger.info(f"Processing gestures: {gesture_ids}")

    # Initialize landmark storage
    landmarks_by_gesture = {i: [] for i in gesture_ids}

    # Get video directories from Train/, Validation/, Test/ folders
    train_dir = video_dir / "Train"
    val_dir = video_dir / "Validation"
    test_dir = video_dir / "Test"

    video_dirs = []
    if train_dir.exists():
        video_dirs.extend(sorted([d for d in train_dir.iterdir() if d.is_dir()]))
    if val_dir.exists():
        video_dirs.extend(sorted([d for d in val_dir.iterdir() if d.is_dir()]))
    if test_dir.exists():
        video_dirs.extend(sorted([d for d in test_dir.iterdir() if d.is_dir()]))

    # Filter to only videos that have labels and are in selected gestures
    video_dirs = [d for d in video_dirs if d.name in all_labels and all_labels[d.name] in gesture_ids]

    # Apply max_videos limit if specified
    if max_videos is not None:
        video_dirs = video_dirs[:max_videos]
        logger.info(f"Processing {len(video_dirs)} videos (limited by --max-videos={max_videos})")
    else:
        logger.info(f"Processing {len(video_dirs)} videos for full training")

    # Initialize MediaPipe Hands for STATIC IMAGE mode
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,  # CRITICAL: Use True for images
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    # Process each video folder
    for video_dir_path in tqdm(video_dirs, desc="Processing videos"):
        video_id = video_dir_path.name
        gesture_id = all_labels[video_id]

        # Find all image files (sorted by frame number)
        image_files = sorted(video_dir_path.glob('*.jpg')) + sorted(video_dir_path.glob('*.png'))

        if len(image_files) == 0:
            logger.warning(f"No images found in {video_dir_path.name}")
            continue

        video_landmarks = []
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

                    video_landmarks.append(landmarks)

            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue

        # Add all landmarks from this video to the gesture
        landmarks_by_gesture[gesture_id].extend(video_landmarks)

    # Log statistics
    total_landmarks = sum(len(v) for v in landmarks_by_gesture.values())
    logger.info(f"✓ Extracted {total_landmarks} landmarks from {len(video_dirs)} videos")

    for gesture_id, landmarks_list in landmarks_by_gesture.items():
        if len(landmarks_list) > 0:
            gesture_name = JESTER_CLASSES[gesture_id] if gesture_id < NUM_JESTER_CLASSES else "Unknown"
            logger.info(f"  Gesture {gesture_id} ({gesture_name}): {len(landmarks_list)} landmarks")

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
        centered = landmarks - landmarks[0]
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
    stride: int = 5,
    mapping: Optional[Dict[int, int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create temporal sequences by combining consecutive images.
    
    Args:
        landmarks_by_gesture: Dict mapping gesture_id to list of [21, 3] landmarks
        window_size: Number of frames per sequence (default: 60)
        stride: Step size for sliding window (default: 5)
        mapping: Optional dict to remap gesture IDs
        
    Returns:
        Tuple of (sequences, labels)
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
            
            # Remap gesture ID if mapping provided
            final_gesture_id = mapping.get(gesture_id, gesture_id) if mapping else gesture_id
            all_labels.append(final_gesture_id)
            
            num_sequences += 1
        
        gesture_name = JESTER_CLASSES[gesture_id] if gesture_id < NUM_JESTER_CLASSES else "Unknown"
        logger.info(f"  Gesture {gesture_id} ({gesture_name}): Created {num_sequences} sequences "
                   f"from {len(landmarks_list)} images")
    
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
    logger.info("Creating train/val/test split (stratified)...")
    
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    
    indices = np.arange(len(sequences))
    labels_np = labels.cpu().numpy()
    
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
    logger.info(f"  Train: {len(train_dataset)} sequences ({len(train_dataset)/len(sequences)*100:.1f}%)")
    logger.info(f"  Val:   {len(val_dataset)} sequences ({len(val_dataset)/len(sequences)*100:.1f}%)")
    logger.info(f"  Test:  {len(test_dataset)} sequences ({len(test_dataset)/len(sequences)*100:.1f}%)")
    
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
    
    ensure_directory_exists(output_dir)
    
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
    video_dir: Path = JESTER_RAW_DIR,
    labels_csv: Optional[Path] = None,
    max_videos: Optional[int] = None,
    output_dir: Path = JESTER_PROCESSED_DIR
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, 
           torch.utils.data.DataLoader]:
    """
    Complete preprocessing pipeline for Jester images.
    
    Supports flexible pretraining on:
    - 8 Quest3-mapped gestures (PRETRAINING_MODE = "quest3")
    - 27 full Jester gestures (PRETRAINING_MODE = "jester")
    - Custom gesture subset (PRETRAINING_MODE = "custom")
    
    Args:
        video_dir: Directory containing gesture_X subdirectories with images
        labels_csv: Optional path to Jester labels CSV
        max_videos: Limit number of videos to process
        output_dir: Where to save processed data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("JESTER PREPROCESSING PIPELINE")
    logger.info(f"Pretraining Mode: {PRETRAINING_MODE}")
    logger.info(f"Number of Classes: {NUM_CLASSES}")
    logger.info(f"Selected Gesture IDs: {SELECTED_GESTURE_IDS}")
    logger.info("="*70)
    
    # Check if dataset exists
    if not video_dir.exists() or len(list(video_dir.glob('*'))) == 0:
        logger.error(f"Video directory not found: {video_dir}")
        if not setup_jester_dataset():
            logger.error("Please download the Jester dataset first")
            raise FileNotFoundError(f"Jester videos not found in {video_dir}")
    
    # Step 1: Extract hand landmarks from images
    logger.info("\n[1/4] Extracting hand landmarks from images...")
    landmarks_by_gesture = extract_hand_landmarks_from_videos(
        video_dir,
        max_videos,
        gesture_ids=SELECTED_GESTURE_IDS
    )
    
    # Step 2: Create temporal sequences
    logger.info("\n[2/4] Creating temporal sequences...")
    sequences, labels = create_temporal_sequences_from_images(
        landmarks_by_gesture,
        window_size=60,
        stride=5,
        mapping=JESTER_TO_QUEST3_MAPPING if PRETRAINING_MODE == "quest3" else None
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
    
    logger.info(f"✓ Data loaded:")
    logger.info(f"  Train: {len(train_dataset)} sequences")
    logger.info(f"  Val:   {len(val_dataset)} sequences")
    logger.info(f"  Test:  {len(test_dataset)} sequences")
    
    return train_loader, val_loader, test_loader
