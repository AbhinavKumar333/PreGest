"""
Quest 3 Dataset Loader for Gesture Recognition
PyTorch Dataset class for loading processed Quest 3 tensors.
"""

import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import random

from .config import (
    QUEST3_PROCESSED_DIR, SEQUENCE_LENGTH, FRAME_SIZE, NUM_CHANNELS,
    TRAIN_CONFIG
)


class Quest3GestureDataset(Dataset):
    """PyTorch Dataset for Quest 3 processed gesture data."""

    def __init__(self, split: str = 'train', augment: bool = False):
        """Initialize Quest 3 dataset.

        Args:
            split: Data split ('train', 'val', 'test')
            augment: Whether to apply data augmentation (training only)
        """
        self.split = split
        self.augment = augment

        # Load processed tensors (handle both single files and chunks)
        self.windows, self.labels = self._load_data()

        # Set up augmentation transforms
        self.setup_augmentation()

    def _load_data(self):
        """Load data from either single file or chunked files."""
        # First try to load single files
        windows_path = QUEST3_PROCESSED_DIR / f"{self.split}_windows.pt"
        labels_path = QUEST3_PROCESSED_DIR / f"{self.split}_labels.pt"

        if windows_path.exists() and labels_path.exists():
            # Load single files
            windows = torch.load(windows_path, weights_only=True)  # (N, 60, 4, 224, 224)
            labels = torch.load(labels_path, weights_only=True)    # (N,)
            return windows, labels

        # If single files don't exist, try loading from chunks
        chunks_metadata_path = QUEST3_PROCESSED_DIR / f"{self.split}_chunks_metadata.json"

        if chunks_metadata_path.exists():
            # Load chunked data
            with open(chunks_metadata_path, 'r') as f:
                metadata = json.load(f)

            num_chunks = metadata['num_chunks']
            all_windows = []
            all_labels = []

            for chunk_idx in range(num_chunks):
                chunk_windows_path = QUEST3_PROCESSED_DIR / f"{self.split}_windows_chunk_{chunk_idx}.pt"
                chunk_labels_path = QUEST3_PROCESSED_DIR / f"{self.split}_labels_chunk_{chunk_idx}.pt"

                if not chunk_windows_path.exists() or not chunk_labels_path.exists():
                    raise FileNotFoundError(f"Missing chunk {chunk_idx} for {self.split} split")

                chunk_windows = torch.load(chunk_windows_path, weights_only=True)
                chunk_labels = torch.load(chunk_labels_path, weights_only=True)

                all_windows.append(chunk_windows)
                all_labels.append(chunk_labels)

            # Concatenate all chunks
            windows = torch.cat(all_windows, dim=0)
            labels = torch.cat(all_labels, dim=0)

            print(f"Loaded {self.split} data: {num_chunks} chunks → {len(labels)} total samples")
            return windows, labels

        # If neither exists
        raise FileNotFoundError(
            f"No processed data found for '{self.split}' split. "
            f"Expected either:\n"
            f"  {windows_path} + {labels_path}\n"
            f"  OR chunked files matching: {QUEST3_PROCESSED_DIR}/{self.split}_*_chunk_*.pt"
        )

    def setup_augmentation(self):
        """Set up data augmentation transforms."""
        self.transform = None  # Always initialize

        if not self.augment:
            return

        # Spatial augmentations for RGB channels only
        self.spatial_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=(-15, 15),  # ±15° rotation
                translate=(0.1, 0.1),  # ±10% translation
                scale=(0.9, 1.1),  # ±10% scaling
                fill=0
            ),
            transforms.ToTensor()
        ])

        # Color augmentations
        self.color_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=0.3,  # ±30% brightness
                contrast=0.3,    # ±30% contrast
                saturation=0.3,  # ±30% saturation
                hue=0.1         # ±10° hue shift
            ),
            transforms.ToTensor()
        ])

    def apply_augmentation(self, window: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to a temporal window.

        Args:
            window: Input window tensor (60, 4, 224, 224)

        Returns:
            Augmented window tensor
        """
        if not self.augment or self.transform is None:
            return window

        augmented_frames = []

        for frame_idx in range(SEQUENCE_LENGTH):
            # Extract RGB and mask channels
            rgb_frame = window[frame_idx, :3]  # (3, 224, 224)
            mask_frame = window[frame_idx, 3:]  # (1, 224, 224)

            # Convert to PIL for torchvision transforms
            rgb_pil = transforms.ToPILImage()(rgb_frame)

            # Apply spatial transforms to RGB
            rgb_augmented = self.spatial_transform(rgb_pil)

            # Apply color transforms
            rgb_augmented = self.color_transform(transforms.ToPILImage()(rgb_augmented))

            # Combine RGB and mask (mask unchanged)
            augmented_frame = torch.cat([rgb_augmented, mask_frame], dim=0)
            augmented_frames.append(augmented_frame)

        return torch.stack(augmented_frames)  # (60, 4, 224, 224)

    def temporal_augmentation(self, window: torch.Tensor) -> torch.Tensor:
        """Apply temporal augmentations.

        Args:
            window: Input window tensor (60, 4, 224, 224)

        Returns:
            Temporally augmented window
        """
        if not self.augment:
            return window

        # Temporal jitter: randomly shift window start by ±5 frames
        if random.random() < 0.5:
            max_shift = 5
            shift = random.randint(-max_shift, max_shift)

            if shift > 0:
                # Shift forward: pad beginning with first frame
                pad_frames = window[:shift]  # First 'shift' frames
                window = torch.cat([pad_frames, window[:-shift]], dim=0)
            elif shift < 0:
                # Shift backward: pad end with last frame
                shift = abs(shift)
                pad_frames = window[-shift:]  # Last 'shift' frames
                window = torch.cat([window[shift:], pad_frames], dim=0)

        # Frame dropout: randomly drop 5% of frames and interpolate
        if random.random() < 0.3:  # 30% chance
            dropout_prob = 0.05
            dropout_mask = torch.rand(SEQUENCE_LENGTH) > dropout_prob

            # For dropped frames, interpolate from neighbors
            for i in range(SEQUENCE_LENGTH):
                if not dropout_mask[i]:
                    # Find nearest valid frames
                    left_idx = i - 1
                    right_idx = i + 1

                    while left_idx >= 0 and not dropout_mask[left_idx]:
                        left_idx -= 1
                    while right_idx < SEQUENCE_LENGTH and not dropout_mask[right_idx]:
                        right_idx += 1

                    if left_idx >= 0 and right_idx < SEQUENCE_LENGTH:
                        # Linear interpolation
                        alpha = (i - left_idx) / (right_idx - left_idx)
                        window[i] = (1 - alpha) * window[left_idx] + alpha * window[right_idx]
                    elif left_idx >= 0:
                        # Use left neighbor
                        window[i] = window[left_idx]
                    elif right_idx < SEQUENCE_LENGTH:
                        # Use right neighbor
                        window[i] = window[right_idx]
                    # If no valid neighbors, keep original frame

        return window

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (window_tensor, label)
        """
        window = self.windows[idx]  # (60, 4, 224, 224)
        label = self.labels[idx]    # scalar

        # Apply augmentations if training
        if self.augment:
            window = self.apply_augmentation(window)
            window = self.temporal_augmentation(window)

        return window, label


def get_quest3_dataloaders(batch_size: int = 16,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create Quest 3 dataloaders for train/val/test.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = Quest3GestureDataset('train', augment=True)
    val_dataset = Quest3GestureDataset('val', augment=False)
    test_dataset = Quest3GestureDataset('test', augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def get_quest3_dataset_info() -> dict:
    """Get information about the Quest 3 dataset splits.

    Returns:
        Dict with dataset statistics
    """
    info = {}

    for split in ['train', 'val', 'test']:
        try:
            dataset = Quest3GestureDataset(split, augment=False)
            windows = dataset.windows
            labels = dataset.labels

            # Count classes
            unique_labels, counts = torch.unique(labels, return_counts=True)
            class_dist = dict(zip(unique_labels.tolist(), counts.tolist()))

            info[split] = {
                'num_samples': len(dataset),
                'class_distribution': class_dist,
                'window_shape': tuple(windows.shape),
                'label_shape': tuple(labels.shape)
            }
        except FileNotFoundError:
            info[split] = None

    return info


if __name__ == "__main__":
    # Test dataset loading
    print("Testing Quest 3 dataset loading...")

    try:
        # Get dataset info
        info = get_quest3_dataset_info()

        for split, split_info in info.items():
            if split_info:
                print(f"\n{split.upper()} split:")
                print(f"  Samples: {split_info['num_samples']}")
                print(f"  Classes: {split_info['class_distribution']}")
                print(f"  Window shape: {split_info['window_shape']}")
            else:
                print(f"\n{split.upper()} split: Not found")

        # Test dataloader creation
        print("\nTesting dataloader creation...")
        train_loader, val_loader, test_loader = get_quest3_dataloaders(batch_size=4)

        # Test one batch
        for batch in train_loader:
            rgb_mask, labels = batch
            print(f"\nSample batch shape: {rgb_mask.shape}")
            print(f"Sample labels shape: {labels.shape}")
            print(f"Sample labels: {labels.tolist()}")
            break

        print("\n✅ Quest 3 dataset loading successful!")

    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
