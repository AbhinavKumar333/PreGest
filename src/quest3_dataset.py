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
from collections import OrderedDict
import random

from .config import (
    QUEST3_PROCESSED_DIR, SEQUENCE_LENGTH, FRAME_SIZE, NUM_CHANNELS,
    TRAIN_CONFIG, QUEST3_GESTURES, NUM_QUEST3_CLASSES
)

class Quest3GestureDataset(Dataset):
    """PyTorch Dataset for Quest 3 processed gesture data."""
    
    def __init__(self, split: str = 'train', augment: bool = False,
                 phase2_config: dict = None):
        """Initialize Quest 3 dataset.
        
        Args:
            split: Data split ('train', 'val', 'test')
            augment: Whether to apply data augmentation (training only)
            phase2_config: Enhanced augmentation config for Phase 2 improvements
        """
        self.split = split
        self.augment = augment
        self.phase2_config = phase2_config or {}
        
        # Load processed tensors (handle both single files and chunks)
        self.windows, self.labels = self._load_data()
        
        # Set up augmentation transforms
        self.setup_augmentation()
    
    def _load_data(self):
        """Load data from either single file or chunked files (lazy loading for chunks)."""
        # First try to load single files
        windows_path = QUEST3_PROCESSED_DIR / f"{self.split}_windows.pt"
        labels_path = QUEST3_PROCESSED_DIR / f"{self.split}_labels.pt"
        
        if windows_path.exists() and labels_path.exists():
            # Load single files
            windows = torch.load(windows_path, weights_only=True)  # (N, 60, 4, 224, 224)
            labels = torch.load(labels_path, weights_only=True)  # (N,)
            self.use_lazy_loading = False
            return windows, labels
        
        # If single files don't exist, use lazy loading for chunks
        chunks_metadata_path = QUEST3_PROCESSED_DIR / f"{self.split}_chunks_metadata.json"
        
        if chunks_metadata_path.exists():
            # Use lazy loading - don't load all chunks at once!
            with open(chunks_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.num_chunks = metadata['num_chunks']
            self.use_lazy_loading = True
            
            # Build index: map global sample index to (chunk_idx, sample_idx_within_chunk)
            self.chunk_index_map = []
            total_samples = 0
            
            for chunk_idx in range(self.num_chunks):
                chunk_labels_path = QUEST3_PROCESSED_DIR / f"{self.split}_labels_chunk_{chunk_idx}.pt"
                
                if not chunk_labels_path.exists():
                    raise FileNotFoundError(f"Missing chunk {chunk_idx} for {self.split} split")
                
                # Only load labels to count samples (labels are small)
                chunk_labels = torch.load(chunk_labels_path, weights_only=True)
                num_samples_in_chunk = len(chunk_labels)
                
                # Map each sample in this chunk
                for local_idx in range(num_samples_in_chunk):
                    self.chunk_index_map.append((chunk_idx, local_idx, chunk_labels[local_idx]))
                
                total_samples += num_samples_in_chunk
            
            print(f"🔧 Lazy loading enabled: {self.split} split = {total_samples} samples from {self.num_chunks} chunks")
            
            # Cache for recently loaded chunks (keep last 2 chunks in memory)
            # Using OrderedDict to guarantee FIFO eviction order
            self.chunk_cache = OrderedDict()
            self.cache_size = 2
            
            # Return dummy tensors (not used with lazy loading)
            return None, None
        
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
                contrast=0.3,  # ±30% contrast
                saturation=0.3,  # ±30% saturation
                hue=0.1  # ±10° hue shift
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
    
    def apply_phase2_augmentations(self, window: torch.Tensor, gesture_name: str,
                                   label_idx: int) -> torch.Tensor:
        """Apply Phase 2 targeted augmentations based on confusion analysis.
        
        Args:
            window: Input window tensor (60, 4, 224, 224)
            gesture_name: Name of the gesture
            label_idx: Index of the gesture label
        
        Returns:
            Augmented window tensor
        """
        if not self.phase2_config:
            return window
        
        augmented_window = window.clone()
        
        # Apply temporal augmentations based on Phase 2 config
        temporal_config = self.phase2_config.get('temporal', {})
        if temporal_config:
            augmented_window = self.apply_temporal_phase2_augmentations(
                augmented_window, gesture_name, temporal_config)
        
        # Apply spatial augmentations based on Phase 2 config
        spatial_config = self.phase2_config.get('spatial', {})
        if spatial_config:
            augmented_window = self.apply_spatial_phase2_augmentations(
                augmented_window, gesture_name, spatial_config)
        
        # Apply gesture-specific augmentations
        gesture_config = self.phase2_config.get('gesture_specific', {})
        if gesture_name in gesture_config:
            gesture_specific = gesture_config[gesture_name]
            augmented_window = self.apply_gesture_specific_augmentations(
                augmented_window, gesture_name, gesture_specific)
        
        return augmented_window
    def apply_temporal_phase2_augmentations(self, window: torch.Tensor, gesture_name: str,
                                           config: dict) -> torch.Tensor:
        """Apply enhanced temporal augmentations targeting confusion areas."""
        augmented = window.clone()
        
        # Speed variation (±15% speed changes)
        speed_var = config.get('speed_variation', [0.85, 1.15])
        if random.random() < 0.3:  # 30% chance
            speed_factor = random.uniform(speed_var[0], speed_var[1])
            if speed_factor > 1.0:
                # Speed up: remove frames
                num_frames = int(SEQUENCE_LENGTH / speed_factor)
                indices = torch.linspace(0, SEQUENCE_LENGTH-1, num_frames).long()
                augmented = augmented[indices]
                
                # Pad or crop back to SEQUENCE_LENGTH
                if len(augmented) < SEQUENCE_LENGTH:
                    last_frame = augmented[-1:].repeat(SEQUENCE_LENGTH - len(augmented), 1, 1, 1)
                    augmented = torch.cat([augmented, last_frame], dim=0)
                elif len(augmented) > SEQUENCE_LENGTH:
                    augmented = augmented[:SEQUENCE_LENGTH]
            elif speed_factor < 1.0:
                # Slow down: duplicate frames
                num_duplicates = int(SEQUENCE_LENGTH * (1/speed_factor - 1))
                duplicate_indices = torch.randint(0, SEQUENCE_LENGTH, (num_duplicates,))
                new_frames = augmented[duplicate_indices]
                all_frames = torch.cat([augmented, new_frames], dim=0)
                # Random sampling back to SEQUENCE_LENGTH
                indices = torch.randperm(len(all_frames))[:SEQUENCE_LENGTH]
                augmented = all_frames[indices]
        
        # Enhanced temporal jitter (±3 frames)
        temporal_jitter = config.get('temporal_jitter', 3)
        if random.random() < 0.4:  # 40% chance
            max_shift = temporal_jitter
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                pad_frames = augmented[:shift]
                augmented = torch.cat([pad_frames, augmented[:-shift]], dim=0)
            elif shift < 0:
                shift = abs(shift)
                pad_frames = augmented[-shift:]
                augmented = torch.cat([augmented[shift:], pad_frames], dim=0)
        
        # Gesture phase shift (±3 frames)
        phase_shift = config.get('gesture_phase_shift', [-3, 3])
        if random.random() < 0.25:  # 25% chance
            shift = random.randint(phase_shift[0], phase_shift[1])
            if shift > 0:
                start_pad = augmented[:shift]
                augmented = torch.cat([start_pad, augmented[:-shift]], dim=0)
            elif shift < 0:
                shift = abs(shift)
                end_pad = augmented[-shift:]
                augmented = torch.cat([augmented[shift:], end_pad], dim=0)
        
        return augmented
    
    def apply_spatial_phase2_augmentations(self, window: torch.Tensor, gesture_name: str,
                                        config: dict) -> torch.Tensor:
        """Apply enhanced spatial augmentations targeting confusion areas."""
        augmented = window.clone()
        
        # Elastic deformation (temporarily disabled due to shape issues)
        # TODO: Fix elastic deformation broadcasting
        # elastic_config = config.get('elastic_deformation', {})
        # if elastic_config and random.random() < 0.2:  # 20% chance
        #     # Implementation would go here
        pass
        
        # Occlusion simulation (helps for swipe gesture discrimination)
        occlusion_prob = config.get('occlusion_simulation', 0.15)
        if random.random() < occlusion_prob:
            for frame_idx in range(SEQUENCE_LENGTH):
                # Randomly occlude parts of the hand
                mask = np.random.rand(3, 224, 224) > 0.1  # 10% occlusion per channel
                rgb_frame = augmented[frame_idx, :3].cpu().numpy()
                rgb_frame[~mask] = 0  # Black out occluded areas
                augmented[frame_idx, :3] = torch.from_numpy(rgb_frame).float()
        
        # Enhanced brightness jitter (±20%)
        brightness_jitter = config.get('brightness_jitter', 0.2)
        if random.random() < 0.3:  # 30% chance
            factor = 1.0 + random.uniform(-brightness_jitter, brightness_jitter)
            for frame_idx in range(SEQUENCE_LENGTH):
                rgb_frame = augmented[frame_idx, :3]
                augmented[frame_idx, :3] = torch.clamp(rgb_frame * factor, 0, 1)
        
        return augmented

    
    def apply_gesture_specific_augmentations(self, window: torch.Tensor, gesture_name: str,
                                            config: dict) -> torch.Tensor:
        """Apply gesture-specific augmentations targeting top confusion areas."""
        augmented = window.clone()
        
        # Targeted augmentations based on gesture type
        if gesture_name == 'release':
            # Fix confusion with flat_palm_stop: enhance temporal hold duration
            hold_variation = config.get('hold_duration_variation', 0.3)
            if random.random() < 0.4:  # 40% chance
                # Vary finger closure timing
                variation_factor = 1.0 + random.uniform(-hold_variation, hold_variation)
                new_length = max(int(SEQUENCE_LENGTH * variation_factor), 1)  # Ensure at least 1
                
                if new_length > SEQUENCE_LENGTH:
                    # Extend: pad with last few frames
                    extension = new_length - SEQUENCE_LENGTH
                    last_frames = augmented[-extension:]
                    augmented = torch.cat([augmented, last_frames], dim=0)
                    # Take random subsequence
                    start_idx = random.randint(0, len(augmented) - SEQUENCE_LENGTH)
                    augmented = augmented[start_idx:start_idx + SEQUENCE_LENGTH]
                elif new_length < SEQUENCE_LENGTH:
                    # Shorten: compress sequence
                    step = SEQUENCE_LENGTH / new_length
                    indices = torch.linspace(0, SEQUENCE_LENGTH-1, SEQUENCE_LENGTH).long()
                    augmented = augmented[indices]
        
        elif gesture_name in ['swipe_left', 'swipe_right']:
            # Fix confusion between left/right: enhance trajectory smoothness
            smoothness = config.get('trajectory_smoothness', 0.8)
            if random.random() < 0.35:  # 35% chance
                # Add trajectory noise to make discrimination harder/tougher
                trajectory_noise = np.random.normal(0, 1-smoothness, (SEQUENCE_LENGTH, 2))
                
                for frame_idx in range(SEQUENCE_LENGTH):
                    # Apply small trajectory perturbations to RGB frames
                    rgb_frame = augmented[frame_idx, :3].numpy()
                    # Simple noise addition instead of complex displacement
                    noise_level = (1 - smoothness) * 0.1  # Scale noise based on smoothness
                    noise = np.random.normal(0, noise_level, rgb_frame.shape)
                    distorted = np.clip(rgb_frame + noise, 0, 1)
                    augmented[frame_idx, :3] = torch.from_numpy(distorted)
        
        elif gesture_name in ['grab', 'pinch_select']:
            # Fix confusion: enhance finger features
            close_variation = config.get('finger_close_variation', 0.25)
            precision = config.get('pinch_precision', 0.9)
            
            if random.random() < 0.45:  # 45% chance
                # Vary finger closure timing
                close_timing = int(SEQUENCE_LENGTH * random.uniform(0.7, 0.9))
                
                for frame_idx in range(close_timing, SEQUENCE_LENGTH):
                    # Add noise to finger regions (would ideally target hand mask)
                    rgb_frame = augmented[frame_idx, :3]
                    noise_factor = random.uniform(-close_variation, close_variation)
                    rgb_frame = torch.clamp(rgb_frame + noise_factor * torch.rand_like(rgb_frame), 0, 1)
                    augmented[frame_idx, :3] = rgb_frame
        
        return augmented
    
    def apply_displacement_field(self, image: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """Apply displacement field to create elastic distortion effect."""
        # Handle both (C, H, W) and (H, W, C) formats
        if image.shape[0] == 3:  # Channels first (C, H, W)
            # Transpose to (H, W, C) for processing
            image = np.transpose(image, (1, 2, 0))
            channels_first = True
        else:
            channels_first = False
        
        h, w = image.shape[:2]
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Scale displacement to smaller values for subtle effect
        displacement = displacement * 0.01
        new_y = np.clip(y_coords + displacement[:, :, 1], 0, h-1).astype(int)
        new_x = np.clip(x_coords + displacement[:, :, 0], 0, w-1).astype(int)
        
        distorted = image[new_y, new_x]
        
        # Convert back to channels first if needed
        if channels_first:
            distorted = np.transpose(distorted, (2, 0, 1))
        
        return distorted

    def __len__(self) -> int:
        """Return dataset length."""
        if self.use_lazy_loading:
            return len(self.chunk_index_map)
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (window_tensor, label)
        """
        # Lazy loading: load chunk on-demand
        if self.use_lazy_loading:
            chunk_idx, local_idx, label = self.chunk_index_map[idx]
            
            # Check cache first
            if chunk_idx not in self.chunk_cache:
                # Load chunk
                chunk_windows_path = QUEST3_PROCESSED_DIR / f"{self.split}_windows_chunk_{chunk_idx}.pt"
                chunk_windows = torch.load(chunk_windows_path, weights_only=True)
                
                # Add to cache
                self.chunk_cache[chunk_idx] = chunk_windows
                
                # Limit cache size (keep only most recent chunks)
                if len(self.chunk_cache) > self.cache_size:
                    # Remove oldest chunk (first key)
                    oldest_key = next(iter(self.chunk_cache))
                    del self.chunk_cache[oldest_key]
            
            # Get window from cached chunk
            window = self.chunk_cache[chunk_idx][local_idx]  # (60, 4, 224, 224)
        else:
            # Regular loading
            window = self.windows[idx]  # (60, 4, 224, 224)
            label = self.labels[idx]    # scalar

        # Apply augmentations if training
        if self.augment:
            window = self.apply_augmentation(window)
            window = self.temporal_augmentation(window)

        # Apply Phase 2 targeted augmentations based on gesture type
        if self.augment and self.phase2_config:
            label_int = int(label.item())
            gesture_name = QUEST3_GESTURES[label_int]
            window = self.apply_phase2_augmentations(window, gesture_name, label_int)

        return window, label



def get_quest3_dataloaders(batch_size: int = 16,
                          num_workers: int = 0,  # Changed from 4 to 0 for memory efficiency
                          phase2_config: dict = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create Quest 3 dataloaders for train/val/test.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes (0 = single process, more memory efficient)
        phase2_config: Enhanced augmentation config for Phase 2 improvements
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = Quest3GestureDataset('train', augment=True, phase2_config=phase2_config)
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
