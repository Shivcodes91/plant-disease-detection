# =============================================================
# ml_training/dataset.py
#
# This file contains everything related to loading and
# preparing our plant disease images for training.
#
# KEY CONCEPTS:
# - Dataset class: tells PyTorch HOW to load one image
# - DataLoader: tells PyTorch HOW MANY images to load at once
# - Augmentation: artificially creates image variations
# - Train/Val/Test split: divides data for training and testing
# =============================================================

# ---- IMPORTS ----
import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =============================================================
# SECTION 1: CLASS MAPPING
# =============================================================
# Our model outputs a NUMBER (0-14), not a disease name
# This dictionary maps numbers ↔ disease names
# We build it automatically from the dataset folders

def build_class_mapping(dataset_path: str) -> Tuple[Dict, Dict]:
    """
    Scans the dataset folder and builds two dictionaries:
    1. class_to_idx: disease name → number  e.g. "Tomato_Early_blight" → 3
    2. idx_to_class: number → disease name  e.g. 3 → "Tomato_Early_blight"
    
    Args:
        dataset_path: path to folder containing disease subfolders
    
    Returns:
        Tuple of (class_to_idx, idx_to_class)
    """
    path = Path(dataset_path)
    
    # Get all subfolders, sorted alphabetically
    # Sorting ensures the mapping is always consistent
    # (same disease always gets the same number)
    class_names = sorted([
        d.name for d in path.iterdir()
        if d.is_dir() and len(list(d.glob('*.jpg')) + 
                               list(d.glob('*.JPG'))) > 0
    ])
    
    # Build forward mapping: name → index
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Build reverse mapping: index → name
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    return class_to_idx, idx_to_class


# =============================================================
# SECTION 2: COLLECT ALL IMAGE PATHS
# =============================================================

def collect_image_paths(dataset_path: str, 
                        class_to_idx: Dict) -> Tuple[List, List]:
    """
    Walks through all disease folders and collects:
    - Path to every image file
    - The class label (number) for that image
    
    Think of this as building a master list:
    [
        ("data/raw/.../Tomato_Early_blight/img001.jpg", 3),
        ("data/raw/.../Tomato_healthy/img002.jpg", 7),
        ...
    ]
    
    Args:
        dataset_path: path to dataset root
        class_to_idx: mapping from class name to index
    
    Returns:
        Tuple of (image_paths list, labels list)
    """
    path = Path(dataset_path)
    image_paths = []
    labels = []
    
    for class_name, class_idx in class_to_idx.items():
        class_folder = path / class_name
        
        if not class_folder.exists():
            print(f"  ⚠ Folder not found: {class_name}")
            continue
        
        # Collect all image files in this folder
        # We check multiple extensions to be thorough
        images = (list(class_folder.glob('*.jpg')) +
                  list(class_folder.glob('*.JPG')) +
                  list(class_folder.glob('*.jpeg')) +
                  list(class_folder.glob('*.png')))
        
        # Add each image path and its label to our lists
        for img_path in images:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"✓ Collected {len(image_paths):,} images across "
          f"{len(class_to_idx)} classes")
    
    return image_paths, labels


# =============================================================
# SECTION 3: AUGMENTATION PIPELINES
# =============================================================
# Augmentation = artificially creating new image variations
# 
# WHY? Our model must work on photos taken by farmers with:
# - Different phone cameras
# - Different lighting (morning, noon, cloudy)
# - Different angles (top-down, side view)
# - Different distances (close up, far away)
#
# We simulate all these variations during training so the
# model becomes robust to real-world conditions.
#
# We use ALBUMENTATIONS library — faster than torchvision
# and has more augmentation options

def get_train_transforms(image_size: int = 224):
    """
    Augmentation pipeline for TRAINING data.
    Applies random transformations to create variety.
    
    Each augmentation has a probability (p=0.5 means 50% chance
    of being applied to any given image).
    """
    return A.Compose([
        
        # ---- SPATIAL TRANSFORMS ----
        # These change the position/orientation of the image
        
        # Resize to our target size first
        A.Resize(image_size, image_size),
        
        # Randomly flip horizontally (mirror image)
        # A diseased leaf looks the same mirrored
        # p=0.5 = applied to 50% of images
        A.HorizontalFlip(p=0.5),
        
        # Randomly flip vertically
        # Less common but still valid
        A.VerticalFlip(p=0.3),
        
        # Randomly rotate by up to 45 degrees
        # Farmers don't always hold phone perfectly straight
        A.Rotate(limit=45, p=0.5),
        
        # Random zoom in/out and shift
        # ShiftScaleRotate combines shift + scale + rotate
       A.Affine(
    translate_percent=0.1,
    scale=(0.8, 1.2),
    rotate=(-15, 15),
    p=0.5
),
        
        # ---- COLOR/LIGHTING TRANSFORMS ----
        # These simulate different lighting conditions
        
        # Randomly change brightness, contrast, saturation, hue
        # brightness_limit=0.3 means ±30% brightness change
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.5
        ),
        
        # Randomly convert to grayscale
        # Forces model to learn shape, not just color
        A.ToGray(p=0.1),
        
        # ---- NOISE/BLUR TRANSFORMS ----
        # These simulate camera quality differences
        
        # Add random Gaussian noise (simulates cheap cameras)
        A.GaussNoise(p=0.2),
        
        # Random blur (simulates out-of-focus photos)
        A.OneOf([
            A.MotionBlur(blur_limit=5),   # motion blur
            A.GaussianBlur(blur_limit=5), # gaussian blur
        ], p=0.2),
        
        # ---- CUTOUT TRANSFORMS ----
        # These force the model to not rely on one single feature
        
        # Randomly erase rectangular regions of the image
        # Forces model to look at the whole leaf, not just one spot
    A.CoarseDropout(
    num_holes_range=(1, 8),
    hole_height_range=(8, 16),
    hole_width_range=(8, 16),
    fill=0,
    p=0.3
),
        
        # ---- NORMALIZATION ----
        # Must always be last
        # Normalize using ImageNet statistics
        # (because EfficientNet was pretrained on ImageNet)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # Convert numpy array → PyTorch tensor
        # Changes shape from (H, W, C) to (C, H, W)
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224):
    """
    Transform pipeline for VALIDATION and TEST data.
    
    IMPORTANT: No random augmentations here!
    We want consistent, reproducible results when evaluating.
    Only resize and normalize — nothing random.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


# =============================================================
# SECTION 4: CUSTOM DATASET CLASS
# =============================================================
# PyTorch's Dataset class is like a smart list of images.
# It knows:
# 1. How many images there are (__len__)
# 2. How to load and return image number N (__getitem__)
#
# The DataLoader uses these two methods to load batches.

class PlantDiseaseDataset(Dataset):
    """
    Custom PyTorch Dataset for Plant Disease images.
    
    PyTorch's DataLoader calls this class to get images.
    It asks for image[0], image[1], image[2]... etc.
    We tell it how to load and preprocess each one.
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[int],
                 transform=None,
                 class_to_idx: Dict = None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: list of file paths to all images
            labels: list of class indices (matching image_paths)
            transform: augmentation pipeline to apply
            class_to_idx: mapping from class name to index
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx
        
        # Validate that paths and labels match
        assert len(image_paths) == len(labels), \
            f"Mismatch: {len(image_paths)} images but {len(labels)} labels"
        
        print(f"  Dataset created: {len(self.image_paths):,} images")
    
    def __len__(self) -> int:
        """
        Returns total number of images in dataset.
        DataLoader calls this to know when to stop.
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads and returns ONE image and its label.
        
        This is called thousands of times during training.
        DataLoader calls this with idx=0, 1, 2, 3...
        
        Args:
            idx: index of the image to load
        
        Returns:
            Tuple of (image_tensor, label)
        """
        # Get the file path for this index
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open the image file
        # convert('RGB') ensures we always get 3 channels
        # Some images might be RGBA (4 channels) or grayscale (1 channel)
        # We force RGB for consistency
        image = Image.open(img_path).convert('RGB')
        
        # Convert PIL image to numpy array
        # Albumentations works with numpy arrays, not PIL images
        # np.array() converts: PIL Image → numpy array of shape (H, W, 3)
        image = np.array(image)
        
        # Apply augmentation transforms if provided
        if self.transform:
            # Albumentations expects: image=numpy_array
            # It returns a dictionary with 'image' key
            augmented = self.transform(image=image)
            image = augmented['image']  # extract the transformed image
        
        # 'image' is now a PyTorch tensor of shape (3, H, W)
        # 'label' is an integer like 3 or 7
        return image, label


# =============================================================
# SECTION 5: DATA SPLITTING
# =============================================================

def split_data(image_paths: List[str], 
               labels: List[int],
               train_ratio: float = 0.70,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               random_seed: int = 42) -> Dict:
    """
    Splits data into train, validation, and test sets.
    
    WHY THREE SPLITS?
    - Train (70%): Model learns from these images
    - Validation (15%): We check accuracy during training
                        (model never trains on these)
    - Test (15%): Final evaluation after all training done
                  (completely unseen data)
    
    WHY stratify?
    - Ensures each split has proportional class representation
    - Without this, one split might have no Potato images!
    
    Args:
        image_paths: all image file paths
        labels: all labels
        train_ratio: fraction for training (0.70 = 70%)
        val_ratio: fraction for validation
        test_ratio: fraction for testing
        random_seed: for reproducibility (same split every run)
    
    Returns:
        Dictionary with train/val/test paths and labels
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: separate test set from the rest
    # test_size=test_ratio means 15% goes to test
    # stratify=labels ensures balanced class distribution
    train_val_paths, test_paths, train_val_labels, test_labels = \
        train_test_split(
            image_paths, labels,
            test_size=test_ratio,
            stratify=labels,        # maintain class balance
            random_state=random_seed
        )
    
    # Second split: separate validation from training
    # val_size = val_ratio / (train_ratio + val_ratio)
    # e.g. 0.15 / (0.70 + 0.15) = 0.176
    val_size = val_ratio / (train_ratio + val_ratio)
    
    train_paths, val_paths, train_labels, val_labels = \
        train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_size,
            stratify=train_val_labels,
            random_state=random_seed
        )
    
    splits = {
        'train': {'paths': train_paths, 'labels': train_labels},
        'val':   {'paths': val_paths,   'labels': val_labels},
        'test':  {'paths': test_paths,  'labels': test_labels}
    }
    
    print(f"\n✓ Data split complete:")
    print(f"   Train : {len(train_paths):,} images ({train_ratio*100:.0f}%)")
    print(f"   Val   : {len(val_paths):,} images ({val_ratio*100:.0f}%)")
    print(f"   Test  : {len(test_paths):,} images ({test_ratio*100:.0f}%)")
    
    return splits


# =============================================================
# SECTION 6: WEIGHTED SAMPLER (handles class imbalance)
# =============================================================

def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to handle class imbalance.
    
    PROBLEM: Tomato YellowLeaf has 6,416 images but
             Potato healthy has only 304 images.
    
    Without fixing this, the model sees Tomato YellowLeaf
    21x more than Potato healthy → gets biased!
    
    SOLUTION: WeightedRandomSampler gives rare classes
    a higher chance of being selected in each batch.
    This way every class gets roughly equal attention.
    
    Args:
        labels: list of class indices for training set
    
    Returns:
        WeightedRandomSampler object
    """
    # Count images per class
    # np.bincount([0,1,1,2,0]) → [2, 2, 1] (counts per class)
    class_counts = np.bincount(labels)
    
    # Calculate weight per class
    # Classes with fewer images get higher weight
    # e.g. if Potato_healthy has 213 train images:
    #   weight = 1/213 = 0.0047 (high weight → sampled more)
    # if Tomato_YellowLeaf has 4491 train images:
    #   weight = 1/4491 = 0.00022 (low weight → sampled less)
    class_weights = 1.0 / class_counts
    
    # Assign weight to each individual sample
    # Every image gets the weight of its class
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # Create the sampler
    # num_samples = total samples to draw per epoch
    # replacement=True means same image can be picked again
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"✓ Weighted sampler created")
    print(f"   Most common class weight : {class_weights.min():.6f}")
    print(f"   Rarest class weight      : {class_weights.max():.6f}")
    
    return sampler


# =============================================================
# SECTION 7: CREATE DATALOADERS
# =============================================================

def create_dataloaders(dataset_path: str,
                       image_size: int = 224,
                       batch_size: int = 32,
                       num_workers: int = 0) -> Dict:
    """
    Master function that does everything:
    1. Builds class mapping
    2. Collects all image paths
    3. Splits into train/val/test
    4. Creates Dataset objects
    5. Creates DataLoader objects
    
    This is the function we'll call from our training script.
    
    Args:
        dataset_path: path to PlantVillage folder
        image_size: resize all images to this size (224x224)
        batch_size: how many images per batch (32 is standard)
        num_workers: parallel data loading workers
                     (0 = main thread, safe for Windows)
    
    Returns:
        Dictionary with everything needed for training
    """
    print("=" * 55)
    print("  BUILDING DATA PIPELINE")
    print("=" * 55)
    
    # Step 1: Build class mapping
    print("\n[1/5] Building class mapping...")
    class_to_idx, idx_to_class = build_class_mapping(dataset_path)
    num_classes = len(class_to_idx)
    print(f"      {num_classes} classes found")
    
    # Step 2: Collect all image paths and labels
    print("\n[2/5] Collecting image paths...")
    image_paths, labels = collect_image_paths(dataset_path, class_to_idx)
    
    # Step 3: Split data
    print("\n[3/5] Splitting data...")
    splits = split_data(image_paths, labels)
    
    # Step 4: Create augmentation transforms
    print("\n[4/5] Setting up augmentation transforms...")
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    print("      Train transforms: 10 augmentations active")
    print("      Val transforms  : resize + normalize only")
    
    # Step 5: Create Dataset objects
    print("\n[5/5] Creating DataLoaders...")
    
    train_dataset = PlantDiseaseDataset(
        image_paths=splits['train']['paths'],
        labels=splits['train']['labels'],
        transform=train_transform,
        class_to_idx=class_to_idx
    )
    
    val_dataset = PlantDiseaseDataset(
        image_paths=splits['val']['paths'],
        labels=splits['val']['labels'],
        transform=val_transform,
        class_to_idx=class_to_idx
    )
    
    test_dataset = PlantDiseaseDataset(
        image_paths=splits['test']['paths'],
        labels=splits['test']['labels'],
        transform=val_transform,  # same as val — no augmentation
        class_to_idx=class_to_idx
    )
    
    # Create weighted sampler for training
    # (handles class imbalance)
    train_sampler = create_weighted_sampler(splits['train']['labels'])
    
    # Create DataLoaders
    # DataLoader wraps a Dataset and handles:
    # - Batching (groups images into batches of batch_size)
    # - Shuffling (randomizes order each epoch)
    # - Parallel loading (num_workers processes)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,    # use weighted sampler (not shuffle)
        num_workers=num_workers,
        pin_memory=True,          # speeds up GPU transfer
        drop_last=True            # drop incomplete final batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,            # no shuffle for validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n{'='*55}")
    print(f"  DATA PIPELINE READY!")
    print(f"{'='*55}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Train batches   : {len(train_loader)}")
    print(f"  Val batches     : {len(val_loader)}")
    print(f"  Test batches    : {len(test_loader)}")
    print(f"  Image size      : {image_size}x{image_size}")
    print(f"  Classes         : {num_classes}")
    
    # Save class mapping to JSON file
    # We'll need this later when making predictions
    mapping_path = Path('data') / 'class_mapping.json'
    mapping_path.parent.mkdir(exist_ok=True)
    
    with open(mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {str(k): v for k, v in idx_to_class.items()},
            'num_classes': num_classes
        }, f, indent=2)
    print(f"\n✓ Class mapping saved to: {mapping_path}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': num_classes,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }