# =============================================================
# ml_training/test_pipeline.py
#
# Tests our data pipeline to make sure everything works.
# Run with: python ml_training/test_pipeline.py
# =============================================================

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add ml_training to path so we can import dataset.py
sys.path.append(str(Path(__file__).parent))
from dataset import create_dataloaders

# =============================================================
# CONFIGURATION
# =============================================================
# Update this path to match where your dataset actually is
DATASET_PATH = "data/raw/PlantVillage/PlantVillage"

IMAGE_SIZE  = 224   # pixels
BATCH_SIZE  = 32    # images per batch
NUM_WORKERS = 0     # 0 is safe for Windows

# =============================================================
# BUILD THE PIPELINE
# =============================================================
print("Building data pipeline...")
data = create_dataloaders(
    dataset_path=DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# =============================================================
# TEST 1: Check one batch loads correctly
# =============================================================
print("\n" + "="*55)
print("TEST 1: Loading one batch")
print("="*55)

# iter() creates an iterator from the dataloader
# next() gets the first batch
train_loader = data['train_loader']
images, labels = next(iter(train_loader))

# images shape: [32, 3, 224, 224]
# 32 = batch size, 3 = RGB channels, 224x224 = image size
print(f"✓ Batch images shape : {images.shape}")
print(f"✓ Batch labels shape : {labels.shape}")
print(f"✓ Image dtype        : {images.dtype}")
print(f"✓ Label dtype        : {labels.dtype}")
print(f"✓ Image value range  : {images.min():.3f} to {images.max():.3f}")
print(f"✓ Unique labels in batch: {labels.unique().tolist()}")

# =============================================================
# TEST 2: Visualize augmented images
# =============================================================
print("\n" + "="*55)
print("TEST 2: Visualizing augmented training images")
print("="*55)

# We need to DENORMALIZE the images to display them
# During preprocessing we normalized: (pixel - mean) / std
# To display, we reverse: pixel * std + mean
def denormalize(tensor):
    """Reverse normalization for display purposes"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # Reverse the normalization formula
    return torch.clamp(tensor * std + mean, 0, 1)

idx_to_class = data['idx_to_class']

# Display 16 images from the batch in a 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
axes_flat = axes.flatten()

for i in range(16):
    ax = axes_flat[i]
    
    # Get image and denormalize
    img_tensor = images[i]
    img_display = denormalize(img_tensor)
    
    # permute changes shape from (C,H,W) to (H,W,C)
    # matplotlib expects (H, W, C) format
    img_np = img_display.permute(1, 2, 0).numpy()
    
    ax.imshow(img_np)
    
    # Get class name for this label
    label_idx = labels[i].item()  # .item() converts tensor → python int
    class_name = idx_to_class[label_idx]
    
    # Clean up name for display
    clean_name = class_name.replace('___', ' ').replace('__',' ').replace('_', ' ')
    
    # Color: green for healthy, red for diseased
    color = 'green' if 'healthy' in class_name.lower() else 'red'
    
    ax.set_title(clean_name, fontsize=7, color=color, fontweight='bold')
    ax.axis('off')

fig.suptitle('Sample Training Batch — Augmented Images\n(Notice: some are flipped, rotated, or color-shifted)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data/sample_batch.png', dpi=120, bbox_inches='tight')
plt.show()
print("✓ Sample batch saved to data/sample_batch.png")

# =============================================================
# TEST 3: Check class balance
# =============================================================
print("\n" + "="*55)
print("TEST 3: Checking class distribution in train set")
print("="*55)

# Count label distribution across multiple batches
from collections import Counter
label_counts = Counter()

# Check first 50 batches
print("Sampling 50 batches to check class balance...")
for i, (_, batch_labels) in enumerate(train_loader):
    label_counts.update(batch_labels.tolist())
    if i >= 49:  # stop after 50 batches
        break

print("\nClass distribution (50 batches sampled):")
for idx in sorted(label_counts.keys()):
    class_name = idx_to_class[idx]
    count = label_counts[idx]
    bar = '█' * (count // 5)
    print(f"  {class_name:45} | {count:4} | {bar}")

print("\n✓ If counts are roughly equal → weighted sampler working!")

# =============================================================
# FINAL SUMMARY
# =============================================================
print("\n" + "="*55)
print("  ALL PIPELINE TESTS PASSED!")
print("="*55)
print(f"""
Pipeline is ready for training:
  • Train  : {data['train_size']:,} images
  • Val    : {data['val_size']:,} images  
  • Test   : {data['test_size']:,} images
  • Classes: {data['num_classes']}
  • Augmentations: flips, rotations, color jitter, noise

Next step → Day 4: Build the model!
""")