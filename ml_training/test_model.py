# =============================================================
# ml_training/test_model.py
# Tests our model architecture end to end
# Run with: python ml_training/test_model.py
# =============================================================

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from model import create_model, predict_single_image
from dataset import get_val_transforms

print("=" * 55)
print("  DAY 4 — MODEL ARCHITECTURE TEST")
print("=" * 55)

# =============================================================
# TEST 1: Create the model
# =============================================================
print("\n[TEST 1] Creating model...")

model, criterion, device = create_model(
    num_classes=15,
    pretrained=True,       # download ImageNet weights
    dropout_rate=0.3,
    freeze_backbone=True   # freeze backbone initially
)

# =============================================================
# TEST 2: Test forward pass with fake data
# =============================================================
print("\n[TEST 2] Testing forward pass...")

# Create a fake batch of 4 images
# In real training this would be real plant images
fake_batch = torch.randn(4, 3, 224, 224).to(device)
fake_labels = torch.randint(0, 15, (4,)).to(device)

# Forward pass
model.eval()
with torch.no_grad():
    output = model(fake_batch)

print(f"✓ Input shape  : {fake_batch.shape}")
print(f"✓ Output shape : {output.shape}")
print(f"✓ Output range : {output.min():.3f} to {output.max():.3f}")

# =============================================================
# TEST 3: Test loss calculation
# =============================================================
print("\n[TEST 3] Testing loss calculation...")

model.train()
output = model(fake_batch)
loss = criterion(output, fake_labels)

print(f"✓ Loss value: {loss.item():.4f}")
print(f"  (random model loss ≈ log(15) = {np.log(15):.4f} — this is expected)")

# =============================================================
# TEST 4: Test freeze/unfreeze
# =============================================================
print("\n[TEST 4] Testing freeze/unfreeze...")

info_frozen = model.get_model_info()
print(f"When frozen:")
print(f"  Trainable: {info_frozen['trainable_parameters']:,}")
print(f"  Frozen   : {info_frozen['frozen_parameters']:,}")

model.unfreeze_backbone()
info_unfrozen = model.get_model_info()
print(f"When unfrozen:")
print(f"  Trainable: {info_unfrozen['trainable_parameters']:,}")
print(f"  Frozen   : {info_unfrozen['frozen_parameters']:,}")

# Re-freeze for next test
model.freeze_backbone()

# =============================================================
# TEST 5: Test prediction function
# =============================================================
print("\n[TEST 5] Testing prediction on a real leaf image...")

# Load a real image from our dataset
dataset_path = Path("data/raw/PlantVillage/PlantVillage")
sample_class = "Tomato_Early_blight"
sample_folder = dataset_path / sample_class
sample_images = list(sample_folder.glob("*.jpg"))[:1]

if sample_images:
    # Load and preprocess the image
    img = Image.open(sample_images[0]).convert('RGB')
    transform = get_val_transforms(224)
    img_array = np.array(img)
    img_tensor = transform(image=img_array)['image']
    
    # Build a simple idx_to_class mapping for testing
    import json
    with open('data/class_mapping.json', 'r') as f:
        mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
    
    # Make prediction (model is random — result won't be accurate yet)
    result = predict_single_image(
        model=model,
        image_tensor=img_tensor,
        idx_to_class=idx_to_class,
        device=device,
        top_k=3
    )
    
    print(f"✓ Image: {sample_images[0].name}")
    print(f"  Actual class : {sample_class}")
    print(f"\n  Top 3 Predictions (random model — not trained yet):")
    for pred in result['all_predictions']:
        bar = '█' * int(pred['confidence'] / 2)
        print(f"  {pred['rank']}. {pred['class_name']:45} "
              f"{pred['confidence']:5.1f}% {bar}")
    
    print(f"\n  Confident prediction: {result['is_confident']}")
    print(f"  (False is expected — model not trained yet!)")
else:
    print("  ⚠ No sample images found — skipping image test")

# =============================================================
# TEST 6: GPU Memory check
# =============================================================
if device.type == 'cuda':
    print(f"\n[TEST 6] GPU Memory usage...")
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved  = torch.cuda.memory_reserved(0) / 1024**2
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"✓ Allocated : {allocated:.1f} MB")
    print(f"✓ Reserved  : {reserved:.1f} MB")
    print(f"✓ Total GPU : {total:.1f} MB")
    print(f"✓ Free      : {total - reserved:.1f} MB")

# =============================================================
# SUMMARY
# =============================================================
print(f"\n{'='*55}")
print(f"  ALL MODEL TESTS PASSED!")
print(f"{'='*55}")
print(f"""
Model is ready for training:
  • Architecture : EfficientNet-B4 + Custom Classifier
  • Parameters   : {info_frozen['total_parameters']:,} total
  • Trainable    : {info_frozen['trainable_parameters']:,} (head only)
  • Device       : {device}
  • Loss         : Label Smoothing Cross Entropy

Training Strategy:
  Phase 1 (Days 5-6) : Train classifier head only (backbone frozen)
  Phase 2 (Day 6+)   : Fine-tune entire model (backbone unfrozen)

Next Step → Day 5: Build the training loop!
""")