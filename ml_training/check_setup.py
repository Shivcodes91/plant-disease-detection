# check_setup.py
# -------------------------------------------------------
# This script checks that every package is installed
# correctly and that PyTorch can process images.
# Run with: python ml_training/check_setup.py
# -------------------------------------------------------

# 'import' loads a library so we can use its features
# Think of it like: "go get that toolbox from the shelf"
import sys
import torch          # our ML framework — everything AI runs through this
import torchvision    # image-specific tools built on top of PyTorch
import numpy as np    # fast number crunching — used everywhere in ML
import PIL            # Pillow — opens image files (jpg, png, etc.)
import fastapi        # our future API framework
import sqlalchemy     # our future database connector
import albumentations # image augmentation library
import timm           # pretrained model zoo (has EfficientNet)

print("=" * 55)
print("  PLANT DISEASE DETECTION — DAY 1 SETUP CHECK")
print("=" * 55)

# ----- CHECK PYTHON -----
# sys.version gives the full Python version string
print(f"\n✓ Python  : {sys.version.split()[0]}")

# ----- CHECK PYTORCH -----
# torch.__version__ is a string like "2.1.0"
print(f"✓ PyTorch : {torch.__version__}")

# torch.cuda.is_available() checks if an NVIDIA GPU is detected
# CUDA is NVIDIA's system that lets PyTorch use the GPU
gpu = torch.cuda.is_available()
if gpu:
    print(f"✓ GPU     : {torch.cuda.get_device_name(0)}")
else:
    # No GPU is completely fine — CPU works, just slower
    print("✓ GPU     : Not found — using CPU (perfectly fine!)")

# 'device' is a PyTorch object that tells operations where to run
# We'll use this everywhere in our code
device = torch.device("cuda" if gpu else "cpu")
print(f"✓ Device  : {device}")

# ----- WHAT IS A TENSOR? -----
# A tensor is PyTorch's core data structure
# Basically a multi-dimensional array of numbers
# An image becomes a tensor: shape [3, 224, 224]
#   → 3 color channels (Red, Green, Blue)
#   → 224 pixels tall
#   → 224 pixels wide
# torch.randn creates a tensor filled with random numbers
# from a normal distribution (bell curve centered at 0)
test_tensor = torch.randn(3, 224, 224)
print(f"\n✓ Test tensor shape  : {test_tensor.shape}")
print(f"  (3=RGB channels, 224x224=image size)")
print(f"  Min value: {test_tensor.min():.3f}")
print(f"  Max value: {test_tensor.max():.3f}")

# ----- TEST IMAGE OPENING -----
from PIL import Image

# Create a fake green 224x224 image (simulates a leaf photo)
# In real use: Image.open("path/to/leaf.jpg")
# "RGB" = Red Green Blue color mode
# color=(34, 139, 34) = forest green in RGB values
fake_leaf = Image.new("RGB", size=(224, 224), color=(34, 139, 34))
print(f"\n✓ Fake leaf image    : {fake_leaf.size} px, mode={fake_leaf.mode}")

# ----- WHAT IS A TRANSFORM PIPELINE? -----
# Raw images can't go into a model directly
# We must convert them into tensors and normalize them
# torchvision.transforms lets us chain these steps together
from torchvision import transforms

# transforms.Compose takes a LIST of steps and runs them in order
transform_pipeline = transforms.Compose([

    # Step 1: Resize
    # No matter what size photo the farmer uploads,
    # resize it to exactly 224x224 pixels
    # EfficientNet-B4 expects 380x380, but 224 works for testing
    transforms.Resize((224, 224)),

    # Step 2: ToTensor
    # Converts PIL Image → PyTorch tensor
    # Also scales pixel values: 0-255 becomes 0.0-1.0
    # Shape changes from (H, W, C) to (C, H, W) — PyTorch's format
    transforms.ToTensor(),

    # Step 3: Normalize
    # Shifts and scales values using ImageNet statistics
    # WHY? EfficientNet was pretrained on ImageNet dataset
    # It "learned" to expect these specific value ranges
    # If we don't normalize, the model gets confused
    # mean and std are per-channel (R, G, B)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # average pixel value per channel in ImageNet
        std=[0.229, 0.224, 0.225]    # spread of pixel values per channel
    )
])

# Apply the pipeline to our fake leaf image
image_tensor = transform_pipeline(fake_leaf)
print(f"\n✓ After transform:")
print(f"  Shape  : {image_tensor.shape}  ← [channels, height, width]")
print(f"  Values : {image_tensor.min():.3f} to {image_tensor.max():.3f}  ← normalized")

# ----- LOAD EFFICIENTNET-B4 -----
# timm (PyTorch Image Models) has hundreds of pretrained models
# EfficientNet-B4 is our choice — good balance of speed and accuracy
# pretrained=False → just load the architecture, no downloaded weights
# num_classes=38 → 38 types of plant diseases in our dataset
print(f"\n✓ Loading EfficientNet-B4 architecture...")
model = timm.create_model(
    'efficientnet_b4',
    pretrained=False,
    num_classes=38
)

# Count total parameters (weights the model learns during training)
# p.numel() = number of elements in that parameter tensor
# sum(...) adds them all up
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters    : {total_params:,}")
print(f"  Trainable parameters: {trainable:,}")

# ----- RUN A FORWARD PASS -----
# A forward pass = feeding an image through the model
# to get a prediction (38 scores, one per disease)

# unsqueeze(0) adds a batch dimension
# Models process BATCHES of images, not single images
# Shape goes from [3, 224, 224] → [1, 3, 224, 224]
#                                   ^ batch size of 1
batch = image_tensor.unsqueeze(0)
print(f"\n✓ Input batch shape  : {batch.shape}  ← [batch, channels, H, W]")

# model.eval() = evaluation mode
# Turns off Dropout and BatchNorm training behavior
# Always use this when NOT training
model.eval()

# torch.no_grad() = don't track gradients
# Gradients are needed during training to update weights
# During testing/inference we don't need them → saves memory
with torch.no_grad():
    output = model(batch)

# output shape: [1, 38]
# 1 = batch size, 38 = one score per disease class
print(f"✓ Output shape       : {output.shape}  ← [batch, 38 classes]")

# Apply softmax to convert raw scores → probabilities (0 to 1, sum to 1)
# dim=1 means apply softmax across the 38 class dimension
probabilities = torch.softmax(output, dim=1)
# Get the highest probability and its index
top_prob, top_class = torch.max(probabilities, dim=1)
print(f"✓ Predicted class    : {top_class.item()} (confidence: {top_prob.item():.2%})")
print(f"  (Random model = random prediction — this is expected!)")

# ----- FINAL SUMMARY -----
print("\n" + "=" * 55)
print("  ALL CHECKS PASSED!")
print("  Your machine is ready for Day 2.")
print("  Next: Download dataset & explore the data.")
print("=" * 55)