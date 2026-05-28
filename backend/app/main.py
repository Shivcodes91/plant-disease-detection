# =============================================================
# backend/app/main.py
#
# FastAPI Backend for Plant Disease Detection
#
# ENDPOINTS:
# GET  /              → health check
# GET  /diseases      → list all 15 diseases
# GET  /remedies/{class_name} → full remedy for a disease
# POST /predict       → upload image → get prediction + remedy
#
# Run with: python run_api.py
# =============================================================

import io
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path so we can import our modules
# This file is at backend/app/main.py
# parent.parent.parent = project root
import sys
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(BASE_DIR))

from ml_training.model import load_checkpoint
from ml_training.dataset import get_val_transforms
from ml_training.gradcam import GradCAM, apply_heatmap
from backend.app.db.remedies_db import get_quick_summary, get_all_diseases
import cv2
import base64

# =============================================================
# APP INITIALIZATION
# =============================================================

app = FastAPI(
    title="Plant Disease Detection API",
    description="""
    AI-powered plant disease detection system.
    Upload a leaf image to get:
    - Disease prediction with confidence score
    - Grad-CAM heatmap showing disease location
    - Full treatment and remedy information
    """,
    version="1.0.0"
)

# CORS — allows our Streamlit frontend to call this API
# Without this, browser blocks cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, specify exact URLs
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# GLOBAL VARIABLES
# Loaded once at startup, reused for every request
# Loading on every request would be 10-20 seconds per call
# =============================================================

MODEL       = None   # our trained EfficientNet-B4
IDX_TO_CLASS = None  # mapping: 0 → "Tomato_Early_blight"
DEVICE      = None   # cuda or cpu
TRANSFORM   = None   # image preprocessing pipeline


def load_model():
    """
    Load model into memory at server startup.

    Uses absolute paths so it works both:
    - Locally on Windows (your machine)
    - On Render's Linux server (deployment)

    Path logic:
    This file: backend/app/main.py
    parent     = backend/app/
    parent.parent = backend/
    parent.parent.parent = project root ← BASE_DIR
    """
    global MODEL, IDX_TO_CLASS, DEVICE, TRANSFORM

    print("=" * 55)
    print("  LOADING PLANT DISEASE MODEL")
    print("=" * 55)

    # ---- DEVICE SETUP ----
    # Use GPU if available (local), otherwise CPU (Render free tier)
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"\n  Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- BUILD ABSOLUTE PATHS ----
    # Path(__file__) = absolute path to this file
    # .resolve() = converts to absolute, resolves symlinks
    # Works on both Windows and Linux
    model_path   = BASE_DIR / 'backend' / 'models' / 'best_model.pt'
    mapping_path = BASE_DIR / 'data' / 'class_mapping.json'

    print(f"\n  Base directory : {BASE_DIR}")
    print(f"  Model path     : {model_path}")
    print(f"  Mapping path   : {mapping_path}")

    # ---- VERIFY FILES EXIST ----
    if not model_path.exists():
        raise FileNotFoundError(
            f"\n❌ Model file not found!\n"
            f"   Expected at: {model_path}\n"
            f"   Make sure best_model.pt is committed to GitHub\n"
            f"   and not in .gitignore"
        )

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"\n❌ Class mapping not found!\n"
            f"   Expected at: {mapping_path}"
        )

    # ---- LOAD MODEL ----
    print(f"\n  Loading EfficientNet-B4 checkpoint...")
    MODEL, checkpoint = load_checkpoint(
        str(model_path),
        device=DEVICE
    )
    MODEL.eval()  # set to evaluation mode (no dropout, stable batchnorm)

    # ---- LOAD CLASS MAPPING ----
    # idx_to_class: {0: "Pepper__bell___Bacterial_spot", 1: "...", ...}
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    IDX_TO_CLASS = {
        int(k): v
        for k, v in mapping['idx_to_class'].items()
    }

    # ---- LOAD TRANSFORM PIPELINE ----
    # Same preprocessing used during training:
    # resize → normalize with ImageNet mean/std → tensor
    TRANSFORM = get_val_transforms(224)

    print(f"\n{'='*55}")
    print(f"  MODEL READY!")
    print(f"{'='*55}")
    print(f"  Val accuracy : {checkpoint['val_accuracy']:.2f}%")
    print(f"  Classes      : {len(IDX_TO_CLASS)}")
    print(f"  Device       : {DEVICE}")
    print(f"  Epoch        : {checkpoint['epoch']}")


# ---- RUN LOAD ON STARTUP ----
@app.on_event("startup")
async def startup_event():
    """Called automatically when FastAPI server starts."""
    load_model()


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def preprocess_image(image_bytes: bytes):
    """
    Convert uploaded image bytes → model-ready tensor.

    Steps:
    1. Open bytes as PIL Image
    2. Convert to RGB (handles RGBA, grayscale, etc.)
    3. Resize to 224x224 for display
    4. Apply val transforms (resize + normalize → tensor)

    Args:
        image_bytes: raw bytes from uploaded file

    Returns:
        Tuple of (image_tensor, original_numpy_array)
    """
    # Open image from raw bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Convert PIL → numpy array (albumentations needs numpy)
    img_array = np.array(image)

    # Keep a resized copy for Grad-CAM overlay
    original = cv2.resize(img_array, (224, 224))

    # Apply preprocessing transforms
    # Returns dict with 'image' key containing the tensor
    transformed = TRANSFORM(image=img_array)
    tensor = transformed['image']

    return tensor, original


def tensor_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy RGB image → base64 string.

    We send images as base64 in JSON so the frontend
    can display them without needing file storage.

    Args:
        image_array: RGB numpy array [H, W, 3]

    Returns:
        Data URL string: "data:image/jpeg;base64,/9j/..."
    """
    # cv2 works with BGR, convert from RGB
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Encode to JPEG bytes (quality 85 = good balance)
    _, buffer = cv2.imencode(
        '.jpg', bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    # Convert bytes → base64 string
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def generate_gradcam_base64(
    image_tensor: torch.Tensor,
    original_image: np.ndarray
) -> Optional[str]:
    """
    Generate Grad-CAM attention heatmap and return as base64.

    Grad-CAM shows WHICH parts of the leaf the model focused
    on when making its prediction. Red = high attention.

    Args:
        image_tensor: preprocessed tensor [3, 224, 224]
        original_image: original numpy image for overlay

    Returns:
        Base64 heatmap overlay, or None if error
    """
    try:
        # Create GradCAM instance with hooks into last conv layer
        gradcam = GradCAM(MODEL, DEVICE)

        # Generate the attention map
        # cam = 2D numpy array [H, W] with values 0-1
        cam, _, _ = gradcam.generate(image_tensor)

        # Remove hooks to free memory
        gradcam.remove_hooks()

        # Overlay heatmap on original image
        # alpha=0.4 means 40% heatmap, 60% original
        overlay = apply_heatmap(original_image, cam, alpha=0.4)

        return tensor_to_base64(overlay)

    except Exception as e:
        print(f"⚠ Grad-CAM error: {e}")
        return None


# =============================================================
# API ENDPOINTS
# =============================================================

@app.get("/")
async def root():
    """
    Health check endpoint.
    Call this to verify the API is running and model is loaded.
    """
    return {
        "status": "online",
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }


@app.get("/diseases")
async def get_diseases():
    """
    Returns list of all 15 detectable diseases.

    Used by the Disease Library page in the frontend.
    Includes: class_name, display_name, plant, severity, is_healthy
    """
    diseases = get_all_diseases()
    return {
        "total": len(diseases),
        "diseases": diseases
    }


@app.get("/remedies/{class_name}")
async def get_remedy(class_name: str):
    """
    Returns full remedy information for a specific disease.

    Args:
        class_name: exact class name e.g. Tomato_Early_blight

    Returns:
        Full remedy dict with symptoms, treatments, prevention, timeline
    """
    from backend.app.db.remedies_db import get_remedy as _get_remedy
    remedy = _get_remedy(class_name)

    if not remedy or not remedy.get('found'):
        raise HTTPException(
            status_code=404,
            detail=f"Remedy not found for: {class_name}"
        )
    return remedy


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_gradcam: bool = True
):
    """
    MAIN ENDPOINT — Image → Disease Prediction + Remedy.

    Flow:
    1. Validate uploaded file (type, size)
    2. Preprocess image (resize, normalize, tensorize)
    3. Run model inference → 15 class scores
    4. Get top 3 predictions with confidence
    5. Look up remedy for top prediction
    6. Generate Grad-CAM heatmap
    7. Return everything as JSON

    Args:
        file: uploaded image (jpg, png, webp)
        include_gradcam: set False to skip heatmap (faster)

    Returns:
        JSON with prediction, top_predictions, remedy, gradcam_image
    """
    start_time = time.time()

    # ---- VALIDATE: File must be an image ----
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, webp, etc.)"
        )

    # ---- READ FILE ----
    image_bytes = await file.read()

    # ---- VALIDATE: Max 10MB ----
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image too large. Maximum size is 10MB."
        )

    try:
        # ---- PREPROCESS ----
        image_tensor, original_image = preprocess_image(
            image_bytes
        )

        # ---- INFERENCE ----
        # Add batch dimension: [3,224,224] → [1,3,224,224]
        image_batch = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():  # no gradient tracking during inference
            outputs = MODEL(image_batch)
            # Convert raw scores → probabilities (sum to 1.0)
            probabilities = F.softmax(outputs, dim=1)

        # ---- TOP 3 PREDICTIONS ----
        # topk returns (values, indices) of highest 3 probabilities
        top_probs, top_indices = torch.topk(
            probabilities, k=3, dim=1
        )
        top_probs   = top_probs[0].cpu().numpy().tolist()
        top_indices = top_indices[0].cpu().numpy().tolist()

        # Build readable prediction list
        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = IDX_TO_CLASS[idx]
            top_predictions.append({
                'rank': len(top_predictions) + 1,
                'class_name': class_name,
                'display_name': (class_name
                    .replace('___', ' ')
                    .replace('__', ' ')
                    .replace('_', ' ')),
                'confidence': round(prob * 100, 2),
                'is_healthy': 'healthy' in class_name.lower()
            })

        # ---- PRIMARY PREDICTION ----
        primary       = top_predictions[0]
        primary_class = IDX_TO_CLASS[top_indices[0]]

        # ---- REMEDY LOOKUP ----
        # Get treatment info for the detected disease
        remedy = get_quick_summary(primary_class)

        # ---- GRAD-CAM ----
        # Only generate if confidence > 30% (low conf = noisy heatmap)
        gradcam_image = None
        if include_gradcam and primary['confidence'] > 30:
            gradcam_image = generate_gradcam_base64(
                image_tensor, original_image
            )

        # ---- BUILD RESPONSE ----
        inference_time = round((time.time() - start_time) * 1000)

        response = {
            "success": True,
            "inference_time_ms": inference_time,

            "prediction": {
                "class_name": primary_class,
                "display_name": primary['display_name'],
                "confidence": primary['confidence'],
                "is_healthy": primary['is_healthy'],
                "is_confident": primary['confidence'] > 60,
                "severity": remedy.get('severity', 'unknown')
            },

            "top_predictions": top_predictions,
            "remedy": remedy,
            "gradcam_image": gradcam_image,
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """
    Returns model architecture info and evaluation statistics.
    Used by the Model Statistics page in the frontend.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )

    info = MODEL.get_model_info()

    # Load evaluation report if it exists
    # Generated by: python ml_training/evaluate.py
    eval_report = {}
    eval_path = BASE_DIR / 'data' / 'evaluation_report.json'
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)

    return {
        "model_name": "EfficientNet-B4",
        "num_classes": MODEL.num_classes,
        "total_parameters": info['total_parameters'],
        "model_size_mb": round(info['model_size_mb'], 1),
        "device": str(DEVICE),
        "evaluation": eval_report
    }