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
# Run with: uvicorn backend.app.main:app --reload
# =============================================================

import io
import json
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
from pydantic import BaseModel

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

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
# GLOBAL MODEL (loaded once at startup)
# =============================================================
# We load the model once when the server starts
# NOT on every request — that would be very slow

MODEL = None
IDX_TO_CLASS = None
DEVICE = None
TRANSFORM = None

def load_model():
    """Load model into memory at server startup."""
    global MODEL, IDX_TO_CLASS, DEVICE, TRANSFORM

    print("Loading plant disease model...")
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load trained model
    MODEL, checkpoint = load_checkpoint(
        'backend/models/best_model.pt',
        device=DEVICE
    )
    MODEL.eval()

    # Load class mapping
    with open('data/class_mapping.json', 'r') as f:
        mapping = json.load(f)
    IDX_TO_CLASS = {
        int(k): v for k, v in mapping['idx_to_class'].items()
    }

    # Preprocessing transform
    TRANSFORM = get_val_transforms(224)

    print(f"✓ Model loaded on {DEVICE}")
    print(f"✓ Val accuracy: {checkpoint['val_accuracy']:.2f}%")
    print(f"✓ Classes: {len(IDX_TO_CLASS)}")


# Load model when app starts
@app.on_event("startup")
async def startup_event():
    load_model()


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def preprocess_image(image_bytes: bytes) -> tuple:
    """
    Convert uploaded image bytes to model-ready tensor.

    Args:
        image_bytes: raw bytes from uploaded file

    Returns:
        Tuple of (tensor, original_numpy_image)
    """
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Convert to numpy for albumentations
    img_array = np.array(image)

    # Keep original for display
    original = cv2.resize(img_array, (224, 224))

    # Apply preprocessing transforms
    transformed = TRANSFORM(image=img_array)
    tensor = transformed['image']

    return tensor, original


def tensor_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy image array to base64 string.
    Base64 lets us send images as text in JSON response.

    Args:
        image_array: RGB numpy array

    Returns:
        Base64 encoded string
    """
    # Convert RGB to BGR for cv2
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Encode as JPEG bytes
    _, buffer = cv2.imencode('.jpg', bgr, 
                              [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Convert bytes to base64 string
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def generate_gradcam_base64(image_tensor: torch.Tensor,
                             original_image: np.ndarray) -> str:
    """
    Generate Grad-CAM heatmap and return as base64.

    Args:
        image_tensor: preprocessed image tensor
        original_image: original numpy image for overlay

    Returns:
        Base64 encoded heatmap overlay image
    """
    try:
        gradcam = GradCAM(MODEL, DEVICE)
        cam, _, _ = gradcam.generate(image_tensor)
        gradcam.remove_hooks()

        # Apply heatmap overlay
        overlay = apply_heatmap(original_image, cam, alpha=0.4)

        return tensor_to_base64(overlay)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


# =============================================================
# API ENDPOINTS
# =============================================================

@app.get("/")
async def root():
    """Health check endpoint."""
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
    Used by frontend to show disease encyclopedia.
    """
    diseases = get_all_diseases()
    return {
        "total": len(diseases),
        "diseases": diseases
    }


@app.get("/remedies/{class_name}")
async def get_remedy(class_name: str):
    """
    Returns full remedy information for a disease.

    Args:
        class_name: disease class name
                    e.g. Tomato_Early_blight

    Returns:
        Full remedy with treatments, prevention, timeline
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
    Main prediction endpoint.

    Accepts an uploaded image file and returns:
    - Disease prediction with confidence
    - Top 3 predictions
    - Grad-CAM heatmap (optional)
    - Remedy information

    Args:
        file: uploaded image file (jpg, png)
        include_gradcam: whether to generate heatmap

    Returns:
        Full prediction result with remedy
    """
    start_time = time.time()

    # ---- VALIDATE FILE ----
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, etc.)"
        )

    # Read file bytes
    image_bytes = await file.read()

    # Check file size (max 10MB)
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image too large. Maximum size is 10MB."
        )

    try:
        # ---- PREPROCESS IMAGE ----
        image_tensor, original_image = preprocess_image(
            image_bytes
        )

        # ---- RUN INFERENCE ----
        image_batch = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = MODEL(image_batch)
            probabilities = F.softmax(outputs, dim=1)

        # Get top 3 predictions
        top_probs, top_indices = torch.topk(
            probabilities, k=3, dim=1
        )
        top_probs   = top_probs[0].cpu().numpy().tolist()
        top_indices = top_indices[0].cpu().numpy().tolist()

        # Build top predictions list
        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = IDX_TO_CLASS[idx]
            top_predictions.append({
                'rank': len(top_predictions) + 1,
                'class_name': class_name,
                'display_name': class_name
                    .replace('___', ' ')
                    .replace('__', ' ')
                    .replace('_', ' '),
                'confidence': round(prob * 100, 2),
                'is_healthy': 'healthy' in class_name.lower()
            })

        # Primary prediction
        primary = top_predictions[0]
        primary_class = IDX_TO_CLASS[top_indices[0]]

        # ---- GET REMEDY ----
        remedy = get_quick_summary(primary_class)

        # ---- GENERATE GRAD-CAM ----
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

            # Primary prediction
            "prediction": {
                "class_name": primary_class,
                "display_name": primary['display_name'],
                "confidence": primary['confidence'],
                "is_healthy": primary['is_healthy'],
                "is_confident": primary['confidence'] > 60,
                "severity": remedy.get('severity', 'unknown')
            },

            # Top 3 alternatives
            "top_predictions": top_predictions,

            # Remedy summary
            "remedy": remedy,

            # Grad-CAM heatmap as base64
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
    """Returns model information and statistics."""
    if MODEL is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded")

    info = MODEL.get_model_info()

    # Load evaluation report if it exists
    eval_report = {}
    eval_path = Path('data/evaluation_report.json')
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