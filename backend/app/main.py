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
import time
import urllib.request
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
# parent = backend/app
# parent.parent = backend
# parent.parent.parent = project root ← BASE_DIR
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

# CORS — allows Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# GLOBAL VARIABLES
# Loaded once at startup, reused for every request
# =============================================================

MODEL        = None
IDX_TO_CLASS = None
DEVICE       = None
TRANSFORM    = None

# ---- HuggingFace URLs ----
# Replace YOUR_HF_USERNAME with your actual HuggingFace username
# e.g. if your profile is huggingface.co/shivraahi → use "shivraahi"
HF_USERNAME  = "shivcodes91"
HF_REPO      = "plant-disease-efficientnet"
HF_BASE_URL  = f"https://huggingface.co/{HF_USERNAME}/{HF_REPO}/resolve/main"


def download_file(url: str, dest_path: Path, label: str):
    """
    Downloads a file from URL to dest_path with progress logging.

    Args:
        url: download URL
        dest_path: where to save the file
        label: display name for logging
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label}...")
    print(f"  From : {url}")
    print(f"  To   : {dest_path}")

    def progress(count, block_size, total_size):
        if total_size > 0:
            percent = count * block_size * 100 / total_size
            # Print progress every 10%
            if int(percent) % 10 == 0:
                print(f"  Progress: {min(percent, 100):.0f}%")

    urllib.request.urlretrieve(url, str(dest_path), progress)
    print(f"  ✓ {label} downloaded!")


def load_model():
    """
    Loads the plant disease model.

    Strategy:
    1. Check if model exists locally (always true when running locally)
    2. If not found (on Render) → download from HuggingFace
    3. Load model weights and class mapping
    4. Set up preprocessing transform
    """
    global MODEL, IDX_TO_CLASS, DEVICE, TRANSFORM

    print("=" * 55)
    print("  LOADING PLANT DISEASE MODEL")
    print("=" * 55)

    # ---- DEVICE ----
    # cuda = GPU (local with RTX 3050)
    # cpu  = Render free tier (no GPU)
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"\n  Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ---- PATHS ----
    model_path   = BASE_DIR / 'backend' / 'models' / 'best_model.pt'
    mapping_path = BASE_DIR / 'data' / 'class_mapping.json'

    print(f"  Base   : {BASE_DIR}")
    print(f"  Model  : {model_path}")
    print(f"  Mapping: {mapping_path}")

    # ---- DOWNLOAD MODEL IF MISSING ----
    # On Render: model not in repo (190MB > GitHub 100MB limit)
    # Solution: download from HuggingFace on first startup
    # On local machine: file exists, skip download
    if not model_path.exists():
        print(f"\n  Model not found locally.")
        print(f"  Downloading from HuggingFace Hub...")
        download_file(
            url=f"{HF_BASE_URL}/best_model.pt",
            dest_path=model_path,
            label="best_model.pt (~190MB)"
        )

    # ---- DOWNLOAD CLASS MAPPING IF MISSING ----
    if not mapping_path.exists():
        print(f"\n  Class mapping not found locally.")
        download_file(
            url=f"{HF_BASE_URL}/class_mapping.json",
            dest_path=mapping_path,
            label="class_mapping.json"
        )

    # ---- LOAD MODEL ----
    print(f"\n  Loading EfficientNet-B4 checkpoint...")
    MODEL, checkpoint = load_checkpoint(
        str(model_path),
        device=DEVICE
    )
    MODEL.eval()

    # ---- LOAD CLASS MAPPING ----
    # Maps index → class name
    # e.g. {0: "Pepper__bell___Bacterial_spot", ...}
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    IDX_TO_CLASS = {
        int(k): v
        for k, v in mapping['idx_to_class'].items()
    }

    # ---- PREPROCESSING PIPELINE ----
    # Same transforms used during training
    TRANSFORM = get_val_transforms(224)

    print(f"\n{'='*55}")
    print(f"  MODEL READY!")
    print(f"{'='*55}")
    print(f"  Val accuracy : {checkpoint['val_accuracy']:.2f}%")
    print(f"  Epoch        : {checkpoint['epoch']}")
    print(f"  Classes      : {len(IDX_TO_CLASS)}")
    print(f"  Device       : {DEVICE}")


# ---- STARTUP EVENT ----
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

    Args:
        image_bytes: raw bytes from uploaded file

    Returns:
        Tuple of (image_tensor, original_numpy_array)
    """
    image    = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(image)
    original  = cv2.resize(img_array, (224, 224))
    tensor    = TRANSFORM(image=img_array)['image']
    return tensor, original


def tensor_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy RGB image → base64 data URL string.
    Lets us send images inside JSON responses.

    Args:
        image_array: RGB numpy array [H, W, 3]

    Returns:
        "data:image/jpeg;base64,..."
    """
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(
        '.jpg', bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def generate_gradcam_base64(
    image_tensor: torch.Tensor,
    original_image: np.ndarray
) -> Optional[str]:
    """
    Generate Grad-CAM heatmap overlay as base64.
    Shows WHERE on the leaf the model detected disease.
    Red = high attention, Blue = low attention.

    Args:
        image_tensor: preprocessed tensor [3, 224, 224]
        original_image: original numpy image for overlay

    Returns:
        Base64 heatmap overlay string, or None if error
    """
    try:
        gradcam = GradCAM(MODEL, DEVICE)
        cam, _, _ = gradcam.generate(image_tensor)
        gradcam.remove_hooks()
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
    """Health check — verify API is running."""
    return {
        "status": "online",
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }


@app.get("/diseases")
async def get_diseases():
    """Returns list of all 15 detectable diseases."""
    diseases = get_all_diseases()
    return {
        "total": len(diseases),
        "diseases": diseases
    }


@app.get("/remedies/{class_name}")
async def get_remedy(class_name: str):
    """
    Returns full remedy for a disease class.

    Args:
        class_name: e.g. Tomato_Early_blight
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
    MAIN ENDPOINT: Upload leaf image → get disease + remedy.

    Flow:
    1. Validate file (type + size)
    2. Preprocess image
    3. Run model inference → probabilities
    4. Get top 3 predictions
    5. Look up remedy for top prediction
    6. Generate Grad-CAM heatmap
    7. Return full JSON response

    Args:
        file: uploaded image (jpg/png/webp, max 10MB)
        include_gradcam: set False to skip heatmap (faster)
    """
    start_time = time.time()

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, webp, etc.)"
        )

    image_bytes = await file.read()

    # Validate file size (max 10MB)
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image too large. Maximum size is 10MB."
        )

    try:
        # Preprocess
        image_tensor, original_image = preprocess_image(image_bytes)

        # Inference
        image_batch   = image_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs       = MODEL(image_batch)
            probabilities = F.softmax(outputs, dim=1)

        # Top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)
        top_probs   = top_probs[0].cpu().numpy().tolist()
        top_indices = top_indices[0].cpu().numpy().tolist()

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

        primary       = top_predictions[0]
        primary_class = IDX_TO_CLASS[top_indices[0]]

        # Remedy lookup
        remedy = get_quick_summary(primary_class)

        # Grad-CAM (skip if low confidence — noisy heatmap)
        gradcam_image = None
        if include_gradcam and primary['confidence'] > 30:
            gradcam_image = generate_gradcam_base64(
                image_tensor, original_image
            )

        inference_time = round((time.time() - start_time) * 1000)

        return JSONResponse(content={
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
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Returns model architecture info and evaluation stats."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info      = MODEL.get_model_info()
    eval_path = BASE_DIR / 'data' / 'evaluation_report.json'
    eval_report = {}
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