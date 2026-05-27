# =============================================================
# streamlit_app.py
#
# Plant Disease Detection — Streamlit Frontend
#
# Run with: streamlit run streamlit_app.py
#
# PAGES:
# 🔍 Detect Disease  → upload image → get prediction
# 📚 Disease Library → browse all 15 diseases
# 📊 Model Stats     → accuracy, confusion matrix
# ℹ️  About          → project info
# =============================================================

import streamlit as st
import requests
import json
import base64
import time
from pathlib import Path
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# =============================================================
# CONFIGURATION
# =============================================================

# API URL — change this when you deploy to production
API_URL = "http://localhost:8000"

# Page config — MUST be first Streamlit command
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# CUSTOM CSS — makes the app look professional
# =============================================================

st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #f8fdf8;
    }

    /* Hide default Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom card style */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e0f0e0;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,100,0,0.08);
    }

    /* Severity badges */
    .severity-severe {
        background: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    .severity-moderate {
        background: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    .severity-none {
        background: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }

    /* Confidence bar */
    .confidence-bar {
        background: #e8f5e9;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }

    /* Header style */
    .page-header {
        font-size: 28px;
        font-weight: 700;
        color: #1a5c2a;
        margin-bottom: 4px;
    }

    /* Treatment box */
    .treatment-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }

    /* Warning box */
    .warning-box {
        background: #fff7ed;
        border-left: 4px solid #ea580c;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }

    /* Metric card */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e0f0e0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        return response.status_code == 200
    except:
        return False


def predict_disease(image_bytes: bytes, filename: str) -> dict:
    """
    Sends image to API and returns prediction results.

    Args:
        image_bytes: raw image file bytes
        filename: original filename

    Returns:
        API response dictionary
    """
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            params={"include_gradcam": True},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the backend is running!"}
    except Exception as e:
        return {"error": str(e)}


def get_all_diseases() -> list:
    """Fetch all diseases from API."""
    try:
        response = requests.get(f"{API_URL}/diseases", timeout=5)
        if response.status_code == 200:
            return response.json().get("diseases", [])
    except:
        pass
    return []


def get_full_remedy(class_name: str) -> dict:
    """Fetch full remedy from API."""
    try:
        response = requests.get(
            f"{API_URL}/remedies/{class_name}",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def get_model_info() -> dict:
    """Fetch model information from API."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def base64_to_image(b64_string: str) -> Image:
    """Convert base64 string back to PIL Image."""
    # Remove the data URL prefix
    if "base64," in b64_string:
        b64_string = b64_string.split("base64,")[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes))


def severity_badge(severity: str) -> str:
    """Returns HTML badge for severity level."""
    icons = {"severe": "🔴", "moderate": "🟡", "none": "🟢"}
    icon = icons.get(severity, "⚪")
    return f"{icon} {severity.upper()}"


# =============================================================
# SIDEBAR NAVIGATION
# =============================================================

with st.sidebar:
    st.markdown("## 🌿 Plant Disease Detector")
    st.markdown("*AI-powered diagnosis for your crops*")
    st.divider()

    page = st.selectbox(
        "Navigate to",
        ["🔍 Detect Disease",
         "📚 Disease Library",
         "📊 Model Statistics",
         "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    # API status indicator
    api_online = check_api_health()
    if api_online:
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline")
        st.caption("Run: `python run_api.py`")

    st.divider()
    st.caption("Model: EfficientNet-B4")
    st.caption("Accuracy: 98.9%")
    st.caption("Classes: 15 diseases")
    st.caption("Plants: Tomato, Potato, Pepper")


# =============================================================
# PAGE 1: DETECT DISEASE
# =============================================================

if page == "🔍 Detect Disease":

    st.markdown(
        '<p class="page-header">🔍 Plant Disease Detection</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        "Upload a clear photo of a plant leaf to diagnose diseases "
        "and get treatment recommendations."
    )

    # Upload section
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📸 Upload Leaf Image")

        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear, well-lit photo of the leaf"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image",
                     use_container_width=True)

            # Tips
            with st.expander("📌 Tips for best results"):
                st.markdown("""
                - 📷 Take photo in good lighting
                - 🍃 Focus on the affected leaf area
                - 📐 Keep leaf flat and fully visible
                - 🔍 Capture at least 1-2 diseased spots
                - 📱 Works with phone camera photos
                """)

    with col2:
        if uploaded_file is not None:
            st.markdown("#### 🧠 Analysis Results")

            if not api_online:
                st.error(
                    "❌ Backend API is not running!\n\n"
                    "Open a terminal and run:\n"
                    "```\npython run_api.py\n```"
                )
            else:
                with st.spinner("🔬 Analyzing leaf... This may take a few seconds"):
                    image_bytes = uploaded_file.getvalue()
                    result = predict_disease(
                        image_bytes, uploaded_file.name
                    )

                if "error" in result:
                    st.error(f"❌ {result['error']}")

                elif result.get("success"):
                    pred = result["prediction"]
                    remedy = result["remedy"]

                    # ---- MAIN RESULT CARD ----
                    is_healthy = pred["is_healthy"]

                    if is_healthy:
                        st.success("✅ **Your plant appears HEALTHY!**")
                    else:
                        severity = pred.get("severity", "moderate")
                        if severity == "severe":
                            st.error(
                                f"🚨 **Disease Detected — SEVERE**"
                            )
                        else:
                            st.warning(
                                f"⚠️ **Disease Detected — {severity.upper()}**"
                            )

                    # Disease name
                    st.markdown(
                        f"### {remedy.get('display_name', pred['display_name'])}"
                    )

                    # Confidence meter
                    confidence = pred["confidence"]
                    conf_color = (
                        "green" if confidence > 80
                        else "orange" if confidence > 60
                        else "red"
                    )
                    st.markdown(f"**Confidence: {confidence:.1f}%**")
                    st.progress(confidence / 100)

                    if not pred.get("is_confident"):
                        st.warning(
                            "⚠️ Low confidence prediction. "
                            "Try a clearer image."
                        )

                    # Key info row
                    info_col1, info_col2, info_col3 = st.columns(3)
                    with info_col1:
                        st.metric("Plant", remedy.get("plant", "Unknown"))
                    with info_col2:
                        st.metric("Type", remedy.get("pathogen_type") or "N/A")
                    with info_col3:
                        sev = pred.get("severity", "unknown")
                        st.metric("Severity", sev.upper())

                    st.divider()

                    # ---- TOP 3 PREDICTIONS ----
                    with st.expander("📊 Top 3 Predictions"):
                        for p in result.get("top_predictions", []):
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(
                                    f"**{p['rank']}. {p['display_name']}**"
                                )
                                st.progress(p["confidence"] / 100)
                            with col_b:
                                st.markdown(
                                    f"**{p['confidence']:.1f}%**"
                                )

                    # ---- OVERVIEW ----
                    if remedy.get("overview"):
                        st.markdown("#### 📋 Overview")
                        st.info(remedy["overview"])

                    # ---- IMMEDIATE ACTION ----
                    if remedy.get("immediate_action") and not is_healthy:
                        st.markdown("#### ⚡ Immediate Action Required")
                        st.markdown(
                            f'<div class="warning-box">'
                            f'🚨 {remedy["immediate_action"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    # ---- TREATMENT ----
                    if remedy.get("first_treatment") and not is_healthy:
                        treatment = remedy["first_treatment"]
                        st.markdown("#### 💊 Recommended Treatment")
                        st.markdown(
                            f'<div class="treatment-box">'
                            f'<strong>{treatment["name"]}</strong><br>'
                            f'📏 Dosage: {treatment["dosage"]}<br>'
                            f'🔄 Frequency: {treatment["frequency"]}<br>'
                            f'💡 Note: {treatment["notes"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        if remedy.get("has_organic_options"):
                            st.success(
                                "🌱 Organic treatment options available! "
                                "Check Disease Library for details."
                            )

                    # ---- GRAD-CAM ----
                    if result.get("gradcam_image"):
                        st.divider()
                        st.markdown(
                            "#### 🔥 AI Attention Map (Grad-CAM)"
                        )
                        st.caption(
                            "Red/yellow areas show where the AI "
                            "detected disease symptoms"
                        )
                        gradcam_img = base64_to_image(
                            result["gradcam_image"]
                        )
                        st.image(
                            gradcam_img,
                            caption="Disease region highlighted by AI",
                            use_container_width=True
                        )

                    # ---- INFERENCE TIME ----
                    st.caption(
                        f"⏱️ Analysis completed in "
                        f"{result['inference_time_ms']}ms"
                    )

        else:
            # Empty state
            st.markdown("""
            <div style='text-align:center; padding: 60px 20px;
                        color: #666; border: 2px dashed #ccc;
                        border-radius: 12px; margin-top: 20px'>
                <h2>📤</h2>
                <h3>Upload a leaf image to get started</h3>
                <p>Supports: JPG, PNG, WEBP</p>
                <p>Works with tomato, potato, and pepper leaves</p>
            </div>
            """, unsafe_allow_html=True)


# =============================================================
# PAGE 2: DISEASE LIBRARY
# =============================================================

elif page == "📚 Disease Library":

    st.markdown(
        '<p class="page-header">📚 Disease Library</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        "Browse all 15 detectable plant diseases with full "
        "treatment information."
    )

    diseases = get_all_diseases()

    if not diseases:
        st.error("Cannot load diseases. Is the API running?")
    else:
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            plant_filter = st.selectbox(
                "Filter by plant",
                ["All Plants", "Tomato", "Potato", "Pepper"]
            )
        with col2:
            severity_filter = st.selectbox(
                "Filter by severity",
                ["All", "severe", "moderate", "none"]
            )
        with col3:
            health_filter = st.selectbox(
                "Show",
                ["All", "Diseased only", "Healthy only"]
            )

        # Apply filters
        filtered = diseases.copy()
        if plant_filter != "All Plants":
            filtered = [
                d for d in filtered
                if d["plant"].lower().startswith(
                    plant_filter.lower()
                )
            ]
        if severity_filter != "All":
            filtered = [
                d for d in filtered
                if d["severity"] == severity_filter
            ]
        if health_filter == "Diseased only":
            filtered = [d for d in filtered if not d["is_healthy"]]
        elif health_filter == "Healthy only":
            filtered = [d for d in filtered if d["is_healthy"]]

        st.markdown(
            f"Showing **{len(filtered)}** diseases"
        )
        st.divider()

        # Display each disease
        for disease in filtered:
            severity = disease["severity"]
            is_healthy = disease["is_healthy"]

            # Severity icon
            icon = (
                "🟢" if is_healthy
                else "🔴" if severity == "severe"
                else "🟡"
            )

            with st.expander(
                f"{icon} {disease['display_name']} — "
                f"{disease['plant']} | "
                f"{severity.upper()}"
            ):
                # Load full remedy
                with st.spinner("Loading remedy..."):
                    full_remedy = get_full_remedy(
                        disease["class_name"]
                    )

                if not full_remedy:
                    st.warning("Could not load remedy details")
                    continue

                col_a, col_b = st.columns([1, 1])

                with col_a:
                    # Overview
                    st.markdown("**Overview**")
                    st.markdown(
                        full_remedy.get("overview", "N/A")
                    )

                    # Pathogen info
                    if full_remedy.get("pathogen_name"):
                        st.markdown(
                            f"🦠 **Pathogen:** "
                            f"*{full_remedy['pathogen_name']}*"
                        )
                        st.markdown(
                            f"🔬 **Type:** "
                            f"{full_remedy.get('pathogen_type', 'N/A')}"
                        )

                    # Symptoms
                    symptoms = full_remedy.get("symptoms", [])
                    if symptoms:
                        st.markdown("**Symptoms**")
                        for s in symptoms:
                            st.markdown(f"• {s}")

                with col_b:
                    # Treatments
                    treatments = full_remedy.get(
                        "treatments", {}
                    )

                    # Chemical treatments
                    chem = treatments.get("chemical_treatments", [])
                    if chem:
                        st.markdown("**💊 Chemical Treatments**")
                        for t in chem:
                            st.markdown(
                                f"**{t['name']}** "
                                f"({', '.join(t.get('brand_examples', []))})"
                            )
                            st.caption(
                                f"Dosage: {t['dosage']} | "
                                f"Every: {t['frequency']}"
                            )

                    # Organic treatments
                    organic = treatments.get(
                        "organic_treatments", []
                    )
                    if organic:
                        st.markdown("**🌱 Organic Options**")
                        for t in organic:
                            st.markdown(f"• **{t['name']}**")
                            st.caption(
                                f"{t['dosage']} | {t['frequency']}"
                            )

                    # Prevention
                    prevention = full_remedy.get("prevention", [])
                    if prevention:
                        st.markdown("**🛡️ Prevention**")
                        for p in prevention[:4]:
                            st.markdown(f"• {p}")

                # Treatment timeline
                timeline = full_remedy.get("treatment_timeline", {})
                if timeline and not is_healthy:
                    st.markdown("**📅 Treatment Timeline**")
                    timeline_cols = st.columns(
                        min(len(timeline), 4)
                    )
                    for i, (period, action) in enumerate(
                        list(timeline.items())[:4]
                    ):
                        with timeline_cols[i % 4]:
                            st.markdown(
                                f"**{period.replace('_', ' ').title()}**"
                            )
                            st.caption(action)


# =============================================================
# PAGE 3: MODEL STATISTICS
# =============================================================

elif page == "📊 Model Statistics":

    st.markdown(
        '<p class="page-header">📊 Model Statistics</p>',
        unsafe_allow_html=True
    )

    model_info = get_model_info()
    eval_data = model_info.get("evaluation", {})

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Test Accuracy",
            f"{eval_data.get('overall_accuracy', 98.39)}%",
            delta="↑ vs random baseline"
        )
    with col2:
        st.metric(
            "Avg Confidence",
            f"{eval_data.get('avg_confidence', 89.03)}%"
        )
    with col3:
        st.metric(
            "Model Parameters",
            f"{model_info.get('total_parameters', 18603351):,}"
        )
    with col4:
        st.metric(
            "Model Size",
            f"{model_info.get('model_size_mb', 71.0)} MB"
        )

    st.divider()

    # Per class accuracy chart
    per_class = eval_data.get("per_class_results", [])

    if per_class:
        st.markdown("#### Per-Class Accuracy")

        df = pd.DataFrame(per_class)
        df["short_name"] = df["class_name"].apply(
            lambda x: x.replace("Pepper__bell___", "Pepper ")
                       .replace("Potato___", "Potato ")
                       .replace("Tomato__", "Tomato ")
                       .replace("Tomato_", "Tomato ")
                       .replace("_", " ")
        )
        df = df.sort_values("accuracy")

        # Color bars by accuracy
        colors = [
            "#16a34a" if a >= 98
            else "#ca8a04" if a >= 95
            else "#dc2626"
            for a in df["accuracy"]
        ]

        fig = go.Figure(go.Bar(
            x=df["accuracy"],
            y=df["short_name"],
            orientation='h',
            marker_color=colors,
            text=[f"{a:.1f}%" for a in df["accuracy"]],
            textposition='outside'
        ))

        fig.update_layout(
            xaxis_title="Accuracy (%)",
            xaxis=dict(range=[85, 101]),
            height=500,
            margin=dict(l=0, r=60, t=20, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Confusion matrix
    st.markdown("#### Confusion Matrix")
    confusion_path = Path("data/confusion_matrix.png")
    if confusion_path.exists():
        st.image(
            str(confusion_path),
            caption="Confusion Matrix — Test Set (98.39% accuracy)",
            use_container_width=True
        )
    else:
        st.info(
            "Run `python ml_training/evaluate.py` "
            "to generate confusion matrix"
        )

    # Training curves
    st.markdown("#### Training Curves")
    curves_path = Path("data/training_curves.png")
    if curves_path.exists():
        st.image(
            str(curves_path),
            caption="Training History — 25 Epochs",
            use_container_width=True
        )

    # Model architecture
    with st.expander("🏗️ Model Architecture Details"):
        st.markdown("""
        | Component | Details |
        |-----------|---------|
        | Base Model | EfficientNet-B4 |
        | Pretrained | ImageNet (1.2M images) |
        | Input Size | 224 × 224 pixels |
        | Classifier | 1792 → 512 → 256 → 15 |
        | Loss Function | Label Smoothing CE (0.1) |
        | Optimizer | AdamW |
        | Scheduler | CosineAnnealingLR |
        | Training Strategy | Freeze → Unfreeze (fine-tuning) |
        | Mixed Precision | AMP (float16) |
        | Augmentations | 10 (flip, rotate, color, noise) |
        """)


# =============================================================
# PAGE 4: ABOUT
# =============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<p class="page-header">ℹ️ About This Project</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🌿 Plant Disease Detection System

        An AI-powered system that helps farmers identify plant
        diseases from leaf photos and get instant treatment
        recommendations.

        ### 🎯 What it does
        - **Detects 15 plant diseases** across Tomato, Potato,
          and Pepper crops
        - **98.9% accuracy** on validation data
        - **Grad-CAM visualization** shows exactly where on
          the leaf the disease was detected
        - **Full remedy information** including chemical and
          organic treatment options, dosages, and prevention tips

        ### 🛠️ Technology Stack
        | Layer | Technology |
        |-------|-----------|
        | AI Model | EfficientNet-B4 (PyTorch) |
        | Explainability | Grad-CAM |
        | Backend API | FastAPI |
        | Frontend | Streamlit |
        | Dataset | PlantVillage (41,274 images) |

        ### 📊 Model Performance
        - **Validation Accuracy:** 98.87%
        - **Test Accuracy:** 98.39%
        - **Average Confidence:** 89.03%
        - **Training:** 25 epochs on RTX 3050 GPU
        """)

    with col2:
        st.markdown("### 🌱 Supported Crops")

        crops = {
            "🍅 Tomato": [
                "Bacterial Spot",
                "Early Blight",
                "Late Blight",
                "Leaf Mold",
                "Septoria Leaf Spot",
                "Spider Mites",
                "Target Spot",
                "Yellow Leaf Curl Virus",
                "Mosaic Virus",
                "Healthy"
            ],
            "🥔 Potato": [
                "Early Blight",
                "Late Blight",
                "Healthy"
            ],
            "🫑 Pepper": [
                "Bacterial Spot",
                "Healthy"
            ]
        }

        for crop, diseases in crops.items():
            with st.expander(f"{crop} — {len(diseases)} classes"):
                for d in diseases:
                    icon = "🟢" if d == "Healthy" else "🔴"
                    st.markdown(f"{icon} {d}")

        st.divider()
        st.markdown("### 👨‍💻 Developer")
        st.markdown("**Shivang Rai**")
        st.markdown("Built with ❤️ using PyTorch, FastAPI & Streamlit")

        st.divider()
        st.markdown("### 🔗 Links")
        st.markdown("[GitHub Repository](https://github.com/Shivcodes91/plant-disease-detection)")