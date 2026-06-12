# =============================================================
# app.py — LeafScan AI (Standalone Streamlit App)
#
# This single file does EVERYTHING:
# - Downloads model from HuggingFace automatically
# - Runs inference directly (no separate API needed)
# - Full UI with all 4 pages
#
# Deploy on Streamlit Cloud — free, 1GB RAM
# Run locally: streamlit run app.py
# =============================================================

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import json
import base64
import io
import urllib.request
from pathlib import Path
from PIL import Image
import cv2
import pandas as pd
import plotly.graph_objects as go

import sys
BASE_DIR = Path(__file__).parent.resolve()
sys.path.append(str(BASE_DIR))

from ml_training.model import load_checkpoint
from ml_training.dataset import get_val_transforms
from ml_training.gradcam import GradCAM, apply_heatmap
from backend.app.db.remedies_db import (
    get_quick_summary,
    get_all_diseases,
    get_remedy as get_full_remedy_db
)

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="LeafScan AI — Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# CSS
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0d1117;}
.block-container{padding:2rem 2.5rem 4rem;max-width:1400px;}
#MainMenu,footer,header{visibility:hidden;}
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f1923 0%,#0a1628 100%);
    border-right:1px solid #1e3a2f;
}
section[data-testid="stSidebar"] *{color:#c8e6c9 !important;}
.hero-title{
    font-family:'Sora',sans-serif;font-size:2.4rem;font-weight:700;
    background:linear-gradient(135deg,#4ade80 0%,#22d3ee 50%,#a78bfa 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;line-height:1.2;margin-bottom:0.3rem;
}
.hero-sub{font-size:1rem;color:#6b8e7a;margin-bottom:1.5rem;}
.section-title{
    font-family:'Sora',sans-serif;font-size:1rem;font-weight:600;
    color:#4ade80;letter-spacing:0.03em;text-transform:uppercase;
    margin:1.5rem 0 0.8rem;display:flex;align-items:center;gap:8px;
}
.section-title::after{content:'';flex:1;height:1px;
    background:linear-gradient(90deg,#1e3a2f,transparent);}
.result-healthy{background:linear-gradient(135deg,#052e16,#0a1628);
    border:1px solid #16a34a;border-radius:16px;padding:1.5rem;
    position:relative;overflow:hidden;}
.result-healthy::before{content:'';position:absolute;top:0;left:0;right:0;
    height:3px;background:linear-gradient(90deg,#16a34a,#4ade80);}
.result-diseased{background:linear-gradient(135deg,#1c0a02,#0a1628);
    border:1px solid #ea580c;border-radius:16px;padding:1.5rem;
    position:relative;overflow:hidden;}
.result-diseased::before{content:'';position:absolute;top:0;left:0;right:0;
    height:3px;background:linear-gradient(90deg,#ea580c,#fbbf24);}
.result-severe{background:linear-gradient(135deg,#1c0202,#0a1628);
    border:1px solid #dc2626;border-radius:16px;padding:1.5rem;
    position:relative;overflow:hidden;}
.result-severe::before{content:'';position:absolute;top:0;left:0;right:0;
    height:3px;background:linear-gradient(90deg,#dc2626,#f97316);}
.disease-name{font-family:'Sora',sans-serif;font-size:1.8rem;font-weight:700;
    color:#f0fdf4;margin:0.5rem 0;line-height:1.2;}
.disease-name-healthy{color:#4ade80;}
.disease-name-severe{color:#f87171;}
.disease-name-moderate{color:#fbbf24;}
.pill{display:inline-block;padding:4px 16px;border-radius:999px;
    font-size:0.78rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;}
.pill-healthy{background:#052e16;color:#4ade80;border:1px solid #16a34a;}
.pill-moderate{background:#1c1002;color:#fbbf24;border:1px solid #d97706;}
.pill-severe{background:#1c0202;color:#f87171;border:1px solid #dc2626;}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:1rem 0;}
.metric-block{background:#0f1923;border:1px solid #1e3a2f;border-radius:12px;
    padding:14px 16px;text-align:center;}
.metric-label{font-size:0.72rem;color:#4b7063;text-transform:uppercase;
    letter-spacing:0.1em;font-weight:500;margin-bottom:4px;}
.metric-value{font-family:'Sora',sans-serif;font-size:1.15rem;font-weight:600;color:#e2e8f0;}
.action-box{background:linear-gradient(135deg,#1c1002,#0f1923);
    border-left:4px solid #f59e0b;border-radius:0 12px 12px 0;
    padding:14px 18px;margin:10px 0;font-size:0.95rem;color:#fde68a;font-weight:500;}
.treatment-box{background:linear-gradient(135deg,#052e16,#0f1923);
    border-left:4px solid #4ade80;border-radius:0 12px 12px 0;padding:14px 18px;margin:10px 0;}
.treatment-name{font-family:'Sora',sans-serif;font-size:1rem;font-weight:600;
    color:#4ade80;margin-bottom:8px;}
.treatment-row{display:flex;gap:8px;margin:4px 0;font-size:0.88rem;color:#86efac;}
.stat-card{background:linear-gradient(135deg,#0f1923,#0a1628);border:1px solid #1e3a2f;
    border-radius:14px;padding:1.4rem 1.2rem;text-align:center;position:relative;overflow:hidden;}
.stat-card::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,#4ade80,transparent);}
.stat-number{font-family:'Sora',sans-serif;font-size:2.4rem;font-weight:700;
    color:#4ade80;line-height:1;margin-bottom:6px;}
.stat-label{font-size:0.8rem;color:#4b7063;text-transform:uppercase;
    letter-spacing:0.1em;font-weight:500;}
div[data-testid="stFileUploader"]{background:#0a1f14 !important;
    border:2px dashed #1e3a2f !important;border-radius:14px !important;}
.stProgress>div>div{background:linear-gradient(90deg,#4ade80,#22d3ee) !important;
    border-radius:999px !important;}
div[data-testid="stExpander"]{background:#0f1923 !important;
    border:1px solid #1e3a2f !important;border-radius:12px !important;}
.stSelectbox>div>div{background:#0f1923 !important;border-color:#1e3a2f !important;
    color:#c8e6c9 !important;border-radius:10px !important;}
div[data-testid="metric-container"]{background:#0f1923 !important;
    border:1px solid #1e3a2f !important;border-radius:12px !important;padding:16px !important;}
.stDivider{border-color:#1e3a2f !important;}
h1,h2,h3,h4{color:#e2e8f0 !important;font-family:'Sora',sans-serif !important;}
p,li,.stMarkdown{color:#94a3b8 !important;}
.stCaption{color:#4b7063 !important;}
</style>
""", unsafe_allow_html=True)


# =============================================================
# MODEL LOADING — cached, runs only once
# =============================================================

HF_BASE = "https://huggingface.co/shivcodes91/plant-disease-efficientnet/resolve/main"

@st.cache_resource(show_spinner=False)
def load_everything():
    """
    Loads model, class mapping, transform.
    @st.cache_resource means this runs ONCE and stays in memory.
    Subsequent page refreshes reuse the cached model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path   = BASE_DIR / 'backend' / 'models' / 'best_model.pt'
    mapping_path = BASE_DIR / 'data' / 'class_mapping.json'

    # Download from HuggingFace if not present
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("🌿 Downloading AI model (~190MB) — first launch only, please wait..."):
            urllib.request.urlretrieve(f"{HF_BASE}/best_model.pt", str(model_path))

    if not mapping_path.exists():
        urllib.request.urlretrieve(f"{HF_BASE}/class_mapping.json", str(mapping_path))

    model, checkpoint = load_checkpoint(str(model_path), device=device)
    model.eval()

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
    transform    = get_val_transforms(224)

    return model, idx_to_class, device, transform, checkpoint


# =============================================================
# INFERENCE HELPERS
# =============================================================

def run_prediction(image_bytes, model, idx_to_class, device, transform):
    """Run model inference on uploaded image bytes."""
    image     = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(image)
    original  = cv2.resize(img_array, (224, 224))
    tensor    = transform(image=img_array)['image']

    batch = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probs   = F.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probs, k=3, dim=1)
    top_probs   = top_probs[0].cpu().numpy().tolist()
    top_indices = top_indices[0].cpu().numpy().tolist()

    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = idx_to_class[idx]
        predictions.append({
            'rank'        : len(predictions) + 1,
            'class_name'  : class_name,
            'display_name': class_name.replace('___',' ').replace('__',' ').replace('_',' '),
            'confidence'  : round(prob * 100, 2),
            'is_healthy'  : 'healthy' in class_name.lower()
        })

    return predictions, tensor, original


def get_gradcam_image(model, device, tensor, original):
    """Generate Grad-CAM heatmap overlay."""
    try:
        gradcam = GradCAM(model, device)
        cam, _, _ = gradcam.generate(tensor)
        gradcam.remove_hooks()
        return apply_heatmap(original, cam, alpha=0.4)
    except Exception as e:
        st.warning(f"Grad-CAM skipped: {e}")
        return None


def severity_pill(sev, is_healthy=False):
    if is_healthy:
        return '<span class="pill pill-healthy">✓ Healthy</span>'
    icons = {"severe":"⚠ Severe","moderate":"⚡ Moderate","none":"✓ None"}
    cls   = {"severe":"pill-severe","moderate":"pill-moderate","none":"pill-none"}
    return f'<span class="pill {cls.get(sev,"pill-moderate")}">{icons.get(sev,sev.upper())}</span>'


def conf_class(c):
    return "conf-high" if c > 80 else "conf-medium" if c > 60 else "conf-low"


# =============================================================
# LOAD MODEL (with nice loading screen)
# =============================================================

with st.spinner("🌿 Initializing LeafScan AI..."):
    try:
        MODEL, IDX_TO_CLASS, DEVICE, TRANSFORM, CHECKPOINT = load_everything()
        model_loaded = True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model_loaded = False


# =============================================================
# SIDEBAR
# =============================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0 1rem;'>
        <div style='font-size:3rem;margin-bottom:8px;'>🌿</div>
        <div style='font-family:Sora,sans-serif;font-size:1.3rem;
                    font-weight:700;color:#4ade80;'>LeafScan AI</div>
        <div style='font-size:0.8rem;color:#4b7063;margin-top:4px;'>
            Plant Disease Detection
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.selectbox("nav",
        ["🔍  Detect Disease","📚  Disease Library",
         "📊  Model Statistics","ℹ️   About"],
        label_visibility="collapsed")

    st.divider()

    # Model status
    if model_loaded:
        st.markdown("""
        <div style='display:flex;align-items:center;gap:8px;
                    background:#052e16;border:1px solid #16a34a;
                    border-radius:10px;padding:10px 14px;'>
            <div style='width:8px;height:8px;border-radius:50%;
                        background:#4ade80;box-shadow:0 0 6px #4ade80;'></div>
            <span style='font-size:0.88rem;color:#4ade80;font-weight:500;'>
                Model Ready
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.error("Model failed to load")

    st.divider()

    for label, val, color in [
        ("Model",    "EfficientNet-B4", "#c8e6c9"),
        ("Accuracy", "98.9%",           "#4ade80"),
        ("Dataset",  "41,274 images",   "#c8e6c9"),
        ("Classes",  "15 diseases",     "#c8e6c9"),
    ]:
        st.markdown(f"""
        <div style='background:#0f1923;border:1px solid #1e3a2f;
                    border-radius:10px;padding:10px 14px;margin-bottom:6px;'>
            <div style='font-size:0.7rem;color:#4b7063;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:3px;'>{label}</div>
            <div style='font-size:0.9rem;color:{color};font-weight:500;'>{val}</div>
        </div>""", unsafe_allow_html=True)


# =============================================================
# PAGE 1 — DETECT DISEASE
# =============================================================

if page == "🔍  Detect Disease":

    st.markdown("""
    <div class="hero-title">Plant Disease Detection</div>
    <div class="hero-sub">Upload a leaf photo — AI diagnoses instantly and recommends treatments</div>
    """, unsafe_allow_html=True)
    st.divider()

    left, right = st.columns([1,1], gap="large")

    with left:
        st.markdown('<div class="section-title">📸 Upload Leaf Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop leaf image", type=["jpg","jpeg","png","webp"],
                                     label_visibility="collapsed")
        if uploaded:
            st.image(Image.open(uploaded), use_container_width=True,
                     caption=f"📎 {uploaded.name}")
            with st.expander("📌 Tips for best accuracy"):
                st.markdown("""
                - 📷 Use good natural lighting
                - 🍃 Focus on the diseased area
                - 📐 Keep leaf flat and fully in frame
                - 🔍 Capture 1–2 lesions clearly
                """)
        else:
            st.markdown("""
            <div style='background:#0a1f14;border:2px dashed #1e3a2f;border-radius:16px;
                        padding:50px 20px;text-align:center;margin-top:10px;'>
                <div style='font-size:3rem;margin-bottom:12px;'>🍃</div>
                <div style='font-family:Sora,sans-serif;font-size:1.1rem;
                            font-weight:600;color:#4b7063;'>Drop your leaf image here</div>
                <div style='font-size:0.88rem;color:#2d4a3e;margin-top:6px;'>
                    JPG · PNG · WEBP &nbsp;·&nbsp; Tomato · Potato · Pepper
                </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">🧠 Analysis Results</div>', unsafe_allow_html=True)

        if uploaded is None:
            st.markdown("""
            <div style='background:#0f1923;border:1px solid #1e3a2f;border-radius:16px;
                        padding:60px 30px;text-align:center;'>
                <div style='font-size:2.5rem;margin-bottom:12px;'>🔬</div>
                <div style='font-size:1rem;font-weight:600;color:#4b7063;'>
                    Waiting for image...
                </div>
            </div>""", unsafe_allow_html=True)

        elif not model_loaded:
            st.error("Model not loaded. Please refresh the page.")

        else:
            with st.spinner("🔬 Analyzing leaf tissue..."):
                predictions, tensor, original = run_prediction(
                    uploaded.getvalue(), MODEL, IDX_TO_CLASS, DEVICE, TRANSFORM
                )

            primary    = predictions[0]
            remedy     = get_quick_summary(primary['class_name'])
            is_healthy = primary['is_healthy']
            severity   = remedy.get('severity','moderate')
            confidence = primary['confidence']
            dname      = remedy.get('display_name', primary['display_name'])

            card_cls = ("result-healthy" if is_healthy
                        else "result-severe" if severity=="severe"
                        else "result-diseased")
            name_cls = ("disease-name-healthy" if is_healthy
                        else "disease-name-severe" if severity=="severe"
                        else "disease-name-moderate")
            status   = ("✅ Your plant is healthy!"
                        if is_healthy
                        else "🚨 Critical disease!" if severity=="severe"
                        else "⚠️ Disease detected")

            st.markdown(f"""
            <div class="{card_cls}">
                <div style='display:flex;justify-content:space-between;
                            align-items:flex-start;flex-wrap:wrap;gap:8px;'>
                    <div>
                        <div style='font-size:0.88rem;color:#4b7063;
                                    font-weight:500;margin-bottom:4px;'>{status}</div>
                        <div class="disease-name {name_cls}">{dname}</div>
                        <div style='margin-top:8px;'>{severity_pill(severity,is_healthy)}</div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-family:Sora,sans-serif;font-size:2.2rem;
                                    font-weight:700;color:{"#4ade80" if confidence>80 else "#fbbf24" if confidence>60 else "#f87171"};'>
                            {confidence:.1f}%
                        </div>
                        <div style='font-size:0.8rem;color:#4b7063;'>confidence</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.progress(confidence / 100)
            if confidence <= 60:
                st.warning("⚠️ Low confidence — try a clearer photo.")

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-block">
                    <div class="metric-label">Plant</div>
                    <div class="metric-value">{remedy.get('plant','Unknown')}</div>
                </div>
                <div class="metric-block">
                    <div class="metric-label">Type</div>
                    <div class="metric-value">{remedy.get('pathogen_type') or 'N/A'}</div>
                </div>
                <div class="metric-block">
                    <div class="metric-label">Severity</div>
                    <div class="metric-value">{severity.upper()}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            with st.expander("📊 Top 3 Predictions"):
                for p in predictions:
                    c1,c2 = st.columns([4,1])
                    with c1:
                        st.markdown(f"{'🟢' if p['is_healthy'] else '🔴'} **{p['rank']}. {p['display_name']}**")
                        st.progress(p['confidence']/100)
                    with c2:
                        st.markdown(f"<div style='text-align:right;font-weight:700;color:#4ade80;padding-top:4px;'>{p['confidence']:.1f}%</div>",
                                    unsafe_allow_html=True)

            if remedy.get('overview'):
                st.markdown('<div class="section-title">📋 Overview</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background:#0f1923;border:1px solid #1e3a2f;border-radius:12px;
                            padding:14px 18px;font-size:0.95rem;color:#94a3b8;line-height:1.7;'>
                    {remedy['overview']}
                </div>""", unsafe_allow_html=True)

            if remedy.get('immediate_action') and not is_healthy:
                st.markdown('<div class="section-title">⚡ Immediate Action</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="action-box">
                    🚨 <strong>Act Now:</strong> {remedy['immediate_action']}
                </div>""", unsafe_allow_html=True)

            if remedy.get('first_treatment') and not is_healthy:
                t = remedy['first_treatment']
                brands = ", ".join(t.get('brand_examples',[]))
                st.markdown('<div class="section-title">💊 Recommended Treatment</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="treatment-box">
                    <div class="treatment-name">{t['name']}</div>
                    {'<div class="treatment-row"><span>🏷️</span><span>Brand: '+brands+'</span></div>' if brands else ''}
                    <div class="treatment-row"><span>📏</span><span>Dosage: {t['dosage']}</span></div>
                    <div class="treatment-row"><span>🔄</span><span>Frequency: {t['frequency']}</span></div>
                    <div class="treatment-row"><span>⏰</span><span>Timing: {t.get('timing','N/A')}</span></div>
                    <div class="treatment-row"><span>💡</span><span>{t['notes']}</span></div>
                </div>""", unsafe_allow_html=True)
                if remedy.get('has_organic_options'):
                    st.markdown("""
                    <div style='display:inline-flex;align-items:center;gap:6px;
                                background:#052e16;border:1px solid #4ade80;
                                border-radius:8px;padding:8px 14px;
                                font-size:0.85rem;color:#4ade80;font-weight:500;'>
                        🌱 Organic options available — see Disease Library
                    </div>""", unsafe_allow_html=True)

            # Grad-CAM
            if confidence > 30:
                st.markdown('<div class="section-title">🔥 AI Attention Map</div>', unsafe_allow_html=True)
                st.caption("Red = disease focus · Blue = low attention")
                with st.spinner("Generating heatmap..."):
                    overlay = get_gradcam_image(MODEL, DEVICE, tensor, original)
                if overlay is not None:
                    st.image(overlay, use_container_width=True)


# =============================================================
# PAGE 2 — DISEASE LIBRARY
# =============================================================

elif page == "📚  Disease Library":

    st.markdown("""
    <div class="hero-title">Disease Library</div>
    <div class="hero-sub">Browse all 15 diseases with full treatment and prevention info</div>
    """, unsafe_allow_html=True)
    st.divider()

    diseases = get_all_diseases()

    f1,f2,f3 = st.columns(3)
    with f1: plant_f  = st.selectbox("🌱 Plant",    ["All Plants","Tomato","Potato","Pepper"])
    with f2: sev_f    = st.selectbox("🎯 Severity", ["All","severe","moderate","none"])
    with f3: health_f = st.selectbox("🔍 Show",     ["All","Diseased only","Healthy only"])

    filtered = diseases.copy()
    if plant_f  != "All Plants":    filtered = [d for d in filtered if d["plant"].lower().startswith(plant_f.lower())]
    if sev_f    != "All":           filtered = [d for d in filtered if d["severity"]==sev_f]
    if health_f == "Diseased only": filtered = [d for d in filtered if not d["is_healthy"]]
    elif health_f == "Healthy only":filtered = [d for d in filtered if d["is_healthy"]]

    sc = sum(1 for d in filtered if d["severity"]=="severe")
    mc = sum(1 for d in filtered if d["severity"]=="moderate")
    hc = sum(1 for d in filtered if d["is_healthy"])

    st.markdown(f"""
    <div style='display:flex;gap:10px;margin:12px 0;flex-wrap:wrap;'>
        <div style='background:#0f1923;border:1px solid #1e3a2f;border-radius:10px;
                    padding:8px 16px;font-size:0.88rem;color:#c8e6c9;'>
            📋 Showing <strong>{len(filtered)}</strong></div>
        <div style='background:#1c0202;border:1px solid #dc2626;border-radius:10px;
                    padding:8px 16px;font-size:0.88rem;color:#f87171;'>
            🔴 Severe: <strong>{sc}</strong></div>
        <div style='background:#1c1002;border:1px solid #d97706;border-radius:10px;
                    padding:8px 16px;font-size:0.88rem;color:#fbbf24;'>
            🟡 Moderate: <strong>{mc}</strong></div>
        <div style='background:#052e16;border:1px solid #16a34a;border-radius:10px;
                    padding:8px 16px;font-size:0.88rem;color:#4ade80;'>
            🟢 Healthy: <strong>{hc}</strong></div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    for disease in filtered:
        sev = disease["severity"]; ih = disease["is_healthy"]
        icon = "🟢" if ih else "🔴" if sev=="severe" else "🟡"
        with st.expander(f"{icon}  {disease['display_name']}  ·  {disease['plant']}  ·  {'HEALTHY' if ih else sev.upper()}"):
            full = get_full_remedy_db(disease["class_name"])
            if not full: st.warning("Could not load details."); continue

            ca, cb = st.columns([1,1], gap="large")
            with ca:
                st.markdown('<div class="section-title">📋 Overview</div>', unsafe_allow_html=True)
                st.markdown(full.get("overview","N/A"))
                if full.get("pathogen_name"):
                    st.markdown(f"""
                    <div style='background:#0f1923;border:1px solid #1e3a2f;border-radius:10px;
                                padding:12px 14px;margin:8px 0;'>
                        <div style='font-size:0.72rem;color:#4b7063;text-transform:uppercase;
                                    letter-spacing:0.08em;margin-bottom:5px;'>Pathogen</div>
                        <div style='color:#c8e6c9;font-weight:500;'>🦠 <em>{full['pathogen_name']}</em></div>
                        <div style='color:#4b7063;font-size:0.85rem;margin-top:3px;'>🔬 {full.get('pathogen_type','N/A')}</div>
                    </div>""", unsafe_allow_html=True)
                symptoms = full.get("symptoms",[])
                if symptoms:
                    st.markdown('<div class="section-title">🔍 Symptoms</div>', unsafe_allow_html=True)
                    for s in symptoms: st.markdown(f"• {s}")

            with cb:
                treatments = full.get("treatments",{})
                chem = treatments.get("chemical_treatments",[])
                if chem:
                    st.markdown('<div class="section-title">💊 Treatments</div>', unsafe_allow_html=True)
                    for t in chem:
                        brands = ", ".join(t.get("brand_examples",[]))
                        st.markdown(f"""
                        <div style='background:#0f1923;border:1px solid #1e3a2f;
                                    border-left:3px solid #4ade80;border-radius:0 10px 10px 0;
                                    padding:10px 14px;margin:6px 0;'>
                            <div style='font-weight:600;color:#c8e6c9;'>{t['name']}</div>
                            <div style='font-size:0.8rem;color:#4b7063;'>{brands}</div>
                            <div style='font-size:0.8rem;color:#6b8e7a;margin-top:3px;'>
                                📏 {t['dosage']} &nbsp;·&nbsp; 🔄 {t['frequency']}
                            </div>
                        </div>""", unsafe_allow_html=True)

                organic = treatments.get("organic_treatments",[])
                if organic:
                    st.markdown('<div class="section-title">🌱 Organic</div>', unsafe_allow_html=True)
                    for t in organic:
                        st.markdown(f"""
                        <div style='background:#052e16;border:1px solid #166534;border-radius:10px;
                                    padding:8px 12px;margin:5px 0;font-size:0.88rem;color:#86efac;'>
                            🌿 <strong>{t['name']}</strong>
                            <span style='color:#4b7063;'> — {t['dosage']}</span>
                        </div>""", unsafe_allow_html=True)

                prevention = full.get("prevention",[])
                if prevention:
                    st.markdown('<div class="section-title">🛡️ Prevention</div>', unsafe_allow_html=True)
                    for p in prevention[:5]: st.markdown(f"• {p}")

            timeline = full.get("treatment_timeline",{})
            if timeline and not ih:
                st.markdown('<div class="section-title">📅 Timeline</div>', unsafe_allow_html=True)
                items = list(timeline.items())[:4]
                tcols = st.columns(len(items))
                for i,(period,action) in enumerate(items):
                    with tcols[i]:
                        st.markdown(f"""
                        <div style='background:#0f1923;border:1px solid #1e3a2f;border-radius:10px;
                                    padding:10px 12px;text-align:center;'>
                            <div style='font-size:0.72rem;color:#4b7063;font-weight:600;
                                        text-transform:uppercase;letter-spacing:0.08em;
                                        margin-bottom:5px;'>{period.replace('_',' ').title()}</div>
                            <div style='font-size:0.82rem;color:#94a3b8;line-height:1.5;'>{action}</div>
                        </div>""", unsafe_allow_html=True)


# =============================================================
# PAGE 3 — MODEL STATISTICS
# =============================================================

elif page == "📊  Model Statistics":

    st.markdown("""
    <div class="hero-title">Model Statistics</div>
    <div class="hero-sub">Performance metrics and evaluation results</div>
    """, unsafe_allow_html=True)
    st.divider()

    eval_path = BASE_DIR / 'data' / 'evaluation_report.json'
    eval_data = {}
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:2rem;'>
        <div class="stat-card">
            <div class="stat-number">{eval_data.get('overall_accuracy',98.39)}%</div>
            <div class="stat-label">Test Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{eval_data.get('avg_confidence',89.03)}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">18.6M</div>
            <div class="stat-label">Parameters</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">25</div>
            <div class="stat-label">Epochs Trained</div>
        </div>
    </div>""", unsafe_allow_html=True)

    per_class = eval_data.get("per_class_results",[])
    if per_class:
        st.markdown('<div class="section-title">📈 Per-Class Accuracy</div>', unsafe_allow_html=True)
        df = pd.DataFrame(per_class)
        df["short_name"] = df["class_name"].apply(
            lambda x: x.replace("Pepper__bell___","Pepper ").replace("Potato___","Potato ")
                       .replace("Tomato__","Tomato ").replace("Tomato_","Tomato ").replace("_"," ")
        )
        df = df.sort_values("accuracy")
        colors = ["#4ade80" if a>=98 else "#fbbf24" if a>=95 else "#f87171" for a in df["accuracy"]]
        fig = go.Figure(go.Bar(
            x=df["accuracy"], y=df["short_name"], orientation='h',
            marker_color=colors, marker_line_width=0,
            text=[f"{a:.1f}%" for a in df["accuracy"]],
            textposition='outside', textfont=dict(color="#c8e6c9",size=12)
        ))
        fig.update_layout(
            xaxis=dict(range=[85,102],tickfont=dict(color="#4b7063"),gridcolor="#1e3a2f",
                       title=dict(text="Accuracy (%)",font=dict(color="#4b7063"))),
            yaxis=dict(tickfont=dict(color="#c8e6c9")),
            height=520, margin=dict(l=0,r=70,t=10,b=40),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', bargap=0.3
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section-title">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        cm_path = BASE_DIR / "data" / "confusion_matrix.png"
        if cm_path.exists(): st.image(str(cm_path), use_container_width=True)
        else: st.info("Run `python ml_training/evaluate.py` to generate")
    with c2:
        st.markdown('<div class="section-title">📉 Training Curves</div>', unsafe_allow_html=True)
        cp = BASE_DIR / "data" / "training_curves.png"
        if cp.exists(): st.image(str(cp), use_container_width=True)


# =============================================================
# PAGE 4 — ABOUT
# =============================================================

elif page == "ℹ️   About":

    st.markdown("""
    <div class="hero-title">About LeafScan AI</div>
    <div class="hero-sub">End-to-end AI system for plant disease diagnosis and treatment</div>
    """, unsafe_allow_html=True)
    st.divider()

    c1,c2 = st.columns([3,2], gap="large")
    with c1:
        st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
        st.markdown("LeafScan AI helps farmers identify plant diseases instantly from a single leaf photo and get actionable treatment advice.")

        for icon,title,desc in [
            ("🧠","98.9% Accuracy","EfficientNet-B4 trained on 41,274 PlantVillage images"),
            ("🔥","Grad-CAM Heatmaps","Visual explanation of where disease was detected"),
            ("💊","Full Remedy Database","Chemical + organic treatments with dosage and timing"),
            ("⚡","Real-time Inference","Results in seconds"),
            ("🌱","15 Disease Classes","Tomato, Potato and Pepper"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:14px;align-items:flex-start;background:#0f1923;
                        border:1px solid #1e3a2f;border-radius:12px;padding:14px 16px;margin:8px 0;'>
                <div style='font-size:1.6rem;flex-shrink:0;'>{icon}</div>
                <div>
                    <div style='font-family:Sora,sans-serif;font-weight:600;
                                color:#c8e6c9;font-size:0.95rem;'>{title}</div>
                    <div style='font-size:0.85rem;color:#4b7063;margin-top:3px;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with c2:
        for label,val,color in [
            ("Validation Accuracy","98.87%","#4ade80"),
            ("Test Accuracy","98.39%","#4ade80"),
            ("Avg Confidence","89.03%","#22d3ee"),
            ("Training Epochs","25","#a78bfa"),
            ("Test Images","6,192","#fbbf24"),
            ("Correct Predictions","6,092","#4ade80"),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        background:#0f1923;border:1px solid #1e3a2f;border-radius:10px;
                        padding:10px 14px;margin:6px 0;'>
                <span style='font-size:0.88rem;color:#4b7063;'>{label}</span>
                <span style='font-family:Sora,sans-serif;font-weight:700;
                             font-size:1rem;color:{color};'>{val}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#052e16,#0a1628);
                    border:1px solid #16a34a;border-radius:14px;
                    padding:20px;text-align:center;'>
            <div style='font-size:2rem;margin-bottom:8px;'>👨‍💻</div>
            <div style='font-family:Sora,sans-serif;font-size:1.1rem;
                        font-weight:700;color:#4ade80;'>Shivang Rai</div>
            <div style='font-size:0.85rem;color:#4b7063;margin-top:4px;'>
                Built with ❤️ using PyTorch & Streamlit
            </div>
            <a href="https://github.com/Shivcodes91/plant-disease-detection"
               target="_blank"
               style='display:inline-block;margin-top:12px;background:#0f1923;
                      border:1px solid #1e3a2f;border-radius:8px;padding:8px 18px;
                      color:#c8e6c9;font-size:0.85rem;text-decoration:none;font-weight:500;'>
                🔗 GitHub Repository
            </a>
        </div>""", unsafe_allow_html=True)