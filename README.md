# 🌿 Plant Disease Detection System

An AI-powered plant disease detection system that helps farmers 
identify diseases from leaf photos and get instant treatment recommendations.

![Accuracy](https://img.shields.io/badge/Accuracy-98.9%25-brightgreen)
![Model](https://img.shields.io/badge/Model-EfficientNet--B4-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![API](https://img.shields.io/badge/API-FastAPI-teal)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

## 🎯 What it does

- Detects **15 plant diseases** across Tomato, Potato, and Pepper
- **98.9% validation accuracy** using EfficientNet-B4
- **Grad-CAM heatmaps** show exactly where disease is on the leaf
- **Full remedy database** with chemical + organic treatments
- **Real-time inference** via FastAPI backend

## 🚀 Live Demo

- **Frontend:** [Coming soon after deployment]
- **API Docs:** [Coming soon after deployment]

## 📸 Screenshots

### Disease Detection
Upload a leaf photo → instant diagnosis + treatment plan

### Grad-CAM Visualization
AI highlights exactly which part of the leaf shows disease symptoms

### Disease Library
Browse all 15 diseases with full treatment information

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | EfficientNet-B4 (PyTorch 2.6) |
| Explainability | Grad-CAM |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Dataset | PlantVillage (41,274 images) |
| Training | NVIDIA RTX 3050 GPU |

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | 98.87% |
| Test Accuracy | 98.39% |
| Avg Confidence | 89.03% |
| Training Epochs | 25 |

## 🌱 Supported Diseases

### Tomato (9 classes)
- Bacterial Spot, Early Blight, Late Blight
- Leaf Mold, Septoria Leaf Spot, Spider Mites
- Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

### Potato (3 classes)
- Early Blight, Late Blight, Healthy

### Pepper (2 classes)
- Bacterial Spot, Healthy

## 🏃 Running Locally

### Prerequisites
- Python 3.10+
- NVIDIA GPU (recommended) or CPU

### Setup

```bash
# Clone the repo
git clone https://github.com/Shivcodes91/plant-disease-detection
cd plant-disease-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start everything (Windows)
start.bat

# Or manually:
# Terminal 1
python run_api.py

# Terminal 2
streamlit run streamlit_app.py
```

### Open in browser
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

## 📁 Project Structure
plant-disease-detection/
├── backend/
│   ├── app/
│   │   ├── main.py          ← FastAPI application
│   │   ├── db/
│   │   │   └── remedies_db.py
│   └── models/
│       └── best_model.pt    ← Trained model
├── ml_training/
│   ├── dataset.py           ← Data pipeline
│   ├── model.py             ← EfficientNet model
│   ├── train.py             ← Training script
│   ├── evaluate.py          ← Evaluation
│   └── gradcam.py           ← Grad-CAM visualization
├── data/
│   ├── remedies.json        ← Disease remedy database
│   ├── class_mapping.json   ← Class mappings
│   └── test_samples/        ← Sample test images
├── streamlit_app.py         ← Frontend UI
├── run_api.py               ← API startup script
└── requirements.txt


## 👨‍💻 Developer

**Shivang Rai**
- GitHub: [@Shivcodes91](https://github.com/Shivcodes91)

## 📄 License

MIT License — feel free to use for learning and projects!
Step 12 — Final Commit
bash
git add .
git commit -m "Day 10: Polish, README, requirements.txt, test samples, start script"
git push origin main
Day 10 Checklist
Run through these before Day 11:

 use_column_width warnings gone
 Immediate action box shows correctly
 Treatment shows in 2-column layout
 All 4 pages work without errors
 requirements.txt created
 README.md written
 start.bat works — double click starts both services
 Test samples copied to data/test_samples/
 Git pushed
Tell me when all checks pass and we go straight to Day 11 — Deployment! Your app will be live on the internet for anyone to use. 🚀🌿


# ============================================================= # streamlit_app.py # # Plant Disease Detection — Streamlit Frontend # # Run with: streamlit run streamlit_app.py # # PAGES: # 🔍 Detect Disease → upload image → get prediction # 📚 Disease Library → browse all 15 diseases # �

pasted

update my full streamlit_app.py   and amke sure to polish ui more like it look more attractive beuatiful and extra good not just simple basic with every colour combination matches  extra good and add all things which can be polished and make thinks little largers in appereance little larger only.....

May 27
Let me build you a stunning, fully polished Streamlit app!


Check frontend design skill
Check frontend design skill
Now let me create the fully polished Streamlit app:


Created a file, ran a command
Created a file, ran a command

# ============================================================= # streamlit_app.py # # Plant Disease Detection — Streamlit Frontend # # Run with: streamlit run streamlit_app.py # # PAGES: # 🔍 Detect Disease → upload image → get prediction # 📚 Disease Library → browse all 15 diseases # �

pasted

update my full streamlit_app.py   and amke sure to polish ui more like it look more attractive beuatiful and extra good not just simple basic with every colour combination matches  extra good and add all things which can be polished and make thinks little largers in appereance little larger only.....

11:30 PM

Presented file
Presented file
Here's your fully polished streamlit_app.py! Here's what changed:

Visual overhaul:

Deep dark theme (
#0d1117 background) with rich green accents — like a pro developer tool
Gradient hero titles using Sora font (imported from Google Fonts)
Glowing colored top-border cards (green for healthy, orange for moderate, red for severe)
Animated pulsing API status indicator
Result page improvements:

Big confidence number displayed prominently with color coding (green/yellow/red)
Severity pill badges with proper colors
3-column metric blocks (Plant / Type / Severity)
Treatment shown with icon rows instead of plain text
Organic badge styled as a green pill
Sidebar:

Logo + brand name at top
Info cards for model details (not plain captions)
Glowing dot on API status indicator
Disease Library:

Color-coded count badges (severe/moderate/healthy counts)
Treatment cards with green left border
Organic options in green pill style
Model Stats:

4 large stat cards with green underline glow
Dark themed Plotly chart
HTML table for architecture (not markdown table)
About page:

Feature cards with icon + title + desc layout
Performance items as styled key-value rows
Developer card with GitHub link button
Replace your existing streamlit_app.py with this file, then run streamlit run streamlit_app.py to see it! 🌿


Streamlit app
PY 

## 👨‍💻 Developer

**Shivang Rai**
- GitHub: [@Shivcodes91](https://github.com/Shivcodes91)

## 📄 License

MIT License — feel free to use for learning and projects!
