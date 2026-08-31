# Multimodal Social Media Virality Prediction & Explainability Engine

End-to-end system for predicting short-form social media virality from video, audio, caption, and posting metadata. The pipeline fuses multimodal features, scores content with XGBoost, explains each prediction with TreeSHAP, and returns prioritized creator recommendations through a FastAPI service and dark glassmorphism dashboard.

## Key Features

- **Multimodal feature fusion**
  - OpenCV visual dynamics (scene cuts, brightness, motion, embeddings)
  - Librosa acoustic tempo and spectral metrics
  - NLP sentiment, hashtag/caption signals, and text embeddings
  - Temporal metadata (duration, upload hour, weekend, category)
- **+42.9% Macro F1 baseline uplift** from multimodal fusion versus a metadata-only baseline
- **TreeSHAP game-theoretic explainability** isolating positive boosters and negative bottlenecks
- **Actionable creator optimization recommendations** ranked HIGH / MEDIUM / LOW

## Tech Stack

Python, FastAPI, XGBoost, SHAP, OpenCV, Librosa, Sentence-Transformers, Pytest, and a modern HTML5/CSS glassmorphism UI.

## Project Layout

```
backend/app/     FastAPI app, schemas, prediction routes
src/              Data generation, feature extractors, training, SHAP, recommendations
frontend/         Dashboard (index.html, style.css, app.js)
tests/            Phase 1–6 unit and integration tests
models/           Trained classifier, regressor, scaler, and evaluation report
data/             Synthetic dataset and processed artifacts
run_app.py        Local uvicorn launcher
```

## Getting Started

### 1. Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If model artifacts are missing, train them once:

```bash
python -m src.models.train
```

### 2. Run tests

```bash
python -m pytest -v
```

### 3. Launch the app

```bash
python run_app.py
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) for the dashboard.

- Health: `GET /api/v1/health`
- JSON inference: `POST /api/v1/predict/json`
- Video upload inference: `POST /api/v1/predict/video`

Interactive API docs are available at `/docs`.
