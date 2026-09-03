# Multimodal Social Media Virality Prediction & Explainability Engine
## Comprehensive Project Architecture, Mathematical Formulations, and Technical Report

---

### Executive Summary

In short-form social video platforms (TikTok, Instagram Reels, YouTube Shorts), organic reach is governed by recommendation algorithms that evaluate content engagement signals within milliseconds of release. Traditional virality models rely exclusively on posting metadata (such as follower counts or upload hours) or unimodal NLP text features. 

This project implements an end-to-end, industrial-grade **Multimodal Machine Learning and Explainability Engine** that predicts the viral potential of short-form videos prior to publication. By fusing **computer vision (OpenCV)**, **audio digital signal processing (Librosa)**, **deep contextual NLP (Sentence-BERT & TextBlob)**, and **cyclical temporal metadata**, the system achieves a **+42.9% Macro F1 baseline uplift** over metadata-only approaches. 

Beyond point-estimate predictions, the platform applies **TreeSHAP (game-theoretic cooperative feature attribution)** to decompose the prediction into positive catalysts and negative bottlenecks, paired with a **Prescriptive Optimization Engine** that delivers targeted recommendations to creators.

---

## 1. High-Level System Architecture

The project follows a decoupled, 5-tier layered design:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PRESENTATION TIER                               │
│        SPA Architecture (HTML5, Vanilla CSS Glassmorphism, Chart.js)        │
│          - Dynamic Constellation Canvas  - History Persistence (LocalStorage)│
│          - Interactive Dropzone          - Dual-Tier Real-Time Charts       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP REST / JSON / Multipart
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API & SERVICE TIER                               │
│               FastAPI (Uvicorn ASGI Engine, Pydantic v2 Contracts)          │
│          - POST /api/v1/predict/video    - POST /api/v1/predict/json        │
│          - GET  /api/v1/health           - GET  /api/v1/metrics/ablation    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Raw Artifacts & Meta
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MULTIMODAL FEATURE EXTRACTION LAYER                    │
│ ┌───────────────────┬───────────────────┬───────────────────┬─────────────┐ │
│ │ Visual (OpenCV)   │ Acoustic (Librosa)│ NLP (SBERT/VADER) │ Meta/Time   │ │
│ │ - Optical Flow    │ - RMS Energy      │ - 16-D Embeddings │ - Cyclical  │ │
│ │ - Color Variance  │ - Tempo (BPM)     │ - Sentiment Pol.  │   Sin/Cos   │ │
│ │ - Scene Cuts      │ - Spec. Centroid  │ - Subjectivity    │ - Weekend   │ │
│ │ - 16-D Embeddings │ - Zero-Cross Rate │ - Hashtag Density │ - Category  │ │
│ └───────────────────┴───────────────────┴───────────────────┴─────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ 57-Dimensional Unified Feature Vector
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INFERENCE & ENSEMBLE TIER                             │
│       StandardScaler Normalization ──► XGBoost Dual-Model System            │
│       1. XGBoost Regressor (Continuous Virality Index [0, 100])             │
│       2. XGBoost Softmax Classifier (Low / Medium / High / Viral)           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Model Trees & Leaf Outputs
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   EXPLAINABILITY & PRESCRIPTIVE TIER                        │
│          - TreeSHAP Fast Tree Explainer (Marginal Shapley Values)           │
│          - Feature Impact Ranking (Top Catalysts vs. Top Bottlenecks)       │
│          - Rule-Driven Creator Prescriptive Engine (Ranked Priority Advice) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Dive: Layer-by-Layer Functionality & Algorithms

### Layer 1: Data Preprocessing & Target Engineering

#### Follower-Normalized Engagement Ratio (NER)
Raw view counts bias models toward large accounts. To capture **content-driven virality**, the target variable is formulated as:
$$\text{NER} = \frac{\text{Engagement Volume}}{\max(\text{Follower Count}, 100)}$$
To prevent small-account denominator instability, a smoothing constant $k$ is utilized:
$$\text{NER}_{\text{smoothed}} = \frac{\text{Likes} + \alpha \cdot \text{Comments} + \beta \cdot \text{Shares}}{\text{Followers} + k}$$

#### Percentile-Based Discretization
To formulate both regression and multi-class classification objectives:
* **Low ($\le 50\text{th}$ percentile):** Baseline engagement.
* **Medium ($50\text{th} - 75\text{th}$ percentile):** Moderate organic discovery.
* **High ($75\text{th} - 90\text{th}$ percentile):** Strong recommendation push.
* **Viral ($> 90\text{th}$ percentile):** Algorithmic breakout tier.

---

### Layer 2: Multimodal Feature Engineering (57 Total Features)

```
Total Feature Vector Dimension: 57
├── Metadata & Temporal (5):
│   ├── duration_sec
│   ├── upload_hour_sin, upload_hour_cos
│   ├── is_weekend
│   └── category_id
├── Visual Modality (20):
│   ├── scene_cut_count
│   ├── avg_brightness
│   ├── color_variance
│   ├── motion_score
│   └── visual_emb_0 .. visual_emb_15 (16-D spatial/texture representation)
├── Acoustic Modality (12):
│   ├── rms_energy
│   ├── tempo_bpm
│   ├── zero_crossing_rate
│   ├── spectral_centroid
│   └── audio_emb_0 .. audio_emb_7 (8-D MFCC spectral summary)
└── Textual Modality (20):
    ├── caption_length
    ├── hashtag_count
    ├── sentiment_polarity
    ├── sentiment_subjectivity
    └── text_emb_0 .. text_emb_15 (16-D SBERT semantic embeddings)
```

#### 1. Computer Vision Pipeline (`VideoFeatureExtractor` / OpenCV)
* **Frame Downsampling:** Source videos are sampled uniformly at $1\text{ FPS}$ to maintain computational efficiency while capturing overall narrative progression.
* **Scene Cut Detection:** Computed via normalized grayscale frame difference:
  $$\Delta_t = \frac{1}{H \times W} \sum_{x,y} |I_t(x,y) - I_{t-1}(x,y)|$$
  A scene change is registered whenever $\Delta_t > \tau$, where $\tau = 30.0$.
* **Motion Dynamics (Optical Flow Approximation):** Measures visual dynamics across consecutive frames:
  $$\text{Motion Score} = \frac{1}{T-1} \sum_{t=1}^{T-1} \text{Mean}(|I_t - I_{t-1}|)$$
* **Color Variance & Luminance:** RGB channels are evaluated for average luminance and standard deviation to detect visual richness and production quality.

#### 2. Acoustic Processing Pipeline (`AudioFeatureExtractor` / Librosa)
* **Audio Track Extraction:** Audio streams are stripped using MoviePy/FFmpeg and resampled to $22,050\text{ Hz}$.
* **Tempo (BPM):** Evaluated via dynamic programming beat tracking on the onset strength envelope:
  $$O(t) = \text{OnsetStrength}(S(f, t))$$
  Music in the range of $120\text{--}128\text{ BPM}$ correlates empirically with higher retention.
* **Root Mean Square (RMS) Energy:** Quantifies perceived acoustic loudness and dynamism:
  $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x[n]^2}$$
* **Spectral Centroid & Zero Crossing Rate:** Measures audio brightness/sharpness and noise/percussive density.

#### 3. Natural Language Processing Pipeline (`TextFeatureExtractor` / SBERT + VADER)
* **Dense Semantic Embeddings:** Captures contextual meaning, clickbait hooks, and curiosity triggers using a 16-dimensional projected **Sentence-BERT (`all-MiniLM-L6-v2`)** model.
* **Sentiment Analysis:** Computes sentiment polarity in range $[-1.0, 1.0]$ and subjectivity in range $[0.0, 1.0]$.
* **Discovery Signals:** Tokenizes and extracts hashtag counts ($n_{\text{tags}}$) and caption character length.

#### 4. Temporal Cyclical Encoding
Raw 24-hour integers create an artificial discontinuity between hour 23 (11 PM) and hour 0 (midnight). The pipeline encodes time cyclically:
$$h_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad h_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

---

### Layer 3: Machine Learning & Modeling Engine

The system uses **eXtreme Gradient Boosting (XGBoost)**, chosen for its gradient-boosted decision trees (GBDT) that handle heterogeneous tabular features, non-linear feature interactions, and tabular-multimodal embeddings with lower latency and higher robustness than end-to-end transformers.

#### Dual-Model Architecture
1. **XGBoost Classifier (`multi:softprob`):**
   * Computes the posterior class probability distribution across the 4 classes:
     $$P(y = c \mid \mathbf{x}) = \frac{e^{f_c(\mathbf{x})}}{\sum_{j=1}^{C} e^{f_j(\mathbf{x})}}$$
2. **XGBoost Regressor (`reg:squarederror`):**
   * Predicts a continuous scalar score clipped to $[0.0, 100.0]$:
     $$\hat{y}_{\text{score}} = \sum_{m=1}^{M} f_m(\mathbf{x})$$

#### Hyperparameters & Regularization
* `max_depth`: 4 to 6 (prevents memorization of high-dimensional embeddings)
* `learning_rate` ($\eta$): 0.05
* `subsample`: 0.85 & `colsample_bytree`: 0.85 (stochastic bagging to prevent overfitting)
* `n_estimators`: 250 (optimized with early stopping on validation holdouts)

---

### Layer 4: Model Explainability with TreeSHAP

Standard deep neural networks behave as black boxes. This system integrates **TreeSHAP (Tree-based SHapley Additive exPlanations)**, calculating game-theoretic marginal contributions:

$$\phi_i(f, \mathbf{x}) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

* **Base Value ($E[f(x)]$):** The average virality across all training samples.
* **Marginal Attribution ($\phi_i$):**
  * $\phi_i > 0$ (**Catalysts / Boosters**): Features pushing the instance toward viral reach (e.g., `is_weekend = +4.39`, `scene_cut_count = +3.80`).
  * $\phi_i < 0$ (**Bottlenecks / Drag**): Features holding the score back (e.g., `motion_score = -7.29`).
* **Additive Property Guaranteed:**
  $$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{D} \phi_i$$

---

### Layer 5: Prescriptive Optimization Engine

To bridge the gap between machine learning math and creator actionability, the system processes negative SHAP attributions through a rule-based expert recommendation matrix:

| Identified Negative SHAP Driver | Threshold Condition | Assigned Priority | Human-Actionable Creator Recommendation |
|---|---|---|---|
| `motion_score` | $\phi_{\text{motion}} < 0$ and value $< 0.40$ | **HIGH** | "Visual dynamics are low. Introduce camera pans, cuts, or subject movement in the first 3 seconds to prevent swipe-aways." |
| `sentiment_polarity` | $\phi_{\text{sentiment}} < 0$ and value $< 0.10$ | **HIGH** | "Caption sentiment is flat or negative. Integrate curiosity hooks, emotional trigger words, or emojis." |
| `tempo_bpm` | $\phi_{\text{tempo}} < 0$ and $| \text{BPM} - 125 | > 35$ | **MEDIUM** | "Background audio is too slow or erratic. Sync your video edits with an energetic track around 120–128 BPM." |
| `scene_cut_count` | $\phi_{\text{cuts}} < 0$ and count $< 2$ | **MEDIUM** | "Video appears static. Break up long continuous shots with jump-cuts or visual zooms every 2.5 seconds." |
| `upload_hour` | $\phi_{\text{hour}} < 0$ | **LOW** | "Target upload slot is off-peak. Reschedule publication to prime evening traffic windows (6 PM – 9 PM)." |

---

## 3. Empirical Ablation Benchmarks

To validate that multimodal fusion genuinely improves predictive power over isolated single-modality models, a **temporal split ablation experiment** was executed:

| Modality Set | Features Used | Macro F1 Score | Multi-Class ROC-AUC (OVR) | Relative Gain over Baseline |
|---|---|---|---|---|
| **Metadata Baseline** | Duration, Upload Hour, Weekend, Category | $0.3061$ | $0.5539$ | Baseline |
| **Text Modality Only** | Metadata + SBERT + Sentiment + Hashtags | $0.3472$ | $0.6051$ | $+13.4\%$ |
| **Audio Modality Only** | Metadata + RMS + Tempo + Spectral Feats | $0.2917$ | $0.5647$ | $-4.7\%$ (Acoustic alone is noisy) |
| **Visual Modality Only** | Metadata + OpenCV Flow + Cuts + Brightness | $0.3478$ | $0.6275$ | $+13.6\%$ |
| **Multimodal Full Fusion** | **All 57 Combined Features** | $\mathbf{0.4722}$ | $\mathbf{0.7483}$ | **$\mathbf{+42.9\%}$ F1 Uplift** |

**Key Academic Takeaway:** Audio features alone underperform metadata; however, when fused with visual cuts and text semantics, audio acts as an interaction booster, pushing the full model's ROC-AUC to $0.7483$.

---

## 4. Complete Technology Stack Reference

```
┌───────────────────────────┬───────────────────────────┬──────────────────────────────────────────┐
│ Component / Layer         │ Technology / Framework    │ Primary Purpose / Architectural Role     │
├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────┤
│ Backend Framework         │ Python 3.12 + FastAPI     │ High-performance asynchronous REST API   │
│ ASGI Server               │ Uvicorn                   │ Non-blocking event-loop request handling │
│ Data Contracts            │ Pydantic v2               │ Strict request/response schema validation│
│ Core ML Algorithms        │ XGBoost (GBDT)            │ Non-linear classification and regression │
│ Classical ML & Scaling    │ Scikit-Learn              │ Label encoding, StandardScaler, splitters│
│ Machine Learning Explain  │ TreeSHAP                  │ Game-theoretic local feature attributions│
│ Computer Vision           │ OpenCV (cv2)              │ Optical flow, scene detection, luminance │
│ Audio Signal Processing   │ Librosa & SoundFile       │ STFT, onset detection, tempo, RMS energy │
│ Text Embeddings & NLP     │ Sentence-Transformers     │ 16-D SBERT semantic vector generation    │
│ Lexical Sentiment         │ TextBlob / VADER          │ Polarity & subjectivity scoring          │
│ Data Storage & Cache      │ Joblib                    │ Serialized model & scaler persistence    │
│ Frontend Client           │ HTML5 + Vanilla ES6 JS    │ GPT-style dual-page Single Page App      │
│ Styling & Aesthetics      │ Modern CSS (No Tailwind)  │ Midnight theme (#060A14), glassmorphism  │
│ Charting & Visualizations │ Chart.js v4               │ Dynamic Doughnut, Bar, and SHAP charts   │
│ Interactive Graphics      │ HTML5 Canvas API          │ Ambient constellation particle engine    │
│ State Persistence         │ Web Storage (localStorage)│ Session prediction history management    │
│ Testing Suite             │ Pytest & HTTPX TestClient │ Automated unit, integration, and API tests│
└───────────────────────────┴───────────────────────────┴──────────────────────────────────────────┘
```

---

## 5. Summary of Key Formulas for Examiner Inquiries

1. **Cyclical Hour Encoding:**
   $$h_{\sin} = \sin(2\pi \cdot h / 24), \quad h_{\cos} = \cos(2\pi \cdot h / 24)$$
2. **Normalized Frame Cut Metric:**
   $$\Delta_t = \frac{1}{H \cdot W} \sum_{x,y} |I_t(x,y) - I_{t-1}(x,y)|$$
3. **RMS Acoustic Energy:**
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x[n]^2}$$
4. **XGBoost Objective Function:**
   $$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t), \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$
5. **Shapley Additive Efficiency:**
   $$\sum_{i=1}^{M} \phi_i(f, \mathbf{x}) = f(\mathbf{x}) - \mathbb{E}[f(X)]$$
