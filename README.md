# Multimodal Social Media Virality Prediction & Explainability Engine

An end-to-end **Machine Learning system for pre-publication social media video virality prediction**. The engine analyzes visual, acoustic, textual, and temporal characteristics of a video to estimate its expected performance tier before publication.

The system combines **XGBoost-based classification, multimodal feature engineering, TreeSHAP explainability, OpenCV, Librosa, Sentence-Transformers, FastAPI, and an interactive OLED-style dashboard** into a reproducible research and deployment pipeline.

---

## Research Objective

The core research question is:

> **Can multimodal pre-publication content characteristics predict social media video virality more effectively than individual modalities or scheduling metadata alone?**

Unlike post-publication analytics systems, the predictive pipeline is restricted to information available **before a video is published**. This prevents downstream engagement signals from leaking into the prediction process.

---

# Academic Evaluation Scope

This project is structured as a final-year AIML engineering project for an Emerging Technologies evaluation. It combines research planning, system design, implementation, testing, and an industry-oriented prototype.

## Problem Analysis and Research Planning

The project addresses the difficulty of estimating short-form content performance before publication. Existing creator decisions often depend on raw historical views, scheduling intuition, or isolated content signals. These approaches do not adequately separate content characteristics from account size and platform effects.

The project investigates the following research question:

> Can multimodal pre-publication signals improve virality-tier prediction over scheduling and metadata features alone?

The research plan includes:

- A reach-normalized virality target.
- Visual, acoustic, textual, and temporal feature extraction.
- Leakage-aware temporal validation.
- Modality ablation experiments.
- Explainable predictions and creator-oriented recommendations.

## System Design and Development

The implementation is organized as a layered machine-learning system:

1. Raw content and metadata ingestion.
2. Visual, audio, text, and temporal feature extraction.
3. Feature scaling and multimodal fusion.
4. XGBoost classification and regression.
5. TreeSHAP or low-resource feature attribution.
6. FastAPI inference services.
7. Interactive browser dashboard.

The system is designed to demonstrate technical complexity beyond a single-model prediction script. It includes serialized model artifacts, API contracts, validation schemas, automated tests, ablation reporting, explainability, confidence estimation, and deployment configuration.

## Testing, Research Outcome, and Project Management

The project evaluates both predictive performance and system reliability. Evaluation artifacts include:

- Macro-F1, weighted F1, precision, and recall.
- Multi-class ROC-AUC where applicable.
- Confusion matrices.
- Regression MAE, RMSE, and R².
- Metadata-versus-multimodal comparison.
- Text, audio, and visual ablation results.
- Temporal train/test validation.
- API, data-pipeline, feature-extractor, and explainability tests.

The repository also contains architecture documentation, reproducibility instructions, model artifacts, deployment configuration, and presentation planning material. Team contributions should be recorded separately in the final project documentation.

## Research and Engineering Contribution

The primary contribution is an integrated, leakage-aware, explainable multimodal framework for estimating pre-publication social-media content potential. The work combines established machine-learning methods into a reproducible engineering system and evaluates the value of each modality through controlled ablation experiments.

The current benchmark uses the dataset available in this repository. Results should therefore be interpreted within that dataset scope. The model estimates historical patterns and does not guarantee future platform performance. Real-world deployment would require a larger, ethically collected longitudinal dataset and creator-disjoint evaluation.

---

# Research & Methodology

## 1. Mathematically Defined Virality Target

Raw view counts are strongly influenced by account size. To reduce this bias, virality is modeled relative to an account's follower base.

Two formulations are evaluated:

### Threshold-Floor Model

\[
V_1 =
\frac{\text{Views}}
{\max(\text{Followers},100)}
\]

The denominator floor prevents extremely small accounts from producing unstable ratios.

### Additive-Smoothed Model

\[
V_2 =
\frac{\text{Views}}
{\text{Followers}+K}
\]

where:

\[
K = 1000
\]

Both formulations are evaluated empirically to determine whether the resulting virality classifications remain stable, particularly for low-follower accounts.

### Virality Tiering

The continuous virality ratio is converted into ordinal performance tiers using empirical percentile thresholds:

| Percentile | Tier |
|---|---|
| < P50 | Low |
| P50–P75 | Medium |
| P75–P90 | High |
| P90–P98 | Very High |
| ≥ P98 | Viral |

The percentile thresholds are calculated from the training data and then applied to held-out data to prevent target-distribution leakage.

---

## 2. Pre-Publication Feature Constraint

The predictive feature vector is intentionally restricted to information available before publication:

\[
X =
\{
F_{\text{visual}},
F_{\text{audio}},
F_{\text{text}},
F_{\text{temporal}}
\}
\]

### Included

- Visual motion and scene dynamics
- Color and composition statistics
- Scene-cut characteristics
- Audio spectral characteristics
- MFCC features
- BPM and acoustic descriptors
- Caption/text embeddings
- Sentiment/polarity
- Upload day and scheduled publication time

### Explicitly Excluded

The predictive pipeline does **not** use post-publication engagement signals such as:

- Views
- Likes
- Shares
- Comments
- Watch-time
- Audience retention
- Post-publication engagement velocity

This separation is a core requirement of the experimental design.

---

# 3. Leakage-Free Validation

Random train/test splitting can produce overly optimistic results in social-media datasets because related content, trending audio, seasonal events, and recurring topics can appear in both partitions.

Therefore, the project evaluates generalization using two complementary protocols.

### Temporal Train/Test Split

The dataset is ordered chronologically:

- **First 80% → Training**
- **Final 20% → Testing**

The test set therefore represents a later period than the training data.

This provides a more realistic approximation of deployment, where the model predicts future content using historical observations.

### Leave-One-Category-Out (LOCO)

Cross-category generalization is evaluated by withholding an entire content category during testing.

Example:

```text
Training:
Gaming
Beauty
Technology
Fitness

Testing:
Dance
```

The model never sees samples from the held-out category during training.

This evaluates whether learned multimodal patterns generalize beyond the domains represented in the training set.

---

# 4. Class-Imbalance Handling

Virality distributions are inherently skewed, with extreme performance concentrated in a relatively small number of samples.

Instead of applying SMOTE directly to the heterogeneous multimodal feature space, the training pipeline uses **class-aware sample weighting**.

For sample \(i\):

\[
w_i =
\frac{N}
{C \times N_{c_i}}
\]

where:

- \(N\) = total number of training samples
- \(C\) = number of classes
- \(N_{c_i}\) = number of samples belonging to sample \(i\)'s class

This increases the contribution of underrepresented virality tiers during model optimization without generating synthetic multimodal feature vectors.

---

# 5. Multimodal Feature Engineering

The system extracts complementary representations from four signal groups.

### Text

- Sentence-Transformer embeddings
- Caption semantics
- Sentiment/polarity features

### Audio

- MFCCs
- Spectral centroid
- Spectral characteristics
- BPM
- Other acoustic descriptors

### Visual

- Motion dynamics
- Scene transitions
- Frame-level statistics
- Color/composition characteristics

### Temporal Metadata

- Upload day
- Scheduled publication hour
- Temporal publication features

These representations are combined into a unified feature space for downstream classification.

---

# 6. Empirical Modality Ablation

Ablation experiments measure how much predictive signal each modality contributes.

| Model Variant | Feature Input | Macro F1 | Weighted F1 | Multi-Class ROC-AUC |
|:---|:---|:---:|:---:|:---:|
| **Baseline** | Temporal & Schedule Metadata | *Generated* | *Generated* | *Generated* |
| **Ablation 1** | Text Only | *Generated* | *Generated* | *Generated* |
| **Ablation 2** | Audio Only | *Generated* | *Generated* | *Generated* |
| **Ablation 3** | Visual Only | *Generated* | *Generated* | *Generated* |
| **Fusion Engine** | **Text + Audio + Visual + Temporal Metadata** | **Generated** | **Generated** | **Generated** |

The results are generated reproducibly by:

```text
src/models/train_ablation.py
```

and exported to:

```text
reports/ablation_results.json
```

### Research Goal

The ablation framework is designed to answer:

1. How much predictive signal does each modality provide independently?
2. Does multimodal fusion outperform individual modalities?
3. Does adding temporal metadata improve the multimodal model?
4. Which modalities contribute the most to generalization?

No performance values are hard-coded into the README. All reported metrics originate from reproducible experiments.

---

# 7. Explainability

Predictions are accompanied by model explanations using **TreeSHAP**.

SHAP values identify which features contributed positively or negatively toward a prediction, allowing the system to move beyond:

> "This video is predicted to go viral."

toward:

> "The prediction was primarily influenced by these visual, acoustic, textual, and temporal characteristics."

This provides feature-level interpretability for both researchers and end users.

---

# System Architecture

```text
                Video + Caption + Schedule
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Visual          Audio           Text
       OpenCV         Librosa      S-BERT/TextBlob
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Feature Engineering
                         │
                         ▼
                  Multimodal Fusion
                         │
                         ▼
                     XGBoost
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Virality Tier           TreeSHAP
              │                     │
              ▼                     ▼
       Prediction API        Explanation Layer
              │                     │
              └──────────┬──────────┘
                         ▼
                  Interactive Dashboard
```

---

# Technology Stack

### Machine Learning

- **XGBoost** — Multiclass classification
- **Scikit-Learn** — preprocessing, evaluation, and experimentation
- **TreeSHAP** — model explainability

### Feature Extraction

- **OpenCV** — visual and motion features
- **Librosa** — acoustic and spectral features
- **Sentence-Transformers** — semantic text embeddings
- **TextBlob** — sentiment/polarity analysis

### API & Serving

- **FastAPI**
- **Pydantic v2**
- **Uvicorn**
- **Joblib**

### Frontend

- HTML5
- CSS
- Vanilla JavaScript (ES6+)
- OLED-inspired dark interface
- Neutral light mode

### Testing

- **Pytest**
- Unit tests
- Integration tests

---

# Project Structure

```text
multimodal-virality-predictor/
│
├── src/
│   ├── features/
│   │   ├── visual/
│   │   ├── audio/
│   │   └── text/
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── train_ablation.py
│   │   └── predict.py
│   │
│   └── api/
│
├── reports/
│   └── ablation_results.json
│
├── tests/
│
├── frontend/
│
├── run_app.py
├── requirements.txt
└── README.md
```

*Directory structure may evolve as the research pipeline develops.*

---

# Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sidhesh795/multimodal-virality-predictor.git
cd multimodal-virality-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the ablation experiments

```bash
python src/models/train_ablation.py
```

Results are written to:

```text
reports/ablation_results.json
```

### 4. Run the test suite

```bash
python -m pytest -v
```

### 5. Launch the dashboard

```bash
python run_app.py
```

---

# Reproducibility

The project is designed around reproducible experimentation.

Research results should be generated from the pipeline rather than manually entered into documentation.

Key experimental artifacts include:

- Dataset preprocessing configuration
- Feature extraction configuration
- Training configuration
- Temporal train/test partitions
- Virality threshold definitions
- Ablation results
- Evaluation metrics
- Model artifacts
- Explainability outputs

---

# Research Contributions

This project focuses on five primary contributions:

1. **Pre-publication virality prediction** using multimodal content characteristics.
2. **Reach-normalized virality formulation** rather than raw view-count prediction.
3. **Leakage-aware temporal evaluation** for realistic future-content prediction.
4. **Modality ablation analysis** to quantify the contribution of individual signals.
5. **Explainable predictions** using TreeSHAP rather than treating the model as a black box.

---

# Project Status

| Component | Status |
|---|---|
| Multimodal feature extraction | In Development |
| Virality target formulation | Implemented |
| Leakage prevention | Implemented |
| Temporal evaluation | Implemented / Under Validation |
| LOCO evaluation | In Development |
| Class-weighted training | Implemented |
| Ablation framework | Implemented |
| SHAP explainability | Implemented |
| FastAPI serving | Implemented |
| Interactive dashboard | Implemented |
| Real-world dataset integration | In Progress |

---

# Research Disclaimer

The system estimates **historical virality patterns from pre-publication characteristics**. It does not guarantee that a video will become viral.

Virality is influenced by platform algorithms, audience behavior, timing, trends, distribution mechanisms, and other external factors that cannot be fully observed from the content itself.

Therefore, model performance is evaluated empirically using held-out data rather than assumed from heuristic rules.

---

## Core Research Question

> **Can multimodal pre-publication signals provide a reliable and explainable estimate of social media video virality across time and content categories?**
