"""
Global Configuration for Multimodal Social Media Virality Prediction System.
Defines paths, feature column definitions, constants, and seeds.
"""

from pathlib import Path

# Base & Directory Paths
# backend/app/config.py -> parent x 3 = Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
SYNTHETIC_DATASET_PATH = DATA_RAW_DIR / "synthetic_virality_dataset.csv"

# Reproducibility
RANDOM_SEED = 42

# Feature Definitions
METADATA_FEATURES = [
    "duration_sec",
    "upload_hour_sin",
    "upload_hour_cos",
    "is_weekend",
    "category_id",
]

VISUAL_SCALAR_FEATURES = [
    "scene_cut_count",
    "avg_brightness",
    "color_variance",
    "motion_score",
]
VISUAL_EMBEDDING_FEATURES = [f"visual_emb_{i}" for i in range(16)]

AUDIO_SCALAR_FEATURES = [
    "rms_energy",
    "tempo_bpm",
    "zero_crossing_rate",
    "spectral_centroid",
]
AUDIO_EMBEDDING_FEATURES = [f"audio_emb_{i}" for i in range(8)]

TEXT_SCALAR_FEATURES = [
    "caption_length",
    "hashtag_count",
    "sentiment_polarity",
    "sentiment_subjectivity",
]
TEXT_EMBEDDING_FEATURES = [f"text_emb_{i}" for i in range(16)]

# Combined Feature List
ALL_SCALAR_FEATURES = (
    METADATA_FEATURES
    + VISUAL_SCALAR_FEATURES
    + AUDIO_SCALAR_FEATURES
    + TEXT_SCALAR_FEATURES
)

ALL_EMBEDDING_FEATURES = (
    VISUAL_EMBEDDING_FEATURES
    + AUDIO_EMBEDDING_FEATURES
    + TEXT_EMBEDDING_FEATURES
)

FEATURE_COLUMNS = ALL_SCALAR_FEATURES + ALL_EMBEDDING_FEATURES

# Target Definitions
TARGET_NER = "ner"
TARGET_CONTINUOUS = "virality_score"
TARGET_CATEGORY = "virality_category"

VIRALITY_CATEGORIES = ["Low", "Medium", "High", "Viral"]
