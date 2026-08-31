"""
Unit Tests for Multimodal Social Media Virality Prediction Data Pipeline.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from backend.app.config import SYNTHETIC_DATASET_PATH, VIRALITY_CATEGORIES
from backend.app.schemas import FeatureVector, ContentInput, PredictionResult


@pytest.fixture(scope="module")
def dataset_df():
    """Fixture that loads the synthetic dataset CSV file."""
    dataset_path = Path(SYNTHETIC_DATASET_PATH)
    assert dataset_path.exists(), f"Dataset file does not exist at {dataset_path}"
    df = pd.read_csv(dataset_path)
    return df


def test_dataset_file_exists_and_non_empty():
    """Verify that synthetic dataset CSV file exists and is non-empty."""
    dataset_path = Path(SYNTHETIC_DATASET_PATH)
    assert dataset_path.exists(), "Synthetic dataset file missing"
    assert dataset_path.stat().st_size > 0, "Synthetic dataset file is empty"


def test_dataset_shape_and_no_nan(dataset_df):
    """Verify dataset has exactly 1,000 rows and zero NaN values across all columns."""
    assert dataset_df.shape[0] == 1000, f"Expected 1000 rows, got {dataset_df.shape[0]}"
    # Verify expected column count: 17 scalar + 16 visual_emb + 8 audio_emb + 16 text_emb + 3 targets = 60 cols
    assert dataset_df.shape[1] >= 57, f"Unexpected column count: {dataset_df.shape[1]}"
    assert dataset_df.isna().sum().sum() == 0, "Dataset contains NaN values"


def test_virality_category_classes(dataset_df):
    """Verify that virality_category column contains all 4 expected classes."""
    assert "virality_category" in dataset_df.columns
    unique_categories = set(dataset_df["virality_category"].unique())
    expected_categories = set(VIRALITY_CATEGORIES)
    assert unique_categories == expected_categories, (
        f"Expected categories {expected_categories}, got {unique_categories}"
    )


def test_pydantic_schema_validation(dataset_df):
    """Verify that sample dataset rows validate against the Pydantic FeatureVector schema."""
    sample_row = dataset_df.iloc[0]

    # Extract visual, audio, text embeddings
    visual_emb = [float(sample_row[f"visual_emb_{i}"]) for i in range(16)]
    audio_emb = [float(sample_row[f"audio_emb_{i}"]) for i in range(8)]
    text_emb = [float(sample_row[f"text_emb_{i}"]) for i in range(16)]

    feature_vector = FeatureVector(
        duration_sec=float(sample_row["duration_sec"]),
        upload_hour_sin=float(sample_row["upload_hour_sin"]),
        upload_hour_cos=float(sample_row["upload_hour_cos"]),
        is_weekend=int(sample_row["is_weekend"]),
        category_id=int(sample_row["category_id"]),
        scene_cut_count=int(sample_row["scene_cut_count"]),
        avg_brightness=float(sample_row["avg_brightness"]),
        color_variance=float(sample_row["color_variance"]),
        motion_score=float(sample_row["motion_score"]),
        visual_embedding=visual_emb,
        rms_energy=float(sample_row["rms_energy"]),
        tempo_bpm=float(sample_row["tempo_bpm"]),
        zero_crossing_rate=float(sample_row["zero_crossing_rate"]),
        spectral_centroid=float(sample_row["spectral_centroid"]),
        audio_embedding=audio_emb,
        caption_length=int(sample_row["caption_length"]),
        hashtag_count=int(sample_row["hashtag_count"]),
        sentiment_polarity=float(sample_row["sentiment_polarity"]),
        sentiment_subjectivity=float(sample_row["sentiment_subjectivity"]),
        text_embedding=text_emb,
    )

    assert feature_vector.duration_sec == float(sample_row["duration_sec"])
    assert len(feature_vector.visual_embedding) == 16
    assert len(feature_vector.audio_embedding) == 8
    assert len(feature_vector.text_embedding) == 16

    # Test flat dict conversion
    flat_dict = feature_vector.to_flat_dict()
    assert len(flat_dict) == 57


def test_other_schemas():
    """Verify ContentInput and PredictionResult Pydantic schemas."""
    input_data = ContentInput(
        caption="Check out this awesome video! #viral #trending",
        duration_sec=15.5,
        upload_hour=18,
        is_weekend=1,
        category_id=2,
    )
    assert input_data.duration_sec == 15.5

    result = PredictionResult(
        virality_score=85.5,
        predicted_category="Viral",
        probabilities={"Low": 0.05, "Medium": 0.10, "High": 0.15, "Viral": 0.70},
        explainability_breakdown={"motion_score": 0.35, "tempo_bpm": 0.25},
    )
    assert result.virality_score == 85.5
    assert result.predicted_category == "Viral"
