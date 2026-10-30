"""
Unit Tests for Model Architecture, Training, and Prediction Engine.
Tests model serialization, evaluation metrics comparison, and inference predictor.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from backend.app.config import MODELS_DIR, FEATURE_COLUMNS, VIRALITY_CATEGORIES
from backend.app.schemas import PredictionResult
from src.features.pipeline import MultimodalFeaturePipeline
from src.models.train import train_models
from src.models.predict import ViralityPredictor


@pytest.fixture(scope="module", autouse=True)
def ensure_models_trained():
    """Ensures models are trained and artifacts exist before running tests."""
    clf_path = MODELS_DIR / "virality_classifier.joblib"
    if not clf_path.exists():
        train_models()


def test_model_artifacts_exist():
    """Verify that all model artifacts and evaluation reports are saved in models/."""
    expected_files = [
        "virality_classifier.joblib",
        "virality_regressor.joblib",
        "scaler.joblib",
        "label_encoder.joblib",
        "baseline_classifier.joblib",
        "evaluation_report.json",
    ]
    for fname in expected_files:
        fpath = MODELS_DIR / fname
        assert fpath.exists(), f"Expected artifact {fname} missing in {MODELS_DIR}"
        assert fpath.stat().st_size > 0, f"Artifact {fname} is empty"


def test_evaluation_report_metrics():
    """Verify evaluation report content and ensure Multimodal model outperforms Baseline."""
    report_path = MODELS_DIR / "evaluation_report.json"
    assert report_path.exists(), "evaluation_report.json missing"

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert "multimodal_classification" in report
    assert "baseline_classification" in report
    assert "virality_regression" in report
    assert "fusion_uplift" in report

    # 1. Classification Metrics Checks
    multimodal_f1 = report["multimodal_classification"]["f1_macro"]
    baseline_f1 = report["baseline_classification"]["f1_macro"]
    assert multimodal_f1 > baseline_f1, (
        f"Expected Multimodal F1 ({multimodal_f1}) to outperform Baseline F1 ({baseline_f1})"
    )
    assert report["fusion_uplift"]["multimodal_outperforms_baseline"] is True

    # 2. Regression Metrics Checks
    reg_metrics = report["virality_regression"]
    assert reg_metrics["r2_score"] > 0.4, f"Low R2 score: {reg_metrics['r2_score']}"
    assert reg_metrics["mae"] < 20.0, f"High MAE: {reg_metrics['mae']}"


def test_virality_predictor_inference():
    """Test ViralityPredictor inference on pipeline output and validate against Pydantic schema."""
    predictor = ViralityPredictor()
    pipeline = MultimodalFeaturePipeline()

    # Generate a realistic sample feature row via MultimodalFeaturePipeline
    caption = "Viral dance challenge taking over! #dance #trending #viral"
    metadata = {
        "duration_sec": 18.0,
        "upload_hour": 20,
        "is_weekend": 1,
        "category_id": 1,
    }
    sample_df = pipeline.extract_all(video_path=None, caption=caption, metadata=metadata)

    assert sample_df.shape == (1, 57)
    assert list(sample_df.columns) == FEATURE_COLUMNS

    # Run inference
    pred_dict = predictor.predict(sample_df)

    assert isinstance(pred_dict, dict)
    assert "virality_score" in pred_dict
    assert "predicted_category" in pred_dict
    assert "probabilities" in pred_dict
    assert 0.0 <= pred_dict["confidence"] <= 1.0
    assert pred_dict["uncertainty_band"] in {"low", "medium", "high"}

    # Check bounds and types
    assert 0.0 <= pred_dict["virality_score"] <= 100.0
    assert pred_dict["predicted_category"] in VIRALITY_CATEGORIES

    probs = pred_dict["probabilities"]
    assert set(probs.keys()) == set(VIRALITY_CATEGORIES)
    prob_sum = sum(probs.values())
    assert 0.99 <= prob_sum <= 1.01, f"Probabilities do not sum to 1.0: {prob_sum}"

    # Validate against Pydantic PredictionResult schema
    validated_model = predictor.predict_result(sample_df)
    assert isinstance(validated_model, PredictionResult)
    assert validated_model.predicted_category == pred_dict["predicted_category"]
    assert validated_model.virality_score == pred_dict["virality_score"]
