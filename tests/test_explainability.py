"""
Unit Tests for Model Explainability (SHAP) and Recommendation Engine.
Tests ViralityExplainer, RecommendationEngine, and integrated ViralityPredictor.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from backend.app.config import MODELS_DIR, FEATURE_COLUMNS, VIRALITY_CATEGORIES
from backend.app.schemas import PredictionResult
from src.features.pipeline import MultimodalFeaturePipeline
from src.models.train import train_models
from src.models.predict import ViralityPredictor
from src.explainability.shap_engine import ViralityExplainer
from src.explainability.recommendations import RecommendationEngine


@pytest.fixture(scope="module", autouse=True)
def ensure_models_trained():
    """Ensures models and explainers are ready."""
    clf_path = MODELS_DIR / "virality_classifier.joblib"
    if not clf_path.exists():
        train_models()


@pytest.fixture(scope="module")
def sample_feature_df():
    """Generates a sample multimodal feature DataFrame."""
    pipeline = MultimodalFeaturePipeline()
    caption = "A quiet vlog #lifestyle #daily"
    metadata = {
        "duration_sec": 45.0,
        "upload_hour": 3,
        "is_weekend": 0,
        "category_id": 0,
    }
    return pipeline.extract_all(video_path=None, caption=caption, metadata=metadata)


def test_virality_explainer(sample_feature_df):
    """Test SHAP tree explainer feature attribution calculation."""
    explainer = ViralityExplainer()
    shap_res = explainer.explain_instance(sample_feature_df)

    assert isinstance(shap_res, dict)
    assert "base_value" in shap_res
    assert "predicted_score" in shap_res
    assert "top_positive_features" in shap_res
    assert "top_negative_features" in shap_res
    assert "feature_impacts" in shap_res

    assert isinstance(shap_res["base_value"], float)
    assert isinstance(shap_res["predicted_score"], float)
    assert len(shap_res["feature_impacts"]) == 57

    # Check top positive drivers
    for item in shap_res["top_positive_features"]:
        assert "feature" in item
        assert "shap_value" in item
        assert "raw_value" in item
        assert item["shap_value"] > 0

    # Check top negative drivers
    for item in shap_res["top_negative_features"]:
        assert "feature" in item
        assert "shap_value" in item
        assert "raw_value" in item
        assert item["shap_value"] < 0


def test_recommendation_engine(sample_feature_df):
    """Test recommendation generation and priority assignment from negative SHAP drivers."""
    engine = RecommendationEngine()

    mock_shap_analysis = {
        "top_negative_features": [
            {"feature": "scene_cut_count", "shap_value": -2.45, "raw_value": 0.0},
            {"feature": "avg_brightness", "shap_value": -1.80, "raw_value": 128.0},
            {"feature": "motion_score", "shap_value": -0.65, "raw_value": 0.0},
            {"feature": "hashtag_count", "shap_value": -0.30, "raw_value": 1.0},
        ]
    }

    recommendations = engine.generate_recommendations(mock_shap_analysis, sample_feature_df)

    assert isinstance(recommendations, list)
    assert len(recommendations) >= 4

    for rec in recommendations:
        assert "feature" in rec
        assert "shap_value" in rec
        assert "suggestion" in rec
        assert "priority" in rec
        assert rec["priority"] in ["HIGH", "MEDIUM", "LOW"]
        assert len(rec["suggestion"]) > 10

    # Top drivers must have HIGH priority
    assert recommendations[0]["priority"] == "HIGH"
    assert recommendations[1]["priority"] == "HIGH"


def test_end_to_end_prediction_with_explainability(sample_feature_df):
    """Test full inference pipeline with integrated SHAP explanations and recommendations."""
    predictor = ViralityPredictor()
    pred_dict = predictor.predict(sample_feature_df)

    assert "virality_score" in pred_dict
    assert "predicted_category" in pred_dict
    assert "probabilities" in pred_dict
    assert "explainability_breakdown" in pred_dict
    assert "shap_explanation" in pred_dict
    assert "recommendations" in pred_dict

    # Check that explainability breakdown and recommendations are populated
    assert len(pred_dict["explainability_breakdown"]) > 0
    assert len(pred_dict["recommendations"]) > 0

    # Validate against Pydantic PredictionResult schema
    validated_model = predictor.predict_result(sample_feature_df)
    assert isinstance(validated_model, PredictionResult)
    assert validated_model.virality_score == pred_dict["virality_score"]
    assert validated_model.predicted_category == pred_dict["predicted_category"]
    assert len(validated_model.recommendations) > 0
