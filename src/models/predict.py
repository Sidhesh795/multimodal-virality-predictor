"""
Inference Engine and Virality Predictor.
Loads trained model artifacts and generates classification probabilities,
predicted categories, continuous virality scores, SHAP explanations,
and prioritized optimization recommendations.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import os
import joblib
import numpy as np
import pandas as pd

from backend.app.config import MODELS_DIR, FEATURE_COLUMNS, VIRALITY_CATEGORIES
from backend.app.schemas import PredictionResult
from src.explainability.shap_engine import ViralityExplainer
from src.explainability.recommendations import RecommendationEngine


class ViralityPredictor:
    """
    Multimodal Virality Prediction & Explainability Inference Service.
    Loads classifier, regressor, scaler, explainer, and recommendation engine.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.label_encoder = None
        self.explainer = None
        self.recommendation_engine = None
        self.enable_shap = os.getenv("VIRALITY_ENABLE_SHAP", "1") != "0"
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized XGBoost models, scaler, label encoder, and explainers."""
        clf_path = self.models_dir / "virality_classifier.joblib"
        reg_path = self.models_dir / "virality_regressor.joblib"
        scaler_path = self.models_dir / "scaler.joblib"
        encoder_path = self.models_dir / "label_encoder.joblib"

        if not clf_path.exists() or not reg_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"Model artifacts missing in {self.models_dir}. Please run train.py first."
            )

        self.classifier = joblib.load(clf_path)
        self.regressor = joblib.load(reg_path)
        self.scaler = joblib.load(scaler_path)

        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
        else:
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(VIRALITY_CATEGORIES)

        if self.enable_shap:
            self.explainer = ViralityExplainer(models_dir=self.models_dir)
        self.recommendation_engine = RecommendationEngine()

    def _fast_explanation(self, feature_df: pd.DataFrame, predicted_score: float) -> Dict[str, Any]:
        """Return a low-resource, model-based attribution for hosted inference.

        This fallback uses scaled feature values weighted by the XGBoost model's
        feature importance. It is intentionally labelled as a fallback rather
        than presented as a TreeSHAP explanation.
        """
        X = feature_df[FEATURE_COLUMNS].copy()
        scaled = self.scaler.transform(X)[0]
        importances = np.asarray(self.regressor.feature_importances_, dtype=float)
        contributions = scaled * importances
        items = []
        for idx, name in enumerate(FEATURE_COLUMNS):
            items.append({
                "feature": name,
                "shap_value": round(float(contributions[idx]), 4),
                "raw_value": round(float(X.iloc[0][name]), 4),
            })
        ordered = sorted(items, key=lambda item: item["shap_value"], reverse=True)
        negatives = [item for item in reversed(ordered) if item["shap_value"] < 0][:5]
        positives = [item for item in ordered if item["shap_value"] > 0][:5]
        impacts = sorted(items, key=lambda item: abs(item["shap_value"]), reverse=True)
        return {
            "explanation_method": "xgboost_feature_importance_fallback",
            "base_value": 0.0,
            "predicted_score": round(float(predicted_score), 2),
            "top_positive_features": positives,
            "top_negative_features": negatives,
            "top_impacts": {item["feature"]: item["shap_value"] for item in impacts[:10]},
            "feature_impacts": {item["feature"]: item["shap_value"] for item in items},
            "modality_contributions": {},
            "modality_strength": {},
        }

    def predict(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs inference and explainability on a multimodal feature DataFrame.

        Parameters:
            feature_df (pd.DataFrame): 1-row or N-row DataFrame with FEATURE_COLUMNS.

        Returns:
            Dict[str, Any]: Prediction result matching PredictionResult schema.
        """
        # Ensure all columns exist in expected order
        X = feature_df[FEATURE_COLUMNS].copy()
        X_scaled = self.scaler.transform(X)

        # 1. Classification & Class Probabilities
        cat_indices = self.classifier.predict(X_scaled)
        cat_probs = self.classifier.predict_proba(X_scaled)

        first_cat_idx = int(cat_indices[0])
        predicted_category = str(self.label_encoder.inverse_transform([first_cat_idx])[0])

        classes = list(self.label_encoder.classes_)
        probabilities = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(classes, cat_probs[0])
        }
        confidence = float(np.max(cat_probs[0]))
        if confidence >= 0.75:
            uncertainty_band = "high"
        elif confidence >= 0.50:
            uncertainty_band = "medium"
        else:
            uncertainty_band = "low"

        # 2. Continuous Score Regression
        raw_scores = self.regressor.predict(X_scaled)
        virality_score = float(np.clip(raw_scores[0], 0.0, 100.0))

        # 3. SHAP Explainability Decomposition
        if self.enable_shap:
            shap_analysis = self.explainer.explain_instance(feature_df)
        else:
            shap_analysis = self._fast_explanation(feature_df, virality_score)
        explainability_breakdown = shap_analysis.get("top_impacts", {})

        # 4. Actionable Recommendation Generation
        recommendations = self.recommendation_engine.generate_recommendations(
            shap_analysis, feature_df
        )

        return {
            "virality_score": round(virality_score, 2),
            "predicted_category": predicted_category,
            "probabilities": probabilities,
            "confidence": round(confidence, 4),
            "uncertainty_band": uncertainty_band,
            "explainability_breakdown": explainability_breakdown,
            "shap_explanation": shap_analysis,
            "recommendations": recommendations,
        }

    def predict_result(self, feature_df: pd.DataFrame) -> PredictionResult:
        """Runs inference and returns a validated Pydantic PredictionResult model."""
        res_dict = self.predict(feature_df)
        return PredictionResult(**res_dict)

