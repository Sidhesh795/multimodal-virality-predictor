"""
SHAP Explainability Engine for Virality Prediction.
Computes local feature attributions and isolates positive and negative virality drivers.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
import shap

from backend.app.config import (
    MODELS_DIR,
    FEATURE_COLUMNS,
    METADATA_FEATURES,
    VISUAL_SCALAR_FEATURES,
    VISUAL_EMBEDDING_FEATURES,
    AUDIO_SCALAR_FEATURES,
    AUDIO_EMBEDDING_FEATURES,
    TEXT_SCALAR_FEATURES,
    TEXT_EMBEDDING_FEATURES,
)


class ViralityExplainer:
    """
    SHAP-based TreeExplainer for analyzing multimodal feature attributions
    on virality predictions.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.regressor = None
        self.scaler = None
        self.explainer = None
        self._load_explainer()

    def _load_explainer(self):
        """Loads serialized regressor and scaler and builds TreeExplainer."""
        reg_path = self.models_dir / "virality_regressor.joblib"
        scaler_path = self.models_dir / "scaler.joblib"

        if not reg_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"Model artifacts missing in {self.models_dir}. Please run train.py first."
            )

        self.regressor = joblib.load(reg_path)
        self.scaler = joblib.load(scaler_path)
        self.explainer = shap.TreeExplainer(self.regressor)

    def explain_instance(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes SHAP feature importance attributions for a single content record.

        Parameters:
            feature_df (pd.DataFrame): 1-row DataFrame containing all FEATURE_COLUMNS.

        Returns:
            Dict[str, Any]: Detailed SHAP breakdown including top positive and negative drivers.
        """
        X = feature_df[FEATURE_COLUMNS].copy()
        X_scaled = self.scaler.transform(X)

        raw_shap = self.explainer.shap_values(X_scaled)
        if isinstance(raw_shap, list):
            sample_shap = raw_shap[0][0]
        elif len(raw_shap.shape) == 2:
            sample_shap = raw_shap[0]
        else:
            sample_shap = raw_shap

        # Base value
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_value = float(expected_val[0])
        else:
            base_value = float(expected_val)

        # Raw predicted score
        predicted_score = float(np.clip(base_value + np.sum(sample_shap), 0.0, 100.0))

        # Collect feature attributions
        features_list = []
        feature_impacts_dict = {}

        for idx, col in enumerate(FEATURE_COLUMNS):
            raw_val = float(feature_df[col].iloc[0])
            s_val = float(sample_shap[idx])
            feature_impacts_dict[col] = round(s_val, 4)

            features_list.append({
                "feature": col,
                "shap_value": round(s_val, 4),
                "raw_value": round(raw_val, 4),
            })

        # Rank positive and negative drivers
        sorted_by_shap = sorted(features_list, key=lambda x: x["shap_value"], reverse=True)
        top_positive = [f for f in sorted_by_shap if f["shap_value"] > 0][:5]
        top_negative = [f for f in reversed(sorted_by_shap) if f["shap_value"] < 0][:5]

        # Top overall absolute impacts
        sorted_by_abs = sorted(features_list, key=lambda x: abs(x["shap_value"]), reverse=True)
        top_impacts = {f["feature"]: f["shap_value"] for f in sorted_by_abs[:10]}

        modality_groups = {
            "metadata": METADATA_FEATURES,
            "visual": VISUAL_SCALAR_FEATURES + VISUAL_EMBEDDING_FEATURES,
            "audio": AUDIO_SCALAR_FEATURES + AUDIO_EMBEDDING_FEATURES,
            "text": TEXT_SCALAR_FEATURES + TEXT_EMBEDDING_FEATURES,
        }
        modality_contributions = {
            modality: round(
                float(sum(feature_impacts_dict.get(name, 0.0) for name in names)),
                4,
            )
            for modality, names in modality_groups.items()
        }
        modality_strength = {
            modality: round(abs(value), 4)
            for modality, value in modality_contributions.items()
        }

        return {
            "base_value": round(base_value, 2),
            "predicted_score": round(predicted_score, 2),
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "top_impacts": top_impacts,
            "feature_impacts": feature_impacts_dict,
            "modality_contributions": modality_contributions,
            "modality_strength": modality_strength,
        }
