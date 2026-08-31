"""
Model Evaluation and Metrics Reporting for Virality Prediction.
Calculates classification metrics, regression metrics, and fusion uplift.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "confusion_matrix": cm,
        "class_names": class_names or [],
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Computes continuous regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
    }


def save_evaluation_report(
    multimodal_classification: Dict[str, Any],
    baseline_classification: Dict[str, Any],
    virality_regression: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Combines metrics, computes multimodal fusion uplift over baseline,
    and saves report to JSON file.
    """
    f1_multimodal = multimodal_classification["f1_macro"]
    f1_baseline = baseline_classification["f1_macro"]
    f1_uplift_abs = round(f1_multimodal - f1_baseline, 4)
    f1_uplift_pct = round(
        ((f1_multimodal - f1_baseline) / max(f1_baseline, 1e-6)) * 100.0, 2
    )

    report = {
        "multimodal_classification": multimodal_classification,
        "baseline_classification": baseline_classification,
        "virality_regression": virality_regression,
        "fusion_uplift": {
            "f1_macro_multimodal": f1_multimodal,
            "f1_macro_baseline": f1_baseline,
            "f1_macro_uplift_absolute": f1_uplift_abs,
            "f1_macro_uplift_percent": f1_uplift_pct,
            "multimodal_outperforms_baseline": f1_multimodal > f1_baseline,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
