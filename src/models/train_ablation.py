"""
Modality ablation study for virality classification.

Trains XGBoost classifiers on the same chronological split for Metadata,
Text, Audio, Visual, and Multimodal Full Fusion feature groups. Uses
sklearn balanced sample weights instead of SMOTE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from backend.app.config import (
    AUDIO_EMBEDDING_FEATURES,
    AUDIO_SCALAR_FEATURES,
    BASE_DIR,
    FEATURE_COLUMNS,
    METADATA_FEATURES,
    RANDOM_SEED,
    SYNTHETIC_DATASET_PATH,
    TARGET_CATEGORY,
    TEXT_EMBEDDING_FEATURES,
    TEXT_SCALAR_FEATURES,
    VIRALITY_CATEGORIES,
    VISUAL_EMBEDDING_FEATURES,
    VISUAL_SCALAR_FEATURES,
)
from src.data.splitters import TemporalSplitter
from src.data.target_builder import compute_virality_v2, discretize_virality

REPORTS_DIR = BASE_DIR / "reports"
ABLATION_RESULTS_PATH = REPORTS_DIR / "ablation_results.json"

MODALITY_GROUPS: Dict[str, List[str]] = {
    "Metadata": list(METADATA_FEATURES),
    "Text": list(TEXT_SCALAR_FEATURES) + list(TEXT_EMBEDDING_FEATURES),
    "Audio": list(AUDIO_SCALAR_FEATURES) + list(AUDIO_EMBEDDING_FEATURES),
    "Visual": list(VISUAL_SCALAR_FEATURES) + list(VISUAL_EMBEDDING_FEATURES),
    "Multimodal Full Fusion": list(FEATURE_COLUMNS),
}


def _ensure_created_at(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a deterministic synthetic timestamp if the dataset has none."""
    if "created_at" in df.columns:
        out = df.copy()
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
        return out

    rng = np.random.RandomState(RANDOM_SEED)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    offsets = rng.randint(0, 365 * 24 * 3600, size=len(df))
    out = df.copy()
    out["created_at"] = start + pd.to_timedelta(offsets, unit="s")
    return out


def _resolve_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Prefer follower-normalized V2 labels when views/followers exist;
    otherwise encode the existing virality_category column.
    """
    if {"views", "followers"}.issubset(df.columns):
        v2 = compute_virality_v2(df["views"], df["followers"], k=1000)
        return discretize_virality(v2).to_numpy()

    if TARGET_CATEGORY not in df.columns:
        raise ValueError(
            "Dataset must include virality_category or views+followers columns."
        )

    encoder = LabelEncoder()
    encoder.fit(VIRALITY_CATEGORIES)
    return encoder.transform(df[TARGET_CATEGORY].astype(str))


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> Dict[str, Any]:
    precision = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    labels = np.arange(n_classes)
    try:
        roc_auc = float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        )
    except ValueError:
        roc_auc = float("nan")

    return {
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "roc_auc_ovr_macro": None if np.isnan(roc_auc) else round(roc_auc, 4),
    }


def train_ablation_models(
    dataset_path: Path | None = None,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    """Train one XGBoost classifier per modality group on a shared temporal split."""
    dataset_path = Path(dataset_path or SYNTHETIC_DATASET_PATH)
    output_path = Path(output_path or ABLATION_RESULTS_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run generate_dataset first."
        )

    df = _ensure_created_at(pd.read_csv(dataset_path))
    y = _resolve_labels(df)

    splitter = TemporalSplitter(df, timestamp_col="created_at", train_ratio=0.8)
    train_idx, test_idx = splitter.split()

    n_classes = int(max(len(VIRALITY_CATEGORIES), int(y.max()) + 1))
    results: Dict[str, Any] = {
        "split": {
            "strategy": "temporal",
            "timestamp_col": "created_at",
            "train_ratio": 0.8,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "sample_weight": "balanced",
        "modalities": {},
    }

    y_train = y[train_idx]
    y_test = y[test_idx]
    sample_weight = compute_sample_weight("balanced", y_train)

    for modality_name, columns in MODALITY_GROUPS.items():
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"{modality_name} is missing columns: {missing}")

        X_train = df.iloc[train_idx][columns]
        X_test = df.iloc[test_idx][columns]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            objective="multi:softprob",
            num_class=n_classes,
        )
        clf.fit(X_train_scaled, y_train, sample_weight=sample_weight)

        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)
        metrics = _classification_metrics(y_test, y_pred, y_proba, n_classes)
        metrics["n_features"] = len(columns)
        results["modalities"][modality_name] = metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    report = train_ablation_models()
    print(f"Ablation results saved to {ABLATION_RESULTS_PATH}")
    for name, metrics in report["modalities"].items():
        print(
            f"{name}: F1-macro={metrics['f1_macro']:.4f} "
            f"ROC-AUC(OVR)={metrics['roc_auc_ovr_macro']}"
        )
