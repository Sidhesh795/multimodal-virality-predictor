"""
Model Training and Multimodal Fusion Engine.
Trains XGBoost classification and regression models and evaluates fusion uplift.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from backend.app.config import (
    SYNTHETIC_DATASET_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    FEATURE_COLUMNS,
    METADATA_FEATURES,
    TARGET_CATEGORY,
    TARGET_CONTINUOUS,
    VIRALITY_CATEGORIES,
)
from src.models.evaluate import (
    evaluate_classification,
    evaluate_regression,
    save_evaluation_report,
)
from src.data.splitters import TemporalSplitter


def train_models():
    """
    Executes full training pipeline:
    1. Loads synthetic dataset.
    2. Performs a chronological 80/20 train/test split.
    3. Fits StandardScaler on feature columns.
    4. Trains Multimodal XGBClassifier and XGBRegressor.
    5. Trains Metadata Baseline XGBClassifier.
    6. Serializes model artifacts and evaluation report.
    """
    print(f"Loading dataset from {SYNTHETIC_DATASET_PATH}...")
    dataset_path = Path(SYNTHETIC_DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run generate_dataset first.")

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")

    # 1. Prepare Features and Targets
    X = df[FEATURE_COLUMNS].copy()
    y_category = df[TARGET_CATEGORY].values
    y_score = df[TARGET_CONTINUOUS].values

    # Encode categorical target to 0, 1, 2, 3 aligned with VIRALITY_CATEGORIES
    label_encoder = LabelEncoder()
    label_encoder.fit(VIRALITY_CATEGORIES)
    y_category_encoded = label_encoder.transform(y_category)

    # 2. Chronological 80/20 split. The generated dataset does not carry
    # platform timestamps, so row order is treated as the deterministic
    # collection order for this offline benchmark.
    if "created_at" not in df.columns:
        df = df.copy()
        df["created_at"] = pd.date_range(
            start="2024-01-01", periods=len(df), freq="h", tz="UTC"
        )
    splitter = TemporalSplitter(df, timestamp_col="created_at", train_ratio=0.8)
    train_idx, test_idx = splitter.split()
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_cat_train = y_category_encoded[train_idx]
    y_cat_test = y_category_encoded[test_idx]
    y_score_train = y_score[train_idx]
    y_score_test = y_score[test_idx]
    sample_weight = compute_sample_weight("balanced", y_cat_train)

    print(f"Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

    # 3. Fit StandardScaler on Continuous Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Multimodal XGBClassifier
    print("Training Multimodal XGBClassifier...")
    clf = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train_scaled, y_cat_train, sample_weight=sample_weight)
    y_cat_pred = clf.predict(X_test_scaled)
    multimodal_cls_metrics = evaluate_classification(
        y_cat_test, y_cat_pred, class_names=list(label_encoder.classes_)
    )
    print(f"Multimodal Classification F1-Macro: {multimodal_cls_metrics['f1_macro']:.4f}")
    print(f"Multimodal Classification Accuracy: {multimodal_cls_metrics['accuracy']:.4f}")

    # 5. Train Multimodal XGBRegressor
    print("Training Multimodal XGBRegressor...")
    reg = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
    )
    reg.fit(X_train_scaled, y_score_train)
    y_score_pred = reg.predict(X_test_scaled)
    # Clip predictions to [0, 100]
    y_score_pred = np.clip(y_score_pred, 0.0, 100.0)
    virality_reg_metrics = evaluate_regression(y_score_test, y_score_pred)
    print(f"Virality Score Regressor MAE: {virality_reg_metrics['mae']:.4f}")
    print(f"Virality Score Regressor R2:  {virality_reg_metrics['r2_score']:.4f}")

    # 6. Train Baseline Model (Metadata Features Only)
    print("Training Metadata Baseline Model...")
    baseline_scaler = StandardScaler()
    X_train_base = baseline_scaler.fit_transform(X_train[METADATA_FEATURES])
    X_test_base = baseline_scaler.transform(X_test[METADATA_FEATURES])

    baseline_clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.08,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    baseline_clf.fit(X_train_base, y_cat_train, sample_weight=sample_weight)
    y_base_pred = baseline_clf.predict(X_test_base)
    baseline_cls_metrics = evaluate_classification(
        y_cat_test, y_base_pred, class_names=list(label_encoder.classes_)
    )
    print(f"Baseline Classification F1-Macro: {baseline_cls_metrics['f1_macro']:.4f}")

    # 7. Save Evaluation Report
    report_path = MODELS_DIR / "evaluation_report.json"
    report = save_evaluation_report(
        multimodal_cls_metrics,
        baseline_cls_metrics,
        virality_reg_metrics,
        report_path,
    )
    report["evaluation_protocol"] = {
        "split_strategy": "temporal",
        "train_ratio": 0.8,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "class_weighting": "balanced_sample_weight",
        "random_seed": RANDOM_SEED,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved to {report_path}")
    print(f"Multimodal F1 Uplift: +{report['fusion_uplift']['f1_macro_uplift_percent']}% over Baseline")

    # 8. Serialize Model Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf_path = MODELS_DIR / "virality_classifier.joblib"
    reg_path = MODELS_DIR / "virality_regressor.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    encoder_path = MODELS_DIR / "label_encoder.joblib"
    baseline_path = MODELS_DIR / "baseline_classifier.joblib"

    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(baseline_clf, baseline_path)

    print(f"Models and scalers serialized successfully to {MODELS_DIR}")
    return report


if __name__ == "__main__":
    train_models()
