"""
FastAPI Routes for Virality Prediction API.
Defines endpoints for JSON prediction, video file upload prediction, and health checks.
"""

from pathlib import Path
import json
import os
import tempfile
import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends, status

from backend.app.config import BASE_DIR, DATA_RAW_DIR, REPORTS_DIR
from backend.app.schemas import ContentInput, PredictionResult
from src.features.pipeline import MultimodalFeaturePipeline
from src.models.predict import ViralityPredictor

router = APIRouter()

# Global predictor & pipeline singletons (lazy loaded or injected via lifespan/request)
_predictor: Optional[ViralityPredictor] = None
_pipeline: Optional[MultimodalFeaturePipeline] = None


def get_predictor() -> ViralityPredictor:
    """Dependency provider for ViralityPredictor."""
    global _predictor
    if _predictor is None:
        _predictor = ViralityPredictor()
    return _predictor


def get_pipeline() -> MultimodalFeaturePipeline:
    """Dependency provider for MultimodalFeaturePipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MultimodalFeaturePipeline()
    return _pipeline


@router.get(
    "/health",
    summary="API Health Check",
    response_model=Dict[str, Any],
)
def health_check(predictor: ViralityPredictor = Depends(get_predictor)):
    """Returns status, service metadata, and model loading state."""
    model_healthy = (
        predictor.classifier is not None and predictor.regressor is not None
    )
    return {
        "status": "healthy" if model_healthy else "degraded",
        "service": "Multimodal Social Media Virality Prediction API",
        "version": "1.0.0",
        "model_loaded": model_healthy,
        "feature_count": len(predictor.classifier.feature_importances_)
        if model_healthy
        else 0,
        "capabilities": [
            "multimodal_feature_extraction",
            "virality_tier_classification",
            "continuous_score_regression",
            "tree_shap_explainability"
            if os.getenv("VIRALITY_ENABLE_SHAP", "1") != "0"
            else "low_resource_feature_attribution",
            "confidence_estimation",
        ],
    }


@router.get(
    "/metrics/ablation",
    summary="Modality Ablation Benchmark Metrics",
    response_model=Dict[str, Any],
)
def get_ablation_metrics():
    """Returns saved ablation study metrics for dashboard charts."""
    report_path = REPORTS_DIR / "ablation_results.json"
    if not report_path.exists():
        report_path = BASE_DIR / "reports" / "ablation_results.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ablation results not found. Run src.models.train_ablation first.",
        )
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post(
    "/predict/json",
    summary="Predict Virality from JSON Payload",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
)
def predict_from_json(
    payload: ContentInput,
    predictor: ViralityPredictor = Depends(get_predictor),
    pipeline: MultimodalFeaturePipeline = Depends(get_pipeline),
):
    """
    Ingests short-form content metadata and caption text, extracts features,
    and returns virality scores, predicted tiers, SHAP analysis, and recommendations.
    """
    try:
        metadata = {
            "duration_sec": payload.duration_sec,
            "upload_hour": payload.upload_hour,
            "is_weekend": payload.is_weekend,
            "category_id": payload.category_id,
        }

        # Extract features
        feature_df = pipeline.extract_all(
            video_path=payload.media_file_path,
            caption=payload.caption,
            metadata=metadata,
        )

        # Run inference
        result = predictor.predict(feature_df)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/predict/video",
    summary="Predict Virality from Uploaded Video",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
)
async def predict_from_video(
    caption: str = Form(..., description="Post caption text"),
    duration_sec: float = Form(15.0, ge=1.0, le=300.0, description="Video duration in seconds"),
    upload_hour: int = Form(12, ge=0, le=23, description="Upload hour (0-23)"),
    is_weekend: int = Form(0, ge=0, le=1, description="1 if weekend, else 0"),
    category_id: int = Form(0, ge=0, description="Content category ID"),
    file: UploadFile = File(..., description="Uploaded video file (.mp4, .mov, .avi, etc.)"),
    predictor: ViralityPredictor = Depends(get_predictor),
    pipeline: MultimodalFeaturePipeline = Depends(get_pipeline),
):
    """
    Ingests an uploaded short-form video file and metadata, runs multimodal feature extraction
    (OpenCV visual analysis, audio processing, NLP caption analysis), and returns prediction results.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    # Check file extension
    ext = Path(file.filename).suffix.lower()
    allowed_exts = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {allowed_exts}",
        )

    # Prepare temp directory
    temp_dir = DATA_RAW_DIR / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / f"upload_{uuid.uuid4().hex}{ext}"

    try:
        # Save uploaded file
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        with open(temp_file_path, "wb") as f:
            f.write(contents)

        metadata = {
            "duration_sec": duration_sec,
            "upload_hour": upload_hour,
            "is_weekend": is_weekend,
            "category_id": category_id,
        }

        # Multimodal feature extraction
        feature_df = pipeline.extract_all(
            video_path=str(temp_file_path),
            caption=caption,
            metadata=metadata,
        )

        # Run inference
        result = predictor.predict(feature_df)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing video: {str(e)}",
        )
    finally:
        # Ensure temp file cleanup
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass
