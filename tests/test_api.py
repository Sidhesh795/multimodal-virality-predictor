"""
Unit and Integration Tests for FastAPI Backend Services & API Endpoints.
Tests GET /health, POST /predict/json, POST /predict/video, and error handling.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import DATA_RAW_DIR, VIRALITY_CATEGORIES
from backend.app.schemas import PredictionResult
from tests.test_feature_extractors import create_dummy_video


@pytest.fixture(scope="module")
def client():
    """Fixture providing FastAPI TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def dummy_video_file():
    """Fixture providing a temporary 2-second MP4 video file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    create_dummy_video(tmp_path, duration_sec=2, fps=30)
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


def test_health_endpoint(client):
    """Test GET /api/v1/health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "service" in data
    assert "version" in data


def test_predict_json_endpoint(client):
    """Test POST /api/v1/predict/json with valid ContentInput payload."""
    payload = {
        "caption": "Check out this amazing short-form video! #viral #trending",
        "duration_sec": 15.5,
        "upload_hour": 18,
        "is_weekend": 1,
        "category_id": 2,
    }
    response = client.post("/api/v1/predict/json", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Validate schema fields
    assert "virality_score" in data
    assert "predicted_category" in data
    assert "probabilities" in data
    assert "shap_explanation" in data
    assert "recommendations" in data

    assert 0.0 <= data["virality_score"] <= 100.0
    assert data["predicted_category"] in VIRALITY_CATEGORIES
    assert len(data["recommendations"]) > 0

    # Pydantic schema validation
    result_model = PredictionResult(**data)
    assert result_model.predicted_category == data["predicted_category"]


def test_predict_video_endpoint(client, dummy_video_file):
    """Test POST /api/v1/predict/video with multipart form upload."""
    form_data = {
        "caption": "Dance trend challenge! #dance #viral",
        "duration_sec": 12.0,
        "upload_hour": 20,
        "is_weekend": 1,
        "category_id": 1,
    }

    with open(dummy_video_file, "rb") as video_f:
        files = {
            "file": ("test_clip.mp4", video_f, "video/mp4")
        }
        response = client.post("/api/v1/predict/video", data=form_data, files=files)

    assert response.status_code == 200
    data = response.json()

    assert "virality_score" in data
    assert "predicted_category" in data
    assert 0.0 <= data["virality_score"] <= 100.0
    assert data["predicted_category"] in VIRALITY_CATEGORIES

    # Verify temp upload directory cleanup
    temp_dir = DATA_RAW_DIR / "temp_uploads"
    if temp_dir.exists():
        temp_files = list(temp_dir.glob("*"))
        assert len(temp_files) == 0, f"Temp upload files not cleaned up: {temp_files}"


def test_predict_invalid_input(client):
    """Test error handling for malformed JSON and invalid video formats."""
    # 1. Invalid JSON (missing required caption)
    invalid_payload = {
        "duration_sec": 15.0,
        # missing caption
    }
    resp_json = client.post("/api/v1/predict/json", json=invalid_payload)
    assert resp_json.status_code == 422

    # 2. Unsupported file format (.txt)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"not a video")
        tmp_txt = tmp.name

    try:
        with open(tmp_txt, "rb") as f:
            files = {"file": ("test.txt", f, "text/plain")}
            form_data = {"caption": "invalid file test"}
            resp_file = client.post("/api/v1/predict/video", data=form_data, files=files)
        assert resp_file.status_code == 400
        assert "Unsupported file format" in resp_file.json()["detail"]
    finally:
        Path(tmp_txt).unlink(missing_ok=True)
