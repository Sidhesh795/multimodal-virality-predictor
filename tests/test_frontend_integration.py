"""
Integration tests for Phase 6 interactive web frontend.
Verifies static dashboard files exist and GET / serves HTML.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.config import BASE_DIR
from backend.app.main import app


FRONTEND_DIR = BASE_DIR / "frontend"


@pytest.fixture(scope="module")
def client():
    """Fixture providing FastAPI TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


def test_frontend_files_exist_and_non_empty():
    """Dashboard HTML, CSS, and JS assets must exist and contain content."""
    for name in ("index.html", "style.css", "app.js"):
        path = FRONTEND_DIR / name
        assert path.exists(), f"Missing frontend file: {path}"
        assert path.stat().st_size > 0, f"Empty frontend file: {path}"
        assert path.read_text(encoding="utf-8").strip(), f"Blank frontend file: {path}"


def test_root_serves_html(client):
    """GET / should return 200 OK with HTML dashboard content."""
    response = client.get("/")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "html" in content_type.lower() or "<html" in response.text.lower()
    body = response.text.lower()
    assert "<html" in body
    assert "virality" in body
    assert "cdn.jsdelivr.net/npm/chart.js" in body
    assert 'id="ablation-chart"' in response.text
    assert 'id="prob-chart"' in response.text
    assert 'id="shap-chart"' in response.text


def test_static_assets_served(client):
    """GET /style.css and GET /app.js should be served by StaticFiles."""
    css = client.get("/style.css")
    js = client.get("/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert len(css.content) > 0
    assert len(js.content) > 0
