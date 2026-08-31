"""
Unit Tests for Multimodal Feature Extractors and Pipeline.
Tests VideoFeatureExtractor, AudioFeatureExtractor, TextFeatureExtractor,
and MultimodalFeaturePipeline.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2

from backend.app.config import FEATURE_COLUMNS
from backend.app.schemas import FeatureVector
from src.features.video_extractor import VideoFeatureExtractor
from src.features.audio_extractor import AudioFeatureExtractor
from src.features.text_extractor import TextFeatureExtractor
from src.features.pipeline import MultimodalFeaturePipeline


def create_dummy_video(file_path: str, duration_sec: int = 3, fps: int = 30) -> str:
    """
    Creates a temporary dummy video file with color scene changes for testing.

    Parameters:
        file_path (str): Output video file path.
        duration_sec (int): Video duration in seconds.
        fps (int): Video frames per second.

    Returns:
        str: Absolute path of created video file.
    """
    width, height = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Create 3 distinct color scenes to stimulate scene cut and motion detection
        if i < total_frames // 3:
            frame = np.full((height, width, 3), (200, 50, 50), dtype=np.uint8)
        elif i < 2 * total_frames // 3:
            frame = np.full((height, width, 3), (50, 200, 50), dtype=np.uint8)
        else:
            frame = np.full((height, width, 3), (50, 50, 200), dtype=np.uint8)

        # Add moving square to stimulate motion
        square_x = (i * 2) % (width - 10)
        frame[20:30, square_x : square_x + 10] = (255, 255, 255)
        out.write(frame)

    out.release()
    return str(file_path)


@pytest.fixture(scope="module")
def sample_video_path():
    """Fixture providing a temporary dummy video file path."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    create_dummy_video(tmp_path, duration_sec=3, fps=30)
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


def test_video_feature_extractor(sample_video_path):
    """Test video feature extraction on dummy video."""
    extractor = VideoFeatureExtractor(target_fps_sample=1.0)
    feats = extractor.extract_features(sample_video_path)

    assert isinstance(feats, dict)
    assert "scene_cut_count" in feats
    assert "avg_brightness" in feats
    assert "color_variance" in feats
    assert "motion_score" in feats
    assert "visual_embedding" in feats

    assert feats["scene_cut_count"] >= 0
    assert 0.0 <= feats["avg_brightness"] <= 255.0
    assert feats["color_variance"] >= 0.0
    assert 0.0 <= feats["motion_score"] <= 1.0
    assert len(feats["visual_embedding"]) == 16
    assert all(isinstance(x, float) for x in feats["visual_embedding"])


def test_video_feature_extractor_corrupt_fallback():
    """Test fallback when video is missing or corrupt."""
    extractor = VideoFeatureExtractor()
    feats = extractor.extract_features("non_existent_file.mp4")

    assert isinstance(feats, dict)
    assert feats["scene_cut_count"] == 0
    assert len(feats["visual_embedding"]) == 16


def test_audio_feature_extractor_fallback():
    """Test audio feature extraction fallback on video with no separate audio stream."""
    extractor = AudioFeatureExtractor()
    feats = extractor.extract_features("non_existent_audio.mp4")

    assert isinstance(feats, dict)
    assert "rms_energy" in feats
    assert "tempo_bpm" in feats
    assert "zero_crossing_rate" in feats
    assert "spectral_centroid" in feats
    assert "audio_embedding" in feats

    assert feats["rms_energy"] >= 0.0
    assert feats["tempo_bpm"] > 0.0
    assert len(feats["audio_embedding"]) == 8


def test_text_feature_extractor():
    """Test text feature extraction for captions."""
    extractor = TextFeatureExtractor(use_transformer=True)
    caption = "Check out this amazing viral video! #trending #viral #foryou"
    feats = extractor.extract_features(caption)

    assert isinstance(feats, dict)
    assert feats["caption_length"] == len(caption)
    assert feats["hashtag_count"] == 3
    assert -1.0 <= feats["sentiment_polarity"] <= 1.0
    assert 0.0 <= feats["sentiment_subjectivity"] <= 1.0
    assert len(feats["text_embedding"]) == 16
    assert all(isinstance(x, float) for x in feats["text_embedding"])

    # Test empty caption handling
    empty_feats = extractor.extract_features("")
    assert empty_feats["caption_length"] == 0
    assert empty_feats["hashtag_count"] == 0
    assert len(empty_feats["text_embedding"]) == 16


def test_multimodal_feature_pipeline(sample_video_path):
    """Test multimodal pipeline feature combination and DataFrame schema compliance."""
    pipeline = MultimodalFeaturePipeline()
    caption = "Incredible moment captured! #fyp #wow"
    metadata = {
        "duration_sec": 12.5,
        "upload_hour": 19,
        "is_weekend": 1,
        "category_id": 3,
    }

    # 1. Test DataFrame extraction
    df = pipeline.extract_all(sample_video_path, caption, metadata)

    assert df.shape == (1, 57), f"Expected shape (1, 57), got {df.shape}"
    assert list(df.columns) == FEATURE_COLUMNS
    assert df.isna().sum().sum() == 0, "Pipeline output DataFrame contains NaN values"

    # 2. Test FeatureVector Pydantic model extraction
    feature_vec = pipeline.extract_feature_vector(sample_video_path, caption, metadata)
    assert isinstance(feature_vec, FeatureVector)
    assert feature_vec.duration_sec == 12.5
    assert feature_vec.category_id == 3
    assert feature_vec.is_weekend == 1
    assert len(feature_vec.visual_embedding) == 16
    assert len(feature_vec.audio_embedding) == 8
    assert len(feature_vec.text_embedding) == 16
