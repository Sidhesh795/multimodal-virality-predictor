"""
Pydantic Schemas for Multimodal Social Media Virality Prediction.
Defines input payloads, feature vectors, and prediction results.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class ContentInput(BaseModel):
    """Input payload for raw short-form content metadata and text."""

    caption: str = Field(..., description="Caption text of the post")
    duration_sec: float = Field(
        ..., ge=1.0, le=300.0, description="Video duration in seconds"
    )
    upload_hour: int = Field(
        ..., ge=0, le=23, description="Hour of content upload (0-23)"
    )
    is_weekend: int = Field(
        0, ge=0, le=1, description="1 if uploaded on weekend, 0 otherwise"
    )
    category_id: int = Field(0, ge=0, description="Content category identifier")
    media_file_path: Optional[str] = Field(
        None, description="Optional path to associated video/audio media file"
    )


class FeatureVector(BaseModel):
    """
    Explicit schema for combined multimodal feature vector extracted from raw content.
    Includes metadata, visual, audio, and text metrics and vector embeddings.
    """

    # Metadata features
    duration_sec: float = Field(..., description="Content duration in seconds")
    upload_hour_sin: float = Field(
        ..., ge=-1.0, le=1.0, description="Sine transform of upload hour"
    )
    upload_hour_cos: float = Field(
        ..., ge=-1.0, le=1.0, description="Cosine transform of upload hour"
    )
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend flag (0 or 1)")
    category_id: int = Field(..., ge=0, description="Category ID")

    # Visual metrics & embedding
    scene_cut_count: int = Field(..., ge=0, description="Number of scene cuts")
    avg_brightness: float = Field(
        ..., ge=0.0, le=255.0, description="Average frame brightness"
    )
    color_variance: float = Field(
        ..., ge=0.0, description="Frame color variance"
    )
    motion_score: float = Field(
        ..., ge=0.0, le=1.0, description="Frame motion intensity score"
    )
    visual_embedding: List[float] = Field(
        ..., description="16-dimensional visual embedding vector"
    )

    # Audio metrics & embedding
    rms_energy: float = Field(..., ge=0.0, description="Root Mean Square energy")
    tempo_bpm: float = Field(..., ge=0.0, description="Tempo in beats per minute")
    zero_crossing_rate: float = Field(
        ..., ge=0.0, description="Zero crossing rate"
    )
    spectral_centroid: float = Field(
        ..., ge=0.0, description="Audio spectral centroid"
    )
    audio_embedding: List[float] = Field(
        ..., description="8-dimensional audio embedding vector"
    )

    # Text metrics & embedding
    caption_length: int = Field(
        ..., ge=0, description="Length of caption string"
    )
    hashtag_count: int = Field(
        ..., ge=0, description="Number of hashtags in caption"
    )
    sentiment_polarity: float = Field(
        ..., ge=-1.0, le=1.0, description="Sentiment polarity (-1.0 to 1.0)"
    )
    sentiment_subjectivity: float = Field(
        ..., ge=0.0, le=1.0, description="Sentiment subjectivity (0.0 to 1.0)"
    )
    text_embedding: List[float] = Field(
        ..., description="16-dimensional text embedding vector"
    )

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattens feature vector including individual embedding indices."""
        d = {
            "duration_sec": self.duration_sec,
            "upload_hour_sin": self.upload_hour_sin,
            "upload_hour_cos": self.upload_hour_cos,
            "is_weekend": self.is_weekend,
            "category_id": self.category_id,
            "scene_cut_count": self.scene_cut_count,
            "avg_brightness": self.avg_brightness,
            "color_variance": self.color_variance,
            "motion_score": self.motion_score,
            "rms_energy": self.rms_energy,
            "tempo_bpm": self.tempo_bpm,
            "zero_crossing_rate": self.zero_crossing_rate,
            "spectral_centroid": self.spectral_centroid,
            "caption_length": self.caption_length,
            "hashtag_count": self.hashtag_count,
            "sentiment_polarity": self.sentiment_polarity,
            "sentiment_subjectivity": self.sentiment_subjectivity,
        }
        for i, val in enumerate(self.visual_embedding):
            d[f"visual_emb_{i}"] = val
        for i, val in enumerate(self.audio_embedding):
            d[f"audio_emb_{i}"] = val
        for i, val in enumerate(self.text_embedding):
            d[f"text_emb_{i}"] = val
        return d


class PredictionResult(BaseModel):
    """
    Output payload containing virality score, predicted category,
    probability breakdown, explainability feature impacts, and recommendations.
    """

    virality_score: float = Field(
        ..., ge=0.0, le=100.0, description="Continuous virality score (0-100)"
    )
    predicted_category: str = Field(
        ..., description="Performance category: Low, Medium, High, or Viral"
    )
    probabilities: Dict[str, float] = Field(
        ..., description="Probability distribution across virality categories"
    )
    explainability_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Feature importance/SHAP impact breakdown"
    )
    shap_explanation: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Detailed SHAP decomposition and driver breakdown"
    )
    recommendations: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="Actionable recommendations for virality optimization"
    )

