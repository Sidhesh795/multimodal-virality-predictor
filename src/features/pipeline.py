"""
Multimodal Feature Extraction Pipeline for Virality Prediction.
Combines video, audio, text extractors, and metadata transformations
into a standardized feature DataFrame and Pydantic FeatureVector.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from backend.app.config import FEATURE_COLUMNS
from backend.app.schemas import FeatureVector
from src.features.video_extractor import VideoFeatureExtractor
from src.features.audio_extractor import AudioFeatureExtractor
from src.features.text_extractor import TextFeatureExtractor


class MultimodalFeaturePipeline:
    """
    Unified multimodal feature extraction engine.
    Ingests video, caption, and metadata to produce aligned feature representations.
    """

    def __init__(
        self,
        video_extractor: Optional[VideoFeatureExtractor] = None,
        audio_extractor: Optional[AudioFeatureExtractor] = None,
        text_extractor: Optional[TextFeatureExtractor] = None,
    ):
        self.video_extractor = video_extractor or VideoFeatureExtractor()
        self.audio_extractor = audio_extractor or AudioFeatureExtractor()
        self.text_extractor = text_extractor or TextFeatureExtractor()

    def extract_feature_dict(
        self,
        video_path: Optional[str],
        caption: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts all multimodal features and returns a flattened dictionary.

        Parameters:
            video_path (Optional[str]): Path to video file.
            caption (str): Post caption text.
            metadata (Optional[Dict[str, Any]]): Metadata dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing all 57 features.
        """
        meta = metadata or {}

        # 1. Process Metadata
        duration_sec = float(meta.get("duration_sec", 15.0))
        if "upload_hour_sin" in meta and "upload_hour_cos" in meta:
            upload_hour_sin = float(meta["upload_hour_sin"])
            upload_hour_cos = float(meta["upload_hour_cos"])
        else:
            upload_hour = int(meta.get("upload_hour", 12))
            upload_hour_sin = float(np.sin(2 * np.pi * upload_hour / 24.0))
            upload_hour_cos = float(np.cos(2 * np.pi * upload_hour / 24.0))

        is_weekend = int(meta.get("is_weekend", 0))
        category_id = int(meta.get("category_id", 0))

        # 2. Extract Features from Modalities
        video_feats = self.video_extractor.extract_features(video_path or "")
        audio_path = meta.get("audio_path", video_path) or ""
        audio_feats = self.audio_extractor.extract_features(audio_path)
        text_feats = self.text_extractor.extract_features(caption or "")

        # 3. Assemble Flat Feature Dictionary
        row: Dict[str, Any] = {
            # Metadata
            "duration_sec": duration_sec,
            "upload_hour_sin": upload_hour_sin,
            "upload_hour_cos": upload_hour_cos,
            "is_weekend": is_weekend,
            "category_id": category_id,
            # Visual Scalars
            "scene_cut_count": int(video_feats["scene_cut_count"]),
            "avg_brightness": float(video_feats["avg_brightness"]),
            "color_variance": float(video_feats["color_variance"]),
            "motion_score": float(video_feats["motion_score"]),
            # Audio Scalars
            "rms_energy": float(audio_feats["rms_energy"]),
            "tempo_bpm": float(audio_feats["tempo_bpm"]),
            "zero_crossing_rate": float(audio_feats["zero_crossing_rate"]),
            "spectral_centroid": float(audio_feats["spectral_centroid"]),
            # Text Scalars
            "caption_length": int(text_feats["caption_length"]),
            "hashtag_count": int(text_feats["hashtag_count"]),
            "sentiment_polarity": float(text_feats["sentiment_polarity"]),
            "sentiment_subjectivity": float(text_feats["sentiment_subjectivity"]),
        }

        # Visual Embeddings (16-dim)
        vis_emb = video_feats.get("visual_embedding", [0.0] * 16)
        for i in range(16):
            row[f"visual_emb_{i}"] = float(vis_emb[i]) if i < len(vis_emb) else 0.0

        # Audio Embeddings (8-dim)
        aud_emb = audio_feats.get("audio_embedding", [0.0] * 8)
        for i in range(8):
            row[f"audio_emb_{i}"] = float(aud_emb[i]) if i < len(aud_emb) else 0.0

        # Text Embeddings (16-dim)
        txt_emb = text_feats.get("text_embedding", [0.0] * 16)
        for i in range(16):
            row[f"text_emb_{i}"] = float(txt_emb[i]) if i < len(txt_emb) else 0.0

        return row

    def extract_feature_vector(
        self,
        video_path: Optional[str],
        caption: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeatureVector:
        """
        Extracts multimodal features and validates them against Pydantic FeatureVector.
        """
        flat_dict = self.extract_feature_dict(video_path, caption, metadata)

        vis_emb = [flat_dict[f"visual_emb_{i}"] for i in range(16)]
        aud_emb = [flat_dict[f"audio_emb_{i}"] for i in range(8)]
        txt_emb = [flat_dict[f"text_emb_{i}"] for i in range(16)]

        return FeatureVector(
            duration_sec=flat_dict["duration_sec"],
            upload_hour_sin=flat_dict["upload_hour_sin"],
            upload_hour_cos=flat_dict["upload_hour_cos"],
            is_weekend=flat_dict["is_weekend"],
            category_id=flat_dict["category_id"],
            scene_cut_count=flat_dict["scene_cut_count"],
            avg_brightness=flat_dict["avg_brightness"],
            color_variance=flat_dict["color_variance"],
            motion_score=flat_dict["motion_score"],
            visual_embedding=vis_emb,
            rms_energy=flat_dict["rms_energy"],
            tempo_bpm=flat_dict["tempo_bpm"],
            zero_crossing_rate=flat_dict["zero_crossing_rate"],
            spectral_centroid=flat_dict["spectral_centroid"],
            audio_embedding=aud_emb,
            caption_length=flat_dict["caption_length"],
            hashtag_count=flat_dict["hashtag_count"],
            sentiment_polarity=flat_dict["sentiment_polarity"],
            sentiment_subjectivity=flat_dict["sentiment_subjectivity"],
            text_embedding=txt_emb,
        )

    def extract_all(
        self,
        video_path: Optional[str],
        caption: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Extracts multimodal features and returns a single-row DataFrame aligned with FEATURE_COLUMNS.

        Parameters:
            video_path (Optional[str]): Path to video file.
            caption (str): Post caption text.
            metadata (Optional[Dict[str, Any]]): Metadata dictionary.

        Returns:
            pd.DataFrame: 1-row DataFrame containing all feature columns.
        """
        flat_dict = self.extract_feature_dict(video_path, caption, metadata)
        df = pd.DataFrame([flat_dict])

        # Ensure all columns in FEATURE_COLUMNS exist in exact order
        df = df.reindex(columns=FEATURE_COLUMNS)
        return df
