"""
Actionable Recommendation Engine for Social Media Virality Optimization.
Translates SHAP feature attributions into concrete, domain-specific creator recommendations.
"""

from typing import Dict, Any, List
import pandas as pd


class RecommendationEngine:
    """
    Expert rule-driven recommendation generator based on negative SHAP feature
    contributions and multimodal performance metrics.
    """

    # Domain-specific actionable advice mappings
    SUGGESTION_MAP = {
        "scene_cut_count": (
            "Visual pacing is slow. Increase scene transitions or visual cuts (every 2-3 seconds) "
            "to maintain audience retention."
        ),
        "avg_brightness": (
            "Video lighting is dim. Increase scene brightness, exposure, or contrast to improve "
            "visual clarity and scroll-stopping appeal."
        ),
        "motion_score": (
            "Visual dynamics are low. Introduce more movement, camera dynamics, or visual action "
            "in the first 3 seconds."
        ),
        "rms_energy": (
            "Audio dynamics are soft. Boost speech volume or select higher-energy background music "
            "to enhance auditory presence."
        ),
        "tempo_bpm": (
            "Audio tempo is sub-optimal. Experiment with upbeat audio tracks (120-130 BPM) "
            "to drive energetic viewer engagement."
        ),
        "zero_crossing_rate": (
            "Audio clarity is noisy or inconsistent. Enhance voiceover clarity and reduce background noise."
        ),
        "spectral_centroid": (
            "Audio brightness is low. Enhance treble and vocal frequencies for crisp sound delivery."
        ),
        "sentiment_polarity": (
            "Caption sentiment is neutral or flat. Try adding emotional triggers, curiosity hooks, "
            "or enthusiastic phrasing."
        ),
        "sentiment_subjectivity": (
            "Caption tone is overly detached. Use personal, opinionated, or storytelling language."
        ),
        "hashtag_count": (
            "Hashtag usage is low. Include 3-5 relevant niche and trending hashtags to expand search discovery."
        ),
        "caption_length": (
            "Caption length could be optimized. Aim for concise, hook-driven captions with a clear call-to-action."
        ),
        "upload_hour_sin": (
            "Upload timing is off-peak. Schedule posting during prime audience engagement hours (6 PM - 9 PM)."
        ),
        "upload_hour_cos": (
            "Upload timing is off-peak. Schedule posting during prime audience engagement hours (6 PM - 9 PM)."
        ),
        "duration_sec": (
            "Video duration could be tuned. Keep short-form videos punchy (15-25 seconds) "
            "to maximize completion rate and looping."
        ),
        "is_weekend": (
            "Posting schedule: Weekend posting windows often yield higher leisure engagement."
        ),
    }

    def generate_recommendations(
        self,
        shap_analysis: Dict[str, Any],
        feature_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Generates prioritized creator recommendations based on negative SHAP impacts.

        Parameters:
            shap_analysis (Dict[str, Any]): Output dictionary from ViralityExplainer.
            feature_df (pd.DataFrame): Input feature DataFrame.

        Returns:
            List[Dict[str, Any]]: List of recommendation objects with priority and suggestion.
        """
        top_negatives = shap_analysis.get("top_negative_features", [])
        recommendations = []
        seen_suggestions = set()

        for idx, item in enumerate(top_negatives):
            feat_name = item["feature"]
            shap_val = float(item["shap_value"])

            # Map embeddings to generic modal recommendations if not in map
            if feat_name.startswith("visual_emb"):
                suggestion = (
                    "Visual aesthetic composition can be improved. Enhance color saturation "
                    "and subject focal framing."
                )
                feat_label = "visual_composition"
            elif feat_name.startswith("audio_emb"):
                suggestion = (
                    "Audio timbre/music choice has negative impact. Try popular trending sound bites "
                    "or clearer vocal mixing."
                )
                feat_label = "audio_timbre"
            elif feat_name.startswith("text_emb"):
                suggestion = (
                    "Caption semantic structure is weak. Use strong opening hooks and direct questions."
                )
                feat_label = "caption_semantics"
            else:
                suggestion = self.SUGGESTION_MAP.get(
                    feat_name,
                    f"Optimize feature '{feat_name}' to enhance overall virality score."
                )
                feat_label = feat_name

            if suggestion in seen_suggestions:
                continue
            seen_suggestions.add(suggestion)

            # Assign priority: HIGH for top 2 negative drivers or shap_val <= -1.0, otherwise MEDIUM
            priority = "HIGH" if (idx < 2 or shap_val <= -1.0) else "MEDIUM"

            recommendations.append({
                "feature": feat_label,
                "shap_value": shap_val,
                "suggestion": suggestion,
                "priority": priority,
            })

        # Fallback if no negative drivers were detected
        if not recommendations:
            recommendations.append({
                "feature": "overall",
                "shap_value": 0.0,
                "suggestion": (
                    "Content metrics are well-balanced across all modalities. "
                    "Maintain current production and hook quality!"
                ),
                "priority": "LOW",
            })

        return recommendations
