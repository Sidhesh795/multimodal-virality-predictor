"""
Synthetic Dataset Generator for Multimodal Social Media Virality Prediction.
Generates 1,000 realistic short-form content records containing metadata, visual,
audio, and text features along with ground truth continuous virality scores
and categorical performance labels.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backend.app.config import (
    SYNTHETIC_DATASET_PATH,
    RANDOM_SEED,
    VIRALITY_CATEGORIES,
)


def generate_synthetic_dataset(
    num_samples: int = 1000, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generates a synthetic multimodal dataset of short-form video records.

    Parameters:
        num_samples (int): Number of records to generate (default 1000).
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Complete synthetic dataset.
    """
    np.random.seed(seed)

    # 1. Metadata Features
    duration_sec = np.random.uniform(5.0, 60.0, num_samples)
    upload_hour = np.random.randint(0, 24, num_samples)
    upload_hour_sin = np.sin(2 * np.pi * upload_hour / 24.0)
    upload_hour_cos = np.cos(2 * np.pi * upload_hour / 24.0)
    is_weekend = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    category_id = np.random.choice([0, 1, 2, 3, 4], size=num_samples)

    # 2. Visual Features
    scene_cut_count = np.random.randint(1, 16, num_samples)
    avg_brightness = np.random.uniform(20.0, 240.0, num_samples)
    color_variance = np.random.uniform(10.0, 100.0, num_samples)
    motion_score = np.random.uniform(0.05, 0.95, num_samples)
    visual_embeddings = np.random.normal(0, 1, size=(num_samples, 16))
    # Normalize visual embedding vectors
    visual_embeddings = visual_embeddings / np.linalg.norm(
        visual_embeddings, axis=1, keepdims=True
    )

    # 3. Audio Features
    rms_energy = np.random.uniform(0.01, 0.5, num_samples)
    tempo_bpm = np.random.uniform(70.0, 170.0, num_samples)
    zero_crossing_rate = np.random.uniform(0.01, 0.3, num_samples)
    spectral_centroid = np.random.uniform(500.0, 4000.0, num_samples)
    audio_embeddings = np.random.normal(0, 1, size=(num_samples, 8))
    audio_embeddings = audio_embeddings / np.linalg.norm(
        audio_embeddings, axis=1, keepdims=True
    )

    # 4. Text Features
    caption_length = np.random.randint(10, 251, num_samples)
    hashtag_count = np.random.randint(0, 11, num_samples)
    sentiment_polarity = np.random.uniform(-1.0, 1.0, num_samples)
    sentiment_subjectivity = np.random.uniform(0.0, 1.0, num_samples)
    text_embeddings = np.random.normal(0, 1, size=(num_samples, 16))
    text_embeddings = text_embeddings / np.linalg.norm(
        text_embeddings, axis=1, keepdims=True
    )

    # 5. Target Calculation (Realistic Multimodal Signal Synthesis)
    # Composite score built from optimal combinations of features + noise
    tempo_optimality = 1.0 - np.clip(np.abs(tempo_bpm - 125.0) / 55.0, 0, 1)
    duration_optimality = 1.0 - np.clip(np.abs(duration_sec - 18.0) / 42.0, 0, 1)
    sentiment_score = (sentiment_polarity + 1.0) / 2.0
    embedding_signal = (
        visual_embeddings[:, 0] * 0.4
        + audio_embeddings[:, 0] * 0.3
        + text_embeddings[:, 0] * 0.3
    )

    raw_signal = (
        0.20 * motion_score
        + 0.15 * tempo_optimality
        + 0.15 * duration_optimality
        + 0.15 * (hashtag_count / 10.0)
        + 0.10 * sentiment_score
        + 0.10 * (scene_cut_count / 15.0)
        + 0.05 * is_weekend
        + 0.10 * embedding_signal
        + np.random.normal(0, 0.08, num_samples)
    )

    # Convert to Normalized Engagement Ratio (NER)
    min_sig, max_sig = np.min(raw_signal), np.max(raw_signal)
    ner = (raw_signal - min_sig) / (max_sig - min_sig)

    # Scale to continuous virality_score [0, 100]
    virality_score = np.round(ner * 100.0, 2)

    # Build DataFrame
    data = {
        "duration_sec": duration_sec,
        "upload_hour_sin": upload_hour_sin,
        "upload_hour_cos": upload_hour_cos,
        "is_weekend": is_weekend,
        "category_id": category_id,
        "scene_cut_count": scene_cut_count,
        "avg_brightness": avg_brightness,
        "color_variance": color_variance,
        "motion_score": motion_score,
        "rms_energy": rms_energy,
        "tempo_bpm": tempo_bpm,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid": spectral_centroid,
        "caption_length": caption_length,
        "hashtag_count": hashtag_count,
        "sentiment_polarity": sentiment_polarity,
        "sentiment_subjectivity": sentiment_subjectivity,
    }

    # Add 16 visual embedding columns
    for i in range(16):
        data[f"visual_emb_{i}"] = visual_embeddings[:, i]

    # Add 8 audio embedding columns
    for i in range(8):
        data[f"audio_emb_{i}"] = audio_embeddings[:, i]

    # Add 16 text embedding columns
    for i in range(16):
        data[f"text_emb_{i}"] = text_embeddings[:, i]

    # Add targets
    data["ner"] = ner
    data["virality_score"] = virality_score

    df = pd.DataFrame(data)

    # Categorical Virality Binning (Low, Medium, High, Viral)
    df["virality_category"] = pd.qcut(
        df["virality_score"], q=4, labels=VIRALITY_CATEGORIES
    ).astype(str)

    return df


def main():
    print("Generating synthetic dataset (1,000 records)...")
    df = generate_synthetic_dataset(num_samples=1000, seed=RANDOM_SEED)

    # Ensure output directory exists
    output_path = Path(SYNTHETIC_DATASET_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("Class distribution:")
    print(df["virality_category"].value_counts())


if __name__ == "__main__":
    main()
