"""
Text Feature Extractor for Multimodal Social Media Virality Prediction.
Extracts caption metrics, sentiment analysis via TextBlob, and 16-dimensional
text embeddings via SentenceTransformers or deterministic projections.
"""

from typing import Dict, Any, List
import hashlib
import numpy as np


class TextFeatureExtractor:
    """
    Extracts linguistic and semantic features from social media captions:
    - caption_length: Total character count.
    - hashtag_count: Total count of hashtags.
    - sentiment_polarity: Sentiment polarity in [-1.0, 1.0].
    - sentiment_subjectivity: Sentiment subjectivity in [0.0, 1.0].
    - text_embedding: 16-dimensional continuous semantic representation.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_transformer: bool = True):
        self.model_name = model_name
        self.use_transformer = use_transformer
        self._model = None
        self._model_failed = False

    def _get_transformer_model(self):
        """Lazy load SentenceTransformer model if available."""
        if not self.use_transformer or self._model_failed:
            return None
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model_failed = True
                self._model = None
        return self._model

    def _compute_deterministic_embedding(self, text: str, dim: int = 16) -> List[float]:
        """
        Generates a deterministic continuous embedding of fixed dimension from text.
        Used as a high-speed offline fallback or embedding compression.
        """
        if not text:
            return [0.0] * dim

        # Use hashes of n-grams and tokens to build deterministic pseudorandom seed
        vec = np.zeros(dim, dtype=float)
        words = text.lower().split()
        for idx, word in enumerate(words):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            for d in range(dim):
                weight = ((h >> (d * 4)) & 0xFF) / 255.0 - 0.5
                vec[d] += weight / (idx + 1) ** 0.5

        # Also encode overall text hash
        full_h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        for d in range(dim):
            vec[d] += (((full_h >> (d * 7)) & 0xFF) / 255.0 - 0.5) * 0.5

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return [float(x) for x in vec]

    def extract_features(self, caption: str) -> Dict[str, Any]:
        """
        Extracts linguistic, sentiment, and embedding metrics from the caption.

        Parameters:
            caption (str): Input text caption of the post.

        Returns:
            Dict[str, Any]: Extracted text features.
        """
        if caption is None:
            caption = ""

        caption_length = len(caption)
        hashtag_count = caption.count("#")

        # 1. Sentiment Analysis
        sentiment_polarity = 0.0
        sentiment_subjectivity = 0.0

        if caption.strip():
            try:
                from textblob import TextBlob
                blob = TextBlob(caption)
                sentiment_polarity = float(blob.sentiment.polarity)
                sentiment_subjectivity = float(blob.sentiment.subjectivity)
            except Exception:
                sentiment_polarity = 0.0
                sentiment_subjectivity = 0.0

        # 2. Text Embedding (16-dimensional)
        text_embedding = None
        model = self._get_transformer_model()

        if model is not None and caption.strip():
            try:
                raw_emb = model.encode(caption, convert_to_numpy=True)
                # Compress / slice 384-dim embedding to 16-dim via linear pooling / slice
                # Taking 16 evenly spaced sub-blocks averaged
                if len(raw_emb) >= 16:
                    step = len(raw_emb) // 16
                    compressed = np.array([np.mean(raw_emb[i * step : (i + 1) * step]) for i in range(16)])
                else:
                    compressed = np.pad(raw_emb, (0, 16 - len(raw_emb)))
                norm = np.linalg.norm(compressed)
                if norm > 1e-6:
                    compressed = compressed / norm
                text_embedding = [float(x) for x in compressed]
            except Exception:
                text_embedding = None

        if text_embedding is None:
            text_embedding = self._compute_deterministic_embedding(caption, dim=16)

        return {
            "caption_length": int(caption_length),
            "hashtag_count": int(hashtag_count),
            "sentiment_polarity": round(sentiment_polarity, 4),
            "sentiment_subjectivity": round(sentiment_subjectivity, 4),
            "text_embedding": text_embedding,
        }
