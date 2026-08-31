"""
Audio Feature Extractor for Multimodal Social Media Virality Prediction.
Extracts acoustic metrics and 8-dimensional audio MFCC embeddings using librosa.
"""

from typing import Dict, Any, List
import warnings
import numpy as np


class AudioFeatureExtractor:
    """
    Extracts acoustic features from audio or video files:
    - rms_energy: Root Mean Square energy (signal volume/intensity).
    - tempo_bpm: Estimated tempo in Beats Per Minute.
    - zero_crossing_rate: Rate of sign-changes in audio signal.
    - spectral_centroid: Center of mass of the audio spectrum.
    - audio_embedding: 8-dimensional normalized MFCC feature summary.
    """

    def __init__(self, target_sr: int = 22050, max_duration_sec: float = 60.0):
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec

    def _get_fallback_features(self) -> Dict[str, Any]:
        """Returns default median fallback metrics for silent, missing, or unreadable audio."""
        return {
            "rms_energy": 0.10,
            "tempo_bpm": 120.0,
            "zero_crossing_rate": 0.05,
            "spectral_centroid": 1500.0,
            "audio_embedding": [0.0] * 8,
        }

    def extract_features(self, video_or_audio_path: str) -> Dict[str, Any]:
        """
        Extracts acoustic metrics and 8-dim MFCC embedding from the given media file.

        Parameters:
            video_or_audio_path (str): Path to audio file (.wav, .mp3) or video file (.mp4).

        Returns:
            Dict[str, Any]: Extracted acoustic features.
        """
        if not video_or_audio_path:
            return self._get_fallback_features()

        try:
            import librosa

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(
                    str(video_or_audio_path),
                    sr=self.target_sr,
                    mono=True,
                    duration=self.max_duration_sec,
                )

            if y is None or len(y) == 0 or np.all(y == 0):
                return self._get_fallback_features()

            # 1. RMS Energy
            rms = librosa.feature.rms(y=y)
            rms_energy = float(np.mean(rms))

            # 2. Tempo (BPM)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, (np.ndarray, list)):
                tempo_val = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo_val = float(tempo)
            if np.isnan(tempo_val) or tempo_val <= 0:
                tempo_val = 120.0

            # 3. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y=y)
            zero_crossing_rate = float(np.mean(zcr))

            # 4. Spectral Centroid
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid = float(np.mean(sc))

            # 5. 8-dimensional MFCC embedding summary
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)
            mfcc_means = np.mean(mfccs, axis=1)
            norm = np.linalg.norm(mfcc_means)
            if norm > 1e-6:
                mfcc_means = mfcc_means / norm
            audio_embedding = [float(x) for x in mfcc_means]

            return {
                "rms_energy": round(rms_energy, 4),
                "tempo_bpm": round(tempo_val, 2),
                "zero_crossing_rate": round(zero_crossing_rate, 4),
                "spectral_centroid": round(spectral_centroid, 2),
                "audio_embedding": audio_embedding,
            }

        except Exception:
            return self._get_fallback_features()
