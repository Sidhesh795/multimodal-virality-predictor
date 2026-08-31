"""
Video Feature Extractor for Multimodal Social Media Virality Prediction.
Extracts frame-level metrics and 16-dimensional visual embeddings using OpenCV.
"""

from typing import Dict, Any, List
import numpy as np
import cv2


class VideoFeatureExtractor:
    """
    Extracts visual features from video files:
    - scene_cut_count: Count of sudden scene transitions based on histogram differences.
    - avg_brightness: Average luminance across frames.
    - color_variance: Standard deviation of HSV color components across frames.
    - motion_score: Mean frame-to-frame pixel differences.
    - visual_embedding: 16-dimensional spatial representation vector.
    """

    def __init__(self, target_fps_sample: float = 1.0, hist_diff_threshold: float = 0.45):
        """
        Parameters:
            target_fps_sample (float): Frames per second to sample (default 1.0 for fast extraction).
            hist_diff_threshold (float): Threshold for detecting scene cuts via histogram distance.
        """
        self.target_fps_sample = target_fps_sample
        self.hist_diff_threshold = hist_diff_threshold

    def _get_fallback_features(self) -> Dict[str, Any]:
        """Returns clean fallback metrics for empty, corrupt, or unreadable video files."""
        return {
            "scene_cut_count": 0,
            "avg_brightness": 128.0,
            "color_variance": 30.0,
            "motion_score": 0.0,
            "visual_embedding": [0.0] * 16,
        }

    def extract_features(self, video_path: str) -> Dict[str, Any]:
        """
        Extracts visual metrics and embedding from the given video file path.

        Parameters:
            video_path (str): Path to MP4 or video file.

        Returns:
            Dict[str, Any]: Extracted visual features.
        """
        if not video_path:
            return self._get_fallback_features()

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return self._get_fallback_features()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return self._get_fallback_features()

            # Downsample to ~1 frame per second
            frame_interval = max(1, int(round(fps / self.target_fps_sample)))

            sampled_frames = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    # Resize to standard compact resolution (160x120) for rapid computation
                    small_frame = cv2.resize(frame, (160, 120))
                    sampled_frames.append(small_frame)
                frame_idx += 1

            cap.release()

            if not sampled_frames:
                return self._get_fallback_features()

            # 1. Scene Cuts and Motion Score
            scene_cuts = 0
            motion_diffs = []
            prev_hist = None
            prev_gray = None

            brightness_list = []
            hsv_vars = []
            spatial_embeddings = []

            for frame in sampled_frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Brightness
                brightness_list.append(float(np.mean(gray)))

                # Color variance (mean std across H, S, V channels)
                hsv_vars.append(float(np.mean(np.std(hsv, axis=(0, 1)))))

                # Spatial 4x4 representation for 16-dim visual embedding
                grid_4x4 = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA)
                spatial_embeddings.append(grid_4x4.flatten().astype(float))

                # Color Histogram for scene cut detection
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                if prev_hist is not None:
                    # Bhattacharyya distance in [0, 1]
                    hist_dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    if hist_dist > self.hist_diff_threshold:
                        scene_cuts += 1

                if prev_gray is not None:
                    # Motion: normalized absolute difference
                    diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))) / 255.0
                    motion_diffs.append(diff)

                prev_hist = hist
                prev_gray = gray

            avg_brightness = float(np.mean(brightness_list)) if brightness_list else 128.0
            avg_color_variance = float(np.mean(hsv_vars)) if hsv_vars else 30.0
            motion_score = float(np.mean(motion_diffs)) if motion_diffs else 0.0

            # 16-dim visual embedding from averaged 4x4 spatial grids, normalized
            mean_embedding = np.mean(spatial_embeddings, axis=0)
            norm = np.linalg.norm(mean_embedding)
            if norm > 1e-6:
                mean_embedding = mean_embedding / norm
            visual_embedding = [float(x) for x in mean_embedding]

            return {
                "scene_cut_count": int(scene_cuts),
                "avg_brightness": round(avg_brightness, 2),
                "color_variance": round(avg_color_variance, 2),
                "motion_score": round(float(np.clip(motion_score, 0.0, 1.0)), 4),
                "visual_embedding": visual_embedding,
            }

        except Exception:
            return self._get_fallback_features()
