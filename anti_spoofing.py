"""
anti_spoofing.py
----------------
Anti-Spoofing & Liveness Detection Engine.

Protects the attendance system against presentation attacks (spoofing):
  - Printed paper photos & ID cards
  - Smartphone, tablet, & computer screen displays
  - Photo cutouts & 2D masks

Architecture:
  - Primary: MiniFASNet V2 / ONNX Deep Anti-Spoofing Model (if onnxruntime & model available).
  - Secondary: Multi-factor Texture & Fourier Frequency Analysis (Moiré screen patterns,
    Laplacian variance blur/flatness, and HSV skin reflection co-occurrence).
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try importing onnxruntime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning("onnxruntime not installed. Falling back to multi-factor texture analysis.")


class AntiSpoofDetector:
    """Anti-Spoofing & Liveness Detector for Face Attendance System."""

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
        self.session = None
        self._init_onnx_model()

    def _init_onnx_model(self):
        """Initialize MiniFASNet V2 ONNX session if available."""
        if not HAS_ONNX:
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'ml_models', 'minifasnet_v2.onnx')

        if os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                logger.info(f"Loaded MiniFASNet V2 ONNX model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                self.session = None
        else:
            logger.info(f"ONNX model asset not found at {model_path}. Using multi-factor texture & frequency analyzer.")

    def check_texture_and_moire(self, face_bgr: np.ndarray) -> float:
        """
        Analyze microscopic texture, Moiré screen lines, and color reflections.
        Returns a score in [0.0, 1.0] where 1.0 = Live Human, 0.0 = Printed Photo/Screen.
        """
        if face_bgr is None or face_bgr.size == 0:
            return 0.0

        h, w = face_bgr.shape[:2]
        if h < 20 or w < 20:
            return 0.0

        # 1. Laplacian Variance (Focus / Blur & Texture Sharpness)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Flat printed cards or smooth screens have low or artificially uniform variance
        # Live faces typically have dynamic texture variation between 100 and 1500
        norm_lap = min(1.0, max(0.0, lap_var / 350.0))

        # 2. Moiré / High-Frequency Screen Pattern Analysis (2D Discrete Fourier Transform)
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-5)

        # High frequency content ratio (screens exhibit repetitive grid spikes in spectrum)
        center_h, center_w = h // 2, w // 2
        r = min(center_h, center_w) // 4
        mask = np.ones((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_w, center_h), r, 0, -1)
        
        high_freq_power = np.mean(magnitude_spectrum[mask == 1])
        low_freq_power = np.mean(magnitude_spectrum[mask == 0]) + 1e-5
        freq_ratio = high_freq_power / low_freq_power

        # Screen spoofs have disproportionately high grid frequency energy (ratio > 1.2)
        freq_score = 1.0 if freq_ratio < 1.15 else max(0.1, 1.0 - (freq_ratio - 1.15) * 3)

        # 3. Color Space & Specular Reflection (HSV Channel Statistics)
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        sat_mean, sat_std = np.mean(sat), np.std(sat)
        val_std = np.std(val)

        # Plastic ID cards & printed photos have lower color saturation variance (sat_std < 15)
        sat_score = min(1.0, max(0.2, sat_std / 30.0))

        # Combine multi-factor texture signals
        combined_texture_score = (norm_lap * 0.35) + (freq_score * 0.40) + (sat_score * 0.25)
        return float(np.clip(combined_texture_score, 0.0, 1.0))

    def predict(self, face_bgr: np.ndarray, full_frame_bgr: np.ndarray | None = None) -> tuple[bool, float, str]:
        """
        Predict whether a face crop is a Live Human or a Spoof (ID card/photo/screen).

        Parameters
        ----------
        face_bgr : np.ndarray
            Cropped face image.
        full_frame_bgr : np.ndarray, optional
            Full frame image for wider context analysis.

        Returns
        -------
        (is_real: bool, confidence_score: float, reason: str)
        """
        if face_bgr is None or face_bgr.size == 0:
            return False, 0.0, "Invalid face image"

        # 1. Use ONNX MiniFASNet Model if available
        if self.session is not None:
            try:
                # MiniFASNet V2 expects 80x80 RGB input in (1, 3, 80, 80) float32 [0, 1]
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (80, 80))
                face_tensor = np.transpose(face_resized, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = np.expand_dims(face_tensor, axis=0)

                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: face_tensor})
                raw_logits = outputs[0][0]

                # Softmax probabilities
                exp_logits = np.exp(raw_logits - np.max(raw_logits))
                probs = exp_logits / np.sum(exp_logits)
                real_score = float(probs[1]) if len(probs) > 1 else float(probs[0])

                is_real = real_score >= self.threshold
                reason = "Live human face verified" if is_real else "Spoof attempt detected (ID card, photo, or screen)"
                return is_real, real_score, reason

            except Exception as e:
                logger.error(f"ONNX inference error: {e}. Falling back to texture analyzer.")

        # 2. Fallback to Multi-Factor Texture & Fourier Frequency Analyzer
        texture_score = self.check_texture_and_moire(face_bgr)
        is_real = texture_score >= self.threshold
        reason = "Live human face verified" if is_real else "Spoof attempt detected (Static photo / ID card reflection)"

        return is_real, texture_score, reason


# ── Module Singleton ─────────────────────────────────────────────────────────
_detector_instance: AntiSpoofDetector | None = None


def get_anti_spoof_detector(threshold: float = 0.65) -> AntiSpoofDetector:
    """Return a shared AntiSpoofDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AntiSpoofDetector(threshold=threshold)
    return _detector_instance
