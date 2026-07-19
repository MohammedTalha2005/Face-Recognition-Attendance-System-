"""
face_detection.py
-----------------
MediaPipe Tasks-based face detector.
Returns bounding boxes as (x, y, w, h) tuples — same interface as Haar,
so the rest of the code needs minimal changes.
"""

import os
import cv2
import numpy as np
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)


class FaceDetector:
    """Wraps MediaPipe Tasks Face Detection for easy reuse."""

    def __init__(self, min_detection_confidence: float = 0.5):
        # Locate the blaze_face_short_range.tflite model in ml_models
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'ml_models', 'blaze_face_short_range.tflite')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe Face Detector model not found at: {model_path}")

        logger.info(f"Initializing MediaPipe Tasks FaceDetector with model: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, img_bgr: np.ndarray) -> list:
        """
        Run MediaPipe face detection on a BGR image.

        Returns
        -------
        list of (x, y, w, h) integer tuples in pixel coordinates.
        Returns an empty list if no faces are found.
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        try:
            h, w = img_bgr.shape[:2]

            # MediaPipe Tasks expects mp.Image in RGB format
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            results = self._detector.detect(mp_image)

            boxes = []
            if results.detections:
                for det in results.detections:
                    bbox = det.bounding_box
                    x = max(0, int(bbox.origin_x))
                    y = max(0, int(bbox.origin_y))
                    bw = int(bbox.width)
                    bh = int(bbox.height)

                    # Clamp to image bounds
                    x2 = min(x + bw, w)
                    y2 = min(y + bh, h)
                    bw = max(0, x2 - x)
                    bh = max(0, y2 - y)

                    if bw > 0 and bh > 0:
                        boxes.append((x, y, bw, bh))

            return boxes
        except Exception as e:
            logger.error(f"MediaPipe Tasks face detection failed: {e}")
            return []

    def crop_faces(self, img_bgr: np.ndarray, padding: float = 0.1) -> list:
        """
        Detect and return cropped face images (BGR) with optional padding.

        Parameters
        ----------
        img_bgr : np.ndarray   Input BGR image.
        padding : float        Fractional padding around the bounding box (0.1 = 10%).

        Returns
        -------
        list of (face_bgr, (x, y, w, h)) tuples.
        """
        boxes = self.detect(img_bgr)
        h, w = img_bgr.shape[:2]
        crops = []
        for (x, y, bw, bh) in boxes:
            pad_x = int(bw * padding)
            pad_y = int(bh * padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            crops.append((img_bgr[y1:y2, x1:x2], (x, y, bw, bh)))
        return crops

    def close(self):
        if hasattr(self, '_detector') and hasattr(self._detector, 'close'):
            try:
                self._detector.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── module-level singleton (lazy-initialized) ────────────────────────────────
_detector_instance: FaceDetector | None = None


def get_detector() -> FaceDetector:
    """Return a shared FaceDetector instance (created once per process)."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector(min_detection_confidence=0.5)
    return _detector_instance

