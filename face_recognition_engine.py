"""
face_recognition_engine.py
---------------------------
FaceNet-based face recognition engine.

Replaces OpenCV LBPH + classifier.xml with:
  - keras-facenet for 512-D embeddings
  - Cosine similarity for matching
  - Embeddings stored in PostgreSQL (FaceEmbedding table)

No retraining required — new students are registered instantly.
"""

import numpy as np
import cv2
import json
import logging

logger = logging.getLogger(__name__)

# ── FaceNet model singleton ──────────────────────────────────────────────────
_facenet_model = None


def get_facenet():
    """Lazy-load FaceNet model (loaded once per process)."""
    global _facenet_model
    if _facenet_model is None:
        try:
            from keras_facenet import FaceNet
            _facenet_model = FaceNet()
            logger.info("FaceNet model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FaceNet model: {e}")
            raise
    return _facenet_model


# ── Embedding generation ─────────────────────────────────────────────────────

def generate_embedding(face_img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Generate a 512-D FaceNet embedding for a single face crop.

    Parameters
    ----------
    face_img_bgr : np.ndarray
        BGR face crop (any size; will be resized to 160×160 internally).

    Returns
    -------
    np.ndarray of shape (512,) or None on failure.
    """
    try:
        model = get_facenet()

        # FaceNet expects RGB, shape (H, W, 3)
        face_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to 160×160 (FaceNet input size)
        face_resized = cv2.resize(face_rgb, (160, 160))

        # embeddings() expects a list/array of images: shape (N, H, W, 3)
        faces_array = np.expand_dims(face_resized, axis=0)  # (1, 160, 160, 3)
        embeddings = model.embeddings(faces_array)           # (1, 512)

        emb = embeddings[0]  # (512,)

        # L2-normalize so cosine similarity == dot product
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return emb / norm

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray | None:
    """
    Average a list of normalized embeddings into one representative embedding.

    Returns a normalized (512,) array, or None if the list is empty.
    """
    valid = [e for e in embeddings if e is not None and e.shape == (512,)]
    if not valid:
        return None
    avg = np.mean(valid, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


# ── Serialization helpers ────────────────────────────────────────────────────

def embedding_to_json(embedding: np.ndarray) -> str:
    """Serialize a (512,) numpy array to a JSON string for DB storage."""
    return json.dumps(embedding.tolist())


def json_to_embedding(json_str: str) -> np.ndarray:
    """Deserialize a JSON string back to a (512,) numpy array."""
    return np.array(json.loads(json_str), dtype=np.float32)


# ── Similarity & matching ────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized vectors.
    Returns a value in [-1, 1]; higher = more similar.
    Since embeddings are already L2-normalized, this is just the dot product.
    """
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two embedding vectors (lower = more similar)."""
    return float(np.linalg.norm(a - b))


def find_best_match(
    query_embedding: np.ndarray,
    stored_embeddings: list[dict],
    threshold: float = 0.6,
    metric: str = "cosine"
) -> dict | None:
    """
    Compare query_embedding against all stored embeddings and return the best match.

    Parameters
    ----------
    query_embedding : np.ndarray
        (512,) embedding of the face to recognize.
    stored_embeddings : list of dict
        Each dict has keys: 'student_id', 'embedding' (np.ndarray or JSON str).
    threshold : float
        Cosine similarity threshold (0–1). Faces with similarity > threshold
        are considered a match. Default 0.6.
        For Euclidean: faces with distance < threshold are a match; use ~0.9.
    metric : str
        'cosine' (default) or 'euclidean'.

    Returns
    -------
    dict with keys: student_id, score, metric — or None if no match found.
    """
    best_score = -np.inf if metric == "cosine" else np.inf
    best_student_id = None

    for record in stored_embeddings:
        emb = record['embedding']
        if isinstance(emb, str):
            emb = json_to_embedding(emb)

        if metric == "cosine":
            score = cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_student_id = record['student_id']
        else:  # euclidean
            score = euclidean_distance(query_embedding, emb)
            if score < best_score:
                best_score = score
                best_student_id = record['student_id']

    if best_student_id is None:
        return None

    # Check threshold
    if metric == "cosine":
        match = best_score >= threshold
        # Convert cosine similarity (0–1) to a readable % confidence
        confidence_pct = int(best_score * 100)
    else:
        match = best_score <= threshold
        # Convert euclidean distance to confidence %: 0 dist = 100%, 2 dist = 0%
        confidence_pct = max(0, int((1 - best_score / 2) * 100))

    if not match:
        return None

    return {
        'student_id': best_student_id,
        'score': best_score,
        'confidence': confidence_pct,
        'metric': metric
    }
