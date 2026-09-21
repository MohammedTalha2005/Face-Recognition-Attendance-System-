"""
routes/face_recognition.py
--------------------------
Face registration (capture) and recognition endpoints.

Stack:
  Detection  → MediaPipe  (replaces Haar Cascade)
  Embeddings → FaceNet    (replaces LBPH + classifier.xml)
  Storage    → PostgreSQL FaceEmbedding table (replaces data/*.jpg + ml_models/classifier.xml)

There is NO train-model step anymore — embeddings are computed at capture time.
"""

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from models import db, Student, Attendance
import cv2
import numpy as np
import base64
import logging
from datetime import datetime
from config import Config

# Import new modules
from face_detection import get_detector
from anti_spoofing import get_anti_spoof_detector
from vector_store import get_vector_store
from face_recognition_engine import (
    generate_embedding,
    average_embeddings,
    find_best_match,
)

face_bp = Blueprint('face', __name__)
logger = logging.getLogger(__name__)


# ── Config endpoint ──────────────────────────────────────────────────────────

@face_bp.route('/config')
@login_required
def get_config():
    """Get face recognition configuration."""
    return jsonify({
        'face_samples_count': Config.FACE_SAMPLES_COUNT,
        'face_similarity_threshold': Config.FACE_SIMILARITY_THRESHOLD,
        'engine': 'FaceNet + MediaPipe'
    })


@face_bp.route('/')
@login_required
def index():
    """Face recognition page."""
    return render_template('face_recognition.html')


# ── Helpers ──────────────────────────────────────────────────────────────────

def _decode_base64_image(b64_str: str) -> np.ndarray | None:
    """Decode a base64 data-URL into a BGR numpy array."""
    try:
        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]
        img_data = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Base64 decode failed: {e}")
        return None


# ── Registration: capture photos → generate & store embedding ────────────────

@face_bp.route('/capture-photos/<int:student_id>', methods=['POST'])
@login_required
def capture_photos(student_id):
    """
    Accept 15–20 webcam frames, detect faces with MediaPipe, generate FaceNet
    embeddings, average them, and store one representative embedding in PostgreSQL.

    No JPEG files saved. No model training needed.
    """
    student = Student.query.get_or_404(student_id)

    try:
        image_data = request.json.get('images', [])
        if not image_data:
            return jsonify({'success': False, 'message': 'No images provided'})

        detector = get_detector()
        embeddings_collected = []
        frames_processed = 0
        frames_with_face = 0

        for idx, img_b64 in enumerate(image_data):
            img = _decode_base64_image(img_b64)
            if img is None:
                logger.warning(f"Frame {idx}: could not decode — skipping")
                continue
            frames_processed += 1

            # MediaPipe face detection
            face_crops = detector.crop_faces(img, padding=0.15)
            if not face_crops:
                logger.debug(f"Frame {idx}: no face detected")
                continue

            if len(face_crops) > 1:
                logger.warning(f"Frame {idx}: multiple faces detected ({len(face_crops)}) — skipping to prevent profile contamination")
                continue

            frames_with_face += 1
            # Single verified face
            face_bgr, _ = face_crops[0]

            # FaceNet embedding
            emb = generate_embedding(face_bgr)
            if emb is not None:
                embeddings_collected.append(emb)

        if not embeddings_collected:
            return jsonify({
                'success': False,
                'message': (
                    f'Could not extract face embeddings. '
                    f'{frames_with_face}/{frames_processed} frames had a face detected. '
                    f'Please ensure good lighting, face the camera directly, and ensure only one person is in frame.'
                )
            })

        # Average all valid embeddings → one representative vector
        avg_emb = average_embeddings(embeddings_collected)
        if avg_emb is None:
            return jsonify({'success': False, 'message': 'Failed to compute average embedding'})

        # Upsert 512-D master vector directly into ChromaDB Vector Store
        vector_store = get_vector_store()
        vector_store.upsert_student(
            student_id=student.student_id,
            embedding=avg_emb,
            metadata={'name': student.name, 'roll_no': student.roll_no, 'department': student.department}
        )

        # Mark student as having face data
        student.photo_sample = 'Yes'
        db.session.commit()

        logger.info(
            f"Registered student {student.student_id}: "
            f"{len(embeddings_collected)} embeddings averaged."
        )

        return jsonify({
            'success': True,
            'message': (
                f'Face registered successfully! '
                f'Generated embedding from {len(embeddings_collected)} frames '
                f'({frames_with_face} faces detected out of {frames_processed} frames).'
            ),
            'count': len(embeddings_collected)
        })

    except Exception as e:
        db.session.rollback()
        logger.exception(f"capture_photos error for student {student_id}")
        return jsonify({'success': False, 'message': str(e)})


# ── Recognition: live frame → identify student → mark attendance ─────────────

@face_bp.route('/recognize', methods=['POST'])
@login_required
def recognize():
    """
    Accept one webcam frame, detect face with MediaPipe, generate FaceNet
    embedding, compare against all stored embeddings, mark attendance.
    """
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Invalid request format'})

        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'})

        # Decode frame
        img = _decode_base64_image(image_data)
        if img is None:
            return jsonify({'success': False, 'message': 'Invalid image format'})

        # MediaPipe detection
        detector = get_detector()
        face_crops = detector.crop_faces(img, padding=0.15)

        if not face_crops:
            return jsonify({
                'success': False,
                'message': 'No face detected. Please face the camera directly in good lighting.'
            })

        # Strict Single-Person Gate: reject if more than one person is in the frame
        if len(face_crops) > 1:
            return jsonify({
                'success': False,
                'message': f'Multiple faces detected ({len(face_crops)} people in frame). Please step forward one person at a time.'
            })

        # Check ChromaDB Vector Store
        vector_store = get_vector_store()
        if vector_store.get_count() == 0:
            return jsonify({
                'success': False,
                'message': 'No students registered yet. Please register students first.'
            })

        anti_spoof_detector = get_anti_spoof_detector(threshold=Config.ANTI_SPOOF_THRESHOLD)
        recognized_students = []
        spoof_detected = False

        for face_bgr, _ in face_crops:
            # Anti-Spoofing / Liveness Check
            is_real, liveness_score, reason = anti_spoof_detector.predict(face_bgr, img)
            if not is_real:
                logger.warning(f"Spoof detected (score: {liveness_score:.3f}): {reason}")
                spoof_detected = True
                continue

            query_emb = generate_embedding(face_bgr)
            if query_emb is None:
                continue

            # Query ChromaDB vector database in < 1ms
            match = vector_store.find_best_match(
                query_emb,
                threshold=Config.FACE_SIMILARITY_THRESHOLD
            )

            if match is None:
                continue

            student = Student.query.filter_by(student_id=match['student_id']).first()
            if not student:
                logger.warning(f"Match found for unknown student_id: {match['student_id']}")
                continue

            # Mark attendance
            today = datetime.now().strftime('%Y-%m-%d')
            existing = Attendance.query.filter_by(
                student_id=student.student_id,
                date=today
            ).first()

            if not existing:
                attendance = Attendance(
                    student_id=student.student_id,
                    roll_no=student.roll_no,
                    name=student.name,
                    department=student.department,
                    time=datetime.now().strftime('%H:%M:%S'),
                    date=today,
                    status='Present'
                )
                db.session.add(attendance)
                db.session.commit()
                logger.info(f"Attendance marked: {student.name} ({student.student_id})")

            recognized_students.append({
                'student_id': student.student_id,
                'name': student.name,
                'roll_no': student.roll_no,
                'department': student.department,
                'confidence': match['confidence'],
                'attendance_marked': not existing,
                'message': 'Attendance marked' if not existing else 'Already marked today'
            })

        if recognized_students:
            return jsonify({'success': True, 'students': recognized_students})
        elif spoof_detected:
            return jsonify({
                'success': False,
                'is_spoof': True,
                'message': (
                    '⚠️ Spoof attempt detected! '
                    'Please present a live human face, not an ID card, paper photo, or screen.'
                )
            })
        else:
            return jsonify({
                'success': False,
                'message': (
                    'Face detected but not recognized. '
                    'Please ensure the student is registered and try again with better lighting.'
                )
            })

    except Exception as e:
        db.session.rollback()
        logger.exception("recognize() error")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'})


# ── Reset embeddings (re-registration utility) ───────────────────────────────

@face_bp.route('/reset-embeddings/<int:student_id>', methods=['POST'])
@login_required
def reset_embeddings(student_id):
    """Delete stored embeddings for a student so they can re-register."""
    student = Student.query.get_or_404(student_id)
    try:
        get_vector_store().delete_student(student.student_id)
        student.photo_sample = 'No'
        db.session.commit()
        return jsonify({
            'success': True,
            'message': f'Cleared embedding(s) for {student.name}. Student can now re-register.'
        })
    except Exception as e:
        db.session.rollback()
        logger.exception(f"reset_embeddings error for student {student_id}")
        return jsonify({'success': False, 'message': str(e)})


# ── Embedding status ─────────────────────────────────────────────────────────

@face_bp.route('/embedding-status/<int:student_id>')
@login_required
def embedding_status(student_id):
    """Check if a student has face embeddings registered."""
    student = Student.query.get_or_404(student_id)
    has_sample = (student.photo_sample == 'Yes')
    return jsonify({
        'student_id': student.student_id,
        'name': student.name,
        'has_embedding': has_sample,
        'embedding_count': 1 if has_sample else 0
    })
