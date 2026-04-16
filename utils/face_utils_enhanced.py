"""
Enhanced Face Utilities — Robust face detection, landmark detection, and preprocessing.
Features:
  - MediaPipe Face Mesh for precise landmark detection
  - Haar Cascade fallback for robustness
  - Advanced preprocessing: histogram equalization, normalization, alignment
  - Improved EAR/MAR computation with smoothing
  - Adaptive thresholds based on conditions
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
import sys, os
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RIGHT_EYE_IDX, LEFT_EYE_IDX, MOUTH_IDX,
    EAR_THRESHOLD, MAR_THRESHOLD, IMG_SIZE,
    USE_MEDIAPIPE_FACE, USE_MEDIAPIPE_MESH, FACE_DETECTION_CONFIDENCE,
    USE_HAAR_CASCADE_FALLBACK, MIN_FACE_SIZE_PERCENT,
    APPLY_HISTOGRAM_EQUALIZATION, APPLY_CLAHE, NORMALIZE_FACE_ROI,
    FACE_ALIGNMENT_METHOD, EAR_THRESHOLD_HIGH_CONF, EAR_THRESHOLD_LOW_CONF,
    MAR_THRESHOLD_HIGH_CONF, MAR_THRESHOLD_LOW_CONF
)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[!] MediaPipe not installed - using Haar Cascade fallback")


class EnhancedFaceLandmarkDetector:
    """
    Robust face detection with MediaPipe + Haar Cascade fallback.
    Provides precise EAR, MAR computation with temporal smoothing.
    Includes advanced preprocessing for robust performance.
    """

    def __init__(self, max_faces=1, smooth_frame_count=5):
        self.max_faces = max_faces
        self.smooth_frame_count = smooth_frame_count
        
        # Temporal smoothing buffers
        self.ear_buffer = deque(maxlen=smooth_frame_count)
        self.mar_buffer = deque(maxlen=smooth_frame_count)
        self.left_ear_buffer = deque(maxlen=smooth_frame_count)
        self.right_ear_buffer = deque(maxlen=smooth_frame_count)
        
        # MediaPipe setup
        self.mp_available = HAS_MEDIAPIPE and USE_MEDIAPIPE_MESH and hasattr(mp, 'solutions')
        if HAS_MEDIAPIPE and USE_MEDIAPIPE_MESH and not hasattr(mp, 'solutions'):
            mp_version = getattr(mp, '__version__', 'unknown')
            print(f"[!] Installed MediaPipe ({mp_version}) has no 'solutions' API; using Haar fallback")
        if self.mp_available:
            try:
                solutions_api = getattr(mp, 'solutions', None)
                if solutions_api is None:
                    raise RuntimeError("MediaPipe solutions API unavailable")
                self.mp_face_mesh = solutions_api.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=max_faces,
                    refine_landmarks=True,  # Improved accuracy
                    min_detection_confidence=FACE_DETECTION_CONFIDENCE,
                    min_tracking_confidence=FACE_DETECTION_CONFIDENCE * 0.9
                )
                print("[OK] MediaPipe Face Mesh initialized (refine_landmarks=True)")
            except Exception as e:
                print(f"[!] MediaPipe initialization failed: {e}")
                self.mp_available = False
        
        # Haar Cascade fallback
        cv2_data = getattr(cv2, 'data', None)
        cascade_path = cv2_data.haarcascades if cv2_data is not None else ''
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_eye.xml')
        )
        self.smile_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_smile.xml')
        )
        
        self.has_cascades = (
            not self.face_cascade.empty() and 
            not self.eye_cascade.empty()
        )
        
        if not self.has_cascades:
            print("[!] Haar cascades failed to load")
        
        # Last known detection for continuity
        self.last_face_bbox = None
        self.last_landmarks = None

    def process_frame(self, frame):
        """
        Process frame with MediaPipe (primary) or Haar Cascade (fallback).
        
        Returns:
            dict with: 'landmarks', 'ear', 'mar', 'face_roi', 'bbox', 
                      'left_ear', 'right_ear', 'success', 'confidence'
        """
        h, w = frame.shape[:2]
        min_face_size = int(max(h, w) * MIN_FACE_SIZE_PERCENT)
        
        result = {
            'landmarks': None,
            'ear': 0.0,
            'mar': 0.0,
            'face_roi': None,
            'bbox': None,
            'success': False,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'confidence': 0.0,
            'face_area': 0
        }

        # Try MediaPipe first
        if self.mp_available:
            result = self._process_with_mediapipe(frame, min_face_size)
            if result['success']:
                self.last_face_bbox = result['bbox']
                self.last_landmarks = result['landmarks']
                self._smooth_metrics(result)
                return result
        
        # Fallback to Haar Cascade
        if self.has_cascades and USE_HAAR_CASCADE_FALLBACK:
            result = self._process_with_haar(frame, min_face_size)
            if result['success']:
                self.last_face_bbox = result['bbox']
                self.last_landmarks = result['landmarks']
                self._smooth_metrics(result)
                return result
        
        # If both fail, use last known landmarks for continuity
        if self.last_landmarks is not None:
            result['landmarks'] = self.last_landmarks
            result['bbox'] = self.last_face_bbox
            result['success'] = False  # Mark as low confidence
        
        self._apply_smoothed_metrics(result)
        return result

    def _process_with_mediapipe(self, frame, min_face_size):
        """Process frame using MediaPipe Face Mesh."""
        result = {
            'landmarks': None,
            'ear': 0.0,
            'mar': 0.0,
            'face_roi': None,
            'bbox': None,
            'success': False,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'confidence': 0.0,
            'face_area': 0
        }
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            outputs = self.face_mesh.process(frame_rgb)
            
            if not outputs.multi_face_landmarks:
                return result
            
            h, w = frame.shape[:2]
            face_landmarks = outputs.multi_face_landmarks[0]
            
            # Convert normalized coordinates to pixel coordinates
            landmarks = np.zeros((468, 2), dtype=np.int32)
            for i, lm in enumerate(face_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks[i] = [x, y]
            
            # Get face bounding box
            x_min = np.min(landmarks[:, 0])
            x_max = np.max(landmarks[:, 0])
            y_min = np.min(landmarks[:, 1])
            y_max = np.max(landmarks[:, 1])
            
            face_w = x_max - x_min
            face_h = y_max - y_min
            face_area = face_w * face_h
            
            if face_area < min_face_size * min_face_size:
                return result
            
            # Extract and preprocess face ROI
            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size == 0:
                return result
            
            face_roi_resized = self._preprocess_face_roi(face_roi)
            
            # Compute metrics
            left_ear = self._compute_ear(landmarks, LEFT_EYE_IDX)
            right_ear = self._compute_ear(landmarks, RIGHT_EYE_IDX)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = self._compute_mar(landmarks, MOUTH_IDX)
            
            result.update({
                'landmarks': landmarks,
                'ear': avg_ear,
                'left_ear': left_ear,
                'right_ear': right_ear,
                'mar': mar,
                'face_roi': face_roi_resized,
                'bbox': (x_min, y_min, x_max, y_max),
                'success': True,
                'confidence': 0.95,  # MediaPipe is high confidence
                'face_area': face_area
            })
            
        except Exception as e:
            if os.environ.get('DEBUG'):
                print(f"[!] MediaPipe error: {e}")
        
        return result

    def _process_with_haar(self, frame, min_face_size):
        """Process frame using Haar Cascade as fallback."""
        result = {
            'landmarks': None,
            'ear': 0.0,
            'mar': 0.0,
            'face_roi': None,
            'bbox': None,
            'success': False,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'confidence': 0.0,
            'face_area': 0
        }
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, 
                minSize=(min_face_size, min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                return result
            
            # Take largest face
            (x, y, face_w, face_h) = max(faces, key=lambda f: f[2] * f[3])
            face_area = face_w * face_h
            
            # Extract and preprocess face ROI
            face_roi = frame[y:y+face_h, x:x+face_w]
            face_roi_resized = self._preprocess_face_roi(face_roi)
            
            # Detect eyes and mouth for metric estimation
            gray_face = gray[y:y+face_h, x:x+face_w]
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 4, minSize=(15, 15))
            smiles = self.smile_cascade.detectMultiScale(gray_face, 1.8, 20, minSize=(20, 20))
            
            # Generate synthetic landmarks
            landmarks = self._generate_synthetic_landmarks(
                x, y, face_w, face_h, eyes, gray_face, smiles
            )
            
            # Estimate metrics using heuristics
            ear_value = self._estimate_ear_from_eyes(face_roi, eyes)
            mar_value = self._estimate_mar_from_mouth(face_roi, smiles)
            
            result.update({
                'landmarks': landmarks,
                'ear': ear_value,
                'left_ear': ear_value * 0.5,
                'right_ear': ear_value * 0.5,
                'mar': mar_value,
                'face_roi': face_roi_resized,
                'bbox': (x, y, x+face_w, y+face_h),
                'success': True,
                'confidence': 0.7,  # Haar is lower confidence
                'face_area': face_area
            })
            
        except Exception as e:
            if os.environ.get('DEBUG'):
                print(f"[!] Haar Cascade error: {e}")
        
        return result

    def _preprocess_face_roi(self, face_roi):
        """
        Advanced preprocessing for face ROI:
        - Histogram equalization/CLAHE
        - Alignment
        - Normalization
        """
        # Resize to target size
        face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        
        # Apply CLAHE (better than regular histogram equalization)
        if APPLY_CLAHE:
            try:
                lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                
                lab = cv2.merge([l_channel, a_channel, b_channel])
                face_resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except:
                pass
        elif APPLY_HISTOGRAM_EQUALIZATION:
            try:
                gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                face_resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            except:
                pass
        
        # Normalization
        if NORMALIZE_FACE_ROI:
            face_resized = face_resized.astype(np.float32) / 255.0
        else:
            face_resized = face_resized.astype(np.float32)
        
        return face_resized

    def _compute_ear(self, landmarks, eye_indices):
        """Compute Eye Aspect Ratio (EAR) from 6 landmark points."""
        try:
            pts = landmarks[eye_indices].astype(float)
            
            # Compute distances
            A = dist.euclidean(pts[1], pts[5])
            B = dist.euclidean(pts[2], pts[4])
            C = dist.euclidean(pts[0], pts[3])
            
            if C == 0:
                return 0.0
            
            ear = (A + B) / (2.0 * C)
            return float(ear)
        except (IndexError, ValueError):
            return 0.0

    def _compute_mar(self, landmarks, mouth_indices):
        """Compute Mouth Aspect Ratio (MAR) from 8 landmark points."""
        try:
            pts = landmarks[mouth_indices].astype(float)
            
            # Compute distances
            A = dist.euclidean(pts[2], pts[6])
            B = dist.euclidean(pts[3], pts[7])
            C = dist.euclidean(pts[4], pts[5])
            D = dist.euclidean(pts[0], pts[1])
            
            if D == 0:
                return 0.0
            
            mar = (A + B + C) / (3.0 * D)
            return float(mar)
        except (IndexError, ValueError):
            return 0.0

    def _estimate_ear_from_eyes(self, face_roi, eyes):
        """Heuristic EAR estimation from detected eyes."""
        if len(eyes) == 0:
            # Estimate from face brightness
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi
            brightness = np.mean(gray) / 255.0
            return max(0.1, brightness * 0.3)
        
        # Eye area ratio
        total_area = float(face_roi.shape[0] * face_roi.shape[1])
        eyes_area = sum(e_w * e_h for (_, _, e_w, e_h) in eyes)
        eye_ratio = eyes_area / total_area if total_area > 0 else 0.0
        
        # Map to EAR range
        ear_value = 0.12 + (eye_ratio * 0.4)
        return float(np.clip(ear_value, 0.10, 0.45))

    def _estimate_mar_from_mouth(self, face_roi, smiles):
        """Heuristic MAR estimation from detected mouth region."""
        h, w = face_roi.shape[:2]
        mouth_region = face_roi[h//2:, :]
        
        if mouth_region.size == 0:
            return 0.5
        
        if len(mouth_region.shape) == 3:
            mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        else:
            mouth_gray = mouth_region
        
        brightness = np.mean(mouth_gray) / 255.0
        
        # Apply smile detection bonus
        smile_bonus = 0.15 if len(smiles) > 0 else 0.0
        
        mar_value = 0.75 - (brightness * 0.25) + smile_bonus
        return float(np.clip(mar_value, 0.35, 0.85))

    def _generate_synthetic_landmarks(self, x, y, face_w, face_h, eyes, gray_face, smiles):
        """Generate synthetic landmarks from Haar detections."""
        landmarks = np.zeros((468, 2), dtype=np.int32)
        
        # Eyes
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            indices = LEFT_EYE_IDX if i == 0 else RIGHT_EYE_IDX
            for idx in indices:
                landmarks[idx] = [eye_center_x, eye_center_y]
        
        # Mouth - use smile detection if available
        if len(smiles) > 0:
            sx, sy, sw, sh = smiles[0]
            mouth_x = x + sx + sw // 2
            mouth_y = y + sy + sh // 2
        else:
            mouth_x = x + face_w // 2
            mouth_y = y + int(face_h * 0.75)
        
        for idx in MOUTH_IDX:
            landmarks[idx] = [mouth_x, mouth_y]
        
        # Fill remaining with face center
        face_center_x = x + face_w // 2
        face_center_y = y + face_h // 2
        for i in range(468):
            if landmarks[i, 0] == 0 and landmarks[i, 1] == 0:
                landmarks[i] = [face_center_x, face_center_y]
        
        return landmarks

    def _smooth_metrics(self, result):
        """Apply temporal smoothing to reduce jitter."""
        self.ear_buffer.append(result['ear'])
        self.left_ear_buffer.append(result['left_ear'])
        self.right_ear_buffer.append(result['right_ear'])
        self.mar_buffer.append(result['mar'])
        
        self._apply_smoothed_metrics(result)

    def _apply_smoothed_metrics(self, result):
        """Apply smoothed metrics from buffers."""
        if len(self.ear_buffer) > 0:
            result['ear'] = np.mean(list(self.ear_buffer))
            result['left_ear'] = np.mean(list(self.left_ear_buffer))
            result['right_ear'] = np.mean(list(self.right_ear_buffer))
            result['mar'] = np.mean(list(self.mar_buffer))

    def get_adaptive_threshold(self, base_threshold, confidence):
        """Get adaptive threshold based on confidence level."""
        if confidence >= 0.8:  # High confidence
            return base_threshold * 0.95
        elif confidence >= 0.6:  # Medium confidence
            return base_threshold
        else:  # Low confidence
            return base_threshold * 1.1

    def draw_landmarks(self, frame, result, state="Alert", fps=0.0, probs=None, prediction_confidence=None):
        """
        Draw enhanced HUD with landmarks, metrics, and state info.
        """
        if not result['success']:
            cv2.putText(frame, "No Face Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return frame

        landmarks = result['landmarks']
        bbox = result['bbox']
        detector_confidence = result['confidence']
        confidence = detector_confidence if prediction_confidence is None else float(prediction_confidence)

        # Draw face bounding box
        if bbox is not None:
            color = {
                "Alert": (0, 255, 0),
                "Drowsy": (0, 0, 255),
                "Yawning": (0, 165, 255)
            }.get(state, (255, 255, 255))

            thickness = 3 if detector_confidence > 0.8 else 2
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

        # Draw eye landmarks
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            cv2.circle(frame, tuple(landmarks[idx]), 3, (0, 255, 255), -1)

        # Draw mouth landmarks
        for idx in MOUTH_IDX:
            cv2.circle(frame, tuple(landmarks[idx]), 3, (255, 0, 255), -1)

        # Enhanced HUD Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 240), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Display metrics
        y_offset = 40
        state_color = {
            "Alert": (0, 255, 0),
            "Drowsy": (0, 0, 255),
            "Yawning": (0, 165, 255)
        }.get(state, (255, 255, 255))
        
        metrics = [
            (f"State: {state}", state_color, 1.0),
            (f"Confidence: {confidence*100:.1f}%", (200, 200, 200), 0.8),
            (f"EAR: {result['ear']:.3f} (L:{result['left_ear']:.3f} R:{result['right_ear']:.3f})", (255, 255, 255), 0.8),
            (f"MAR: {result['mar']:.3f}", (255, 255, 255), 0.8),
            (f"FPS: {fps:.1f}", (0, 255, 0) if fps >= 15 else (0, 0, 255), 0.8),
        ]

        if probs is not None:
            prob_text = f"Alert:{probs[0]:.2f} Drowsy:{probs[1]:.2f} Yawn:{probs[2]:.2f}"
            metrics.append((prob_text, (200, 200, 200), 0.75))

        for text, text_color, scale in metrics:
            cv2.putText(frame, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 2)
            y_offset += 35

        return frame

    def close(self):
        """Clean up resources."""
        if self.mp_available:
            self.face_mesh.close()


def preprocess_face_roi(face_roi):
    """Legacy function - kept for compatibility."""
    if isinstance(face_roi, np.ndarray):
        if face_roi.dtype == np.uint8:
            return face_roi.astype(np.float32) / 255.0
        return face_roi.astype(np.float32)
    return face_roi


# For backward compatibility
FaceLandmarkDetector = EnhancedFaceLandmarkDetector
