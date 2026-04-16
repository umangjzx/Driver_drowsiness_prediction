"""
Enhanced Real-time Driver Drowsiness Detector

Features:
  - Temporal smoothing for stable predictions
  - Ensemble inference for robustness
  - Adaptive thresholds based on confidence
  - Improved alarm logic with hysteresis
  - Better HUD visualization with statistics
  - Performance monitoring
"""
import os, sys, time, collections, glob, inspect
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_DIR, SEQUENCE_LENGTH, IMG_SIZE, CLASS_NAMES,
    CAMERA_INDEX, ALARM_CONSECUTIVE_FRAMES, EAR_THRESHOLD, MAR_THRESHOLD,
    DISPLAY_WIDTH, DISPLAY_HEIGHT, USE_ENSEMBLE, ENSEMBLE_CONFIG,
    TEMPORAL_SMOOTHING_WINDOW, CONFIDENCE_THRESHOLD,
    EAR_THRESHOLD_HIGH_CONF, EAR_THRESHOLD_LOW_CONF,
    MAR_THRESHOLD_HIGH_CONF, MAR_THRESHOLD_LOW_CONF,
    ALARM_TRIGGER_THRESHOLD, ALARM_HYSTERESIS, ALARM_MIN_DURATION,
    PREDICT_EVERY_N_FRAMES, VERBOSE_MODE, LOG_METRICS_INTERVAL,
    USE_HUGGINGFACE, HF_MODEL_ID, HF_BACKEND, HF_CLASS_LABELS,
    HF_CONFIDENCE_THRESHOLD, CALIBRATION_TARGET_FRAMES,
    CALIBRATION_MIN_CONFIDENCE, EYE_CLOSED_STREAK_TRIGGER,
    EYE_CLOSED_STREAK_STRONG, EAR_BASELINE_RATIO_MEDIAPIPE,
    EAR_BASELINE_RATIO_HAAR, NO_FACE_RESET_SECONDS
)
from utils.face_utils_enhanced import EnhancedFaceLandmarkDetector, preprocess_face_roi
from utils.audio_alarm import AlarmSystem


class EnhancedDrowsinessDetector:
    """
    High-performance drowsiness detector with ensemble support and temporal smoothing.
    """
    
    def __init__(
        self,
        model_path=None,
        use_ensemble=False,
        use_temporal_smoothing=True,
        force_rule_based=False,
        use_huggingface=False,
        hf_model_id=None,
        hf_backend="auto",
    ):
        self.model_path = model_path or os.path.join(MODEL_DIR, 'best_model.keras')
        if not os.path.exists(self.model_path):
            self.model_path = os.path.join(MODEL_DIR, 'final_model.keras')
        
        self.use_ensemble = use_ensemble and USE_ENSEMBLE
        self.use_temporal_smoothing = use_temporal_smoothing
        self.force_rule_based = force_rule_based
        self.use_huggingface = bool(use_huggingface or USE_HUGGINGFACE)
        self.hf_model_id = hf_model_id or HF_MODEL_ID
        self.hf_backend = hf_backend or HF_BACKEND
        self.hf_classifier = None
        self.models = []
        
        # Load inference backend(s)
        if self.use_huggingface:
            self._load_huggingface_model()
        else:
            self._load_models()
        
        # Face detection
        self.detector = EnhancedFaceLandmarkDetector(max_faces=1)
        self.alarm = AlarmSystem()
        
        # Frame and metrics buffers
        self.frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
        self.ear_mar_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_buffer = collections.deque(maxlen=TEMPORAL_SMOOTHING_WINDOW)
        self.confidence_buffer = collections.deque(maxlen=TEMPORAL_SMOOTHING_WINDOW)
        
        # State tracking
        self.drowsy_counter = 0
        self.yawn_counter = 0
        self.alert_counter = 0
        self.current_state = "Alert"
        self.current_confidence = 0.0
        self.fps = 0.0
        self.prediction_history = collections.deque(maxlen=100)
        self.eye_closed_streak = 0
        self.personal_ear_baseline = None
        self.personal_mar_baseline = None
        self.calibration_complete = False
        self.calibration_ear_samples = collections.deque(maxlen=240)
        self.calibration_mar_samples = collections.deque(maxlen=240)
        self.last_face_seen_time = time.time()
        self.adaptive_ear_threshold = EAR_THRESHOLD
        self.adaptive_mar_threshold = MAR_THRESHOLD
        self.current_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Alarm state with hysteresis
        self.alarm_armed = False
        self.alarm_intensity = 0.0
        self.last_alarm_time = 0.0
        
        # Statistics
        self.frame_count = 0
        self.processing_times = collections.deque(maxlen=100)

        # If Face Mesh is unavailable (common on Python 3.13), use a safe fallback mode
        if not self.force_rule_based and not getattr(self.detector, 'mp_available', False):
            if self.use_huggingface and self.hf_classifier is not None:
                print("[!] Face Mesh unavailable; using Hugging Face inference with Haar landmark fallback")
            else:
                self.force_rule_based = True
                if len(self.models) > 0:
                    print("[!] Face Mesh unavailable; switching to rule-based inference for real-time stability")
                    print("[i] Tip: Python 3.11 + MediaPipe with solutions API gives best model-based behavior")

    def _load_models(self):
        """Load primary model and optional ensemble models."""
        try:
            import tensorflow as tf
            
            if os.path.exists(self.model_path):
                print(f"[i] Loading model: {self.model_path}")
                model = self._load_model_with_compat(tf, self.model_path)
                self.models.append(model)
                print("[OK] Model loaded")
                self._warmup_model(model)
            else:
                print(f"[!] Model not found at {self.model_path}")
                print("[i] Running in rule-based mode (EAR/MAR thresholds only)")
                return
            
            # Load ensemble models if enabled
            if self.use_ensemble:
                self._load_ensemble_models()
        
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            self.models = []

    def _load_huggingface_model(self):
        """Load Hugging Face zero-shot classifier backend."""
        try:
            from utils.huggingface_inference import HuggingFaceDrowsinessClassifier

            print(f"[i] Loading Hugging Face model: {self.hf_model_id}")
            self.hf_classifier = HuggingFaceDrowsinessClassifier(
                model_id=self.hf_model_id,
                class_labels=HF_CLASS_LABELS,
                confidence_threshold=HF_CONFIDENCE_THRESHOLD,
                backend=self.hf_backend,
            )
            self.use_huggingface = True
            print(f"[OK] Hugging Face backend ready ({self.hf_classifier.backend})")
        except Exception as e:
            print(f"[!] Hugging Face backend unavailable: {e}")
            print("[i] Falling back to TensorFlow / rule-based inference")
            self.use_huggingface = False
            self.hf_classifier = None
            self._load_models()

    def _load_ensemble_models(self):
        """Load additional models for ensemble inference."""
        try:
            import tensorflow as tf
            
            # Look for other trained models in MODEL_DIR
            model_files = glob.glob(os.path.join(MODEL_DIR, '*.keras'))
            model_files = [f for f in model_files if f != self.model_path]
            
            ensemble_count = 0
            for model_file in model_files[:ENSEMBLE_CONFIG['max_models'] - 1]:
                try:
                    print(f"[i] Loading ensemble model: {model_file}")
                    model = self._load_model_with_compat(tf, model_file)
                    self.models.append(model)
                    self._warmup_model(model)
                    ensemble_count += 1
                except Exception as e:
                    print(f"[!] Failed to load ensemble model: {e}")
            
            if ensemble_count > 0:
                print(f"[OK] Loaded {ensemble_count} ensemble model(s) (total: {len(self.models)})")
        
        except Exception as e:
            print(f"[!] Ensemble loading failed: {e}")

    def _warmup_model(self, model):
        """Warm up model with dummy inference."""
        try:
            dummy = np.zeros((1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            model.predict(dummy, verbose=0)
        except:
            pass

    @staticmethod
    def _compat_batchnorm_class(tf):
        """BatchNormalization wrapper to ignore legacy renorm kwargs during deserialization."""
        class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
            def __init__(self, *args, renorm=None, renorm_clipping=None, renorm_momentum=None, **kwargs):
                super().__init__(*args, **kwargs)

        return CompatBatchNormalization

    def _load_model_with_compat(self, tf, model_path):
        """Load model with compatibility fallbacks across Keras/TensorFlow versions."""
        load_signature = {}
        try:
            load_signature = inspect.signature(tf.keras.models.load_model).parameters
        except Exception:
            pass

        compat_bn = self._compat_batchnorm_class(tf)
        attempts = [
            ("default", None, False),
            ("default-no-safe-mode", None, None),
            ("compat-bn", {"BatchNormalization": compat_bn}, False),
            ("compat-bn-no-safe-mode", {"BatchNormalization": compat_bn}, None),
        ]

        errors = []
        for label, custom_objects, safe_mode in attempts:
            kwargs = {"compile": False}
            if custom_objects is not None:
                kwargs["custom_objects"] = custom_objects
            if safe_mode is not None and "safe_mode" in load_signature:
                kwargs["safe_mode"] = safe_mode

            try:
                model = tf.keras.models.load_model(model_path, **kwargs)
                if label != "default":
                    print(f"[i] Model loaded using compatibility mode: {label}")
                return model
            except Exception as e:
                errors.append(f"{label}: {e}")

        raise RuntimeError("All model loading attempts failed:\n" + "\n".join(errors))

    def predict(self, threshold=CONFIDENCE_THRESHOLD):
        """
        Make prediction with optional ensemble and smoothing.
        Returns: (state, probabilities, confidence)
        """
        required_frames = 1 if (self.use_huggingface and self.hf_classifier is not None) else SEQUENCE_LENGTH
        if len(self.frame_buffer) < required_frames:
            return "Alert", np.array([1.0, 0.0, 0.0]), 0.0

        if self.use_huggingface and self.hf_classifier is not None:
            state, probs, max_confidence = self.hf_classifier.predict(self.frame_buffer[-1])
            probs = np.array(probs, dtype=np.float32)

            # Fuse EAR-based eye-closure evidence so sustained closed eyes are not missed.
            state, probs, max_confidence = self._apply_eye_closure_fusion(state, probs, max_confidence)

            # During sustained eye closure, avoid smoothing lag so Drowsy appears immediately.
            if self.eye_closed_streak >= EYE_CLOSED_STREAK_TRIGGER:
                self.prediction_buffer.append(np.array(probs, dtype=np.float32))
                self.confidence_buffer.append(max_confidence)
                return state, np.array(probs, dtype=np.float32), float(max_confidence)

            if self.use_temporal_smoothing:
                self.prediction_buffer.append(probs)
                self.confidence_buffer.append(max_confidence)

                probs_smooth = np.mean(list(self.prediction_buffer), axis=0)
                confidence_smooth = float(np.mean(list(self.confidence_buffer)))

                pred_class = int(np.argmax(probs_smooth))
                state = CLASS_NAMES[pred_class]
                max_confidence = confidence_smooth
                probs = np.array(probs_smooth, dtype=np.float32)

            return state, probs, max_confidence

        if self.force_rule_based:
            return self._rule_based_predict(use_haar_tuned_thresholds=not getattr(self.detector, 'mp_available', False))
        
        if len(self.models) == 0:
            return self._rule_based_predict()
        
        # Prepare input
        seq = np.array(list(self.frame_buffer), dtype=np.float32)
        seq = np.expand_dims(seq, axis=0)
        
        if len(self.models) == 1:
            # Single model inference
            probs = self.models[0].predict(seq, verbose=0)[0]
        else:
            # Ensemble inference
            all_probs = []
            for model in self.models:
                probs_i = model.predict(seq, verbose=0)[0]
                all_probs.append(probs_i)
            
            # Ensemble aggregation
            all_probs = np.array(all_probs)
            if ENSEMBLE_CONFIG['method'] == 'average':
                probs = np.mean(all_probs, axis=0)
            elif ENSEMBLE_CONFIG['method'] == 'weighted_average':
                # Weight by model index (newer models have higher weight)
                weights = np.linspace(0.5, 1.5, len(all_probs))
                weights /= weights.sum()
                probs = np.average(all_probs, axis=0, weights=weights)
            else:  # voting
                probs = np.mean(all_probs, axis=0)
        
        # Extract prediction and confidence
        pred_class = int(np.argmax(probs))
        max_confidence = float(probs[pred_class])
        state = CLASS_NAMES[pred_class]
        
        # Apply temporal smoothing if enabled
        if self.use_temporal_smoothing:
            self.prediction_buffer.append(probs)
            self.confidence_buffer.append(max_confidence)
            
            # Average predictions over window
            probs_smooth = np.mean(list(self.prediction_buffer), axis=0)
            confidence_smooth = np.mean(list(self.confidence_buffer))
            
            pred_class = int(np.argmax(probs_smooth))
            state = CLASS_NAMES[pred_class]
            max_confidence = confidence_smooth
        
        return state, probs, max_confidence

    def _update_personal_calibration(self, ear, mar, detector_confidence):
        """Build personal EAR/MAR baselines to improve per-user detection quality."""
        if not np.isfinite(ear) or not np.isfinite(mar):
            return
        if detector_confidence < CALIBRATION_MIN_CONFIDENCE:
            return

        self.calibration_ear_samples.append(float(ear))
        self.calibration_mar_samples.append(float(mar))

        if self.calibration_complete:
            # Keep baselines fresh only during likely-alert moments.
            if self.current_state == "Alert":
                ear_base = float(self.personal_ear_baseline) if self.personal_ear_baseline is not None else float(ear)
                mar_base = float(self.personal_mar_baseline) if self.personal_mar_baseline is not None else float(mar)
                self.personal_ear_baseline = 0.98 * ear_base + 0.02 * float(ear)
                self.personal_mar_baseline = 0.98 * mar_base + 0.02 * float(mar)
            return

        if len(self.calibration_ear_samples) < CALIBRATION_TARGET_FRAMES:
            return

        ear_arr = np.array(self.calibration_ear_samples, dtype=np.float32)
        mar_arr = np.array(self.calibration_mar_samples, dtype=np.float32)

        self.personal_ear_baseline = float(np.percentile(ear_arr, 70))
        self.personal_mar_baseline = float(np.percentile(mar_arr, 50))

        if getattr(self.detector, 'mp_available', False):
            self.adaptive_ear_threshold = float(np.clip(self.personal_ear_baseline * EAR_BASELINE_RATIO_MEDIAPIPE, 0.15, 0.24))
            self.adaptive_mar_threshold = float(np.clip(max(MAR_THRESHOLD, self.personal_mar_baseline * 1.18), 0.58, 0.82))
        else:
            self.adaptive_ear_threshold = float(np.clip(self.personal_ear_baseline * EAR_BASELINE_RATIO_HAAR, 0.12, 0.20))
            self.adaptive_mar_threshold = float(np.clip(max(0.70, self.personal_mar_baseline * 1.12), 0.68, 0.86))

        self.calibration_complete = True
        print(
            f"[OK] Calibration complete | EAR baseline:{self.personal_ear_baseline:.3f} "
            f"EAR thr:{self.adaptive_ear_threshold:.3f} MAR thr:{self.adaptive_mar_threshold:.3f}"
        )

    def _decay_state_when_no_face(self):
        """Graceful decay of alert state when the face is temporarily lost."""
        gap = time.time() - self.last_face_seen_time
        if gap <= NO_FACE_RESET_SECONDS:
            return

        self.eye_closed_streak = max(0, self.eye_closed_streak - 1)
        self.yawn_counter = max(0, self.yawn_counter - 1)
        self.drowsy_counter = max(0, self.drowsy_counter - 1)
        self.alert_counter += 1

        if self.alarm_armed and gap > (NO_FACE_RESET_SECONDS + 0.8):
            self.alarm_armed = False
            self.alarm.stop()

    def _apply_eye_closure_fusion(self, state, probs, confidence):
        """Fuse EAR signal with HF output to improve closed-eye drowsiness recall."""
        if len(self.ear_mar_buffer) < 5:
            return state, probs, confidence

        history = np.array([v[2] for v in self.ear_mar_buffer if np.isfinite(v[2])], dtype=np.float32)
        if history.size < 5:
            return state, probs, confidence

        recent_ear = float(np.mean(history[-5:]))
        baseline_ear = float(self.personal_ear_baseline) if self.personal_ear_baseline is not None else float(np.percentile(history, 70))
        variability = float(np.std(history))

        # Dynamic threshold: blend baseline ratio and variability-aware threshold.
        if getattr(self.detector, 'mp_available', False):
            ratio_threshold = baseline_ear * EAR_BASELINE_RATIO_MEDIAPIPE
            threshold_cap = 0.24
        else:
            ratio_threshold = baseline_ear * EAR_BASELINE_RATIO_HAAR
            threshold_cap = 0.19

        dynamic_threshold = baseline_ear - max(0.012, 1.2 * variability)
        eye_close_threshold = max(dynamic_threshold, ratio_threshold)
        eye_close_threshold = min(eye_close_threshold, threshold_cap)
        eye_close_threshold = max(0.10, eye_close_threshold)

        eyes_closed = recent_ear < eye_close_threshold
        if eyes_closed:
            self.eye_closed_streak += 1
        else:
            self.eye_closed_streak = max(0, self.eye_closed_streak - 1)

        if self.eye_closed_streak < EYE_CLOSED_STREAK_TRIGGER:
            return state, probs, confidence

        fused_probs = np.array(probs, dtype=np.float32)
        closure_strength = min(1.0, (eye_close_threshold - recent_ear) / max(eye_close_threshold, 1e-6))
        drowsy_boost = 0.45 + 0.45 * closure_strength

        fused_probs[1] = max(float(fused_probs[1]), drowsy_boost)
        fused_probs[0] = float(fused_probs[0]) * 0.45
        fused_probs[2] = float(fused_probs[2]) * 0.85

        total = float(np.sum(fused_probs))
        if total > 1e-6:
            fused_probs /= total

        pred_class = int(np.argmax(fused_probs))
        fused_state = CLASS_NAMES[pred_class]
        fused_confidence = float(fused_probs[pred_class])

        if fused_state != "Drowsy":
            fused_state = "Drowsy"
            fused_confidence = max(fused_confidence, float(fused_probs[1]))

        return fused_state, fused_probs, fused_confidence

    def _rule_based_predict(self, use_haar_tuned_thresholds=False):
        """Fallback rule-based prediction using EAR/MAR thresholds."""
        if len(self.ear_mar_buffer) < SEQUENCE_LENGTH:
            return "Alert", np.array([1.0, 0.0, 0.0]), 0.0
        
        recent = list(self.ear_mar_buffer)
        avg_ear = np.mean([r[2] for r in recent[-10:]])
        avg_mar = np.mean([r[3] for r in recent[-10:]])
        
        # Adaptive thresholds
        ear_threshold = self.adaptive_ear_threshold
        mar_threshold = self.adaptive_mar_threshold

        # Haar-only fallback tends to under-estimate EAR and over-estimate MAR.
        # Use conservative thresholds to reduce false drowsy/yawn triggers.
        if use_haar_tuned_thresholds:
            ear_threshold = min(ear_threshold, 0.14)
            mar_threshold = max(mar_threshold, 0.72)
        
        if avg_ear < ear_threshold:
            confidence = 1.0 - (avg_ear / ear_threshold)
            return "Drowsy", np.array([0.1, 0.8, 0.1]), min(0.95, confidence)
        elif avg_mar > mar_threshold:
            confidence = (avg_mar - mar_threshold) / (1.0 - mar_threshold)
            return "Yawning", np.array([0.1, 0.1, 0.8]), min(0.95, confidence)
        
        return "Alert", np.array([0.8, 0.1, 0.1]), 0.9

    def update_adaptive_thresholds(self, confidence):
        """Update EAR/MAR thresholds based on detection confidence."""
        if confidence >= 0.85:  # High confidence
            self.adaptive_ear_threshold = EAR_THRESHOLD_HIGH_CONF
            self.adaptive_mar_threshold = MAR_THRESHOLD_HIGH_CONF
        elif confidence >= 0.65:  # Medium confidence
            ratio = (confidence - 0.65) / (0.85 - 0.65)
            self.adaptive_ear_threshold = (
                EAR_THRESHOLD_LOW_CONF * (1 - ratio) + 
                EAR_THRESHOLD_HIGH_CONF * ratio
            )
            self.adaptive_mar_threshold = (
                MAR_THRESHOLD_LOW_CONF * (1 - ratio) + 
                MAR_THRESHOLD_HIGH_CONF * ratio
            )
        else:  # Low confidence
            self.adaptive_ear_threshold = EAR_THRESHOLD_LOW_CONF
            self.adaptive_mar_threshold = MAR_THRESHOLD_LOW_CONF

    def _update_alarm_state(self, state, confidence):
        """Update alarm with hysteresis and smoothing."""
        current_time = time.time()
        confidence_gate = ALARM_TRIGGER_THRESHOLD
        yawn_gate = ALARM_TRIGGER_THRESHOLD
        if self.force_rule_based:
            confidence_gate = min(ALARM_TRIGGER_THRESHOLD, 0.45)
        if self.use_huggingface and self.hf_classifier is not None:
            confidence_gate = min(confidence_gate, HF_CONFIDENCE_THRESHOLD)
        if self.force_rule_based or (self.use_huggingface and self.hf_classifier is not None):
            yawn_gate = min(0.20, confidence_gate)
        
        strong_eye_closure = self.eye_closed_streak >= EYE_CLOSED_STREAK_TRIGGER

        if (state == "Drowsy" and confidence >= confidence_gate) or strong_eye_closure:
            inc = 1
            if self.eye_closed_streak >= EYE_CLOSED_STREAK_STRONG:
                inc += 1
            self.drowsy_counter += inc
            self.yawn_counter = 0
            self.alert_counter = max(0, self.alert_counter - 1)
            self.alarm_intensity = min(1.0, self.alarm_intensity + 0.1)
        elif (self.force_rule_based or (self.use_huggingface and self.hf_classifier is not None)) and state == "Yawning" and confidence >= yawn_gate:
            # In fallback/HF modes, sustained yawning contributes to risk buildup.
            self.yawn_counter += 1
            if self.yawn_counter % 2 == 0:
                self.drowsy_counter += 1
            self.alert_counter = max(0, self.alert_counter - 1)
            self.alarm_intensity = min(1.0, self.alarm_intensity + 0.06)
        else:
            self.yawn_counter = 0
            self.drowsy_counter = max(0, self.drowsy_counter - 2)
            self.alert_counter += 1
            self.alarm_intensity = max(0.0, self.alarm_intensity - 0.15)
        
        # Arm alarm with hysteresis
        trigger_threshold = ALARM_CONSECUTIVE_FRAMES * (1.0 - ALARM_HYSTERESIS)
        if self.drowsy_counter >= trigger_threshold and not self.alarm_armed:
            self.alarm_armed = True
            self.last_alarm_time = current_time
        elif self.drowsy_counter < ALARM_CONSECUTIVE_FRAMES * ALARM_HYSTERESIS and self.alarm_armed:
            self.alarm_armed = False
        
        # Trigger audio alarm if armed
        if self.alarm_armed and (current_time - self.last_alarm_time) > ALARM_MIN_DURATION:
            self.alarm.play(intensity=self.alarm_intensity)
        else:
            self.alarm.stop()

    def run(self):
        """Main detection loop."""
        print("\n" + "=" * 70)
        print("   ENHANCED REAL-TIME DROWSINESS DETECTOR")
        print("=" * 70)
        print(f"[i] Ensemble: {'Enabled' if self.use_ensemble else 'Disabled'} ({len(self.models)} model(s))")
        print(f"[i] Temporal Smoothing: {'Enabled' if self.use_temporal_smoothing else 'Disabled'}")

        inference_mode = "Model"
        if self.use_huggingface and self.hf_classifier is not None:
            inference_mode = "Hugging Face"
        elif self.force_rule_based or len(self.models) == 0:
            inference_mode = "Rule-based"

        print(f"[i] Inference mode: {inference_mode}")
        if self.use_huggingface and self.hf_classifier is not None:
            print(f"[i] HF backend: {self.hf_classifier.backend} ({self.hf_model_id})")
        print(f"[i] Prediction every {PREDICT_EVERY_N_FRAMES} frame(s)\n")
        
        # Open camera
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[X] Cannot open webcam!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        print(f"[OK] Webcam opened - {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
        print("[i] Press 'Q' to quit, 'R' to reset alarm, 'S' to save frame\n")
        
        prev_time = time.time()
        frame_count = 0
        required_sequence_frames = 1 if (self.use_huggingface and self.hf_classifier is not None) else SEQUENCE_LENGTH

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # FPS calculation
                curr_time = time.time()
                self.fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time
                self.frame_count += 1
                
                # Face detection
                result = self.detector.process_frame(frame)
                
                if result['success'] and result['face_roi'] is not None:
                    self.last_face_seen_time = time.time()

                    # Preprocess and buffer
                    face_roi = preprocess_face_roi(result['face_roi'])
                    self.frame_buffer.append(face_roi)
                    self.ear_mar_buffer.append([
                        result['left_ear'], result['right_ear'],
                        result['ear'], result['mar']
                    ])
                    
                    # Predict every N frames
                    frame_count += 1
                    if (frame_count % PREDICT_EVERY_N_FRAMES == 0 and 
                        len(self.frame_buffer) >= required_sequence_frames):
                        state, probs, confidence = self.predict()
                        self.current_state = state
                        self.current_probs = np.array(probs, dtype=np.float32)
                        self.current_confidence = confidence
                        self.prediction_history.append(state)

                        # Update per-user calibration using high-quality detections.
                        self._update_personal_calibration(result['ear'], result['mar'], result['confidence'])
                        
                        # Update adaptive thresholds
                        self.update_adaptive_thresholds(result['confidence'])
                    
                    # Update alarm logic
                    self._update_alarm_state(self.current_state, self.current_confidence)
                    
                    # Log metrics periodically
                    if VERBOSE_MODE and self.frame_count % LOG_METRICS_INTERVAL == 0:
                        print(f"[F{self.frame_count}] State:{self.current_state} " +
                              f"Conf:{self.current_confidence:.2f} " +
                              f"EAR:{result['ear']:.3f} MAR:{result['mar']:.3f} EyeStreak:{self.eye_closed_streak} " +
                              f"FPS:{self.fps:.1f}")
                else:
                    self._decay_state_when_no_face()
                
                # Draw enhanced HUD
                frame = self._draw_enhanced_hud(frame, result)
                
                # Display
                cv2.imshow("Enhanced Drowsiness Detector", frame)
                
                # Calculate frame processing time
                frame_time = time.time() - frame_start
                self.processing_times.append(frame_time)
                
                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[i] Quitting...")
                    break
                elif key == ord('r'):
                    self.drowsy_counter = 0
                    self.yawn_counter = 0
                    self.alert_counter = 0
                    self.alarm_armed = False
                    self.alarm.stop()
                    print("[i] Alarm reset")
                elif key == ord('s'):
                    filename = f"detection_frame_{self.frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[i] Saved frame: {filename}")
        
        except KeyboardInterrupt:
            print("\n[i] Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.alarm.cleanup()
            self.detector.close()
            self._print_statistics()
            print("[OK] Detector stopped")

    def _draw_enhanced_hud(self, frame, result):
        """Draw professional HUD with all information."""
        h, w = frame.shape[:2]
        
        if not result['success']:
            cv2.putText(frame, "No Face Detected", (w//2-150, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            return frame
        
        # Draw landmarks and bounding box
        frame = self.detector.draw_landmarks(
            frame, result, self.current_state, self.fps, 
            probs=self.current_probs,
            prediction_confidence=self.current_confidence
        )
        
        # Drowsiness alarm bar
        bar_h = 20
        bar_y = h - 40
        bar_limit = 300
        bar_pct = min(1.0, self.drowsy_counter / max(1, ALARM_CONSECUTIVE_FRAMES))
        
        bar_color = (0, 255, 0) if bar_pct < 0.5 else (0, 165, 255) if bar_pct < 0.8 else (0, 0, 255)
        cv2.rectangle(frame, (10, bar_y), (10 + int(bar_limit*bar_pct), bar_y+bar_h), bar_color, -1)
        cv2.rectangle(frame, (10, bar_y), (10 + bar_limit, bar_y+bar_h), (255, 255, 255), 2)
        cv2.putText(frame, f"Alert: {self.drowsy_counter}/{ALARM_CONSECUTIVE_FRAMES}",
                    (320, bar_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not self.calibration_complete:
            text = f"Calibrating: {len(self.calibration_ear_samples)}/{CALIBRATION_TARGET_FRAMES}"
            cv2.putText(frame, text, (10, bar_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        elif self.personal_ear_baseline is not None:
            text = f"EAR thr:{self.adaptive_ear_threshold:.3f} base:{self.personal_ear_baseline:.3f}"
            cv2.putText(frame, text, (10, bar_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2)
        
        # Alarm flash
        if self.alarm_armed:
            if int(time.time() * 4) % 2 == 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(frame, "!!! DROWSINESS ALERT - WAKE UP !!!",
                           (w//2-320, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        return frame

    def _print_statistics(self):
        """Print performance statistics."""
        print("\n" + "=" * 70)
        print("   PERFORMANCE STATISTICS")
        print("=" * 70)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Average FPS: {np.mean([1/t for t in self.processing_times if t > 0]):.1f}")
        
        if len(self.processing_times) > 0:
            avg_time = np.mean(self.processing_times) * 1000
            print(f"Average frame time: {avg_time:.2f} ms")
        
        if len(self.prediction_history) > 0:
            from collections import Counter
            counts = Counter(self.prediction_history)
            print(f"\nPrediction distribution:")
            for state, count in counts.most_common():
                pct = count / len(self.prediction_history) * 100
                print(f"  {state}: {count} ({pct:.1f}%)")
        
        print(f"Total drowsiness detections: {self.drowsy_counter}")
        print("=" * 70)


# For backward compatibility
DrowsinessDetector = EnhancedDrowsinessDetector


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Real-time Drowsiness Detector")
    parser.add_argument('--model', type=str, default=None, help="Path to model file")
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument('--ensemble', action='store_true', help="Enable ensemble mode")
    parser.add_argument('--no-smooth', action='store_true', help="Disable temporal smoothing")
    parser.add_argument('--rule-based', action='store_true', help="Force EAR/MAR rule-based inference")
    parser.add_argument('--hf', action='store_true', help="Use Hugging Face zero-shot image classification")
    parser.add_argument('--hf-model', type=str, default=None, help="Hugging Face model id")
    parser.add_argument('--hf-backend', choices=['auto', 'local', 'hosted'], default=HF_BACKEND,
                        help="Hugging Face backend mode")
    
    args = parser.parse_args()
    
    detector = EnhancedDrowsinessDetector(
        model_path=args.model,
        use_ensemble=args.ensemble,
        use_temporal_smoothing=not args.no_smooth,
        force_rule_based=args.rule_based,
        use_huggingface=args.hf or USE_HUGGINGFACE,
        hf_model_id=args.hf_model,
        hf_backend=args.hf_backend
    )
    
    detector.run()
