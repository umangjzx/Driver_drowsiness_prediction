"""
Configuration constants for Driver Drowsiness Detection System.
All hyperparameters, paths, and thresholds are defined here.
"""

import os

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
ALARM_SOUND_PATH = os.path.join(ASSETS_DIR, "alarm.wav")

# Minimal runtime mode: only ensure assets directory exists.
os.makedirs(ASSETS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Class Labels
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["Alert", "Drowsy", "Yawning"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ──────────────────────────────────────────────────────────────
# Image & Sequence Settings
# ──────────────────────────────────────────────────────────────
IMG_SIZE = 64                   # Face ROI crop size (64x64)
IMG_CHANNELS = 3                # RGB
SEQUENCE_LENGTH = 20            # Sliding window of 20 frames
BATCH_SIZE = 16
SHUFFLE_BUFFER = 1000

# ──────────────────────────────────────────────────────────────
# Model Hyperparameters
# ──────────────────────────────────────────────────────────────
CNN_FILTERS = [32, 64, 128]     # Conv2D filter counts per block
CNN_KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
LSTM_UNITS = 128
LSTM_DROPOUT = 0.3
DENSE_UNITS = 64
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-4

# ──────────────────────────────────────────────────────────────
# Training Settings
# ──────────────────────────────────────────────────────────────
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# ──────────────────────────────────────────────────────────────
# Data Augmentation
# ──────────────────────────────────────────────────────────────
AUG_ROTATION_RANGE = 15         # degrees
AUG_BRIGHTNESS_RANGE = (0.7, 1.3)
AUG_HORIZONTAL_FLIP = True
AUG_NOISE_STDDEV = 0.02
AUG_ZOOM_RANGE = 0.1

# ──────────────────────────────────────────────────────────────
# Facial Landmark Detection (MediaPipe)
# ──────────────────────────────────────────────────────────────
# Eye landmark indices for MediaPipe Face Mesh (468 landmarks)
# Right eye
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# Left eye
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# Mouth (outer lips)
MOUTH_IDX = [61, 291, 39, 181, 0, 17, 269, 405]

# EAR / MAR Thresholds
EAR_THRESHOLD = 0.21            # Below this → eyes closed
MAR_THRESHOLD = 0.65            # Above this → mouth open (yawning)
EAR_CONSEC_FRAMES = 20          # Consecutive frames for drowsy alarm

# ──────────────────────────────────────────────────────────────
# Real-time Inference
# ──────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
TARGET_FPS = 30
ALARM_CONSECUTIVE_FRAMES = 5   # Trigger alarm after this many drowsy frames
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# ──────────────────────────────────────────────────────────────
# Synthetic Data Generation
# ──────────────────────────────────────────────────────────────
SYNTHETIC_SAMPLES_PER_CLASS = 500

# ──────────────────────────────────────────────────────────────
# Advanced Model Architecture (Optional)
# ──────────────────────────────────────────────────────────────
USE_ATTENTION_MECHANISM = True      # Add Attention layers for better performance
USE_BIDIRECTIONAL_LSTM = True       # Use BiLSTM instead of LSTM
USE_LAYER_NORMALIZATION = True      # Layer normalization instead of batch norm
ATTENTION_HEADS = 4                 # For multi-head attention
DROPOUT_ATTENTION = 0.2             # Dropout in attention layer

# ──────────────────────────────────────────────────────────────
# Inference & Post-Processing
# ──────────────────────────────────────────────────────────────
TEMPORAL_SMOOTHING_WINDOW = 5       # Smooth predictions over N frames
CONFIDENCE_THRESHOLD = 0.5          # Min confidence for prediction
USE_ENSEMBLE = True                 # Use ensemble for robustness
ENSEMBLE_CONFIG = {
    'model_paths': [],              # Auto-populated with available models
    'max_models': 3,                # Max models to ensemble
    'method': 'weighted_average',   # 'average', 'weighted_average', 'voting'
}

# Optional Hugging Face backend for Alert/Drowsy/Yawning classification
USE_HUGGINGFACE = True
HF_MODEL_ID = "openai/clip-vit-base-patch32"
HF_BACKEND = "local"               # 'auto', 'local', or 'hosted'
HF_CONFIDENCE_THRESHOLD = 0.22
HF_CLASS_LABELS = {
    "Alert": "an alert driver with eyes open",
    "Drowsy": "a drowsy sleepy driver with eyes closed",
    "Yawning": "a yawning driver with mouth open",
}

# ──────────────────────────────────────────────────────────────
# Face Detection Robustness
# ──────────────────────────────────────────────────────────────
USE_MEDIAPIPE_FACE = True           # Primary: MediaPipe Face Detection
USE_MEDIAPIPE_MESH = True           # Use MediaPipe Face Mesh for landmarks
FACE_DETECTION_CONFIDENCE = 0.6     # MediaPipe detection threshold
FACE_TRACKING_CONFIDENCE = 0.5      # MediaPipe tracking threshold
USE_HAAR_CASCADE_FALLBACK = True    # Fallback to Haar cascades
MIN_FACE_SIZE_PERCENT = 0.05        # Minimum face size as % of frame
MAX_FACES_TO_TRACK = 1              # Number of faces to track

# ──────────────────────────────────────────────────────────────
# Preprocessing Enhancements
# ──────────────────────────────────────────────────────────────
APPLY_HISTOGRAM_EQUALIZATION = True # Improve contrast in poor lighting
APPLY_CLAHE = True                  # Contrast Limited Adaptive Histogram Equalization
NORMALIZE_FACE_ROI = True           # Normalize face ROI to [0, 1]
FACE_ALIGNMENT_METHOD = 'landmarks' # 'landmarks' or 'geometric'

# ──────────────────────────────────────────────────────────────
# Robustness Thresholds
# ──────────────────────────────────────────────────────────────
# Adaptive EAR/MAR thresholds based on confidence
EAR_THRESHOLD_HIGH_CONF = 0.18      # Higher confidence = lower threshold
EAR_THRESHOLD_LOW_CONF = 0.25       # Lower confidence = higher threshold
MAR_THRESHOLD_HIGH_CONF = 0.60
MAR_THRESHOLD_LOW_CONF = 0.70

# Runtime calibration and eye-closure fusion
CALIBRATION_TARGET_FRAMES = 45      # Frames for personal baseline estimation
CALIBRATION_MIN_CONFIDENCE = 0.55   # Minimum detector confidence for calibration sample
EYE_CLOSED_STREAK_TRIGGER = 2       # Closed-eye streak to start drowsy fusion
EYE_CLOSED_STREAK_STRONG = 5        # Closed-eye streak for stronger alarm escalation
EAR_BASELINE_RATIO_MEDIAPIPE = 0.82 # Closed-eye threshold ratio vs personal EAR baseline
EAR_BASELINE_RATIO_HAAR = 0.90      # Less aggressive ratio for Haar fallback
NO_FACE_RESET_SECONDS = 1.2         # Grace period before decaying alert state if no face

# Alarm triggering with temporal awareness
ALARM_TRIGGER_THRESHOLD = 0.70      # Drowsiness probability to trigger alarm
ALARM_HYSTERESIS = 0.15             # Hysteresis for alarm on/off
ALARM_MIN_DURATION = 0.5            # Minimum alarm duration in seconds

# ──────────────────────────────────────────────────────────────
# Logging & Monitoring
# ──────────────────────────────────────────────────────────────
VERBOSE_MODE = True                 # Print detailed debug info
LOG_PREDICTIONS = True              # Log all predictions
LOG_METRICS_INTERVAL = 100          # Log metrics every N frames
SAVE_DETECTION_VIDEO = False        # Save output video (file path or False)

# ──────────────────────────────────────────────────────────────
# Performance Optimization
# ──────────────────────────────────────────────────────────────
USE_QUANTIZATION = False            # Use quantized model (faster but slightly less accurate)
USE_GPU = True                      # Use GPU for inference
PREDICT_EVERY_N_FRAMES = 1          # Predict every frame (1=all, 2=half, 3=third, etc.)
CACHE_PREPROCESSED_FRAMES = True    # Cache preprocessed frames for faster inference

# ──────────────────────────────────────────────────────────────
# Dataset & Training Improvements
# ──────────────────────────────────────────────────────────────
AUGMENTATION_PROB = 0.7             # Probability to apply augmentation
USE_MIXUP_AUGMENTATION = True       # Use mixup for stronger augmentation
MIXUP_ALPHA = 0.2                   # Mixup alpha parameter
USE_CUTMIX_AUGMENTATION = False     # Use CutMix augmentation
CUTMIX_ALPHA = 1.0
SYNTHETIC_TOTAL_SEQUENCES = SYNTHETIC_SAMPLES_PER_CLASS * NUM_CLASSES
