"""
Production Training Pipeline for Drowsiness Detector

Complete pipeline with:
- Data loading from YAWDD, CEW, or custom datasets
- Augmentation (brightness, blur, rotation, occlusion)
- Transfer learning with MobileNetV3 + BiLSTM
- Validation and early stopping
- Model checkpointing and export
"""
import os
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple, Generator, Dict, List
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, Callback
)

from config import (
    IMG_SIZE, IMG_CHANNELS, SEQUENCE_LENGTH, NUM_CLASSES,
    CLASS_NAMES, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE,
    REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, LEARNING_RATE,
    VALIDATION_SPLIT, AUG_ROTATION_RANGE, AUG_BRIGHTNESS_RANGE,
    AUG_HORIZONTAL_FLIP, AUG_NOISE_STDDEV
)


class DataAugmentationPipeline:
    """Advanced data augmentation for robustness."""
    
    def __init__(self):
        self.image_gen = ImageDataGenerator(
            rotation_range=AUG_ROTATION_RANGE,
            brightness_range=AUG_BRIGHTNESS_RANGE,
            horizontal_flip=AUG_HORIZONTAL_FLIP,
            fill_mode='nearest'
        )
    
    def apply_augmentation(self, frame: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Apply random augmentations with probability p."""
        if np.random.rand() > p:
            return frame
        
        frame = frame.astype(np.float32)
        
        # Random brightness
        if np.random.rand() < 0.4:
            brightness = np.random.uniform(0.7, 1.3)
            frame = np.clip(frame * brightness, 0, 255)
        
        # Random blur
        if np.random.rand() < 0.3:
            kernel_size = np.random.choice([3, 5, 7])
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        # Random noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, AUG_NOISE_STDDEV * 255, frame.shape)
            frame = np.clip(frame + noise, 0, 255)
        
        # Random rotation (by warping)
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-AUG_ROTATION_RANGE, AUG_ROTATION_RANGE)
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
        
        # Random occlusion (glasses, face mask simulation)
        if np.random.rand() < 0.2:
            frame = self._apply_occlusion(frame)
        
        return np.uint8(np.clip(frame, 0, 255))
    
    @staticmethod
    def _apply_occlusion(frame: np.ndarray) -> np.ndarray:
        """Simulate occlusions (glasses, masks)."""
        h, w = frame.shape[:2]
        occlusion_type = np.random.choice(['glasses', 'mask', 'blur_region'])
        
        if occlusion_type == 'glasses':
            # Top half occlusion (like glasses)
            y1, y2 = h // 4, h // 2
            x1, x2 = w // 4, 3 * w // 4
        elif occlusion_type == 'mask':
            # Lower half occlusion (like mask)
            y1, y2 = h // 2, 3 * h // 4
            x1, x2 = w // 4, 3 * w // 4
        else:  # blur_region
            y1, y2 = np.random.randint(0, h // 2), np.random.randint(h // 2, h)
            x1, x2 = np.random.randint(0, w // 2), np.random.randint(w // 2, w)
        
        # Apply slight blur/darkening
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (5, 5), 0)
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * 0.7  # Darken
        
        return frame


class DrownsinessDataset:
    """Load and manage drowsiness detection datasets."""
    
    def __init__(self, augmentation_prob: float = 0.7):
        self.augmentation = DataAugmentationPipeline()
        self.augmentation_prob = augmentation_prob
        self.X = []
        self.y = []
    
    def load_from_directory(
        self,
        dataset_dir: str,
        class_subdirs: Dict[str, str] | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from directory structure.
        
        Expected structure:
            dataset_dir/
                alert/
                drowsy/
                yawning/
        """
        if class_subdirs is None:
            class_subdirs = {
                'Alert': 'alert',
                'Drowsy': 'drowsy',
                'Yawning': 'yawning'
            }
        
        print(f"[*] Loading dataset from {dataset_dir}")
        
        for class_name, subdir_name in class_subdirs.items():
            class_dir = os.path.join(dataset_dir, subdir_name)
            if not os.path.exists(class_dir):
                print(f"[!] Class directory not found: {class_dir}")
                continue
            
            class_idx = CLASS_NAMES.index(class_name)
            image_paths = list(Path(class_dir).glob("*.jpg")) + \
                          list(Path(class_dir).glob("*.png"))
            
            print(f"[*] Loading {len(image_paths)} images for {class_name}")
            
            for img_path in image_paths:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.X.append(img)
                        self.y.append(class_idx)
                except Exception as e:
                    print(f"[!] Error loading {img_path}: {e}")
        
        print(f"[OK] Loaded {len(self.X)} total images")
        return np.array(self.X), np.array(self.y)
    
    def load_sequences_from_directory(
        self,
        dataset_dir: str,
        sequence_length: int = SEQUENCE_LENGTH
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load temporal sequences (frames grouped by video/sequence).
        
        Expected structure:
            dataset_dir/
                alert/
                    video1/
                        frame_0.jpg
                        frame_1.jpg
                        ...
                    video2/
                        ...
                drowsy/
                    ...
        """
        print(f"[*] Loading sequences from {dataset_dir}")
        
        sequences = []
        labels = []
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(dataset_dir, class_name.lower())
            if not os.path.exists(class_dir):
                continue
            
            # Iterate through videos
            for video_dir in Path(class_dir).iterdir():
                if not video_dir.is_dir():
                    continue
                
                # Load frames
                frame_paths = sorted(
                    list(video_dir.glob("*.jpg")) +
                    list(video_dir.glob("*.png"))
                )
                
                # Create sequences
                for i in range(0, len(frame_paths) - sequence_length, 1):
                    sequence = []
                    for j in range(sequence_length):
                        img = cv2.imread(str(frame_paths[i + j]))
                        if img is not None:
                            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            sequence.append(img)
                    
                    if len(sequence) == sequence_length:
                        sequences.append(sequence)
                        labels.append(class_idx)
        
        print(f"[OK] Loaded {len(sequences)} sequences")
        return np.array(sequences), np.array(labels)
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize images to [0, 1]."""
        return X.astype(np.float32) / 255.0
    
    def apply_augmentation(self, X: np.ndarray) -> np.ndarray:
        """Apply augmentation to entire batch."""
        X_aug = []
        for img in X:
            if np.random.rand() < self.augmentation_prob:
                img = self.augmentation.apply_augmentation(img)
            X_aug.append(img)
        return np.array(X_aug)


def build_bilstm_model(
    input_shape: Tuple = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS),
    sequence_length: int = SEQUENCE_LENGTH,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE
) -> keras.Model:
    """
    Build MobileNetV3 + BiLSTM model for temporal drowsiness detection.
    """
    # Input: sequence of frames
    inputs = keras.Input(shape=(sequence_length, *input_shape))
    
    # Pre-trained MobileNetV3 backbone (without top)
    base_model = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze for transfer learning
    
    # Apply CNN to each frame in sequence
    x = layers.TimeDistributed(base_model)(inputs)  # [batch, seq_len, feature_dim]
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)  # [batch, seq_len, 576]
    
    # BiLSTM for temporal modeling
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3)
    )(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


class TrainedModelEvaluationCallback(Callback):
    """Callback to evaluate on validation set periodically."""
    
    def __init__(self, X_val, y_val, interval: int = 5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.interval = interval
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            val_loss, val_acc, _, _ = self.model.evaluate(
                self.X_val, self.y_val, verbose=0
            )
            print(f"  [Val] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    model_save_dir: str = "saved_models",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE
) -> Tuple[keras.Model, Dict]:
    """
    Train drowsiness detection model.
    
    Args:
        X_train: Training sequences [N, seq_len, H, W, C]
        y_train: Training labels [N]
        X_val: Validation sequences
        y_val: Validation labels
        model_save_dir: Directory to save models
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        (trained_model, training_history)
    """
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Convert labels to one-hot
    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    
    # Build model
    print("[*] Building model...")
    model = build_bilstm_model()
    print(model.summary())
    
    # Create validation split if not provided
    if X_val is None or y_val is None:
        val_split = VALIDATION_SPLIT
        X_val = None
        y_val = None
    else:
        val_split = None
        y_val_oh = keras.utils.to_categorical(y_val, NUM_CLASSES)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(model_save_dir, 'best_model.keras'),
            monitor='val_accuracy' if X_val is not None else 'accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(model_save_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train
    print(f"[*] Training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh) if X_val is not None else None,
        validation_split=val_split if val_split else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(model_save_dir, 'final_model.keras')
    model.save(final_model_path)
    print(f"[OK] Model saved to {final_model_path}")
    
    return model, history.history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train drowsiness detector")
    parser.add_argument('--data-dir', type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument('--sequences', action='store_true',
                       help="Load sequences instead of individual frames")
    parser.add_argument('--output-dir', type=str, default="saved_models",
                       help="Output directory for models")
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--augmentation-prob', type=float, default=0.7)
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = DrownsinessDataset(augmentation_prob=args.augmentation_prob)
    
    if args.sequences:
        X, y = dataset.load_sequences_from_directory(args.data_dir)
    else:
        X, y = dataset.load_from_directory(args.data_dir)
        # Convert to sequences manually
        print(f"[*] Converting frames to sequences...")
        sequences, seq_labels = [], []
        class_indices = {i: np.where(y == i)[0] for i in range(NUM_CLASSES)}
        
        for class_idx in range(NUM_CLASSES):
            indices = class_indices[class_idx]
            for i in range(0, len(indices) - SEQUENCE_LENGTH, 1):
                seq = X[indices[i:i+SEQUENCE_LENGTH]]
                sequences.append(seq)
                seq_labels.append(class_idx)
        
        X = np.array(sequences)
        y = np.array(seq_labels)
    
    # Normalize
    X = dataset.normalize(X)
    
    # Augment training data
    print(f"[*] Applying augmentation...")
    X = dataset.apply_augmentation(X)
    
    # Train
    model, history = train_model(X, y, epochs=args.epochs, batch_size=args.batch_size,
                                model_save_dir=args.output_dir)
    
    print("[OK] Training complete!")
