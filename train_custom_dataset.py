"""
Specialized Training Script for Drowsiness Detector
Handles the custom dataset structure with drowsy subcategories
"""
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, Callback
)
import time
from datetime import datetime

from config import (
    IMG_SIZE, IMG_CHANNELS, SEQUENCE_LENGTH, NUM_CLASSES,
    CLASS_NAMES, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE,
    REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, LEARNING_RATE,
    VALIDATION_SPLIT, AUG_ROTATION_RANGE, AUG_BRIGHTNESS_RANGE,
    AUG_HORIZONTAL_FLIP, AUG_NOISE_STDDEV
)


class DataAugmentationPipeline:
    """Advanced data augmentation for robustness."""
    
    @staticmethod
    def apply_augmentation(frame: np.ndarray, p: float = 0.5) -> np.ndarray:
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
        
        return np.uint8(np.clip(frame, 0, 255))


def load_custom_dataset(dataset_dir: str, augmentation_prob: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from custom structure:
    dataset_dir/
        drowsy/
            sleepyCombination/
            slowBlinkWithNodding/
            yawning/
        notdrowsy/
    """
    X = []
    y = []
    augmentor = DataAugmentationPipeline()
    
    # Map: notdrowsy -> Alert (0), drowsy subfolder combinations -> Drowsy (1) / Yawning (2)
    class_mapping = {
        'notdrowsy': 0,           # Alert
        'sleepyCombination': 1,   # Drowsy
        'slowBlinkWithNodding': 1,# Drowsy
        'yawning': 2              # Yawning
    }
    
    print(f"[*] Loading dataset from {dataset_dir}")
    print(f"[*] Augmentation probability: {augmentation_prob}")
    
    total_images = 0
    for class_name, class_idx in class_mapping.items():
        # Handle nested structure for drowsy
        if class_name == 'notdrowsy':
            class_dir = os.path.join(dataset_dir, class_name)
        else:
            # These are under drowsy/
            class_dir = os.path.join(dataset_dir, 'drowsy', class_name)
        
        if not os.path.exists(class_dir):
            print(f"[!] Class directory not found: {class_dir}")
            continue
        
        image_paths = list(Path(class_dir).glob("*.jpg")) + \
                      list(Path(class_dir).glob("*.png"))
        
        class_display = f"Class {class_idx} ({class_name})"
        print(f"[*] Loading {len(image_paths)} images for {class_display}")
        
        for idx, img_path in enumerate(image_paths):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Apply augmentation
                if np.random.rand() < augmentation_prob:
                    img = augmentor.apply_augmentation(img)
                
                X.append(img)
                y.append(class_idx)
                total_images += 1
                
                # Progress indicator
                if (idx + 1) % 5000 == 0:
                    print(f"   [{idx + 1}/{len(image_paths)}]")
                    
            except Exception as e:
                if (idx + 1) % 10000 == 0:  # Log every 10k attempts
                    print(f"   [Error loading image {idx}: {type(e).__name__}]")
        
        print(f"   [OK] {len(image_paths)} images loaded for {class_display}")
    
    print(f"\n[OK] Total images loaded: {total_images}")
    
    X = np.array(X, dtype=np.uint8)
    y = np.array(y)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for class_idx, count in zip(unique, counts):
        pct = (count / len(y)) * 100
        print(f"  Class {class_idx} ({CLASS_NAMES[class_idx]}): {count} ({pct:.1f}%)")
    
    return X, y


def build_bilstm_model(
    input_shape: Tuple = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS),
    sequence_length: int = SEQUENCE_LENGTH,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE
) -> keras.Model:
    """Build MobileNetV3 + BiLSTM model for temporal drowsiness detection."""
    
    # Input: sequence of frames
    inputs = keras.Input(shape=(sequence_length, *input_shape))
    
    # Pre-trained MobileNetV3 backbone (without top)
    base_model = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze for transfer learning
    
    print(f"[*] MobileNetV3 base model loaded (output dim: 576)")
    
    # Apply CNN to each frame in sequence
    x = keras.layers.TimeDistributed(base_model)(inputs)
    x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D())(x)
    
    # Add dropout after CNN features
    x = keras.layers.Dropout(0.2)(x)
    
    # BiLSTM for temporal modeling (bidirectional processes sequence in both directions)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)
    )(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Dense layers with BatchNormalization
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Output with softmax for multi-class
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with better metrics
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


class TrainingProgressCallback(keras.callbacks.Callback):
    """Custom callback for detailed training progress."""
    
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        rate = elapsed / (epoch + 1)
        remaining = rate * (self.total_epochs - epoch - 1)
        
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        print(f"\n   [Epoch {epoch+1}/{self.total_epochs}] "
              f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


class SequenceDataGenerator(keras.utils.Sequence):
    """Memory-efficient data generator for sequences."""
    
    def __init__(self, X, y, batch_size=16, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices].astype(np.float32)
        y_batch = self.y[batch_indices]
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_model_on_custom_dataset(
    dataset_dir: str,
    output_dir: str = "saved_models",
    epochs: int = EPOCHS,
    batch_size: int = 16,  # Increased from 8 for better gradient estimates
    augmentation_prob: float = 0.7
):
    """Train model on custom dataset with memory-efficient generators."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("TRAINING DROWSINESS DETECTOR ON CUSTOM DATASET (Memory Efficient)")
    print("="*70)
    
    # Load dataset
    print(f"\n[STEP 1] Loading dataset from {dataset_dir}...")
    X, y = load_custom_dataset(dataset_dir, augmentation_prob=augmentation_prob)
    
    # Normalize
    print(f"\n[STEP 2] Normalizing images...")
    X = X.astype(np.float32) / 255.0
    print(f"  Shape: {X.shape}")
    print(f"  Range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Convert frames to sequences
    print(f"\n[STEP 3] Creating sequences (window size: {SEQUENCE_LENGTH})...")
    sequences = []
    seq_labels = []
    
    # Group images by class
    for class_idx in range(NUM_CLASSES):
        indices = np.where(y == class_idx)[0]
        print(f"  Class {class_idx} ({CLASS_NAMES[class_idx]}): {len(indices)} images -> ", end="")
        
        # Create overlapping sequences
        seq_count = 0
        for i in range(0, len(indices) - SEQUENCE_LENGTH, max(1, SEQUENCE_LENGTH // 2)):
            seq = X[indices[i:i+SEQUENCE_LENGTH]]
            sequences.append(seq)
            seq_labels.append(class_idx)
            seq_count += 1
        
        print(f"{seq_count} sequences")
    
    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(seq_labels)
    
    print(f"\n  Total sequences: {len(X_seq)}")
    print(f"  Shape: {X_seq.shape}")
    print(f"  Memory size: ~{(X_seq.nbytes / 1024 / 1024):.0f} MB")
    
    # Convert labels to one-hot
    y_one_hot = keras.utils.to_categorical(y_seq, NUM_CLASSES)
    
    # Train/validation split
    split_idx = int(len(X_seq) * (1 - VALIDATION_SPLIT))
    X_train = X_seq[:split_idx]
    y_train = y_one_hot[:split_idx]
    X_val = X_seq[split_idx:]
    y_val = y_one_hot[split_idx:]
    
    print(f"\n[STEP 4] Train/validation split:")
    print(f"  Training: {len(X_train)} sequences (~{(X_train.nbytes / 1024 / 1024):.0f} MB)")
    print(f"  Validation: {len(X_val)} sequences (~{(X_val.nbytes / 1024 / 1024):.0f} MB)")
    
    # Build model
    print(f"\n[STEP 5] Building model...")
    model = build_bilstm_model()
    print(f"\nModel summary:")
    model.summary()
    
    # Create data generators
    print(f"\n[STEP 6] Creating data generators...")
    train_gen = SequenceDataGenerator(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen = SequenceDataGenerator(X_val, y_val, batch_size=batch_size, shuffle=False)
    
    print(f"  Training batches per epoch: {len(train_gen)}")
    print(f"  Validation batches per epoch: {len(val_gen)}")
    print(f"  Batch size: {batch_size}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1
        ),
        TrainingProgressCallback(epochs)
    ]
    
    # Calculate class weights to handle imbalance
    print(f"\n[STEP 7] Calculating class weights for imbalanced data...")
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(y_seq)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_seq)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"  Class weights: {class_weight_dict}")
    
    # Train
    print(f"\n[STEP 8] Training model for {epochs} epochs...")
    print("="*70)
    
    start_time = time.time()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print("="*70)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(final_model_path)
    print(f"\n[OK] Final model saved to: {final_model_path}")
    
    # Print training summary
    print(f"\n[TRAINING SUMMARY]")
    print(f"  Total training time: {training_time/60:.1f} minutes")
    print(f"  Best validation accuracy: {max(history.history.get('val_accuracy', [0])):.4f}")
    print(f"  Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    # Generate visualizations
    print(f"\n[STEP 9] Generating training visualizations...")
    try:
        from visualizations import DrowsinessVisualizer
        
        viz_dir = os.path.join(output_dir, 'visualizations')
        visualizer = DrowsinessVisualizer(output_dir=viz_dir)
        
        # Plot training history
        visualizer.plot_training_history(history.history)
        
        print(f"[OK] Visualizations saved to: {viz_dir}")
    except Exception as e:
        print(f"[!] Warning: Could not generate visualizations: {type(e).__name__}: {e}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train drowsiness detector on custom dataset")
    parser.add_argument('--data-dir', type=str, default=r"d:\archive (10)\Multi class\train",
                       help="Path to dataset directory")
    parser.add_argument('--output-dir', type=str, default="saved_models",
                       help="Output directory for models")
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=16,
                       help="Batch size (default 16 for better gradients)")
    parser.add_argument('--augmentation-prob', type=float, default=0.7,
                       help="Probability of augmentation")
    
    args = parser.parse_args()
    
    print(f"\nDataset path: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Augmentation probability: {args.augmentation_prob}")
    
    # Train
    model, history = train_model_on_custom_dataset(
        dataset_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augmentation_prob=args.augmentation_prob
    )
    
    print("\n[OK] Training complete! Model ready for evaluation and deployment.")
