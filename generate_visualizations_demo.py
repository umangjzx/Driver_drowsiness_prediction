"""
Quick Start Guide for Generating Visualizations

This script demonstrates how to generate visualizations from:
1. Training history
2. Real-time detection metrics
3. Model evaluation results

Run: python generate_visualizations_demo.py
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualizations import DrowsinessVisualizer
from metrics_collector import MetricsCollector


def demo_training_visualizations():
    """Generate visualizations from training history."""
    print("\n" + "="*70)
    print("DEMO: TRAINING VISUALIZATIONS")
    print("="*70 + "\n")
    
    visualizer = DrowsinessVisualizer(output_dir="visualization_results/training")
    
    # Simulate training history
    epochs = 50
    history = {
        'loss': np.linspace(1.0, 0.3, epochs) + np.random.normal(0, 0.05, epochs),
        'accuracy': np.linspace(0.3, 0.92, epochs) + np.random.normal(0, 0.02, epochs),
        'val_loss': np.linspace(0.95, 0.35, epochs) + np.random.normal(0, 0.06, epochs),
        'val_accuracy': np.linspace(0.35, 0.88, epochs) + np.random.normal(0, 0.03, epochs)
    }
    
    # Clip to valid ranges
    history['loss'] = np.clip(history['loss'], 0, 2)
    history['accuracy'] = np.clip(history['accuracy'], 0, 1)
    history['val_loss'] = np.clip(history['val_loss'], 0, 2)
    history['val_accuracy'] = np.clip(history['val_accuracy'], 0, 1)
    
    print("[*] Generating training curves...")
    visualizer.plot_training_history(history)
    
    print("\n[OK] Training visualizations complete!")
    print(f"[*] Output: visualization_results/training/")


def demo_evaluation_visualizations():
    """Generate visualizations from model evaluation."""
    print("\n" + "="*70)
    print("DEMO: MODEL EVALUATION VISUALIZATIONS")
    print("="*70 + "\n")
    
    visualizer = DrowsinessVisualizer(output_dir="visualization_results/evaluation")
    
    # Simulate evaluation data
    np.random.seed(42)
    n_samples = 500
    
    # Generate realistic confusion patterns
    y_true = np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.41, 0.14])
    
    # Predictions with ~92% accuracy
    y_pred = y_true.copy()
    errors_idx = np.random.choice(n_samples, int(n_samples * 0.08), replace=False)
    y_pred[errors_idx] = (y_pred[errors_idx] + np.random.randint(1, 3, len(errors_idx))) % 3
    
    # Generate probabilities
    y_pred_proba = np.random.rand(n_samples, 3)
    for i in range(n_samples):
        y_pred_proba[i, y_pred[i]] += np.random.uniform(0.2, 0.5)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    print("[*] Generating confusion matrix...")
    visualizer.plot_confusion_matrix(y_true, y_pred)
    
    print("[*] Generating classification metrics...")
    visualizer.plot_classification_metrics(y_true, y_pred)
    
    print("[*] Generating ROC-AUC curves...")
    visualizer.plot_roc_auc_curves(y_true, y_pred_proba)
    
    print("\n[OK] Evaluation visualizations complete!")
    print(f"[*] Output: visualization_results/evaluation/")


def demo_realtime_metrics_visualizations():
    """Generate visualizations from real-time detection metrics."""
    print("\n" + "="*70)
    print("DEMO: REAL-TIME METRICS VISUALIZATIONS")
    print("="*70 + "\n")
    
    collector = MetricsCollector(max_samples=1000)
    
    # Simulate real-time detection
    print("[*] Simulating real-time detection metrics...")
    np.random.seed(42)
    
    n_frames = 300
    predictions_list = np.random.choice(['Alert', 'Drowsy', 'Yawning'], n_frames, p=[0.6, 0.3, 0.1])
    
    for i in range(n_frames):
        # Simulate EAR values (lower when drowsy)
        if predictions_list[i] == 'Drowsy':
            ear = np.random.uniform(0.05, 0.18)
        else:
            ear = np.random.uniform(0.20, 0.35)
        
        # Simulate MAR values (higher when yawning)
        if predictions_list[i] == 'Yawning':
            mar = np.random.uniform(0.65, 0.85)
        else:
            mar = np.random.uniform(0.30, 0.60)
        
        # Simulate confidence
        confidence = np.random.uniform(0.6, 0.98)
        
        # Simulate FPS
        fps = np.random.uniform(28, 30)
        
        # Simulate latency
        latency_ms = np.random.uniform(30, 40)
        
        # Add blink event occasionally
        blink_event = None
        if np.random.rand() < 0.05:
            blink_event = (i, np.random.uniform(0.1, 0.2))
        
        collector.add_frame_metrics(
            ear=ear,
            mar=mar,
            prediction=predictions_list[i],
            confidence=confidence,
            fps=fps,
            latency_ms=latency_ms,
            blink_event=blink_event
        )
    
    print(f"[OK] Collected metrics for {collector.frames_processed} frames")
    
    # Print summary
    collector.print_summary()
    
    # Generate visualizations
    viz_dict = collector.generate_visualizations(
        output_dir="visualization_results/realtime_metrics"
    )
    
    # Save metrics
    collector.save_metrics(output_dir="visualization_results/realtime_metrics")
    
    print("\n[OK] Real-time metrics visualizations complete!")
    print(f"[*] Output: visualization_results/realtime_metrics/")


def main():
    """Run all visualization demos."""
    print("\n" + "="*70)
    print("DROWSINESS DETECTION - VISUALIZATION GENERATION DEMO")
    print("="*70)
    
    print("\nThis script demonstrates how to generate publication-quality")
    print("visualizations for:")
    print("  1. Training history (loss, accuracy curves)")
    print("  2. Model evaluation (confusion matrix, metrics, ROC curves)")
    print("  3. Real-time detection (EAR, MAR, confidence, FPS)")
    
    print("\n" + "-"*70)
    print("GENERATING DEMO VISUALIZATIONS")
    print("-"*70)
    
    # Run demos
    demo_training_visualizations()
    demo_evaluation_visualizations()
    demo_realtime_metrics_visualizations()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    
    print("\nGenerated Files:")
    print("  ✓ visualization_results/training/training_history.png")
    print("  ✓ visualization_results/evaluation/confusion_matrix.png")
    print("  ✓ visualization_results/evaluation/classification_metrics.png")
    print("  ✓ visualization_results/evaluation/roc_auc_curves.png")
    print("  ✓ visualization_results/realtime_metrics/ear_over_time.png")
    print("  ✓ visualization_results/realtime_metrics/mar_over_time.png")
    print("  ✓ visualization_results/realtime_metrics/blink_rate.png")
    print("  ✓ visualization_results/realtime_metrics/prediction_distribution.png")
    print("  ✓ visualization_results/realtime_metrics/confidence_scores.png")
    print("  ✓ visualization_results/realtime_metrics/fps_latency.png")
    
    print("\nUSAGE IN YOUR CODE:")
    print("""
# For Training:
from visualizations import DrowsinessVisualizer
visualizer = DrowsinessVisualizer()
visualizer.plot_training_history(history.history)

# For Real-time Detection:
from metrics_collector import MetricsCollector
collector = MetricsCollector()
collector.add_frame_metrics(ear=0.25, mar=0.5, ...)
collector.generate_visualizations()

# For Model Evaluation:
from sklearn.metrics import confusion_matrix
y_true, y_pred = [0, 1, 2, ...], [0, 1, 1, ...]
visualizer.plot_confusion_matrix(y_true, y_pred)
visualizer.plot_classification_metrics(y_true, y_pred)
visualizer.plot_roc_auc_curves(y_true, y_pred_proba)
    """)
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
