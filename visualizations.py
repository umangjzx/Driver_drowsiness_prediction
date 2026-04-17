"""
Comprehensive Visualization Module for Drowsiness Detection System

Generates publication-quality plots for:
- Model performance (confusion matrix, precision, recall, F1)
- Real-time metrics (EAR, MAR, blink rate)
- Training analysis (accuracy/loss curves)
- Detection insights (prediction distribution, confidence scores)

All plots are saved as high-resolution PNG images.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family'] = 'sans-serif'

# Color palette for classes
CLASS_COLORS = {
    'Alert': '#2ecc71',      # Green
    'Drowsy': '#e74c3c',     # Red
    'Yawning': '#f39c12'     # Orange
}

RGB_COLORS = ['#2ecc71', '#e74c3c', '#f39c12']  # For indexing


class DrowsinessVisualizer:
    """Generate comprehensive visualizations for drowsiness detection system."""
    
    def __init__(self, output_dir: str = "visualization_results"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualization PNG files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Visualization output directory: {self.output_dir}")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = ['Alert', 'Drowsy', 'Yawning'],
        figsize: Tuple = (10, 8),
        cmap: str = 'Blues'
    ) -> str:
        """
        Create and save confusion matrix visualization.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: Names of classes
            figsize: Figure size
            cmap: Color map
            
        Returns:
            Path to saved image
        """
        from sklearn.metrics import confusion_matrix
        
        print("[*] Generating Confusion Matrix...")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            cbar_kws={'label': 'Count'},
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
            linewidths=2,
            linecolor='white'
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Drowsiness Detection Model', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add accuracy text
        accuracy = np.trace(cm) / cm.sum()
        textstr = f'Overall Accuracy: {accuracy:.2%}'
        fig.text(0.5, 0.02, textstr, ha='center', fontsize=11, 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def plot_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = ['Alert', 'Drowsy', 'Yawning'],
        figsize: Tuple = (14, 6)
    ) -> str:
        """
        Create and save classification metrics visualization.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: Names of classes
            figsize: Figure size
            
        Returns:
            Path to saved image
        """
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        print("[*] Generating Classification Metrics...")
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        accuracy = accuracy_score(y_true, y_pred)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        metrics = [
            ('Precision', precision),
            ('Recall', recall),
            ('F1-Score', f1)
        ]
        
        for idx, (metric_name, values) in enumerate(metrics):
            ax = axes[idx]
            
            bars = ax.bar(
                class_names,
                values,
                color=RGB_COLORS,
                edgecolor='black',
                linewidth=2,
                alpha=0.85
            )
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f'{value:.3f}',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=10
                )
            
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.set_title(f'{metric_name} by Class', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle(
            f'Classification Metrics - Overall Accuracy: {accuracy:.2%}',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'classification_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def plot_roc_auc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: List[str] = ['Alert', 'Drowsy', 'Yawning'],
        figsize: Tuple = (12, 6)
    ) -> str:
        """
        Create and save ROC-AUC curves.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities (n_samples, n_classes)
            class_names: Names of classes
            figsize: Figure size
            
        Returns:
            Path to saved image
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        print("[*] Generating ROC-AUC Curves...")
        
        # Binarize labels
        y_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve for each class
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(
                fpr, tpr,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})',
                linewidth=2.5,
                color=RGB_COLORS[i]
            )
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC-AUC Curves - Drowsiness Detection Model',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'roc_auc_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def plot_training_history(
        self,
        history: Dict,
        figsize: Tuple = (14, 5)
    ) -> str:
        """
        Create and save training history plots.
        
        Args:
            history: Training history dictionary from keras model.fit()
            figsize: Figure size
            
        Returns:
            Path to saved image
        """
        print("[*] Generating Training History Plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot 1: Loss
        ax1 = axes[0]
        ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
        
        ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss (Categorical Crossentropy)', fontsize=11, fontweight='bold')
        ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Accuracy
        ax2 = axes[1]
        ax2.plot(epochs, history['accuracy'], 'g-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
        if 'val_accuracy' in history:
            ax2.plot(epochs, history['val_accuracy'], 'orange', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
        
        ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        fig.suptitle('Model Training History', fontsize=14, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'training_history.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def plot_eye_aspect_ratio(
        self,
        ear_values: List[float],
        timestamps: Optional[List[float]] = None,
        threshold: float = 0.21,
        figsize: Tuple = (14, 6)
    ) -> str:
        """
        Create and save Eye Aspect Ratio (EAR) plot over time.
        
        Args:
            ear_values: List of EAR values
            timestamps: Optional timestamps (defaults to frame numbers)
            threshold: EAR threshold for drowsiness
            figsize: Figure size
            
        Returns:
            Path to saved image
        """
        print("[*] Generating Eye Aspect Ratio (EAR) Plot...")
        
        if timestamps is None:
            timestamps = range(len(ear_values))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot EAR values
        ax.plot(timestamps, ear_values, 'b-', linewidth=2, label='EAR', alpha=0.8)
        ax.fill_between(timestamps, ear_values, alpha=0.2, color='blue')
        
        # Plot threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5,
                   label=f'Drowsiness Threshold ({threshold:.2f})', alpha=0.8)
        
        # Highlight drowsy regions
        drowsy_regions = np.array(ear_values) < threshold
        for i, is_drowsy in enumerate(drowsy_regions):
            if is_drowsy:
                ax.axvspan(timestamps[i] - 0.5, timestamps[i] + 0.5, 
                          alpha=0.15, color='red')
        
        ax.set_xlabel('Frame / Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Eye Aspect Ratio (EAR)', fontsize=12, fontweight='bold')
        ax.set_title('Eye Aspect Ratio (EAR) Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, max(ear_values) * 1.2 if ear_values else 1])
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'ear_over_time.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def plot_mouth_aspect_ratio(
        self,
        mar_values: List[float],
        timestamps: Optional[List[float]] = None,
        threshold: float = 0.65,
        figsize: Tuple = (14, 6)
    ) -> str:
        """
        Create and save Mouth Aspect Ratio (MAR) plot over time.
        
        Args:
            mar_values: List of MAR values
            timestamps: Optional timestamps (defaults to frame numbers)
            threshold: MAR threshold for yawning
            figsize: Figure size
            
        Returns:
            Path to saved image
        """
        print("[*] Generating Mouth Aspect Ratio (MAR) Plot...")
        
        if timestamps is None:
            timestamps = range(len(mar_values))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot MAR values
        ax.plot(timestamps, mar_values, 'orange', linewidth=2, label='MAR', alpha=0.8)
        ax.fill_between(timestamps, mar_values, alpha=0.2, color='orange')
        
        # Plot threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5,
                   label=f'Yawning Threshold ({threshold:.2f})', alpha=0.8)
        
        # Highlight yawning regions
        yawning_regions = np.array(mar_values) > threshold
        for i, is_yawning in enumerate(yawning_regions):
            if is_yawning:
                ax.axvspan(timestamps[i] - 0.5, timestamps[i] + 0.5,
                          alpha=0.15, color='orange')
        
        ax.set_xlabel('Frame / Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mouth Aspect Ratio (MAR)', fontsize=12, fontweight='bold')
        ax.set_title('Mouth Aspect Ratio (MAR) Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, max(mar_values) * 1.2 if mar_values else 1])
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'mar_over_time.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def plot_blink_rate(
        self,
        blink_events: List[Tuple[float, float]],
        window_size: int = 60,
        figsize: Tuple = (14, 6)
    ) -> str:
        """
        Create and save blink rate plot over time.
        
        Args:
            blink_events: List of (timestamp, duration) tuples for each blink
            window_size: Window size for calculating blink rate (frames)
            figsize: Figure size
            
        Returns:
            Path to saved image
        """
        print("[*] Generating Blink Rate Plot...")
        
        if not blink_events:
            print("[!] No blink events found. Skipping blink rate plot.")
            return None
        
        # Convert to array
        blink_times = np.array([b[0] if isinstance(b, tuple) else b for b in blink_events])
        
        # Calculate blink rate in sliding windows
        max_time = int(max(blink_times)) + 1
        blink_rates = []
        time_points = []
        
        for t in range(max_time - window_size):
            blinks_in_window = np.sum((blink_times >= t) & (blink_times < t + window_size))
            blink_rate = (blinks_in_window / window_size) * 100  # As percentage
            blink_rates.append(blink_rate)
            time_points.append(t)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot blink rate
        ax.plot(time_points, blink_rates, 'purple', linewidth=2.5, marker='o', 
                markersize=4, label='Blink Rate', alpha=0.8)
        ax.fill_between(time_points, blink_rates, alpha=0.2, color='purple')
        
        # Normal blink rate is ~15-20 blinks per minute (~0.25-0.33 per second)
        # For window of 60 frames at 30fps, that's ~30-60 frames per minute
        normal_rate = 20  # As percentage for our window
        ax.axhline(y=normal_rate, color='green', linestyle='--', linewidth=2,
                   label=f'Normal Rate (~{normal_rate}%)', alpha=0.7)
        ax.axhline(y=normal_rate * 0.5, color='orange', linestyle='--', linewidth=2,
                   label=f'Low Blink Rate (~{normal_rate * 0.5:.0f}%)', alpha=0.7)
        
        ax.set_xlabel('Time (frames)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Blink Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Blink Rate Over Time (Window: {window_size} frames)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, max(blink_rates) * 1.2 if blink_rates else 50])
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'blink_rate.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def plot_prediction_distribution(
        self,
        predictions: Dict[str, int],
        class_names: List[str] = ['Alert', 'Drowsy', 'Yawning'],
        figsize: Tuple = (10, 6)
    ) -> str:
        """
        Create and save prediction distribution bar chart.
        
        Args:
            predictions: Dictionary with class names as keys and prediction counts as values
            class_names: Names of classes
            figsize: Figure size
            
        Returns:
            Path to saved image
        """
        print("[*] Generating Prediction Distribution Plot...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        counts = [predictions.get(name, 0) for name in class_names]
        total = sum(counts)
        
        if total == 0:
            print("[!] No predictions found. Skipping distribution plot.")
            return None
        
        percentages = [(c / total) * 100 for c in counts]
        
        bars = ax.bar(class_names, counts, color=RGB_COLORS, 
                     edgecolor='black', linewidth=2, alpha=0.85)
        
        # Add value labels on bars
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Number of Frames', fontsize=12, fontweight='bold')
        ax.set_title(f'Prediction Distribution (Total: {total} frames)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add summary text
        summary_text = f'Drowsy Rate: {percentages[1]:.1f}%'
        ax.text(0.98, 0.97, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'prediction_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def plot_confidence_scores(
        self,
        confidences: List[float],
        predictions: List[str],
        class_names: List[str] = ['Alert', 'Drowsy', 'Yawning'],
        figsize: Tuple = (14, 6)
    ) -> str:
        """
        Create and save confidence score plot over time.
        
        Args:
            confidences: List of confidence scores
            predictions: List of predicted class names
            class_names: Names of classes
            figsize: Figure size
            
        Returns:
            Path to saved image
        """
        print("[*] Generating Confidence Scores Plot...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        frames = range(len(confidences))
        
        # Color points by prediction
        color_map = {name: RGB_COLORS[i] for i, name in enumerate(class_names)}
        colors = [color_map.get(pred, 'gray') for pred in predictions]
        
        # Plot confidence scores
        scatter = ax.scatter(frames, confidences, c=colors, s=50, alpha=0.6,
                            edgecolors='black', linewidth=0.5)
        
        # Add line plot
        ax.plot(frames, confidences, 'gray', linewidth=1, alpha=0.3)
        
        # Add threshold line
        threshold = 0.5
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Decision Threshold ({threshold})', alpha=0.7)
        
        ax.set_xlabel('Frame', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Confidence Scores Over Time',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim([0, 1.05])
        ax.grid(alpha=0.3)
        
        # Add legend for class colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=RGB_COLORS[i], edgecolor='black',
                                label=class_names[i])
                          for i in range(len(class_names))]
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--',
                                         linewidth=2, label='Decision Threshold'))
        ax.legend(handles=legend_elements, fontsize=10, loc='lower right')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'confidence_scores.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def plot_fps_latency(
        self,
        fps_history: List[float],
        latency_ms_history: List[float],
        figsize: Tuple = (14, 6)
    ) -> str:
        """
        Create and save FPS and latency plots.
        
        Args:
            fps_history: List of FPS measurements
            latency_ms_history: List of latency measurements in milliseconds
            figsize: Figure size
            
        Returns:
            Path to saved image
        """
        print("[*] Generating FPS & Latency Plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        frames = range(len(fps_history))
        
        # Plot 1: FPS
        ax1 = axes[0]
        ax1.plot(frames, fps_history, 'g-', linewidth=2, alpha=0.8)
        ax1.fill_between(frames, fps_history, alpha=0.2, color='green')
        ax1.axhline(y=np.mean(fps_history), color='darkgreen', linestyle='--',
                   linewidth=2, label=f'Mean FPS: {np.mean(fps_history):.1f}')
        ax1.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax1.set_ylabel('FPS (Frames Per Second)', fontsize=11, fontweight='bold')
        ax1.set_title('Frame Rate Performance', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, max(fps_history) * 1.2 if fps_history else 60])
        
        # Plot 2: Latency
        ax2 = axes[1]
        ax2.plot(frames, latency_ms_history, 'b-', linewidth=2, alpha=0.8)
        ax2.fill_between(frames, latency_ms_history, alpha=0.2, color='blue')
        ax2.axhline(y=np.mean(latency_ms_history), color='darkblue', linestyle='--',
                   linewidth=2, label=f'Mean Latency: {np.mean(latency_ms_history):.1f}ms')
        ax2.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Latency (milliseconds)', fontsize=11, fontweight='bold')
        ax2.set_title('Inference Latency', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, max(latency_ms_history) * 1.2 if latency_ms_history else 100])
        
        fig.suptitle('Real-Time Performance Metrics', fontsize=14, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'fps_latency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()
        
        return str(output_path)
    
    def generate_summary_report(self, visualizations: Dict[str, str]) -> str:
        """
        Create a summary report listing all generated visualizations.
        
        Args:
            visualizations: Dictionary mapping visualization names to file paths
            
        Returns:
            Path to summary report JSON file
        """
        print("[*] Generating Summary Report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'visualizations': visualizations,
            'total_plots': len(visualizations)
        }
        
        report_path = self.output_dir / 'visualization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[OK] Saved: {report_path}")
        
        # Also create markdown summary
        markdown_path = self.output_dir / 'VISUALIZATIONS.md'
        with open(markdown_path, 'w') as f:
            f.write("# Drowsiness Detection System - Visualizations\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Generated Plots\n\n")
            for name, path in visualizations.items():
                rel_path = Path(path).name
                f.write(f"### {name}\n")
                f.write(f"![{name}]({rel_path})\n\n")
        
        print(f"[OK] Saved: {markdown_path}")
        
        return str(report_path)


# ============================================================================
# Demo and testing functions
# ============================================================================

def create_demo_visualizations():
    """Create demo visualizations with synthetic data."""
    
    print("\n" + "="*70)
    print("DROWSINESS DETECTION - VISUALIZATION DEMO")
    print("="*70 + "\n")
    
    visualizer = DrowsinessVisualizer(output_dir="visualization_results")
    viz_dict = {}
    
    # 1. Confusion Matrix
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 200)
    y_pred = y_true + np.random.randint(-1, 2, 200)
    y_pred = np.clip(y_pred, 0, 2)
    
    viz_dict['Confusion Matrix'] = visualizer.plot_confusion_matrix(y_true, y_pred)
    
    # 2. Classification Metrics
    viz_dict['Classification Metrics'] = visualizer.plot_classification_metrics(y_true, y_pred)
    
    # 3. ROC-AUC Curves
    y_pred_proba = np.random.rand(len(y_true), 3)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    viz_dict['ROC-AUC Curves'] = visualizer.plot_roc_auc_curves(y_true, y_pred_proba)
    
    # 4. Training History
    history = {
        'loss': [0.95, 0.85, 0.75, 0.65, 0.55, 0.50, 0.48],
        'accuracy': [0.52, 0.60, 0.68, 0.75, 0.80, 0.82, 0.83],
        'val_loss': [0.92, 0.82, 0.72, 0.68, 0.60, 0.55, 0.52],
        'val_accuracy': [0.54, 0.62, 0.70, 0.74, 0.78, 0.80, 0.81]
    }
    viz_dict['Training History'] = visualizer.plot_training_history(history)
    
    # 5. Eye Aspect Ratio
    ear_values = np.sin(np.linspace(0, 4*np.pi, 300)) * 0.1 + 0.22
    ear_values = np.clip(ear_values, 0.05, 0.4)
    viz_dict['Eye Aspect Ratio'] = visualizer.plot_eye_aspect_ratio(ear_values.tolist(), threshold=0.21)
    
    # 6. Mouth Aspect Ratio
    mar_values = np.random.uniform(0.3, 0.8, 300)
    viz_dict['Mouth Aspect Ratio'] = visualizer.plot_mouth_aspect_ratio(mar_values.tolist(), threshold=0.65)
    
    # 7. Blink Rate
    blink_events = [(i, 0.1) for i in range(0, 300, 20)]
    viz_dict['Blink Rate'] = visualizer.plot_blink_rate(blink_events, window_size=60)
    
    # 8. Prediction Distribution
    predictions = {'Alert': 150, 'Drowsy': 100, 'Yawning': 50}
    viz_dict['Prediction Distribution'] = visualizer.plot_prediction_distribution(predictions)
    
    # 9. Confidence Scores
    confidences = np.random.uniform(0.3, 0.95, 300)
    predictions_list = np.random.choice(['Alert', 'Drowsy', 'Yawning'], 300)
    viz_dict['Confidence Scores'] = visualizer.plot_confidence_scores(
        confidences.tolist(), predictions_list.tolist()
    )
    
    # 10. FPS & Latency
    fps_history = np.random.uniform(25, 31, 300)
    latency_history = np.random.uniform(30, 40, 300)
    viz_dict['FPS & Latency'] = visualizer.plot_fps_latency(fps_history.tolist(), latency_history.tolist())
    
    # Generate summary report
    report_path = visualizer.generate_summary_report(viz_dict)
    
    print("\n" + "="*70)
    print(f"[OK] All visualizations generated successfully!")
    print(f"[*] Output directory: {visualizer.output_dir}")
    print(f"[*] Total plots: {len(viz_dict)}")
    print("="*70 + "\n")
    
    return visualizer, viz_dict


if __name__ == "__main__":
    # Run demo
    visualizer, viz_dict = create_demo_visualizations()
    
    print("\nGenerated Visualizations:")
    for name, path in viz_dict.items():
        print(f"  ✓ {name}")
