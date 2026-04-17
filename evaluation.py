"""
Production-Grade Evaluation Framework for Drowsiness Detector

Computes precision, recall, F1, confusion matrix, ROC-AUC, and per-class metrics
from labeled test videos or frame sequences.
"""
import os
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List
import json
from datetime import datetime

from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from realtime_detector_enhanced import EnhancedDrowsinessDetector
from config import CLASS_NAMES, IMG_SIZE


class DrowsinessDetectorEvaluator:
    """Comprehensive evaluation suite for drowsiness detector."""
    
    def __init__(self, detector: EnhancedDrowsinessDetector):
        """Initialize evaluator with a detector instance."""
        self.detector = detector
        self.predictions = []
        self.ground_truths = []
        self.confidences = defaultdict(list)
        self.results = {}
        
    def evaluate_video(
        self,
        video_path: str,
        ground_truth_labels: List[int] | None = None,
        stride: int = 1
    ) -> Dict:
        """
        Evaluate detector on a video file.
        
        Args:
            video_path: Path to video file
            ground_truth_labels: Frame-by-frame labels (0=Alert, 1=Drowsy, 2=Yawning)
                                If None, no metrics computed
            stride: Process every Nth frame
            
        Returns:
            Dictionary with frame predictions and metadata
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        frame_idx = 0
        frame_predictions = []
        face_detected_ratio = 0
        total_frames_processed = 0
        
        print(f"[*] Evaluating video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % stride != 0:
                continue
                
            total_frames_processed += 1
            
            # Resize for display (optional)
            if frame.shape[0] > 1080:
                scale = 1080 / frame.shape[0]
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), 1080))
            
            # Run detector
            try:
                result = self.detector.process_frame(frame)
                state = result.get('state', 'Alert')
                confidence = result.get('confidence', 0.0)
                face_detected = result.get('face_detected', False)
                
                # Map state to class index
                class_idx = CLASS_NAMES.index(state) if state in CLASS_NAMES else 0
                
                frame_predictions.append({
                    'frame': frame_idx,
                    'state': state,
                    'class_idx': class_idx,
                    'confidence': float(confidence),
                    'face_detected': face_detected
                })
                
                if face_detected:
                    face_detected_ratio += 1
                    
            except Exception as e:
                print(f"[!] Error processing frame {frame_idx}: {e}")
                frame_predictions.append({
                    'frame': frame_idx,
                    'state': 'Alert',
                    'class_idx': 0,
                    'confidence': 0.0,
                    'face_detected': False
                })
        
        cap.release()
        
        results = {
            'video': video_path,
            'total_frames': frame_idx,
            'processed_frames': total_frames_processed,
            'face_detection_ratio': face_detected_ratio / max(total_frames_processed, 1),
            'predictions': frame_predictions
        }
        
        # Compute metrics if ground truth provided
        if ground_truth_labels is not None:
            if len(ground_truth_labels) != len(frame_predictions):
                print(f"[!] Warning: label count ({len(ground_truth_labels)}) != "
                      f"prediction count ({len(frame_predictions)})")
            
            preds = [p['class_idx'] for p in frame_predictions]
            truths = ground_truth_labels[:len(preds)]
            confs = [p['confidence'] for p in frame_predictions]
            
            metrics = self._compute_metrics(truths, preds, confs)
            results['metrics'] = metrics
            
            self.ground_truths.extend(truths)
            self.predictions.extend(preds)
            self.confidences['all'].extend(confs)
        
        return results
    
    def evaluate_frame_directory(
        self,
        frame_dir: str,
        labels_file: str | None = None
    ) -> Dict:
        """
        Evaluate on a directory of frame images with optional labels.
        
        Args:
            frame_dir: Directory containing *.jpg or *.png frames
            labels_file: JSON file with frame -> label mapping
            
        Returns:
            Evaluation results dictionary
        """
        frame_paths = sorted(
            list(Path(frame_dir).glob("*.jpg")) + 
            list(Path(frame_dir).glob("*.png"))
        )
        
        if not frame_paths:
            raise IOError(f"No frames found in {frame_dir}")
        
        # Load labels if provided
        labels_map = {}
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                labels_map = json.load(f)
        
        frame_predictions = []
        print(f"[*] Evaluating {len(frame_paths)} frames from {frame_dir}")
        
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            
            frame_name = frame_path.stem
            
            try:
                result = self.detector.process_frame(frame)
                state = result.get('state', 'Alert')
                confidence = result.get('confidence', 0.0)
                
                class_idx = CLASS_NAMES.index(state) if state in CLASS_NAMES else 0
                
                pred_dict = {
                    'frame': frame_name,
                    'state': state,
                    'class_idx': class_idx,
                    'confidence': float(confidence)
                }
                
                # Add ground truth if available
                if frame_name in labels_map:
                    gt_label = labels_map[frame_name]
                    if isinstance(gt_label, str):
                        gt_idx = CLASS_NAMES.index(gt_label) if gt_label in CLASS_NAMES else 0
                    else:
                        gt_idx = int(gt_label)
                    pred_dict['ground_truth_idx'] = gt_idx
                    self.ground_truths.append(gt_idx)
                    self.predictions.append(class_idx)
                    self.confidences['all'].append(float(confidence))
                
                frame_predictions.append(pred_dict)
                
            except Exception as e:
                print(f"[!] Error processing {frame_name}: {e}")
        
        results = {
            'directory': frame_dir,
            'total_frames': len(frame_predictions),
            'predictions': frame_predictions
        }
        
        if self.ground_truths:
            metrics = self._compute_metrics(
                self.ground_truths, self.predictions,
                self.confidences['all']
            )
            results['metrics'] = metrics
        
        return results
    
    def _compute_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        confidences: List[float] | None = None
    ) -> Dict:
        """Compute all evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['precision'] = float(precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        ))
        metrics['recall'] = float(recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        ))
        metrics['f1'] = float(f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        ))
        
        # Per-class metrics
        report = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0
        )
        metrics['per_class'] = {
            CLASS_NAMES[i]: {
                'precision': report[CLASS_NAMES[i]]['precision'],
                'recall': report[CLASS_NAMES[i]]['recall'],
                'f1': report[CLASS_NAMES[i]]['f1-score'],
                'support': int(report[CLASS_NAMES[i]]['support'])
            }
            for i in range(len(CLASS_NAMES))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC-AUC (one-vs-rest for multiclass)
        try:
            if confidences and len(set(y_true)) > 1:
                # Use confidence as probability estimate
                metrics['auc'] = float(roc_auc_score(
                    y_true, np.array(confidences), average='weighted', multi_class='ovr'
                ))
        except Exception:
            metrics['auc'] = None
        
        # Accuracy
        metrics['accuracy'] = float(np.mean(np.array(y_pred) == np.array(y_true)))
        
        return metrics
    
    def generate_report(self, output_dir: str = "evaluation_reports") -> str:
        """
        Generate comprehensive evaluation report with visualizations.
        
        Returns:
            Path to generated report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"report_{timestamp}.json")
        
        report = {
            'timestamp': timestamp,
            'detector_config': {
                'model': getattr(self.detector, 'model_path', 'unknown'),
                'use_ensemble': self.detector.use_ensemble,
                'use_temporal_smoothing': self.detector.use_temporal_smoothing,
                'use_huggingface': self.detector.use_huggingface,
            },
            'summary': {
                'total_predictions': len(self.predictions),
                'total_ground_truths': len(self.ground_truths),
                'classes': CLASS_NAMES
            }
        }
        
        if self.predictions and self.ground_truths:
            metrics = self._compute_metrics(
                self.ground_truths, self.predictions,
                self.confidences.get('all', [])
            )
            report['metrics'] = metrics
            
            # Generate confusion matrix visualization
            self._plot_confusion_matrix(
                self.ground_truths, self.predictions,
                os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
            )
            
            # Generate metrics summary plot
            self._plot_metrics_summary(
                metrics,
                os.path.join(output_dir, f"metrics_{timestamp}.png")
            )
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[OK] Report saved to: {report_path}")
        return report_path
    
    def _plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        save_path: str
    ):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
        )
        plt.title('Confusion Matrix')
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[OK] Confusion matrix saved to: {save_path}")
    
    def _plot_metrics_summary(self, metrics: Dict, save_path: str):
        """Plot metrics summary."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Overall metrics
        overall = {
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Accuracy': metrics['accuracy']
        }
        axes[0].bar(overall.keys(), overall.values(), color='steelblue')
        axes[0].set_ylim([0, 1])
        axes[0].set_title('Overall Metrics')
        axes[0].set_ylabel('Score')
        for i, v in enumerate(overall.values()):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Per-class F1
        f1_scores = {
            CLASS_NAMES[i]: metrics['per_class'][CLASS_NAMES[i]]['f1']
            for i in range(len(CLASS_NAMES))
        }
        axes[1].bar(f1_scores.keys(), f1_scores.values(), color='forestgreen')
        axes[1].set_ylim([0, 1])
        axes[1].set_title('Per-Class F1 Scores')
        axes[1].set_ylabel('F1 Score')
        
        # Support
        support = {
            CLASS_NAMES[i]: metrics['per_class'][CLASS_NAMES[i]]['support']
            for i in range(len(CLASS_NAMES))
        }
        axes[2].bar(support.keys(), support.values(), color='coral')
        axes[2].set_title('Class Distribution (Support)')
        axes[2].set_ylabel('Samples')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[OK] Metrics summary saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate drowsiness detector")
    parser.add_argument('--video', type=str, help="Path to test video")
    parser.add_argument('--frames-dir', type=str, help="Directory with test frames")
    parser.add_argument('--labels', type=str, help="JSON labels file")
    parser.add_argument('--output-dir', type=str, default="evaluation_reports",
                       help="Output directory for reports")
    parser.add_argument('--model', type=str, help="Model path")
    parser.add_argument('--hf', action='store_true', help="Use Hugging Face backend")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = EnhancedDrowsinessDetector(
        model_path=args.model,
        use_huggingface=args.hf
    )
    
    # Run evaluation
    evaluator = DrowsinessDetectorEvaluator(detector)
    
    if args.video:
        print(f"[*] Evaluating video: {args.video}")
        results = evaluator.evaluate_video(args.video)
        print(f"[OK] Evaluated {len(results['predictions'])} frames")
    
    elif args.frames_dir:
        print(f"[*] Evaluating frame directory: {args.frames_dir}")
        results = evaluator.evaluate_frame_directory(args.frames_dir, args.labels)
        print(f"[OK] Evaluated {len(results['predictions'])} frames")
    
    # Generate report
    evaluator.generate_report(args.output_dir)
