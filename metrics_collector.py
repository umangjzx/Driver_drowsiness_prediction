"""
Real-Time Metrics Collection for Drowsiness Detection System

Collects performance metrics during real-time detection including:
- Eye Aspect Ratio (EAR) values
- Mouth Aspect Ratio (MAR) values
- Blink events
- Confidence scores
- Predictions
- FPS and latency measurements

These metrics can be used to generate visualizations post-detection.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np


class MetricsCollector:
    """Collect real-time detection metrics for visualization and analysis."""
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_samples: Maximum number of samples to keep in memory
        """
        self.max_samples = max_samples
        self.reset()
    
    def reset(self):
        """Reset all collected metrics."""
        self.timestamps = []
        self.ear_values = []
        self.mar_values = []
        self.predictions = []  # List of class names
        self.confidences = []  # List of confidence scores
        self.fps_values = []
        self.latency_ms_values = []
        self.blink_events = []  # List of (frame_num, duration) tuples
        self.frames_processed = 0
        self.start_time = None
    
    def add_frame_metrics(
        self,
        ear: Optional[float] = None,
        mar: Optional[float] = None,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        fps: Optional[float] = None,
        latency_ms: Optional[float] = None,
        blink_event: Optional[Tuple] = None
    ):
        """
        Add metrics for a single frame.
        
        Args:
            ear: Eye Aspect Ratio value
            mar: Mouth Aspect Ratio value
            prediction: Predicted class name (Alert, Drowsy, Yawning)
            confidence: Confidence score (0-1)
            fps: Current frames per second
            latency_ms: Inference latency in milliseconds
            blink_event: (frame_num, duration) tuple for blink
        """
        if self.start_time is None:
            self.start_time = datetime.now()
        
        # Track frame number
        self.frames_processed += 1
        timestamp = (datetime.now() - self.start_time).total_seconds()
        self.timestamps.append(timestamp)
        
        # Add optional metrics
        if ear is not None:
            self.ear_values.append(ear)
        if mar is not None:
            self.mar_values.append(mar)
        if prediction is not None:
            self.predictions.append(prediction)
        if confidence is not None:
            self.confidences.append(confidence)
        if fps is not None:
            self.fps_values.append(fps)
        if latency_ms is not None:
            self.latency_ms_values.append(latency_ms)
        if blink_event is not None:
            self.blink_events.append(blink_event)
        
        # Limit memory usage
        if self.frames_processed > self.max_samples:
            self._trim_oldest_samples()
    
    def _trim_oldest_samples(self):
        """Remove oldest 10% of samples to prevent memory overflow."""
        trim_idx = self.max_samples // 10
        
        self.timestamps = self.timestamps[trim_idx:]
        if self.ear_values:
            self.ear_values = self.ear_values[trim_idx:]
        if self.mar_values:
            self.mar_values = self.mar_values[trim_idx:]
        if self.predictions:
            self.predictions = self.predictions[trim_idx:]
        if self.confidences:
            self.confidences = self.confidences[trim_idx:]
        if self.fps_values:
            self.fps_values = self.fps_values[trim_idx:]
        if self.latency_ms_values:
            self.latency_ms_values = self.latency_ms_values[trim_idx:]
    
    def get_statistics(self) -> Dict:
        """
        Get statistical summary of collected metrics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_frames': self.frames_processed,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
        }
        
        # EAR statistics
        if self.ear_values:
            stats['ear'] = {
                'mean': np.mean(self.ear_values),
                'std': np.std(self.ear_values),
                'min': np.min(self.ear_values),
                'max': np.max(self.ear_values)
            }
        
        # MAR statistics
        if self.mar_values:
            stats['mar'] = {
                'mean': np.mean(self.mar_values),
                'std': np.std(self.mar_values),
                'min': np.min(self.mar_values),
                'max': np.max(self.mar_values)
            }
        
        # Prediction distribution
        if self.predictions:
            from collections import Counter
            pred_counts = Counter(self.predictions)
            total = len(self.predictions)
            stats['predictions'] = {
                'distribution': dict(pred_counts),
                'percentages': {k: (v/total)*100 for k, v in pred_counts.items()}
            }
        
        # Confidence statistics
        if self.confidences:
            stats['confidence'] = {
                'mean': np.mean(self.confidences),
                'std': np.std(self.confidences),
                'min': np.min(self.confidences),
                'max': np.max(self.confidences)
            }
        
        # FPS statistics
        if self.fps_values:
            stats['fps'] = {
                'mean': np.mean(self.fps_values),
                'std': np.std(self.fps_values),
                'min': np.min(self.fps_values),
                'max': np.max(self.fps_values)
            }
        
        # Latency statistics
        if self.latency_ms_values:
            stats['latency_ms'] = {
                'mean': np.mean(self.latency_ms_values),
                'std': np.std(self.latency_ms_values),
                'min': np.min(self.latency_ms_values),
                'max': np.max(self.latency_ms_values)
            }
        
        # Blink statistics
        if self.blink_events:
            blink_count = len(self.blink_events)
            duration_sec = stats['duration_seconds']
            blinks_per_minute = (blink_count / duration_sec * 60) if duration_sec > 0 else 0
            stats['blinks'] = {
                'total_count': blink_count,
                'per_minute': blinks_per_minute,
                'average_duration': np.mean([b[1] for b in self.blink_events]) if self.blink_events else 0
            }
        
        return stats
    
    def save_metrics(self, output_dir: str = "detection_metrics"):
        """
        Save collected metrics to JSON file.
        
        Args:
            output_dir: Directory to save metrics
            
        Returns:
            Path to saved JSON file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        metrics = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_frames': self.frames_processed,
                'duration_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            },
            'data': {
                'timestamps': self.timestamps,
                'ear_values': self.ear_values,
                'mar_values': self.mar_values,
                'predictions': self.predictions,
                'confidences': self.confidences,
                'fps_values': self.fps_values,
                'latency_ms_values': self.latency_ms_values,
                'blink_events': self.blink_events
            },
            'statistics': self.get_statistics()
        }
        
        output_path = Path(output_dir) / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"[*] Metrics saved to: {output_path}")
        
        return str(output_path)
    
    def generate_visualizations(self, output_dir: str = "visualization_results"):
        """
        Generate visualizations from collected metrics.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        from visualizations import DrowsinessVisualizer
        
        print(f"\n[*] Generating visualizations from collected metrics...")
        
        visualizer = DrowsinessVisualizer(output_dir=output_dir)
        viz_dict = {}
        
        # Generate plots if we have data
        if self.ear_values:
            viz_dict['Eye Aspect Ratio'] = visualizer.plot_eye_aspect_ratio(
                self.ear_values, self.timestamps
            )
        
        if self.mar_values:
            viz_dict['Mouth Aspect Ratio'] = visualizer.plot_mouth_aspect_ratio(
                self.mar_values, self.timestamps
            )
        
        if self.blink_events:
            viz_dict['Blink Rate'] = visualizer.plot_blink_rate(self.blink_events)
        
        if self.predictions:
            from collections import Counter
            pred_counts = Counter(self.predictions)
            viz_dict['Prediction Distribution'] = visualizer.plot_prediction_distribution(
                dict(pred_counts)
            )
        
        if self.confidences and self.predictions:
            viz_dict['Confidence Scores'] = visualizer.plot_confidence_scores(
                self.confidences, self.predictions
            )
        
        if self.fps_values and self.latency_ms_values:
            viz_dict['FPS & Latency'] = visualizer.plot_fps_latency(
                self.fps_values, self.latency_ms_values
            )
        
        # Generate summary report
        if viz_dict:
            visualizer.generate_summary_report(viz_dict)
        
        print(f"[OK] Generated {len(viz_dict)} visualizations")
        
        return viz_dict
    
    def print_summary(self):
        """Print summary of collected metrics to console."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("DETECTION METRICS SUMMARY")
        print("="*70)
        
        print(f"\n[Frames & Duration]")
        print(f"  Total Frames: {stats['total_frames']}")
        print(f"  Duration: {stats['duration_seconds']:.1f} seconds")
        
        if 'ear' in stats:
            print(f"\n[Eye Aspect Ratio (EAR)]")
            print(f"  Mean:  {stats['ear']['mean']:.3f}")
            print(f"  Std:   {stats['ear']['std']:.3f}")
            print(f"  Min:   {stats['ear']['min']:.3f}")
            print(f"  Max:   {stats['ear']['max']:.3f}")
        
        if 'mar' in stats:
            print(f"\n[Mouth Aspect Ratio (MAR)]")
            print(f"  Mean:  {stats['mar']['mean']:.3f}")
            print(f"  Std:   {stats['mar']['std']:.3f}")
            print(f"  Min:   {stats['mar']['min']:.3f}")
            print(f"  Max:   {stats['mar']['max']:.3f}")
        
        if 'predictions' in stats:
            print(f"\n[Predictions]")
            for pred, count in stats['predictions']['distribution'].items():
                pct = stats['predictions']['percentages'][pred]
                print(f"  {pred:10s}: {count:5d} ({pct:5.1f}%)")
        
        if 'confidence' in stats:
            print(f"\n[Confidence Scores]")
            print(f"  Mean:  {stats['confidence']['mean']:.3f}")
            print(f"  Std:   {stats['confidence']['std']:.3f}")
            print(f"  Min:   {stats['confidence']['min']:.3f}")
            print(f"  Max:   {stats['confidence']['max']:.3f}")
        
        if 'fps' in stats:
            print(f"\n[Frame Rate (FPS)]")
            print(f"  Mean:  {stats['fps']['mean']:.1f}")
            print(f"  Std:   {stats['fps']['std']:.1f}")
            print(f"  Min:   {stats['fps']['min']:.1f}")
            print(f"  Max:   {stats['fps']['max']:.1f}")
        
        if 'latency_ms' in stats:
            print(f"\n[Latency (milliseconds)]")
            print(f"  Mean:  {stats['latency_ms']['mean']:.1f} ms")
            print(f"  Std:   {stats['latency_ms']['std']:.1f} ms")
            print(f"  Min:   {stats['latency_ms']['min']:.1f} ms")
            print(f"  Max:   {stats['latency_ms']['max']:.1f} ms")
        
        if 'blinks' in stats:
            print(f"\n[Blink Statistics]")
            print(f"  Total Blinks:        {stats['blinks']['total_count']}")
            print(f"  Blinks per Minute:   {stats['blinks']['per_minute']:.1f}")
            print(f"  Avg Blink Duration:  {stats['blinks']['average_duration']:.2f}s")
        
        print("\n" + "="*70 + "\n")


# Convenience functions for easy integration

def create_metrics_collector(max_samples: int = 10000) -> MetricsCollector:
    """Create a metrics collector instance."""
    return MetricsCollector(max_samples=max_samples)
