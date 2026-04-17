"""
Production Performance Profiler for Drowsiness Detector

Tracks:
- FPS (frames per second)
- Latency breakdown (face detection, inference, postprocessing)
- Memory usage (RAM, GPU)
- Thermal metrics
- Model efficiency (throughput, latency per backend)
"""
import os
import time
import psutil
import cv2
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import GPUtil
    HAS_GPU_UTIL = True
except ImportError:
    HAS_GPU_UTIL = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


@dataclass
class LatencyMetrics:
    """Stores latency breakdown for a single frame."""
    face_detection: float = 0.0
    face_preprocessing: float = 0.0
    model_inference: float = 0.0
    postprocessing: float = 0.0
    total: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'face_detection_ms': self.face_detection * 1000,
            'face_preprocessing_ms': self.face_preprocessing * 1000,
            'model_inference_ms': self.model_inference * 1000,
            'postprocessing_ms': self.postprocessing * 1000,
            'total_ms': self.total * 1000
        }


class PerformanceProfiler:
    """Comprehensive performance monitoring for detector."""
    
    def __init__(self, window_size: int = 300):
        """
        Initialize profiler.
        
        Args:
            window_size: Number of frames to track for rolling averages
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.memory_snapshots = deque(maxlen=window_size)
        self.gpu_snapshots = deque(maxlen=window_size)
        
        self.frame_count = 0
        self.total_time = 0.0
        self.start_time = time.time()
        
        self.process = psutil.Process()
        self.phase_timers = defaultdict(lambda: deque(maxlen=window_size))
        
    def start_phase(self, phase_name: str) -> float:
        """Start timing a phase."""
        return time.time()
    
    def end_phase(self, phase_name: str, start_time: float):
        """End timing a phase."""
        elapsed = time.time() - start_time
        self.phase_timers[phase_name].append(elapsed)
    
    def record_frame_time(self, frame_time: float):
        """Record frame processing time."""
        self.frame_times.append(frame_time)
        self.total_time += frame_time
        self.frame_count += 1
    
    def record_latency(self, metrics: LatencyMetrics):
        """Record detailed latency breakdown."""
        self.latencies.append(metrics)
    
    def record_memory(self):
        """Record memory usage."""
        try:
            rss = self.process.memory_info().rss / 1024 / 1024  # MB
            self.memory_snapshots.append(rss)
        except:
            pass
    
    def record_gpu_memory(self):
        """Record GPU memory usage."""
        if not HAS_GPU_UTIL or not HAS_TF:
            return
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_mem = gpus[0].memoryUsed
                self.gpu_snapshots.append(gpu_mem)
        except:
            pass
    
    @property
    def fps(self) -> float:
        """Frames per second (rolling average)."""
        if not self.frame_times:
            return 0.0
        frame_times_list = list(self.frame_times)
        if len(frame_times_list) < 2:
            return 0.0
        total_sec = sum(frame_times_list)
        return len(frame_times_list) / total_sec if total_sec > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if not self.frame_times:
            return 0.0
        avg_sec = np.mean(list(self.frame_times))
        return avg_sec * 1000
    
    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency."""
        if len(self.frame_times) < 20:
            return 0.0
        latencies_ms = np.array(list(self.frame_times)) * 1000
        return float(np.percentile(latencies_ms, 95))
    
    @property
    def p99_latency_ms(self) -> float:
        """99th percentile latency."""
        if len(self.frame_times) < 50:
            return 0.0
        latencies_ms = np.array(list(self.frame_times)) * 1000
        return float(np.percentile(latencies_ms, 99))
    
    @property
    def avg_memory_mb(self) -> float:
        """Average memory usage in MB."""
        if not self.memory_snapshots:
            return 0.0
        return float(np.mean(list(self.memory_snapshots)))
    
    @property
    def peak_memory_mb(self) -> float:
        """Peak memory usage."""
        if not self.memory_snapshots:
            return 0.0
        return float(np.max(list(self.memory_snapshots)))
    
    @property
    def avg_gpu_memory_mb(self) -> float:
        """Average GPU memory."""
        if not self.gpu_snapshots:
            return 0.0
        return float(np.mean(list(self.gpu_snapshots)))
    
    def get_phase_stats(self, phase_name: str) -> Dict:
        """Get statistics for a specific phase."""
        if phase_name not in self.phase_timers or not self.phase_timers[phase_name]:
            return {}
        
        times = np.array(list(self.phase_timers[phase_name])) * 1000  # Convert to ms
        
        return {
            'avg_ms': float(np.mean(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'total_calls': len(times)
        }
    
    def get_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_frames': self.frame_count,
            'uptime_seconds': time.time() - self.start_time,
            'fps': {
                'current': self.fps,
                'overall': self.frame_count / (time.time() - self.start_time + 1e-6)
            },
            'latency_ms': {
                'average': self.avg_latency_ms,
                'p95': self.p95_latency_ms,
                'p99': self.p99_latency_ms
            },
            'memory_mb': {
                'average': self.avg_memory_mb,
                'peak': self.peak_memory_mb,
                'gpu_average': self.avg_gpu_memory_mb
            },
            'phases': {
                phase: self.get_phase_stats(phase)
                for phase in self.phase_timers.keys()
            }
        }
        
        return summary
    
    def print_report(self):
        """Print formatted performance report."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Frames: {summary['total_frames']}")
        print(f"Uptime: {summary['uptime_seconds']:.1f}s")
        print(f"Overall FPS: {summary['fps']['overall']:.2f}")
        print(f"Current FPS: {summary['fps']['current']:.2f}")
        
        print("\n--- Latency Breakdown ---")
        print(f"Avg Latency: {summary['latency_ms']['average']:.2f} ms")
        print(f"P95 Latency: {summary['latency_ms']['p95']:.2f} ms")
        print(f"P99 Latency: {summary['latency_ms']['p99']:.2f} ms")
        
        print("\n--- Memory Usage ---")
        print(f"Avg RAM: {summary['memory_mb']['average']:.1f} MB")
        print(f"Peak RAM: {summary['memory_mb']['peak']:.1f} MB")
        if summary['memory_mb']['gpu_average'] > 0:
            print(f"Avg GPU Memory: {summary['memory_mb']['gpu_average']:.1f} MB")
        
        if summary['phases']:
            print("\n--- Phase Timings ---")
            for phase, stats in summary['phases'].items():
                if stats:
                    print(f"{phase}:")
                    print(f"  Avg: {stats['avg_ms']:.2f}ms (calls: {stats['total_calls']})")
                    print(f"  P95: {stats['p95_ms']:.2f}ms")
        
        print("="*60 + "\n")
    
    def plot_performance(self, output_path: str = "performance_report.png"):
        """Plot performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # FPS over time
        if self.frame_times:
            rolling_fps = []
            times = list(self.frame_times)
            for i in range(0, len(times), max(1, len(times) // 100)):
                window = times[max(0, i-30):i+1]
                if window:
                    rolling_fps.append(len(window) / sum(window))
            axes[0, 0].plot(rolling_fps)
            axes[0, 0].set_title('FPS Over Time')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].grid()
        
        # Latency distribution
        if self.frame_times:
            latencies_ms = np.array(list(self.frame_times)) * 1000
            axes[0, 1].hist(latencies_ms, bins=50, edgecolor='black')
            axes[0, 1].set_title('Latency Distribution')
            axes[0, 1].set_xlabel('Latency (ms)')
            axes[0, 1].axvline(np.mean(latencies_ms), color='r', linestyle='--', label='Mean')
            axes[0, 1].legend()
            axes[0, 1].grid()
        
        # Memory over time
        if self.memory_snapshots:
            axes[1, 0].plot(list(self.memory_snapshots))
            axes[1, 0].set_title('Memory Usage')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].grid()
        
        # Phase breakdown
        if self.phase_timers:
            phase_names = list(self.phase_timers.keys())
            phase_avgs = [
                np.mean(list(self.phase_timers[p])) * 1000
                for p in phase_names
            ]
            axes[1, 1].barh(phase_names, phase_avgs)
            axes[1, 1].set_title('Average Phase Latencies')
            axes[1, 1].set_xlabel('Time (ms)')
            axes[1, 1].grid()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"[OK] Performance plot saved to {output_path}")
        plt.close()
    
    def save_report(self, output_dir: str = "performance_reports"):
        """Save detailed performance report as JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"perf_{timestamp}.json")
        
        summary = self.get_summary()
        
        # Add additional metrics
        summary['latency_breakdown'] = []
        for latency in list(self.latencies)[-50:]:  # Last 50 frames
            summary['latency_breakdown'].append(latency.to_dict())
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[OK] Report saved to {report_path}")
        
        # Also save plot
        plot_path = os.path.join(output_dir, f"perf_{timestamp}.png")
        self.plot_performance(plot_path)


class ProfilerContext:
    """Context manager for profiling code blocks."""
    
    def __init__(self, profiler: PerformanceProfiler, phase_name: str):
        self.profiler = profiler
        self.phase_name = phase_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = self.profiler.start_phase(self.phase_name)
        return self
    
    def __exit__(self, *args):
        self.profiler.end_phase(self.phase_name, self.start_time)


if __name__ == "__main__":
    # Example usage
    profiler = PerformanceProfiler()
    
    # Simulate some work
    for i in range(100):
        start = time.time()
        
        # Face detection phase
        with ProfilerContext(profiler, 'face_detection'):
            time.sleep(0.005)
        
        # Inference phase
        with ProfilerContext(profiler, 'inference'):
            time.sleep(0.010)
        
        # Postprocessing phase
        with ProfilerContext(profiler, 'postprocessing'):
            time.sleep(0.002)
        
        frame_time = time.time() - start
        profiler.record_frame_time(frame_time)
        profiler.record_memory()
    
    profiler.print_report()
    profiler.save_report()
