"""
Training Monitor - Real-time tracking of drowsiness detector training
Monitors saved_models directory for model checkpoints and logs
"""
import os
import time
from pathlib import Path
from datetime import datetime
import json

def monitor_training():
    """Monitor training progress by checking saved_models directory."""
    
    saved_models_dir = "saved_models"
    os.makedirs(saved_models_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("DROWSINESS DETECTOR TRAINING MONITOR")
    print("="*70)
    print(f"Monitoring directory: {saved_models_dir}/")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    model_found = False
    logs_dir = os.path.join(saved_models_dir, "logs")
    
    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking training progress...")
        
        # Check for model checkpoints
        best_model = os.path.join(saved_models_dir, "best_model.keras")
        final_model = os.path.join(saved_models_dir, "final_model.keras")
        
        if os.path.exists(best_model) and not model_found:
            size_mb = os.path.getsize(best_model) / (1024 * 1024)
            print(f"  ✓ Best model checkpoint found! ({size_mb:.1f} MB)")
            model_found = True
        
        if os.path.exists(final_model):
            size_mb = os.path.getsize(final_model) / (1024 * 1024)
            print(f"  ✓ Final model saved! ({size_mb:.1f} MB)")
            print(f"\n[OK] Training complete!")
            return
        
        # Check for TensorBoard logs
        if os.path.exists(logs_dir):
            event_files = list(Path(logs_dir).glob("events.out.tfevents.*"))
            if event_files:
                print(f"  ✓ TensorBoard logs: {len(event_files)} file(s)")
        
        # Check for other model files
        model_files = list(Path(saved_models_dir).glob("*.keras"))
        if model_files:
            print(f"  ✓ Model files: {len(model_files)}")
            for mf in model_files:
                size_mb = mf.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(mf.stat().st_mtime)
                print(f"    - {mf.name}: {size_mb:.1f} MB (modified: {mtime.strftime('%H:%M:%S')})")
        
        print()
        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n[*] Monitoring stopped by user")
