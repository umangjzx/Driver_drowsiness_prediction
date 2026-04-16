"""
Audio Alarm System — Generates and plays alarm sounds for drowsiness alerts.
Uses pygame for cross-platform audio playback and generates WAV alarms programmatically.
"""

import os
import struct
import wave
import math
import time
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALARM_SOUND_PATH, ASSETS_DIR


def generate_alarm_wav(filepath=None, duration=2.0, frequency=880, sample_rate=44100):
    """
    Generate an alarm WAV file programmatically.
    Creates a dual-tone siren effect.
    
    Args:
        filepath: Path to save the WAV file
        duration: Duration in seconds
        frequency: Base frequency in Hz
        sample_rate: Sample rate in Hz
    """
    if filepath is None:
        filepath = ALARM_SOUND_PATH

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    n_samples = int(sample_rate * duration)
    samples = []

    for i in range(n_samples):
        t = i / sample_rate
        # Siren effect: oscillating frequency between freq and freq*1.5
        siren_freq = frequency + (frequency * 0.5) * math.sin(2 * math.pi * 3 * t)
        # Main tone
        sample = 0.6 * math.sin(2 * math.pi * siren_freq * t)
        # Add urgency harmonic
        sample += 0.3 * math.sin(2 * math.pi * siren_freq * 1.5 * t)
        # Pulsing envelope
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 8 * t)
        sample *= envelope
        # Clamp
        sample = max(-1.0, min(1.0, sample))
        samples.append(int(sample * 32767))

    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for s in samples:
            wav_file.writeframes(struct.pack('<h', s))

    print(f"[✓] Alarm sound generated: {filepath}")
    return filepath


class AlarmSystem:
    """
    Manages audio alarm playback for drowsiness detection.
    Uses pygame.mixer for non-blocking audio playback.
    """

    def __init__(self, alarm_path=None):
        self.alarm_path = alarm_path or ALARM_SOUND_PATH
        self.is_playing = False
        self.mixer_initialized = False
        self.last_beep_time = 0.0
        self.beep_interval = 0.35
        self._init_mixer()
        self._ensure_alarm_exists()

    def _init_mixer(self):
        """Initialize pygame mixer for audio playback."""
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self.mixer_initialized = True
            print("[✓] Audio system initialized (pygame)")
        except Exception as e:
            print(f"[⚠] pygame mixer init failed: {e}")
            print("[i] Alarm will use system beep as fallback")
            self.mixer_initialized = False

    def _ensure_alarm_exists(self):
        """Generate alarm sound if it doesn't exist."""
        if not os.path.exists(self.alarm_path):
            print("[i] Generating alarm sound...")
            generate_alarm_wav(self.alarm_path)

    def play(self, intensity=1.0):
        """Start playing the alarm (non-blocking, loops)."""
        intensity = float(np.clip(intensity, 0.0, 1.0))
        volume = max(0.15, intensity)

        if self.mixer_initialized:
            try:
                import pygame
                if not self.is_playing:
                    pygame.mixer.music.load(self.alarm_path)
                    pygame.mixer.music.play(-1)  # Loop indefinitely
                    self.is_playing = True
                    print("[🔔] ALARM TRIGGERED!")
                pygame.mixer.music.set_volume(volume)
            except Exception as e:
                print(f"[⚠] Alarm playback failed: {e}")
                self._fallback_beep(intensity)
        else:
            now = time.time()
            if (not self.is_playing) or (now - self.last_beep_time >= self.beep_interval):
                self._fallback_beep(intensity)
                self.last_beep_time = now
            self.is_playing = True

    def stop(self):
        """Stop the alarm."""
        if not self.is_playing:
            return

        if self.mixer_initialized:
            try:
                import pygame
                pygame.mixer.music.stop()
            except Exception:
                pass

        self.is_playing = False

    def _fallback_beep(self, intensity=1.0):
        """Fallback: use system beep on Windows."""
        try:
            import winsound
            level = float(np.clip(intensity, 0.0, 1.0))
            frequency = int(700 + (500 * level))
            duration_ms = int(120 + (260 * level))
            winsound.Beep(frequency, duration_ms)
        except Exception:
            print("\a")  # Terminal bell

    def cleanup(self):
        """Clean up audio resources."""
        self.stop()
        if self.mixer_initialized:
            try:
                import pygame
                pygame.mixer.quit()
            except Exception:
                pass


# Auto-generate alarm on import if missing
if not os.path.exists(ALARM_SOUND_PATH):
    generate_alarm_wav()
