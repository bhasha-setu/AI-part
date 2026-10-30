"""Push-to-talk microphone capture: Enter to start, Enter to stop.

Push-to-talk (not always-on listening) keeps the mic closed while the reply plays, so
the tutor never transcribes its own voice. macOS asks for mic permission for the
terminal app the first time.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def record_push_to_talk(ask=input) -> np.ndarray:
    """Record until the learner presses Enter. Returns float32 mono at 16 kHz."""
    import sounddevice as sd

    frames: list[np.ndarray] = []

    def callback(indata, n_frames, time_info, status):
        frames.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=callback):
        ask("  ● recording… press Enter to stop ")
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)


def save_wav(samples: np.ndarray, path: Path) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
