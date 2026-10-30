"""Stage 1: speech to text with Whisper.

Two engines, same behaviour:
    MLXWhisperSTT     mlx-whisper on the Apple GPU (M-series Macs). ~5x faster than CPU.
    WhisperSTT        faster-whisper on CPU (CTranslate2). Fallback for non-Apple machines.

The model is multilingual and never forced to English: a Hindi speaker's answer may be
Hindi, English or a mix, and forcing "en" turns spoken Hindi into nonsense English.
Language is detected first, then restricted to ``languages`` (Whisper often hears Hindi
as Urdu; the closest allowed language is used instead).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union

import numpy as np

from .types import Transcript

SAMPLE_RATE = 16000
MIN_SECONDS = 0.3
Audio = Union[np.ndarray, str, Path]   # float32 mono 16 kHz, or a path to an audio file

# Short names for config and --whisper-model.
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


class STT(Protocol):
    name: str

    def transcribe(self, audio: Audio) -> Transcript: ...


def trim_to_speech(samples: np.ndarray) -> np.ndarray:
    """Cut leading and trailing non-speech with Silero VAD (shipped with faster-whisper,
    a few ms on CPU). Returns an empty array when there is no speech at all: Whisper
    invents text on silence ("Thank you.") and push-to-talk always records some."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    spans = get_speech_timestamps(samples, VadOptions(min_silence_duration_ms=500,
                                                      speech_pad_ms=200))
    if not spans:
        return samples[:0]
    return samples[spans[0]["start"]:spans[-1]["end"]]


def pick_language(probs: dict[str, float], allowed: Sequence[str]) -> tuple[str, float]:
    """The most likely language among the allowed ones (all, if none allowed)."""
    candidates = {lang: p for lang, p in probs.items() if lang in allowed} if allowed else {}
    best = max((candidates or probs).items(), key=lambda lp: lp[1])
    return best[0], float(best[1])


class WhisperSTT:
    def __init__(self, model: str = "small", *, device: str = "cpu",
                 compute_type: str = "int8", languages: Sequence[str] = ("en", "hi"),
                 beam_size: int = 1, vad_filter: bool = True) -> None:
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper:{model}"
        self.languages = tuple(languages)
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def _load(self, audio: Audio) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        from faster_whisper import decode_audio

        return decode_audio(str(audio), sampling_rate=SAMPLE_RATE)

    def pick_language(self, probs: list[tuple[str, float]]) -> tuple[str, float]:
        return pick_language(dict(probs), self.languages)

    def transcribe(self, audio: Audio) -> Transcript:
        samples = self._load(audio)
        seconds = len(samples) / SAMPLE_RATE
        if seconds < MIN_SECONDS:
            return Transcript("", self.languages[0] if self.languages else "en", None, seconds)
        _, _, probs = self._model.detect_language(samples, vad_filter=self.vad_filter)
        language, prob = self.pick_language(probs)
        segments, _ = self._model.transcribe(samples, language=language,
                                             beam_size=self.beam_size,
                                             vad_filter=self.vad_filter,
                                             condition_on_previous_text=False,
                                             without_timestamps=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text, language, round(prob, 3), round(seconds, 2))


class MLXWhisperSTT:
    """Whisper on the Apple GPU. The first call compiles kernels, so the constructor
    runs one warm-up transcription; the first download of a model can take minutes.

    Language detection runs a full encoder pass. For a big model that doubles the cost
    (large-v3-turbo: ~1.1 s detect + ~1.3 s transcribe), so ``detect_model`` lets a small
    model pick the language (~0.2 s) and the big one only transcribe."""

    def __init__(self, model: str = "small", *, languages: Sequence[str] = ("en", "hi"),
                 detect_model: Optional[str] = None, vad: bool = True,
                 warm_up: bool = True) -> None:
        import mlx.core as mx
        from mlx_whisper.load_models import load_model
        from mlx_whisper.transcribe import ModelHolder

        self.repo = MLX_REPOS.get(model, model)
        detect_repo = MLX_REPOS.get(detect_model, detect_model) if detect_model else self.repo
        self.name = f"mlx-whisper:{model}" + (f"+detect:{detect_model}" if detect_repo != self.repo else "")
        self.languages = tuple(languages)
        self.vad = vad
        if detect_repo == self.repo:
            self._detector = ModelHolder.get_model(self.repo, mx.float16)
        else:
            # Held here, not in mlx_whisper's one-model cache, which would reload weights
            # every time detection and transcription alternate between two models.
            self._detector = load_model(detect_repo, dtype=mx.float16)
        if warm_up:   # with VAD off, so the models really run once
            self._run(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def _load(self, audio: Audio) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        from mlx_whisper.audio import load_audio

        return np.asarray(load_audio(str(audio)), dtype=np.float32)

    def _detect(self, samples: np.ndarray) -> tuple[str, float]:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        model = self._detector
        mel = log_mel_spectrogram(samples, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
        return pick_language(probs, self.languages)

    def _run(self, samples: np.ndarray) -> tuple[str, str, float]:
        import mlx_whisper

        language, prob = self._detect(samples)
        result = mlx_whisper.transcribe(samples, path_or_hf_repo=self.repo, language=language,
                                        condition_on_previous_text=False,
                                        without_timestamps=True, verbose=None)
        return result["text"].strip(), language, prob

    def transcribe(self, audio: Audio) -> Transcript:
        samples = self._load(audio)
        seconds = round(len(samples) / SAMPLE_RATE, 2)
        if self.vad:
            samples = trim_to_speech(samples)
        if len(samples) / SAMPLE_RATE < MIN_SECONDS:
            return Transcript("", self.languages[0] if self.languages else "en", None, seconds)
        text, language, prob = self._run(samples)
        return Transcript(text, language, round(prob, 3), seconds)


class TypedSTT:
    """Stands in for STT when the learner types (``--text``) or in tests."""
    name = "typed"

    def __init__(self, language: str = "en") -> None:
        self.language = language

    def transcribe(self, audio: Audio) -> Transcript:
        return Transcript(str(audio), self.language)


def timed_transcribe(stt: STT, audio: Audio) -> tuple[Transcript, float]:
    start = time.perf_counter()
    transcript = stt.transcribe(audio)
    return transcript, (time.perf_counter() - start) * 1000


def load_stt(cfg: dict, *, override_model: Optional[str] = None,
             override_engine: Optional[str] = None) -> STT:
    engine = override_engine or cfg.get("engine", "mlx")
    model = override_model or cfg.get("model", "small")
    languages = cfg.get("languages", ["en", "hi"])
    if engine == "mlx":
        try:
            return MLXWhisperSTT(model, languages=languages, detect_model=cfg.get("detect_model"),
                                 vad=cfg.get("vad_filter", True))
        except ImportError:
            print("mlx-whisper not available; falling back to faster-whisper on CPU")
    return WhisperSTT(model, device=cfg.get("device", "cpu"),
                      compute_type=cfg.get("compute_type", "int8"), languages=languages,
                      beam_size=cfg.get("beam_size", 1), vad_filter=cfg.get("vad_filter", True))
