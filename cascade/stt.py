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

import platform
import sys
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


def pick_language(probs: dict[str, float], allowed: Sequence[str], *,
                  prefer: Optional[str] = None, switch_share: float = 0.5,
                  aliases: Optional[dict[str, str]] = None) -> tuple[str, float]:
    """The most likely language among the allowed ones (all, if none allowed).

    ``aliases`` adds one language's probability to another's (Whisper hears spoken Hindi
    as Urdu: {"ur": "hi"}). With ``prefer``, another language is chosen only when it has
    more than ``switch_share`` of the allowed languages' total. Whisper's confidence is
    low for accented English (p 0.2-0.3 on 2026-10-30), and a coin flip sent an English
    answer ("I am from Dehradun, Uttarakhand") to Hindi, where it came out as "आप देरादून
    उत्रगन". In an English lesson, English is the safer guess."""
    merged = dict(probs)
    for src, dst in (aliases or {}).items():
        if src in merged and dst != src:
            merged[dst] = merged.get(dst, 0.0) + merged.pop(src)
    candidates = {lang: p for lang, p in merged.items() if lang in allowed} if allowed else {}
    best = max((candidates or merged).items(), key=lambda lp: lp[1])
    if prefer in candidates and best[0] != prefer:
        total = sum(candidates.values())
        if total <= 0 or best[1] / total <= switch_share:
            return prefer, float(candidates[prefer])
    return best[0], float(best[1])


def language_scores(probs: dict[str, float], allowed: Sequence[str],
                    aliases: Optional[dict[str, str]] = None) -> dict[str, float]:
    """The allowed languages' probabilities (aliases folded in), for the log."""
    out = {lang: float(probs.get(lang, 0.0)) for lang in allowed}
    for src, dst in (aliases or {}).items():
        if dst in out:
            out[dst] += float(probs.get(src, 0.0))
    return {lang: round(p, 3) for lang, p in out.items()}


class WhisperSTT:
    def __init__(self, model: str = "small", *, device: str = "cpu",
                 compute_type: str = "int8", languages: Sequence[str] = ("en", "hi"),
                 beam_size: int = 1, vad_filter: bool = True, prefer: Optional[str] = None,
                 switch_share: float = 0.5, aliases: Optional[dict] = None,
                 cpu_threads: int = 0, num_workers: int = 1) -> None:
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper:{model}"
        self.languages = tuple(languages)
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.choice = {"prefer": prefer, "switch_share": switch_share, "aliases": aliases}
        self._model = WhisperModel(model, device=device, compute_type=compute_type,
                                   cpu_threads=cpu_threads, num_workers=num_workers)

    def _load(self, audio: Audio) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        from faster_whisper import decode_audio

        return decode_audio(str(audio), sampling_rate=SAMPLE_RATE)

    def pick_language(self, probs: list[tuple[str, float]]) -> tuple[str, float]:
        return pick_language(dict(probs), self.languages, **self.choice)

    def transcribe(self, audio: Audio) -> Transcript:
        samples = self._load(audio)
        seconds = len(samples) / SAMPLE_RATE
        if seconds < MIN_SECONDS:
            return Transcript("", self.languages[0] if self.languages else "en", None, seconds)
        _, _, probs = self._model.detect_language(samples, vad_filter=self.vad_filter)
        language, prob = self.pick_language(probs)
        scores = language_scores(dict(probs), self.languages, self.choice["aliases"])
        segments, _ = self._model.transcribe(samples, language=language,
                                             beam_size=self.beam_size,
                                             vad_filter=self.vad_filter,
                                             condition_on_previous_text=False,
                                             without_timestamps=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text, language, round(prob, 3), round(seconds, 2), scores)


class MLXWhisperSTT:
    """Whisper on the Apple GPU. The first call compiles kernels, so the constructor
    runs one warm-up transcription; the first download of a model can take minutes.

    Language detection runs a full encoder pass. For a big model that doubles the cost
    (large-v3-turbo: ~1.1 s detect + ~1.3 s transcribe), so ``detect_model`` lets a small
    model pick the language (~0.2 s) and the big one only transcribe."""

    def __init__(self, model: str = "small", *, languages: Sequence[str] = ("en", "hi"),
                 detect_model: Optional[str] = None, vad: bool = True,
                 warm_up: bool = True, prefer: Optional[str] = None,
                 switch_share: float = 0.5, aliases: Optional[dict] = None) -> None:
        import mlx.core as mx
        from mlx_whisper.load_models import load_model
        from mlx_whisper.transcribe import ModelHolder

        self.repo = MLX_REPOS.get(model, model)
        detect_repo = MLX_REPOS.get(detect_model, detect_model) if detect_model else self.repo
        self.name = f"mlx-whisper:{model}" + (f"+detect:{detect_model}" if detect_repo != self.repo else "")
        self.languages = tuple(languages)
        self.vad = vad
        self.choice = {"prefer": prefer, "switch_share": switch_share, "aliases": aliases}
        self.last_scores: dict = {}
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
        self.last_scores = language_scores(probs, self.languages, self.choice["aliases"])
        return pick_language(probs, self.languages, **self.choice)

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
        return Transcript(text, language, round(prob, 3), seconds, self.last_scores)


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


def is_apple_silicon() -> bool:
    """True if running on Apple Silicon (M-series Mac) where MLX is supported."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def resolve_stt_config(cfg: dict, *, override_engine: Optional[str] = None,
                       override_model: Optional[str] = None) -> tuple[str, str]:
    """Resolve (engine, model) based on the running platform and user configuration."""
    apple_silicon = is_apple_silicon()
    raw_engine = override_engine or cfg.get("engine", "auto")

    if raw_engine in ("auto", None):
        engine = "mlx" if apple_silicon else "faster-whisper"
    elif raw_engine == "mlx" and not apple_silicon:
        if override_engine == "mlx":
            print(f"Note: mlx-whisper is only supported on Apple Silicon (macOS arm64).\n"
                  f"  Switching to faster-whisper for {platform.system()} ({platform.machine()}).")
        engine = "faster-whisper"
    else:
        engine = raw_engine

    if override_model:
        model = override_model
    elif engine == "mlx":
        model = cfg.get("model", "large-v3-turbo")
    else:
        # For faster-whisper (especially on CPU), prefer small for interactive latency
        model = cfg.get("cpu_model") or cfg.get("cpu_fallback_model") or (
            cfg.get("model") if cfg.get("model") != "large-v3-turbo" else "small"
        )
    return engine, model


def load_stt(cfg: dict, *, override_model: Optional[str] = None,
             override_engine: Optional[str] = None) -> STT:
    engine, model = resolve_stt_config(cfg, override_engine=override_engine,
                                       override_model=override_model)
    languages = cfg.get("languages", ["en", "hi"])
    choice = {"prefer": cfg.get("prefer_language"), "switch_share": cfg.get("switch_share", 0.5),
              "aliases": cfg.get("language_aliases")}
    if engine == "mlx":
        try:
            return MLXWhisperSTT(model, languages=languages, detect_model=cfg.get("detect_model"),
                                 vad=cfg.get("vad_filter", True), **choice)
        except ImportError:
            # Usually the project venv isn't active on Apple Silicon.
            print(f"mlx-whisper is not installed for {sys.executable}.\n"
                  "  Is the project venv active? (source ../.venv/bin/activate)\n"
                  "  Falling back to faster-whisper 'small' on CPU (slower).")
            model = override_model or cfg.get("cpu_fallback_model", "small")
    return WhisperSTT(model, device=cfg.get("device", "cpu"),
                      compute_type=cfg.get("compute_type", "int8"), languages=languages,
                      beam_size=cfg.get("beam_size", 1), vad_filter=cfg.get("vad_filter", True),
                      cpu_threads=cfg.get("cpu_threads", 0), num_workers=cfg.get("num_workers", 1),
                      **choice)
