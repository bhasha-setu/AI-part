"""The data that moves between the three stages of the cascade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Transcript:
    """What STT heard. ``language`` is a Whisper code ("en", "hi"); ``language_prob``
    is Whisper's confidence in it (None when the text was typed)."""
    text: str
    language: str
    language_prob: Optional[float] = None
    audio_seconds: Optional[float] = None


@dataclass
class Segment:
    """One run of reply text in a single language, so each can get its own TTS voice.
    The tutor in experiment_1 already replies in English/Hindi segments, so it can
    later take the LLM stage's place without changing STT or TTS."""
    language: str
    text: str


@dataclass
class TurnTiming:
    """Milliseconds for one turn, all measured on the wall clock from the moment the
    learner's input ends.

    stt_ms           transcription
    llm_first_ms     STT end -> first reply sentence from the LLM
    first_audio_ms   input end -> first reply audio starts: what the learner waits
    total_ms         input end -> last audio finished
    """
    stt_ms: float = 0.0
    llm_first_ms: Optional[float] = None
    first_audio_ms: Optional[float] = None
    total_ms: float = 0.0


@dataclass
class TurnResult:
    transcript: Transcript
    reply: list[str]                       # sentences, in the order they were spoken
    timing: TurnTiming
    llm: Optional[str] = None              # which backend answered
    llm_error: Optional[str] = None
    tts_chars: int = 0                     # characters sent to TTS (ElevenLabs bills these)
    fallback_used: bool = False

    @property
    def reply_text(self) -> str:
        return " ".join(self.reply)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reply_text"] = self.reply_text
        return d
