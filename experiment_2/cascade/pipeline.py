"""STT -> LLM -> TTS, one turn at a time, timed on the wall clock.

The LLM streams sentences; each is spoken as soon as it arrives, so the learner hears
the first sentence while the rest is still being generated.

Failure policy: if the LLM can't start a reply (no network, every key rate-limited),
the learner hears a short fallback line instead of silence and the turn is left out of
the history. If it fails after some sentences were spoken, the spoken part is kept.
"""

from __future__ import annotations

import time

from .llm import ChatLLM, LLMUnavailable, split_by_script
from .stt import STT, Audio, timed_transcribe
from .tts import TTS
from .types import Segment, Transcript, TurnResult, TurnTiming

TUTOR_DEBUG_KEYS = ("phase", "source", "verdict", "hint_level", "next_hint_level",
                    "problems", "reason")


class CascadePipeline:
    def __init__(self, stt: STT, llm: ChatLLM, tts: TTS, *, play: bool = True,
                 fallback_reply: str = "Sorry, I couldn't think of a reply. Please say that again.",
                 no_speech_reply: str = "I didn't hear anything. Please try again.",
                 max_history_turns: int = 12, stop_words=()) -> None:
        self.stt, self.llm, self.tts = stt, llm, tts
        self.play = play
        self.fallback_reply = fallback_reply
        self.no_speech_reply = no_speech_reply
        self.max_history_turns = max_history_turns
        self.stop_words = {w.lower() for w in stop_words}
        self.history: list[dict] = []   # [{"role": "user"|"model", "text": ...}]

    def _say(self, sentence: str, t0: float, result: TurnResult) -> None:
        """Speak one sentence; record when the turn's first audio started."""
        units = (split_by_script(sentence) if self.tts.splits_by_script
                 else [Segment("mixed", sentence)])
        for unit in units:
            before = time.perf_counter()
            started = self.tts.speak(unit.text, unit.language, play=self.play)
            result.tts_chars += len(unit.text)
            if result.timing.first_audio_ms is None and started is not None:
                result.timing.first_audio_ms = (before - t0) * 1000 + started

    def _finish(self, result: TurnResult, t0: float) -> TurnResult:
        result.done = bool(getattr(self.llm, "done", False))
        debug = getattr(self.llm, "last_debug", None)
        if debug is not None:
            result.tutor = {k: debug.get(k) for k in TUTOR_DEBUG_KEYS if k in debug}
        result.timing.total_ms = (time.perf_counter() - t0) * 1000
        return result

    def greet(self) -> TurnResult | None:
        """Speak the LLM stage's opening line, if it has one (the tutor does)."""
        if not hasattr(self.llm, "opening"):
            return None
        t0 = time.perf_counter()
        result = TurnResult(Transcript("", "en"), [], TurnTiming())
        for sentence in self.llm.opening():
            if result.timing.llm_first_ms is None:
                result.timing.llm_first_ms = (time.perf_counter() - t0) * 1000
            result.reply.append(sentence)
            self._say(sentence, t0, result)
        result.llm = getattr(self.llm, "last_used", None) or self.llm.name
        return self._finish(result, t0)

    def turn(self, audio: Audio) -> TurnResult:
        t0 = time.perf_counter()
        transcript, stt_ms = timed_transcribe(self.stt, audio)
        result = TurnResult(transcript, [], TurnTiming(stt_ms=stt_ms))

        if transcript.text.strip().lower().strip(".!?। ") in self.stop_words:
            result.stopped = True   # checked first, so "quit" is never graded as an answer
        elif not transcript.text:
            result.fallback_used = True
            result.reply.append(self.no_speech_reply)
            self._say(self.no_speech_reply, t0, result)
        else:
            history = self.history[-2 * self.max_history_turns:]
            try:
                for sentence in self.llm.stream(history, transcript.text):
                    if result.timing.llm_first_ms is None:
                        result.timing.llm_first_ms = (time.perf_counter() - t0) * 1000 - stt_ms
                    result.reply.append(sentence)
                    self._say(sentence, t0, result)
            except LLMUnavailable as exc:
                result.llm_error = str(exc)
            result.llm = getattr(self.llm, "last_used", None) or self.llm.name
            if result.reply:
                self.history += [{"role": "user", "text": transcript.text},
                                 {"role": "model", "text": result.reply_text}]
            else:
                result.fallback_used = True
                result.reply.append(self.fallback_reply)
                self._say(self.fallback_reply, t0, result)

        return self._finish(result, t0)
