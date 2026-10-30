"""Speech adapters. STT and TTS are external services; these are their interfaces and
the stubs used by tests and the text terminal. No audio library is imported anywhere
in the server or the core: an adapter for a real service lives outside this package
and only has to satisfy the protocol.
"""

from __future__ import annotations

from typing import Protocol


class SpeechToText(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class StubSTT:
    """Treats the 'audio' as UTF-8 text, so typed input stands in for speech."""

    def transcribe(self, audio: bytes) -> str:
        return audio.decode("utf-8")


class StubTTS:
    """Records what would be spoken and returns it as UTF-8 bytes."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        return text.encode("utf-8")


class StubLaptopClient:
    """Stands in for the laptop (mic and speaker): audio -> STT -> server -> TTS."""

    def __init__(self, server, stt: SpeechToText | None = None, tts: TextToSpeech | None = None) -> None:
        self.server = server
        self.stt = stt or StubSTT()
        self.tts = tts or StubTTS()

    def start(self):
        reply = self.server.start()
        self.tts.synthesize(reply.text)
        return reply

    def speak(self, audio: bytes):
        reply = self.server.handle(self.stt.transcribe(audio))
        self.tts.synthesize(reply.text)
        return reply
