"""Stage 3: text to speech.

Every engine has ``speak(text, language, play=True) -> ms until audio starts`` and
blocks until the audio has finished. With ``play=False`` it still synthesises (so the
time is real) but plays nothing: for benchmarks.

    MacNativeTTS     macOS voices in-process (NSSpeechSynthesizer via pyobjc): voices
                     stay loaded, so speech starts in ~20-80 ms. One voice per language,
                     so the pipeline splits replies by script first.
    MacSayTTS        the `say` command. Same voices, but ~700 ms per call to start a
                     process and load the voice. Fallback when pyobjc is missing.
    ElevenLabsTTS    ElevenLabs streaming API (Flash v2.5): a multilingual voice reads
                     mixed Hindi/English text in one call, so no script split. Audio is
                     played as it arrives. Costs credits per character.
    SilentTTS        --no-tts and tests.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Protocol


class TTS(Protocol):
    name: str
    splits_by_script: bool   # True: give it one language at a time

    def speak(self, text: str, language: str, play: bool = True) -> Optional[float]: ...


class MacNativeTTS:
    name = "macos-native"
    splits_by_script = True

    def __init__(self, voices: dict[str, str], rate: int = 175) -> None:
        from AppKit import NSSpeechSynthesizer

        available = list(NSSpeechSynthesizer.availableVoices())
        self._synths = {}
        for lang, voice in voices.items():
            ident = next((v for v in available if v.endswith("." + voice) or v == voice), None)
            if ident is None:
                raise RuntimeError(f"macOS voice {voice!r} not installed")
            synth = NSSpeechSynthesizer.alloc().initWithVoice_(ident)
            synth.setRate_(rate)
            self._synths[lang] = synth
        self._tmp = tempfile.TemporaryDirectory()

    def _wait(self, synth, until_speaking: bool) -> None:
        from Foundation import NSDate, NSRunLoop

        loop = NSRunLoop.currentRunLoop()
        while bool(synth.isSpeaking()) != until_speaking:
            loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.005))

    def speak(self, text: str, language: str, play: bool = True) -> Optional[float]:
        synth = self._synths.get(language) or self._synths["en"]
        start = time.perf_counter()
        if play:
            synth.startSpeakingString_(text)
        else:
            from Foundation import NSURL

            synth.startSpeakingString_toURL_(text, NSURL.fileURLWithPath_(
                str(Path(self._tmp.name) / "out.aiff")))
        self._wait(synth, until_speaking=True)
        started = (time.perf_counter() - start) * 1000
        self._wait(synth, until_speaking=False)
        if not play:
            # isSpeaking turns on at once when writing to a file, so report the whole
            # synthesis time instead: an upper bound on the time to first sound.
            return (time.perf_counter() - start) * 1000
        return started


class MacSayTTS:
    name = "macos-say"
    splits_by_script = True

    def __init__(self, voices: dict[str, str], rate: int = 175) -> None:
        self.voices = voices
        self.rate = rate

    def speak(self, text: str, language: str, play: bool = True) -> Optional[float]:
        voice = self.voices.get(language, self.voices.get("en"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.wav"
            start = time.perf_counter()
            subprocess.run(["say", "-v", voice, "-r", str(self.rate), "-o", str(path),
                            "--data-format=LEI16@22050", text], check=True, capture_output=True)
            started = (time.perf_counter() - start) * 1000
            if play:
                subprocess.run(["afplay", str(path)], check=True)
        return started


class ElevenLabsTTS:
    """Streams 16 kHz PCM from ElevenLabs and plays chunks as they arrive. The key is
    passed in and only ever sent as the xi-api-key header."""
    name = "elevenlabs"
    splits_by_script = False
    SAMPLE_RATE = 16000

    def __init__(self, api_key: str, voice_id: str, model: str = "eleven_flash_v2_5",
                 timeout_seconds: float = 15, transport=None) -> None:
        import httpx

        self.voice_id = voice_id
        self.model = model
        self.name = f"elevenlabs:{model}"
        self._client = httpx.Client(base_url="https://api.elevenlabs.io",
                                    headers={"xi-api-key": api_key},
                                    timeout=timeout_seconds, transport=transport)

    def speak(self, text: str, language: str, play: bool = True) -> Optional[float]:
        start = time.perf_counter()
        first: Optional[float] = None
        out = None
        if play:
            import sounddevice as sd

            out = sd.RawOutputStream(samplerate=self.SAMPLE_RATE, channels=1, dtype="int16")
            out.start()
        try:
            with self._client.stream(
                    "POST", f"/v1/text-to-speech/{self.voice_id}/stream",
                    params={"output_format": f"pcm_{self.SAMPLE_RATE}"},
                    json={"text": text, "model_id": self.model}) as response:
                if response.status_code != 200:
                    response.read()
                    raise RuntimeError(f"ElevenLabs HTTP {response.status_code}: "
                                       f"{response.text[:200]}")
                leftover = b""
                for chunk in response.iter_bytes():
                    if first is None:
                        first = (time.perf_counter() - start) * 1000
                    if out is not None:
                        data = leftover + chunk
                        cut = len(data) - len(data) % 2   # whole int16 samples only
                        out.write(data[:cut])
                        leftover = data[cut:]
        finally:
            if out is not None:
                out.stop()   # waits for buffered audio to finish
                out.close()
        return first


class FallbackTTS:
    """A cloud engine with a local backup. If the primary fails (no credits, a voice the
    plan can't use, no network), the reason is printed once and the backup speaks from
    then on, so a conversation never crashes or goes silent over a TTS error."""
    splits_by_script = False   # the primary takes mixed text; the backup splits it itself

    def __init__(self, primary: TTS, backup: TTS, warn=print) -> None:
        self.primary, self.backup, self.warn = primary, backup, warn
        self.failed: Optional[str] = None

    @property
    def name(self) -> str:
        return self.backup.name + " (fallback)" if self.failed else f"{self.primary.name} > {self.backup.name}"

    def _backup(self, text: str, play: bool) -> Optional[float]:
        from .llm import split_by_script

        first = None
        for seg in split_by_script(text) if self.backup.splits_by_script else [None]:
            lang, chunk = (seg.language, seg.text) if seg else ("mixed", text)
            started = self.backup.speak(chunk, lang, play=play)
            first = started if first is None else first
        return first

    def speak(self, text: str, language: str, play: bool = True) -> Optional[float]:
        if self.failed is None:
            try:
                return self.primary.speak(text, language, play=play)
            except Exception as exc:   # HTTP 402/401/429, network: all mean "use the backup"
                self.failed = str(exc)[:200]
                self.warn(f"[{self.primary.name} failed, switching to {self.backup.name}: {self.failed}]")
        return self._backup(text, play)


class SilentTTS:
    name = "silent"
    splits_by_script = True

    def speak(self, text: str, language: str, play: bool = True) -> Optional[float]:
        return None


def load_tts(cfg: dict, env: dict) -> TTS:
    engine = cfg.get("engine", "macos-native")
    if engine == "elevenlabs":
        e = cfg["elevenlabs"]
        key = env.get(e.get("key_env", "ELEVENLABS_API_KEY"))
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set in the repo-root .env")
        voice_id = env.get(e.get("voice_id_env", "")) or e["voice_id"]
        primary = ElevenLabsTTS(key, voice_id.strip(), e.get("model", "eleven_flash_v2_5"))
        return FallbackTTS(primary, load_tts(dict(cfg, engine="macos-native"), env))
    voices, rate = cfg["voices"], cfg.get("rate", 175)
    if engine == "macos-native":
        try:
            return MacNativeTTS(voices, rate)
        except ImportError:
            print("pyobjc not installed; using the slower `say` command")
    return MacSayTTS(voices, rate)
