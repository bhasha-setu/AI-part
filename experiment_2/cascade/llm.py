"""Stage 2: text in, reply sentences out (streamed).

``ChatLLM.stream(history, text)`` yields the reply one sentence at a time, so TTS can
start speaking the first sentence while the rest is still being generated. This is the
seam where the pedagogy (experiment_1's tutor) plugs in later.

Backends:
    GeminiChat        Google Gemini (google-genai), streaming.
    OpenAICompatChat  any OpenAI-compatible endpoint, e.g. NVIDIA's hosted models
                      (https://integrate.api.nvidia.com/v1), streaming over SSE.
    ProviderChain     tries backends in order; one that fails (429, timeout) is skipped
                      for a cool-down so the next turn doesn't wait on it again.
    EchoLLM           offline stand-in.

``split_by_script`` cuts a sentence into Hindi (Devanagari) and English runs for TTS
engines that need one voice per language (macOS).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Iterable, Iterator, Optional, Protocol

from .types import Segment


class LLMUnavailable(Exception):
    """No usable reply. The message is a short reason code, safe to log."""


class ChatLLM(Protocol):
    name: str

    def stream(self, history: list[dict], text: str) -> Iterator[str]: ...


# --- text helpers ------------------------------------------------------------------------

def _is_devanagari(ch: str) -> bool:
    return "ऀ" <= ch <= "ॿ" and ch not in "।॥"


def split_by_script(text: str) -> list[Segment]:
    """Cut text into Hindi (Devanagari) and English runs. Spaces, digits and
    punctuation stay with the run they sit in, so nothing is lost or reordered."""
    segments: list[Segment] = []
    for ch in text:
        if _is_devanagari(ch):
            lang = "hi"
        elif ch.isalpha():
            lang = "en"
        else:
            lang = None   # neutral: joins the current run
        if segments and (lang is None or segments[-1].language == lang):
            segments[-1].text += ch
        elif lang is None:
            segments.append(Segment("en", ch))
        else:
            segments.append(Segment(lang, ch))
    # A run of only spaces/punctuation can't be spoken; fold it into its neighbour.
    merged: list[Segment] = []
    for seg in segments:
        if merged and (not any(c.isalnum() for c in seg.text) or merged[-1].language == seg.language):
            merged[-1].text += seg.text
        else:
            merged.append(seg)
    return [Segment(s.language, s.text.strip()) for s in merged if s.text.strip()]


# A sentence ends at . ! ? or the Hindi danda (।), followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def sentences_from_chunks(chunks: Iterable[str]) -> Iterator[str]:
    """Re-cut a stream of arbitrary text chunks into whole sentences."""
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        parts = _SENTENCE_END.split(buffer)
        for sentence in parts[:-1]:
            if sentence.strip():
                yield sentence.strip()
        buffer = parts[-1]
    if buffer.strip():
        yield buffer.strip()


def _clean_for_speech(text: str) -> str:
    """Drop markdown symbols a TTS voice would read out ("asterisk")."""
    return re.sub(r"[*_#`>]+", "", text)


# --- backends ---------------------------------------------------------------------------

class GeminiChat:
    """Streaming multi-turn chat on Gemini. The key is passed in, never printed or
    logged; ``name`` carries a key label (e.g. "key2"), not the key."""

    def __init__(self, model: str, system_prompt: str, *, api_key: str,
                 key_label: str = "", thinking_level: Optional[str] = "minimal",
                 timeout_seconds: float = 15, max_output_tokens: int = 300,
                 client=None) -> None:
        self.model = model
        self.name = f"gemini:{model}" + (f"[{key_label}]" if key_label else "")
        self.system_prompt = system_prompt
        self.thinking_level = thinking_level
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            max_output_tokens=self.max_output_tokens,
            thinking_config=(types.ThinkingConfig(thinking_level=self.thinking_level)
                             if self.thinking_level else None),
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)))

    def _chunks(self, history: list[dict], text: str) -> Iterator[str]:
        contents = [{"role": h["role"], "parts": [{"text": h["text"]}]} for h in history]
        contents.append({"role": "user", "parts": [{"text": text}]})
        try:
            for chunk in self._client.models.generate_content_stream(
                    model=self.model, contents=contents, config=self._config()):
                candidates = getattr(chunk, "candidates", None) or []
                finish = getattr(candidates[0], "finish_reason", None) if candidates else None
                if getattr(finish, "name", finish) == "MAX_TOKENS":
                    raise LLMUnavailable("max_tokens")   # don't speak a sentence cut in half
                if chunk.text:
                    yield chunk.text
        except LLMUnavailable:
            raise
        except Exception as exc:   # network, quota (429), bad model name: all recoverable
            code = getattr(exc, "code", None)
            raise LLMUnavailable(f"api_error:{type(exc).__name__}"
                                 + (f":{code}" if code else "")) from exc

    def stream(self, history: list[dict], text: str) -> Iterator[str]:
        for sentence in sentences_from_chunks(self._chunks(history, text)):
            yield _clean_for_speech(sentence)


class OpenAICompatChat:
    """Streaming chat on any OpenAI-compatible endpoint (NVIDIA build.nvidia.com,
    and others). Uses httpx directly, so no extra SDK."""

    def __init__(self, model: str, system_prompt: str, *, api_key: str, base_url: str,
                 label: str = "openai-compat", timeout_seconds: float = 15,
                 max_output_tokens: int = 300, temperature: float = 0.6,
                 extra_body: Optional[dict] = None, transport=None) -> None:
        import httpx

        self.model = model
        self.name = f"{label}:{model}"
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.extra_body = extra_body or {}   # provider-specific, e.g. turning thinking off
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds,
                                    headers={"Authorization": f"Bearer {api_key}"},
                                    transport=transport)

    def _chunks(self, history: list[dict], text: str) -> Iterator[str]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += [{"role": "assistant" if h["role"] == "model" else "user",
                      "content": h["text"]} for h in history]
        messages.append({"role": "user", "content": text})
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": self.max_output_tokens, "temperature": self.temperature,
                **self.extra_body}
        try:
            with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code != 200:
                    raise LLMUnavailable(f"http_{response.status_code}")
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choice = (json.loads(data).get("choices") or [{}])[0]
                    if choice.get("finish_reason") == "length":
                        raise LLMUnavailable("max_tokens")
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
        except LLMUnavailable:
            raise
        except Exception as exc:
            raise LLMUnavailable(f"api_error:{type(exc).__name__}") from exc

    def stream(self, history: list[dict], text: str) -> Iterator[str]:
        for sentence in sentences_from_chunks(self._chunks(history, text)):
            yield _clean_for_speech(sentence)


class ProviderChain:
    """Try each backend in order until one starts replying. A backend that fails before
    its first sentence is benched for ``cooldown_seconds``. Once a backend has produced
    a sentence, a later failure is raised: the learner already heard part of a reply,
    and starting over with another model would repeat or contradict it."""

    def __init__(self, backends: list, cooldown_seconds: float = 60, clock=time.monotonic) -> None:
        if not backends:
            raise ValueError("ProviderChain needs at least one backend")
        self.backends = backends
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._benched_until: dict[str, float] = {}
        self.last_used: Optional[str] = None
        self.last_errors: list[str] = []

    @property
    def name(self) -> str:
        return " > ".join(b.name for b in self.backends)

    def stream(self, history: list[dict], text: str) -> Iterator[str]:
        self.last_errors = []
        now = self._clock()
        ready = [b for b in self.backends if self._benched_until.get(b.name, 0) <= now]
        for backend in ready or self.backends:   # all benched: try them anyway
            produced = False
            try:
                for sentence in backend.stream(history, text):
                    produced = True
                    self.last_used = backend.name
                    yield sentence
                if produced:
                    return
                self.last_errors.append(f"{backend.name}:empty")
            except LLMUnavailable as exc:
                if produced:
                    raise
                self.last_errors.append(f"{backend.name}:{exc}")
                self._benched_until[backend.name] = self._clock() + self.cooldown_seconds
        raise LLMUnavailable("all_failed:" + ";".join(self.last_errors))


class EchoLLM:
    """Offline stand-in: repeats what it heard. Lets the whole cascade run with no
    network, to measure STT and TTS on their own."""
    name = "echo"

    def stream(self, history: list[dict], text: str) -> Iterator[str]:
        yield f"You said: {text}" if text else "I didn't hear anything."


# --- building the chain from config ----------------------------------------------------

def _keys(env, prefix: str) -> list[tuple[str, str]]:
    """PREFIX, PREFIX_2, PREFIX_3, ... -> [(label, key)], skipping unset ones."""
    found = []
    for i in range(1, 10):
        var = prefix if i == 1 else f"{prefix}_{i}"
        if env.get(var):
            found.append((f"key{i}", env[var]))
    return found


def build_chain(cfg: dict, env=None) -> ProviderChain:
    """One backend per (provider, key) listed in ``cfg["providers"]`` whose key is set."""
    env = os.environ if env is None else env
    prompt = cfg["system_prompt"]
    timeout = cfg.get("timeout_seconds", 15)
    max_tokens = cfg.get("max_output_tokens", 300)
    backends = []
    for p in cfg.get("providers", []):
        for label, key in _keys(env, p["key_env"]):
            if p["type"] == "gemini":
                backends.append(GeminiChat(p["model"], prompt, api_key=key, key_label=label,
                                           thinking_level=p.get("thinking_level", "minimal"),
                                           timeout_seconds=timeout, max_output_tokens=max_tokens))
            elif p["type"] == "openai_compat":
                backends.append(OpenAICompatChat(p["model"], prompt, api_key=key,
                                                 base_url=p["base_url"],
                                                 label=f"{p.get('label', 'openai')}[{label}]",
                                                 timeout_seconds=timeout,
                                                 max_output_tokens=max_tokens,
                                                 extra_body=p.get("extra_body")))
            else:
                raise ValueError(f"unknown provider type {p['type']!r}")
    if not backends:
        names = ", ".join(p["key_env"] for p in cfg.get("providers", []))
        raise RuntimeError(f"no LLM key set (looked for {names} in the repo-root .env)")
    return ProviderChain(backends, cooldown_seconds=cfg.get("cooldown_seconds", 60))
