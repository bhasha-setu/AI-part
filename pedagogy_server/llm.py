"""A provider-neutral seam for "send instructions plus data, get JSON text back".

    JSONModel      protocol: generate_json(system=, user=, schema=) -> raw text.
                   Raises LLMUnavailable when there is no usable text.
    GeminiModel    the Google Gemini client. Needs GEMINI_API_KEY and ``google-genai``.
    ScriptedModel  returns queued outputs, for tests. Never touches the network.

The judge, the scenario generator and the roleplay partner all sit on this seam. None
of them trusts what comes back: each validates the text itself and has a fallback.
Swapping provider means writing one more class with ``generate_json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Protocol

KEY_VAR = "GEMINI_API_KEY"


class LLMUnavailable(Exception):
    """The model gave no usable text. The message is a short reason code."""


class LLMConfigError(RuntimeError):
    """The client can't be built (no key, or the SDK isn't installed)."""


class JSONModel(Protocol):
    name: str

    def generate_json(self, *, system: str, user: str, schema: dict) -> str: ...


def _gemini_errors() -> tuple:
    """Failures that become LLMUnavailable. Programming errors still raise."""
    errors = [TimeoutError, ConnectionError]
    try:
        import httpx
        from google.genai import errors as genai_errors
    except ImportError:
        pass
    else:
        errors += [genai_errors.APIError, httpx.TimeoutException, httpx.TransportError]
    return tuple(errors)


class GeminiModel:
    """Structured JSON output through ``google-genai``. The key is read from the
    environment and never stored on the instance, printed or logged."""

    def __init__(self, model: str, *, thinking_level: Optional[str] = None,
                 timeout_seconds: float = 15, max_output_tokens: Optional[int] = None,
                 client=None, env: Optional[dict] = None) -> None:
        self.model = model
        self.thinking_level = thinking_level
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.name = f"gemini:{model}"
        if client is None:
            env = os.environ if env is None else env
            if not env.get(KEY_VAR):
                raise LLMConfigError(f"{KEY_VAR} is not set; the Gemini client needs it")
            try:
                from google import genai
            except ImportError as exc:
                raise LLMConfigError("the Gemini client needs the 'google-genai' package "
                                     "(pip install google-genai)") from exc
            client = genai.Client(api_key=env[KEY_VAR])
        self._client = client

    def __repr__(self) -> str:
        return f"GeminiModel(model={self.model!r}, thinking_level={self.thinking_level!r})"

    def build_config(self, system: str, schema: dict):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=self.max_output_tokens,
            thinking_config=(types.ThinkingConfig(thinking_level=self.thinking_level)
                             if self.thinking_level else None),
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
        )

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=user, config=self.build_config(system, schema))
        except _gemini_errors() as exc:
            code = getattr(exc, "code", None)
            raise LLMUnavailable(f"api_error:{type(exc).__name__}"
                                 + (f":{code}" if code else "")) from exc
        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        finish = getattr(finish, "name", finish)
        if finish == "MAX_TOKENS":
            raise LLMUnavailable("max_tokens")
        if finish not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
            raise LLMUnavailable(f"finish:{finish}")   # SAFETY, RECITATION, ...
        text = getattr(response, "text", None)
        if not text:
            raise LLMUnavailable("no_text")
        return text


class ScriptedModel:
    """Returns the queued outputs in order. An Exception in the queue is raised; a dict
    is sent as JSON text. ``calls`` keeps every (system, user, schema)."""

    name = "scripted"

    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.calls: list = []

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if not self.outputs:
            raise LLMUnavailable("script_exhausted")
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, str) else json.dumps(out)


def load_dotenv(path: str | Path = ".env", env: Optional[dict] = None) -> list:
    """Copy KEY=value lines from ``path`` into ``env`` (default os.environ) without
    overriding what is already set. Returns the names loaded, never the values.
    Called by entry points (web app, CLIs), not by library code."""
    env = os.environ if env is None else env
    path = Path(path)
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and name not in env:
            env[name] = value
            loaded.append(name)
    return loaded
