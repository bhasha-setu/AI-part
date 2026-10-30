"""The LLM-judge interface and its implementations.

    LLMJudge        protocol: judge(JudgeRequest) -> raw text. Raises JudgeUnavailable.
    StubJudge       deterministic, rule-backed. The default, and what tests and CI use.
    ScriptedJudge   returns queued outputs, for testing rejections and fallbacks.
    AnthropicJudge  the real client. Needs ANTHROPIC_API_KEY in the environment and the
                    ``anthropic`` package. Never called in tests.

This is a different seam from ``guided_practice.evaluator.LLMJudge``, which the rule
evaluator still uses for its own rule-miss step inside the fallback.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Protocol

from guided_practice.evaluator import RuleEvaluator, Verdict

from .disclosure import reveals_target
from .models import JUDGE_RESPONSE_SCHEMA, JudgeRequest

# Server-side refusal fallback: if Claude's safety classifiers decline, the API re-runs
# the request on a recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _api_errors() -> tuple:
    """Failures that fall back to the rules. Programming errors (a wrong argument, a
    bug) are not in this list: they raise, so they can't hide behind the fallback."""
    errors = [TimeoutError, ConnectionError]
    try:
        import anthropic
    except ImportError:
        pass
    else:
        errors.append(anthropic.APIError)   # status, connection and timeout errors
    return tuple(errors)


class JudgeUnavailable(Exception):
    """The judge gave no usable text. The message is a short reason code."""


class JudgeConfigError(RuntimeError):
    """The real judge can't be built (no key, or the SDK isn't installed)."""


class LLMJudge(Protocol):
    name: str

    def judge(self, request: JudgeRequest) -> str: ...


class StubJudge:
    """A well-behaved judge, emulated with the course's own rules.

    It returns the rule verdict, the matched error id, the content correction only where
    the hint level allows it, and a fallback line as the reply. Deterministic, so it
    checks the pipeline's plumbing. It says nothing about how a real model judges.
    """

    name = "stub"

    def __init__(self, course, strings=None) -> None:
        from .config import load_reply_strings

        self.course = course
        self.strings = strings or load_reply_strings(course.ui.code)
        self.rules = RuleEvaluator()
        self.calls: list = []
        self._scenarios: dict = {}

    def _item(self, request: JudgeRequest):
        if request.scenario_id not in self._scenarios:
            self._scenarios[request.scenario_id] = self.course.load_scenario(request.scenario_id)
        return self._scenarios[request.scenario_id].item_by_id(request.item_id)

    def judge(self, request: JudgeRequest) -> str:
        self.calls.append(request)
        item = self._item(request)
        ev = self.rules.evaluate(request.transcript_normalised, item)
        error_id, correction = None, None
        if ev.verdict == Verdict.SURFACE_ERROR:
            error_id = ev.matched_error_id or None
            if ev.feedback and not reveals_target(ev.feedback, item, request.hint_level,
                                                  self.course.ui, request.shown_to_learner):
                correction = ev.feedback
        rule_decisive = ev.verdict in (Verdict.CORRECT, Verdict.SURFACE_ERROR, Verdict.OFF_TOPIC)
        return json.dumps({
            "verdict": ev.verdict,
            "error_id": error_id,
            "correction": correction,
            "reply": self.strings.guided(ev.verdict, request.hint_level),
            "confidence": 0.95 if rule_decisive else 0.7,
        })


class ScriptedJudge:
    """Returns the queued outputs in order. An Exception in the queue is raised."""

    name = "scripted"

    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.calls: list = []

    def judge(self, request: JudgeRequest) -> str:
        self.calls.append(request)
        if not self.outputs:
            raise JudgeUnavailable("script_exhausted")
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, str) else json.dumps(out)


class AnthropicJudge:
    """The real judge, through the Anthropic SDK.

    The key is read by the SDK from ANTHROPIC_API_KEY; this class only checks that the
    variable is set and never stores, prints or logs it. Output is constrained with a
    JSON-schema structured output, and validation.py still checks every rule.
    """

    name = "anthropic"

    def __init__(self, cfg, client=None, env: Optional[dict] = None) -> None:
        self.cfg = cfg
        if client is None:
            env = os.environ if env is None else env
            if not env.get("ANTHROPIC_API_KEY"):
                raise JudgeConfigError("ANTHROPIC_API_KEY is not set; the real judge needs it")
            try:
                import anthropic
            except ImportError as exc:
                raise JudgeConfigError("the real judge needs the 'anthropic' package "
                                       "(pip install anthropic)") from exc
            client = anthropic.Anthropic(timeout=cfg.timeout_seconds, max_retries=cfg.max_retries)
        self._client = client

    def __repr__(self) -> str:
        return f"AnthropicJudge(model={self.cfg.model!r}, effort={self.cfg.effort!r})"

    def build_params(self, request: JudgeRequest) -> dict:
        return {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "system": self.cfg.system_prompt,
            "messages": [{
                "role": "user",
                "content": json.dumps(request.to_payload(), ensure_ascii=False),
            }],
            "output_config": {
                "effort": self.cfg.effort,
                "format": {"type": "json_schema", "schema": JUDGE_RESPONSE_SCHEMA},
            },
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def judge(self, request: JudgeRequest) -> str:
        try:
            response = self._client.beta.messages.create(**self.build_params(request))
        except _api_errors() as exc:  # API, network or timeout failure -> rule fallback
            status = getattr(exc, "status_code", None)
            raise JudgeUnavailable(f"api_error:{type(exc).__name__}"
                                   + (f":{status}" if status else "")) from exc
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            raise JudgeUnavailable("refusal")
        if stop == "max_tokens":
            raise JudgeUnavailable("max_tokens")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise JudgeUnavailable("no_text_block")
        return text


def make_judge(kind: str, course, cfg=None):
    """``stub`` (default) or ``anthropic``."""
    from .config import load_judge_config

    if kind == "stub":
        return StubJudge(course)
    if kind == "anthropic":
        return AnthropicJudge(cfg or load_judge_config())
    raise ValueError(f"unknown judge '{kind}' (expected stub or anthropic)")
