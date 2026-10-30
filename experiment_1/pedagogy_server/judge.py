"""The LLM-judge interface and its implementations.

    LLMJudge        protocol: judge(JudgeRequest) -> raw text. Raises JudgeUnavailable.
    StubJudge       deterministic, rule-backed. The default, and what tests and CI use.
    ScriptedJudge   returns queued outputs, for testing rejections and fallbacks.
    GeminiJudge     a real client, through the provider-neutral llm.JSONModel seam.
                    Needs GEMINI_API_KEY and ``google-genai``. Never called in tests.

This is a different seam from ``guided_practice.evaluator.LLMJudge``, which the rule
evaluator still uses for its own rule-miss step inside the fallback.
"""

from __future__ import annotations

import json
from typing import Optional, Protocol

from guided_practice.evaluator import RuleEvaluator, Verdict

from .disclosure import reveals_target
from .models import JUDGE_RESPONSE_SCHEMA, JudgeRequest

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

    def __init__(self, course, strings=None, cfg=None) -> None:
        from .config import load_judge_config, load_reply_strings

        self.course = course
        self.strings = strings or load_reply_strings(course.ui.code)
        self.cfg = cfg or load_judge_config()
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
                                                  self.course.ui, request.shown_to_learner,
                                                  self.cfg.reveal_accepted_min_tokens):
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


class GeminiJudge:
    """The real judge on any ``llm.JSONModel`` (Gemini by default). Same request, same
    schema; validation.py still checks every rule."""

    name = "gemini"

    def __init__(self, cfg, model=None) -> None:
        self.cfg = cfg
        if model is None:
            from .llm import GeminiModel, LLMConfigError

            try:
                model = GeminiModel(cfg.gemini_model, thinking_level=cfg.gemini_thinking_level or None,
                                    timeout_seconds=cfg.timeout_seconds)
            except LLMConfigError as exc:
                raise JudgeConfigError(str(exc)) from exc
        self._model = model

    def __repr__(self) -> str:
        return f"GeminiJudge({self._model!r})"

    def judge(self, request: JudgeRequest) -> str:
        from .llm import LLMUnavailable

        try:
            return self._model.generate_json(
                system=self.cfg.system_prompt,
                user=json.dumps(request.to_payload(), ensure_ascii=False),
                schema=JUDGE_RESPONSE_SCHEMA)
        except LLMUnavailable as exc:
            raise JudgeUnavailable(str(exc)) from exc


JUDGE_KINDS = ("stub", "gemini")


def make_judge(kind: str, course, cfg=None):
    """``stub`` (default) or ``gemini``."""
    from .config import load_judge_config

    if kind == "stub":
        return StubJudge(course, cfg=cfg)
    if kind == "gemini":
        return GeminiJudge(cfg or load_judge_config())
    raise ValueError(f"unknown judge '{kind}' (expected one of {', '.join(JUDGE_KINDS)})")
