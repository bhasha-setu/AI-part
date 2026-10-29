"""Verdict engine.

The rule evaluator classifies a normalised transcript against one item, in a
fixed order:

  1. empty            -> off_topic      (no LLM call)
  2. accepted phrasing-> correct
  3. common error     -> surface_error  (with its one-line feedback)
  4. typo-repaired, then accepted -> correct
  5. otherwise        -> meaning_error  (delegated to the LLM-judge seam)

Only step 5 touches the (stubbed) LLM. Everything else is pure rules, so the
main path has no model calls, as required.

Research basis: LLM feedback matches teachers on surface errors but is weaker on
content and argument, and it tends to overcorrect (Pasha 2026; Craven & Fredrick
2026; Lin 2024; framework §7). So surface errors are handled by rules, and the LLM
is only a fallback. The evidence is mostly for English targets, so this design is
English-specific until re-validated for other languages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from .matching import fill, typo_correct

logger = logging.getLogger("guided_practice.llm_judge")


class Verdict:
    CORRECT = "correct"
    SURFACE_ERROR = "surface_error"
    MEANING_ERROR = "meaning_error"
    OFF_TOPIC = "off_topic"


@dataclass
class EvalResult:
    verdict: str
    feedback: str = ""
    why: str = ""
    slots: dict = field(default_factory=dict)
    matched_error_id: str = ""


class Evaluator(Protocol):
    """The verdict-engine interface the Session depends on."""

    def evaluate(self, norm: str, item) -> "EvalResult": ...


class LLMJudge(Protocol):
    """Interface for an LLM meaning-check. The main path never needs this."""

    def judge(self, norm: str, item) -> str: ...


class StubLLMJudge:
    """TODO stub. Always returns ``meaning_error`` and records the call.

    Replace with a real few-shot judge later (see README / framework §7). The
    recorded calls let tests assert the seam is actually exercised.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def judge(self, norm: str, item) -> str:
        record = {"item_id": getattr(item, "id", None), "transcript": norm}
        self.calls.append(record)
        logger.info("LLM judge (stub) called: %s", record)
        return Verdict.MEANING_ERROR


class RuleEvaluator:
    def __init__(self, llm: LLMJudge | None = None) -> None:
        self.llm = llm if llm is not None else StubLLMJudge()

    def evaluate(self, norm: str, item) -> EvalResult:
        # 1. empty -> off_topic
        if not norm:
            return EvalResult(Verdict.OFF_TOPIC)

        # 2. accepted phrasing
        slots = item.match_accepted(norm)
        if slots is not None:
            return EvalResult(Verdict.CORRECT, slots=slots)

        # 3. common (surface) error
        for ce in item.common_errors:
            m = ce.match(norm)
            if m is not None:
                return EvalResult(
                    Verdict.SURFACE_ERROR,
                    feedback=fill(ce.feedback, m),
                    why=fill(ce.why, m),
                    slots=m,
                    matched_error_id=ce.id,
                )

        # 4. bounded typo tolerance, then re-check accepted
        corrected, changed = typo_correct(norm, item.vocab, item.lang)
        if changed:
            slots = item.match_accepted(corrected)
            if slots is not None:
                return EvalResult(Verdict.CORRECT, slots=slots)

        # 5. otherwise -> LLM seam (stub returns meaning_error)
        verdict = self.llm.judge(norm, item)
        return EvalResult(verdict)
