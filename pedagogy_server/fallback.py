"""Rule-based fallback responder.

Used when the judge is unavailable or its output is rejected. The verdict comes from
the existing RuleEvaluator (accepted forms, listed errors, bounded typo repair), the
correction from the content, and every other line from ``fallback_lines`` in the UI
config: one per stage, and for guided practice one per verdict and hint level. The
result still goes through the guardrails, like a judge reply.
"""

from __future__ import annotations

from guided_practice.evaluator import EvalResult, Verdict

from .guardrails import Candidate


class FallbackResponder:
    def __init__(self, strings) -> None:
        self.strings = strings

    def stage_line(self, stage: str) -> str:
        return self.strings.stage_lines[stage]

    def guided_line(self, verdict: str, level: int) -> str:
        return self.strings.guided(verdict, level)

    def candidate(self, rule: EvalResult, level: int) -> Candidate:
        error_id = rule.matched_error_id or None
        if rule.verdict == Verdict.SURFACE_ERROR and rule.feedback:
            # The content's one-line correction is the whole reply, as in Session.
            return Candidate(rule.verdict, error_id, rule.feedback, "", "fallback")
        return Candidate(rule.verdict, error_id, None, self.guided_line(rule.verdict, level),
                         "fallback")
