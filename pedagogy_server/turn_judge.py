"""One guided-practice turn through the judge path:

    frame -> JudgeRequest -> LLMJudge -> validate -> (fallback) -> guardrails -> EvalResult

The returned EvalResult goes to the existing Session, which still decides the hint
level, retries, the item outcome and the next item. The pipeline and the eval runner
share this class, so the eval measures the same path the learner hears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from guided_practice.evaluator import EvalResult, RuleEvaluator, Verdict
from guided_practice.normalise import normalise

from .config import load_judge_config, load_reply_strings, replies_may_use_learner_language
from .disclosure import shown_prompts
from .fallback import FallbackResponder
from .guardrails import Candidate, apply_guardrails
from .judge import JudgeUnavailable
from .models import JudgeRequest, KnownError
from .validation import validate_judge_output

RAW_LOG_LIMIT = 2000  # characters of raw judge output kept in the log


@dataclass(frozen=True)
class JudgedTurn:
    evaluation: EvalResult
    request: JudgeRequest
    source: str                    # judge | judge:low_confidence | fallback
    rejections: tuple
    guardrail_actions: tuple
    judge_verdict: Optional[str]
    rule_verdict: str
    rule_error_id: str
    correction_shown: Optional[str]
    raw_output: Optional[str]

    def log_data(self) -> dict:
        return {
            "source": self.source,
            "verdict": self.evaluation.verdict,
            "error_id": self.evaluation.matched_error_id or None,
            "judge_verdict": self.judge_verdict,
            "rule_verdict": self.rule_verdict,
            "rule_error_id": self.rule_error_id or None,
            "rejections": list(self.rejections),
            "guardrail_actions": list(self.guardrail_actions),
            "correction_shown": self.correction_shown,
            "feedback": self.evaluation.feedback,
            "judge_raw": self.raw_output[:RAW_LOG_LIMIT] if self.raw_output else None,
        }


class TurnJudge:
    def __init__(self, course, judge, cfg=None, strings=None) -> None:
        self.course = course
        self.judge = judge
        self.cfg = cfg or load_judge_config()
        self.strings = strings or load_reply_strings(course.ui.code)
        self.fallback = FallbackResponder(self.strings)
        self.rules = RuleEvaluator()
        self.learner_language_ok = replies_may_use_learner_language(course.profile)

    def build_request(self, scenario, item, level: int, transcript: str, norm: str,
                      shown_corrections=()) -> JudgeRequest:
        profile = self.course.profile
        rule = self.cfg.learner_language_rule["allowed" if self.learner_language_ok else "forbidden"]
        constraints = self.cfg.constraints + (rule.format(
            learner_language=profile.learner_language, reply_language=self.course.ui.code),)
        return JudgeRequest(
            target_language=profile.target_language,
            learner_language=profile.learner_language,
            cefr_level=profile.cefr_level,
            reply_language=self.course.ui.code,
            scenario_id=scenario.id,
            scenario_title=scenario.title,
            item_id=item.id,
            prompt=item.prompt,
            target_form=item.model_answer,
            accepted_forms=tuple(item.accepted),
            known_errors=tuple(KnownError(ce.id, ce.example, ce.correction_example)
                               for ce in item.common_errors),
            hint_level=level,
            shown_to_learner=tuple(shown_prompts(item, level, self.course.ui)) + tuple(shown_corrections),
            transcript_raw=transcript,
            transcript_normalised=norm,
            rubric=self.cfg.rubric,
            constraints=constraints,
        )

    def judge_turn(self, scenario, item, level: int, transcript: str, state) -> JudgedTurn:
        norm = normalise(transcript, self.course.lang)
        rule = self.rules.evaluate(norm, item)
        already_shown = tuple(state.corrections_shown.get(item.id, ()))
        request = self.build_request(scenario, item, level, transcript, norm, already_shown)

        raw, rejections, judge_verdict, cand = None, [], None, None
        try:
            raw = self.judge.judge(request)
        except JudgeUnavailable as exc:
            rejections = [f"judge_unavailable:{exc}"]
        else:
            result = validate_judge_output(raw, item=item, level=level, norm=norm,
                                           ui=self.course.ui, strings=self.strings,
                                           cfg=self.cfg, already_shown=already_shown)
            rejections = list(result.rejections)
            if result.ok:
                resp = result.response
                judge_verdict = resp.verdict
                floor = self.cfg.accepted_form_floor and item.match_accepted(norm) is not None
                if result.low_confidence and not floor:
                    # Not trusted as correct: advance the ladder as for a meaning error.
                    cand = Candidate(Verdict.MEANING_ERROR, None, None,
                                     self.fallback.guided_line(Verdict.MEANING_ERROR, level),
                                     "judge:low_confidence")
                else:
                    cand = Candidate(resp.verdict, resp.error_id, resp.correction, resp.reply, "judge")
        if cand is None:
            cand = self.fallback.candidate(rule, level)

        guarded = apply_guardrails(cand, item=item, norm=norm, level=level, ui=self.course.ui,
                                   strings=self.strings, cfg=self.cfg, state=state,
                                   already_shown=already_shown)
        if guarded.correction_shown:
            state.record_correction(item.id, guarded.error_id, guarded.correction_shown)

        evaluation = EvalResult(verdict=guarded.verdict, feedback=guarded.feedback,
                                slots=rule.slots, matched_error_id=guarded.error_id or "")
        return JudgedTurn(
            evaluation=evaluation, request=request, source=cand.source,
            rejections=tuple(rejections), guardrail_actions=guarded.actions,
            judge_verdict=judge_verdict, rule_verdict=rule.verdict,
            rule_error_id=rule.matched_error_id, correction_shown=guarded.correction_shown,
            raw_output=raw,
        )
