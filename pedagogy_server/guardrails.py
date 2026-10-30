"""Guardrails: server rules applied after validation to every candidate reply, the
judge's and the rule fallback's alike. They decide whether a correction is shown.

    form_not_produced:<id>       the cited error's pattern doesn't match what the learner
                                 said, so the learner didn't produce that form
    unlisted_error_not_corrected the error isn't in the content, so the server can't
                                 check it (allow_unlisted_corrections)
    already_corrected:<id>       this error id was corrected earlier in the session
    correction_cap_reached       corrections_per_session reached; encourage only
    early_reveal                 the text would show the target form before the hint
                                 level allows it
    length_cap                   the line the learner hears is over the cap

A held-back correction is replaced by the encouragement line, not trimmed: a reply
written around a correction can't be trusted once the correction is cut out.
Guardrails don't change the verdict, so the ladder still retries or advances as usual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .disclosure import count_markers, count_sentences, reveals_target

SURFACE = "surface_error"
CORRECT = "correct"


@dataclass(frozen=True)
class Candidate:
    verdict: str
    error_id: Optional[str]
    correction: Optional[str]
    reply: str
    source: str               # judge | judge:low_confidence | fallback


@dataclass(frozen=True)
class GuardedTurn:
    verdict: str
    error_id: Optional[str]
    feedback: str             # the line the learner hears before the next prompt
    correction_shown: Optional[str]
    actions: tuple


def _join(*parts) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def apply_guardrails(cand: Candidate, *, item, norm: str, level: int, ui, strings, cfg,
                     state, already_shown=()) -> GuardedTurn:
    actions = []
    reply_corrects = count_markers(cand.reply, strings.correction_markers) > 0
    correction_text = _join(cand.reply if reply_corrects else "", cand.correction or "")

    if correction_text:
        if cand.error_id is None:
            if not cfg.allow_unlisted_corrections:
                actions.append("unlisted_error_not_corrected")
        else:
            listed = next((ce for ce in item.common_errors if ce.id == cand.error_id), None)
            if listed is None or listed.match(norm) is None:
                actions.append(f"form_not_produced:{cand.error_id}")
            if state.corrected_error_ids[cand.error_id] >= cfg.corrections_per_error_id:
                actions.append(f"already_corrected:{cand.error_id}")
        if state.corrections_used >= cfg.corrections_per_session:
            actions.append("correction_cap_reached")
        if reveals_target(correction_text, item, level, ui, already_shown):
            actions.append("early_reveal")

    fallback_line = strings.encourage if cand.verdict == SURFACE else strings.guided(cand.verdict, level)
    if actions:
        feedback, shown = fallback_line, None
    else:
        feedback = _join(cand.reply, cand.correction or "") or fallback_line
        shown = correction_text or None
        if cand.verdict != CORRECT and reveals_target(feedback, item, level, ui, already_shown):
            actions.append("early_reveal")
            feedback, shown = fallback_line, None

    sentence_cap = cfg.reply_max_sentences + (1 if shown else 0)
    if (len(feedback) > cfg.feedback_max_chars
            or count_sentences(feedback, strings.sentence_terminators) > sentence_cap):
        actions.append("length_cap")
        feedback, shown = fallback_line, None

    return GuardedTurn(verdict=cand.verdict, error_id=cand.error_id, feedback=feedback,
                       correction_shown=shown, actions=tuple(actions))
