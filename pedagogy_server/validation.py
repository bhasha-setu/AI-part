"""Server-side validation of the judge's raw output. The judge is never trusted.

Each rejection is a reason code. Any rejection sends the turn to the rule-based
fallback, and the codes are logged. Rules, in order:

    malformed_json                  not parseable as JSON
    not_an_object                   JSON, but not an object
    missing_field:<f>               a required field is absent
    unexpected_field:<f>            a field outside the schema (strict schema)
    bad_type:<f>                    wrong JSON type
    bad_verdict                     verdict not one of the four values
    confidence_out_of_range         confidence outside [0, 1]
    empty_reply                     reply is blank
    correction_without_surface_error  error_id or correction set on another verdict
    unknown_error_id                error_id is not one of the item's listed errors
    correction_not_one_line         correction has a line break
    correction_too_long             correction over correction_max_chars
    correction_reveals_target       correction shows more than the hint level allows
    reply_too_long                  over reply_max_sentences or reply_max_chars
    reply_reveals_target            a non-correct reply shows the target form early
    too_many_corrections            reply markers plus the correction field exceed one
    accepted_form_contradicted      the transcript is an exact accepted form but the
                                    verdict isn't correct (accepted_form_floor)

Low confidence is not a rejection: ``low_confidence`` is set and the caller treats
the verdict as meaning_error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .disclosure import count_markers, count_sentences, reveals_target
from .models import JUDGE_RESPONSE_SCHEMA, VERDICTS, JudgeResponse

_REQUIRED = tuple(JUDGE_RESPONSE_SCHEMA["required"])


@dataclass
class ValidationResult:
    response: Optional[JudgeResponse] = None
    rejections: list = field(default_factory=list)
    low_confidence: bool = False

    @property
    def ok(self) -> bool:
        return self.response is not None and not self.rejections


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_judge_output(raw: str, *, item, level: int, norm: str, ui, strings, cfg,
                          already_shown=()) -> ValidationResult:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return ValidationResult(rejections=["malformed_json"])
    if not isinstance(data, dict):
        return ValidationResult(rejections=["not_an_object"])

    shape = [f"missing_field:{f}" for f in _REQUIRED if f not in data]
    shape += [f"unexpected_field:{f}" for f in data if f not in _REQUIRED]
    if shape:
        return ValidationResult(rejections=shape)

    types = []
    if not isinstance(data["verdict"], str):
        types.append("bad_type:verdict")
    for f in ("error_id", "correction"):
        if data[f] is not None and not isinstance(data[f], str):
            types.append(f"bad_type:{f}")
    if not isinstance(data["reply"], str):
        types.append("bad_type:reply")
    if not _is_number(data["confidence"]):
        types.append("bad_type:confidence")
    if types:
        return ValidationResult(rejections=types)

    verdict = data["verdict"]
    error_id = data["error_id"] or None
    correction = (data["correction"] or "").strip() or None
    reply = data["reply"].strip()
    confidence = float(data["confidence"])
    out = []

    if verdict not in VERDICTS:
        out.append("bad_verdict")
    if not 0.0 <= confidence <= 1.0:
        out.append("confidence_out_of_range")
    if not reply:
        out.append("empty_reply")
    if verdict != "surface_error" and (error_id or correction):
        out.append("correction_without_surface_error")
    if error_id and error_id not in {ce.id for ce in item.common_errors}:
        out.append("unknown_error_id")

    if correction:
        if "\n" in correction or "\r" in correction:
            out.append("correction_not_one_line")
        if len(correction) > cfg.correction_max_chars:
            out.append("correction_too_long")
        if reveals_target(correction, item, level, ui, already_shown):
            out.append("correction_reveals_target")

    if (count_sentences(reply, strings.sentence_terminators) > cfg.reply_max_sentences
            or len(reply) > cfg.reply_max_chars):
        out.append("reply_too_long")
    if verdict != "correct" and reveals_target(reply, item, level, ui, already_shown):
        out.append("reply_reveals_target")
    if count_markers(reply, strings.correction_markers) + (1 if correction else 0) > 1:
        out.append("too_many_corrections")

    if cfg.accepted_form_floor and verdict != "correct" and item.match_accepted(norm) is not None:
        out.append("accepted_form_contradicted")

    response = JudgeResponse(verdict=verdict, error_id=error_id, correction=correction,
                             reply=reply, confidence=confidence)
    return ValidationResult(response=response, rejections=out,
                            low_confidence=confidence < cfg.min_confidence)
