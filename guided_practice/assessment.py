"""Assessment and feedback stage: summary structure and a stub.

The summary has seven fields, and each one records where its value came from:
  basis "log"   computed from the turn log
  basis "stub"  not computed yet; value is None

Task completion and independence are computable from the log now. Clarity,
grammar, vocabulary and comprehension need a judgement rule that doesn't exist
yet, so they stay stubs.

Research basis: judgements on discourse and argument are where LLM feedback is
weakest (Pasha 2026; Lyu et al. 2024). So comprehension and clarity are not handed
to an LLM here, and any LLM judge needs a separate decision record. Comprehension
will need the roleplay transcript, which is why assess() takes one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class AssessedValue:
    value: Optional[object]
    basis: str                  # "log" | "stub"
    note: str = ""


@dataclass
class AssessmentSummary:
    status: str                 # "stubbed" if any field is a stub, else "completed"
    task_completion: AssessedValue
    clarity: AssessedValue
    grammar: AssessedValue
    vocabulary: AssessedValue
    comprehension: AssessedValue
    independence: AssessedValue
    improvement_goal: AssessedValue


class Assessor(Protocol):
    def assess(self, outcomes: list, records: list, session_id: str,
               roleplay_transcript: list = ()) -> AssessmentSummary: ...


def _stub(note: str) -> AssessedValue:
    return AssessedValue(value=None, basis="stub", note=note)


class StubAssessor:
    """Computes what the log supports; stubs the rest.

    TODO: clarity, grammar, vocabulary, comprehension (needs a judgement rule and
    a decision on whether an LLM may judge these; see docs/reasoning_core.md).
    """

    def assess(self, outcomes: list, records: list, session_id: str,
               roleplay_transcript: list = ()) -> AssessmentSummary:
        attempted = len(outcomes)
        achieved = sum(1 for oc in outcomes if oc.outcome != "not_achieved")
        independent = sum(1 for oc in outcomes if oc.independent)
        scored = [r for r in records
                  if r.get("session_id") == session_id and r.get("turn_type", "scored") == "scored"]
        error_ids = Counter(r["matched_error_id"] for r in scored if r.get("matched_error_id"))
        top_error = error_ids.most_common(1)[0][0] if error_ids else None
        return AssessmentSummary(
            status="stubbed",
            task_completion=AssessedValue(
                value=(achieved / attempted) if attempted else None, basis="log",
                note="share of items achieved",
            ),
            clarity=_stub("TODO: judgement rule not defined"),
            grammar=_stub("TODO: judgement rule not defined"),
            vocabulary=_stub("TODO: judgement rule not defined"),
            comprehension=_stub("TODO: roleplay stage not implemented"),
            independence=AssessedValue(
                value=(independent / attempted) if attempted else None, basis="log",
                note="share of items solved at or below independent_max_level",
            ),
            improvement_goal=AssessedValue(
                value=top_error, basis="log",
                note="most frequent surface-error id; None if no surface errors",
            ),
        )
