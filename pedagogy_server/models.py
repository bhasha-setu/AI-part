"""Typed data models for the server.

Reused from the core, not redefined:
    guided_practice.session.Session   the hint ladder (one per guided attempt)
    guided_practice.policy.Decision   what the lesson policy decided, with its reason

New here:
    ServerSession   per-lesson server state: correction counts and what was shown
    Frame           what the server knows before a turn
    JudgeRequest    what the judge sees (no policy internals)
    JudgeResponse   what the judge must return
    Turn            one transcript in, one reply out
    Reply           what goes to TTS
    LogRecord       one line of the decision log
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

from guided_practice.policy import Decision  # noqa: F401  (re-exported: the lesson's decision type)

VERDICTS = ("correct", "surface_error", "meaning_error", "off_topic")


@dataclass
class ServerSession:
    """State the server keeps for one lesson. Guardrail counters span every guided
    attempt in the lesson, so a repeat doesn't reset the correction cap."""

    session_id: str
    learner_id: str
    course_id: str
    scenario_id: str
    corrections_used: int = 0
    corrected_error_ids: Counter = field(default_factory=Counter)
    corrections_shown: dict = field(default_factory=dict)   # item_id -> [correction text]
    turns: list = field(default_factory=list)

    def record_correction(self, item_id: str, error_id: Optional[str], text: str) -> None:
        self.corrections_used += 1
        if error_id:
            self.corrected_error_ids[error_id] += 1
        self.corrections_shown.setdefault(item_id, []).append(text)


@dataclass(frozen=True)
class Frame:
    stage: str
    expects: str                 # continue | answer | done | held
    item_id: str = ""
    hint_level: Optional[int] = None
    awaiting_follow_up: bool = False
    corrections_used: int = 0
    corrected_error_ids: tuple = ()
    lesson_status: str = ""


@dataclass(frozen=True)
class KnownError:
    """A content-listed error, sent to the judge as data to match against."""

    id: str
    example: str
    correction_example: str


@dataclass(frozen=True)
class JudgeRequest:
    """Everything the judge sees. It never carries retry counts, caps, Leitner state,
    the policy or the stage order: the judge can't choose a hint level or end a lesson."""

    target_language: str
    learner_language: str
    cefr_level: str
    reply_language: str
    scenario_id: str
    scenario_title: str
    item_id: str
    prompt: str
    target_form: str
    accepted_forms: tuple
    known_errors: tuple          # of KnownError
    hint_level: int
    shown_to_learner: tuple      # prompts (and corrections) already on screen for this item
    transcript_raw: str
    transcript_normalised: str
    rubric: str
    constraints: tuple

    def to_payload(self) -> dict:
        data = asdict(self)
        data["known_errors"] = [asdict(e) for e in self.known_errors]
        data["accepted_forms"] = list(self.accepted_forms)
        data["shown_to_learner"] = list(self.shown_to_learner)
        data["constraints"] = list(self.constraints)
        data["placeholder_note"] = "{name}-style slots in forms stand for any value the learner chooses."
        return data


@dataclass(frozen=True)
class JudgeResponse:
    verdict: str
    error_id: Optional[str]
    correction: Optional[str]
    reply: str
    confidence: float


# The JSON schema the judge must satisfy. Sent to the real client as a structured-output
# format, and enforced again server-side by validation.py (the server never trusts it).
JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        # anyOf, not a type array: it is in the structured-outputs schema subset.
        "error_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "correction": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "reply": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["verdict", "error_id", "correction", "reply", "confidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Reply:
    """What the server returns for TTS. ``text`` is what is spoken."""

    text: str
    stage: str
    expects: str
    feedback: str = ""
    prompt: str = ""
    item_id: str = ""


@dataclass(frozen=True)
class Turn:
    index: int
    frame: Frame
    transcript_raw: str
    reply: Reply


@dataclass
class LogRecord:
    """One line of ``decisions.jsonl``. ``kind`` is turn | policy | lesson_end."""

    timestamp: str
    session_id: str
    turn_index: int
    kind: str
    stage: str
    reason: str
    data: dict = field(default_factory=dict)
