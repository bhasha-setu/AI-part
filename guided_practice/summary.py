"""End-of-session summary, computed from the turn log.

Working from the JSONL log (not from in-memory session state) keeps the summary
honest and reusable for later delayed-recall analysis: the same functions run
over logs written days ago.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import LevelConfig
from .evaluator import Verdict
from .session import MAX_LEVEL

SCORED = "scored"


def _level(level: Optional[LevelConfig]) -> LevelConfig:
    if level is not None:
        return level
    from .course import default_course

    return default_course().level


def _scored(records: list[dict], session_id: Optional[str] = None) -> list[dict]:
    rows = [r for r in records if r.get("turn_type", SCORED) == SCORED]
    if session_id is not None:
        rows = [r for r in rows if r.get("session_id") == session_id]
    return rows


def items_attempted(records: list[dict], session_id: Optional[str] = None) -> list[str]:
    """Item ids that had at least one scored turn, in first-seen order."""
    seen: list[str] = []
    for r in _scored(records, session_id):
        if r["item_id"] not in seen:
            seen.append(r["item_id"])
    return seen


def solved_level_per_item(
    records: list[dict], session_id: Optional[str] = None
) -> dict[str, Optional[int]]:
    """Map each attempted item to the hint level of its first correct turn.

    ``None`` means the item was attempted but never answered acceptably
    (not achieved).
    """
    result: dict[str, Optional[int]] = {}
    for item_id in items_attempted(records, session_id):
        result[item_id] = None
    for r in _scored(records, session_id):
        item_id = r["item_id"]
        if r["verdict"] == Verdict.CORRECT and result.get(item_id) is None:
            result[item_id] = r["hint_level"]
    return result


def independence_score(
    records: list[dict], session_id: Optional[str] = None,
    level: Optional[LevelConfig] = None,
) -> float:
    """Share of attempted items solved independently.

    "Independent" means solved at or below the level's ``independent_max_level``.
    Items repeated at level 3 or never achieved stay in the denominator.
    Returns 0.0 when no items were attempted.
    """
    cutoff = _level(level).independent_max_level
    levels = solved_level_per_item(records, session_id)
    if not levels:
        return 0.0
    independent = sum(1 for lvl in levels.values() if lvl is not None and lvl <= cutoff)
    return independent / len(levels)


def outcome_per_item(
    records: list[dict], session_id: Optional[str] = None
) -> dict[str, str]:
    """Map each attempted item to its recorded outcome.

    Prefers the ``outcome`` written on the item's closing turn; falls back to
    the hint level (``repeated`` at level 3, else ``produced``) when absent.
    """
    result: dict[str, str] = {}
    for item_id in items_attempted(records, session_id):
        result[item_id] = "not_achieved"
    for r in _scored(records, session_id):
        if r.get("outcome"):
            result[r["item_id"]] = r["outcome"]
    # Fallback for any item whose closing turn lacked an explicit outcome.
    levels = solved_level_per_item(records, session_id)
    for item_id, lvl in levels.items():
        if result.get(item_id) in (None, "not_achieved") and lvl is not None:
            result[item_id] = "repeated" if lvl >= MAX_LEVEL else "produced"
    return result


def review_items(
    records: list[dict], session_id: Optional[str] = None,
    level: Optional[LevelConfig] = None,
) -> list[str]:
    """Items to flag for review: solved at ``review_min_level`` or above,
    or not achieved. This is a teacher-facing list. The scheduler reschedules
    every attempted item separately (decision 23 in the README)."""
    cutoff = _level(level).review_min_level
    levels = solved_level_per_item(records, session_id)
    return [item for item, lvl in levels.items() if lvl is None or lvl >= cutoff]


@dataclass
class SessionSummary:
    session_id: Optional[str]
    items_attempted: list[str] = field(default_factory=list)
    hint_level_per_item: dict = field(default_factory=dict)
    outcome_per_item: dict = field(default_factory=dict)
    independence_score: float = 0.0
    review_items: list[str] = field(default_factory=list)


def summarise(records: list[dict], session_id: Optional[str] = None,
              level: Optional[LevelConfig] = None) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        items_attempted=items_attempted(records, session_id),
        hint_level_per_item=solved_level_per_item(records, session_id),
        outcome_per_item=outcome_per_item(records, session_id),
        independence_score=independence_score(records, session_id, level),
        review_items=review_items(records, session_id, level),
    )
