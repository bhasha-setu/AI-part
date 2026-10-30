"""Minimal learner model (Leitner boxes), JSON-persisted.

Tracks, per learner and item, a Leitner box and a next-due timestamp. The clock
is injected so behaviour is deterministic and testable.

State keys are namespaced by course: with ``course_id`` set, an item is stored as
``"<course_id>::<item_id>"``, so the same item id in two courses cannot collide.
``course_id=None`` keeps the unscoped keys used before courses existed; it is a
compatibility shim.

Scheduling policy (thresholds and intervals live in the level config):
  * Solved at or below ``promote_max_level``  -> promote one box.
  * Solved at or below ``hold_max_level``     -> keep the box.
  * Anything else (level 3 repeat, not achieved) -> reset to box 1.

The interval values are hypotheses. The direction of spacing is from Hanson &
Brown (2019). That study's learners are assumed to be EFL, following framework §11,
and this has not been verified (see docs/reasoning_core.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .config import LevelConfig


@dataclass
class ItemState:
    item_id: str
    box: int
    due: str          # ISO 8601 timestamp
    last_seen: str    # ISO 8601 timestamp


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _default_level() -> LevelConfig:
    from .course import default_course

    return default_course().level


class LearnerModel:
    def __init__(self, learner_id: str, states: Optional[dict] = None,
                 course_id: Optional[str] = None) -> None:
        self.learner_id = learner_id
        self.course_id = course_id
        self.states: dict[str, ItemState] = states or {}

    # --- keys ---------------------------------------------------------------

    def key(self, item_id: str) -> str:
        return item_id if self.course_id is None else f"{self.course_id}::{item_id}"

    # --- persistence --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, learner_id: str,
             course_id: Optional[str] = None) -> "LearnerModel":
        path = Path(path)
        if not path.exists():
            return cls(learner_id, course_id=course_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = data.get(learner_id, {})
        states = {k: ItemState(**v) for k, v in rec.items()}
        return cls(learner_id, states, course_id=course_id)

    def save(self, path: str | Path) -> None:
        # Merge, don't replace: a learner's record holds states for every course,
        # and saving one course must not erase another course's states.
        path = Path(path)
        data = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        record = data.get(self.learner_id, {})
        record.update({k: asdict(v) for k, v in self.states.items()})
        data[self.learner_id] = record
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    # --- queries ------------------------------------------------------------

    def state_for(self, item_id: str) -> Optional[ItemState]:
        return self.states.get(self.key(item_id))

    def is_due(self, item_id: str, now: datetime) -> bool:
        st = self.state_for(item_id)
        return st is not None and _parse(st.due) <= now

    # --- updates ------------------------------------------------------------

    def record_result(
        self,
        item_id: str,
        solved_level: Optional[int],
        achieved: bool,
        now: datetime,
        level: Optional[LevelConfig] = None,
    ) -> ItemState:
        level = level or _default_level()
        intervals = level.leitner_box_intervals_days
        max_box = max(intervals)

        prev = self.state_for(item_id)
        box = prev.box if prev else 1

        if not achieved:
            box = 1
        elif solved_level is not None and solved_level <= level.promote_max_level:
            box = min(box + 1, max_box)
        elif solved_level is not None and solved_level <= level.hold_max_level:
            box = max(box, 1)  # unchanged
        else:
            box = 1

        due = now + timedelta(days=intervals[box])
        st = ItemState(item_id=item_id, box=box, due=_iso(due), last_seen=_iso(now))
        self.states[self.key(item_id)] = st
        return st
