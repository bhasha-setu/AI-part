"""Policy interface for the lesson loop.

Stages report outcomes. They don't decide what happens next. After every stage
the Lesson asks a Policy, which returns a Decision:

    advance       move to the next stage
    repeat        run the same stage again (capped by max_stage_attempts)
    drop_easier   restart the item at easier content (not implemented yet)
    end           hold the lesson here

``HoldPolicy`` is a placeholder. It advances completed and stubbed stages and
holds on anything else. The reasoning core (Step 3) replaces it. No test should
depend on HoldPolicy's exact reasons, because those are placeholder text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

ACTIONS = ("advance", "repeat", "drop_easier", "end")


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action '{self.action}'; expected one of {ACTIONS}")
        if not self.reason:
            raise ValueError("every decision needs a reason string")


class Policy(Protocol):
    def after_stage(self, stage: str, record, history: list) -> Decision: ...


class HoldPolicy:
    """Placeholder policy. Replaced by the Step 3 reasoning core."""

    def after_stage(self, stage: str, record, history: list) -> Decision:
        # A skipped stage (nothing due) advances. Holding would block a learner
        # who has nothing to review. The skip reason is in the stage record.
        if record.status in ("completed", "stubbed", "skipped"):
            return Decision("advance", f"placeholder: {stage} {record.status}")
        return Decision("end", f"placeholder: {stage} {record.status}; holding lesson")
