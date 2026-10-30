"""Roleplay stage: turn-driven interface and TODO stub.

Real-world roleplay is the first conversation without scaffolding, so it is where
transfer gets tested. It needs its own design pass first (docs/reasoning_core.md,
open questions).

The interface is turn-driven, like Session, so a real implementation can take
learner turns and an AI partner can reply:

    rp = roleplay.start(scenario, learner_id, history)
    while not rp.done:
        partner_line = rp.submit(learner_text)
    result = rp.result()

Turns are text only, but each carries ``audio: None`` so speech adapters can be
added later without changing the data shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RoleplayResult:
    status: str                       # "stubbed" | "completed"
    transcript: list = field(default_factory=list)   # [{speaker, text, audio}]
    reason: str = ""


class RoleplaySession(Protocol):
    done: bool
    prompt: str

    def submit(self, text: str) -> str: ...
    def result(self) -> RoleplayResult: ...


class RoleplayStage(Protocol):
    def start(self, scenario, learner_id: str, history: list) -> RoleplaySession: ...


class _StubSession:
    """Finishes at once with status "stubbed". Does not block the lesson."""

    def __init__(self) -> None:
        self.done = True
        self.prompt = ""

    def submit(self, text: str) -> str:
        raise RuntimeError("stub roleplay is already done")

    def result(self) -> RoleplayResult:
        return RoleplayResult(status="stubbed", transcript=[],
                              reason="TODO: roleplay stage not implemented")


class StubRoleplay:
    """TODO: open conversation with an AI partner, text first.

    Using an LLM here is a new decision and needs its own README record. Until then
    this stub reports "stubbed", which does not block the lesson.
    """

    def start(self, scenario, learner_id: str, history: list) -> RoleplaySession:
        return _StubSession()
