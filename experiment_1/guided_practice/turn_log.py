"""Per-turn logging as JSON Lines.

One JSON object per turn, appended to a ``.jsonl`` file so later delayed-recall
analysis is a simple line scan. Follow-up turns are logged with
``turn_type="follow_up"`` and are excluded from scoring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TurnRecord:
    session_id: str
    scenario: str
    learner_id: str
    item_id: str
    hint_level: int
    transcript_raw: str
    transcript_normalised: str
    verdict: str
    feedback_shown: str
    timestamp: str
    turn_type: str = "scored"       # "scored" | "follow_up"
    matched_error_id: str = ""
    # Set only on the turn that closes an item:
    outcome: str = ""               # "produced" | "repeated" | "not_achieved"
    surface_errors_before_correct: int = 0
    course_id: str = ""             # keeps logs from different courses apart
    extra: dict = field(default_factory=dict)


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: TurnRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        path = Path(path)
        if not path.exists():
            return []
        out = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
