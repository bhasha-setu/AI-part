"""Learners and their progress through the curriculum.

Two separate progressions are kept per learner, and shown separately:

    curriculum progress   which lessons are done (stage 1 Basics, 2 Everyday, 3 Real
                          world). Kept here, in progress.json.
    Hindi help level      how much Hindi the tutor uses (Hindi first -> English only).
                          the tutor's tutor keeps it (profiles.json), moving one level
                          after each lesson.

A lesson is "complete" when every one of its phrases was achieved (said correctly,
with or without hints). Otherwise the learner can redo it or move on anyway
("moved_on": the missed phrases stay due in spaced review). A quit mid-lesson records
nothing.

State lives in ``runs/learners/<username>/`` (git-ignored): progress.json, plus
the tutor's learners.json (spaced review) and profiles.json (Hindi help level).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .tutor_brain import use_tutor_packages

use_tutor_packages()

from guided_practice.content import Scenario, load_scenario  # noqa: E402

COMPLETE, MOVED_ON = "complete", "moved_on"
DONE_STATUSES = (COMPLETE, MOVED_ON)


def safe_username(name: str) -> str:
    """Lowercase letters, digits, - and _ only: the name is a folder name."""
    clean = re.sub(r"[^a-z0-9_-]+", "-", (name or "").strip().casefold()).strip("-")[:32]
    if not clean:
        raise ValueError("a username needs at least one letter or digit")
    return clean


def write_json_atomic(path: Path, data) -> None:
    """Write via a temp file and rename, so Ctrl-C can't leave half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# --- the curriculum ---------------------------------------------------------------------

@dataclass(frozen=True)
class LessonRef:
    id: str
    stage: int            # 1-based
    stage_name: str
    index: int            # 1-based, within the stage
    stage_size: int

    @property
    def label(self) -> str:
        return f"Stage {self.stage} · {self.stage_name} · lesson {self.index} of {self.stage_size}"


class Curriculum:
    def __init__(self, manifest: Path) -> None:
        self.root = Path(manifest).parent
        raw = yaml.safe_load(Path(manifest).read_text(encoding="utf-8"))
        self.course_id = raw["course"]
        self.stages = [{"name": s["name"], "goal": s.get("goal", ""), "lessons": list(s["lessons"])}
                       for s in raw["stages"]]
        self.lessons = [LessonRef(lid, n, s["name"], i, len(s["lessons"]))
                        for n, s in enumerate(self.stages, 1)
                        for i, lid in enumerate(s["lessons"], 1)]
        self._by_id = {ref.id: ref for ref in self.lessons}
        if len(self._by_id) != len(self.lessons):
            raise ValueError("a lesson id appears twice in the curriculum")

    def ref(self, lesson_id: str) -> LessonRef:
        return self._by_id[lesson_id]

    def path(self, lesson_id: str) -> Path:
        return self.root / "lessons" / f"{lesson_id}.yaml"

    def load(self, lesson_id: str, lang) -> Scenario:
        return load_scenario(self.path(lesson_id), lang)

    def title(self, lesson_id: str) -> str:
        raw = yaml.safe_load(self.path(lesson_id).read_text(encoding="utf-8"))
        return raw["scenario"]["title"]

    def stage_lessons(self, stage: int) -> list[LessonRef]:
        return [ref for ref in self.lessons if ref.stage == stage]


# --- one learner's progress ---------------------------------------------------------------

class LearnerProgress:
    def __init__(self, root: Path, username: str) -> None:
        self.username = safe_username(username)
        self.dir = Path(root) / self.username
        self.path = self.dir / "progress.json"
        self.is_new = not self.path.exists()
        self.data = (json.loads(self.path.read_text(encoding="utf-8")) if not self.is_new
                     else {"username": self.username,
                           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "lessons": {}})

    def save(self) -> None:
        write_json_atomic(self.path, self.data)
        self.is_new = False

    def lesson(self, lesson_id: str) -> dict:
        return self.data["lessons"].get(lesson_id, {})

    def status(self, lesson_id: str) -> Optional[str]:
        return self.lesson(lesson_id).get("status")

    def attempts(self, lesson_id: str) -> list:
        return self.lesson(lesson_id).get("attempts", [])

    def is_done(self, lesson_id: str) -> bool:
        return self.status(lesson_id) in DONE_STATUSES

    def next_lesson(self, curriculum: Curriculum) -> Optional[LessonRef]:
        """The first lesson, in curriculum order, that isn't complete or moved on."""
        return next((ref for ref in curriculum.lessons if not self.is_done(ref.id)), None)

    def stage_done(self, curriculum: Curriculum, stage: int) -> bool:
        lessons = curriculum.stage_lessons(stage)
        return bool(lessons) and all(self.is_done(ref.id) for ref in lessons)

    def record(self, lesson_id: str, summary, *, variant: Optional[str] = None) -> dict:
        """Store a finished attempt. Returns it; ``attempt["all_achieved"]`` says whether
        the lesson is now complete. A lesson already complete stays complete on redo."""
        missed = [r["item_id"] for r in summary.items if not r["achieved"]]
        attempt = {
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "achieved": len(summary.items) - len(missed),
            "total": len(summary.items),
            "missed": missed,
            "from_memory": round(summary.from_memory, 2),
            "hindi_turns": summary.hindi_turns,
            "reviewed": [{"said": r.get("said"), "fixed": r.get("fixed"), "type": r.get("type")}
                         for r in summary.reviewed],
            "variant": variant,
            "all_achieved": not missed,
        }
        entry = self.data["lessons"].setdefault(lesson_id, {"attempts": []})
        entry["attempts"].append(attempt)
        if not missed:
            entry["status"] = COMPLETE
        else:
            entry.setdefault("status", "in_progress")
        self.save()
        return attempt

    def move_on(self, lesson_id: str) -> None:
        """Mark a lesson done although some phrases were missed."""
        entry = self.data["lessons"].setdefault(lesson_id, {"attempts": []})
        if entry.get("status") != COMPLETE:
            entry["status"] = MOVED_ON
        self.save()


def status_mark(status: Optional[str]) -> str:
    return {COMPLETE: "✓ complete", MOVED_ON: "~ moved on", "in_progress": "… started"}.get(status, "  not started")
