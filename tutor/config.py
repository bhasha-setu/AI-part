"""Tutor config (config/tutor.yaml) and the learner's support stage.

The support stage is the Hindi -> English transition: stage 0 teaches in Hindi, the
last stage is English only. It is stored per learner and moves one step per lesson.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from guided_practice.config import CONFIG_DIR, unwrap


@dataclass(frozen=True)
class Stage:
    name: str
    rule: str


@dataclass(frozen=True)
class TutorConfig:
    model: str
    thinking_level: str
    timeout_seconds: float
    max_output_tokens: int
    retries_per_turn: int
    hindi_script: str
    start_stage: int
    stages: tuple
    promote_min_from_memory: float
    promote_max_hindi_turns: int
    demote_max_from_memory: float
    demote_min_hindi_turns: int
    hindi_bridge_level: int
    converse_min_turns: int
    converse_max_turns: int
    review_max_errors: int
    learner_text_max_chars: int
    system_prompt: str
    lines: dict

    def en(self, key: str, **values) -> str:
        text = self.lines[key]
        return text.format(**values) if values else text

    def hi(self, key: str) -> str:
        return self.lines[key][self.hindi_script]

    @property
    def last_stage(self) -> int:
        return len(self.stages) - 1

    def bridges_from_hindi(self, stage: int) -> bool:
        """Every stage but the last answers Hindi with the English for it."""
        return stage < self.last_stage


def load_tutor_config(path: Path | None = None) -> TutorConfig:
    raw = unwrap(yaml.safe_load((path or CONFIG_DIR / "tutor.yaml").read_text(encoding="utf-8")))
    script = raw["hindi_script"]
    if script not in raw["script_rule"]:
        raise ValueError(f"hindi_script must be one of {sorted(raw['script_rule'])}")
    stages = tuple(Stage(s["name"], " ".join(s["rule"].split())) for s in raw["stages"])
    if not 0 <= raw["start_stage"] < len(stages):
        raise ValueError("start_stage is not a stage index")
    return TutorConfig(
        model=raw["model"],
        thinking_level=raw.get("thinking_level") or "",
        timeout_seconds=float(raw["timeout_seconds"]),
        max_output_tokens=int(raw["max_output_tokens"]),
        retries_per_turn=int(raw["retries_per_turn"]),
        hindi_script=script,
        start_stage=int(raw["start_stage"]),
        stages=stages,
        promote_min_from_memory=float(raw["promote_min_from_memory"]),
        promote_max_hindi_turns=int(raw["promote_max_hindi_turns"]),
        demote_max_from_memory=float(raw["demote_max_from_memory"]),
        demote_min_hindi_turns=int(raw["demote_min_hindi_turns"]),
        hindi_bridge_level=int(raw["hindi_bridge_level"]),
        converse_min_turns=int(raw["converse_min_turns"]),
        converse_max_turns=int(raw["converse_max_turns"]),
        review_max_errors=int(raw["review_max_errors"]),
        learner_text_max_chars=int(raw["learner_text_max_chars"]),
        system_prompt=raw["system_prompt"].replace("{script_rule}", raw["script_rule"][script]),
        lines=raw["lines"],
    )


def next_stage(cfg: TutorConfig, stage: int, from_memory: float, hindi_turns: int) -> int:
    """The support stage for the next lesson."""
    if from_memory >= cfg.promote_min_from_memory and hindi_turns <= cfg.promote_max_hindi_turns:
        return min(stage + 1, cfg.last_stage)
    if from_memory <= cfg.demote_max_from_memory or hindi_turns >= cfg.demote_min_hindi_turns:
        return max(stage - 1, 0)
    return stage


class Profiles:
    """Per-learner, per-course support stage, in one JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def stage(self, learner_id: str, course_id: str, default: int) -> int:
        return self._read().get(learner_id, {}).get(course_id, {}).get("stage", default)

    def set_stage(self, learner_id: str, course_id: str, stage: int) -> None:
        data = self._read()
        data.setdefault(learner_id, {}).setdefault(course_id, {})["stage"] = stage
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
