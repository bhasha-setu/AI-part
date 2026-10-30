"""Server config: judge thresholds (config/judge.yaml) and reply strings (config/ui/).

Thresholds are audited with the core's ``audit_thresholds``, so every cap and cutoff
carries a source. Learner-facing lines live in the UI file, not in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from guided_practice.config import CONFIG_DIR, ConfigError, audit_thresholds, unwrap

VERDICT_KEYS = ("correct", "surface_error", "meaning_error", "off_topic")
LEVELS = 4  # the hint ladder's four rungs (guided_practice.session.MAX_LEVEL + 1)
STAGE_LINE_KEYS = ("intro", "model_dialogue", "useful_language", "roleplay", "assessment",
                   "complete", "held")


@dataclass(frozen=True)
class JudgeConfig:
    status: str
    model: str
    effort: str
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    min_confidence: float
    reply_max_sentences: int
    reply_max_chars: int
    correction_max_chars: int
    accepted_form_floor: bool
    corrections_per_session: int
    corrections_per_error_id: int
    feedback_max_chars: int
    allow_unlisted_corrections: bool
    system_prompt: str
    rubric: str
    constraints: tuple
    learner_language_rule: dict
    gemini_model: str = ""
    reveal_accepted_min_tokens: int = 0
    gemini_thinking_level: str = ""


@dataclass(frozen=True)
class ReplyStrings:
    """Server-side lines from ``config/ui/<code>.yaml``."""

    code: str
    encourage: str
    retry_without_correction: str   # "{prompt}" is the current hint-level prompt
    stage_lines: dict           # stage -> line
    guided_lines: dict          # verdict -> [line for level 0..3]
    correction_markers: tuple
    sentence_terminators: str

    def guided(self, verdict: str, level: int) -> str:
        lines = self.guided_lines[verdict]
        return lines[min(max(level, 0), len(lines) - 1)]


def _read(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


@lru_cache(maxsize=None)
def load_judge_config(path: str | Path | None = None) -> JudgeConfig:
    path = Path(path) if path else CONFIG_DIR / "judge.yaml"
    raw = _read(path)
    problems = audit_thresholds(raw)
    if problems:
        raise ConfigError(f"{path.name}: " + "; ".join(problems))
    v = unwrap(raw)
    return JudgeConfig(
        status=v["status"],
        model=v["model"],
        effort=v["effort"],
        max_tokens=v["max_tokens"],
        timeout_seconds=v["timeout_seconds"],
        max_retries=v["max_retries"],
        min_confidence=v["min_confidence"],
        reply_max_sentences=v["reply_max_sentences"],
        reply_max_chars=v["reply_max_chars"],
        correction_max_chars=v["correction_max_chars"],
        accepted_form_floor=bool(v["accepted_form_floor"]),
        corrections_per_session=v["corrections_per_session"],
        corrections_per_error_id=v["corrections_per_error_id"],
        feedback_max_chars=v["feedback_max_chars"],
        allow_unlisted_corrections=bool(v["allow_unlisted_corrections"]),
        system_prompt=v["system_prompt"],
        rubric=v["rubric"],
        constraints=tuple(v["constraints"]),
        learner_language_rule=dict(v["learner_language_rule"]),
        gemini_model=v.get("gemini_model", ""),
        reveal_accepted_min_tokens=v.get("reveal_accepted_min_tokens", 0),
        gemini_thinking_level=v.get("gemini_thinking_level") or "",
    )


@lru_cache(maxsize=None)
def load_reply_strings(code: str) -> ReplyStrings:
    raw = _read(CONFIG_DIR / "ui" / f"{code}.yaml")
    lines = raw["fallback_lines"]
    guided = {k: list(lines["guided_practice"][k]) for k in VERDICT_KEYS}
    for verdict, row in guided.items():
        if len(row) != LEVELS:
            raise ConfigError(f"ui/{code}: fallback_lines.guided_practice.{verdict} needs "
                              f"{LEVELS} lines (one per hint level)")
    return ReplyStrings(
        code=code,
        encourage=raw["encourage"],
        retry_without_correction=raw["retry_without_correction"],
        stage_lines={k: lines[k] for k in STAGE_LINE_KEYS},
        guided_lines=guided,
        correction_markers=tuple(m.casefold() for m in raw["correction_markers"]),
        sentence_terminators=raw["sentence_terminators"],
    )


def replies_may_use_learner_language(profile) -> bool:
    """Profile flag read by the server only (the core Profile doesn't carry it)."""
    raw = _read(profile.directory / "profile.yaml")
    return bool(raw.get("replies_may_use_learner_language", False))
