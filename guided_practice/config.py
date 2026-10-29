"""Typed loaders for everything a course can vary.

Code reads config through these functions and never names a language, level or
course. Thresholds are stored as leaves of the form
``{value, source, evidence_scope}``; ``audit_thresholds`` checks that every
leaf has a source, so "cited or marked hypothesis" is enforced by a test rather
than by a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
COURSES_DIR = PROJECT_ROOT / "courses"

# Compatibility default for call sites that predate course profiles.
# New code should pass a Course explicitly.
DEFAULT_COURSE = "english_for_hindi_a1"

SUPPORTED_TOKENISATION = ("whitespace",)
ALLOWED_SOURCE_PREFIXES = ("brief", "research:", "framework:", "hypothesis", "placeholder")
ALLOWED_SCOPES = ("EFL", "general", "none")


class ConfigError(ValueError):
    """Malformed, missing or unsupported config."""


# --- raw YAML helpers --------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node and "source" in node


def iter_leaves(node: Any, path: tuple = ()):
    """Yield ``(path, leaf)`` for every threshold leaf in a config tree."""
    if is_leaf(node):
        yield path, node
    elif isinstance(node, dict):
        for key, child in node.items():
            yield from iter_leaves(child, path + (str(key),))


def unwrap(node: Any) -> Any:
    """Replace every threshold leaf with its bare value."""
    if is_leaf(node):
        return node["value"]
    if isinstance(node, dict):
        return {k: unwrap(v) for k, v in node.items()}
    return node


def audit_thresholds(node: Any, path: tuple = ()) -> list[str]:
    """Return problems with threshold sourcing. Empty list means the config is sound.

    Problems are: a leaf whose source is missing or not one of the allowed
    prefixes; a leaf whose evidence_scope is missing or unknown; and a bare
    number outside a leaf, which would be a magic number.
    """
    problems: list[str] = []
    if is_leaf(node):
        name = ".".join(path)
        source = str(node.get("source", ""))
        if not source.startswith(ALLOWED_SOURCE_PREFIXES):
            problems.append(f"{name}: source '{source}' must start with one of {ALLOWED_SOURCE_PREFIXES}")
        if node.get("evidence_scope") not in ALLOWED_SCOPES:
            problems.append(f"{name}: evidence_scope must be one of {ALLOWED_SCOPES}")
        return problems
    if isinstance(node, dict) and "value" in node:
        # A half-written leaf: a value with no source is a threshold without a citation.
        problems.append(f"{'.'.join(path)}: has 'value' but no 'source'")
        return problems
    if isinstance(node, dict):
        for key, child in node.items():
            problems += audit_thresholds(child, path + (str(key),))
    elif isinstance(node, list):
        for i, child in enumerate(node):
            problems += audit_thresholds(child, path + (str(i),))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        problems.append(f"{'.'.join(path)}: bare number {node} outside a threshold leaf")
    return problems


# --- typed config objects ----------------------------------------------------

@dataclass(frozen=True)
class LanguageConfig:
    code: str
    status: str
    script_notes: str
    tokenisation: str
    cased: bool
    apostrophe_variants: tuple
    contractions: dict          # {contraction: expansion}, both normalised
    leading_fillers: frozenset
    titlecase_slots: frozenset
    typo_min_token_length: int


@dataclass(frozen=True)
class LevelConfig:
    cefr: str
    status: str
    stage_budgets_seconds: dict
    max_items_per_session: int
    max_retries_per_level: int
    max_stage_attempts: int
    independent_max_level: int
    review_min_level: int
    promote_max_level: int
    hold_max_level: int
    leitner_box_intervals_days: dict


@dataclass(frozen=True)
class UIStrings:
    code: str
    status: str
    correct: str
    repeat_ok: str
    hint: str
    retry_prompt: str
    repeat_prompt: str
    lesson_continue: str
    lesson_headings: dict
    slot_display: dict


@dataclass(frozen=True)
class Profile:
    course_id: str
    status: str
    target_language: str
    learner_language: str
    cefr_level: str
    ui_strings: str
    scenarios: tuple
    directory: Path

    @property
    def scenarios_dir(self) -> Path:
        return self.directory / "scenarios"


# --- loaders -----------------------------------------------------------------

def _normalise_key(text: str, variants: tuple) -> str:
    # Mirrors guided_practice.normalise for contraction keys and fillers, so the
    # config and the transcripts are compared in the same form.
    import unicodedata

    t = unicodedata.normalize("NFKC", text)
    for variant in variants:
        t = t.replace(variant, "'")
    return t.casefold().strip()


@lru_cache(maxsize=None)
def load_language(code: str) -> LanguageConfig:
    raw = _read_yaml(CONFIG_DIR / "languages" / f"{code}.yaml")
    problems = audit_thresholds(raw)
    if problems:
        raise ConfigError(f"{code}: " + "; ".join(problems))
    tokenisation = raw["tokenisation"]
    if tokenisation not in SUPPORTED_TOKENISATION:
        raise ConfigError(
            f"{code}: tokenisation '{tokenisation}' is not supported "
            f"(supported: {SUPPORTED_TOKENISATION}). Refusing to guess word boundaries."
        )
    variants = tuple(raw.get("apostrophe_variants") or ())
    contractions = {
        _normalise_key(k, variants): _normalise_key(v, variants)
        for k, v in (raw.get("contractions") or {}).items()
    }
    fillers = frozenset(_normalise_key(f, variants) for f in raw.get("leading_fillers") or [])
    return LanguageConfig(
        code=code,
        status=raw["status"],
        script_notes=raw["script_notes"],
        tokenisation=tokenisation,
        cased=bool(raw["cased"]),
        apostrophe_variants=variants,
        contractions=contractions,
        leading_fillers=fillers,
        titlecase_slots=frozenset(raw.get("titlecase_slots") or []),
        typo_min_token_length=unwrap(raw["typo_min_token_length"]),
    )


@lru_cache(maxsize=None)
def load_level(cefr: str) -> LevelConfig:
    raw = _read_yaml(CONFIG_DIR / "levels" / f"{cefr}.yaml")
    problems = audit_thresholds(raw)
    if problems:
        raise ConfigError(f"{cefr}: " + "; ".join(problems))
    values = unwrap(raw)
    return LevelConfig(
        cefr=values["cefr"],
        status=values["status"],
        stage_budgets_seconds=values["stage_budgets_seconds"],
        max_items_per_session=values["max_items_per_session"],
        max_retries_per_level=values["max_retries_per_level"],
        max_stage_attempts=values["max_stage_attempts"],
        independent_max_level=values["independent_max_level"],
        review_min_level=values["review_min_level"],
        promote_max_level=values["promote_max_level"],
        hold_max_level=values["hold_max_level"],
        leitner_box_intervals_days={int(k): v for k, v in values["leitner_box_intervals_days"].items()},
    )


@lru_cache(maxsize=None)
def load_ui(code: str) -> UIStrings:
    raw = _read_yaml(CONFIG_DIR / "ui" / f"{code}.yaml")
    return UIStrings(
        code=code,
        status=raw["status"],
        correct=raw["correct"],
        repeat_ok=raw["repeat_ok"],
        hint=raw["hint"],
        retry_prompt=raw["retry_prompt"],
        repeat_prompt=raw["repeat_prompt"],
        lesson_continue=raw["lesson_continue"],
        lesson_headings=raw["lesson_headings"],
        slot_display=raw["slot_display"],
    )


@lru_cache(maxsize=None)
def load_profile(course_id: str) -> Profile:
    directory = COURSES_DIR / course_id
    raw = _read_yaml(directory / "profile.yaml")
    if raw["course_id"] != course_id:
        raise ConfigError(f"profile at {directory} declares course_id '{raw['course_id']}'")
    return Profile(
        course_id=course_id,
        status=raw["status"],
        target_language=raw["target_language"],
        learner_language=raw["learner_language"],
        cefr_level=raw["cefr_level"],
        ui_strings=raw["ui_strings"],
        scenarios=tuple(raw["scenarios"]),
        directory=directory,
    )
