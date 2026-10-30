"""Scenario content: dataclasses, YAML loading, compilation and validation.

Scenarios live under ``courses/<course_id>/scenarios/*.yaml`` and are marked
DRAFT. Each item is compiled once, at load time, with its course's language
config, so the same loader works for any target language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .config import LanguageConfig, PROJECT_ROOT, COURSES_DIR, DEFAULT_COURSE
from .matching import compile_raw, compile_template, fill, match
from .normalise import normalise

# Compatibility constant: scenario directory of the default course.
SCENARIOS_DIR = COURSES_DIR / DEFAULT_COURSE / "scenarios"

_SLOT_RE = re.compile(r"\{(\w+)\}")


@dataclass
class CommonError:
    """A typical surface error with its one-line correction."""

    id: str
    pattern: str
    verdict: str
    feedback: str
    why: str = ""
    example: str = ""
    correction_example: str = ""   # the corrected form of `example`
    regex: bool = False
    # compiled
    _rx: object = field(default=None, repr=False)
    _names: list = field(default_factory=list, repr=False)
    _lang: object = field(default=None, repr=False)

    def compile(self, lang: LanguageConfig) -> None:
        self._lang = lang
        if self.regex:
            self._rx, self._names = compile_raw(self.pattern)
        else:
            self._rx, self._names = compile_template(self.pattern, single_word=True, lang=lang)

    def match(self, norm: str) -> Optional[dict]:
        return match(self._rx, self._names, norm, raw=self.regex, lang=self._lang)


@dataclass
class Item:
    id: str
    scenario: str
    prompt: str          # level 0 text
    cue: str             # level 1 text
    first_words: str     # level 2 text
    model_answer: str    # level 3 text (may contain {slots})
    accepted: list[str]
    common_errors: list[CommonError]
    follow_up: Optional[str] = None
    # compiled
    lang: object = field(default=None, repr=False)
    _accepted: list = field(default_factory=list, repr=False)
    vocab: set = field(default_factory=set, repr=False)

    def compile(self, lang: LanguageConfig) -> None:
        self.lang = lang
        self._accepted = [compile_template(t, single_word=False, lang=lang) for t in self.accepted]
        for ce in self.common_errors:
            ce.compile(lang)
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> set:
        words: set[str] = set()
        sources = list(self.accepted) + [self.model_answer, self.first_words]
        for src in sources:
            cleaned = _SLOT_RE.sub(" ", src)
            words.update(normalise(cleaned, self.lang).split())
        return words

    def match_accepted(self, norm: str) -> Optional[dict]:
        for rx, names in self._accepted:
            result = match(rx, names, norm, lang=self.lang)
            if result is not None:
                return result
        return None


@dataclass(frozen=True)
class DialogueTurn:
    """One line of the model conversation. ``audio`` is reserved for later TTS."""

    speaker: str
    text: str
    audio: Optional[str] = None


@dataclass(frozen=True)
class UsefulPhrase:
    """A phrase taught before practice. ``used_by`` lists the item ids that need it."""

    phrase: str
    meaning: str
    used_by: tuple = ()


@dataclass
class Scenario:
    id: str
    title: str
    can_do: str
    model_dialogue: list[DialogueTurn]
    useful_language: list[UsefulPhrase]
    items: list[Item]

    def item_by_id(self, item_id: str) -> Optional[Item]:
        return next((it for it in self.items if it.id == item_id), None)


def _default_lang() -> LanguageConfig:
    from .course import default_course

    return default_course().lang


def _item_from_dict(d: dict, scenario_id: str, lang: LanguageConfig) -> Item:
    errors = [
        CommonError(
            id=e["id"],
            pattern=e["pattern"],
            verdict=e.get("verdict", "surface_error"),
            feedback=e["feedback"],
            why=e.get("why", ""),
            example=e.get("example", ""),
            correction_example=e.get("correction_example", ""),
            regex=bool(e.get("regex", False)),
        )
        for e in d.get("common_errors", [])
    ]
    item = Item(
        id=d["id"],
        scenario=d.get("scenario", scenario_id),
        prompt=d["prompt"],
        cue=d["cue"],
        first_words=d["first_words"],
        model_answer=d["model_answer"],
        accepted=list(d.get("accepted", [])),
        common_errors=errors,
        follow_up=d.get("follow_up"),
    )
    item.compile(lang)
    return item


def load_scenario(path: str | Path, lang: LanguageConfig | None = None) -> Scenario:
    """Load and compile one scenario YAML file for the given target language."""
    path = Path(path)
    lang = lang or _default_lang()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    s = data["scenario"]
    items = [_item_from_dict(it, s["id"], lang) for it in s["items"]]
    dialogue = [
        DialogueTurn(speaker=t["speaker"], text=t["text"], audio=t.get("audio"))
        for t in s.get("model_dialogue", [])
    ]
    phrases = [
        UsefulPhrase(phrase=p["phrase"], meaning=p["meaning"], used_by=tuple(p.get("used_by", [])))
        for p in s.get("useful_language", [])
    ]
    return Scenario(
        id=s["id"],
        title=s["title"],
        can_do=s.get("can_do", ""),
        model_dialogue=dialogue,
        useful_language=phrases,
        items=items,
    )


def load_scenario_by_id(scenario_id: str, lang: LanguageConfig | None = None,
                        directory: Path | None = None) -> Scenario:
    directory = directory or SCENARIOS_DIR
    return load_scenario(directory / f"{scenario_id}.yaml", lang)


# --- validation -------------------------------------------------------------

def _dummy_fill(template: str) -> str:
    """Fill every {slot} with a harmless multi-use placeholder value."""
    return _SLOT_RE.sub(lambda m: "placeholderval", template)


def validate_scenario(scenario: Scenario) -> list[str]:
    """Return a list of content problems (empty list == content is sound).

    Checks that a learner who follows each hint level can actually succeed:
      * the model answer passes its own accepted list,
      * ``first_words`` is a prefix of the model answer,
      * every common-error ``example`` trips that pattern and is NOT accepted,
      * every ``correction_example`` passes the accepted list.
    """
    problems: list[str] = []
    for item in scenario.items:
        model_norm = normalise(_dummy_fill(item.model_answer), item.lang)
        if item.match_accepted(model_norm) is None:
            problems.append(
                f"{item.id}: model_answer does not match its own accepted list"
            )

        fw = normalise(item.first_words, item.lang)
        ma = normalise(_SLOT_RE.sub(" ", item.model_answer), item.lang)
        if fw and not ma.startswith(fw):
            problems.append(
                f"{item.id}: first_words '{fw}' is not a prefix of model_answer '{ma}'"
            )

        for ce in item.common_errors:
            if not ce.example:
                problems.append(f"{item.id}/{ce.id}: common_error has no example")
                continue
            ex = normalise(ce.example, item.lang)
            if ce.match(ex) is None:
                problems.append(
                    f"{item.id}/{ce.id}: example '{ex}' does not match its pattern"
                )
            if item.match_accepted(ex) is not None:
                problems.append(
                    f"{item.id}/{ce.id}: example '{ex}' is wrongly accepted as correct"
                )
            if not ce.correction_example:
                problems.append(
                    f"{item.id}/{ce.id}: common_error has no correction_example"
                )
            else:
                corr = normalise(ce.correction_example, item.lang)
                if item.match_accepted(corr) is None:
                    problems.append(
                        f"{item.id}/{ce.id}: correction '{corr}' is not accepted "
                        f"(following the hint would fail)"
                    )
    return problems
