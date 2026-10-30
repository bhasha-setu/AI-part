"""The scenario generator: the learner says what they want to learn, the LLM proposes
lessons and writes the content, and the server decides whether that content is safe to
teach from.

    plan(request)   -> 2-3 LessonOptions (tasks the learner chooses between)
    build(option)   -> a scenario YAML in the learner's library, or GenerationFailed

The split follows the judge: the LLM chooses *what* to practise and writes the
sentences; the pedagogy (stage order, hint ladder, retries, corrections, scheduling)
stays in the core and never comes from the model.

What the server enforces on generated content, beyond the core's validate_scenario:
  * The error-matching regexes are built here from the model's literal error_forms,
    never written by the model. No error form may match an accepted answer, so a
    correct sentence can't be marked wrong (the worst failure for this product).
  * Counts and lengths stay inside the level config and config/generator.yaml.
  * Target-language sentences are in the target language's script.
  * Neither prompt nor cue gives the answer away.
  * The dialogue models every item's answer before practice asks for it.
  * used_by points at real items, and every item has a phrase.
A build that fails these is retried once with the problem list, then refused; the
caller falls back to authored scenarios.

Generated scenarios are written per learner under ``<state_dir>/generated/`` and are
stamped GENERATED, DRAFT: nobody has reviewed them.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from guided_practice.config import CONFIG_DIR, ConfigError, audit_thresholds, unwrap
from guided_practice.content import _dummy_fill, load_scenario, validate_scenario
from guided_practice.course import Course
from guided_practice.normalise import normalise

from .config import load_judge_config
from .disclosure import contains_any_target, contains_target
from .llm import LLMUnavailable

RAW_LOG_LIMIT = 2000
_SLOT_RE = re.compile(r"\{(\w+)\}")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


# --- config ----------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratorConfig:
    status: str
    plan_model: str
    build_model: str
    build_thinking_level: str
    build_fallback_model: str
    plan_timeout_seconds: float
    build_timeout_seconds: float
    build_attempts: int
    min_options: int
    max_options: int
    option_title_max_chars: int
    option_text_max_chars: int
    max_preview_phrases: int
    min_minutes: int
    max_minutes: int
    request_max_chars: int
    option_ttl_seconds: int
    min_items: int
    model_answer_max_words: int
    min_accepted: int
    max_accepted: int
    max_errors_per_item: int
    max_error_forms: int
    min_dialogue_lines: int
    max_dialogue_lines: int
    text_max_chars: int
    scripts: dict
    plan_system_prompt: str
    build_system_prompt: str


@lru_cache(maxsize=None)
def load_generator_config(path: str | Path | None = None) -> GeneratorConfig:
    path = Path(path) if path else CONFIG_DIR / "generator.yaml"
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    problems = audit_thresholds(raw)
    if problems:
        raise ConfigError(f"{path.name}: " + "; ".join(problems))
    v = unwrap(raw)
    if v["min_options"] > v["max_options"]:
        raise ConfigError(f"{path.name}: min_options is above max_options")
    names = GeneratorConfig.__dataclass_fields__
    return GeneratorConfig(**{k: v[k] for k in names})


# --- schemas sent to the model ----------------------------------------------------

_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "options": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "title": _STR, "can_do": _STR, "situation": _STR,
                "phrases_preview": _STRS, "why": _STR,
                "est_minutes": {"type": "integer"},
                "existing_scenario_id": {"anyOf": [_STR, {"type": "null"}]},
            },
            "required": ["title", "can_do", "situation", "phrases_preview", "why",
                         "est_minutes", "existing_scenario_id"],
        }},
    },
    "required": ["options"],
}

SCENARIO_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _STR,
        "can_do": _STR,
        "model_dialogue": {"type": "array", "items": {
            "type": "object", "properties": {"speaker": _STR, "text": _STR},
            "required": ["speaker", "text"]}},
        "useful_language": {"type": "array", "items": {
            "type": "object", "properties": {"phrase": _STR, "meaning": _STR, "used_by": _STRS},
            "required": ["phrase", "meaning", "used_by"]}},
        "items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": _STR, "prompt": _STR, "cue": _STR, "first_words": _STR,
                "model_answer": _STR, "accepted": _STRS,
                "common_errors": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"id": _STR, "error_forms": _STRS, "correction_example": _STR,
                                   "feedback": _STR, "why": _STR},
                    "required": ["id", "error_forms", "correction_example", "feedback", "why"]}},
                "follow_up": {"anyOf": [_STR, {"type": "null"}]},
            },
            "required": ["id", "prompt", "cue", "first_words", "model_answer", "accepted",
                         "common_errors", "follow_up"],
        }},
    },
    "required": ["title", "can_do", "model_dialogue", "useful_language", "items"],
}


# --- data ----------------------------------------------------------------------------

@dataclass(frozen=True)
class LessonOption:
    option_id: str
    title: str
    can_do: str
    situation: str
    phrases_preview: tuple
    why: str
    est_minutes: int
    scenario_id: Optional[str] = None     # set when an existing scenario fits

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phrases_preview"] = list(self.phrases_preview)
        return d


@dataclass
class PlanResult:
    options: list
    source: str                     # llm | fallback
    note: str = ""
    rejections: list = field(default_factory=list)
    raw: Optional[str] = None


@dataclass
class BuildResult:
    scenario_id: str
    path: Path
    title: str
    can_do: str
    item_count: int
    attempts: int
    log: list                       # one entry per attempt: problems and raw output


class GenerationFailed(Exception):
    def __init__(self, reason: str, log: list) -> None:
        super().__init__(reason)
        self.log = log


# --- the learner's library -------------------------------------------------------------

@dataclass(frozen=True)
class LearnerCourse(Course):
    """A Course that also serves the scenarios generated for one learner. Everything
    that loads scenarios goes through ``load_scenario``, so the core needs no change."""

    generated_dir: Path = None

    def generated_ids(self) -> tuple:
        if self.generated_dir is None or not self.generated_dir.exists():
            return ()
        return tuple(sorted(p.stem for p in self.generated_dir.glob("*.yaml")))

    def scenario_ids(self) -> tuple:
        return tuple(self.profile.scenarios) + self.generated_ids()

    def load_scenario(self, scenario_id: str):
        if scenario_id in self.profile.scenarios:
            return super().load_scenario(scenario_id)
        if scenario_id in self.generated_ids():
            return load_scenario(self.generated_dir / f"{scenario_id}.yaml", self.lang)
        raise KeyError(f"scenario '{scenario_id}' is neither in course '{self.course_id}' "
                       f"nor generated for this learner")


def _safe_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)[:64] or "_"


def learner_course(course: Course, state_dir: str | Path, learner_id: str) -> LearnerCourse:
    gen_dir = Path(state_dir) / "generated" / _safe_part(course.course_id) / _safe_part(learner_id)
    return LearnerCourse(profile=course.profile, lang=course.lang, level=course.level,
                         ui=course.ui, generated_dir=gen_dir)


# --- checks -------------------------------------------------------------------------

def _script_of(ch: str) -> str:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return ""


def wrong_script(text: str, script: str) -> bool:
    """True if any letter in ``text`` (outside {slots}) is not in ``script``."""
    text = _SLOT_RE.sub(" ", text)
    return any(ch.isalpha() and _script_of(ch) != script for ch in text)


def _short(text: str, limit: int) -> bool:
    return isinstance(text, str) and 0 < len(text.strip()) <= limit


def check_plan(data, cfg: GeneratorConfig, known_ids) -> list:
    if not isinstance(data, dict) or not isinstance(data.get("options"), list):
        return ["bad_shape"]
    opts = data["options"]
    problems = []
    if not cfg.min_options <= len(opts) <= cfg.max_options:
        problems.append(f"option_count:{len(opts)}")
    for i, o in enumerate(opts):
        if not isinstance(o, dict):
            problems.append(f"option{i}:bad_shape")
            continue
        if not _short(o.get("title"), cfg.option_title_max_chars):
            problems.append(f"option{i}:title")
        for key in ("can_do", "situation", "why"):
            if not _short(o.get(key), cfg.option_text_max_chars):
                problems.append(f"option{i}:{key}")
        phrases = o.get("phrases_preview")
        if (not isinstance(phrases, list) or len(phrases) > cfg.max_preview_phrases
                or not all(_short(p, cfg.option_title_max_chars) for p in phrases)):
            problems.append(f"option{i}:phrases_preview")
        minutes = o.get("est_minutes")
        if not isinstance(minutes, int) or not cfg.min_minutes <= minutes <= cfg.max_minutes:
            problems.append(f"option{i}:est_minutes")
        existing = o.get("existing_scenario_id")
        if existing is not None and existing not in known_ids:
            problems.append(f"option{i}:unknown_scenario_id")
    return problems


def _norm_forms(forms, lang) -> list:
    return [normalise(_dummy_fill(f), lang) for f in forms]


def _error_regex(forms, lang) -> str:
    """Exact-match regex over the normalised error forms. A {slot} matches one or more
    words. Built here, so the model never writes a regex."""
    alts = []
    for form in forms:
        pieces = _SLOT_RE.split(form)
        out = []
        for i, piece in enumerate(pieces):
            if i % 2:           # a slot name
                out.append(r"\S+(?: \S+)*")
            else:
                norm = normalise(piece, lang)
                if norm:
                    out.append(re.escape(norm))
        alts.append(" ".join(out))
    return "^(?:" + "|".join(alts) + ")$"


def check_scenario_data(data, cfg: GeneratorConfig, course: Course) -> list:
    """Structural and pedagogic checks on the model's JSON, before it becomes YAML."""
    if not isinstance(data, dict):
        return ["bad_shape"]
    problems = []
    reveal_min = load_judge_config().reveal_accepted_min_tokens
    script = cfg.scripts.get(course.profile.target_language)
    lang = course.lang
    items = data.get("items")
    max_items = course.level.max_items_per_session
    if not isinstance(items, list) or not cfg.min_items <= len(items) <= max_items:
        return problems + [f"item_count: need {cfg.min_items}-{max_items} items"]
    for key in ("title", "can_do"):
        if not _short(data.get(key), cfg.text_max_chars):
            problems.append(f"{key}: missing or over {cfg.text_max_chars} chars")

    dialogue = data.get("model_dialogue") or []
    if not cfg.min_dialogue_lines <= len(dialogue) <= cfg.max_dialogue_lines:
        problems.append(f"dialogue: need {cfg.min_dialogue_lines}-{cfg.max_dialogue_lines} lines")
    speakers = {d.get("speaker") for d in dialogue if isinstance(d, dict)}
    if len(speakers) != 2:
        problems.append("dialogue: needs exactly two speakers")
    for n, d in enumerate(dialogue):
        if not isinstance(d, dict) or not _short(d.get("text"), cfg.text_max_chars):
            problems.append(f"dialogue[{n}]: bad line")
        elif script and wrong_script(d["text"], script):
            problems.append(f"dialogue[{n}]: not in the target language's script")
    dialogue_text = " ".join(d.get("text", "") for d in dialogue if isinstance(d, dict))

    ids = [it.get("id") for it in items if isinstance(it, dict)]
    if len(set(ids)) != len(items):
        problems.append("items: ids missing or repeated")
    for it in items:
        if not isinstance(it, dict):
            problems.append("items: bad item")
            continue
        iid = it.get("id", "?")
        if not isinstance(iid, str) or not _ID_RE.match(iid):
            problems.append(f"{iid}: id must be lowercase letters, digits and _")
        for key in ("prompt", "cue", "first_words", "model_answer"):
            if not _short(it.get(key), cfg.text_max_chars):
                problems.append(f"{iid}: {key} missing or too long")
        answer = it.get("model_answer") or ""
        if len(_SLOT_RE.sub("x", answer).split()) > cfg.model_answer_max_words:
            problems.append(f"{iid}: model_answer over {cfg.model_answer_max_words} words")
        accepted = it.get("accepted") or []
        if not cfg.min_accepted <= len(accepted) <= cfg.max_accepted:
            problems.append(f"{iid}: need {cfg.min_accepted}-{cfg.max_accepted} accepted forms")
        targets = [answer, it.get("first_words") or ""] + list(accepted)
        if script and any(wrong_script(t, script) for t in targets if isinstance(t, str)):
            problems.append(f"{iid}: answer forms not in the target language's script")

        errors = it.get("common_errors") or []
        if len(errors) > cfg.max_errors_per_item:
            problems.append(f"{iid}: over {cfg.max_errors_per_item} common errors")
        accepted_norm = set(_norm_forms(accepted + [answer], lang))
        for e in errors:
            eid = e.get("id", "?") if isinstance(e, dict) else "?"
            forms = e.get("error_forms") if isinstance(e, dict) else None
            if not isinstance(eid, str) or not _ID_RE.match(eid):
                problems.append(f"{iid}/{eid}: bad error id")
            if not forms or len(forms) > cfg.max_error_forms or not all(isinstance(f, str) and f.strip() for f in forms):
                problems.append(f"{iid}/{eid}: need 1-{cfg.max_error_forms} error_forms")
                continue
            for f in _norm_forms(forms, lang):
                if f in accepted_norm:
                    problems.append(f"{iid}/{eid}: error form '{f}' is also an accepted answer")
            for key in ("feedback", "why", "correction_example"):
                if not _short(e.get(key), cfg.text_max_chars):
                    problems.append(f"{iid}/{eid}: {key} missing or too long")

        follow = it.get("follow_up")
        if follow is not None:
            if not _short(follow, cfg.text_max_chars) or ":" in follow:
                problems.append(f"{iid}: follow_up must be one spoken line, no narration")
            elif script and wrong_script(follow, script):
                problems.append(f"{iid}: follow_up not in the target language's script")

        # The dialogue must model the answer before practice asks for it.
        probe = _ProbeItem(answer, lang, [a for a in accepted if isinstance(a, str)])
        if answer and not contains_target(dialogue_text, probe):
            problems.append(f"{iid}: the dialogue never says the model answer")
        # Same reveal rule as the judge's: the answer or a longer accepted form.
        for key in ("prompt", "cue"):
            if answer and contains_any_target(it.get(key) or "", probe, reveal_min):
                problems.append(f"{iid}: {key} gives the answer away")

    phrases = data.get("useful_language") or []
    covered = set()
    for p in phrases:
        if not isinstance(p, dict) or not _short(p.get("phrase"), cfg.text_max_chars):
            problems.append("useful_language: bad phrase")
            continue
        unknown = [u for u in p.get("used_by") or [] if u not in ids]
        if unknown:
            problems.append(f"useful_language: used_by names unknown items {unknown}")
        covered.update(p.get("used_by") or [])
    missing = [i for i in ids if i not in covered]
    if missing:
        problems.append(f"useful_language: no phrase for items {missing}")
    return problems


@dataclass
class _ProbeItem:
    """Just enough of an Item for disclosure.contains_target."""

    model_answer: str
    lang: object
    accepted: list = field(default_factory=list)


# --- turning model JSON into content YAML ------------------------------------------------

def scenario_yaml(data: dict, scenario_id: str, course: Course, meta: dict) -> tuple:
    """The model's JSON in the content-file format, plus {error_id: error_forms}. Item and error ids get a
    scenario-specific prefix so they can't collide with authored ids in the learner
    model."""
    prefix = scenario_id.rsplit("_", 1)[-1]
    rename = {it["id"]: f"g{prefix}_{it['id']}" for it in data["items"]}
    items, error_forms = [], {}
    for it in data["items"]:
        errors = []
        for e in it.get("common_errors") or []:
            forms = list(e["error_forms"])
            error_forms[f"g{prefix}_{e['id']}"] = forms
            errors.append({
                "id": f"g{prefix}_{e['id']}",
                "pattern": _error_regex(forms, course.lang),
                "regex": True,
                "verdict": "surface_error",
                "feedback": e["feedback"],
                "why": e["why"],
                "example": _dummy_fill(forms[0]),
                "correction_example": _dummy_fill(e["correction_example"]),
            })
        items.append({
            "id": rename[it["id"]],
            "scenario": scenario_id,
            "prompt": it["prompt"],
            "cue": it["cue"],
            "first_words": it["first_words"],
            "model_answer": it["model_answer"],
            "accepted": list(it["accepted"]),
            "common_errors": errors,
            "follow_up": it.get("follow_up"),
        })
    return {
        "generated": meta,
        "scenario": {
            "id": scenario_id,
            "title": data["title"],
            "can_do": data["can_do"],
            "model_dialogue": [{"speaker": d["speaker"], "text": d["text"], "audio": None}
                               for d in data["model_dialogue"]],
            "useful_language": [{"phrase": p["phrase"], "meaning": p["meaning"],
                                 "used_by": [rename.get(u, u) for u in p.get("used_by") or []]}
                                for p in data["useful_language"]],
            "items": items,
        },
    }, error_forms


HEADER = ("# GENERATED, DRAFT. Written by an LLM and checked by pedagogy_server/generator.py.\n"
          "# Not reviewed by a teacher. The 'generated' block records how it was made.\n")


def _overlap_problems(scenario, error_forms: dict) -> list:
    """On the compiled content: no error pattern matches a correct answer (the model
    answer or any accepted form), and every error form is caught by its pattern and
    not accepted."""
    problems = []
    for item in scenario.items:
        correct = [item.model_answer] + list(item.accepted)
        for ce in item.common_errors:
            for form in correct:
                norm = normalise(_dummy_fill(form), item.lang)
                if ce.match(norm) is not None:
                    problems.append(f"{item.id}/{ce.id}: error pattern matches the correct answer '{norm}'")
            for form in error_forms.get(ce.id, ()):
                norm = normalise(_dummy_fill(form), item.lang)
                if ce.match(norm) is None:
                    problems.append(f"{item.id}/{ce.id}: error form '{norm}' is not caught")
                if item.match_accepted(norm) is not None:
                    problems.append(f"{item.id}/{ce.id}: error form '{norm}' is accepted as correct")
    return problems


def _write_and_check(doc: dict, path: Path, course: Course, error_forms: dict) -> list:
    """Write to a temporary file, load it with the core loader, run validate_scenario.
    The file is moved into place only when it is clean."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(HEADER + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    try:
        scenario = load_scenario(tmp, course.lang)
    except Exception as exc:     # a malformed file is a content problem, not a crash
        tmp.unlink(missing_ok=True)
        return [f"load_failed: {type(exc).__name__}: {exc}"]
    problems = validate_scenario(scenario) + _overlap_problems(scenario, error_forms)
    if problems:
        tmp.unlink(missing_ok=True)
        return problems
    tmp.replace(path)
    return []


# --- the generator ---------------------------------------------------------------------

def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:30].strip("_") or "lesson"


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"\w+", text.casefold()) if len(t) > 2}


class ScenarioGenerator:
    """``plan_model`` and ``build_model`` are llm.JSONModel objects; either may be None,
    in which case plan falls back to authored scenarios and build is refused.
    ``build_fallback_model`` takes over for the remaining attempts after the build model
    is unavailable."""

    def __init__(self, plan_model=None, build_model=None, cfg: Optional[GeneratorConfig] = None,
                 clock=time.time, build_fallback_model=None) -> None:
        self.plan_model = plan_model
        self.build_model = build_model
        self.build_fallback_model = build_fallback_model
        self.cfg = cfg or load_generator_config()
        self.clock = clock

    @property
    def enabled(self) -> bool:
        return self.build_model is not None

    # -- plan --

    def _catalogue(self, course) -> list:
        out = []
        for sid in course.scenario_ids():
            sc = course.load_scenario(sid)
            out.append({"scenario_id": sid, "title": sc.title, "can_do": sc.can_do})
        return out

    def plan(self, request: str, course) -> PlanResult:
        request = (request or "").strip()[: self.cfg.request_max_chars]
        catalogue = self._catalogue(course)
        known = {c["scenario_id"] for c in catalogue}
        if self.plan_model is None:
            return self._fallback_plan(request, catalogue, "planner_off", [], None)
        payload = {
            "learner_request": request,
            "target_language": course.profile.target_language,
            "learner_language": course.profile.learner_language,
            "cefr_level": course.profile.cefr_level,
            "interface_language": course.ui.code,
            "existing_scenarios": catalogue,
            "options_wanted": f"{self.cfg.min_options}-{self.cfg.max_options}",
            "minutes_range": [self.cfg.min_minutes, self.cfg.max_minutes],
        }
        try:
            raw = self.plan_model.generate_json(system=self.cfg.plan_system_prompt,
                                                user=json.dumps(payload, ensure_ascii=False),
                                                schema=PLAN_SCHEMA)
        except LLMUnavailable as exc:
            return self._fallback_plan(request, catalogue, f"llm_unavailable:{exc}", [], None)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._fallback_plan(request, catalogue, "not_json", ["not_json"], raw)
        problems = check_plan(data, self.cfg, known)
        if problems:
            return self._fallback_plan(request, catalogue, "rejected", problems, raw)
        options = [self._option(o, request, i) for i, o in enumerate(data["options"])]
        return PlanResult(options=options, source="llm", rejections=[],
                          raw=raw[:RAW_LOG_LIMIT])

    def _option(self, o: dict, request: str, i: int) -> LessonOption:
        digest = hashlib.sha1(f"{request}|{o['title']}|{i}|{self.clock()}".encode()).hexdigest()[:6]
        return LessonOption(
            option_id=f"opt-{digest}", title=o["title"].strip(), can_do=o["can_do"].strip(),
            situation=o["situation"].strip(), phrases_preview=tuple(o["phrases_preview"]),
            why=o["why"].strip(), est_minutes=o["est_minutes"],
            scenario_id=o.get("existing_scenario_id"))

    def _fallback_plan(self, request, catalogue, reason, problems, raw) -> PlanResult:
        """Rank existing scenarios by word overlap with the request."""
        words = _tokens(request)
        ranked = sorted(catalogue, key=lambda c: -len(words & _tokens(c["title"] + " " + c["can_do"])))
        options = [LessonOption(option_id=f"opt-{c['scenario_id']}", title=c["title"],
                                can_do=c["can_do"], situation="", phrases_preview=(), why="",
                                est_minutes=self.cfg.min_minutes, scenario_id=c["scenario_id"])
                   for c in ranked[: self.cfg.max_options]]
        return PlanResult(options=options, source="fallback",
                          note="I couldn't plan a new lesson just now, so here are lessons that already exist.",
                          rejections=[reason] + list(problems),
                          raw=raw[:RAW_LOG_LIMIT] if raw else None)

    # -- build --

    def build(self, option: LessonOption, course: LearnerCourse, request: str = "") -> BuildResult:
        if option.scenario_id:
            raise ValueError("this option is an existing scenario; start it directly")
        if self.build_model is None:
            raise GenerationFailed("generator_off", [])
        digest = hashlib.sha1(f"{option.option_id}|{self.clock()}".encode()).hexdigest()[:6]
        scenario_id = f"gen_{_slug(option.title)}_{digest}"
        path = course.generated_dir / f"{scenario_id}.yaml"
        payload = {
            "task": option.to_dict(),
            "learner_request": request,
            "target_language": course.profile.target_language,
            "learner_language": course.profile.learner_language,
            "cefr_level": course.profile.cefr_level,
            "interface_language": course.ui.code,
            "limits": {
                "items": [self.cfg.min_items, course.level.max_items_per_session],
                "model_answer_max_words": self.cfg.model_answer_max_words,
                "accepted_forms": [self.cfg.min_accepted, self.cfg.max_accepted],
                "common_errors_per_item": [0, self.cfg.max_errors_per_item],
                "dialogue_lines": [self.cfg.min_dialogue_lines, self.cfg.max_dialogue_lines],
                "max_chars_per_text": self.cfg.text_max_chars,
            },
        }
        log = []
        model = self.build_model
        for attempt in range(1, self.cfg.build_attempts + 1):
            entry = {"attempt": attempt, "model": getattr(model, "name", "?"), "problems": [], "raw": None}
            log.append(entry)
            try:
                raw = model.generate_json(
                    system=self.cfg.build_system_prompt,
                    user=json.dumps(payload, ensure_ascii=False), schema=SCENARIO_SCHEMA)
            except LLMUnavailable as exc:
                entry["problems"] = [f"llm_unavailable:{exc}"]
                model = self.build_fallback_model or model
                continue
            entry["raw"] = raw[:RAW_LOG_LIMIT]
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                entry["problems"] = ["not_json"]
            else:
                entry["problems"] = check_scenario_data(data, self.cfg, course)
                if not entry["problems"]:
                    meta = {"status": "GENERATED_DRAFT", "request": request,
                            "option": option.to_dict(), "model": getattr(model, "name", "?"),
                            "attempt": attempt, "created_unix": int(self.clock())}
                    doc, forms = scenario_yaml(data, scenario_id, course, meta)
                    entry["problems"] = _write_and_check(doc, path, course, forms)
                    if not entry["problems"]:
                        return BuildResult(scenario_id=scenario_id, path=path, title=data["title"],
                                           can_do=data["can_do"], item_count=len(data["items"]),
                                           attempts=attempt, log=log)
            # Retry with the problems, so the model can fix exactly those.
            payload["previous_attempt_problems"] = entry["problems"][:20]
        raise GenerationFailed("content failed validation", log)


def make_generator(kind: str) -> ScenarioGenerator:
    """``off`` (fallback planning only) or ``gemini``."""
    if kind == "off":
        return ScenarioGenerator()
    if kind == "gemini":
        from .llm import GeminiModel

        cfg = load_generator_config()
        plan = GeminiModel(cfg.plan_model, timeout_seconds=cfg.plan_timeout_seconds)
        build = GeminiModel(cfg.build_model, thinking_level=cfg.build_thinking_level or None,
                            timeout_seconds=cfg.build_timeout_seconds)
        backup = (GeminiModel(cfg.build_fallback_model, timeout_seconds=cfg.build_timeout_seconds)
                  if cfg.build_fallback_model else None)
        return ScenarioGenerator(plan, build, cfg, build_fallback_model=backup)
    raise ValueError(f"unknown generator '{kind}' (expected off or gemini)")
