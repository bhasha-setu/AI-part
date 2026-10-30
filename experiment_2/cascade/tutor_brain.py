"""The experiment_1 tutor as the cascade's LLM stage.

The tutor (experiment_1/tutor) owns the lesson: practice phrases with a hint ladder, a
short roleplay, a review of 2-3 mistakes, then a summary and the next support stage.
This module only adapts it to voice; experiment_1 is imported read-only, never changed:

    TutorBrain     ChatLLM for the pipeline: speaks the tutor's opening, passes each
                   transcript to ``session.turn()``, yields the reply sentence by sentence,
                   and saves the summary when the lesson is done.
    JSONChain      the tutor's model: tries each key in order (Gemini 1, Gemini 2, ...)
                   with a cool-down. When every key fails it raises experiment_1's
                   LLMUnavailable, so the tutor falls back to its own rule-based lines.
    VoiceGuard     wraps the model: drops "mistakes" that are speech-recognition artefacts
                   (spelling, punctuation, how a name is written) before the tutor sees
                   them, so they never reach the review. The prompt says so too
                   (``system_prompt_append``); this makes it hold even when the model slips.

Voice-specific settings come from experiment_2/config.yaml (``tutor:``), merged over
experiment_1/config/tutor.yaml: Hindi in Devanagari (the Hindi voice can't read Roman
Hindi) and minimal thinking (latency).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import yaml

from .llm import sentences_from_chunks

EXPERIMENT_1 = Path(__file__).resolve().parent.parent.parent / "experiment_1"


def use_experiment_1() -> None:
    """Make experiment_1's packages importable (appended, so nothing shadows ours)."""
    path = str(EXPERIMENT_1)
    if path not in sys.path:
        sys.path.append(path)


use_experiment_1()

from guided_practice.course import load_course  # noqa: E402
from guided_practice.learner_model import LearnerModel  # noqa: E402
from pedagogy_server.llm import GeminiModel, LLMUnavailable  # noqa: E402
from tutor.config import Profiles, TutorConfig, load_tutor_config  # noqa: E402
from tutor.session import TutorSession, _now  # noqa: E402


# --- the tutor's model: one JSON call per turn, across several keys -----------------------

class JSONChain:
    """``generate_json`` over several backends, in order. A backend that fails for a
    provider reason (429, timeout, network) is skipped for ``cooldown_seconds``. A
    ``max_tokens`` failure is the request's fault, not the provider's, so it isn't benched."""

    def __init__(self, backends: list, cooldown_seconds: float = 60, clock=time.monotonic) -> None:
        if not backends:
            raise ValueError("JSONChain needs at least one backend")
        self.backends = backends
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._benched_until: dict[str, float] = {}
        self.last_used: Optional[str] = None
        self.errors: list[str] = []      # since the last successful call

    @property
    def name(self) -> str:
        return " > ".join(b.name for b in self.backends)

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        now = self._clock()
        ready = [b for b in self.backends if self._benched_until.get(b.name, 0) <= now]
        self.errors = []
        for backend in ready or self.backends:
            try:
                raw = backend.generate_json(system=system, user=user, schema=schema)
            except LLMUnavailable as exc:
                self.errors.append(f"{backend.name}:{exc}")
                if str(exc) != "max_tokens":
                    self._benched_until[backend.name] = self._clock() + self.cooldown_seconds
                continue
            self.last_used = backend.name
            return raw
        self.last_used = None
        raise LLMUnavailable("all_failed:" + ";".join(self.errors))


def build_json_chain(tutor_cfg: TutorConfig, providers: list, env=None, *,
                     cooldown_seconds: float = 60) -> JSONChain:
    """One GeminiModel per Gemini key in ``providers`` (GEMINI_API_KEY, _2, ...)."""
    env = os.environ if env is None else env
    backends = []
    for p in providers:
        if p["type"] != "gemini":
            continue   # only Gemini is checked against the tutor's JSON schemas so far
        for i in range(1, 10):
            var = p["key_env"] if i == 1 else f"{p['key_env']}_{i}"
            if not env.get(var):
                continue
            model = GeminiModel(p.get("model", tutor_cfg.model),
                                thinking_level=tutor_cfg.thinking_level or None,
                                timeout_seconds=tutor_cfg.timeout_seconds,
                                max_output_tokens=tutor_cfg.max_output_tokens,
                                env={"GEMINI_API_KEY": env[var]})
            model.name = f"gemini:{model.model}[key{i}]"
            backends.append(model)
    if not backends:
        raise RuntimeError("no Gemini key set for the tutor (GEMINI_API_KEY in the repo-root .env)")
    return JSONChain(backends, cooldown_seconds=cooldown_seconds)


def load_voice_tutor_config(overrides: dict) -> TutorConfig:
    """experiment_1's tutor.yaml with ``overrides`` applied. Goes through the YAML (not
    dataclasses.replace) because the script rule is baked into the system prompt at load."""
    raw = yaml.safe_load((EXPERIMENT_1 / "config" / "tutor.yaml").read_text(encoding="utf-8"))
    overrides = dict(overrides or {})
    append = overrides.pop("system_prompt_append", "")
    raw.update(overrides)
    if append:
        raw["system_prompt"] = raw["system_prompt"].rstrip() + "\n\n" + append.strip() + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as f:
        yaml.safe_dump(raw, f, allow_unicode=True)
        path = Path(f.name)
    try:
        return load_tutor_config(path)
    finally:
        path.unlink(missing_ok=True)


# Error types that a transcript can't show to be the learner's: Whisper wrote them.
TRANSCRIPT_ARTEFACTS = {"spelling", "punctuation", "capitalization", "capitalisation",
                        "proper_noun", "name", "transcription"}


class VoiceGuard:
    """Filters the model's conversation output: errors of a TRANSCRIPT_ARTEFACTS type
    are removed. Everything else passes through unchanged."""

    def __init__(self, model) -> None:
        self.model = model
        self.dropped: list[dict] = []   # what was removed, for the log

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def last_used(self) -> Optional[str]:
        return getattr(self.model, "last_used", None)

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        raw = self.model.generate_json(system=system, user=user, schema=schema)
        if "errors" not in schema.get("properties", {}):
            return raw
        try:
            out = json.loads(raw)
        except ValueError:
            return raw   # the tutor reports bad JSON itself and retries
        errors = out.get("errors") or []
        kept = [e for e in errors
                if str(e.get("type", "")).strip().lower() not in TRANSCRIPT_ARTEFACTS]
        if len(kept) == len(errors):
            return raw
        self.dropped += [e for e in errors if e not in kept]
        out["errors"] = kept
        return json.dumps(out, ensure_ascii=False)


# --- the brain the pipeline talks to ------------------------------------------------------

class TutorBrain:
    """One lesson. ``stream`` ignores the pipeline's chat history: the session keeps
    its own state (phase, hint level, mistakes)."""

    def __init__(self, cfg: TutorConfig, model, *, course_id: str, scenario_id: Optional[str],
                 learner_id: str, state_dir: Path, stage: Optional[int] = None) -> None:
        self.cfg = cfg
        self.model = model
        self.course = load_course(course_id)
        self.scenario = self.course.load_scenario(scenario_id or self.course.scenario_ids()[0])
        self.learner_id = learner_id
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.profiles = Profiles(self.state_dir / "profiles.json")
        if stage is None:
            stage = self.profiles.stage(learner_id, self.course.course_id, cfg.start_stage)
        self.session = TutorSession(self.scenario, cfg, model, stage, log=self._log)
        self.name = f"tutor[{getattr(model, 'name', 'offline')}]"
        self.last_debug: dict = {}
        self.done = False

    def _log(self, event: dict) -> None:
        with (self.state_dir / "tutor_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    @property
    def last_used(self) -> Optional[str]:
        """Which key answered the last turn, or "rules" when the tutor fell back."""
        if self.last_debug.get("source") != "llm":
            return "rules"
        return getattr(self.model, "last_used", None) or getattr(self.model, "name", None)

    def _sentences(self, reply) -> Iterator[str]:
        self.last_debug = reply.debug
        self.done = reply.done
        for seg in reply.segments:
            yield from sentences_from_chunks([seg.text])

    def opening(self) -> Iterator[str]:
        return self._sentences(self.session.start())

    def stream(self, history: list[dict], text: str) -> Iterator[str]:
        return self._sentences(self.session.turn(text))

    @property
    def header(self) -> str:
        stage = self.cfg.stages[self.session.stage]
        return f"{self.scenario.title} · support stage {self.session.stage}: {stage.name}"

    def finish(self, now=None) -> list[str]:
        """Save spaced review and the next support stage; return a short report."""
        summary = self.session.summary()
        now = now or _now()
        lm_path = self.state_dir / "learners.json"
        lm = LearnerModel.load(lm_path, self.learner_id, course_id=self.course.course_id)
        for rec in self.session.records:
            lm.record_result(rec.item_id, rec.solved_level, rec.achieved, now, self.course.level)
        lm.save(lm_path)
        self.profiles.set_stage(self.learner_id, self.course.course_id, summary.stage_after)
        lines = []
        for rec in summary.items:
            how = ("from memory" if rec["achieved"] and rec["solved_level"] <= 1 and not rec["used_hindi"]
                   else "with help" if rec["achieved"] else "not yet")
            lines.append(f"  {rec['item_id']}: {how}" + (", started in Hindi" if rec["used_hindi"] else ""))
        lines.append(f"From memory: {summary.from_memory:.0%}")
        lines += [f"  Worked on: \"{r['said']}\" -> \"{r['fixed']}\"" for r in summary.reviewed]
        if summary.stage_after != summary.stage_before:
            word = "less" if summary.stage_after > summary.stage_before else "more"
            lines.append(f"Next lesson: {word} Hindi support (stage {summary.stage_after}).")
        return lines


def build_tutor(cfg: dict, llm_cfg: dict, *, offline: bool, state_dir: Path,
                scenario: Optional[str] = None, stage: Optional[int] = None,
                env=None) -> TutorBrain:
    tutor_cfg = load_voice_tutor_config(cfg.get("overrides", {}))
    model = None if offline else VoiceGuard(build_json_chain(
        tutor_cfg, llm_cfg.get("providers", []), env,
        cooldown_seconds=llm_cfg.get("cooldown_seconds", 60)))
    return TutorBrain(tutor_cfg, model, course_id=cfg.get("course", "english_for_hindi_a1"),
                      scenario_id=scenario or cfg.get("scenario"),
                      learner_id=cfg.get("learner", "voice-learner"),
                      state_dir=state_dir, stage=stage)
