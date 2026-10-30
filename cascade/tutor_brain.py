"""The tutor (tutor/) as the cascade's LLM stage.

The tutor (tutor/) owns the lesson: practice phrases with a hint ladder, a
short roleplay, a review of 2-3 mistakes, then a summary and the next support stage.
This module only adapts it to voice; the tutor packages aren't changed for it:

    TutorBrain     ChatLLM for the pipeline: speaks the tutor's opening, passes each
                   transcript to ``session.turn()``, yields the reply sentence by sentence,
                   and saves the summary when the lesson is done.
    JSONChain      the tutor's model: tries each key in order (Gemini 1, Gemini 2, ...)
                   with a cool-down. When every key fails it raises the tutor's
                   LLMUnavailable, so the tutor falls back to its own rule-based lines.
    VoiceGuard     wraps the model: drops "mistakes" that are speech-recognition artefacts
                   (spelling, punctuation, how a name is written) before the tutor sees
                   them, so they never reach the review. The prompt says so too
                   (``system_prompt_append``); this makes it hold even when the model slips.

Voice-specific settings come from config.yaml (``tutor:``), merged over
config/tutor.yaml: Hindi in Devanagari (the Hindi voice can't read Roman
Hindi) and minimal thinking (latency).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import yaml

from .llm import sentences_from_chunks

ROOT = Path(__file__).resolve().parent.parent


def use_tutor_packages() -> None:
    """Make the tutor packages (tutor/, guided_practice/, pedagogy_server/, next to
    cascade/) importable when this file is run from another folder."""
    path = str(ROOT)
    if path not in sys.path:
        sys.path.append(path)


use_tutor_packages()

from guided_practice.course import load_course  # noqa: E402
from guided_practice.learner_model import LearnerModel  # noqa: E402
from pedagogy_server.llm import GeminiModel, LLMUnavailable  # noqa: E402
from tutor.config import Profiles, TutorConfig, load_tutor_config  # noqa: E402
from tutor.session import TutorSession, _now  # noqa: E402


# --- the tutor's model: one JSON call per turn, across several keys -----------------------

class JSONChain:
    """``generate_json`` over several backends, in order. A backend that fails for a
    provider reason (429, timeout, network) is skipped for ``cooldown_seconds``. A
    ``max_tokens`` failure is the request's fault, not the provider's, so it isn't benched.

    ``hedge_after_seconds``: Gemini sometimes stalls for 10-20 s on a call that usually
    takes 1.5 s (seen 2026-10-30: 18.9 s and 12.8 s in one lesson). If a backend hasn't
    answered by then, the next one gets the same request too and the first answer wins.
    The slow call is left to finish in the background (its result is dropped)."""

    def __init__(self, backends: list, cooldown_seconds: float = 60, clock=time.monotonic,
                 hedge_after_seconds: Optional[float] = None) -> None:
        if not backends:
            raise ValueError("JSONChain needs at least one backend")
        self.backends = backends
        self.cooldown_seconds = cooldown_seconds
        self.hedge_after_seconds = hedge_after_seconds
        self._clock = clock
        self._benched_until: dict[str, float] = {}
        self.last_used: Optional[str] = None
        self.errors: list[str] = []      # since the last successful call
        self.hedged = 0                  # calls where a second backend was started

    @property
    def name(self) -> str:
        return " > ".join(b.name for b in self.backends)

    def warm_up(self) -> None:
        """Open each key's connection with a model lookup (no generation quota used)."""
        for b in self.backends:
            client = getattr(b, "_client", None)
            if client is not None:
                client.models.get(model=b.model)

    def _failed(self, backend, exc: LLMUnavailable) -> None:
        self.errors.append(f"{backend.name}:{exc}")
        if str(exc) != "max_tokens":
            self._benched_until[backend.name] = self._clock() + self.cooldown_seconds

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        now = self._clock()
        ready = [b for b in self.backends if self._benched_until.get(b.name, 0) <= now] or self.backends
        self.errors = []
        if self.hedge_after_seconds is None or len(ready) < 2:
            for backend in ready:
                try:
                    raw = backend.generate_json(system=system, user=user, schema=schema)
                except LLMUnavailable as exc:
                    self._failed(backend, exc)
                    continue
                self.last_used = backend.name
                return raw
        else:
            raw = self._hedged(ready, system=system, user=user, schema=schema)
            if raw is not None:
                return raw
        self.last_used = None
        raise LLMUnavailable("all_failed:" + ";".join(self.errors))

    def _hedged(self, ready: list, **request) -> Optional[str]:
        import queue
        import threading

        results: queue.Queue = queue.Queue()
        waiting = list(ready)

        def run(backend):
            try:
                results.put((backend, backend.generate_json(**request), None))
            except LLMUnavailable as exc:
                results.put((backend, None, exc))
            except Exception as exc:   # never leave the caller waiting on a dead thread
                results.put((backend, None, LLMUnavailable(f"error:{type(exc).__name__}")))

        def launch():
            threading.Thread(target=run, args=(waiting.pop(0),), daemon=True).start()

        launch()
        running = 1
        while running:
            try:
                backend, raw, exc = results.get(timeout=self.hedge_after_seconds if waiting else None)
            except queue.Empty:   # the running call is slow: start the next backend too
                launch()
                running += 1
                self.hedged += 1
                continue
            running -= 1
            if exc is None:
                self.last_used = backend.name
                return raw
            self._failed(backend, exc)
            if waiting and not running:   # failed fast: go straight to the next one
                launch()
                running += 1
        return None


def build_json_chain(tutor_cfg: TutorConfig, providers: list, env=None, *,
                     cooldown_seconds: float = 60,
                     hedge_after_seconds: Optional[float] = None) -> JSONChain:
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
    return JSONChain(backends, cooldown_seconds=cooldown_seconds,
                     hedge_after_seconds=hedge_after_seconds)


def deep_merge(base: dict, extra: dict) -> dict:
    """Merge ``extra`` into ``base`` in place; nested dicts are merged key by key."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# --- one consistent speaker ------------------------------------------------------------------
# Every line is spoken by one voice (ElevenLabs "George" is male), so the roleplay partner
# must be a man too. Role names (Waiter, Doctor) are kept; a partner with a personal name
# that isn't a known male name is renamed, in the dialogue and wherever the name appears.
MALE_NAMES = ("Arjun", "Rohan", "Vikram", "Sanjay", "Mohan", "Karan", "Rahul", "Amit", "Ravi", "Aman")
ROLE_NAMES = {"waiter", "doctor", "interviewer", "receptionist", "shopkeeper", "assistant",
              "manager", "clerk", "conductor", "pharmacist", "driver", "officer", "colleague"}


def male_partner(scenario):
    """The scenario with its roleplay partner (the first speaker) played by a man."""
    import re

    from guided_practice.content import DialogueTurn

    speakers = [t.speaker for t in scenario.model_dialogue]
    if not speakers:
        return scenario
    partner = speakers[0]
    if partner.lower() in ROLE_NAMES or partner in MALE_NAMES:
        return scenario
    others = set(speakers)
    new = next(n for n in MALE_NAMES if n not in others)
    rx = re.compile(rf"\b{re.escape(partner)}\b")
    dialogue = [DialogueTurn(new if t.speaker == partner else t.speaker, rx.sub(new, t.text), t.audio)
                for t in scenario.model_dialogue]
    return dataclasses.replace(scenario, model_dialogue=dialogue)


def load_voice_tutor_config(overrides: dict) -> TutorConfig:
    """the tutor's tutor.yaml with ``overrides`` applied. Goes through the YAML (not
    dataclasses.replace) because the script rule is baked into the system prompt at load."""
    raw = yaml.safe_load((ROOT / "config" / "tutor.yaml").read_text(encoding="utf-8"))
    overrides = dict(overrides or {})
    append = overrides.pop("system_prompt_append", "")
    deep_merge(raw, overrides)
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


# A tip on the learner's own sentence, every turn it helps. the tutor keeps corrections
# for the review at the end of the roleplay (and its checks reject a reply that quotes a
# fix), so the tip is a separate field the tutor's reply never sees: the app says it.
TIP = {"type": "object",
       "properties": {"said": {"type": "string"}, "better": {"type": "string"},
                      "type": {"type": "string"}, "why": {"type": "string"}}}
TIP_RULE = """\
Tips (the "tip" field): the app says your tip to the learner just before your reply.
- Read every English sentence the learner said this turn, even when it answers the
  question well. Fill tip when one has a real mistake (grammar, a wrong or missing word,
  word order, tense) or when a native speaker would say it differently (an awkward
  phrase, a filler like "and yeah"): said = their sentence, better = the corrected or
  more natural whole sentence (keep their meaning and their own details), type =
  grammar, word_choice, word_order, natural or filler, why = a short simple reason in
  English (at most 12 words). One tip per turn: the most useful one.
- Leave tip out when their English was already good, when they spoke Hindi, or when the
  only difference is spelling, punctuation, capitals or how a name or place is written:
  those come from the speech recogniser, not the learner.
- Your "say" still never corrects them: the tip is separate.
"""
TIP_MAX = 200
_LATIN = re.compile(r"^[\x00-\x7F’‘“”–—…]*$")
_WORDS = re.compile(r"[a-z0-9']+")


def check_tip(tip, learner_text: str) -> tuple[Optional[dict], str]:
    """(tip, "") when it is worth saying, else (None, why it was dropped)."""
    if not isinstance(tip, dict) or not str(tip.get("better") or "").strip():
        return None, ""
    better = str(tip["better"]).strip()
    if str(tip.get("type", "")).strip().lower() in TRANSCRIPT_ARTEFACTS:
        return None, "transcript_artefact"
    if not _LATIN.match(learner_text):
        return None, "learner_spoke_hindi"
    if not _LATIN.match(better) or len(better) > TIP_MAX:
        return None, "not_english_or_too_long"
    words = lambda t: _WORDS.findall(t.lower().replace("’", "'"))
    if words(better) in (words(learner_text), words(str(tip.get("said") or ""))):
        return None, "same_words"   # only spelling, capitals or punctuation differ
    return {"said": str(tip.get("said") or learner_text).strip(), "better": better,
            "type": str(tip.get("type") or "").strip(), "why": str(tip.get("why") or "").strip()}, ""


def gives_answer_away(text: str, item, lang) -> bool:
    """Does ``text`` contain the practice item's answer or its first words?"""
    from guided_practice.normalise import normalise

    norm = normalise(text, lang)
    answer = normalise(re.sub(r"\{\w+\}", " ", item.model_answer), lang)
    first = normalise(item.first_words, lang)
    return bool(answer and answer in norm or (len(first.split()) >= 2 and first in norm))


class VoiceGuard:
    """Wraps the tutor's model for voice.

    * errors of a TRANSCRIPT_ARTEFACTS type are removed (Whisper wrote them);
    * with ``tips``, practice and conversation calls also ask for a ``tip`` on the
      learner's sentence (TIP_RULE); the latest one is kept on ``last_tip`` for TutorBrain.
    Everything else passes through unchanged."""

    def __init__(self, model, *, tips: bool = False) -> None:
        self.model = model
        self.tips = tips
        self.dropped: list[dict] = []   # what was removed, for the log
        self.last_tip: Optional[dict] = None

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def last_used(self) -> Optional[str]:
        return getattr(self.model, "last_used", None)

    def warm_up(self) -> None:
        self.model.warm_up()

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        props = schema.get("properties", {})
        wants_tip = self.tips and ("errors" in props or "verdict" in props)
        if wants_tip:
            schema = {**schema, "properties": {**props, "tip": TIP}}
            system = system.rstrip() + "\n\n" + TIP_RULE
        self.last_tip = None
        raw = self.model.generate_json(system=system, user=user, schema=schema)
        if "errors" not in props and not wants_tip:
            return raw
        try:
            out = json.loads(raw)
        except ValueError:
            return raw   # the tutor reports bad JSON itself and retries
        errors = out.get("errors") or []
        kept = [e for e in errors
                if str(e.get("type", "")).strip().lower() not in TRANSCRIPT_ARTEFACTS]
        if wants_tip:
            # The model often lists a roleplay mistake for the review but leaves tip empty
            # (2 of 4 on 2026-10-30): then the most important listed mistake is the tip.
            top = max(kept, key=lambda e: e.get("importance") or 0, default=None)
            self.last_tip = out.get("tip") or (top and {
                "said": top.get("said"), "better": top.get("fixed"),
                "type": top.get("type"), "why": top.get("why")})
        if len(kept) == len(errors):
            return raw
        self.dropped += [e for e in errors if e not in kept]
        out["errors"] = kept
        return json.dumps(out, ensure_ascii=False)


# --- the brain the pipeline talks to ------------------------------------------------------

class TutorBrain:
    """One lesson. ``stream`` ignores the pipeline's chat history: the session keeps
    its own state (phase, hint level, mistakes).

    ``state_dir`` is the learner's folder: the tutor's spaced review (learners.json)
    and Hindi help level (profiles.json) are kept there."""

    def __init__(self, cfg: TutorConfig, model, *, scenario, course, learner_id: str,
                 state_dir: Path, hindi_help: Optional[int] = None) -> None:
        self.cfg = cfg
        self.model = model
        self.course = course
        self.scenario = scenario
        self.learner_id = learner_id
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.profiles = Profiles(self.state_dir / "profiles.json")
        if hindi_help is None:
            hindi_help = self.profiles.stage(learner_id, course.course_id, cfg.start_stage)
        self.session = TutorSession(scenario, cfg, model, hindi_help, log=self._log)
        self.name = f"tutor[{getattr(model, 'name', 'offline')}]"
        self.last_debug: dict = {}
        self.done = False
        self.summary = None

    def _log(self, event: dict) -> None:
        event = {"lesson": self.scenario.id, **event}
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
        item = self.session.item if self.session.phase == "practice" else None
        chain = getattr(self.model, "model", self.model)
        hedged = getattr(chain, "hedged", 0)
        reply = self.session.turn(text)
        if getattr(chain, "hedged", 0) > hedged:   # key 2 was asked too: key 1 was slow
            reply.debug = {**(reply.debug or {}), "hedged": chain.hedged - hedged}
        tip = self._tip(reply, text, item)
        if tip:
            yield f"Tip: you can say, \"{tip['better']}\""
        yield from self._sentences(reply)
        if tip:
            self.last_debug = {**self.last_debug, "tip": tip}

    def _tip(self, reply, text: str, item) -> Optional[dict]:
        """The model's tip on this turn, if it is worth saying. In practice, after an
        answer that isn't correct yet, a tip that would say the target for the learner
        is dropped: the hint ladder decides how much to give away."""
        raw = getattr(self.model, "last_tip", None)
        debug = reply.debug or {}
        if not raw or debug.get("source") != "llm" or debug.get("phase") not in ("practice", "conversation"):
            return None
        tip, dropped = check_tip(raw, text)
        answer_said = max(debug.get("hint_level") or 0, debug.get("next_hint_level") or 0) >= 3
        if tip and item is not None and debug.get("verdict") != "correct" and not answer_said \
                and gives_answer_away(tip["better"], item, self.course.lang):
            tip, dropped = None, "gives_answer_away"
        if dropped:
            reply.debug = {**debug, "tip_dropped": dropped}
        return tip

    @property
    def hindi_help_label(self) -> str:
        return hindi_help_label(self.cfg, self.session.stage)

    def finish(self, now=None) -> list[str]:
        """Save spaced review and the next Hindi help level; return a short report.
        The summary is kept on ``self.summary`` for the caller's progress record."""
        summary = self.summary = self.session.summary()
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
            lines.append(f"Next lesson: {word} Hindi help ({hindi_help_label(self.cfg, summary.stage_after)}).")
        return lines


def hindi_help_label(cfg: TutorConfig, level: int) -> str:
    """the tutor's support stage, named for people: "Hindi help: English only"."""
    return f"Hindi help: {cfg.stages[level].name.replace('_', ' ')}"


class TutorFactory:
    """Builds the tutor config and the model once; makes a TutorBrain per lesson."""

    def __init__(self, cfg: dict, llm_cfg: dict, *, offline: bool, env=None) -> None:
        self.cfg = load_voice_tutor_config(cfg.get("overrides", {}))
        self.voice_gender = cfg.get("voice_gender", "male")
        self.tips = cfg.get("tips", True)
        self.model = None if offline else VoiceGuard(build_json_chain(
            self.cfg, llm_cfg.get("providers", []), env,
            cooldown_seconds=llm_cfg.get("cooldown_seconds", 60),
            hedge_after_seconds=cfg.get("hedge_after_seconds")), tips=self.tips)
        self.name = f"tutor[{getattr(self.model, 'name', 'offline')}]"

    def lesson(self, scenario, course, *, learner_id: str, state_dir: Path,
               hindi_help: Optional[int] = None, focus: str = "") -> TutorBrain:
        """``focus``: extra instructions for this learner and lesson (a redo's past
        mistakes), appended to the tutor's system prompt for this lesson only."""
        if self.voice_gender == "male":
            scenario = male_partner(scenario)
        cfg = self.cfg
        if focus:
            cfg = dataclasses.replace(cfg, system_prompt=cfg.system_prompt.rstrip() + "\n\n" + focus + "\n")
        return TutorBrain(cfg, self.model, scenario=scenario, course=course,
                          learner_id=learner_id, state_dir=state_dir, hindi_help=hindi_help)
