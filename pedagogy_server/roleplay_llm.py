"""LLM-backed roleplay partner. The LLM proposes, the server validates, a script answers.

Roles (rule, from the model dialogue): the partner is the speaker of the FIRST line
(or ``partner_speaker`` in config/roleplay.yaml); the learner plays the first other
speaker. English course: partner "Asha"/"Waiter", learner "Ravi". Hindi course: A / B.

The partner never corrects. Each model output must be ``{"partner_line", "done"}`` and
pass the checks in ``validate_line``; otherwise the next scripted partner line from the
model dialogue is used (the learner never sees a rejected line). The conversation ends
after ``max_learner_turns``, or earlier when the model sets done AND at least
``min_learner_turns`` happened. Every conversation therefore always ends.

Script check (simple, documented): a line is rejected if it contains a letter whose
unicode script (first word of ``unicodedata.name``) is the learner language's script
and that script differs from the target's. Skipped when the profile flag
``replies_may_use_learner_language`` is true. Scripts come from ``language_scripts``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from guided_practice.config import CONFIG_DIR, ConfigError, audit_thresholds, unwrap
from guided_practice.roleplay import RoleplayResult, StubRoleplay

from .config import _read, load_reply_strings, replies_may_use_learner_language
from .disclosure import contains_target, count_markers, count_sentences
from .llm import GeminiModel, LLMUnavailable

RAW_LOG_CHARS = 2000
LEARNER_TEXT_CHARS = 400   # cap on learner text sent to the model (it is untrusted data)

SCHEMA = {
    "type": "object",
    "properties": {"partner_line": {"type": "string"}, "done": {"type": "boolean"}},
    "required": ["partner_line", "done"],
}


@dataclass(frozen=True)
class RoleplayConfig:
    status: str
    gemini_model: str
    gemini_thinking_level: str
    timeout_seconds: float
    max_tokens: int
    max_learner_turns: int
    min_learner_turns: int
    reply_max_sentences: int
    reply_max_chars: int
    partner_speaker: object
    language_scripts: dict
    fallback_closing: dict
    system_prompt: str


@lru_cache(maxsize=None)
def load_roleplay_config(path=None) -> RoleplayConfig:
    path = Path(path) if path else CONFIG_DIR / "roleplay.yaml"
    raw = _read(path)
    problems = audit_thresholds(raw)
    if problems:
        raise ConfigError(f"{path.name}: " + "; ".join(problems))
    v = unwrap(raw)
    cfg = RoleplayConfig(
        status=v["status"], gemini_model=v["gemini_model"],
        gemini_thinking_level=v.get("gemini_thinking_level") or "",
        timeout_seconds=v["timeout_seconds"], max_tokens=v["max_tokens"],
        max_learner_turns=v["max_learner_turns"], min_learner_turns=v["min_learner_turns"],
        reply_max_sentences=v["reply_max_sentences"], reply_max_chars=v["reply_max_chars"],
        partner_speaker=v.get("partner_speaker"),
        language_scripts=dict(v["language_scripts"]),
        fallback_closing=dict(v["fallback_closing"]),
        system_prompt=v["system_prompt"],
    )
    if cfg.min_learner_turns > cfg.max_learner_turns or cfg.min_learner_turns < 1:
        raise ConfigError(f"{path.name}: need 1 <= min_learner_turns <= max_learner_turns")
    return cfg


def detect_roles(scenario, partner_speaker=None) -> tuple:
    """(partner, learner) speaker names.

    The learner is the speaker who says the items' model answers in the dialogue (the
    generator requires this, and authored content follows it). With no such speaker,
    the partner is the first line's speaker. ``partner_speaker`` overrides both."""
    speakers = list(dict.fromkeys(t.speaker for t in scenario.model_dialogue))
    if partner_speaker:
        return partner_speaker, next((s for s in speakers if s != partner_speaker), "Learner")
    says_answers = {s: 0 for s in speakers}
    for turn in scenario.model_dialogue:
        says_answers[turn.speaker] += sum(contains_target(turn.text, it) for it in scenario.items)
    if speakers and max(says_answers.values()) > 0:
        learner = max(speakers, key=lambda s: says_answers[s])
        return next((s for s in speakers if s != learner), "Partner"), learner
    partner = speakers[0] if speakers else "Partner"
    return partner, next((s for s in speakers if s != partner), "Learner")


def _script_of(ch: str) -> str:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return ""


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def validate_line(raw: str, *, cfg: RoleplayConfig, markers, terminators: str,
                  forbidden_script: str, goals: list) -> tuple:
    """Return (partner_line, done, rejections). ``partner_line`` is None if rejected."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None, False, ["not_json"]
    if not isinstance(data, dict) or set(data) != {"partner_line", "done"}:
        return None, False, ["bad_shape"]
    line, done = data["partner_line"], data["done"]
    if not isinstance(line, str) or not isinstance(done, bool):
        return None, False, ["bad_types"]
    line = line.strip()
    if not line:
        return None, False, ["empty"]
    out = []
    if len(line) > cfg.reply_max_chars:
        out.append("too_long")
    if count_sentences(line, terminators) > cfg.reply_max_sentences:
        out.append("too_many_sentences")
    if count_markers(line, markers):
        out.append("correction_marker")
    if forbidden_script and any(ch.isalpha() and _script_of(ch) == forbidden_script
                                for ch in line):
        out.append("wrong_script")
    plain = _plain(line)
    if any(g and g in plain for g in goals):
        out.append("reveals_goal")
    return (None if out else line), done, out


class LLMRoleplaySession:
    def __init__(self, stage, scenario, learner_id, history) -> None:
        self._stage = stage
        self.scenario = scenario
        self.learner_id = learner_id
        cfg = stage.cfg
        self.partner, self.learner = detect_roles(scenario, cfg.partner_speaker)
        self.transcript: list = []
        self.log: list = []
        self.learner_turns = 0
        self.done = False
        self.prompt = ""
        self.goals = [_plain(it.model_answer) for it in scenario.items
                      if "{" not in it.model_answer]
        self._script = [t.text for t in scenario.model_dialogue if t.speaker == self.partner]
        self._next_script = 0
        self._fallbacks = 0
        self._turn(final=False)

    # -- public ------------------------------------------------------------
    def submit(self, text: str) -> str:
        if self.done:
            raise RuntimeError("roleplay is already done")
        self.transcript.append({"speaker": self.learner, "text": text, "audio": None})
        self.learner_turns += 1
        self._turn(final=self.learner_turns >= self._stage.cfg.max_learner_turns)
        return self.prompt

    def result(self) -> RoleplayResult:
        reason = f"{self.learner_turns} learner turns; {self._fallbacks} fallback lines"
        return RoleplayResult(status="completed", transcript=list(self.transcript),
                              reason=reason)

    # -- internals ---------------------------------------------------------
    def _turn(self, final: bool) -> None:
        cfg = self._stage.cfg
        line, model_done, entry = None, False, {"turn": self.learner_turns, "rejections": [],
                                                "raw": ""}
        if self._stage.model is None:
            entry["rejections"].append("no_model")
        else:
            try:
                raw = self._stage.model.generate_json(
                    system=self._stage.system, user=self._request(final), schema=SCHEMA)
            except LLMUnavailable as exc:
                entry["rejections"].append(f"llm_unavailable:{exc}")
            else:
                entry["raw"] = str(raw)[:RAW_LOG_CHARS]
                line, model_done, rej = validate_line(
                    raw, cfg=cfg, markers=self._stage.markers,
                    terminators=self._stage.terminators,
                    forbidden_script=self._stage.forbidden_script, goals=self.goals)
                entry["rejections"] += rej
        if line is not None:
            entry["source"] = "llm"
            ending = final or (model_done and self.learner_turns >= cfg.min_learner_turns)
            if not self.transcript:      # an LLM opening replaces the dialogue's first line
                self._next_script = max(self._next_script, 1)
        else:
            entry["source"] = "fallback"
            self._fallbacks += 1
            line, ending = self._scripted(final)
        self.done = bool(ending)
        self.prompt = line
        self.transcript.append({"speaker": self.partner, "text": line, "audio": None})
        self.log.append(entry)

    def _scripted(self, final: bool) -> tuple:
        if not final and self._next_script < len(self._script):
            line = self._script[self._next_script]
            self._next_script += 1
            return line, False
        closing = self._stage.cfg.fallback_closing.get(self._stage.target)
        return closing or (self._script[-1] if self._script else "Goodbye."), True

    def _request(self, final: bool) -> str:
        s = self.scenario
        cfg = self._stage.cfg
        return json.dumps({
            "scenario": {"title": s.title, "can_do": s.can_do},
            "partner_role": self.partner,
            "learner_role": self.learner,
            "reference_dialogue": [{"speaker": t.speaker, "text": t.text}
                                   for t in s.model_dialogue],
            "useful_phrases": [p.phrase for p in s.useful_language],
            "goals": [it.model_answer for it in s.items],
            "conversation_so_far": [
                {"speaker": t["speaker"],
                 "text": t["text"][:LEARNER_TEXT_CHARS]} for t in self.transcript],
            "learner_turns_so_far": self.learner_turns,
            "max_learner_turns": cfg.max_learner_turns,
            "final": final,
            "opening": not self.transcript,
        }, ensure_ascii=False)


class LLMRoleplay:
    """RoleplayStage backed by a JSONModel. ``model=None`` is the scripted-only partner."""

    def __init__(self, course, model, cfg: RoleplayConfig | None = None) -> None:
        self.course = course
        self.model = model
        self.cfg = cfg or load_roleplay_config()
        profile = course.profile
        self.target = profile.target_language
        strings = load_reply_strings(profile.ui_strings)
        self.markers = strings.correction_markers
        self.terminators = strings.sentence_terminators
        target_script = self.cfg.language_scripts.get(self.target, "")
        learner_script = self.cfg.language_scripts.get(profile.learner_language, "")
        allowed = replies_may_use_learner_language(profile)
        self.forbidden_script = ("" if allowed or not learner_script
                                 or learner_script == target_script else learner_script)
        self.system = self.cfg.system_prompt.format(
            cefr=profile.cefr_level, target=self.target,
            max_sentences=self.cfg.reply_max_sentences, max_chars=self.cfg.reply_max_chars)

    def start(self, scenario, learner_id: str, history: list) -> LLMRoleplaySession:
        return LLMRoleplaySession(self, scenario, learner_id, history)


def make_roleplay(kind: str, course):
    if kind == "stub":
        return StubRoleplay()
    cfg = load_roleplay_config()
    if kind == "scripted":
        return LLMRoleplay(course, None, cfg)
    if kind == "gemini":
        model = GeminiModel(cfg.gemini_model, thinking_level=cfg.gemini_thinking_level or None,
                            timeout_seconds=cfg.timeout_seconds,
                            max_output_tokens=cfg.max_tokens)
        return LLMRoleplay(course, model, cfg)
    raise ValueError(f"unknown roleplay kind: {kind!r} (stub | scripted | gemini)")
