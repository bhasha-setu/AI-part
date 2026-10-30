"""One tutor lesson: a free conversation with a light structure behind it.

The server decides the structure:
    practice   each target phrase in turn, hint level 0 -> 3, one phrase at a time
    converse   a short roleplay; mistakes are noted silently, never corrected
    review     at most ``review_max_errors`` mistakes: the learner tries to fix each first
    done       summary, spaced review update, support stage for the next lesson

The LLM decides the words: its verdict on each answer and what the tutor says. Every
request spells out what the tutor may do next ("if correct, ask X; if not, help at hint
level N"), so the reply comes out right instead of being thrown away afterwards. The
server still checks it: an accepted answer is never marked wrong, and an answer
revealed before its hint level is retried once, then masked ("My name ___ Ravi").

With no model (offline) or when the model fails, the rule verdict and fixed lines in
config/tutor.yaml take over.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from guided_practice.content import Item, Scenario

from pedagogy_server.llm import JSONModel, LLMUnavailable

from .config import TutorConfig, next_stage
from .rules import detect_language, mask, reveal_spans, rule_verdict

PRACTICE, CONVERSE, REVIEW, DONE = "practice", "converse", "review", "done"
NOT_CORRECT = ("form_error", "hindi", "unclear", "off_topic")

_SEGMENTS = {
    "type": "array",
    "items": {"type": "object",
              "properties": {"lang": {"type": "string", "enum": ["en", "hi"]},
                             "text": {"type": "string"}},
              "required": ["lang", "text"]},
}
_ERROR = {"type": "object",
          "properties": {"said": {"type": "string"}, "fixed": {"type": "string"},
                         "type": {"type": "string"}, "why": {"type": "string"}}}
_LANG = {"type": "string", "enum": ["en", "hi", "mixed"]}

INTRO_SCHEMA = {"type": "object", "properties": {"say": _SEGMENTS}, "required": ["say"]}
PRACTICE_SCHEMA = {
    "type": "object",
    "properties": {
        "learner_language": _LANG,
        "verdict": {"type": "string",
                    "enum": ["correct", "form_error", "hindi", "unclear", "off_topic", "question"]},
        "error": _ERROR,
        "say": _SEGMENTS,
    },
    "required": ["learner_language", "verdict", "say"],
}
CONVERSE_SCHEMA = {
    "type": "object",
    "properties": {
        "learner_language": _LANG,
        "say": _SEGMENTS,
        "errors": {"type": "array", "items": {
            "type": "object",
            "properties": {**_ERROR["properties"], "importance": {"type": "integer"}},
            "required": ["said", "fixed", "type", "importance"]}},
        "used_phrases": {"type": "array", "items": {"type": "string"}},
        "english_version": {"type": "string"},
        "wants_to_end": {"type": "boolean"},
    },
    "required": ["learner_language", "say", "errors", "wants_to_end"],
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"fixed_it": {"type": "boolean"}, "say": _SEGMENTS},
    "required": ["fixed_it", "say"],
}

# An opening quote after a space or at the start: how a reply quotes a corrected form.
_QUOTED_RE = re.compile(r"(^|\s)['\"\u201c\u2018][^'\"\u201d\u2019]+['\"\u201d\u2019]")

# Mistakes that respond well to feedback come first in the review (framework §6).
HIGH_YIELD = ("copula", "word_order", "tense", "verb")

LEVEL_RULES = {
    0: "Do NOT say the correct sentence or the missing/wrong word. Point at the problem "
       "(for example 'one small word is missing after name') and ask them to try again.",
    1: "Give this clue in natural words: \"{cue}\". Point at the problem. Do NOT say the "
       "correct sentence.",
    2: "Give the first words, \"{first_words}\", and let them finish. Do NOT say the "
       "whole sentence.",
    3: "Say the full correct sentence, \"{model}\", and ask them to say it.",
}


@dataclass
class Segment:
    lang: str
    text: str


@dataclass
class TutorReply:
    segments: list
    phase: str
    done: bool = False
    debug: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


@dataclass
class ItemRecord:
    item_id: str
    hint_level: int = 0
    answers: list = field(default_factory=list)
    used_hindi: bool = False
    solved_level: Optional[int] = None
    achieved: bool = False
    error: Optional[dict] = None     # the first mistake, for the review


@dataclass
class Summary:
    items: list
    from_memory: float
    hindi_turns: int
    reviewed: list
    stage_before: int
    stage_after: int


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TutorSession:
    def __init__(self, scenario: Scenario, cfg: TutorConfig, model: Optional[JSONModel],
                 stage: int, log: Optional[Callable[[dict], None]] = None) -> None:
        self.scenario = scenario
        self.cfg = cfg
        self.model = model
        self.stage = stage
        self.log = log or (lambda event: None)
        self.phase = PRACTICE
        self.index = 0
        self.records = [ItemRecord(it.id) for it in scenario.items]
        self.hindi_turns = 0
        self.history: list = []          # the conversation: [{"who", "text"}]
        self.converse_errors: list = []
        self.used_phrases: set = set()
        self.review_queue: list = []
        self.reviewed: list = []
        speakers = [t.speaker for t in scenario.model_dialogue]
        self.partner = speakers[0] if speakers else "your friend"
        self._scripted = [t.text for t in scenario.model_dialogue if t.speaker == self.partner]

    # --- public ----------------------------------------------------------------------

    @property
    def item(self) -> Item:
        return self.scenario.items[self.index]

    @property
    def record(self) -> ItemRecord:
        return self.records[self.index]

    def start(self) -> TutorReply:
        phrases = [p.phrase for p in self.scenario.useful_language]
        request = {
            "phase": "intro",
            "support_stage": self._stage_rule(),
            "lesson": {"title": self.scenario.title, "can_do": self.scenario.can_do},
            "phrases": [{"phrase": p.phrase, "meaning": p.meaning}
                        for p in self.scenario.useful_language],
            "what_to_do": (
                "Greet the learner in one short line and say what they will be able to do. "
                "List the phrases exactly as given (one segment, English), with their meaning "
                "as the support stage allows. Then ask the first practice question, based on "
                f"this situation: \"{self.item.prompt}\". Do not give its answer."),
        }
        fallback = [Segment("en", self.cfg.en("intro_en", can_do=self.scenario.can_do,
                                               phrases="; ".join(phrases)))]
        if self._hindi_ok():
            fallback.append(Segment("hi", self.cfg.hi("intro_hi")))
        fallback.append(Segment("en", self.item.prompt))
        say, debug = self._ask(request, INTRO_SCHEMA, lambda out: (out["say"], []), fallback)
        return TutorReply(say, self.phase, debug=debug)

    def turn(self, text: str) -> TutorReply:
        text = (text or "").strip()[: self.cfg.learner_text_max_chars]
        if self.phase == PRACTICE:
            return self._practice(text)
        if self.phase == CONVERSE:
            return self._converse(text)
        if self.phase == REVIEW:
            return self._review(text)
        return TutorReply([Segment("en", self.cfg.en("done_en"))], DONE, done=True)

    def summary(self) -> Summary:
        n = len(self.records) or 1
        memory = sum(1 for r in self.records
                     if r.achieved and r.solved_level is not None and r.solved_level <= 1
                     and not r.used_hindi) / n
        return Summary(items=[asdict(r) for r in self.records], from_memory=memory,
                       hindi_turns=self.hindi_turns, reviewed=self.reviewed,
                       stage_before=self.stage,
                       stage_after=next_stage(self.cfg, self.stage, memory, self.hindi_turns))

    # --- practice --------------------------------------------------------------------

    def _practice(self, text: str) -> TutorReply:
        item, rec = self.item, self.record
        level = rec.hint_level
        rule = rule_verdict(text, item)
        local_lang = detect_language(text, item)
        give_up = level >= 3
        mistake_level = min(level + 1, 3)
        hindi_level = (max(mistake_level, self.cfg.hindi_bridge_level)
                       if self.cfg.bridges_from_hindi(self.stage) else mistake_level)
        nxt = self._next_step()

        if give_up:
            if_not = (f"Say the correct sentence, \"{self._model(rule)}\", say we will "
                      f"practise it again later, then {nxt}")
            if_hindi = if_not
        else:
            if_not = "Help at hint level %d: %s" % (mistake_level, self._level_rule(mistake_level, rule))
            if self.cfg.bridges_from_hindi(self.stage):
                if_hindi = ("Say their meaning is right and help them say it in English. "
                            "Hint level %d: %s" % (hindi_level, self._level_rule(hindi_level, rule)))
            else:
                if_hindi = ("Answer in English only: ask them to try to say it in English. "
                            "Hint level %d: %s" % (hindi_level, self._level_rule(hindi_level, rule)))
        request = {
            "phase": "practice",
            "support_stage": self._stage_rule(),
            "situation": item.prompt,
            "target_sentence": item.model_answer,
            "slots_note": "{name}, {place} etc. stand for the learner's own information.",
            "accepted_forms": item.accepted,
            "known_mistakes": [{"type": ce.id, "example": ce.example,
                                "fixed": ce.correction_example, "why": " ".join(ce.why.split())}
                               for ce in item.common_errors],
            "earlier_answers_to_this_question": rec.answers,
            "learner_answer": text,
            "already_verified_correct": rule["verdict"] == "correct",
            "what_to_do": {
                "if correct": f"React to what they said in a few words, then {nxt}",
                "if form_error, unclear or off_topic": if_not,
                "if hindi (the answer is Hindi or mixed)": if_hindi,
                "if question (they asked you something about English or the task)":
                    "Answer it in one short sentence, then ask them to try the same "
                    "question again. Do not give its answer.",
            },
            "error_field": "For form_error, fill error with what they said, the fixed "
                           "sentence, a short type (e.g. copula, article, preposition, "
                           "word_order, tense) and a one-sentence why.",
        }
        allowed = {"form_error": mistake_level, "unclear": mistake_level,
                   "off_topic": mistake_level, "hindi": hindi_level,
                   "question": level, "correct": 3}
        if give_up:
            allowed = {k: 3 for k in allowed}

        def accept(out):
            verdict = out.get("verdict")
            problems = []
            if verdict not in allowed:
                problems.append("verdict must be one of " + ", ".join(allowed))
            elif rule["verdict"] == "correct" and verdict != "correct":
                problems.append("the answer is an accepted form; it is correct")
            elif local_lang == "hi" and verdict not in ("hindi", "question"):
                problems.append("the answer is in Hindi")
            elif verdict != "correct" and allowed[verdict] < 3:
                text_out = " ".join(s["text"] for s in out["say"])
                if reveal_spans(text_out, item, text):
                    problems.append(f"you said the answer too early; at hint level "
                                    f"{allowed[verdict]} the learner must find it")
            return out["say"], problems

        fallback_out = self._practice_fallback(rule, local_lang, give_up, mistake_level,
                                               hindi_level, nxt)
        say, debug = self._ask(request, PRACTICE_SCHEMA, accept, fallback_out["say"],
                               raw_holder=True)
        out = debug.pop("out", None) or fallback_out
        if any("too early" not in p for p in debug.get("unresolved", [])):
            # the model kept breaking a hard rule: use the rule verdict and fixed lines
            say, out, debug["source"] = fallback_out["say"], fallback_out, "fallback"
        verdict = "correct" if rule["verdict"] == "correct" else out["verdict"]
        if local_lang == "hi" and verdict != "question":
            verdict = "hindi"
        if verdict != "correct" and allowed.get(verdict, 3) < 3 and debug.get("source") == "llm":
            say = self._mask(say, text)
        next_level = allowed.get(verdict, level)
        say = self._language_policy(say, verdict in ("hindi", "question") or local_lang != "en"
                                    or (verdict != "correct" and next_level >= 2))
        debug.update(verdict=verdict, hint_level=level)

        rec.answers.append(text)
        if verdict == "correct":
            rec.solved_level, rec.achieved = level, True
            self._advance()
        elif verdict == "question":
            pass
        else:
            if verdict == "hindi":
                rec.used_hindi = True
                self.hindi_turns += 1
            elif rec.error is None and out.get("error") and out["error"].get("fixed"):
                rec.error = dict(out["error"])
            if give_up:
                self._advance()
            else:
                rec.hint_level = hindi_level if verdict == "hindi" else mistake_level
        debug["next_hint_level"] = None if self.phase != PRACTICE else self.record.hint_level
        return TutorReply(say, self.phase, debug=debug)

    def _advance(self) -> None:
        if self.index + 1 < len(self.records):
            self.index += 1
            return
        self.phase = CONVERSE
        opener = self._next_step_opener()
        self.history.append({"who": self.partner, "text": opener})

    def _next_step(self) -> str:
        if self.index + 1 < len(self.records):
            return f"ask the next practice question, based on: \"{self.scenario.items[self.index + 1].prompt}\"."
        return (f"say that now you will have a short real conversation and you play "
                f"{self.partner}. Then, as {self.partner}, say: \"{self._next_step_opener()}\"")

    def _next_step_opener(self) -> str:
        return self._scripted[0] if self._scripted else "Hello! How are you?"

    def _level_rule(self, level: int, rule: dict) -> str:
        return LEVEL_RULES[level].format(cue=self.item.cue, first_words=self.item.first_words,
                                         model=self._model(rule))

    def _model(self, rule: dict) -> str:
        model = self.item.model_answer
        for k, v in (rule.get("slots") or {}).items():
            model = model.replace("{%s}" % k, v)
        return model

    def _practice_fallback(self, rule, local_lang, give_up, mistake_level, hindi_level, nxt) -> dict:
        verdict = rule["verdict"]
        say = []
        if verdict == "correct":
            say.append(Segment("en", self.cfg.en("correct_en")))
            say.append(Segment("en", self._next_line_fallback()))
        elif give_up:
            say.append(Segment("en", self.cfg.en("give_up_en", model=self._model(rule))))
            say.append(Segment("en", self._next_line_fallback()))
        else:
            level = hindi_level if verdict == "hindi" else mistake_level
            if verdict == "hindi" and self.cfg.bridges_from_hindi(self.stage):
                say.append(Segment("hi", self.cfg.hi("bridge_hi")))
            elif verdict == "hindi":
                say.append(Segment("en", self.cfg.en("try_english_en")))
            elif self._hindi_ok() and level >= 2:
                say.append(Segment("hi", self.cfg.hi("hint_hi")))
            hint = self.cfg.lines["hint_en"][level].format(
                prompt=self.item.prompt, cue=self.item.cue, first_words=self.item.first_words,
                model=self._model(rule))
            say.append(Segment("en", hint))
        return {"verdict": verdict, "error": rule["error"], "say": say}

    def _next_line_fallback(self) -> str:
        if self.index + 1 < len(self.records):
            return self.scenario.items[self.index + 1].prompt
        return (self.cfg.en("converse_start_en", partner=self.partner) + " "
                + self._next_step_opener())

    # --- conversation ----------------------------------------------------------------

    def _converse(self, text: str) -> TutorReply:
        self.history.append({"who": "learner", "text": text})
        local_lang = detect_language(text)
        if local_lang != "en":
            self.hindi_turns += 1
        turns = sum(1 for h in self.history if h["who"] == "learner")
        must_end = turns >= self.cfg.converse_max_turns
        remaining = [it.model_answer for it in self.scenario.items
                     if it.id not in self.used_phrases]
        bridge = ("If they speak Hindi, say the English for what they meant in a few words and "
                  "carry on with the conversation." if self.cfg.bridges_from_hindi(self.stage)
                  else "If they speak Hindi, ask them, in English, to try in English.")
        request = {
            "phase": "conversation",
            "support_stage": self._stage_rule(),
            "you_are": f"{self.partner}, in this situation: {self.scenario.title}. "
                       f"The learner's goal: {self.scenario.can_do}",
            "example_dialogue": [{"who": t.speaker, "text": t.text}
                                 for t in self.scenario.model_dialogue],
            "conversation_so_far": self.history,
            "phrases_not_used_yet": remaining,
            "learner_turn": turns,
            "what_to_do": (
                ("Close the conversation warmly in one short line. Ask nothing."
                 if must_end else
                 "Stay in role. Reply to what they actually said, in one or two short "
                 "sentences, then ask one question that gives them a chance to use a phrase "
                 "not used yet. Never ask what they already told you. ")
                + " Do NOT correct mistakes now: list each real mistake in errors (said, "
                "fixed, type, why, importance 1-3, 3 = most important) for the review "
                "later. " + bridge +
                " A Hindi sentence is not a mistake: never list it in errors. If they spoke "
                "Hindi or mixed, put what they meant in simple correct English in "
                "english_version (the app shows it); otherwise leave it empty."
                " used_phrases: the target phrases they said correctly this turn, as given. "
                "Set wants_to_end when the conversation has naturally finished."),
        }

        def accept(out):
            problems = []
            reply = " ".join(s["text"] for s in out["say"])
            fixes = [e.get("fixed", "") for e in out.get("errors", [])]
            if local_lang == "en" and (_QUOTED_RE.search(reply) or any(
                    f and f.casefold().rstrip(".!?") in reply.casefold() for f in fixes)):
                problems.append("do not correct or quote their English during the conversation; "
                                "only reply in role and list the mistake in errors")
            return out["say"], problems

        fallback = [Segment("en", self.cfg.en("converse_end_en") if must_end
                            else self._scripted_line(turns))]
        say, debug = self._ask(request, CONVERSE_SCHEMA, accept, fallback, raw_holder=True)
        out = debug.pop("out", None) or {}
        spoke_hindi = local_lang != "en" or out.get("learner_language") in ("hi", "mixed")
        if any("correct or quote" in p for p in debug.get("unresolved", [])):
            say = fallback
        say = self._language_policy(say, spoke_hindi)
        english = (out.get("english_version") or "").strip()
        if spoke_hindi and english and self.cfg.bridges_from_hindi(self.stage):
            say = [Segment("en", f"In English: \"{english}\"")] + list(say)
        self.converse_errors.extend(
            e for e in out.get("errors", [])
            if e.get("fixed") and detect_language(e.get("said", "")) == "en")
        by_answer = {it.model_answer: it.id for it in self.scenario.items}
        self.used_phrases.update(by_answer[p] for p in out.get("used_phrases", []) if p in by_answer)
        ended = must_end or (debug.get("source") == "llm" and out.get("wants_to_end")
                             and turns >= self.cfg.converse_min_turns) \
            or (debug.get("source") != "llm" and turns >= len(self._scripted))
        self.history.append({"who": self.partner, "text": " ".join(s.text for s in say)})
        if not ended:
            return TutorReply(say, self.phase, debug=debug)
        say = list(say) + self._start_review()
        return TutorReply(say, self.phase, done=self.phase == DONE, debug=debug)

    def _scripted_line(self, turns: int) -> str:
        if turns < len(self._scripted):
            return self._scripted[turns]
        return self.cfg.en("converse_end_en")

    # --- review ----------------------------------------------------------------------

    def _start_review(self) -> list:
        self.review_queue = self._pick_errors()
        if not self.review_queue:
            self.phase = DONE
            return [Segment("en", self.cfg.en("review_none_en")),
                    Segment("en", self.cfg.en("done_en"))]
        self.phase = REVIEW
        return [Segment("en", self.cfg.en("review_start_en", n=len(self.review_queue)))] \
            + self._review_prompt()

    def _pick_errors(self) -> list:
        def rank(e):
            kind = (e.get("type") or "").lower()
            return (-(e.get("importance") or 2), 0 if any(k in kind for k in HIGH_YIELD) else 1)

        pool = sorted(self.converse_errors, key=rank)
        pool += [r.error for r in self.records
                 if r.error and (not r.achieved or (r.solved_level or 0) >= 2)]
        picked, kinds = [], set()
        for e in pool:
            kind = (e.get("type") or e["fixed"]).lower()
            if kind in kinds:
                continue
            kinds.add(kind)
            picked.append(e)
            if len(picked) >= self.cfg.review_max_errors:
                break
        return picked

    def _review_prompt(self) -> list:
        err = self.review_queue[0]
        say = [Segment("en", self.cfg.en("review_prompt_en", said=err["said"]))]
        if self._hindi_ok():
            say.append(Segment("hi", self.cfg.hi("review_prompt_hi")))
        return say

    def _review(self, text: str) -> TutorReply:
        err = self.review_queue.pop(0)
        request = {
            "phase": "review",
            "support_stage": self._stage_rule(),
            "mistake": err,
            "learner_fix": text,
            "what_to_do": (
                "Decide if their fix is correct English with the same meaning (fixed_it). "
                "If yes, praise in a few words. If not, give the correct sentence. Either "
                "way add one plain sentence on why, comparing with Hindi if it helps."),
        }
        fixed_en = self.cfg.en("review_answer_en", fixed=err["fixed"])
        fallback = [Segment("en", fixed_en)]
        if err.get("why"):
            fallback.append(Segment("en", err["why"]))
        say, debug = self._ask(request, REVIEW_SCHEMA, lambda out: (out["say"], []), fallback,
                               raw_holder=True)
        out = debug.pop("out", None) or {}
        say = self._language_policy(say, True)
        shown = " ".join(x.text for x in say).casefold()
        if not out.get("fixed_it") and err["fixed"].casefold().rstrip(".!?") not in shown:
            # one try per mistake: the learner always leaves with the correct form
            say = [Segment("en", fixed_en)] + list(say)
        self.reviewed.append({**err, "learner_fix": text, "fixed_it": bool(out.get("fixed_it"))})
        say = list(say)
        if self.review_queue:
            say += self._review_prompt()
        else:
            self.phase = DONE
            say.append(Segment("en", self.cfg.en("done_en")))
        return TutorReply(say, self.phase, done=self.phase == DONE, debug=debug)

    # --- the model -------------------------------------------------------------------

    def _stage_rule(self) -> dict:
        st = self.cfg.stages[self.stage]
        return {"stage": self.stage, "of": self.cfg.last_stage, "name": st.name, "rule": st.rule}

    def _language_policy(self, say: list, hindi_needed: bool) -> list:
        """Drop Hindi the support stage doesn't allow. Stages 0 and 1 always allow it,
        the middle stage only when it helps (learner spoke Hindi, is stuck, or asked),
        the last stage never."""
        if self.stage < self.cfg.last_stage - 1:
            return say
        if self.stage < self.cfg.last_stage and hindi_needed:
            return say
        kept = [s for s in say if s.lang != "hi"]
        return kept or say

    def _hindi_ok(self) -> bool:
        return self.cfg.bridges_from_hindi(self.stage)

    def _mask(self, say: list, learner_text: str) -> list:
        out = []
        for s in say:
            spans = reveal_spans(s.text, self.item, learner_text) if s.lang == "en" else []
            out.append(Segment(s.lang, mask(s.text, spans, learner_text)) if spans else s)
        return out

    def _ask(self, request: dict, schema: dict, accept, fallback: list,
             raw_holder: bool = False):
        """Call the model, retrying once with the problems found. Returns the segments
        and a debug dict (with the parsed output under "out" when raw_holder is set)."""
        debug = {"phase": request["phase"], "source": "fallback", "problems": []}
        if self.model is None:
            debug["reason"] = "offline"
            self.log({"request": request, **debug})
            return fallback, debug
        notes = []
        last = None
        for attempt in range(self.cfg.retries_per_turn + 1):
            user = json.dumps({**request, **({"fix_these": notes} if notes else {})},
                              ensure_ascii=False)
            try:
                raw = self.model.generate_json(system=self.cfg.system_prompt, user=user,
                                               schema=schema)
                out = json.loads(raw)
                segments = [Segment(s["lang"], s["text"].strip()) for s in out["say"]
                            if s.get("text", "").strip() and s.get("lang") in ("en", "hi")]
                if not segments:
                    raise ValueError("empty say")
            except LLMUnavailable as exc:
                debug["reason"] = str(exc)
                break
            except (ValueError, KeyError, TypeError) as exc:
                debug["problems"].append(f"bad_output:{type(exc).__name__}")
                notes = ["return valid JSON with a non-empty say list"]
                continue
            out["say"] = [asdict(s) for s in segments]
            _, problems = accept(out)
            last = (segments, out)
            if not problems:
                debug.pop("unresolved", None)
                break
            debug["problems"].extend(problems)
            debug["unresolved"] = problems
            notes = problems
        if last is None:
            self.log({"request": request, **debug})
            return fallback, debug
        segments, out = last
        debug["source"] = "llm"
        if raw_holder:
            debug["out"] = out
        self.log({"request": request, "out": out, **{k: v for k, v in debug.items() if k != "out"}})
        return segments, debug
