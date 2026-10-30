"""A fresh version of a lesson for a redo, shaped by the learner's past attempts.

Same lesson, asked another way: the target phrases, their accepted forms, common
errors, cues, first words and ids are never changed (so guided_practice's validation still
holds and spaced review continues on the same items). The LLM rewrites only:

    setting      a new situation for the lesson (one line)
    prompts      a new situation for each phrase
    partner      a new roleplay partner, with a new example dialogue

Personalisation, from the learner's earlier attempts at this lesson:
    * phrases they missed come first;
    * mistakes from their reviews go into the tutor's instructions for this lesson, so
      the roleplay gives them chances to use those forms (still no mid-conversation
      correction: the tutor's rule).

The server checks the LLM's output like pedagogy_server's generator does (ids match, no
prompt gives the answer away, English script, length limits), retries once with the
problems, then falls back to the original lesson, still reordered and personalised.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Optional

from .tutor_brain import LLMUnavailable, use_tutor_packages

use_tutor_packages()

from guided_practice.content import DialogueTurn, Scenario  # noqa: E402
from guided_practice.normalise import normalise  # noqa: E402

SLOT = re.compile(r"\{(\w+)\}")
LATIN_ONLY = re.compile(r"^[\x00-\x7F’‘“”–—…]*$")
PROMPT_MAX, SETTING_MAX, LINE_MAX, DIALOGUE_MAX = 160, 140, 160, 10

SCHEMA = {
    "type": "object",
    "properties": {
        "setting": {"type": "string"},
        "partner": {"type": "string"},
        "prompts": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "prompt": {"type": "string"}},
            "required": ["id", "prompt"]}},
        "dialogue": {"type": "array", "items": {
            "type": "object",
            "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}},
            "required": ["speaker", "text"]}},
    },
    "required": ["setting", "partner", "prompts", "dialogue"],
}

SYSTEM = (
    "You write a fresh version of a short English speaking lesson for a Hindi-speaking "
    "learner who is repeating it. Keep the same target phrases; change only the "
    "situation. Each prompt describes a situation and what to say, in simple English, "
    "and must NOT contain the answer or its first words. The dialogue is between the "
    "partner and the learner's character and must use every target phrase once, in the "
    "new situation; the partner speaks first and is a man (a male name). Use simple English only, no Hindi, no markdown. Return only JSON.")


def past_mistakes(attempts: list) -> tuple[list[str], list[dict]]:
    """Item ids missed in the latest attempt, and the reviewed mistakes from all of them."""
    missed = list(attempts[-1].get("missed", [])) if attempts else []
    seen, reviewed = set(), []
    for att in attempts:
        for r in att.get("reviewed", []):
            key = (r.get("said"), r.get("fixed"))
            if r.get("fixed") and key not in seen:
                seen.add(key)
                reviewed.append(r)
    return missed, reviewed[-5:]


def focus_note(scenario: Scenario, missed: list[str], reviewed: list[dict]) -> str:
    """Extra tutor instructions for this learner's redo."""
    if not missed and not reviewed:
        return ""
    lines = ["This learner is repeating this lesson."]
    targets = [it.model_answer for it in scenario.items if it.id in missed]
    if targets:
        lines.append("Last time they could not yet say: " + "; ".join(targets)
                     + ". Give them extra chances to use these in the conversation.")
    if reviewed:
        lines.append("Mistakes from their earlier reviews: "
                     + "; ".join(f"\"{r['said']}\" -> \"{r['fixed']}\"" for r in reviewed)
                     + ". In the conversation, ask questions that invite these forms, but "
                       "still do not correct mistakes until the review.")
    return "\n".join(lines)


def check_variant(data: dict, scenario: Scenario, lang) -> list[str]:
    problems = []
    ids = [it.id for it in scenario.items]
    got = [p.get("id") for p in data.get("prompts", [])]
    if sorted(got) != sorted(ids):
        problems.append(f"prompts must have exactly these ids: {ids}")
    items = {it.id: it for it in scenario.items}
    for p in data.get("prompts", []):
        item, text = items.get(p.get("id")), (p.get("prompt") or "").strip()
        if item is None:
            continue
        if not text or len(text) > PROMPT_MAX:
            problems.append(f"{item.id}: prompt must be 1-{PROMPT_MAX} characters")
        norm = normalise(text, lang)
        answer = normalise(SLOT.sub(" ", item.model_answer), lang)
        first = normalise(item.first_words, lang)
        if answer and answer in norm or (len(first.split()) >= 2 and first in norm):
            problems.append(f"{item.id}: the prompt gives the answer away; describe the situation instead")
    setting = (data.get("setting") or "").strip()
    if not setting or len(setting) > SETTING_MAX:
        problems.append(f"setting must be 1-{SETTING_MAX} characters")
    dialogue = data.get("dialogue") or []
    if not 2 <= len(dialogue) <= DIALOGUE_MAX:
        problems.append(f"dialogue must have 2-{DIALOGUE_MAX} lines")
    if len({(d.get("speaker") or "").strip() for d in dialogue}) < 2:
        problems.append("dialogue needs two speakers")
    partner = (data.get("partner") or "").strip()
    if dialogue and (dialogue[0].get("speaker") or "").strip() != partner:
        problems.append("the partner must speak the first dialogue line")   # the tutor plays them
    if any(not (d.get("text") or "").strip() or len(d["text"]) > LINE_MAX for d in dialogue):
        problems.append(f"every dialogue line must be 1-{LINE_MAX} characters")
    texts = [setting, data.get("partner") or ""] + [p.get("prompt") or "" for p in data.get("prompts", [])] \
        + [d.get("text") or "" for d in dialogue]
    if any(not LATIN_ONLY.match(t) for t in texts):
        problems.append("use English (Latin script) only")
    return problems


def reorder(scenario: Scenario, missed: list[str]) -> Scenario:
    """Missed items first, others in their usual order."""
    first = [it for it in scenario.items if it.id in missed]
    rest = [it for it in scenario.items if it.id not in missed]
    return replace(scenario, items=first + rest)


def apply_variant(scenario: Scenario, data: dict) -> Scenario:
    prompts = {p["id"]: p["prompt"].strip() for p in data["prompts"]}
    items = [replace(it, prompt=prompts[it.id]) for it in scenario.items]
    dialogue = [DialogueTurn(d["speaker"].strip(), d["text"].strip()) for d in data["dialogue"]]
    return replace(scenario, items=items, model_dialogue=dialogue,
                   can_do=f"{scenario.can_do} Today's situation: {data['setting'].strip()}")


def make_variant(scenario: Scenario, attempts: list, model, lang, *, attempts_allowed: int = 2):
    """Returns (scenario, focus note, meta). ``meta["source"]`` is "llm" for a fresh
    version or "original" when the LLM was unavailable or its versions failed checks."""
    missed, reviewed = past_mistakes(attempts)
    base = reorder(scenario, missed)
    note = focus_note(scenario, missed, reviewed)
    meta = {"source": "original", "missed_first": missed, "reviewed": reviewed, "problems": []}
    if model is None:
        meta["reason"] = "offline"
        return base, note, meta
    request = {
        "lesson": {"title": scenario.title, "can_do": scenario.can_do},
        "earlier_settings": [scenario.can_do],
        "partner_used_before": sorted({t.speaker for t in scenario.model_dialogue}),
        "target_phrases": [{"id": it.id, "say": it.model_answer, "old_prompt": it.prompt,
                            "slots_note": "{name}, {place} etc. are the learner's own details"}
                           for it in base.items],
        "learner_struggled_with": [it.model_answer for it in base.items if it.id in missed],
        "learner_mistakes": reviewed,
        "what_to_do": "Pick a new, realistic setting different from the earlier one. Write a "
                      "new prompt for every target phrase (same ids) and a new short dialogue "
                      "with a new partner name. If they struggled with a phrase, make its "
                      "situation especially clear.",
    }
    notes = []
    for _ in range(attempts_allowed):
        user = json.dumps({**request, **({"fix_these": notes} if notes else {})}, ensure_ascii=False)
        try:
            data = json.loads(model.generate_json(system=SYSTEM, user=user, schema=SCHEMA))
        except LLMUnavailable as exc:
            meta["reason"] = str(exc)
            return base, note, meta
        except ValueError:
            notes = ["return valid JSON"]
            meta["problems"].append("bad_json")
            continue
        notes = check_variant(data, scenario, lang)
        if not notes:
            meta.update(source="llm", setting=data["setting"], partner=data["partner"])
            return apply_variant(base, data), note, meta
        meta["problems"] += notes
    meta["reason"] = "failed_checks"
    return base, note, meta
