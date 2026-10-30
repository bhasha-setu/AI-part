"""What the learner has been shown, and whether a text reveals the target form early.

The target form is the item's model answer without its slots, normalised with the
item's language config: "I'm from {place}." becomes the token run "i am from". A text
reveals it when it contains every such run and the prompts already on screen at this
hint level do not. So:

    level 0-1   prompt and cue only     a recast "I'm from Delhi." reveals
    level 2     first words shown       "I'm from…" is on screen, so it no longer does
    level 3     model sentence shown    nothing can reveal more

A prompt-style correction ("You need one more word after 'name'.") never reveals.
Limitation: only the model answer counts as the target. An accepted alternative spelled
out in full is not caught. The model dialogue and useful-language stages also show
target forms before practice (reasoning_core.md, C1); the ladder rule ignores them.
"""

from __future__ import annotations

import re

from guided_practice.normalise import normalise
from guided_practice.session import MAX_LEVEL, render_prompt

_SLOT_RE = re.compile(r"\{\w+\}")


def target_segments(item) -> list:
    parts = _SLOT_RE.split(item.model_answer)
    segments = [tuple(normalise(p, item.lang).split()) for p in parts]
    return [s for s in segments if s]


def _has_run(tokens: list, run: tuple) -> bool:
    n = len(run)
    return any(tuple(tokens[i:i + n]) == run for i in range(len(tokens) - n + 1))


def contains_target(text: str, item) -> bool:
    segments = target_segments(item)
    if not text or not segments:
        return False
    tokens = normalise(text, item.lang).split()
    return all(_has_run(tokens, seg) for seg in segments)


def shown_prompts(item, level: int, ui) -> list:
    """The prompts on screen for this item at ``level`` (the ladder only climbs)."""
    return [render_prompt(item, lvl, ui) for lvl in range(min(level, MAX_LEVEL) + 1)]


def reveals_target(text: str, item, level: int, ui, already_shown=()) -> bool:
    if not text or level >= MAX_LEVEL:
        return False
    if not contains_target(text, item):
        return False
    disclosed = " ".join(shown_prompts(item, level, ui) + list(already_shown))
    return not contains_target(disclosed, item)


def count_sentences(text: str, terminators: str) -> int:
    if not text.strip():
        return 0
    pieces = re.split("[" + re.escape(terminators) + "]+", text)
    return sum(1 for p in pieces if p.strip())


def count_markers(text: str, markers) -> int:
    folded = (text or "").casefold()
    return sum(folded.count(m) for m in markers)
