"""What the learner has been shown, and whether a text reveals the target form early.

The target form is the item's model answer without its slots, normalised with the
item's language config: "I'm from {place}." becomes the token run "i am from". A text
reveals it when it contains every such run and the prompts already on screen at this
hint level do not. So:

    level 0-1   prompt and cue only     a recast "I'm from Delhi." reveals
    level 2     first words shown       "I'm from…" is on screen, so it no longer does
    level 3     model sentence shown    nothing can reveal more

A prompt-style correction ("You need one more word after 'name'.") never reveals.
Accepted alternatives count as targets too when ``accepted_min_tokens`` is set (the
server passes config/judge.yaml's reveal_accepted_min_tokens): a judge once recast "i want
two kilos of tomatoes please" at level 0 for the model answer "I would like two kilos of
tomatoes, please.". Shorter accepted forms ("yes", "i agree") are left out, because they
occur in ordinary replies. The model dialogue and useful-language stages also show
target forms before practice (reasoning_core.md, C1); the ladder rule ignores them.
"""

from __future__ import annotations

import re

from guided_practice.normalise import normalise
from guided_practice.session import MAX_LEVEL, render_prompt

_SLOT_RE = re.compile(r"\{\w+\}")


def _segments(form: str, lang) -> list:
    segments = [tuple(normalise(p, lang).split()) for p in _SLOT_RE.split(form)]
    return [s for s in segments if s]


def target_segments(item) -> list:
    return _segments(item.model_answer, item.lang)


def _targets(item, accepted_min_tokens: int) -> list:
    """Segment lists for the model answer and, if asked, the longer accepted forms."""
    targets = [target_segments(item)]
    if accepted_min_tokens:
        for form in getattr(item, "accepted", ()):
            segs = _segments(form, item.lang)
            if sum(len(s) for s in segs) >= accepted_min_tokens and segs not in targets:
                targets.append(segs)
    return [t for t in targets if t]


def _contains(tokens: list, segments: list) -> bool:
    return all(_has_run(tokens, seg) for seg in segments)


def _has_run(tokens: list, run: tuple) -> bool:
    n = len(run)
    return any(tuple(tokens[i:i + n]) == run for i in range(len(tokens) - n + 1))


def contains_target(text: str, item) -> bool:
    segments = target_segments(item)
    if not text or not segments:
        return False
    tokens = normalise(text, item.lang).split()
    return all(_has_run(tokens, seg) for seg in segments)


def contains_any_target(text: str, item, accepted_min_tokens: int = 0) -> bool:
    """The model answer, or an accepted form of at least ``accepted_min_tokens`` words."""
    if not text:
        return False
    tokens = normalise(text, item.lang).split()
    return any(_contains(tokens, t) for t in _targets(item, accepted_min_tokens))


def shown_prompts(item, level: int, ui) -> list:
    """The prompts on screen for this item at ``level`` (the ladder only climbs)."""
    return [render_prompt(item, lvl, ui) for lvl in range(min(level, MAX_LEVEL) + 1)]


def reveals_target(text: str, item, level: int, ui, already_shown=(),
                   accepted_min_tokens: int = 0) -> bool:
    if not text or level >= MAX_LEVEL:
        return False
    tokens = normalise(text, item.lang).split()
    disclosed = normalise(" ".join(shown_prompts(item, level, ui) + list(already_shown)),
                          item.lang).split()
    return any(_contains(tokens, t) and not _contains(disclosed, t)
               for t in _targets(item, accepted_min_tokens))


def count_sentences(text: str, terminators: str) -> int:
    if not text.strip():
        return 0
    pieces = re.split("[" + re.escape(terminators) + "]+", text)
    return sum(1 for p in pieces if p.strip())


def count_markers(text: str, markers) -> int:
    folded = (text or "").casefold()
    return sum(folded.count(m) for m in markers)
