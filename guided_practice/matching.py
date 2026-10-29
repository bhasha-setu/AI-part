"""Template compilation, slot matching, feedback filling and typo tolerance.

Templates use ``{slot}`` placeholders, e.g. ``"my name is {name}"``. Every
literal part of a template is pushed through the same normaliser as the
learner's transcript, so an authored phrasing and a spoken one are compared on
equal footing.

Two slot widths:
  * accepted phrasings  -> ``.+?``  (a slot may span several words: "new delhi")
  * common-error patterns -> ``\\S+`` (single token, which keeps error patterns
    narrow so "i am a student" never trips the dropped-article pattern).

Common-error patterns may instead be authored as raw regexes (``regex: true``)
for cases that need alternation, e.g. ``^(?P<subj>he|she) are\\b``.
"""

from __future__ import annotations

import re
from typing import Optional

from .config import LanguageConfig
from .normalise import normalise

_SLOT_RE = re.compile(r"\{(\w+)\}")
_MARKER = "slotmarker%d"


def compile_template(template: str, single_word: bool = False, lang: LanguageConfig | None = None):
    """Compile a ``{slot}`` template into an anchored regex.

    Returns ``(regex, slot_names)``. ``slot_names`` preserves slot order and may
    contain duplicates; group names are ``g0``, ``g1``, ... positionally.
    """
    names: list[str] = []

    def _mark(m: re.Match) -> str:
        names.append(m.group(1))
        return " " + (_MARKER % (len(names) - 1)) + " "

    marked = _SLOT_RE.sub(_mark, template)
    norm = normalise(marked, lang)  # markers are alphanumeric, so they survive
    pattern = re.escape(norm)
    for i in range(len(names)):
        grp = r"(?P<g%d>\S+)" % i if single_word else r"(?P<g%d>.+?)" % i
        pattern = pattern.replace(re.escape(_MARKER % i), grp)
    return re.compile("^" + pattern + "$"), names


def compile_raw(pattern: str):
    """Compile a raw regex pattern (already in normalised space).

    Returns ``(regex, slot_names)`` where slot names are the regex's *named*
    groups, in definition order.
    """
    rx = re.compile(pattern)
    names = sorted(rx.groupindex, key=lambda k: rx.groupindex[k])
    return rx, names


def _present(name: str, value: str, lang: LanguageConfig | None) -> str:
    """Display form of a captured slot. Casing is a language setting."""
    if lang is not None and lang.cased and name in lang.titlecase_slots:
        return value.title()
    return value


def match(rx: re.Pattern, names: list[str], norm: str, raw: bool = False,
          lang: LanguageConfig | None = None) -> Optional[dict]:
    """Return a slot dict if ``norm`` matches, else ``None``.

    An empty dict means "matched, no slots" and is still a match (not None).
    """
    m = rx.match(norm)
    if m is None:
        return None
    out: dict[str, str] = {}
    if raw:
        for name in names:
            val = m.group(name)
            if val is None:
                continue
            out[name] = _present(name, val.strip(), lang)
    else:
        for i, name in enumerate(names):
            out[name] = _present(name, m.group("g%d" % i).strip(), lang)
    return out


def fill(text: str, slots: dict) -> str:
    """Substitute ``{slot}`` placeholders in feedback text from captured slots."""
    for key, val in slots.items():
        text = text.replace("{%s}" % key, str(val))
    return text


# --- bounded typo tolerance -------------------------------------------------

def _within_edit_distance_one(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` differ by at most one insert/delete/substitute."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:  # one substitution
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    # one insert/delete: walk the shorter against the longer
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    edited = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            if edited:
                return False
            edited = True
            j += 1  # skip the extra char in the longer string
    return True


def typo_correct(norm: str, vocab: set[str], lang: LanguageConfig) -> tuple[str, bool]:
    """Best-effort spelling repair toward an item's own vocabulary.

    Each transcript token not already in ``vocab`` is replaced by a vocabulary
    word within edit distance 1, considering only vocabulary words of at least
    ``lang.typo_min_token_length`` characters. The length guard keeps short
    function words ("a", "is", "am") from acting as correction targets, so genuine
    grammar errors survive to be classified.

    Edit distance runs on code points. For scripts with combining marks, one
    edit can change a vowel sign, so this is a re-validation item per language.

    Returns ``(corrected_text, changed)``.
    """
    targets = [w for w in vocab if len(w) >= lang.typo_min_token_length]
    out: list[str] = []
    changed = False
    for tok in norm.split():
        if tok in vocab:
            out.append(tok)
            continue
        replacement = None
        for w in targets:
            if _within_edit_distance_one(tok, w):
                replacement = w
                break
        if replacement is not None:
            out.append(replacement)
            changed = True
        else:
            out.append(tok)
    return " ".join(out), changed
