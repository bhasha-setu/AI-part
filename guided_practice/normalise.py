"""Transcript normalisation, driven by a per-language config.

Every transcript AND every authored phrasing passes through this function, so
matching compares like with like.

Steps (in order):
  1. Unicode NFKC (composes forms, so NFC and NFD input agree).
  2. Map the language's apostrophe variants to "'" and casefold.
  3. Strip punctuation by Unicode category: drop P* and S*, keep letters (L*),
     combining marks (M*, so Devanagari matras survive) and digits (N*).
     Intra-word apostrophes are kept for contractions.
  4. Drop leading filler words listed in the language config.
  5. Expand the language's contractions ("i'm" -> "i am").
  6. Drop dangling apostrophes and collapse whitespace.

Nothing here names a language. The per-language data comes from
config/languages/<code>.yaml via guided_practice.config.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .config import LanguageConfig

_PUNCT_CATEGORIES = ("P", "S")
_APOSTROPHE = "'"
_WS_RE = re.compile(r"\s+")
_DANGLING_APOS_RE = re.compile(r"(?<!\w)'+|'+(?!\w)")


def _strip_punctuation(text: str) -> str:
    out = []
    for ch in text:
        if ch == _APOSTROPHE or unicodedata.category(ch)[0] not in _PUNCT_CATEGORIES:
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


@lru_cache(maxsize=None)
def _contraction_re(keys: tuple):
    if not keys:
        return None
    alternatives = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    return re.compile(r"(?<![\w'])(" + alternatives + r")(?![\w'])")


def _expand_contractions(text: str, table: dict) -> str:
    rx = _contraction_re(tuple(table))
    if rx is None:
        return text
    return rx.sub(lambda m: table[m.group(1)], text)


def _default_lang() -> LanguageConfig:
    # Compatibility default for call sites that don't pass a language.
    from .course import default_course

    return default_course().lang


def normalise(text: str | None, lang: LanguageConfig | None = None) -> str:
    """Return the normalised form of ``text`` for the given language."""
    if not text:
        return ""
    lang = lang or _default_lang()

    t = unicodedata.normalize("NFKC", text)
    for variant in lang.apostrophe_variants:
        t = t.replace(variant, _APOSTROPHE)
    t = t.casefold().strip()
    t = _strip_punctuation(t)

    tokens = t.split()
    while tokens and tokens[0] in lang.leading_fillers:
        tokens.pop(0)
    t = " ".join(tokens)

    t = _expand_contractions(t, lang.contractions)
    t = _DANGLING_APOS_RE.sub("", t)
    return _WS_RE.sub(" ", t).strip()
