"""Deterministic checks the server runs around the LLM.

    detect_language   en / hi / mixed for a learner answer (Devanagari or Roman Hindi)
    rule_verdict      the offline verdict: accepted form, known mistake, Hindi, unclear
    reveal_spans      where a reply spells out an accepted answer
    mask              blank the revealed words the learner hasn't said: "My name ___ Ravi"
"""

from __future__ import annotations

import re
from typing import Optional

from guided_practice.content import Item
from guided_practice.normalise import normalise

_TOKEN_RE = re.compile(r"[\w']+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Common Hindi words as people type them in Roman script.
HINGLISH = frozenset("""
mera meri mere mujhe mujhko naam hai hain hoon hun hu hoo tha thi mai
kya kyun kaise kahan kaun nahi nahin haan ji aap aapka aapki tum tumhara hum
humara se ka ki ke ko mein raha rahi rahe karta karti karte padhta padhti chahiye
chahta chahti pasand accha achha acha theek thik bahut bhi aur lekin toh kuch
wala wali yahan wahan abhi kal aaj khana pani chai ek
""".split())
# Also ordinary English words: they count as Hindi only next to a word from HINGLISH.
HINGLISH_AMBIGUOUS = frozenset("main the do na".split())


def tokens(text: str) -> list:
    return [m.group(0).casefold() for m in _TOKEN_RE.finditer(text or "")]


def detect_language(text: str, item: Optional[Item] = None) -> str:
    """'hi', 'mixed' or 'en'. Devanagari is always Hindi. Roman text is Hindi when most
    of its words are common Hindi words and the item's English words are absent."""
    if _DEVANAGARI_RE.search(text or ""):
        return "mixed" if re.search(r"[A-Za-z]{3,}", text) else "hi"
    words = tokens(text)
    if not words:
        return "en"
    hindi = [w for w in words if w in HINGLISH]
    if hindi:
        hindi += [w for w in words if w in HINGLISH_AMBIGUOUS]
    english = [w for w in words if item is not None and w in item.vocab and w not in HINGLISH]
    if len(hindi) >= 2 and len(hindi) >= len(words) / 2:
        return "mixed" if len(english) >= 2 else "hi"
    return "mixed" if len(hindi) >= 2 else "en"


def rule_verdict(text: str, item: Item) -> dict:
    """What the content alone can say about an answer. Used offline, as the fallback
    when the LLM is unavailable, and as the floor (an accepted form is always correct)."""
    norm = normalise(text, item.lang)
    if item.match_accepted(norm) is not None:
        return {"verdict": "correct", "error": None}
    for ce in item.common_errors:
        slots = ce.match(norm)
        if slots is not None:
            fixed = ce.correction_example or item.model_answer
            for k, v in slots.items():
                fixed = fixed.replace("{%s}" % k, v)
            fixed = fixed[:1].upper() + fixed[1:]
            return {"verdict": "form_error", "slots": slots,
                    "error": {"said": text.strip(), "fixed": fixed, "type": ce.id,
                              "why": " ".join(ce.why.split())}}
    if detect_language(text, item) != "en":
        return {"verdict": "hindi", "error": None}
    if set(norm.split()) & item.vocab:
        return {"verdict": "unclear", "error": None}
    return {"verdict": "off_topic", "error": None}


def reveal_spans(reply: str, item: Item, learner_text: str, min_tokens: int = 2) -> list:
    """Character spans of ``reply`` that spell out an accepted answer.

    A slot counts only when its value is something the learner said (their own name,
    their city), so "I am happy" is not read as the answer "I am {name}"."""
    found = [(m.start(), m.end(), m.group(0)) for m in _TOKEN_RE.finditer(reply or "")]
    said = set(normalise(learner_text, item.lang).split())
    spans = []
    i = 0
    while i < len(found):
        hit = None
        for j in range(i + min_tokens, len(found) + 1):
            window = normalise(" ".join(t for _, _, t in found[i:j]), item.lang)
            slots = item.match_accepted(window)
            if slots is not None and all(set(v.casefold().split()) <= said for v in slots.values()):
                hit = j
                break
        if hit is None:
            i += 1
            continue
        spans.append((found[i][0], found[hit - 1][1]))
        i = hit
    return spans


def mask(reply: str, spans: list, learner_text: str) -> str:
    """Blank the words inside ``spans`` that the learner didn't say."""
    said = set(tokens(learner_text))
    out, pos = [], 0
    for start, end in spans:
        out.append(reply[pos:start])
        words, blank = [], False
        for m in _TOKEN_RE.finditer(reply[start:end]):
            if m.group(0).casefold() in said:
                words.append(m.group(0))
                blank = False
            elif not blank:
                words.append("___")
                blank = True
        out.append(" ".join(words))
        pos = end
    out.append(reply[pos:])
    return "".join(out)
