# Experiment 1: pedagogy core (text tutor), English for Hindi speakers

> Paused. Built pedagogy-first, before any voice loop; see the [repo README](../README.md)
> for why, and [experiment_2](../experiment_2/README.md) for the cascade it will plug into.

The AI core of a speaking-first language app. A Hindi speaker practises English with an
LLM tutor: they answer by voice or text, in English, Hindi or a mix, and the tutor helps
them move from Hindi to English step by step.

The target learner knows some English (between beginner and intermediate) and needs the
basics: making sentences and speaking without freezing.

## Quick start

```bash
cd experiment_1
pip install -r requirements.txt
pip install google-genai              # for the live model
# GEMINI_API_KEY=... in .env (here or at the repo root; never committed)

python3 -m tutor.cli                  # chat with the tutor (live Gemini)
python3 -m tutor.cli --offline        # same lesson, no network
python3 -m tutor.cli --debug          # show verdict and hint level each turn
python3 -m pytest                     # all tests, offline
```

## How a lesson works

1. **Practice.** Phrases one at a time ("Someone asks your name…"). A wrong answer gets
   a hint, not the answer: pointer → clue → first words → full sentence.
2. **Conversation.** A short roleplay. Mistakes are noted silently, never corrected
   mid-conversation.
3. **Review.** Up to 3 of the learner's own sentences. They try to fix each one first,
   then see the correct form and a one-line reason.
4. **Summary.** What was said from memory, and spaced review for missed phrases.

A Hindi answer is a stepping stone, not a mistake:

```
you>   mera naam Ravi hai
tutor> That's right! Now say it in English. Start with "My name is…"
you>   my name Ravi
tutor> Almost! One small word is missing after "name".
```

**Hindi → English transition.** Four support stages: Hindi first → bilingual → English
with Hindi help → English only. A learner moves one stage after each lesson, based on
how much they said from memory and how often they needed Hindi.

**Who decides what.** The server owns the structure (phase, next phrase, hint level,
caps, spaced review, support stage). The LLM owns the words (verdict, replies, Hindi
bridges). The server enforces the rules: a correct answer is never marked wrong, an
answer given too early is retried and then masked ("My name ___ Ravi"), and Hindi is
dropped where the stage doesn't need it.

## Layout

```
tutor/              the conversational tutor (main experience)
config/tutor.yaml   tutor system prompt, support stages, rules and numbers
courses/            lesson content (YAML): english_for_hindi_a1, hindi_for_english_a1
guided_practice/    lesson content loader, answer matching, spaced review
pedagogy_server/    earlier LLM judge pipeline, generator, roleplay; Gemini client (llm.py)
web/                browser app (another team); API contract in web/API.md
docs/               research reviews, product framework, design notes
tests/              pytest suite, runs offline
```

## Status

Working: the tutor loop, Hindi bridging, support stages, review, spaced review, offline
mode and tests.

Not built yet:
- Tutor wired into the web API (the web app still talks to `pedagogy_server`).
- LLM-generated lessons for the tutor (it uses the hand-written ones).
- Placement check at the start.
- Native-speaker review of the Hindi lines (the model writes them, unchecked).

Notes for voice: speech-to-text must not be forced to English, or spoken Hindi is lost.
Replies come as English/Hindi segments so each can use its own TTS voice. Hindi is
written in Roman script by default; set `hindi_script: devanagari` for better Hindi TTS.

## Docs

- `docs/Language Learning Product Pedagogy and Interaction Framework.pdf`: product and pedagogy framework
- `docs/reasoning_core.md`: lesson rules and their research basis
- `docs/design_notes.md`: full design decisions and history of the earlier builds
