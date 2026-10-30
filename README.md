# Bhasha Setu: a voice tutor for Hindi speakers learning English

A Hindi speaker talks; the tutor answers out loud and teaches lesson by lesson, from
Hindi towards English:

```
mic (push-to-talk) → Whisper on the Apple GPU (STT) → the tutor on Gemini (LLM)
                   → ElevenLabs or macOS voices (TTS) → speaker
```

## Layout

| Folder | What it is |
|---|---|
| `cascade/` | The voice app: STT, LLM, TTS, the pipeline, the CLI, learners and progress, redo versions, extra practice |
| `tutor/` | The tutor: lesson phases (practice with a hint ladder, roleplay, review), Hindi help levels |
| `guided_practice/` | Lesson content, answer matching, spaced review |
| `pedagogy_server/` | The Gemini client and the lesson generator used by extra practice |
| `config.yaml` | Voice app settings (STT, LLM keys and models, TTS, tutor overrides) |
| `config/` | The tutor's own settings (`tutor.yaml`, generator, languages) |
| `curriculum/` | The 3-stage, 12-lesson English course |
| `courses/` | Course metadata (language pair, level) |
| `tests/` | Offline tests (`python -m pytest`) |
| `runs/` | Logs and learner progress (git-ignored) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp your-keys .env        # see the table below; .env is never committed
```

## Run

From the repo root. Outside `.venv`, the CLI restarts itself with `.venv/bin/python`
(otherwise `mlx-whisper` is missing and STT falls back to the slow CPU engine).

```bash
python -m cascade.cli                     # voice lessons: asks your username, shows progress + menu
python -m cascade.cli --text              # the same, typed
python -m cascade.cli --user himanshu --lesson shopping   # one lesson, no menu
python -m cascade.cli --hindi-help 0      # force Hindi first
python -m cascade.cli --chat              # free conversation instead of lessons
python -m cascade.cli --no-tts            # print replies only
python -m cascade.cli --wav a.wav b.wav   # one turn per file, for repeatable runs
python -m cascade.cli --offline           # no network: the tutor's rule-based lines
python -m cascade.cli --tts macos-native  # free macOS voices instead of ElevenLabs
python -m cascade.cli --keep-audio        # save mic input under runs/
python -m pytest                          # offline tests
```

Keys (`.env` at the repo root; numbered extras such as `GEMINI_API_KEY_2` are picked up):

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY` | LLM, first choice |
| `GEMINI_API_KEY_2` | LLM, second (a key from a different account has its own quota) |
| `NVIDIA_API_KEY` | LLM, last fallback (build.nvidia.com, OpenAI-compatible) |
| `ELEVENLABS_API_KEY` | `--tts elevenlabs` (needs the Text to Speech permission) |

The LLMs are tried in that order. One that fails (429, timeout) answers nothing that turn,
the next one does, and the failed one is skipped for 60 s. Gemini quotas are per Google
Cloud project, so keys from one project share one quota; creating extra projects just to
get around free-tier limits may break Google's terms.

The first mic run triggers a macOS microphone permission prompt. The first load of a
Whisper model downloads it (large-v3-turbo is about 1.6 GB).

## The stages

| Stage | Module | Default | Why |
|---|---|---|---|
| STT | `stt.py` | mlx-whisper `large-v3-turbo`, language picked by `small`, Silero VAD first | Runs on the M-series GPU, ~5x faster than faster-whisper on CPU. Multilingual and never forced to English. Detection is limited to `en`/`hi` (Whisper often hears Hindi as Urdu). A small detector saves turbo's ~1 s second encoder pass. The VAD (~15 ms) trims push-to-talk dead air and returns nothing on silence; without it turbo transcribes silence as "Thank you." |
| LLM | `llm.py` | Gemini `gemini-3.5-flash-lite`, minimal thinking, then NVIDIA | Streams sentences. A provider that fails (429, timeout) is skipped for 60 s and the next one answers. |
| TTS | `tts.py` | ElevenLabs Flash v2.5 (George), macOS voices if it fails | ElevenLabs reads mixed Hindi/English in one call and starts in ~0.2 s on a warm connection. The macOS voices (Rishi en_IN, Lekha hi_IN) stay loaded in-process, start in ~30 ms, and need replies split by script. |

`pipeline.py` speaks each sentence as soon as the LLM produces it, times the turn on the
wall clock, keeps the chat history, and speaks a fallback line if no LLM can answer.

## Lessons (the tutor)

The LLM stage becomes the tutor's tutor (`cascade/tutor_brain.py`), imported read-only:
the tutor's files are not changed and its state isn't written to. The tutor runs the
lesson: it greets first, practises each phrase with a hint ladder (Hindi answers get the
English for them), plays a short roleplay without correcting, reviews 2-3 mistakes, then
prints a summary and saves spaced review and the next support stage to
`runs/tutor_state/`. Each turn prints and logs the phase, verdict, hint level and whether
the LLM or the tutor's rules answered.

- Settings: `tutor:` in `config.yaml`, merged over `config/tutor.yaml`. Voice
  needs Hindi in Devanagari (the Hindi voice can't read Roman Hindi), and thinking is
  minimal for latency.
- Tips: after a sentence with a mistake or an unnatural phrase, the tutor first says a
  better way ("Tip: you can say, ...") and prints why. The tip is a separate field in
  the tutor's JSON (`VoiceGuard`), so the roleplay itself still doesn't correct, and the
  review at the end still happens. In practice, a tip that would say the target phrase
  before the hint ladder does is dropped. `tutor.tips: false` turns tips off.
- Keys: the tutor's JSON calls go Gemini key 1, then key 2 (60 s cool-down after a
  failure). If key 1 hasn't answered in `hedge_after_seconds` (4 s), key 2 gets the same
  request and the first answer wins: single calls stalled for 12-19 s on 2026-10-30.
  When every key fails, the tutor's rule-based lines answer, so a lesson never stops.
- NVIDIA isn't used for the tutor: nemotron-3.5-lightning returned valid JSON in 1.4-3.7
  s, but its replies said "मैं नेमोट्रॉन हूँ" ("I am Nemotron") mid-lesson.
- Language: Whisper is unsure of accented English (p 0.2-0.3), so Hindi is picked only
  when it has more than 70% of the en+hi score (Urdu counted as Hindi); both scores are
  logged per turn.
- Keys pressed while the tutor speaks are dropped before the next prompt, so a stray
  Enter can't start and stop a recording at once.
- Saying or typing "quit" stops before the tutor sees it, so it isn't graded.

Live run (2026-10-30, clips made with macOS voices, nothing played): the Hindi answer
was bridged to English, and four correct English answers were judged `correct`. STT
was ~1.45 s and the tutor reply 1.05-1.75 s per turn, so first audio is about 2.5-3.2 s:
the tutor returns one JSON reply, so nothing can be spoken before all of it arrives.
Whisper heard "मेरा नाम रवि है" as "मेरा नाम रविवार है": names are a known STT weak spot.

## Learners, curriculum and progress

**Profiles.** The tutor asks for a username (no password yet; `--user` skips the
question). Everything about that learner is kept in `runs/learners/<username>/`
(git-ignored): `progress.json` (lessons and attempts), the tutor's `learners.json`
(spaced review) and `profiles.json` (Hindi help level), redo versions and generated
lessons.

**Two separate progressions**, shown separately:
- *Curriculum*: which lessons are done. 3 stages × 4 lessons in
  `curriculum/english_for_hindi/`: **Basics** (introducing yourself, family and home,
  daily routine, numbers/time/money), **Everyday situations** (ordering food, shopping,
  directions and transport, doctor and pharmacy), **Real world** (phone appointment, job
  interview, workplace small talk, solving a problem). Lesson files use the tutor's
  format and pass its validator (`tests/test_curriculum.py`). They are DRAFT: written
  with an LLM's help and checked by tests, not yet reviewed by a teacher.
- *Hindi help*: how much Hindi the tutor uses (Hindi first -> English only), moved by
  the tutor's rule after each lesson.

**Completion.** A lesson is complete when every phrase is achieved (with or without
hints). Otherwise the learner chooses: redo now, mark complete and move on (missed
phrases stay due in spaced review), or stop. A lesson quit half-way records nothing.

**Redo = a fresh version** (`cascade/variants.py`). Same target phrases, accepted
answers and mistake patterns (so validation and spaced review still hold); the LLM
writes a new setting, new situation prompts and a new roleplay partner. Phrases missed
last time come first, and past review mistakes go into the tutor's instructions for that
lesson. The server checks the version (no answer given away, English only, lengths) and
falls back to the original lesson, reordered, if the LLM fails twice. About 2 s.

**Extra practice** (`[x]` in the menu, `cascade/extras.py`): the tutor's scenario
generator plans 2-3 new lessons inside a stage, shaped by what the learner found hard,
then writes and validates the chosen one (GENERATED_DRAFT). Planning takes ~2 s,
writing up to ~90 s.

## Provider notes (tested 2026-10-30)

**NVIDIA free endpoints.** `nemotron-3.5-lightning-30b-a3b` with thinking off
(`chat_template_kwargs: {enable_thinking: false}` in `config.yaml`) is the fallback:
~0.65 s when idle, but one call took 11 s, so it stays last. With thinking on it spends all
300 tokens reasoning and returns nothing. `nemotron-3.5-bolt-e4b` was as fast but dropped
punctuation. `gemma-4-31b-it` and `deepseek-v4.1-flash` timed out after 90 s. The Llama
models with free endpoints are safety and vision models, not chat models. Your key's model
list: `python -m cascade.bench --list-nvidia-models`.

**ElevenLabs (free plan).** Flash v2.5 starts audio in ~180-300 ms (the first request of a
session ~4 s, opening the connection) and reads mixed Hindi/English in one call. Only
built-in *Default* voices work through the API on the free plan; Voice Library voices
get HTTP 402 `paid_plan_required`. The config uses George (Default, English accent); to
use your own voice on a paid plan, uncomment `voice_id_env` in `config.yaml`. A reply
costs ~50 credits (0.5 per character), so 10k credits a month is roughly 180 replies;
`--tts macos-native` uses the free macOS voices. If ElevenLabs fails mid-conversation, the error
is printed once and the macOS voices speak from then on.

## Measurements

`python -m cascade.bench` reproduces these (`--stt`, `--tts`, `--pipeline`). Every turn of
the CLI is also logged to `runs/<time>.jsonl`. Times are medians on an M2 MacBook Air
(16 GB), over five short clips made with macOS voices, 2026-10-30.

**First audio** (the learner stops talking → the reply starts):

| | STT | LLM first sentence | Voice starts | First audio |
|---|---|---|---|---|
| Baseline: faster-whisper small (CPU), Gemini, `say` | 2.5 s | 1.0 s | 0.7 s | ~4.5 s (estimate) |
| Now: mlx turbo + small detector, Gemini minimal thinking, native voices | 1.5 s | 0.9 s | <0.05 s | **2.5 s** (measured\*) |
| Speed option: mlx `small` (`--whisper-model small`) | 0.6 s | 0.9 s | <0.05 s | ~1.6 s (estimate) |

\* Measured with nothing played, so "voice starts" is the full synthesis time of the first
piece (an upper bound). In live play the logs count until macOS reports speaking (~2 ms),
which slightly under-reports when sound is actually heard.

**STT** (median ms; transcript of the mixed clip "मेरा नाम Ravi है और मैं Delhi में software engineer हूँ"):

| Engine | English | Hindi | Mixed | Mixed transcript |
|---|---|---|---|---|
| faster-whisper small, CPU | 2,560 | 2,680–2,860 | 3,040 | मैं डल ही में सुछ्ट्वर अंजीनेर हूँ |
| mlx small | 520 | 580–870 | 810 | मैं दल ही में सुट्वर अंजीनेर हूँ |
| mlx medium + small detector | 1,080 | 1,250–1,570 | 1,740 | मैं डल्ही में सॉस्ट्वेर अंजिनेयर हूं |
| mlx large-v3-turbo | 2,320 | 2,350–2,450 | 2,420 | मैं डेल ही में software engineer हूँ |
| **mlx large-v3-turbo + small detector** | **1,400** | **1,470–1,550** | **1,530** | मैं डेल ही में software engineer हूँ |

Only turbo got "मुझे चाय पीना पसंद है" right. It kept "software engineer" in Latin script
in this run, but not in a rerun of the same clip with silence around it
("सॉस्ट्वेर एंजिनेर"). So mixed speech is not solved. Whisper picks one language per clip
and writes everything in that script. For a tutor that must see which English the
learner actually said, this is the main open STT problem. Names get garbled too ("रवे" for
Ravi).

**Still open.** These are synthetic voices; real learner speech (accents, Hinglish,
noise) may rank the models differently. Record a few clips with `--keep-audio`, then
`python -m cascade.bench --stt mlx:small mlx:large-v3-turbo+small -- runs/<time>/*.wav`.

## Next

- Benchmark real learner clips (English, Hindi, Hinglish).
- Compare voice quality: macOS Rishi/Lekha vs ElevenLabs George, with learners.
- Try ElevenLabs Scribe (speech to text; the key has the permission) on mixed
  Hindi-English clips, where Whisper writes English words in Devanagari.
- Transcribe while the learner is still talking (chunked STT), so only the last chunk
  is left when they stop: the next big cut after this one.
- Tutor mode: run full lessons with real learner speech; check the verdicts with
  minimal thinking; measure Gemini calls per lesson against the free-tier limits.
- Tutor latency: stream the tutor's JSON and speak the first `say` segment as soon as it
  is complete.
