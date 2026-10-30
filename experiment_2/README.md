# Experiment 2: cascaded speech-to-speech

A plain voice loop, built before any pedagogy so it can be measured on its own:

```
mic (push-to-talk) → Whisper on the Apple GPU (STT) → Gemini / NVIDIA (LLM, streamed)
                   → macOS voices or ElevenLabs (TTS, sentence by sentence) → speaker
```

## Run

From this folder, with keys in the repo-root `.env`. Activate the project venv first:
outside it, `mlx-whisper` is missing and STT falls back to the slow CPU engine.

```bash
source ../.venv/bin/activate              # once per terminal (or use ../.venv/bin/python)
python -m cascade.cli                     # voice -> voice (Enter to start, Enter to stop)
python -m cascade.cli --text              # text  -> voice
python -m cascade.cli --no-tts            # voice -> text
python -m cascade.cli --text --no-tts     # text  -> text
python -m cascade.cli --wav a.wav b.wav   # one turn per file, for repeatable runs
python -m cascade.cli --offline           # echo instead of an LLM, no network
python -m cascade.cli --tts elevenlabs    # ElevenLabs voice; macOS voices take over if it fails
python -m cascade.cli --whisper-model small
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
| TTS | `tts.py` | macOS voices in-process: Rishi (en_IN), Lekha (hi_IN) | Voices stay loaded, so speech starts in ~30 ms (the `say` command needs ~700 ms per call). Replies are split by script, one voice per language. |

`pipeline.py` speaks each sentence as soon as the LLM produces it, times the turn on the
wall clock, keeps the chat history, and speaks a fallback line if no LLM can answer.

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
costs ~50 credits (0.5 per character), so 10k credits a month is roughly 180 replies:
the free macOS voices stay the default. If ElevenLabs fails mid-conversation, the error
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
- Swap the LLM stage for experiment_1's tutor (`ChatLLM.stream(history, text)` yields
  sentences).
