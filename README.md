# Bhasha Setu: voice tutor for Hindi speakers learning English

The end goal is a spoken tutor: a Hindi speaker talks, an AI tutor answers out loud and
guides them from Hindi towards English. The repo is organised as experiments.

| | Folder | What it is | Status |
|---|---|---|---|
| **Experiment 1** | [`experiment_1/`](experiment_1/README.md) | Pedagogy first: the text tutor (lesson structure, hint ladder, Hindi→English support stages, LLM judge, eval set). | Done as a text prototype; paused |
| **Experiment 2** | [`experiment_2/`](experiment_2/README.md) | Voice first: a cascaded speech-to-speech pipeline, STT → LLM → TTS, with per-stage latency logging. | Working baseline |

## Why two experiments

Experiment 1 built the pedagogy before any voice, and building it that way was causing
problems. Its lesson logic was designed around typed text and has never run through a
real speech loop. A voice loop brings risks that text can't show: STT errors, Hindi
heard as Urdu, latency, replies too long to listen to, and Hindi script the TTS can't
read.

Experiment 2 builds the voice pipeline first and measures it. The plan:

1. **Cascade baseline** (now): Whisper → Gemini (a plain conversation partner) → macOS
   voices. Measure each stage's latency and where it breaks with Hindi and Hinglish.
2. **Improve the cascade** (in progress): Whisper on the Apple GPU, sentence streaming
   into TTS and in-process voices brought first audio from ~4.5 s to ~2.5 s. Next:
   real learner clips, ElevenLabs, chunked STT.
3. **Plug the pedagogy in**: replace the LLM stage with experiment_1's tutor. Both use
   the same seam: text in, English/Hindi reply segments out.

## Setup (shared)

One virtualenv and one `.env` at the repo root serve both experiments.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r experiment_1/requirements.txt -r experiment_2/requirements.txt google-genai
echo "GEMINI_API_KEY=..." > .env        # never committed

(cd experiment_2 && python -m cascade.cli)          # talk to the cascade
(cd experiment_1 && python -m tutor.cli --offline)  # the text tutor
```

Run each experiment's tests from inside its folder (`python -m pytest`).
