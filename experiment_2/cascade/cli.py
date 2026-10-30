"""Talk to the cascade. Input and output can each be voice or text:

    python -m cascade.cli                     # voice -> voice (push-to-talk mic)
    python -m cascade.cli --text              # text  -> voice
    python -m cascade.cli --no-tts            # voice -> text
    python -m cascade.cli --text --no-tts     # text  -> text
    python -m cascade.cli --wav a.wav b.wav   # audio files, one per turn
    python -m cascade.cli --offline           # echo instead of an LLM, no network
    python -m cascade.cli --tts elevenlabs    # override config.yaml's engines
    python -m cascade.cli --stt-engine faster-whisper --whisper-model base

Each turn (transcript, reply, timings, which LLM answered) is appended to
runs/<time>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .llm import EchoLLM, build_chain
from .pipeline import CascadePipeline
from .stt import TypedSTT, load_stt
from .tts import SilentTTS, load_tts
from .types import TurnResult

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
QUIT = {"q", "quit", "exit"}


def load_dotenv(path: Path) -> None:
    """Copy KEY=value lines into the environment without overriding. Never prints values."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def load_env() -> None:
    load_dotenv(EXPERIMENT_ROOT / ".env")
    load_dotenv(EXPERIMENT_ROOT.parent / ".env")   # shared keys at the repo root


def format_turn(i: int, result: TurnResult) -> str:
    t, tm = result.transcript, result.timing
    prob = f" p={t.language_prob:.2f}" if t.language_prob is not None else ""
    ms = lambda v: f"{v:.0f}" if v is not None else "-"
    lines = [f"you   [{t.language}{prob}]> {t.text or '(nothing heard)'}",
             f"tutor> {result.reply_text}"]
    if result.llm_error:
        lines.append(f"       (LLM error: {result.llm_error}"
                     + ("; fallback line used)" if result.fallback_used else ")"))
    lines.append(f"       turn {i}: stt {ms(tm.stt_ms)} · llm first sentence {ms(tm.llm_first_ms)}"
                 f" · first audio {ms(tm.first_audio_ms)} · total {ms(tm.total_ms)} ms"
                 + (f" · via {result.llm}" if result.llm else ""))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--text", action="store_true", help="type instead of speaking")
    mode.add_argument("--wav", nargs="+", type=Path, help="audio files, one per turn")
    parser.add_argument("--offline", action="store_true", help="echo LLM, no network")
    parser.add_argument("--no-tts", action="store_true", help="print replies only")
    parser.add_argument("--tts", help="override tts.engine (macos-native, macos-say, elevenlabs)")
    parser.add_argument("--stt-engine", help="override stt.engine (mlx, faster-whisper)")
    parser.add_argument("--whisper-model", help="override stt.model (small, large-v3-turbo, ...)")
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "config.yaml")
    parser.add_argument("--keep-audio", action="store_true", help="save mic input under runs/")
    return parser


def build_pipeline(args, cfg: dict, *, play: bool = True) -> CascadePipeline:
    """Shared by the CLI and the benchmark. Raises RuntimeError on missing keys."""
    llm_cfg = cfg["llm"]
    llm = EchoLLM() if args.offline else build_chain(llm_cfg)
    if getattr(args, "text", False):
        stt = TypedSTT()
    else:
        print(f"Loading STT ({args.stt_engine or cfg['stt'].get('engine', 'mlx')}:"
              f"{args.whisper_model or cfg['stt']['model']})…", flush=True)
        stt = load_stt(cfg["stt"], override_model=args.whisper_model,
                       override_engine=args.stt_engine)
    tts_cfg = dict(cfg["tts"], **({"engine": args.tts} if args.tts else {}))
    tts = SilentTTS() if args.no_tts else load_tts(tts_cfg, os.environ)
    return CascadePipeline(stt, llm, tts, play=play,
                           fallback_reply=llm_cfg.get("fallback_reply", "Sorry, please say that again."),
                           no_speech_reply=llm_cfg.get("no_speech_reply", "I didn't hear anything."),
                           max_history_turns=llm_cfg.get("max_history_turns", 12))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    load_env()
    try:
        pipeline = build_pipeline(args, cfg)
    except RuntimeError as exc:
        print(f"Can't start: {exc}. (--offline runs without an LLM.)", file=sys.stderr)
        return 2

    runs = EXPERIMENT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = runs / f"{stamp}.jsonl"
    audio_root = runs / stamp if args.keep_audio else None
    print(f"[stt: {pipeline.stt.name} · llm: {pipeline.llm.name} · tts: {pipeline.tts.name}]")
    print(f"[log: {log_path.relative_to(EXPERIMENT_ROOT)}]  Say or type 'quit' to stop.\n")

    def inputs():
        if args.wav:
            yield from args.wav
            return
        while True:
            try:
                if args.text:
                    line = input("you> ").strip()
                    if line.lower() in QUIT:
                        return
                    yield line
                else:
                    if input("Press Enter to talk (or type quit): ").strip().lower() in QUIT:
                        return
                    from .mic import record_push_to_talk

                    yield record_push_to_talk()
            except (EOFError, KeyboardInterrupt):
                return

    with log_path.open("a", encoding="utf-8") as log:
        for i, audio in enumerate(inputs(), 1):
            if audio_root is not None and not isinstance(audio, (str, Path)):
                from .mic import save_wav

                audio_root.mkdir(parents=True, exist_ok=True)
                save_wav(audio, audio_root / f"turn{i}.wav")
            result = pipeline.turn(audio)
            print(format_turn(i, result) + "\n", flush=True)
            record = {"turn": i, "stt": pipeline.stt.name, "tts": pipeline.tts.name,
                      **result.to_dict()}
            if isinstance(audio, Path):
                record["input_file"] = str(audio)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            if result.transcript.text.strip().lower().strip(".!? ") in QUIT:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
