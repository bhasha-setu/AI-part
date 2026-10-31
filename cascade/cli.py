"""Talk to the cascade. Input and output can each be voice or text:

    python -m cascade.cli                     # log in, then voice lessons (ElevenLabs voice)
    python -m cascade.cli --user himanshu --lesson shopping   # one lesson, no menu
    python -m cascade.cli --chat              # free conversation instead of lessons
    python -m cascade.cli --text              # text  -> voice
    python -m cascade.cli --no-tts            # voice -> text
    python -m cascade.cli --text --no-tts     # text  -> text
    python -m cascade.cli --wav a.wav b.wav   # audio files, one per turn
    python -m cascade.cli --offline           # no network (rule-based tutor / echo chat)
    python -m cascade.cli --tts macos-native  # override config.yaml's voice
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
from .stt import TypedSTT, load_stt, resolve_stt_config
from .tts import SilentTTS, load_tts
from .types import TurnResult

ROOT = Path(__file__).resolve().parent.parent
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
    load_dotenv(ROOT / ".env")   # keys at the repo root


def you_line(t) -> str:
    prob = f" p={t.language_prob:.2f}" if t.language_prob is not None else ""
    return f"you   [{t.language}{prob}]> {t.text or '(nothing heard)'}"


def format_turn(i: int, result: TurnResult, *, heard_shown: bool = False) -> str:
    """``heard_shown``: the "you>" line was already printed when STT finished."""
    t, tm = result.transcript, result.timing
    lines = [you_line(t)] if i and not heard_shown else []
    d = result.tutor or {}
    tip = d.get("tip")
    if tip:
        lines.append(f"tip   \"{tip['said']}\" -> \"{tip['better']}\""
                     + (f"  ({tip['why']})" if tip.get("why") else ""))
    lines.append(f"tutor> {result.reply_text}")
    if result.tutor:
        lines.append(f"       [{d.get('phase')} · {d.get('source')} · verdict={d.get('verdict')} · "
                     f"hint {d.get('hint_level')}->{d.get('next_hint_level')}]")
    if result.llm_error:
        lines.append(f"       (LLM error: {result.llm_error}"
                     + ("; fallback line used)" if result.fallback_used else ")"))
    # what the learner felt: the wait before the voice starts, split by stage
    waited = tm.first_audio_ms
    sec = lambda v: f"{v / 1000:.1f}" if v is not None else "-"
    parts = [f"stt {sec(tm.stt_ms)}", f"llm {sec(tm.llm_first_ms)}"]
    if waited is not None:
        parts.append(f"voice {sec(waited - tm.stt_ms - (tm.llm_first_ms or 0))}")
    spoke = f" · tutor spoke {sec(tm.total_ms - waited)} s" if waited is not None else ""
    lines.append(f"       turn {i}: you waited {sec(waited)} s ({' + '.join(parts)}){spoke}"
                 + (f" · via {result.llm}" if result.llm else ""))
    return "\n".join(lines)


def fresh_input(prompt: str = "") -> str:
    """input(), after dropping keys pressed while the tutor was busy: an Enter pressed
    during a long reply would otherwise start and stop the next recording at once."""
    if sys.stdin.isatty():
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except (ImportError, OSError):
            pass
    return input(prompt)


def warm_up(tts, model) -> None:
    """Open the network connections in the background while the learner logs in, so
    the first reply doesn't pay for them (ElevenLabs: ~1.3 s cold vs ~0.2 s warm)."""
    import threading

    def run():
        for target in (getattr(tts, "primary", tts), model):
            try:
                target.warm_up()
            except Exception:
                pass   # only an optimisation

    threading.Thread(target=run, daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--text", action="store_true", help="type instead of speaking")
    mode.add_argument("--wav", nargs="+", type=Path, help="audio files, one per turn")
    parser.add_argument("--offline", action="store_true", help="echo LLM, no network")
    parser.add_argument("--no-tts", action="store_true", help="print replies only")
    parser.add_argument("--tts", choices=["macos-native", "macos-say", "elevenlabs"],
                        help="override tts.engine")
    parser.add_argument("--stt-engine", help="override stt.engine (mlx, faster-whisper)")
    parser.add_argument("--whisper-model", help="override stt.model (small, large-v3-turbo, ...)")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--keep-audio", action="store_true", help="save mic input under runs/")
    parser.add_argument("--chat", action="store_true",
                        help="free conversation instead of lessons (the default is lessons)")
    parser.add_argument("--tutor", action="store_true", help=argparse.SUPPRESS)   # old flag; lessons are the default
    parser.add_argument("--user", help="tutor: your username (skips the question)")
    parser.add_argument("--lesson", help="tutor: run this lesson only (e.g. shopping)")
    parser.add_argument("--hindi-help", type=int, choices=range(4),
                        help="tutor: force the Hindi help level (0 = Hindi first, 3 = English only)")
    return parser


def build_pipeline(args, cfg: dict, *, play: bool = True) -> CascadePipeline:
    """Shared by the CLI and the benchmark. Raises RuntimeError on missing keys."""
    llm_cfg = cfg["llm"]
    if not getattr(args, "chat", False):
        llm = EchoLLM()   # placeholder: each lesson puts its tutor here (tutor_cli)
    else:
        llm = EchoLLM() if args.offline else build_chain(llm_cfg)
    if getattr(args, "text", False):
        stt = TypedSTT()
    else:
        engine_label, model_label = resolve_stt_config(cfg["stt"],
                                                       override_engine=args.stt_engine,
                                                       override_model=args.whisper_model)
        print(f"Loading STT ({engine_label}:{model_label})…", flush=True)
        stt = load_stt(cfg["stt"], override_model=args.whisper_model,
                       override_engine=args.stt_engine)
    tts_cfg = dict(cfg["tts"], **({"engine": args.tts} if args.tts else {}))
    tts = SilentTTS() if args.no_tts else load_tts(tts_cfg, os.environ)
    return CascadePipeline(stt, llm, tts, play=play,
                           fallback_reply=llm_cfg.get("fallback_reply", "Sorry, please say that again."),
                           no_speech_reply=llm_cfg.get("no_speech_reply", "I didn't hear anything."),
                           max_history_turns=llm_cfg.get("max_history_turns", 12),
                           stop_words=QUIT)


def use_project_venv() -> None:
    """Re-run under the repo's .venv when started with another Python: outside it,
    mlx-whisper and the in-process macOS voices are missing and everything is slower."""
    venv_python = ROOT / ".venv" / "bin" / "python"
    if (venv_python.exists() and Path(sys.prefix).resolve() != venv_python.parent.parent.resolve()
            and not os.environ.get("CASCADE_NO_REEXEC")):
        print(f"(not in the project venv; restarting with {venv_python.relative_to(ROOT)})",
              flush=True)
        os.environ["CASCADE_NO_REEXEC"] = "1"   # never loop
        os.execv(str(venv_python), [str(venv_python), "-m", "cascade.cli", *sys.argv[1:]])


def main(argv=None) -> int:
    if argv is None:
        use_project_venv()
    args = build_parser().parse_args(argv)
    args.tutor = not args.chat
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    load_env()
    try:
        pipeline = build_pipeline(args, cfg)
    except RuntimeError as exc:
        print(f"Can't start: {exc}. (--offline runs without an LLM.)", file=sys.stderr)
        return 2

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = runs / f"{stamp}.jsonl"
    audio_root = runs / stamp if args.keep_audio else None
    if args.tutor:
        from .tutor_cli import build_app

        try:
            app = build_app(args, cfg, ROOT, ask=fresh_input)
        except RuntimeError as exc:
            print(f"Can't start: {exc}. (--offline runs without an LLM.)", file=sys.stderr)
            return 2
        llm_name = app.factory.name
        warm_up(pipeline.tts, app.factory.model)
    else:
        llm_name = pipeline.llm.name
    print(f"[stt: {pipeline.stt.name} · llm: {llm_name} · tts: {pipeline.tts.name}]")
    print(f"[log: {log_path.relative_to(ROOT)}]  Say or type 'quit' to stop.")
    inputs = make_inputs(args)
    with log_path.open("a", encoding="utf-8") as log:
        conv = Conversation(pipeline, log, audio_root, echo_heard=not args.text)
        if not args.tutor:
            conv.run(inputs)
            return 0
        app.show_progress()
        lesson_id = args.lesson or (app.next_lesson().id if args.wav and app.next_lesson() else None)
        if lesson_id is None and not args.wav:
            lesson_id = app.menu()
        while lesson_id:
            brain = app.start_lesson(lesson_id)
            pipeline.llm = brain
            conv.run(inputs, greet=True, lesson=lesson_id)
            if not brain.done:
                print("Stopped before the end of the lesson: nothing was recorded.")
                break
            redo = app.after_lesson(lesson_id, brain)
            if redo:
                lesson_id = redo
                continue
            if args.lesson or args.wav:
                break
            app.show_progress()
            lesson_id = app.menu()
    return 0


def make_inputs(args):
    """One shared stream of learner inputs: wav files, typed lines or mic recordings.
    Typing or saying 'quit' ends the current conversation."""
    if args.wav:
        return iter(args.wav)

    def interactive():
        while True:
            try:
                if args.text:
                    line = input("you> ").strip()
                    if line.lower() in QUIT:
                        return
                    yield line
                else:
                    if fresh_input("Press Enter to talk (or type quit): ").strip().lower() in QUIT:
                        return
                    from .mic import SAMPLE_RATE, record_push_to_talk

                    samples = record_push_to_talk(ask=fresh_input)
                    if len(samples) < 0.3 * SAMPLE_RATE:
                        print("  (that was too short: press Enter, speak, then press Enter)")
                        continue
                    yield samples
            except (EOFError, KeyboardInterrupt):
                return

    return interactive()


class Conversation:
    """Runs turns through the pipeline and logs each one. Shared by free chat and by
    every tutor lesson, so turn numbers keep counting across lessons."""

    def __init__(self, pipeline: CascadePipeline, log, audio_root: Path | None, *,
                 echo_heard: bool = True) -> None:
        self.pipeline, self.log, self.audio_root = pipeline, log, audio_root
        self.echo_heard = echo_heard   # print what STT heard at once (not for typed input)
        self.turn = 0

    def _write(self, result: TurnResult, **extra) -> None:
        record = {"turn": self.turn, "stt": self.pipeline.stt.name,
                  "tts": self.pipeline.tts.name, **extra, **result.to_dict()}
        self.log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.log.flush()

    def run(self, inputs, *, greet: bool = False, lesson: str | None = None) -> None:
        extra = {"lesson": lesson} if lesson else {}
        if self.echo_heard:
            self.pipeline.on_heard = lambda t: print(you_line(t), flush=True)
        if greet:
            print("(preparing the lesson…)", flush=True)
            greeting = self.pipeline.greet()
            if greeting is not None:
                print(format_turn(0, greeting) + "\n", flush=True)
                self._write(greeting, greeting=True, **extra)
        for audio in inputs:
            self.turn += 1
            if self.audio_root is not None and not isinstance(audio, (str, Path)):
                from .mic import save_wav

                self.audio_root.mkdir(parents=True, exist_ok=True)
                save_wav(audio, self.audio_root / f"turn{self.turn}.wav")
            result = self.pipeline.turn(audio)
            if result.stopped:
                return
            print(format_turn(self.turn, result, heard_shown=self.echo_heard) + "\n", flush=True)
            self._write(result, **extra, **({"input_file": str(audio)} if isinstance(audio, Path) else {}))
            if result.done:
                return


if __name__ == "__main__":
    sys.exit(main())
