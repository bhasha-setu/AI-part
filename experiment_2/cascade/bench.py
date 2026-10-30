"""Measure the cascade. Every number is a median over --runs runs, after one warm-up.

    python -m cascade.bench --stt mlx:small mlx:large-v3-turbo+small faster-whisper:small -- clips/*.wav
    python -m cascade.bench --tts macos-native macos-say elevenlabs
    python -m cascade.bench --pipeline -- clips/*.wav     # config.yaml's backends, nothing played
    python -m cascade.bench --list-nvidia-models

Synthetic clips (made with `say`) flatter STT; record real ones with
`python -m cascade.cli --keep-audio` and benchmark those.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

from .cli import EXPERIMENT_ROOT, build_parser, build_pipeline, load_env
from .stt import load_stt
from .tts import load_tts

TTS_SENTENCES = [("en", "That sounds so relaxing!"),
                 ("en", "What kind of tea do you like best, with milk or without?"),
                 ("hi", "मुझे चाय पीना पसंद है।"),
                 ("mixed", "In English you can say, मुझे चाय पसंद है, I like tea.")]


def median_ms(fn, runs: int) -> tuple[float, object]:
    fn()   # warm-up
    times, out = [], None
    for _ in range(runs):
        start = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), out


def bench_stt(specs: list[str], clips: list[Path], cfg: dict, runs: int) -> None:
    for spec in specs:
        engine, _, model = spec.partition(":")
        model, _, detect = model.partition("+")   # mlx:large-v3-turbo+small = detect with small
        stt = load_stt(dict(cfg["stt"], detect_model=detect or None),
                       override_engine=engine, override_model=model or None)
        print(f"\n{stt.name}")
        for clip in clips:
            ms, t = median_ms(lambda: stt.transcribe(clip), runs)
            print(f"  {clip.name:<20} {ms:6.0f} ms  [{t.language} {t.language_prob}] {t.text}")


def bench_tts(engines: list[str], cfg: dict, runs: int) -> None:
    for engine in engines:
        try:
            tts = load_tts(dict(cfg["tts"], engine=engine), os.environ)
        except RuntimeError as exc:
            print(f"\n{engine}: skipped ({exc})")
            continue
        print(f"\n{tts.name}  (ms until audio starts, nothing played)")
        for lang, text in TTS_SENTENCES:
            if tts.splits_by_script and lang == "mixed":
                continue
            starts = [tts.speak(text, lang if tts.splits_by_script else "mixed", play=False)
                      for _ in range(runs + 1)][1:]
            print(f"  {statistics.median(starts):6.0f} ms  {text}")


def bench_pipeline(clips: list[Path], cfg: dict, runs: int, delay: float) -> None:
    args = build_parser().parse_args(["--wav", *map(str, clips)])
    pipeline = build_pipeline(args, cfg, play=False)
    print(f"[stt: {pipeline.stt.name} · llm: {pipeline.llm.name} · tts: {pipeline.tts.name}]")
    rows = {"stt_ms": [], "llm_first_ms": [], "first_audio_ms": [], "total_ms": []}
    for r in range(runs):
        for clip in clips:
            pipeline.history.clear()
            result = pipeline.turn(clip)
            for key in rows:
                value = getattr(result.timing, key)
                if value is not None and not result.fallback_used:
                    rows[key].append(value)
            status = f"FALLBACK ({result.llm_error})" if result.fallback_used else result.llm
            print(f"  run {r + 1} {clip.name:<16} first audio "
                  f"{result.timing.first_audio_ms or 0:6.0f} ms · {status} · {result.reply_text[:60]}")
            time.sleep(delay)   # free-tier keys rate-limit fast calls
    print("\nmedian over turns that got a real LLM reply:")
    for key, values in rows.items():
        print(f"  {key:<15} {statistics.median(values):6.0f} ms  (n={len(values)})" if values
              else f"  {key:<15}      -")


def list_nvidia_models() -> None:
    import httpx

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("NVIDIA_API_KEY is not set in the repo-root .env")
        return
    r = httpx.get("https://integrate.api.nvidia.com/v1/models",
                  headers={"Authorization": f"Bearer {key}"}, timeout=15)
    r.raise_for_status()
    for m in sorted(m["id"] for m in r.json()["data"]):
        print(m)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stt", nargs="+", metavar="ENGINE:MODEL")
    parser.add_argument("--tts", nargs="+", metavar="ENGINE")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--list-nvidia-models", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between LLM turns")
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "config.yaml")
    parser.add_argument("clips", nargs="*", type=Path)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    load_env()
    if (args.stt or args.pipeline) and not args.clips:
        parser.error("--stt and --pipeline need audio clips")
    if args.list_nvidia_models:
        list_nvidia_models()
    if args.stt:
        bench_stt(args.stt, args.clips, cfg, args.runs)
    if args.tts:
        bench_tts(args.tts, cfg, args.runs)
    if args.pipeline:
        bench_pipeline(args.clips, cfg, args.runs, args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
