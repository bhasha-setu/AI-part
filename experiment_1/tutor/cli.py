"""Chat with the tutor in the terminal. Type in English, Hindi or a mix.

    python3 -m tutor.cli                          # Gemini (GEMINI_API_KEY in .env)
    python3 -m tutor.cli --offline                # no network: rule verdicts, fixed lines
    python3 -m tutor.cli --stage 0                # force a support stage (0 = Hindi first)
    python3 -m tutor.cli --debug                  # show verdict and hint level per turn

``:quit`` stops. State (support stage, spaced review, log) goes to --state-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional, TextIO

from guided_practice.config import DEFAULT_COURSE, PROJECT_ROOT
from guided_practice.course import load_course
from guided_practice.learner_model import LearnerModel

from pedagogy_server.llm import GeminiModel, LLMConfigError, load_dotenv

from .config import Profiles, TutorConfig, load_tutor_config
from .session import TutorSession

DEFAULT_STATE = PROJECT_ROOT / "runs" / "tutor"


def save_lesson(session: TutorSession, course, learner_id: str, state_dir: Path, now=None):
    """Spaced review for every phrase, and the support stage for the next lesson."""
    from .session import _now

    summary = session.summary()
    now = now or _now()
    lm_path = state_dir / "learners.json"
    lm = LearnerModel.load(lm_path, learner_id, course_id=course.course_id)
    for rec in session.records:
        lm.record_result(rec.item_id, rec.solved_level, rec.achieved, now, course.level)
    lm.save(lm_path)
    Profiles(state_dir / "profiles.json").set_stage(learner_id, course.course_id,
                                                     summary.stage_after)
    return summary


def run(scenario_id: Optional[str], learner_id: str, state_dir: Path, cfg: TutorConfig,
        model, ask: Callable[[str], str], out: TextIO, stage: Optional[int] = None,
        debug: bool = False, course_id: str = DEFAULT_COURSE):
    course = load_course(course_id)
    scenario = course.load_scenario(scenario_id or course.scenario_ids()[0])
    state_dir.mkdir(parents=True, exist_ok=True)
    profiles = Profiles(state_dir / "profiles.json")
    if stage is None:
        stage = profiles.stage(learner_id, course.course_id, cfg.start_stage)
    log_path = state_dir / "tutor_log.jsonl"

    def log(event: dict) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    session = TutorSession(scenario, cfg, model, stage, log=log)

    def show(reply) -> None:
        print(f"\ntutor> {reply.text}", file=out)
        if debug:
            d = reply.debug
            print(f"  [{d.get('phase')} {d.get('source')} verdict={d.get('verdict')} "
                  f"level={d.get('hint_level')}->{d.get('next_hint_level')} "
                  f"problems={d.get('problems')}]", file=out)

    print(f"[{scenario.title} · support stage {stage}: {cfg.stages[stage].name}]", file=out)
    reply = session.start()
    show(reply)
    while not reply.done:
        try:
            line = ask("you> ")
        except EOFError:
            return None
        if line.strip() in (":quit", ":q"):
            return None
        reply = session.turn(line)
        show(reply)

    summary = save_lesson(session, course, learner_id, state_dir)
    print("\n--- How you did ---", file=out)
    for rec in summary.items:
        how = ("from memory" if rec["achieved"] and rec["solved_level"] <= 1 and not rec["used_hindi"]
               else "with help" if rec["achieved"] else "not yet")
        extra = ", started in Hindi" if rec["used_hindi"] else ""
        print(f"  {rec['item_id']}: {how} (hint level {rec['solved_level']}{extra})", file=out)
    print(f"From memory: {summary.from_memory:.0%}", file=out)
    for r in summary.reviewed:
        print(f"  Worked on: \"{r['said']}\" -> \"{r['fixed']}\"", file=out)
    if summary.stage_after != summary.stage_before:
        word = "less" if summary.stage_after > summary.stage_before else "more"
        print(f"Next lesson: {word} Hindi support (stage {summary.stage_after}).", file=out)
    print(f"(log: {log_path})", file=out)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--course", default=DEFAULT_COURSE)
    p.add_argument("--scenario")
    p.add_argument("--learner", default="cli-learner")
    p.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    p.add_argument("--stage", type=int)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    cfg = load_tutor_config()
    model = None
    if not args.offline:
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(PROJECT_ROOT.parent / ".env")   # shared key at the repo root
        try:
            model = GeminiModel(cfg.model, thinking_level=cfg.thinking_level or None,
                                timeout_seconds=cfg.timeout_seconds,
                                max_output_tokens=cfg.max_output_tokens)
        except LLMConfigError as exc:
            print(f"{exc}. Use --offline to run without the model.", file=sys.stderr)
            return 2
    run(args.scenario, args.learner, args.state_dir, cfg, model, input, sys.stdout,
        stage=args.stage, debug=args.debug, course_id=args.course)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
