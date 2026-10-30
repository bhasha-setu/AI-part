"""Text terminal: type as the learner, read what the server would say.

This is the step-1 test harness for the voice loop. Typed lines go through the STT
stub and replies through the TTS stub, so the server sees exactly what it would see
from a laptop client. Press Enter (or type anything) to continue past a presentation
stage. ``:frame`` prints the server's frame; ``:quit`` stops.

    python3 -m pedagogy_server.terminal                         # stub judge
    python3 -m pedagogy_server.terminal --judge anthropic       # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, TextIO

from guided_practice.config import DEFAULT_COURSE, PROJECT_ROOT
from guided_practice.course import load_course

from .adapters import StubLaptopClient, StubSTT, StubTTS
from .judge import JUDGE_KINDS, JudgeConfigError, make_judge
from .llm import load_dotenv
from .pipeline import PedagogyServer

QUIT = (":quit", ":q")
FRAME = ":frame"


def run(client: StubLaptopClient, lines: Iterable[str], out: TextIO, show_frame: bool = False) -> int:
    """Drive a lesson from ``lines``. Returns the number of learner turns sent."""
    server = client.server
    reply = client.start()
    print(f"[{reply.stage}] {reply.text}", file=out)
    sent = 0
    for line in lines:
        line = line.rstrip("\n")
        if line.strip() in QUIT:
            break
        if line.strip() == FRAME:
            print(asdict(server.frame()), file=out)
            continue
        reply = client.speak(line.encode("utf-8"))
        sent += 1
        print(f"[{reply.stage}] {reply.text}", file=out)
        if show_frame:
            print(f"    frame: {asdict(server.frame())}", file=out)
        if server.done:
            break
    return sent


def _stdin_lines(prompt: str):
    while True:
        try:
            yield input(prompt)
        except EOFError:
            return


def main(argv=None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")   # keys only; never printed
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--course", default=DEFAULT_COURSE)
    parser.add_argument("--scenario", default=None, help="defaults to the course's first scenario")
    parser.add_argument("--learner", default="terminal-learner")
    parser.add_argument("--judge", choices=JUDGE_KINDS, default="stub")
    parser.add_argument("--state-dir", default=None, help="where logs and learner state go "
                        "(default: a new temp directory)")
    parser.add_argument("--show-frame", action="store_true")
    args = parser.parse_args(argv)

    course = load_course(args.course)
    scenario = args.scenario or course.scenario_ids()[0]
    state_dir = Path(args.state_dir or tempfile.mkdtemp(prefix="pedagogy-"))
    try:
        judge = make_judge(args.judge, course)
    except JudgeConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    server = PedagogyServer(course, scenario, args.learner, state_dir, judge)
    client = StubLaptopClient(server, StubSTT(), StubTTS())
    print(f"course={course.course_id} scenario={scenario} judge={judge.name} logs={state_dir}")
    run(client, _stdin_lines("you> "), sys.stdout, show_frame=args.show_frame)
    print(f"lesson {server.lesson.status}. decision log: {state_dir / 'decisions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
