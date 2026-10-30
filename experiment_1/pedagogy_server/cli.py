"""The learner-facing CLI: the whole core loop, typed, with the real LLM pieces.

    1. "What do you want to learn?"  -> the generator plans 2-3 lesson options
    2. the learner picks one          -> an existing lesson, or a generated one (checked)
    3. the lesson runs                -> intro, dialogue, phrases, guided practice (judged),
                                         roleplay (LLM partner), summary
    4. state persists in --state-dir  -> generated lessons and spaced review carry over

    python3 -m pedagogy_server.cli                               # Gemini everywhere
    python3 -m pedagogy_server.cli --offline                     # no network: stub judge,
                                                                 # scripted partner, no generator
    python3 -m pedagogy_server.cli --scenario ordering_food      # skip planning
    python3 -m pedagogy_server.cli --show-decisions              # print why after each turn

In a lesson: Enter continues a presentation stage, ``:frame`` shows the server state,
``:quit`` stops. The decision log is ``<state-dir>/decisions.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, TextIO

from guided_practice.config import DEFAULT_COURSE, PROJECT_ROOT
from guided_practice.course import load_course
from guided_practice.lesson import STAGE_GUIDED_PRACTICE

from .generator import GenerationFailed, learner_course, make_generator
from .judge import JUDGE_KINDS, JudgeConfigError, make_judge
from .llm import LLMConfigError, load_dotenv
from .pipeline import PedagogyServer
from .roleplay_llm import make_roleplay

QUIT = (":quit", ":q")
FRAME = ":frame"
DEFAULT_STATE = PROJECT_ROOT / "runs" / "cli"


class Quit(Exception):
    pass


class CLI:
    def __init__(self, course, learner_id: str, state_dir: Path, judge, generator, roleplay_kind: str,
                 ask: Callable[[str], str], out: TextIO, show_decisions: bool = False) -> None:
        self.course = course
        self.learner_id = learner_id
        self.state_dir = state_dir
        self.judge = judge
        self.generator = generator
        self.roleplay_kind = roleplay_kind
        self.ask = ask
        self.out = out
        self.show_decisions = show_decisions

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def read(self, prompt: str) -> str:
        line = self.ask(prompt)
        if line.strip() in QUIT:
            raise Quit
        return line

    # --- choosing a lesson ---------------------------------------------------------------

    def choose(self) -> Optional[str]:
        """Returns a scenario id, or None if the learner gave up."""
        while True:
            request = self.read("\nWhat do you want to learn today? > ").strip()
            plan = self.generator.plan(request, self.course)
            if plan.note:
                self.say(plan.note)
            if self.show_decisions and plan.rejections:
                self.say(f"  [plan {plan.source}: {', '.join(plan.rejections[:6])}]")
            for n, opt in enumerate(plan.options, 1):
                tag = "existing lesson" if opt.scenario_id else "new lesson"
                self.say(f"\n  {n}. {opt.title}  ({tag}, ~{opt.est_minutes} min)")
                self.say(f"     You will be able to: {opt.can_do}")
                if opt.situation:
                    self.say(f"     Situation: {opt.situation}")
                if opt.phrases_preview:
                    self.say(f"     You'll say things like: " + " / ".join(opt.phrases_preview))
                if opt.why:
                    self.say(f"     Why: {opt.why}")
            pick = self.read("\nPick a number, or press Enter to ask again > ").strip()
            if not pick.isdigit() or not 1 <= int(pick) <= len(plan.options):
                continue
            opt = plan.options[int(pick) - 1]
            if opt.scenario_id:
                return opt.scenario_id
            self.say("\nBuilding your lesson... (this can take up to a minute)")
            try:
                built = self.generator.build(opt, self.course, request)
            except GenerationFailed as exc:
                self.say("Sorry, I couldn't build a lesson I'd trust for that. "
                         "Try another option, or one of the existing lessons.")
                if self.show_decisions:
                    for entry in exc.log:
                        self.say(f"  [attempt {entry['attempt']}: {'; '.join(entry['problems'][:5])}]")
                continue
            self.say(f"Ready: {built.title} ({built.item_count} things to say).")
            if self.show_decisions:
                self.say(f"  [built in {built.attempts} attempt(s); saved to {built.path}]")
                for entry in built.log[:-1]:
                    self.say(f"  [attempt {entry['attempt']} rejected: {'; '.join(entry['problems'][:5])}]")
            return built.scenario_id

    # --- running a lesson ---------------------------------------------------------------

    def lesson(self, scenario_id: str) -> PedagogyServer:
        server = PedagogyServer(self.course, scenario_id, self.learner_id, self.state_dir,
                                self.judge, roleplay=make_roleplay(self.roleplay_kind, self.course))
        self._told_skip = False
        reply = server.start()
        self._show(server, reply)
        while not server.done:
            line = self.read("you> " if reply.expects == "answer" else "(Enter to continue) ")
            if line.strip() == FRAME:
                self.say(str(asdict(server.frame())))
                continue
            if reply.expects == "answer" and not line.strip():
                continue
            reply = server.handle(line)
            self._show(server, reply)
        self.summary(server)
        return server

    def _show(self, server, reply) -> None:
        if not self._told_skip and any(r.stage == STAGE_GUIDED_PRACTICE and r.status == "skipped"
                                       for r in server.lesson.records):
            self._told_skip = True
            self.say("\n(Nothing in this lesson is due for practice yet: spaced review brings "
                     "these sentences back on a later day. Going straight to the conversation.)")
        self.say(f"\n[{reply.stage}] {reply.text}")
        if self.show_decisions:
            last = server.log.read(server.log.path)
            turn = next((r for r in reversed(last) if r["kind"] == "turn"), None)
            if turn:
                judge = turn["data"].get("judge")
                partner = turn["data"].get("partner")
                detail = turn["reason"]
                if judge and (judge["rejections"] or judge["guardrail_actions"]):
                    detail += f" | rejected: {judge['rejections']} | guardrails: {judge['guardrail_actions']}"
                if partner and partner.get("rejections"):
                    detail += f" | partner rejected: {partner['rejections']}"
                self.say(f"  [{detail}]")

    def summary(self, server) -> None:
        lesson = server.lesson
        a = lesson.assessment
        self.say("\n--- How you did ---")
        if a is None:
            self.say(f"Lesson {lesson.status}.")
            return
        if a.task_completion.value is not None:
            self.say(f"Said correctly: {a.task_completion.value:.0%} of the items")
        if a.independence.value is not None:
            self.say(f"From memory (little or no help): {a.independence.value:.0%}")
        goal = a.improvement_goal.value
        if goal:
            why = next((ce.why for it in lesson.scenario.items for ce in it.common_errors
                        if ce.id == goal and ce.why), "")
            self.say(f"Work on next: {why or goal}")
        turns = getattr(server.roleplay.session, "learner_turns", 0)
        if turns:
            self.say(f"Conversation: you spoke {turns} times without hints.")
        self.say(f"(decision log: {server.log.path})")


def build_parts(args):
    course = learner_course(load_course(args.course), args.state_dir, args.learner)
    judge = make_judge(args.judge, course)
    generator = make_generator(args.generator)
    return course, judge, generator


def main(argv=None, ask: Callable[[str], str] = input, out: TextIO = sys.stdout) -> int:
    load_dotenv(PROJECT_ROOT / ".env")   # keys only; never printed
    load_dotenv(PROJECT_ROOT.parent / ".env")   # shared key at the repo root
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--course", default=DEFAULT_COURSE)
    parser.add_argument("--learner", default="me")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--judge", choices=JUDGE_KINDS, default="gemini")
    parser.add_argument("--generator", choices=("gemini", "off"), default="gemini")
    parser.add_argument("--roleplay", choices=("gemini", "scripted", "stub"), default="gemini")
    parser.add_argument("--offline", action="store_true",
                        help="no network: stub judge, scripted partner, generator off")
    parser.add_argument("--scenario", help="skip planning and run this lesson")
    parser.add_argument("--show-decisions", action="store_true")
    args = parser.parse_args(argv)
    if args.offline:
        args.judge, args.generator, args.roleplay = "stub", "off", "scripted"

    try:
        course, judge, generator = build_parts(args)
        make_roleplay(args.roleplay, course)       # fail early on a missing key
    except (JudgeConfigError, LLMConfigError) as exc:
        print(f"error: {exc}. Put the key in .env, or run with --offline.", file=sys.stderr)
        return 2
    cli = CLI(course, args.learner, args.state_dir, judge, generator, args.roleplay,
              ask, out, show_decisions=args.show_decisions)
    try:
        scenario_id = args.scenario or cli.choose()
        if scenario_id:
            cli.lesson(scenario_id)
    except (Quit, EOFError, KeyboardInterrupt):
        cli.say("\nBye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
