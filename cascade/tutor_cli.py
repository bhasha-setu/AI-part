"""Tutor mode of the CLI: who is learning, what comes next, then the lessons.

    Your username: himanshu
    Welcome back, himanshu.   Hindi help: english only
    Stage 1 · Basics (1/4 done)
      ✓ complete     Introducing yourself
        not started  Family and home   <- next
      ...
    [Enter] continue · [r] redo a lesson · [s] choose a stage · [q] quit

After a lesson the summary is saved. A lesson is complete when every phrase was
achieved; otherwise the learner can redo it now, move on anyway, or stop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .learners import Curriculum, LearnerProgress, LessonRef, status_mark, write_json_atomic
from .tutor_brain import Profiles, TutorFactory, hindi_help_label, use_tutor_packages

use_tutor_packages()

from guided_practice.course import load_course  # noqa: E402

QUIT = {"q", "quit", "exit"}
CONTINUE = {"", "c", "continue", "n", "next"}


class TutorApp:
    def __init__(self, curriculum: Curriculum, factory: TutorFactory, progress: LearnerProgress,
                 *, ask: Callable[[str], str] = input, out: Callable[[str], None] = print,
                 hindi_help: Optional[int] = None, generator=None) -> None:
        self.curriculum = curriculum
        self.factory = factory
        self.progress = progress
        self.course = load_course(curriculum.course_id)
        self.ask, self.out = ask, out
        self.hindi_help = hindi_help
        self.variant: Optional[dict] = None   # how the current lesson was prepared
        self.generator = generator            # extra practice (cascade/extras.py); None offline
        self.extras: dict = {}                # "extra:<id>" -> generated Scenario

    # --- what the learner sees --------------------------------------------------------

    def title(self, lesson_id: str) -> str:
        try:
            return self.curriculum.title(lesson_id)
        except FileNotFoundError:
            return lesson_id.replace("_", " ").capitalize() + " (not written yet)"

    def available(self, lesson_id: str) -> bool:
        return self.curriculum.path(lesson_id).exists()

    def next_lesson(self) -> Optional[LessonRef]:
        return next((ref for ref in self.curriculum.lessons
                     if not self.progress.is_done(ref.id) and self.available(ref.id)), None)

    def hindi_level(self) -> int:
        if self.hindi_help is not None:
            return self.hindi_help
        profiles = Profiles(self.progress.dir / "profiles.json")
        return profiles.stage(self.progress.username, self.course.course_id,
                              self.factory.cfg.start_stage)

    def show_progress(self) -> None:
        p = self.progress
        hello = f"Welcome, {p.username}! New profile created." if p.is_new else f"Welcome back, {p.username}."
        self.out(f"\n{hello}   {hindi_help_label(self.factory.cfg, self.hindi_level())}")
        nxt = self.next_lesson()
        for n, stage in enumerate(self.curriculum.stages, 1):
            refs = self.curriculum.stage_lessons(n)
            done = sum(p.is_done(r.id) for r in refs)
            self.out(f"Stage {n} · {stage['name']} ({done}/{len(refs)} done)")
            for r in refs:
                marker = "   <- next" if nxt and r.id == nxt.id else ""
                self.out(f"  {status_mark(p.status(r.id)):<14} {self.title(r.id)}{marker}")

    # --- choosing a lesson --------------------------------------------------------------

    def menu(self) -> Optional[str]:
        """Returns the lesson id to run, or None to quit."""
        while True:
            nxt = self.next_lesson()
            cont = (f"[Enter] continue: {self.title(nxt.id)} ({nxt.label})" if nxt
                    else "All lessons are done!")
            choice = self.ask(f"\n{cont}\n[r] redo a lesson · [s] choose a stage · [x] extra practice · [q] quit\n> ").strip().lower().rstrip("\\")
            if choice in QUIT:
                return None
            if choice in CONTINUE and nxt:
                return nxt.id
            if choice == "r":
                picked = self._pick([r for r in self.curriculum.lessons
                                     if self.progress.status(r.id) and self.available(r.id)],
                                    "Redo which lesson?")
                if picked:
                    return picked
            elif choice == "s":
                picked = self._pick_stage()
                if picked:
                    return picked
            elif choice == "x":
                picked = self.extra_practice()
                if picked:
                    return picked
            elif choice not in CONTINUE:
                self.out(f"Didn't catch {choice!r}: press Enter, or type r, s, x or q.")

    # --- extra practice ---------------------------------------------------------------------

    def current_stage(self) -> int:
        nxt = self.next_lesson()
        return nxt.stage if nxt else len(self.curriculum.stages)

    def extra_practice(self) -> Optional[str]:
        """Plan new lessons in a stage, build the chosen one. Returns "extra:<id>"."""
        from .extras import GenerationFailed, extra_request, hard_points, learner_library

        if self.generator is None:
            self.out("Extra practice needs the LLM (it isn't available offline).")
            return None
        default = self.current_stage()
        answer = self.ask(f"Extra practice for which stage? (1-{len(self.curriculum.stages)}, "
                          f"Enter = {default}) ").strip()
        stage = int(answer) if answer.isdigit() and 1 <= int(answer) <= len(self.curriculum.stages) else default
        refs = [r for r in self.curriculum.stage_lessons(stage) if self.available(r.id)]
        scenarios = {r.id: self.curriculum.load(r.id, self.course.lang) for r in refs}
        hard = hard_points(self.progress, [r.id for r in refs], scenarios)
        request = extra_request(self.curriculum.stages[stage - 1], hard,
                                [s.title for s in scenarios.values()])
        library = learner_library(self.course, self.progress)
        self.out("Finding new lessons for you…")
        plan = self.generator.plan(request, library)
        options = [o for o in plan.options if not o.scenario_id]
        if not options:
            self.out("Couldn't plan a new lesson just now. Try again later.")
            return None
        for i, o in enumerate(options, 1):
            self.out(f"  {i}. {o.title}: {o.can_do}")
        pick = self.ask("Which one? (number, Enter to go back) ").strip()
        if not (pick.isdigit() and 1 <= int(pick) <= len(options)):
            return None
        self.out("Writing and checking the lesson (this can take up to a minute)…")
        try:
            built = self.generator.build(options[int(pick) - 1], library, request)
        except GenerationFailed as exc:
            self.out(f"The new lesson didn't pass the checks ({exc}). Try another one.")
            return None
        key = f"extra:{built.scenario_id}"
        self.extras[key] = (library.load_scenario(built.scenario_id), stage)
        return key

    def _pick(self, refs: list, question: str) -> Optional[str]:
        if not refs:
            self.out("Nothing to choose yet.")
            return None
        for i, r in enumerate(refs, 1):
            self.out(f"  {i}. {self.title(r.id)} ({r.label}) {status_mark(self.progress.status(r.id)).strip()}")
        answer = self.ask(f"{question} (number, Enter to go back) ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(refs):
            return refs[int(answer) - 1].id
        return None

    def _pick_stage(self) -> Optional[str]:
        for n, stage in enumerate(self.curriculum.stages, 1):
            self.out(f"  {n}. {stage['name']}: {stage['goal']}")
        answer = self.ask("Which stage? (number, Enter to go back) ").strip()
        if not (answer.isdigit() and 1 <= int(answer) <= len(self.curriculum.stages)):
            return None
        refs = [r for r in self.curriculum.stage_lessons(int(answer)) if self.available(r.id)]
        if not refs:
            self.out("That stage has no lessons yet.")
            return None
        first_open = next((r for r in refs if not self.progress.is_done(r.id)), refs[0])
        return first_open.id

    # --- running one lesson ---------------------------------------------------------------

    def start_lesson(self, lesson_id: str):
        """A first attempt uses the lesson as written. A redo gets a fresh version shaped
        by the earlier attempts (cascade/variants.py)."""
        from .variants import make_variant

        if lesson_id in self.extras:
            scenario, stage = self.extras[lesson_id]
            self.variant = None
            brain = self.factory.lesson(scenario, self.course, learner_id=self.progress.username,
                                        state_dir=self.progress.dir, hindi_help=self.hindi_help)
            self.out(f"\n[{scenario.title} · extra practice, stage {stage} · {brain.hindi_help_label}]")
            return brain
        ref = self.curriculum.ref(lesson_id)
        scenario = self.curriculum.load(lesson_id, self.course.lang)
        attempts = self.progress.attempts(lesson_id)
        focus, self.variant = "", None
        if attempts:
            self.out("Preparing a fresh version of this lesson, based on your last attempt…")
            scenario, focus, meta = make_variant(scenario, attempts, self.factory.model,
                                                 self.course.lang)
            name = f"{lesson_id}-{len(attempts) + 1}"
            write_json_atomic(self.progress.dir / "variants" / f"{name}.json",
                              {"lesson": lesson_id, **meta, "focus": focus,
                               "prompts": {it.id: it.prompt for it in scenario.items}})
            self.variant = {"name": name, "source": meta["source"], "setting": meta.get("setting")}
            if meta["source"] == "llm":
                self.out(f"New situation: {meta['setting']}")
            else:
                self.out("(Same situations as before; the phrases you missed come first.)")
        brain = self.factory.lesson(scenario, self.course, learner_id=self.progress.username,
                                    state_dir=self.progress.dir, hindi_help=self.hindi_help,
                                    focus=focus)
        self.out(f"\n[{scenario.title} · {ref.label} · {brain.hindi_help_label}]")
        return brain

    def after_lesson(self, lesson_id: str, brain) -> Optional[str]:
        """Save the result. Returns the lesson to run next (a redo), or None."""
        report = brain.finish()
        attempt = self.progress.record(lesson_id, brain.summary, variant=self.variant)
        self.out("--- How you did ---")
        self.out("\n".join(report))
        if lesson_id in self.extras:
            self.out(f"Extra practice done: {attempt['achieved']} of {attempt['total']} phrases achieved.")
            return None
        if attempt["all_achieved"]:
            self.out(f"Lesson complete! All {attempt['total']} phrases achieved.")
            return None
        self.out(f"{attempt['achieved']} of {attempt['total']} phrases achieved. To complete the "
                 "lesson, every phrase is needed.")
        choice = self.ask("[r] redo it now · [m] mark complete and move on · [Enter] stop for now\n> ").strip().lower()
        if choice == "r":
            return lesson_id
        if choice == "m":
            self.progress.move_on(lesson_id)
            self.out("Moved on. The missed phrases will come back in later review.")
        return None


def build_app(args, cfg: dict, root: Path, *, ask=input, out=print) -> TutorApp:
    curriculum = Curriculum(root / "curriculum" / "english_for_hindi" / "curriculum.yaml")
    factory = TutorFactory(cfg.get("tutor", {}), cfg["llm"], offline=args.offline)
    username = args.user or ask("Your username: ")
    while True:
        try:
            progress = LearnerProgress(root / "runs" / "learners", username)
            break
        except ValueError as exc:
            username = ask(f"{exc}. Your username: ")
    generator = None
    if not args.offline:
        from .extras import build_generator

        generator = build_generator(cfg["llm"].get("providers", []))
    return TutorApp(curriculum, factory, progress, ask=ask, out=out, hindi_help=args.hindi_help,
                    generator=generator)
