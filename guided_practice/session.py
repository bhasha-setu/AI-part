"""The guided-practice session: a turn-driven state machine.

Usage is deliberately not an ``input()`` loop, so the same code drives the app,
the tests and the demo:

    session = Session(scenario, "learner-1", course=course, logger=logger, ...)
    prompt = session.start()
    while not session.done:
        result = session.submit(transcript, why=False)
        prompt = result.prompt

Hint ladder per item (levels 0-3):
    0 situation   1 cue   2 first words   3 full model sentence (repeat).
The ladder has four rungs. That is the structure of the practice, not a tunable
value, so MAX_LEVEL is a code constant.

Rules, with their thresholds in the level config:
    * correct            -> record level; ask follow-up if any; next item.
    * surface_error      -> show correction, retry at SAME level
                            (max_retries_per_level, then advance).
    * meaning_error /
      off_topic          -> advance one hint level and re-present.
    * level 3 exhausted  -> outcome "not_achieved".

Pedagogical basis (see docs/reasoning_core.md): graduated hints and prompts
before answers follow the framework's section 6 and the prompts-vs-recasts
review; the retry budget is a hypothesis.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from .config import LanguageConfig, LevelConfig, UIStrings
from .content import Item, Scenario
from .course import Course, default_course
from .evaluator import EvalResult, Evaluator, RuleEvaluator, Verdict
from .normalise import normalise
from .turn_log import JsonlLogger, TurnRecord

# Structural: the ladder has four rungs (levels 0-3). Not a tunable threshold.
MAX_LEVEL = 3

_SLOT_RE = re.compile(r"\{(\w+)\}")


def render_prompt(item: Item, level: int, ui: Optional[UIStrings] = None) -> str:
    if level <= 0:
        return item.prompt
    if level == 1:
        return item.cue
    if level == 2:
        return item.first_words
    # level 3: show the model sentence to repeat, slots rendered as placeholders
    ui = ui or default_course().ui
    return _SLOT_RE.sub(lambda m: ui.slot_display.get(m.group(1), "[%s]" % m.group(1)),
                        item.model_answer)


@dataclass
class TurnResult:
    """What the caller shows after a submit."""

    verdict: str                 # verdict of the turn just submitted
    feedback: str                # one line to show the learner (may be empty)
    prompt: str                  # next prompt to present ("" when done)
    done: bool
    item_id: str
    hint_level: int              # level the submitted turn was at
    item_ended: bool = False     # True when this turn closed out an item
    outcome: Optional[str] = None     # "produced" | "repeated" | "not_achieved"
    independent: Optional[bool] = None
    awaiting_follow_up: bool = False


@dataclass
class ItemOutcome:
    item_id: str
    solved_level: Optional[int]
    outcome: str                 # "produced" | "repeated" | "not_achieved"
    independent: bool
    surface_errors_before_correct: int = 0


def _default_clock() -> Callable[[], datetime]:
    return lambda: datetime.now(timezone.utc)


class Session:
    def __init__(
        self,
        scenario: Scenario,
        learner_id: str,
        evaluator: Optional[Evaluator] = None,
        logger: Optional[JsonlLogger] = None,
        items: Optional[list[Item]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        session_id: Optional[str] = None,
        cap: Optional[int] = None,
        why_default: bool = False,
        course: Optional[Course] = None,
    ) -> None:
        self.course = course or default_course()
        self.scenario = scenario
        self.learner_id = learner_id
        self.evaluator = evaluator or RuleEvaluator()
        self.logger = logger
        self.level: LevelConfig = self.course.level
        self.ui: UIStrings = self.course.ui
        self.lang: LanguageConfig = self.course.lang
        self.cap = cap if cap is not None else self.level.max_items_per_session
        self.why_default = why_default
        self.clock = clock or _default_clock()
        self.session_id = session_id or ("sess-" + uuid.uuid4().hex[:8])

        chosen = items if items is not None else scenario.items
        self.items: list[Item] = list(chosen)[: self.cap]

        self._i = 0
        self._level = 0
        self._submitted_level = 0
        self._retries = 0
        self._surface_count = 0
        self._awaiting_follow_up = False
        self.started = False
        self.done = len(self.items) == 0
        self.outcomes: list[ItemOutcome] = []

    # --- public API ---------------------------------------------------------

    def start(self) -> str:
        self.started = True
        if self.done:
            return ""
        return render_prompt(self._item, self._level, self.ui)

    @property
    def current_item(self) -> Optional[Item]:
        return None if self.done else self._item

    @property
    def awaiting_follow_up(self) -> bool:
        return self._awaiting_follow_up

    @property
    def hint_level(self) -> int:
        """The level the next submitted turn will be judged at."""
        return self._level

    def submit(self, transcript: str, why: Optional[bool] = None,
               evaluation: Optional[EvalResult] = None) -> TurnResult:
        """Score one turn. ``evaluation`` lets a caller that judged the turn itself
        (the pedagogy server) supply the verdict and the feedback to show. The
        ladder, retries and outcomes are still decided here."""
        if self.done:
            raise RuntimeError("session already complete")
        if not self.started:
            raise RuntimeError("call start() before submit()")
        show_why = self.why_default if why is None else why

        item = self._item
        # The level the learner was at when this transcript was produced. All
        # TurnResults report this (not the post-advance level), and all logs use
        # it (we always write the log before mutating the level).
        self._submitted_level = self._level

        # A follow-up answer: logged, unscored, then move on. The item's outcome
        # was already recorded on the preceding correct turn.
        if self._awaiting_follow_up:
            self._write_log(item, transcript, "follow_up", feedback="",
                            turn_type="follow_up",
                            extra={"follow_up_question": item.follow_up or ""})
            self._awaiting_follow_up = False
            return self._advance_item(
                "follow_up", "", ended_id=item.id,
                ended_level=self._submitted_level, outcome=None, independent=None,
            )

        ev = evaluation if evaluation is not None else \
            self.evaluator.evaluate(normalise(transcript, self.lang), item)

        if ev.verdict == Verdict.CORRECT:
            return self._handle_correct(item, transcript, ev)
        if ev.verdict == Verdict.SURFACE_ERROR:
            return self._handle_surface(item, transcript, ev, show_why)
        # meaning_error or off_topic
        return self._handle_meaning(item, transcript, ev)

    # --- handlers -----------------------------------------------------------

    def _handle_correct(self, item, transcript, ev) -> TurnResult:
        level = self._submitted_level
        outcome = "repeated" if level >= MAX_LEVEL else "produced"
        independent = level <= self.level.independent_max_level
        # An evaluator may supply its own feedback; the rule evaluator leaves it empty.
        feedback = ev.feedback or (self.ui.repeat_ok if outcome == "repeated" else self.ui.correct)

        # This correct turn closes the item: log the outcome on it.
        self._write_log(item, transcript, Verdict.CORRECT, feedback=feedback,
                        outcome=outcome, surface_count=self._surface_count)
        self._record_outcome(item, solved_level=level, outcome=outcome,
                             independent=independent)

        if item.follow_up:
            self._awaiting_follow_up = True
            return TurnResult(
                verdict=Verdict.CORRECT, feedback=feedback, prompt=item.follow_up,
                done=False, item_id=item.id, hint_level=level,
                item_ended=False, outcome=outcome, independent=independent,
                awaiting_follow_up=True,
            )
        return self._advance_item(Verdict.CORRECT, feedback, ended_id=item.id,
                                  ended_level=level, outcome=outcome,
                                  independent=independent)

    def _handle_surface(self, item, transcript, ev, show_why) -> TurnResult:
        level = self._submitted_level
        self._surface_count += 1
        feedback = ev.feedback
        if show_why and ev.why:
            feedback = f"{feedback} {ev.why}"

        if self._retries < self.level.max_retries_per_level:
            self._retries += 1
            self._write_log(item, transcript, Verdict.SURFACE_ERROR,
                            feedback=feedback, matched_error_id=ev.matched_error_id)
            # Stay at the same level; ask for the corrected form.
            return TurnResult(
                verdict=Verdict.SURFACE_ERROR, feedback=feedback,
                prompt=self.ui.retry_prompt.format(prompt=render_prompt(item, level, self.ui)),
                done=False, item_id=item.id, hint_level=level,
            )
        # retries exhausted at this level -> advance the ladder
        return self._advance_after_fail(item, transcript, Verdict.SURFACE_ERROR,
                                        feedback, matched_error_id=ev.matched_error_id)

    def _handle_meaning(self, item, transcript, ev) -> TurnResult:
        # feedback_shown reflects what the learner actually sees.
        return self._advance_after_fail(item, transcript, ev.verdict, ev.feedback or self.ui.hint)

    # --- level / item transitions ------------------------------------------

    def _advance_after_fail(self, item, transcript, verdict, feedback,
                            matched_error_id="") -> TurnResult:
        """A failed turn (surface-exhausted / meaning / off_topic): climb the
        ladder, or close the item as not achieved at level 3."""
        level = self._submitted_level

        if level < MAX_LEVEL:
            # Log at the submission level, THEN advance.
            self._write_log(item, transcript, verdict, feedback=feedback,
                            matched_error_id=matched_error_id)
            self._level += 1
            self._retries = 0
            return TurnResult(
                verdict=verdict, feedback=feedback,
                prompt=render_prompt(item, self._level, self.ui),
                done=False, item_id=item.id, hint_level=level,
            )

        # At level 3 there's nowhere to climb: allow a couple of repeat retries.
        if self._retries < self.level.max_retries_per_level:
            self._retries += 1
            self._write_log(item, transcript, verdict, feedback=feedback,
                            matched_error_id=matched_error_id)
            return TurnResult(
                verdict=verdict, feedback=feedback,
                prompt=self.ui.repeat_prompt.format(prompt=render_prompt(item, level, self.ui)),
                done=False, item_id=item.id, hint_level=level,
            )

        # exhausted at level 3 -> not achieved (this turn closes the item)
        self._write_log(item, transcript, verdict, feedback=feedback,
                        matched_error_id=matched_error_id, outcome="not_achieved",
                        surface_count=self._surface_count)
        self._record_outcome(item, solved_level=None, outcome="not_achieved",
                             independent=False)
        return self._advance_item(verdict, feedback, ended_id=item.id,
                                  ended_level=level, outcome="not_achieved",
                                  independent=False)

    def _advance_item(self, verdict, feedback, ended_id, ended_level,
                      outcome, independent) -> TurnResult:
        """Move to the next item (or finish). Does NOT log; the closing turn's
        log was already written by the caller."""
        self._i += 1
        self._level = 0
        self._retries = 0
        self._surface_count = 0
        self._awaiting_follow_up = False

        if self._i >= len(self.items):
            self.done = True
            next_prompt = ""
        else:
            next_prompt = render_prompt(self._item, self._level, self.ui)
        return TurnResult(
            verdict=verdict, feedback=feedback, prompt=next_prompt, done=self.done,
            item_id=ended_id, hint_level=ended_level, item_ended=True,
            outcome=outcome, independent=independent,
        )

    # --- helpers ------------------------------------------------------------

    @property
    def _item(self) -> Item:
        return self.items[self._i]

    def _record_outcome(self, item, solved_level, outcome, independent) -> None:
        self.outcomes.append(ItemOutcome(
            item_id=item.id, solved_level=solved_level, outcome=outcome,
            independent=independent,
            surface_errors_before_correct=self._surface_count,
        ))

    def _write_log(self, item, transcript, verdict, feedback="",
                   turn_type="scored", matched_error_id="", outcome="",
                   surface_count=0, extra=None) -> None:
        if self.logger is None:
            return
        self.logger.log(TurnRecord(
            session_id=self.session_id,
            scenario=self.scenario.id,
            learner_id=self.learner_id,
            item_id=item.id,
            hint_level=self._level,
            transcript_raw=transcript,
            transcript_normalised=normalise(transcript, self.lang),
            verdict=verdict,
            feedback_shown=feedback,
            timestamp=self.clock().isoformat(),
            turn_type=turn_type,
            matched_error_id=matched_error_id,
            outcome=outcome,
            surface_errors_before_correct=surface_count,
            extra=extra or {},
            course_id=self.course.course_id,
        ))
