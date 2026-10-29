"""Lesson loop: six stages in order, with a policy deciding what follows each one.

Stages:
    intro           presentation: what the learner will be able to do
    model_dialogue  presentation: a short written conversation (audio-ready data)
    useful_language presentation: phrases the practice needs
    guided_practice recall from memory, via the hint ladder (Session)
    roleplay        STUB
    assessment      STUB for clarity, grammar, vocabulary, comprehension

Presentation stages wait for a ``continue_()`` call. Guided practice waits for
``submit()``. Roleplay and assessment run as soon as they are reached.

Stages report outcomes; they don't judge them. After each stage the Lesson asks
``policy.after_stage(...)`` what to do. A stage that fails never advances the
lesson by itself. Here is how each decision is handled:
    advance      move on
    repeat       re-run the stage, up to ``max_stage_attempts``
    drop_easier  hold the lesson. Choosing easier content is Step 3's job.
    end          hold the lesson

Research basis: the lesson keeps two kinds of practice for different jobs. Drills
(guided recall here) raise accuracy, and task-based roleplay builds fluency
(Tomita 2025; Nguyen 2026; Nawaz et al. 2025). So drills come before roleplay, and
independence measured in drills is not fluency. See docs/reasoning_core.md, C7.

Time budgets are recorded, not enforced. An overrun sets ``over_budget`` and does
not interrupt the stage (see README decision 13 and docs/reasoning_core.md).

    lesson = Lesson(course, "introducing_yourself", "l1", learners_path, log_path,
                    clock=clock)
    view = lesson.start()
    while view.expects not in ("done", "held"):
        view = lesson.continue_() if view.expects == "continue" else lesson.submit(text)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .app import commit_outcomes, create_session
from .assessment import AssessmentSummary, Assessor, StubAssessor
from .course import Course
from .policy import HoldPolicy, Policy
from .roleplay import RoleplayStage, StubRoleplay
from .session import ItemOutcome
from .turn_log import JsonlLogger

STAGE_INTRO = "intro"
STAGE_MODEL_DIALOGUE = "model_dialogue"
STAGE_USEFUL_LANGUAGE = "useful_language"
STAGE_GUIDED_PRACTICE = "guided_practice"
STAGE_ROLEPLAY = "roleplay"
STAGE_ASSESSMENT = "assessment"

STAGE_ORDER = (
    STAGE_INTRO,
    STAGE_MODEL_DIALOGUE,
    STAGE_USEFUL_LANGUAGE,
    STAGE_GUIDED_PRACTICE,
    STAGE_ROLEPLAY,
    STAGE_ASSESSMENT,
)
PRESENTATION_STAGES = (STAGE_INTRO, STAGE_MODEL_DIALOGUE, STAGE_USEFUL_LANGUAGE)
AUTO_STAGES = (STAGE_ROLEPLAY, STAGE_ASSESSMENT)


@dataclass
class StageRecord:
    stage: str
    status: str             # completed | failed | skipped | stubbed
    attempt: int
    budget_seconds: int
    elapsed_seconds: float
    over_budget: bool
    detail: dict = field(default_factory=dict)


@dataclass
class LessonView:
    stage: str
    expects: str            # continue | answer | done | held
    lines: list = field(default_factory=list)
    prompt: str = ""
    feedback: str = ""
    item_id: str = ""
    follow_up: bool = False
    reason: str = ""


class Lesson:
    def __init__(
        self,
        course: Course,
        scenario_id: str,
        learner_id: str,
        learners_path: str | Path,
        log_path: str | Path,
        clock: Optional[Callable[[], datetime]] = None,
        policy: Optional[Policy] = None,
        roleplay: Optional[RoleplayStage] = None,
        assessor: Optional[Assessor] = None,
        lesson_id: str = "lesson",
    ) -> None:
        self.course = course
        self.scenario = course.load_scenario(scenario_id)
        self.learner_id = learner_id
        self.learners_path = Path(learners_path)
        self.logger = JsonlLogger(log_path)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.policy = policy or HoldPolicy()
        self.roleplay = roleplay or StubRoleplay()
        self.assessor = assessor or StubAssessor()
        self.lesson_id = lesson_id

        self.status = "not_started"     # not_started | in_progress | complete | held
        self.records: list[StageRecord] = []
        self.assessment: Optional[AssessmentSummary] = None
        self._index = 0
        self._attempts: dict[str, int] = {}
        self._stage_start: Optional[datetime] = None
        self._session = None
        self._learner = None
        self._guided_items: Optional[list] = None   # items of the first attempt
        self._open_ids: set[str] = set()            # items not achieved in the last attempt
        self._pending_outcomes: dict[str, ItemOutcome] = {}  # latest outcome per item, uncommitted
        self._roleplay = None                       # live roleplay session, if any
        self._last_session_id = ""
        self._last_outcomes: list[ItemOutcome] = []
        self._roleplay_transcript: list = []

    # --- public API ---------------------------------------------------------

    @property
    def current_stage(self) -> str:
        return STAGE_ORDER[self._index] if self._index < len(STAGE_ORDER) else "complete"

    def start(self) -> LessonView:
        if self.status != "not_started":
            raise RuntimeError(f"lesson already {self.status}")
        self.status = "in_progress"
        return self._enter_stage()

    def continue_(self) -> LessonView:
        self._require_in_progress()
        if self.current_stage not in PRESENTATION_STAGES:
            raise RuntimeError(f"stage '{self.current_stage}' does not take continue_()")
        return self._finish_stage("completed", {})

    def submit(self, text: str, why: Optional[bool] = None) -> LessonView:
        self._require_in_progress()
        if self.current_stage == STAGE_ROLEPLAY:
            return self._submit_roleplay(text)
        if self.current_stage != STAGE_GUIDED_PRACTICE:
            raise RuntimeError(f"stage '{self.current_stage}' does not take submit()")

        result = self._session.submit(text, why=why)
        if not self._session.done:
            return LessonView(
                stage=STAGE_GUIDED_PRACTICE, expects="answer",
                prompt=result.prompt, feedback=result.feedback,
                item_id=result.item_id, follow_up=self._session.awaiting_follow_up,
            )

        # The learner model is not updated here. _apply commits once, after the
        # final attempt, so a repeat doesn't double-count the Leitner update.
        attempt_outcomes = list(self._session.outcomes)
        for oc in attempt_outcomes:
            self._pending_outcomes[oc.item_id] = oc
        self._last_outcomes = list(self._pending_outcomes.values())
        self._last_session_id = self._session.session_id
        not_achieved = [oc.item_id for oc in attempt_outcomes if oc.outcome == "not_achieved"]
        self._open_ids = set(not_achieved)
        status = "failed" if not_achieved else "completed"
        detail = {
            "session_id": self._last_session_id,
            "items": [asdict(oc) for oc in attempt_outcomes],
            "not_achieved": not_achieved,
        }
        self._session = None
        return self._finish_stage(status, detail)

    # --- stage machinery ----------------------------------------------------

    def _require_in_progress(self) -> None:
        if self.status != "in_progress":
            raise RuntimeError(f"lesson is {self.status}")

    def _enter_stage(self) -> LessonView:
        stage = self.current_stage
        attempt = self._attempts.get(stage, 0) + 1
        self._attempts[stage] = attempt
        self._stage_start = self.clock()

        if stage in PRESENTATION_STAGES:
            return self._presentation_view(stage)
        if stage == STAGE_GUIDED_PRACTICE:
            return self._start_guided(attempt)
        return self._run_auto(stage)

    def _presentation_view(self, stage: str) -> LessonView:
        ui = self.course.ui
        heading = ui.lesson_headings[stage]
        if stage == STAGE_INTRO:
            body = [self.scenario.can_do]
        elif stage == STAGE_MODEL_DIALOGUE:
            body = [f"{t.speaker}: {t.text}" for t in self.scenario.model_dialogue]
        else:
            body = [f"{p.phrase} — {p.meaning}" for p in self.scenario.useful_language]
        return LessonView(stage=stage, expects="continue",
                          lines=[heading, *body, ui.lesson_continue])

    def _start_guided(self, attempt: int) -> LessonView:
        session_id = f"{self.lesson_id}-gp{attempt}"
        # A repeat re-presents only the items not achieved in the last attempt.
        # Asking the scheduler again would find nothing due right after the first
        # attempt and skip the practice.
        retry_items = None
        if self._guided_items is not None:
            retry_items = [it for it in self._guided_items if it.id in self._open_ids]
        session, learner = create_session(
            self.scenario.id, self.learner_id, self.learners_path,
            now=self.clock(), logger=self.logger, clock=self.clock,
            session_id=session_id, course=self.course, items=retry_items,
        )
        if self._guided_items is None:
            self._guided_items = list(session.items)
        if not session.items:
            # Nothing due and nothing new. Practice is skipped, not failed.
            self._session = None
            self._last_session_id = session_id
            return self._finish_stage("skipped", {"reason": "no items selected"})

        self._session, self._learner = session, learner
        prompt = session.start()
        return LessonView(
            stage=STAGE_GUIDED_PRACTICE, expects="answer", prompt=prompt,
            item_id=session.current_item.id,
            lines=[self.course.ui.lesson_headings[STAGE_GUIDED_PRACTICE]],
        )

    def _run_auto(self, stage: str) -> LessonView:
        if stage == STAGE_ROLEPLAY:
            self._roleplay = self.roleplay.start(self.scenario, self.learner_id, list(self.records))
            if not self._roleplay.done:
                return LessonView(stage=STAGE_ROLEPLAY, expects="answer",
                                  prompt=self._roleplay.prompt,
                                  lines=[self.course.ui.lesson_headings[STAGE_ROLEPLAY]])
            return self._finish_roleplay()

        records = JsonlLogger.read(self.logger.path)
        transcript = self._roleplay_transcript
        summary = self.assessor.assess(self._last_outcomes, records, self._last_session_id,
                                       roleplay_transcript=transcript)
        self.assessment = summary
        status = summary.status
        return self._finish_stage(status, {"summary": asdict(summary)})

    def _submit_roleplay(self, text: str) -> LessonView:
        prompt = self._roleplay.submit(text)
        if not self._roleplay.done:
            return LessonView(stage=STAGE_ROLEPLAY, expects="answer", prompt=prompt)
        return self._finish_roleplay()

    def _finish_roleplay(self) -> LessonView:
        result = self._roleplay.result()
        self._roleplay_transcript = list(result.transcript)
        self._roleplay = None
        status = "stubbed" if result.status == "stubbed" else "completed"
        return self._finish_stage(status, {"reason": result.reason,
                                           "turns": len(result.transcript)})

    def _finish_stage(self, status: str, detail: dict) -> LessonView:
        stage = self.current_stage
        elapsed = (self.clock() - self._stage_start).total_seconds()
        budget = self.course.level.stage_budgets_seconds[stage]
        record = StageRecord(
            stage=stage, status=status, attempt=self._attempts[stage],
            budget_seconds=budget, elapsed_seconds=elapsed,
            over_budget=elapsed > budget, detail=detail,
        )
        self.records.append(record)
        decision = self.policy.after_stage(stage, record, list(self.records))
        return self._apply(decision, stage)

    def _commit_guided(self) -> None:
        # Writes the learner-model update for every item practised this lesson,
        # once. Runs when the guided stage is final, not on each repeat.
        if not self._pending_outcomes:
            return
        commit_outcomes(list(self._pending_outcomes.values()), self._learner,
                        self.course, self.learners_path, self.clock())
        self._pending_outcomes = {}

    def _apply(self, decision, stage: str) -> LessonView:
        cap = self.course.level.max_stage_attempts
        if decision.action == "repeat" and self._attempts[stage] < cap:
            return self._enter_stage()

        self._commit_guided()

        if decision.action == "advance":
            self._index += 1
            if self._index >= len(STAGE_ORDER):
                self.status = "complete"
                return LessonView(stage="complete", expects="done", reason=decision.reason)
            return self._enter_stage()

        if decision.action == "repeat":
            return self._hold(stage, f"max_stage_attempts ({cap}) reached on {stage}")

        if decision.action == "drop_easier":
            return self._hold(stage, "drop_easier needs easier-content selection (Step 3); "
                                     "lesson held")

        return self._hold(stage, decision.reason)

    def _hold(self, stage: str, reason: str) -> LessonView:
        self.status = "held"
        return LessonView(stage=stage, expects="held", reason=reason)
