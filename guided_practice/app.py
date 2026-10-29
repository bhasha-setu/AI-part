"""Session factory + teardown: wires the scheduler and learner model to a
Session, as the spec's ``session = (scenario_id, learner_id, cap)`` implies.

    session, learner = create_session("introducing_yourself", "l1",
                                       learners_path, now)
    prompt = session.start()
    ... drive turns ...
    finish_session(session, learner, learners_path, now)   # persists reviews
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .course import Course, default_course
from .evaluator import Evaluator, RuleEvaluator
from .learner_model import LearnerModel
from .scheduler import select_items
from .session import Session
from .turn_log import JsonlLogger


def create_session(
    scenario_id: str,
    learner_id: str,
    learners_path: str | Path,
    now: datetime,
    cap: Optional[int] = None,
    logger: Optional[JsonlLogger] = None,
    clock: Optional[Callable[[], datetime]] = None,
    evaluator: Optional[Evaluator] = None,
    session_id: Optional[str] = None,
    course: Optional[Course] = None,
    items: Optional[list] = None,
) -> tuple[Session, LearnerModel]:
    """Build a session. Items come from the scheduler unless given explicitly,
    which a lesson uses to repeat the same items after a repeat decision."""
    course = course or default_course()
    scenario = course.load_scenario(scenario_id)
    learner = LearnerModel.load(learners_path, learner_id, course_id=course.course_id)
    cap = cap if cap is not None else course.level.max_items_per_session
    if items is None:
        items = select_items(scenario.items, learner, now, cap=cap)
    else:
        items = list(items)[:cap]
    session = Session(
        scenario=scenario,
        learner_id=learner_id,
        evaluator=evaluator or RuleEvaluator(),
        logger=logger,
        items=items,
        clock=clock,
        session_id=session_id,
        cap=cap,
        course=course,
    )
    return session, learner


def finish_session(
    session: Session,
    learner: LearnerModel,
    learners_path: str | Path,
    now: datetime,
) -> None:
    """Fold the session's per-item outcomes into the learner model and save."""
    commit_outcomes(session.outcomes, learner, session.course, learners_path, now)


def commit_outcomes(
    outcomes: list,
    learner: LearnerModel,
    course: Course,
    learners_path: str | Path,
    now: datetime,
) -> None:
    """Write per-item outcomes into the learner model and save. A lesson calls
    this once, after its final attempt, so a repeat doesn't update twice."""
    for oc in outcomes:
        learner.record_result(
            oc.item_id,
            solved_level=oc.solved_level,
            achieved=(oc.outcome != "not_achieved"),
            now=now,
            level=course.level,
        )
    learner.save(learners_path)
