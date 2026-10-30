"""Integration: scheduler + learner model + session, across two sessions."""

from datetime import datetime, timedelta, timezone

from guided_practice.app import create_session, finish_session
from guided_practice.config import DEFAULT_COURSE
from guided_practice.learner_model import LearnerModel
from guided_practice.scheduler import select_items

DAY1 = datetime(2025, 1, 6, 9, 0, 0, tzinfo=timezone.utc)
DAY2 = DAY1 + timedelta(days=1)


def _clock(base):
    counter = {"n": 0}

    def clock():
        counter["n"] += 1
        return base + timedelta(seconds=counter["n"])

    return clock


def test_review_scheduled_and_selected_next_session(tmp_path):
    learners = tmp_path / "learners.json"

    # Session 1: three items solved independently (level 0) and one dragged to
    # level 2. The level-2 item should be due sooner than the others.
    s1, learner1 = create_session(
        "introducing_yourself", "l1", learners, DAY1,
        clock=_clock(DAY1), session_id="s1",
    )
    s1.start()
    script = [
        "My name is Ravi",      # intro_name L0 correct
        "I'm fine, thank you",  # follow-up (unscored)
        "I'm from Dehradun",    # intro_origin L0 correct
        # intro_work: fail twice to reach L2, then correct at L2
        "I do not know",        # L0 meaning -> L1
        "still nothing",        # L1 meaning -> L2
        "I am a student",       # L2 correct
        "I like cricket",       # intro_interest L0 correct
    ]
    for text in script:
        if s1.done:
            break
        s1.submit(text)
    assert s1.done

    finish_session(s1, learner1, learners, DAY1)

    # The learner model now has a record for every attempted item.
    # State is namespaced by course, so reload with the course that wrote it.
    reloaded = LearnerModel.load(learners, "l1", course_id=DEFAULT_COURSE)
    assert reloaded.state_for("intro_work") is not None
    assert reloaded.state_for("intro_name") is not None

    # intro_work (level 2 -> box 1 -> due in 1 day) is due on DAY2;
    # the independently-solved items (box 2 -> due in 2 days) are NOT.
    scenario_items = s1.scenario.items
    due_day2 = select_items(scenario_items, reloaded, DAY2, cap=5)
    due_ids = [it.id for it in due_day2]
    assert "intro_work" in due_ids
    assert due_ids[0] == "intro_work"            # the due review comes first
    assert "intro_name" not in due_ids           # not yet due (solved cleanly)


def test_create_session_respects_cap(tmp_path):
    learners = tmp_path / "learners.json"
    session, _ = create_session(
        "ordering_food", "l2", learners, DAY1, cap=2, clock=_clock(DAY1),
    )
    assert len(session.items) == 2
