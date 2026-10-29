"""Lesson loop: stage order, gating, the policy hook, stubs and time budgets."""

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from guided_practice.assessment import AssessmentSummary
from guided_practice.config import DEFAULT_COURSE, load_level
from guided_practice.course import load_course
from guided_practice.learner_model import LearnerModel
from guided_practice import lesson as lesson_module
from guided_practice.lesson import Lesson
from guided_practice.policy import Decision, HoldPolicy

BASE = datetime(2025, 1, 6, 9, 0, 0, tzinfo=timezone.utc)
ASSESSED_FIELDS = {"task_completion", "clarity", "grammar", "vocabulary",
                   "comprehension", "independence", "improvement_goal"}


def _clock(step_seconds: float = 1.0):
    n = {"i": 0}

    def clock():
        n["i"] += 1
        return BASE + timedelta(seconds=step_seconds * n["i"])

    return clock


class ScriptedPolicy:
    """Returns the given decision the first time a (stage, attempt) is reached."""

    def __init__(self, decisions: dict):
        self.decisions = dict(decisions)
        self.seen = []

    def after_stage(self, stage, record, history):
        self.seen.append((stage, record.attempt, record.status))
        key = (stage, record.attempt)
        if key in self.decisions:
            return self.decisions.pop(key)
        return HoldPolicy().after_stage(stage, record, history)


def _lesson(tmp_path, policy=None, clock=None, course_id=DEFAULT_COURSE, scenario_id=None):
    course = load_course(course_id)
    scenario_id = scenario_id or course.scenario_ids()[0]
    return Lesson(course, scenario_id, "l1", tmp_path / "learners.json",
                  tmp_path / "log.jsonl", clock=clock or _clock(), policy=policy)


def _drive_to_end(lesson, answer=None, max_turns=200):
    """Advance through presentations; answer guided practice with model answers."""
    view = lesson.start()
    turns = 0
    while view.expects not in ("done", "held"):
        turns += 1
        assert turns < max_turns, "lesson did not terminate"
        if view.expects == "continue":
            view = lesson.continue_()
        elif view.follow_up:
            view = lesson.submit("ok")
        else:
            text = answer(lesson, view) if answer else _model(lesson, view)
            view = lesson.submit(text)
    return view


def _model(lesson, view):
    return lesson.scenario.item_by_id(view.item_id).model_answer.replace("{name}", "Ravi") \
        .replace("{place}", "Dehradun").replace("{job}", "student") \
        .replace("{interest}", "cricket")


# --- ordering and gating ----------------------------------------------------

def test_full_lesson_completes_with_stubs_marked(tmp_path):
    lesson = _lesson(tmp_path)
    view = _drive_to_end(lesson)
    assert view.expects == "done"
    assert lesson.status == "complete"
    stages = [(r.stage, r.status) for r in lesson.records]
    assert stages == [
        ("intro", "completed"),
        ("model_dialogue", "completed"),
        ("useful_language", "completed"),
        ("guided_practice", "completed"),
        ("roleplay", "stubbed"),
        ("assessment", "stubbed"),
    ]


def test_presentation_stage_needs_continue_not_submit(tmp_path):
    lesson = _lesson(tmp_path)
    view = lesson.start()
    assert view.stage == "intro" and view.expects == "continue"
    assert lesson.scenario.can_do in view.lines
    with pytest.raises(RuntimeError, match="does not take submit"):
        lesson.submit("hello")


def test_guided_practice_does_not_take_continue(tmp_path):
    lesson = _lesson(tmp_path)
    lesson.start()
    lesson.continue_()
    lesson.continue_()
    lesson.continue_()
    with pytest.raises(RuntimeError, match="does not take continue_"):
        lesson.continue_()


def test_dialogue_turns_carry_audio_none(tmp_path):
    lesson = _lesson(tmp_path)
    assert lesson.scenario.model_dialogue
    assert all(turn.audio is None for turn in lesson.scenario.model_dialogue)


def test_policy_repeat_is_obeyed_for_guided_practice(tmp_path):
    policy = ScriptedPolicy({("guided_practice", 1): Decision("repeat", "try again")})
    lesson = _lesson(tmp_path, policy=policy)
    _drive_to_end(lesson)
    guided = [r for r in lesson.records if r.stage == "guided_practice"]
    assert [r.attempt for r in guided] == [1, 2]
    assert lesson.status == "complete"


def test_repeated_guided_practice_commits_the_learner_model_once(tmp_path, monkeypatch):
    calls = []
    commit = lesson_module.commit_outcomes

    def counting_commit(*args, **kwargs):
        calls.append(1)
        return commit(*args, **kwargs)

    monkeypatch.setattr(lesson_module, "commit_outcomes", counting_commit)
    policy = ScriptedPolicy({("guided_practice", 1): Decision("repeat", "try again")})
    lesson = _lesson(tmp_path, policy=policy)
    _drive_to_end(lesson)
    assert len(calls) == 1


def test_repeat_is_capped_by_max_stage_attempts(tmp_path):
    cap = load_level("A1").max_stage_attempts
    policy = ScriptedPolicy({("intro", n): Decision("repeat", "again") for n in range(1, cap + 1)})
    lesson = _lesson(tmp_path, policy=policy)
    view = lesson.start()
    while view.expects not in ("held", "done"):
        view = lesson.continue_() if view.expects == "continue" else lesson.submit(_model(lesson, view))
    assert lesson.status == "held"
    assert view.stage == "intro"
    assert f"max_stage_attempts ({cap})" in view.reason


def test_drop_easier_holds_and_does_not_advance(tmp_path):
    policy = ScriptedPolicy({("intro", 1): Decision("drop_easier", "too hard")})
    lesson = _lesson(tmp_path, policy=policy)
    lesson.start()
    view = lesson.continue_()
    assert view.expects == "held"
    assert lesson.status == "held"
    assert lesson.current_stage == "intro"
    assert "Step 3" in view.reason


def test_end_decision_holds_with_its_reason(tmp_path):
    policy = ScriptedPolicy({("intro", 1): Decision("end", "learner asked to stop")})
    lesson = _lesson(tmp_path, policy=policy)
    lesson.start()
    view = lesson.continue_()
    assert view.expects == "held"
    assert view.reason == "learner asked to stop"


def test_failed_guided_practice_holds_under_placeholder_policy(tmp_path):
    lesson = _lesson(tmp_path)
    view = lesson.start()
    lesson.continue_()
    lesson.continue_()
    lesson.continue_()
    turns = 0
    while view.expects != "held":
        turns += 1
        assert turns < 200, "guided practice did not end"
        view = lesson.submit("zzz qqq")
    assert lesson.status == "held"
    assert lesson.records[-1].stage == "guided_practice"
    assert lesson.records[-1].status == "failed"
    assert "placeholder" in view.reason


class AdvanceAll:
    """Test policy: moves on from any stage. Placeholder HoldPolicy holds skips instead."""

    def after_stage(self, stage, record, history):
        return Decision("advance", "test: advance everything")


def test_skipped_guided_practice_advances_under_an_advancing_policy(tmp_path):
    # Every item was seen recently and none is due, so the session has nothing to do.
    course = load_course(DEFAULT_COURSE)
    learners = tmp_path / "learners.json"
    seen = LearnerModel("l1", course_id=course.course_id)
    scenario = course.load_scenario(course.scenario_ids()[0])
    for item in scenario.items:
        seen.record_result(item.id, solved_level=0, achieved=True, now=BASE, level=course.level)
    seen.save(learners)

    lesson = Lesson(course, scenario.id, "l1", learners, tmp_path / "log.jsonl",
                    clock=_clock(), policy=AdvanceAll())
    _drive_to_end(lesson)
    gp = next(r for r in lesson.records if r.stage == "guided_practice")
    assert gp.status == "skipped"
    assert lesson.status == "complete"


def test_skipped_guided_practice_advances_past_the_stage(tmp_path):
    course = load_course(DEFAULT_COURSE)
    learners = tmp_path / "learners.json"
    seen = LearnerModel("l1", course_id=course.course_id)
    scenario = course.load_scenario(course.scenario_ids()[0])
    for item in scenario.items:
        seen.record_result(item.id, solved_level=0, achieved=True, now=BASE, level=course.level)
    seen.save(learners)
    lesson = Lesson(course, scenario.id, "l1", learners, tmp_path / "log.jsonl", clock=_clock())
    view = _drive_to_end(lesson)
    guided = [r for r in lesson.records if r.stage == "guided_practice"]
    assert [r.status for r in guided] == ["skipped"]
    assert view.expects == "done"


# --- time budgets -----------------------------------------------------------

def test_time_overrun_is_recorded_and_does_not_interrupt(tmp_path):
    lesson = _lesson(tmp_path, clock=_clock(step_seconds=1000))
    view = _drive_to_end(lesson)
    assert view.expects == "done"
    assert all(r.over_budget for r in lesson.records if r.stage in ("intro", "model_dialogue"))


def test_within_budget_is_not_flagged(tmp_path):
    lesson = _lesson(tmp_path, clock=_clock(step_seconds=0.1))
    _drive_to_end(lesson)
    assert not any(r.over_budget for r in lesson.records)


# --- stubs and the assessment structure ------------------------------------

def test_assessment_summary_has_all_fields_with_basis(tmp_path):
    lesson = _lesson(tmp_path)
    _drive_to_end(lesson)
    summary = lesson.assessment
    assert isinstance(summary, AssessmentSummary)
    fields = {f.name for f in dataclasses.fields(summary)} - {"status"}
    assert fields == ASSESSED_FIELDS
    for name in ASSESSED_FIELDS:
        value = getattr(summary, name)
        assert value.basis in ("log", "stub"), name


def test_log_backed_fields_are_computed_and_stub_fields_are_not(tmp_path):
    lesson = _lesson(tmp_path)
    _drive_to_end(lesson)
    s = lesson.assessment
    assert s.task_completion.basis == "log" and s.task_completion.value == 1.0
    assert s.independence.basis == "log" and s.independence.value == 1.0
    for name in ("clarity", "grammar", "vocabulary", "comprehension"):
        assert getattr(s, name).basis == "stub"
        assert getattr(s, name).value is None
    assert s.status == "stubbed"


def test_lesson_cannot_be_restarted(tmp_path):
    lesson = _lesson(tmp_path)
    lesson.start()
    with pytest.raises(RuntimeError):
        lesson.start()


def test_decision_rejects_unknown_action_and_empty_reason():
    with pytest.raises(ValueError):
        Decision("skip", "x")
    with pytest.raises(ValueError):
        Decision("advance", "")


class _TwoTurnRoleplay:
    """A fake roleplay that takes two learner turns, then completes."""

    def start(self, scenario, learner_id, history):
        outer = self

        class Session:
            done = False
            prompt = "Partner: Hello, what do you want?"
            turns = []

            def submit(self, text):
                self.turns.append({"speaker": "learner", "text": text, "audio": None})
                if len(self.turns) >= 2:
                    self.done = True
                    return ""
                self.prompt = "Partner: Anything else?"
                return self.prompt

            def result(self):
                from guided_practice.roleplay import RoleplayResult
                return RoleplayResult(status="completed", transcript=list(self.turns))

        return Session()


def test_roleplay_takes_learner_turns_through_the_lesson(tmp_path):
    course = load_course(DEFAULT_COURSE)
    lesson = Lesson(course, course.scenario_ids()[0], "l1", tmp_path / "learners.json",
                    tmp_path / "log.jsonl", clock=_clock(), roleplay=_TwoTurnRoleplay())
    view = lesson.start()
    while view.stage != "roleplay" or view.expects != "answer":
        if view.expects == "continue":
            view = lesson.continue_()
        else:
            view = lesson.submit(_model(lesson, view))
    assert view.prompt == "Partner: Hello, what do you want?"
    view = lesson.submit("I want tea")
    assert view.stage == "roleplay" and view.expects == "answer"
    assert view.prompt == "Partner: Anything else?"
    view = lesson.submit("Nothing, thanks")
    assert view.expects == "done" and lesson.status == "complete"
    rp = next(r for r in lesson.records if r.stage == "roleplay")
    assert rp.status == "completed"
    assert rp.detail["turns"] == 2
