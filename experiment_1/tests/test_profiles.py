"""The same core logic runs on two different course profiles, with no code change.

English-for-Hindi A1 (real DRAFT content) and Hindi-for-English A1 (placeholder
content). The script for each course is derived from that course's own content:
model answers, error examples and their corrections. Nothing here names a language.
"""

import re
from datetime import datetime, timedelta, timezone

import pytest

from guided_practice.config import load_profile
from guided_practice.content import validate_scenario
from guided_practice.course import load_course
from guided_practice.evaluator import RuleEvaluator, StubLLMJudge, Verdict
from guided_practice.lesson import Lesson
from guided_practice.session import Session
from guided_practice.summary import independence_score
from guided_practice.turn_log import JsonlLogger

COURSES = ["english_for_hindi_a1", "hindi_for_english_a1"]
_SLOT = re.compile(r"\{\w+\}")
FILLER = "Ravi"  # a generic slot value; both courses accept it in every slot


def _filled(template: str) -> str:
    return _SLOT.sub(FILLER, template)


def _clock():
    base = datetime(2025, 1, 6, 9, 0, 0, tzinfo=timezone.utc)
    n = {"i": 0}

    def clock():
        n["i"] += 1
        return base + timedelta(seconds=n["i"])

    return clock


def _scenarios():
    for course_id in COURSES:
        for scenario_id in load_profile(course_id).scenarios:
            yield course_id, scenario_id


def _session(course, scenario, item_ids, tmp_path):
    items = [scenario.item_by_id(i) for i in item_ids]
    judge = StubLLMJudge()
    return Session(
        scenario=scenario, learner_id="l1", evaluator=RuleEvaluator(llm=judge),
        logger=JsonlLogger(tmp_path / "log.jsonl"), items=items, clock=_clock(),
        session_id="s1", course=course,
    ), judge


@pytest.mark.parametrize("course_id,scenario_id", list(_scenarios()))
def test_content_validates(course_id, scenario_id):
    scenario = load_course(course_id).load_scenario(scenario_id)
    assert validate_scenario(scenario) == []


@pytest.mark.parametrize("course_id,scenario_id", list(_scenarios()))
def test_model_answer_is_correct_at_level_0(course_id, scenario_id, tmp_path):
    course = load_course(course_id)
    scenario = course.load_scenario(scenario_id)
    for item in scenario.items:
        session, judge = _session(course, scenario, [item.id], tmp_path)
        session.start()
        result = session.submit(_filled(item.model_answer))
        assert result.verdict == Verdict.CORRECT, (item.id, result)
        assert result.hint_level == 0
        assert judge.calls == []  # no LLM on the main path


@pytest.mark.parametrize("course_id,scenario_id", list(_scenarios()))
def test_junk_input_advances_one_level(course_id, scenario_id, tmp_path):
    course = load_course(course_id)
    scenario = course.load_scenario(scenario_id)
    item = scenario.items[0]
    session, _ = _session(course, scenario, [item.id], tmp_path)
    session.start()
    result = session.submit("zzz qqq")
    assert result.verdict == Verdict.MEANING_ERROR
    assert result.prompt == item.cue  # level 1


@pytest.mark.parametrize("course_id,scenario_id", list(_scenarios()))
def test_every_error_example_is_surface_and_does_not_advance(course_id, scenario_id, tmp_path):
    course = load_course(course_id)
    scenario = course.load_scenario(scenario_id)
    checked = 0
    for item in scenario.items:
        for ce in item.common_errors:
            session, _ = _session(course, scenario, [item.id], tmp_path)
            session.start()
            result = session.submit(ce.example)
            assert result.verdict == Verdict.SURFACE_ERROR, (item.id, ce.id)
            assert result.hint_level == 0
            assert result.prompt.startswith(course.ui.retry_prompt.split("{prompt}")[0])
            checked += 1
    assert checked > 0, f"{scenario_id} has no error examples to exercise"


@pytest.mark.parametrize("course_id,scenario_id", list(_scenarios()))
def test_every_correction_is_accepted(course_id, scenario_id, tmp_path):
    course = load_course(course_id)
    scenario = course.load_scenario(scenario_id)
    for item in scenario.items:
        for ce in item.common_errors:
            session, _ = _session(course, scenario, [item.id], tmp_path)
            session.start()
            session.submit(ce.example)  # surface error, same level
            result = session.submit(ce.correction_example)
            assert result.verdict == Verdict.CORRECT, (item.id, ce.id, result)


@pytest.mark.parametrize("course_id", COURSES)
def test_full_session_all_independent(course_id, tmp_path):
    course = load_course(course_id)
    scenario_id = course.scenario_ids()[0]
    scenario = course.load_scenario(scenario_id)
    session = Session(
        scenario=scenario, learner_id="l1", evaluator=RuleEvaluator(llm=StubLLMJudge()),
        logger=JsonlLogger(tmp_path / "log.jsonl"), clock=_clock(), session_id="s1",
        course=course,
    )
    session.start()
    for item in session.items:
        if session.done:
            break
        session.submit(_filled(item.model_answer))
        while session.awaiting_follow_up:
            session.submit("ok")
    records = JsonlLogger.read(tmp_path / "log.jsonl")
    assert independence_score(records, "s1", course.level) == 1.0


@pytest.mark.parametrize("course_id", COURSES)
def test_full_lesson_completes_on_both_profiles(course_id, tmp_path):
    course = load_course(course_id)
    scenario_id = course.scenario_ids()[0]
    lesson = Lesson(course, scenario_id, "l1", tmp_path / "learners.json",
                    tmp_path / "log.jsonl", clock=_clock())
    view = lesson.start()
    while view.expects not in ("done", "held"):
        if view.expects == "continue":
            view = lesson.continue_()
        elif view.follow_up:
            view = lesson.submit("ok")
        else:
            view = lesson.submit(_filled(lesson.scenario.item_by_id(view.item_id).model_answer))
    assert lesson.status == "complete", lesson.records
    stages = [r.stage for r in lesson.records]
    assert stages == ["intro", "model_dialogue", "useful_language",
                      "guided_practice", "roleplay", "assessment"]


def test_both_profiles_load_in_one_process():
    # Loading one course must not leak state into the other.
    en = load_course("english_for_hindi_a1")
    hi = load_course("hindi_for_english_a1")
    assert en.lang.code == "en" and hi.lang.code == "hi"
    assert "um" in en.lang.leading_fillers and "um" not in hi.lang.leading_fillers
    assert en.level is not None and hi.level is not None
