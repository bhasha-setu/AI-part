"""Course namespacing, config-driven thresholds, and course ids in the log."""

import dataclasses
from datetime import datetime, timedelta, timezone

from guided_practice.config import DEFAULT_COURSE, load_level
from guided_practice.course import load_course
from guided_practice.learner_model import LearnerModel
from guided_practice.summary import independence_score, review_items
from guided_practice.turn_log import JsonlLogger

NOW = datetime(2025, 1, 6, 9, 0, 0, tzinfo=timezone.utc)


def test_same_item_id_in_two_courses_does_not_collide(tmp_path):
    path = tmp_path / "learners.json"
    en = LearnerModel("l1", course_id="english_for_hindi_a1")
    hi = LearnerModel("l1", course_id="hindi_for_english_a1")
    en.record_result("intro_name", solved_level=0, achieved=True, now=NOW)
    hi.record_result("intro_name", solved_level=3, achieved=True, now=NOW)
    en.save(path)
    hi.save(path)

    reloaded_en = LearnerModel.load(path, "l1", course_id="english_for_hindi_a1")
    reloaded_hi = LearnerModel.load(path, "l1", course_id="hindi_for_english_a1")
    assert reloaded_en.state_for("intro_name").box == 2   # promoted
    assert reloaded_hi.state_for("intro_name").box == 1   # reset by a level-3 solve


def test_unscoped_model_keeps_legacy_keys(tmp_path):
    lm = LearnerModel("l1")
    lm.record_result("x", solved_level=0, achieved=True, now=NOW)
    assert "x" in lm.states


def test_independence_threshold_comes_from_the_level_config():
    records = [{"session_id": "s", "item_id": "a", "hint_level": 1, "verdict": "correct",
                "turn_type": "scored"}]
    default = load_level("A1")
    strict = dataclasses.replace(default, independent_max_level=0)
    assert independence_score(records, "s", default) == 1.0   # level 1 counts
    assert independence_score(records, "s", strict) == 0.0    # now it does not


def test_review_threshold_comes_from_the_level_config():
    records = [{"session_id": "s", "item_id": "a", "hint_level": 1, "verdict": "correct",
                "turn_type": "scored"}]
    default = load_level("A1")
    lenient = dataclasses.replace(default, review_min_level=1)
    assert review_items(records, "s", default) == []
    assert review_items(records, "s", lenient) == ["a"]


def test_turn_log_records_course_id(tmp_path):
    from guided_practice.session import Session
    from guided_practice.evaluator import RuleEvaluator, StubLLMJudge

    course = load_course(DEFAULT_COURSE)
    scenario = course.load_scenario("introducing_yourself")
    log = JsonlLogger(tmp_path / "log.jsonl")
    s = Session(scenario, "l1", RuleEvaluator(llm=StubLLMJudge()), log,
                clock=lambda: NOW, session_id="s1", course=course)
    s.start()
    s.submit("my name is Ravi")
    records = JsonlLogger.read(tmp_path / "log.jsonl")
    assert records and all(r["course_id"] == DEFAULT_COURSE for r in records)
