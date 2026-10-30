"""Step 5: the rule-based fallback responder."""

from guided_practice.course import load_course
from guided_practice.evaluator import EvalResult, RuleEvaluator
from guided_practice.lesson import STAGE_ORDER, STAGE_GUIDED_PRACTICE
from guided_practice.normalise import normalise
from pedagogy_server.config import LEVELS, VERDICT_KEYS, load_reply_strings
from pedagogy_server.fallback import FallbackResponder

COURSE = load_course("english_for_hindi_a1")
SCENARIO = COURSE.load_scenario("introducing_yourself")
STRINGS = load_reply_strings(COURSE.ui.code)
FB = FallbackResponder(STRINGS)


def test_one_line_per_stage():
    for stage in STAGE_ORDER:
        if stage != STAGE_GUIDED_PRACTICE:
            assert FB.stage_line(stage)
    assert FB.stage_line("complete") and FB.stage_line("held")


def test_one_line_per_verdict_and_hint_level():
    lines = {(v, lvl): FB.guided_line(v, lvl) for v in VERDICT_KEYS for lvl in range(LEVELS)}
    assert all(lines.values())
    assert lines[("correct", 3)] != lines[("correct", 0)]   # level 3 is a repeat, not a production


def test_surface_error_falls_back_to_the_content_correction():
    item = SCENARIO.item_by_id("intro_name")
    rule = RuleEvaluator().evaluate(normalise("my name Ravi", COURSE.lang), item)
    cand = FB.candidate(rule, level=2)
    assert cand.source == "fallback"
    assert cand.error_id == "missing_copula"
    assert cand.correction == "Say: My name is Ravi."
    assert cand.reply == ""


def test_meaning_error_falls_back_to_the_level_line():
    cand = FB.candidate(EvalResult("meaning_error"), level=1)
    assert cand.correction is None and cand.error_id is None
    assert cand.reply == STRINGS.guided("meaning_error", 1)
