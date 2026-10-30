"""Step 2: judge-output validation. One test per rejection rule."""

import json

import pytest

from guided_practice.course import load_course
from guided_practice.normalise import normalise
from pedagogy_server.config import load_judge_config, load_reply_strings
from pedagogy_server.validation import validate_judge_output

COURSE = load_course("english_for_hindi_a1")
SCENARIO = COURSE.load_scenario("introducing_yourself")
CFG = load_judge_config()
STRINGS = load_reply_strings(COURSE.ui.code)

GOOD = {
    "verdict": "surface_error",
    "error_id": "missing_copula",
    "correction": "One word is missing after 'name'.",
    "reply": "Nearly there.",
    "confidence": 0.9,
}


def check(resp, item_id="intro_name", level=0, transcript="my name ravi"):
    raw = resp if isinstance(resp, str) else json.dumps(resp)
    return validate_judge_output(
        raw, item=SCENARIO.item_by_id(item_id), level=level,
        norm=normalise(transcript, COURSE.lang), ui=COURSE.ui, strings=STRINGS, cfg=CFG,
    )


def with_(**changes):
    out = dict(GOOD)
    out.update(changes)
    return out


def test_a_well_formed_response_passes():
    result = check(GOOD)
    assert result.ok, result.rejections
    assert result.response.error_id == "missing_copula"
    assert not result.low_confidence


def test_malformed_json_is_rejected():
    assert check("{verdict: correct").rejections == ["malformed_json"]


def test_json_in_a_code_fence_is_rejected():
    assert check("```json\n" + json.dumps(GOOD) + "\n```").rejections == ["malformed_json"]


def test_non_object_is_rejected():
    assert check("[1, 2]").rejections == ["not_an_object"]


@pytest.mark.parametrize("field", ["verdict", "error_id", "correction", "reply", "confidence"])
def test_missing_field_is_rejected(field):
    resp = dict(GOOD)
    del resp[field]
    assert check(resp).rejections == [f"missing_field:{field}"]


def test_unexpected_field_is_rejected():
    assert check(with_(hint_level=2)).rejections == ["unexpected_field:hint_level"]


def test_verdict_outside_the_four_values_is_rejected():
    assert "bad_verdict" in check(with_(verdict="great", error_id=None, correction=None)).rejections


@pytest.mark.parametrize("value", ["high", True, None])
def test_non_numeric_confidence_is_rejected(value):
    assert check(with_(confidence=value)).rejections == ["bad_type:confidence"]


def test_confidence_out_of_range_is_rejected():
    assert "confidence_out_of_range" in check(with_(confidence=1.5)).rejections


def test_empty_reply_is_rejected():
    assert "empty_reply" in check(with_(reply="  ")).rejections


def test_correction_on_a_non_surface_verdict_is_rejected():
    result = check(with_(verdict="correct"), transcript="my name is ravi")
    assert "correction_without_surface_error" in result.rejections


def test_unknown_error_id_is_rejected():
    assert "unknown_error_id" in check(with_(error_id="invented_error")).rejections


def test_multi_line_correction_is_rejected():
    assert "correction_not_one_line" in check(with_(correction="Add a word.\nAfter name.")).rejections


def test_overlong_correction_is_rejected():
    assert "correction_too_long" in check(with_(correction="x" * (CFG.correction_max_chars + 1))).rejections


def test_correction_revealing_the_target_at_level_0_is_rejected():
    result = check(with_(correction="Say: My name is Ravi."), level=0)
    assert "correction_reveals_target" in result.rejections


def test_the_same_correction_is_allowed_once_first_words_are_shown():
    # Level 2 shows "My name is…", so the correction discloses nothing new.
    result = check(with_(correction="Say: My name is Ravi."), level=2)
    assert result.ok, result.rejections


def test_reply_over_two_sentences_is_rejected():
    assert "reply_too_long" in check(with_(reply="Good. Nearly. Try again.")).rejections


def test_reply_over_the_character_cap_is_rejected():
    assert "reply_too_long" in check(with_(reply="a" * (CFG.reply_max_chars + 1))).rejections


def test_non_correct_reply_revealing_the_target_is_rejected():
    resp = {"verdict": "meaning_error", "error_id": None, "correction": None,
            "reply": "It is my name is Ravi.", "confidence": 0.9}
    assert "reply_reveals_target" in check(resp, transcript="i like tea").rejections


def test_more_than_one_correction_is_rejected():
    resp = with_(reply="Say it instead of skipping.")
    assert "too_many_corrections" in check(resp).rejections


def test_one_correction_inside_the_reply_alone_is_allowed():
    resp = with_(reply="You should say one more word.", correction=None)
    assert check(resp).ok


def test_verdict_contradicting_an_exact_accepted_form_is_rejected():
    resp = {"verdict": "meaning_error", "error_id": None, "correction": None,
            "reply": "Let's try again.", "confidence": 0.9}
    result = check(resp, transcript="my name's Ravi")
    assert "accepted_form_contradicted" in result.rejections


def test_low_confidence_is_flagged_not_rejected():
    result = check(with_(confidence=CFG.min_confidence - 0.1))
    assert result.ok
    assert result.low_confidence
