"""Verdict-engine tests: accepted phrasings, surface errors, typo tolerance,
and pattern narrowness."""

import pytest

from guided_practice.evaluator import RuleEvaluator, StubLLMJudge, Verdict
from guided_practice.normalise import normalise


def ev(scenario, item_id, text):
    item = scenario.item_by_id(item_id)
    judge = StubLLMJudge()
    result = RuleEvaluator(llm=judge).evaluate(normalise(text), item)
    return result, judge


# --- accepted phrasings -----------------------------------------------------

@pytest.mark.parametrize("item_id,text", [
    ("intro_name", "My name is Ravi"),
    ("intro_name", "I'm Ravi"),
    ("intro_origin", "I'm from Dehradun"),
    ("intro_origin", "I come from Dehradun"),
    ("intro_work", "I am a student"),
    ("intro_work", "I work as an engineer"),
    ("intro_interest", "I like cricket"),
    ("intro_interest", "I enjoy reading"),
])
def test_accepted_alternatives_are_correct(intro, item_id, text):
    result, judge = ev(intro, item_id, text)
    assert result.verdict == Verdict.CORRECT
    assert judge.calls == []  # LLM seam never touched for correct answers


@pytest.mark.parametrize("item_id,text", [
    ("food_menu", "Can I have the menu, please?"),
    ("food_agree", "I agree"),
    ("food_agree", "Yes I agree"),
    ("food_ask_price", "How much is it?"),
    ("food_friend_diet", "She is vegetarian"),
    ("food_friend_diet", "She does not eat meat"),
])
def test_food_accepted_alternatives(food, item_id, text):
    result, _ = ev(food, item_id, text)
    assert result.verdict == Verdict.CORRECT


# --- common errors -> surface_error ----------------------------------------

ERROR_CASES = [
    ("introducing_yourself", "intro_name", "my name ravi", "missing_copula"),
    ("introducing_yourself", "intro_origin", "I am Dehradun", "missing_from_preposition"),
    ("introducing_yourself", "intro_work", "I am student", "dropped_article"),
    ("introducing_yourself", "intro_interest", "I am like cricket", "wrong_verb_after_i_am"),
    ("ordering_food", "food_menu", "can I have menu please", "menu_missing_the"),
    ("ordering_food", "food_agree", "I am agree", "wrong_verb_agree"),
    ("ordering_food", "food_ask_price", "How much it is?", "question_word_order"),
    ("ordering_food", "food_friend_diet", "She are vegetarian", "he_she_agreement"),
]


@pytest.mark.parametrize("scen,item_id,text,err_id", ERROR_CASES,
                         ids=[c[3] for c in ERROR_CASES])
def test_common_errors_are_surface(scen, item_id, text, err_id, intro, food):
    scenario = intro if scen == "introducing_yourself" else food
    result, judge = ev(scenario, item_id, text)
    assert result.verdict == Verdict.SURFACE_ERROR
    assert result.matched_error_id == err_id
    assert result.feedback  # one-line correction present
    assert judge.calls == []  # surface errors never hit the LLM


def test_every_authored_example_is_surface(intro, food):
    for scenario in (intro, food):
        for item in scenario.items:
            for ce in item.common_errors:
                result, _ = ev(scenario, item.id, ce.example)
                assert result.verdict == Verdict.SURFACE_ERROR
                assert result.matched_error_id == ce.id


def test_surface_feedback_uses_captured_slot(intro):
    result, _ = ev(intro, "intro_name", "my name ravi")
    assert result.feedback == "Say: My name is Ravi."  # title-cased name


# --- typo tolerance ---------------------------------------------------------

def test_typo_is_corrected_to_accepted(intro):
    result, _ = ev(intro, "intro_name", "my nam is ravi")
    assert result.verdict == Verdict.CORRECT


# --- narrowness: grammar errors must NOT be swallowed by typo/accepted ------

def test_dropped_article_survives_typo_tolerance(intro):
    # "i am student" must stay a surface error, not be "corrected" to correct.
    result, _ = ev(intro, "intro_work", "I am student")
    assert result.verdict == Verdict.SURFACE_ERROR


def test_valid_longer_answer_not_flagged(intro):
    # "I am working in a bank" must not trip a dropped-article pattern.
    result, _ = ev(intro, "intro_work", "I am working in a bank")
    assert result.verdict != Verdict.SURFACE_ERROR


def test_correct_article_answer_not_flagged(intro):
    result, _ = ev(intro, "intro_work", "I am a student")
    assert result.verdict == Verdict.CORRECT


# Realistic A1 answers that must be CORRECT (regression guard for false errors).
MUST_BE_CORRECT = [
    ("intro_work", "I am an engineer"),
    ("intro_work", "I am studying"),
    ("intro_work", "I work in a bank"),
    ("intro_work", "I am working in a bank"),
    ("intro_origin", "I am Indian"),
]


@pytest.mark.parametrize("item_id,text", MUST_BE_CORRECT)
def test_must_be_correct(intro, item_id, text):
    result, _ = ev(intro, item_id, text)
    assert result.verdict == Verdict.CORRECT, result


# Answers that must NEVER be a (wrong) surface correction.
MUST_NOT_BE_SURFACE = [
    ("intro_work", "I work in a bank"),
    ("intro_origin", "I am Indian"),
    ("intro_interest", "I enjoy painting"),
]


@pytest.mark.parametrize("item_id,text", MUST_NOT_BE_SURFACE)
def test_must_not_be_surface_error(intro, item_id, text):
    result, _ = ev(intro, item_id, text)
    assert result.verdict != Verdict.SURFACE_ERROR, result


def test_following_the_correction_then_passes(intro, food):
    """Every surface correction, once said, must be accepted (not an error)."""
    for scenario in (intro, food):
        for item in scenario.items:
            for ce in item.common_errors:
                result, _ = ev(scenario, item.id, ce.correction_example)
                assert result.verdict == Verdict.CORRECT, (item.id, ce.id, result)


# --- fallthrough ------------------------------------------------------------

def test_off_topic_is_empty(intro):
    result, judge = ev(intro, "intro_name", "   ")
    assert result.verdict == Verdict.OFF_TOPIC
    assert judge.calls == []  # empty input never hits the LLM


def test_meaning_error_hits_llm_seam(intro):
    result, judge = ev(intro, "intro_name", "the weather is nice today")
    assert result.verdict == Verdict.MEANING_ERROR
    assert len(judge.calls) == 1  # LLM stub was called exactly once
