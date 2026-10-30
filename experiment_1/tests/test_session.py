"""Session state-machine behaviour tests."""

import pytest

from guided_practice.content import SCENARIOS_DIR, load_scenario
from guided_practice.evaluator import RuleEvaluator, StubLLMJudge, Verdict
from guided_practice.session import Session
from guided_practice.summary import outcome_per_item, summarise
from guided_practice.turn_log import JsonlLogger


def make_session(scenario, items, tmp_path, fixed_clock, cap=5):
    logger = JsonlLogger(tmp_path / "log.jsonl")
    return Session(
        scenario=scenario,
        learner_id="l1",
        evaluator=RuleEvaluator(llm=StubLLMJudge()),
        logger=logger,
        items=items,
        clock=fixed_clock,
        session_id="s1",
        cap=cap,
    )


def one_item(scenario, item_id):
    return [scenario.item_by_id(item_id)]


# --- level advances only after a FAILED attempt, never after correct --------

def test_correct_at_level_0_does_not_advance(intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_origin"), tmp_path, fixed_clock)
    s.start()
    r = s.submit("I'm from Dehradun")
    assert r.verdict == Verdict.CORRECT
    assert r.hint_level == 0           # solved at level 0
    assert r.item_ended and r.outcome == "produced" and r.independent


def test_meaning_error_advances_one_level(intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_name"), tmp_path, fixed_clock)
    p0 = s.start()
    assert p0 == "Someone asks your name. Say it."
    r = s.submit("the weather is nice")   # meaning_error
    assert r.verdict == Verdict.MEANING_ERROR
    assert r.prompt == "Name → ?"         # advanced to level 1 cue


def test_accepted_alternative_passes_at_current_level(intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_name"), tmp_path, fixed_clock)
    s.start()
    s.submit("blah blah")                 # meaning -> level 1
    r = s.submit("I'm Ravi")              # accepted alternative at level 1
    assert r.verdict == Verdict.CORRECT
    assert r.hint_level == 1


# --- surface error retries at the SAME level, does not advance --------------

def test_surface_error_does_not_advance_level(intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_name"), tmp_path, fixed_clock)
    s.start()
    r = s.submit("my name ravi")          # surface error at level 0
    assert r.verdict == Verdict.SURFACE_ERROR
    assert r.hint_level == 0              # still level 0
    assert not r.item_ended
    r2 = s.submit("my name is ravi")      # corrected, still level 0
    assert r2.verdict == Verdict.CORRECT
    assert r2.hint_level == 0
    assert r2.independent                 # solved at level 0 counts as independent


def test_max_two_retries_then_advance(intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_name"), tmp_path, fixed_clock)
    s.start()
    r1 = s.submit("my name ravi")   # surface, retry 1, level 0
    r2 = s.submit("my name ravi")   # surface, retry 2, level 0
    r3 = s.submit("my name ravi")   # surface, retries exhausted -> advance
    assert r1.hint_level == 0 and r2.hint_level == 0
    assert r3.verdict == Verdict.SURFACE_ERROR
    assert r3.prompt == "Name → ?"  # advanced to level 1


# --- full scripted session --------------------------------------------------

def test_full_scripted_session(intro, tmp_path, fixed_clock):
    """A1 learner: hints on items 1 and 3, item 2 at level 0, item 4 at 0."""
    log_path = tmp_path / "log.jsonl"
    logger = JsonlLogger(log_path)
    s = Session(intro, "l1", RuleEvaluator(llm=StubLLMJudge()), logger,
                clock=fixed_clock, session_id="s1", cap=5)

    s.start()
    script = [
        "Hello there",          # item1 L0 meaning -> L1
        "my name Ravi",         # item1 L1 surface (retry)
        "my name is Ravi",      # item1 L1 correct
        "I'm fine, thank you",  # item1 follow-up (unscored)
        "I'm from Dehradun",    # item2 L0 correct
        "I work",               # item3 L0 meaning -> L1
        "nothing comes to mind",  # item3 L1 meaning -> L2
        "I am student",         # item3 L2 surface (retry)
        "I am a student",       # item3 L2 correct
        "I like cricket",       # item4 L0 correct
    ]
    results = []
    for text in script:
        results.append(s.submit(text))
    assert s.done

    summary = summarise(JsonlLogger.read(log_path), session_id="s1")
    assert summary.items_attempted == [
        "intro_name", "intro_origin", "intro_work", "intro_interest",
    ]
    assert summary.hint_level_per_item == {
        "intro_name": 1,
        "intro_origin": 0,
        "intro_work": 2,
        "intro_interest": 0,
    }
    assert summary.independence_score == pytest.approx(0.75)
    assert summary.review_items == ["intro_work"]


# --- not achieved -----------------------------------------------------------

# --- every common-error example: surface_error, no level advance -----------

def _all_error_cases():
    cases = []
    for path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        scen = load_scenario(path)
        for item in scen.items:
            for ce in item.common_errors:
                cases.append((path.stem, item.id, ce.id, ce.example))
    return cases


@pytest.mark.parametrize(
    "scen_name,item_id,err_id,example", _all_error_cases(),
    ids=[f"{c[1]}:{c[2]}" for c in _all_error_cases()],
)
def test_error_example_does_not_advance_level(
    scen_name, item_id, err_id, example, intro, food, tmp_path, fixed_clock
):
    scenario = intro if scen_name == "introducing_yourself" else food
    s = make_session(scenario, one_item(scenario, item_id), tmp_path, fixed_clock)
    s.start()
    r = s.submit(example)
    assert r.verdict == Verdict.SURFACE_ERROR
    assert r.hint_level == 0           # still at level 0, did not advance
    assert not r.item_ended
    assert r.prompt.startswith("Now say the correct form:")


# --- the spec's exact noisy inputs, end to end -----------------------------

@pytest.mark.parametrize("text", [
    "um my name is ravi",
    "  my   name   is   ravi  ",
    "My name is Ravi.",
    "my nam is ravi",       # typo tolerance
])
def test_noisy_inputs_match_at_level_0(text, intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_name"), tmp_path, fixed_clock)
    s.start()
    r = s.submit(text)
    assert r.verdict == Verdict.CORRECT
    assert r.hint_level == 0


def test_spec_noisy_i_am_agree_with_trailing_period(food, tmp_path, fixed_clock):
    # The brief lists "I am agree." as a noisy input that must match correctly.
    s = make_session(food, one_item(food, "food_agree"), tmp_path, fixed_clock)
    s.start()
    r = s.submit("I am agree.")          # trailing period + wrong verb
    assert r.verdict == Verdict.SURFACE_ERROR
    assert r.feedback == "Say: I agree."
    assert r.hint_level == 0             # correction is a same-level retry
    assert not r.item_ended
    r2 = s.submit("I agree")             # following the correction succeeds
    assert r2.verdict == Verdict.CORRECT
    assert r2.hint_level == 0


def test_i_am_go_is_surface(intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_interest"), tmp_path, fixed_clock)
    s.start()
    r = s.submit("I am go swimming")
    assert r.verdict == Verdict.SURFACE_ERROR
    assert r.feedback == "Say: I go swimming."


# --- the 'why' flag ---------------------------------------------------------

def test_why_flag_off_is_one_line(intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_name"), tmp_path, fixed_clock)
    s.start()
    r = s.submit("my name ravi", why=False)
    assert r.feedback == "Say: My name is Ravi."


def test_why_flag_on_adds_explanation(intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_name"), tmp_path, fixed_clock)
    s.start()
    r = s.submit("my name ravi", why=True)
    assert r.feedback.startswith("Say: My name is Ravi.")
    assert len(r.feedback) > len("Say: My name is Ravi.")   # explanation appended


# --- 'repeated' outcome at level 3 -----------------------------------------

def test_repeated_outcome_at_level_3(intro, tmp_path, fixed_clock):
    log_path = tmp_path / "log.jsonl"
    logger = JsonlLogger(log_path)
    s = Session(intro, "l1", RuleEvaluator(llm=StubLLMJudge()), logger,
                items=one_item(intro, "intro_name"), clock=fixed_clock,
                session_id="s1", cap=5)
    s.start()
    s.submit("blah")        # L0 meaning -> L1
    s.submit("blah")        # L1 meaning -> L2
    s.submit("blah")        # L2 meaning -> L3 (shows model sentence)
    r = s.submit("My name is Ravi")   # repeat the model at L3
    assert r.verdict == Verdict.CORRECT
    assert r.hint_level == 3
    assert r.outcome == "repeated"
    assert r.independent is False
    # outcome is recorded in the log, not just in memory
    records = JsonlLogger.read(log_path)
    assert outcome_per_item(records)["intro_name"] == "repeated"


def test_surface_errors_before_correct_is_logged(intro, tmp_path, fixed_clock):
    log_path = tmp_path / "log.jsonl"
    logger = JsonlLogger(log_path)
    s = Session(intro, "l1", RuleEvaluator(llm=StubLLMJudge()), logger,
                items=one_item(intro, "intro_name"), clock=fixed_clock,
                session_id="s1", cap=5)
    s.start()
    s.submit("my name ravi")       # surface error (retry)
    s.submit("my name is ravi")    # correct, closes item
    closing = [r for r in JsonlLogger.read(log_path)
               if r["verdict"] == "correct"][0]
    assert closing["surface_errors_before_correct"] == 1
    assert closing["outcome"] == "produced"


def test_item_not_achieved_after_exhausting_ladder(intro, tmp_path, fixed_clock):
    s = make_session(intro, one_item(intro, "intro_name"), tmp_path, fixed_clock)
    s.start()
    last = None
    for _ in range(12):                  # keep failing on meaning
        if s.done:
            break
        last = s.submit("completely unrelated sentence")
    assert s.done
    assert last.outcome == "not_achieved"
    assert last.independent is False
