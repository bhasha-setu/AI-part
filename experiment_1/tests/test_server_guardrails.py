"""Step 3: guardrails. One test per guardrail, plus the fallback lines fitting the cap."""

from guided_practice.course import load_course
from guided_practice.normalise import normalise
from pedagogy_server.config import LEVELS, VERDICT_KEYS, load_judge_config, load_reply_strings
from pedagogy_server.guardrails import Candidate, apply_guardrails
from pedagogy_server.models import ServerSession

COURSE = load_course("english_for_hindi_a1")
SCENARIO = COURSE.load_scenario("introducing_yourself")
CFG = load_judge_config()
STRINGS = load_reply_strings(COURSE.ui.code)

PROMPT_STYLE = "One word is missing after 'name'."


def guard(cand, item_id="intro_name", level=0, transcript="my name ravi", state=None):
    state = state or ServerSession("s", "l", COURSE.course_id, SCENARIO.id)
    return apply_guardrails(
        cand, item=SCENARIO.item_by_id(item_id), norm=normalise(transcript, COURSE.lang),
        level=level, ui=COURSE.ui, strings=STRINGS, cfg=CFG, state=state,
    )


def surface(correction=PROMPT_STYLE, error_id="missing_copula", reply="Nearly there."):
    return Candidate("surface_error", error_id, correction, reply, "judge")


def test_a_valid_correction_is_shown():
    out = guard(surface())
    assert out.actions == ()
    assert out.correction_shown == PROMPT_STYLE
    assert out.feedback == f"Nearly there. {PROMPT_STYLE}"


def test_correction_cap_stops_correcting_and_encourages():
    state = ServerSession("s", "l", COURSE.course_id, SCENARIO.id)
    state.corrections_used = CFG.corrections_per_session
    out = guard(surface(), state=state)
    assert "correction_cap_reached" in out.actions
    assert out.correction_shown is None
    assert out.feedback == STRINGS.encourage
    assert out.verdict == "surface_error"     # the ladder still sees the surface error


def test_each_error_id_is_corrected_once_per_session():
    state = ServerSession("s", "l", COURSE.course_id, SCENARIO.id)
    state.record_correction("intro_name", "missing_copula", PROMPT_STYLE)
    out = guard(surface(), state=state)
    assert "already_corrected:missing_copula" in out.actions
    assert out.correction_shown is None


def test_a_form_the_learner_did_not_produce_is_not_corrected():
    # The learner said "i like tea"; the missing-copula pattern doesn't match it.
    out = guard(surface(), transcript="i like tea")
    assert "form_not_produced:missing_copula" in out.actions
    assert out.correction_shown is None


def test_an_unlisted_error_is_not_corrected():
    out = guard(surface(error_id=None))
    assert "unlisted_error_not_corrected" in out.actions
    assert out.correction_shown is None


def test_a_correction_hidden_in_the_reply_is_still_guarded():
    state = ServerSession("s", "l", COURSE.course_id, SCENARIO.id)
    state.corrections_used = CFG.corrections_per_session
    out = guard(surface(correction=None, reply="You should say one more word."), state=state)
    assert "correction_cap_reached" in out.actions
    assert out.feedback == STRINGS.encourage


def test_no_early_reveal_of_the_target_form():
    # The rule fallback's own recast ("Say: My name is Ravi.") is held back at level 0...
    fallback = Candidate("surface_error", "missing_copula", "Say: My name is Ravi.", "", "fallback")
    out = guard(fallback, level=0)
    assert "early_reveal" in out.actions
    assert out.feedback == STRINGS.encourage
    # ...and shown at level 2, where "My name is…" is already on screen.
    out = guard(fallback, level=2)
    assert out.actions == ()
    assert out.feedback == "Say: My name is Ravi."


def test_no_early_reveal_in_a_non_correct_reply():
    cand = Candidate("meaning_error", None, None, "Try my name is Ravi.", "judge")
    out = guard(cand, transcript="i like tea")
    assert "early_reveal" in out.actions
    assert out.feedback == STRINGS.guided("meaning_error", 0)


def test_length_cap_on_the_line_the_learner_hears():
    cand = Candidate("correct", None, None, "Great. " * 40, "judge")
    out = guard(cand, transcript="my name is ravi")
    assert "length_cap" in out.actions
    assert out.feedback == STRINGS.guided("correct", 0)


def test_every_fallback_line_fits_the_length_cap():
    for verdict in VERDICT_KEYS:
        for level in range(LEVELS):
            line = STRINGS.guided(verdict, level)
            assert len(line) <= CFG.feedback_max_chars
    assert len(STRINGS.encourage) <= CFG.feedback_max_chars
