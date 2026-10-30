"""The conversational tutor: rules, the server-owned structure, and checks on the LLM."""

import io

import pytest

from guided_practice.course import load_course
from pedagogy_server.llm import LLMUnavailable, ScriptedModel
from tutor.cli import run
from tutor.config import Profiles, load_tutor_config, next_stage
from tutor.rules import detect_language, mask, reveal_spans, rule_verdict
from tutor.session import CONVERSE, DONE, REVIEW, TutorSession

CORRECT = ["my name is Ravi", "I am from Delhi", "I am a student", "I like cricket"]


@pytest.fixture(scope="module")
def course():
    return load_course()


@pytest.fixture
def scenario(course):
    return course.load_scenario("introducing_yourself")


@pytest.fixture(scope="module")
def cfg():
    return load_tutor_config()


def say(*parts, verdict="form_error", lang="en", **extra):
    return {"learner_language": lang, "verdict": verdict,
            "say": [{"lang": l, "text": t} for l, t in parts], **extra}


def to_conversation(scenario, cfg, stage=2):
    s = TutorSession(scenario, cfg, None, stage)
    s.start()
    for answer in CORRECT:
        s.turn(answer)
    assert s.phase == CONVERSE
    return s


# --- rules -----------------------------------------------------------------------------

def test_detects_roman_and_devanagari_hindi(scenario):
    item = scenario.items[0]
    assert detect_language("mera naam Ravi hai", item) == "hi"
    assert detect_language("मेरा नाम रवि है", item) == "hi"
    assert detect_language("my name is Ravi", item) == "en"
    assert detect_language("main student hoon") == "hi"


@pytest.mark.parametrize("text", ["I went to the stadium with the team", "what do you do",
                                  "the main road", "do you like the main course"])
def test_english_words_that_are_also_hindi_stay_english(text):
    assert detect_language(text) == "en"


def test_rule_verdict(scenario):
    item = scenario.items[0]
    assert rule_verdict("My name's Ravi.", item)["verdict"] == "correct"
    wrong = rule_verdict("my name Ravi", item)
    assert wrong["verdict"] == "form_error"
    assert wrong["error"]["type"] == "missing_copula"
    assert wrong["error"]["fixed"].lower() == "my name is ravi"
    assert rule_verdict("mera naam Ravi hai", item)["verdict"] == "hindi"


def test_reveal_is_masked_to_the_missing_word(scenario):
    item = scenario.items[0]
    reply = "Almost! Say: My name is Ravi."
    spans = reveal_spans(reply, item, "my name Ravi")
    assert spans
    assert mask(reply, spans, "my name Ravi") == "Almost! Say: My name ___ Ravi."


def test_slot_words_the_learner_never_said_are_not_a_reveal(scenario):
    item = scenario.items[0]
    assert reveal_spans("I am happy you tried!", item, "my name Ravi") == []


# --- the server-owned structure ---------------------------------------------------------

def test_offline_lesson_runs_to_the_end(scenario, cfg):
    s = TutorSession(scenario, cfg, None, stage=2)
    assert s.start().text
    s.turn("mera naam Ravi hai")          # Hindi: bridged, not failed
    assert s.record.hint_level == cfg.hindi_bridge_level
    s.turn("my name is Ravi")
    for answer in CORRECT[1:]:
        s.turn(answer)
    while s.phase != DONE:
        s.turn("ok")
    summary = s.summary()
    first = summary.items[0]
    assert first["used_hindi"] and first["achieved"]
    assert summary.from_memory == 0.75     # the Hindi-first item is not "from memory"


def test_hindi_at_the_last_stage_asks_for_english_without_jumping_hints(scenario, cfg):
    s = TutorSession(scenario, cfg, None, stage=cfg.last_stage)
    s.start()
    reply = s.turn("mera naam Ravi hai")
    assert s.record.hint_level == 1
    assert all(seg.lang == "en" for seg in reply.segments)


def test_wrong_at_level_3_moves_on_and_is_not_achieved(scenario, cfg):
    s = TutorSession(scenario, cfg, None, stage=2)
    s.start()
    for _ in range(4):
        s.turn("hello there")
    assert s.index == 1
    assert s.records[0].achieved is False


# --- checks on the model ----------------------------------------------------------------

def test_accepted_answer_is_never_marked_wrong(scenario, cfg):
    model = ScriptedModel([say(("en", "Not quite.")), say(("en", "Still no."))])
    s = TutorSession(scenario, cfg, model, stage=2)
    s.turn("my name is Ravi")
    assert s.index == 1 and s.records[0].solved_level == 0
    assert "already_verified_correct" in model.calls[0]["user"]


def test_early_reveal_is_retried_then_masked(scenario, cfg):
    reveal = say(("en", "Almost! Say: My name is Ravi."))
    model = ScriptedModel([reveal, reveal])
    s = TutorSession(scenario, cfg, model, stage=2)
    reply = s.turn("my name Ravi")
    assert len(model.calls) == 2 and "fix_these" in model.calls[1]["user"]
    assert "My name ___ Ravi" in reply.text
    assert s.record.hint_level == 1


def test_retry_that_fixes_the_reveal_is_used_as_is(scenario, cfg):
    model = ScriptedModel([say(("en", "Say: My name is Ravi.")),
                           say(("en", "One small word is missing after 'name'."))])
    s = TutorSession(scenario, cfg, model, stage=2)
    reply = s.turn("my name Ravi")
    assert reply.text == "One small word is missing after 'name'."
    assert reply.debug["problems"]


def test_hindi_lines_are_dropped_when_the_stage_does_not_need_them(scenario, cfg):
    model = ScriptedModel([say(("en", "Great!"), ("hi", "Bahut badhiya!"), ("en", "Where are you from?"),
                               verdict="correct")])
    s = TutorSession(scenario, cfg, model, stage=2)
    reply = s.turn("my name is Ravi")
    assert [seg.lang for seg in reply.segments] == ["en", "en"]


def test_model_unavailable_falls_back_to_fixed_lines(scenario, cfg):
    s = TutorSession(scenario, cfg, ScriptedModel([LLMUnavailable("timeout")]), stage=2)
    reply = s.turn("my name Ravi")
    assert reply.debug["source"] == "fallback" and reply.text


def test_conversation_never_corrects_and_bridges_hindi(scenario, cfg):
    s = to_conversation(scenario, cfg)
    s.model = ScriptedModel([
        # corrects mid-conversation: retried
        {"learner_language": "en", "say": [{"lang": "en", "text": "Nice! We say 'I like playing cricket'."}],
         "errors": [{"said": "I like play cricket", "fixed": "I like playing cricket",
                     "type": "verb", "importance": 3}], "wants_to_end": False},
        {"learner_language": "en", "say": [{"lang": "en", "text": "Nice! Where do you play?"}],
         "errors": [{"said": "I like play cricket", "fixed": "I like playing cricket",
                     "type": "verb", "importance": 3}], "wants_to_end": False},
        # Hindi: bridged; the Hindi sentence is not kept as a mistake
        {"learner_language": "hi", "say": [{"lang": "en", "text": "Cool. What do you study?"}],
         "errors": [{"said": "main park mein khelta hoon", "fixed": "I play in the park",
                     "type": "hindi", "importance": 3}],
         "english_version": "I play in the park.", "wants_to_end": False},
    ])
    first = s.turn("I like play cricket")
    assert first.text == "Nice! Where do you play?"
    second = s.turn("main park mein khelta hoon")
    assert second.segments[0].text == 'In English: "I play in the park."'
    assert [e["said"] for e in s.converse_errors] == ["I like play cricket"]


def test_review_asks_first_then_answers(scenario, cfg):
    s = to_conversation(scenario, cfg)
    s.converse_errors = [{"said": "I going stadium", "fixed": "I went to the stadium",
                          "type": "tense", "why": "Past needs went.", "importance": 3}]
    reply = s.turn("bye")
    while s.phase == CONVERSE:
        reply = s.turn("bye")
    assert s.phase == REVIEW
    assert '"I going stadium"' in reply.text and "went" not in reply.text
    end = s.turn("I go to stadium")
    assert "I went to the stadium" in end.text and end.done


def test_review_shows_at_most_the_cap(scenario, cfg):
    s = to_conversation(scenario, cfg)
    s.converse_errors = [{"said": f"s{i}", "fixed": f"f{i}", "type": f"t{i}", "importance": 1}
                         for i in range(6)]
    assert len(s._pick_errors()) == cfg.review_max_errors


# --- support stage and persistence ------------------------------------------------------

def test_support_stage_moves_one_step(cfg):
    assert next_stage(cfg, 2, 1.0, 0) == 3
    assert next_stage(cfg, 2, 0.2, 0) == 1
    assert next_stage(cfg, 2, 0.6, 1) == 2
    assert next_stage(cfg, 0, 0.0, 9) == 0


def test_cli_offline_saves_stage_and_review(tmp_path, cfg):
    answers = iter(CORRECT + ["hello", "ok", "fine", "bye"])
    out = io.StringIO()
    summary = run(None, "t1", tmp_path, cfg, None, lambda _: next(answers), out)
    assert summary.from_memory == 1.0
    assert Profiles(tmp_path / "profiles.json").stage("t1", "english_for_hindi_a1", -1) == 3
    assert (tmp_path / "learners.json").exists()
    assert "From memory: 100%" in out.getvalue()


def test_review_always_shows_the_correct_form_after_a_failed_fix(scenario, cfg):
    s = to_conversation(scenario, cfg)
    s.converse_errors = [{"said": "I going stadium", "fixed": "I went to the stadium",
                          "type": "tense", "importance": 3}]
    while s.phase == CONVERSE:
        s.turn("bye")
    s.model = ScriptedModel([{"fixed_it": False, "say": [{"lang": "en", "text": "Try again: I ... to the stadium."}]}])
    end = s.turn("I go stadium")
    assert end.text.startswith('We say: "I went to the stadium".')


def test_no_correction_guard_holds_for_english_with_ambiguous_words(scenario, cfg):
    s = to_conversation(scenario, cfg)
    correcting = {"learner_language": "en",
                  "say": [{"lang": "en", "text": "We say 'I went to the shop'. Nice!"}],
                  "errors": [{"said": "I go to the shop the day", "fixed": "I went to the shop",
                              "type": "tense", "importance": 3}], "wants_to_end": False}
    s.model = ScriptedModel([correcting, correcting])
    reply = s.turn("I go to the shop the day")
    assert reply.debug["problems"] and "We say" not in reply.text
    assert s.hindi_turns == 0
