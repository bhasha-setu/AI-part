"""The scenario generator: plan checks, build checks, retry, backup model, fallback,
and the learner library. ScriptedModel only; nothing here touches the network."""

import copy
import json

import pytest

from guided_practice.course import load_course
from guided_practice.evaluator import RuleEvaluator, Verdict
from guided_practice.normalise import normalise
from pedagogy_server.generator import (
    GenerationFailed, LessonOption, ScenarioGenerator, check_plan, check_scenario_data,
    learner_course, load_generator_config, make_generator)
from pedagogy_server.llm import LLMUnavailable, ScriptedModel

BASE = load_course("english_for_hindi_a1")
CFG = load_generator_config()

GOOD_PLAN = {"options": [
    {"title": "Buying vegetables", "can_do": "Ask for vegetables and their price.",
     "situation": "You are at a market stall.", "phrases_preview": ["How much are these?"],
     "why": "You shop every week.", "est_minutes": 5, "existing_scenario_id": None},
    {"title": "Ordering food", "can_do": "Order at a cafe.", "situation": "A cafe.",
     "phrases_preview": [], "why": "It fits.", "est_minutes": 4,
     "existing_scenario_id": "ordering_food"},
]}

GOOD = {
    "title": "Buying vegetables at the market",
    "can_do": "Ask for vegetables and their price at a market.",
    "model_dialogue": [
        {"speaker": "Vendor", "text": "Hello! What do you want today?"},
        {"speaker": "Customer", "text": "I want two kilos of tomatoes, please."},
        {"speaker": "Vendor", "text": "Here you are."},
        {"speaker": "Customer", "text": "How much are these?"},
        {"speaker": "Vendor", "text": "Eighty rupees."},
        {"speaker": "Customer", "text": "Thank you very much."},
    ],
    "useful_language": [
        {"phrase": "I want two kilos of ...", "meaning": "to ask for an amount", "used_by": ["want_tomatoes"]},
        {"phrase": "How much are these?", "meaning": "to ask a price", "used_by": ["ask_price"]},
        {"phrase": "Thank you very much.", "meaning": "to thank", "used_by": ["say_thanks"]},
    ],
    "items": [
        {"id": "want_tomatoes", "prompt": "Tell the seller you need tomatoes, two kilos.",
         "cue": "Tomatoes → ?", "first_words": "I want",
         "model_answer": "I want two kilos of tomatoes, please.",
         "accepted": ["i want two kilos of tomatoes please", "i want two kilos of tomatoes",
                      "i would like two kilos of tomatoes please"],
         "common_errors": [{"id": "plural_kilo", "error_forms": ["i want two kilo tomato please",
                                                                 "i want two kilo tomatoes"],
                            "correction_example": "i want two kilos of tomatoes please",
                            "feedback": "Two of something: check the endings.",
                            "why": "After two, English nouns take -s."}],
         "follow_up": "Here you are."},
        {"id": "ask_price", "prompt": "Find out what the tomatoes cost.", "cue": "Price → ?",
         "first_words": "How much", "model_answer": "How much are these?",
         "accepted": ["how much are these", "how much is this", "how much are they"],
         "common_errors": [{"id": "cost_order", "error_forms": ["how much cost these"],
                            "correction_example": "how much are these",
                            "feedback": "Check the verb.", "why": "Use 'are' with these."}],
         "follow_up": "Eighty rupees."},
        {"id": "say_thanks", "prompt": "Thank the seller warmly.", "cue": "Thanks → ?",
         "first_words": "Thank you", "model_answer": "Thank you very much.",
         "accepted": ["thank you very much", "thank you", "thanks a lot"],
         "common_errors": [], "follow_up": None},
    ],
}

OPTION = LessonOption(option_id="opt-abc123", title="Buying vegetables", can_do="Shop.",
                      situation="Market.", phrases_preview=(), why="Useful.", est_minutes=5)


def course_for(tmp_path):
    return learner_course(BASE, tmp_path, "learner-1")


def gen(outputs, backup=None):
    return ScenarioGenerator(ScriptedModel(outputs), ScriptedModel(outputs), CFG,
                             clock=lambda: 1.0, build_fallback_model=backup)


def broken(path_fn):
    data = copy.deepcopy(GOOD)
    path_fn(data)
    return data


# --- config ---------------------------------------------------------------------------

def test_config_passes_the_threshold_audit():
    assert CFG.build_attempts >= 1 and CFG.min_options <= CFG.max_options


def test_make_generator_kinds():
    assert not make_generator("off").enabled
    with pytest.raises(ValueError):
        make_generator("gpt")


# --- plan ------------------------------------------------------------------------------

def test_plan_accepts_good_options_and_marks_existing_ones(tmp_path):
    course = course_for(tmp_path)
    result = ScenarioGenerator(ScriptedModel([GOOD_PLAN]), None, CFG).plan("vegetables", course)
    assert result.source == "llm"
    assert [o.scenario_id for o in result.options] == [None, "ordering_food"]
    assert len({o.option_id for o in result.options}) == 2


@pytest.mark.parametrize("change, problem", [
    (lambda d: d["options"].pop(), "option_count:1"),
    (lambda d: d["options"][0].update(title="x" * 200), "option0:title"),
    (lambda d: d["options"][0].update(est_minutes=90), "option0:est_minutes"),
    (lambda d: d["options"][1].update(existing_scenario_id="made_up"), "option1:unknown_scenario_id"),
])
def test_plan_checks(change, problem):
    data = copy.deepcopy(GOOD_PLAN)
    change(data)
    assert problem in check_plan(data, CFG, {"ordering_food"})


def test_a_rejected_or_failed_plan_falls_back_to_existing_lessons_ranked_by_overlap(tmp_path):
    course = course_for(tmp_path)
    for output in ["not json", LLMUnavailable("timeout"), {"options": []}]:
        result = ScenarioGenerator(ScriptedModel([output]), None, CFG).plan("ordering food at a cafe", course)
        assert result.source == "fallback" and result.note
        assert result.options[0].scenario_id == "ordering_food"
        assert all(o.scenario_id for o in result.options)


# --- build checks -------------------------------------------------------------------------

def test_the_good_lesson_passes_every_check():
    assert check_scenario_data(GOOD, CFG, BASE) == []


@pytest.mark.parametrize("change, fragment", [
    (lambda d: d["items"].pop(), "item_count"),
    (lambda d: d["items"][0].update(prompt="Say: I want two kilos of tomatoes, please."), "prompt gives the answer away"),
    # Through an accepted alternative, not the model answer (the leak found live).
    (lambda d: d["items"][0].update(prompt="Say: I want two kilos of tomatoes."), "prompt gives the answer away"),
    (lambda d: d["items"][0].update(model_answer="I want two kilos of fresh red tomatoes from your shop today please."),
     "over 10 words"),
    (lambda d: d["model_dialogue"].__delitem__(3), "dialogue never says the model answer"),
    (lambda d: d["items"][1]["common_errors"][0].update(error_forms=["how much is this"]),
     "is also an accepted answer"),
    (lambda d: d["items"][1].update(accepted=["कितने का है", "how much are these"]), "script"),
    (lambda d: d["useful_language"][2].update(used_by=["nope"]), "unknown items"),
    (lambda d: d["items"][0].update(follow_up="The vendor says: here you are."), "no narration"),
    (lambda d: d["items"][2].update(id="Say Thanks"), "lowercase letters"),
    (lambda d: d["model_dialogue"].append({"speaker": "Narrator", "text": "Later."}), "two speakers"),
])
def test_build_checks(change, fragment):
    problems = check_scenario_data(broken(change), CFG, BASE)
    assert any(fragment in p for p in problems), problems


# --- build end to end ------------------------------------------------------------------------

def test_a_good_build_lands_in_the_learner_library_and_runs_through_the_rules(tmp_path):
    course = course_for(tmp_path)
    result = gen([GOOD]).build(OPTION, course, "I want to buy vegetables")
    assert result.attempts == 1 and result.item_count == 3
    assert result.scenario_id in course.scenario_ids()
    assert "GENERATED, DRAFT" in result.path.read_text(encoding="utf-8")

    scenario = course.load_scenario(result.scenario_id)
    item = scenario.items[0]
    assert item.id.startswith("g") and item.id.endswith("_want_tomatoes")
    rules = RuleEvaluator()
    ev = lambda text: rules.evaluate(normalise(text, course.lang), item)
    assert ev("i want two kilo tomato please").verdict == Verdict.SURFACE_ERROR
    assert ev("i want two kilo tomato please").matched_error_id.endswith("_plural_kilo")
    assert ev("I would like two kilos of tomatoes, please.").verdict == Verdict.CORRECT
    # The phrase list was renamed with the items.
    assert scenario.useful_language[0].used_by == (item.id,)


def test_another_learner_does_not_see_the_lesson(tmp_path):
    result = gen([GOOD]).build(OPTION, course_for(tmp_path))
    other = learner_course(BASE, tmp_path, "learner-2")
    assert result.scenario_id not in other.scenario_ids()
    with pytest.raises(KeyError):
        other.load_scenario(result.scenario_id)


def test_a_rejected_first_attempt_is_retried_with_its_problems(tmp_path):
    bad = broken(lambda d: d["items"][1]["common_errors"][0].update(error_forms=["how much is this"]))
    model = ScriptedModel([bad, GOOD])
    result = ScenarioGenerator(None, model, CFG, clock=lambda: 1.0).build(OPTION, course_for(tmp_path))
    assert result.attempts == 2
    retry_payload = json.loads(model.calls[1]["user"])
    assert any("accepted answer" in p for p in retry_payload["previous_attempt_problems"])


def test_an_error_pattern_that_matches_a_correct_slot_answer_is_caught_after_compiling(tmp_path):
    # "i am {job}" as an error form also matches "i am a teacher", which is accepted
    # through the slot. Plain string comparison can't see this; the compiled check can.
    data = broken(lambda d: d["items"][2].update(
        model_answer="I am a {job}.", first_words="I am", accepted=["i am a {job}", "i am an {job}"],
        prompt="Say what you do for work.", cue="Job → ?",
        common_errors=[{"id": "no_article", "error_forms": ["i am {job}"],
                        "correction_example": "i am a teacher", "feedback": "One small word is missing.",
                        "why": "Jobs take a or an."}]))
    data["model_dialogue"].append({"speaker": "Customer", "text": "I am a teacher."})
    data["model_dialogue"].append({"speaker": "Vendor", "text": "Nice."})
    with pytest.raises(GenerationFailed) as exc:
        ScenarioGenerator(None, ScriptedModel([data, data]), CFG, clock=lambda: 1.0).build(
            OPTION, course_for(tmp_path))
    problems = exc.value.log[-1]["problems"]
    assert any("matches the correct answer" in p for p in problems), problems
    assert not list(course_for(tmp_path).generated_dir.glob("*"))   # nothing half-written


def test_an_unavailable_build_model_hands_over_to_the_backup(tmp_path):
    main = ScriptedModel([LLMUnavailable("api_error:ServerError:503")])
    backup = ScriptedModel([GOOD])
    gen_ = ScenarioGenerator(None, main, CFG, clock=lambda: 1.0, build_fallback_model=backup)
    result = gen_.build(OPTION, course_for(tmp_path))
    assert result.attempts == 2 and len(backup.calls) == 1


def test_two_failures_raise_and_write_nothing(tmp_path):
    course = course_for(tmp_path)
    with pytest.raises(GenerationFailed) as exc:
        gen(["not json", LLMUnavailable("timeout")]).build(OPTION, course)
    assert [e["problems"][0] for e in exc.value.log] == ["not_json", "llm_unavailable:timeout"]
    assert course.generated_ids() == ()


def test_build_refuses_existing_scenarios_and_a_disabled_generator(tmp_path):
    existing = LessonOption("opt-x", "t", "c", "s", (), "w", 5, scenario_id="ordering_food")
    with pytest.raises(ValueError):
        gen([GOOD]).build(existing, course_for(tmp_path))
    with pytest.raises(GenerationFailed):
        ScenarioGenerator(cfg=CFG).build(OPTION, course_for(tmp_path))
