"""Fresh lesson versions for redos. Offline: scripted models only."""

import json

import pytest

from cascade.tutor_brain import LLMUnavailable, TutorFactory, use_tutor_packages
from cascade.variants import check_variant, focus_note, make_variant, past_mistakes

use_tutor_packages()
from guided_practice.course import load_course  # noqa: E402

COURSE = load_course("english_for_hindi_a1")
LESSON = COURSE.load_scenario("introducing_yourself")
IDS = [it.id for it in LESSON.items]
ATTEMPTS = [{"missed": ["intro_work"], "reviewed": [
    {"said": "I am student", "fixed": "I am a student", "type": "dropped_article"}]}]

GOOD = {
    "setting": "Your first day at a new office; a colleague comes to say hello.",
    "partner": "Meera",
    "prompts": [{"id": "intro_name", "prompt": "Your new colleague wants to know what to call you."},
                {"id": "intro_origin", "prompt": "She asks which city your family lives in."},
                {"id": "intro_work", "prompt": "She asks about your role in the team."},
                {"id": "intro_interest", "prompt": "Tell her about a hobby you enjoy."}],
    "dialogue": [{"speaker": "Meera", "text": "Hi, I'm Meera. Are you new here?"},
                 {"speaker": "Ravi", "text": "Yes! My name is Ravi."},
                 {"speaker": "Meera", "text": "Where are you from, Ravi?"},
                 {"speaker": "Ravi", "text": "I'm from Pune. I am a designer. I like music."}],
}


class Scripted:
    name = "scripted"

    def __init__(self, *outs):
        self.outs, self.users = list(outs), []

    def generate_json(self, *, system, user, schema):
        self.users.append(json.loads(user))
        out = self.outs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, str) else json.dumps(out)


def test_past_mistakes_and_focus_note():
    missed, reviewed = past_mistakes(ATTEMPTS)
    assert missed == ["intro_work"] and reviewed[0]["fixed"] == "I am a student"
    note = focus_note(LESSON, missed, reviewed)
    assert "I am a {job}" in note and "\"I am student\" -> \"I am a student\"" in note
    assert "do not correct" in note
    assert focus_note(LESSON, [], []) == ""


def test_good_variant_changes_situations_but_never_targets():
    scn, note, meta = make_variant(LESSON, ATTEMPTS, Scripted(GOOD), COURSE.lang)
    assert meta["source"] == "llm" and meta["partner"] == "Meera"
    assert [it.id for it in scn.items][0] == "intro_work"           # missed phrase first
    by_id = {it.id: it for it in scn.items}
    for orig in LESSON.items:
        new = by_id[orig.id]
        assert (new.model_answer, new.accepted, new.first_words, new.cue) == \
               (orig.model_answer, orig.accepted, orig.first_words, orig.cue)
        assert new.prompt != orig.prompt
        assert new.match_accepted("my name is ravi") == orig.match_accepted("my name is ravi")
    assert scn.model_dialogue[0].speaker == "Meera"
    assert "first day at a new office" in scn.can_do
    assert LESSON.items[0].prompt == "Someone asks your name. Say it."   # original untouched


@pytest.mark.parametrize("change, problem", [
    (lambda d: d["prompts"].pop(), "exactly these ids"),
    (lambda d: d["prompts"][0].update(prompt="Say: My name is Ravi."), "gives the answer away"),
    (lambda d: d["prompts"][1].update(prompt="Say I'm from your city."), "gives the answer away"),
    (lambda d: d.update(setting="नया ऑफिस"), "Latin script"),
    (lambda d: d["dialogue"].reverse(), "partner must speak the first"),
    (lambda d: d.update(dialogue=d["dialogue"][:1]), "dialogue must have"),
])
def test_bad_variants_are_caught(change, problem):
    data = json.loads(json.dumps(GOOD))
    change(data)
    assert any(problem in p for p in check_variant(data, LESSON, COURSE.lang))


def test_one_retry_with_problems_then_fallback():
    leaky = json.loads(json.dumps(GOOD))
    leaky["prompts"][0]["prompt"] = "Say: My name is Ravi."
    model = Scripted(leaky, GOOD)
    scn, _, meta = make_variant(LESSON, ATTEMPTS, model, COURSE.lang)
    assert meta["source"] == "llm" and "fix_these" in model.users[1]

    scn, note, meta = make_variant(LESSON, ATTEMPTS, Scripted(leaky, leaky), COURSE.lang)
    assert meta["source"] == "original" and meta["reason"] == "failed_checks"
    assert scn.items[0].id == "intro_work" and note                # still personalised


def test_unavailable_or_offline_falls_back_to_reordered_original():
    for model in (Scripted(LLMUnavailable("api_error:429")), None):
        scn, note, meta = make_variant(LESSON, ATTEMPTS, model, COURSE.lang)
        assert meta["source"] == "original"
        assert [it.prompt for it in scn.items][0] == LESSON.item_by_id("intro_work").prompt


def test_focus_goes_into_this_lessons_prompt_only(tmp_path):
    factory = TutorFactory({"overrides": {"hindi_script": "devanagari"}}, {"providers": []}, offline=True)
    focused = factory.lesson(LESSON, COURSE, learner_id="t", state_dir=tmp_path, focus="FOCUS NOTE")
    plain = factory.lesson(LESSON, COURSE, learner_id="t", state_dir=tmp_path)
    assert "FOCUS NOTE" in focused.cfg.system_prompt
    assert "FOCUS NOTE" not in plain.cfg.system_prompt
