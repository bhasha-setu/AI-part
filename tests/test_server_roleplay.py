"""LLM roleplay partner: validation, fallback, termination, roles, lesson integration."""

import json

import pytest

from guided_practice.config import audit_thresholds
from guided_practice.course import load_course
from guided_practice.lesson import Lesson
from guided_practice.roleplay import StubRoleplay
from pedagogy_server.llm import LLMUnavailable, ScriptedModel
from pedagogy_server.roleplay_llm import (
    LLMRoleplay, detect_roles, load_roleplay_config, make_roleplay)

EN = "english_for_hindi_a1"
HI = "hindi_for_english_a1"


def good(line="Nice. What do you do?", done=False):
    return {"partner_line": line, "done": done}


def start(outputs, course_id=EN, scenario="ordering_food"):
    course = load_course(course_id)
    model = ScriptedModel(outputs)
    rp = LLMRoleplay(course, model).start(course.load_scenario(scenario), "l1", [])
    return rp, model


def play(rp, n=10):
    while not rp.done and n:
        rp.submit("hello")
        n -= 1
    return rp


def test_config_audit_passes():
    from guided_practice.config import CONFIG_DIR
    import yaml
    raw = yaml.safe_load((CONFIG_DIR / "roleplay.yaml").read_text(encoding="utf-8"))
    assert audit_thresholds(raw) == []
    cfg = load_roleplay_config()
    assert cfg.max_learner_turns == 4 and cfg.min_learner_turns == 2


def test_roles_both_courses():
    en = load_course(EN)
    assert detect_roles(en.load_scenario("ordering_food")) == ("Waiter", "Ravi")
    assert detect_roles(en.load_scenario("introducing_yourself")) == ("Asha", "Ravi")
    hi = load_course(HI)
    # A says "मेरा नाम राम है।", the first item's answer, so A is the learner side.
    assert detect_roles(hi.load_scenario("introducing_myself_hi")) == ("B", "A")
    assert detect_roles(en.load_scenario("ordering_food"), "Ravi") == ("Ravi", "Waiter")


def test_full_conversation_ends_at_max_turns_with_closing():
    rp, model = start([good("Welcome! Here is the menu."), good("Tea is nice."),
                       good("It is forty rupees."), good("Anything else?"),
                       good("Thank you. Goodbye!", done=True)])
    assert rp.prompt == "Welcome! Here is the menu."
    play(rp)
    assert rp.learner_turns == 4 and rp.done
    assert rp.transcript[-1]["text"] == "Thank you. Goodbye!"
    assert [t["speaker"] for t in rp.transcript] == ["Waiter", "Ravi"] * 4 + ["Waiter"]
    assert all(set(t) == {"speaker", "text", "audio"} and t["audio"] is None
               for t in rp.transcript)
    assert all(e["source"] == "llm" for e in rp.log)
    assert json.loads(model.calls[-1]["user"])["final"] is True
    res = rp.result()
    assert res.status == "completed" and "4 learner turns; 0 fallback" in res.reason
    with pytest.raises(RuntimeError):
        rp.submit("x")


def test_done_before_min_turns_ignored_then_honoured():
    rp, _ = start([good(), good("Okay.", done=True), good("Bye.", done=True)])
    rp.submit("a")
    assert not rp.done            # 1 < min (2)
    rp.submit("b")
    assert rp.done and rp.learner_turns == 2


def test_system_prompt_and_request_carry_scenario():
    rp, model = start([good()])
    call = model.calls[0]
    req = json.loads(call["user"])
    assert req["scenario"]["title"] == "Ordering food"
    assert req["opening"] is True and req["partner_role"] == "Waiter"
    assert any("menu" in g.lower() for g in req["goals"])
    assert "Never correct" in call["system"] and "A1" in call["system"]


BAD = [
    ("not json", "not_json"),
    ("[1]", "bad_shape"),
    ({"partner_line": "Hi.", "done": False, "x": 1}, "bad_shape"),
    ({"partner_line": "Hi."}, "bad_shape"),
    ({"partner_line": 3, "done": False}, "bad_types"),
    ({"partner_line": "Hi.", "done": "no"}, "bad_types"),
    ({"partner_line": "   ", "done": False}, "empty"),
    ({"partner_line": "Hello there. How are you. Fine. Good.", "done": False},
     "too_many_sentences"),
    ({"partner_line": "x" * 130, "done": False}, "too_long"),
    ({"partner_line": "You should say: I am fine.", "done": False}, "correction_marker"),
    ({"partner_line": "आप कैसे हैं?", "done": False}, "wrong_script"),
    ({"partner_line": "Can I have the menu, please?", "done": False}, "reveals_goal"),
]


@pytest.mark.parametrize("output,code", BAD)
def test_each_rule_rejects_and_falls_back(output, code):
    rp, _ = start([output])
    assert rp.log[0]["source"] == "fallback"
    assert code in rp.log[0]["rejections"]
    assert rp.prompt == "Welcome! Here is the menu."


def test_fallback_walks_script_then_closes():
    rp, _ = start([LLMUnavailable("timeout")] * 5)
    assert rp.log[0]["rejections"] == ["llm_unavailable:timeout"]
    rp.submit("a")
    assert rp.prompt == "Our masala chai is very good today."
    rp.submit("b")
    assert rp.prompt == "It is forty rupees."
    rp.submit("c")
    assert rp.prompt == "Thank you. Goodbye!" and rp.done


def test_always_failing_model_ends_all_courses():
    for cid, sid in ((EN, "introducing_yourself"), (EN, "ordering_food"),
                     (HI, "introducing_myself_hi")):
        rp, _ = start(["garbage"] * 20, cid, sid)
        play(rp)
        assert rp.done and rp.learner_turns <= 4
        assert rp.result().status == "completed"
        assert all(e["source"] == "fallback" for e in rp.log)


def test_llm_opening_is_not_repeated_by_fallback():
    rp, _ = start([good("Hello! Welcome."), "bad"])
    rp.submit("x")
    assert rp.prompt == "Our masala chai is very good today."


def test_raw_truncated_in_log():
    rp, _ = start(["y" * 5000])
    assert len(rp.log[0]["raw"]) == 2000


def test_hindi_course_script_check():
    rp, _ = start([good("Hello!")], HI, "introducing_myself_hi")
    assert "wrong_script" in rp.log[0]["rejections"]   # Latin forbidden for hi target
    rp, _ = start([good("नमस्ते! आप कैसे हैं?")], HI, "introducing_myself_hi")
    assert rp.log[0]["source"] == "llm"


def test_make_roleplay_kinds():
    course = load_course(EN)
    assert isinstance(make_roleplay("stub", course), StubRoleplay)
    sc = make_roleplay("scripted", course).start(course.load_scenario("ordering_food"), "l", [])
    play(sc)
    assert sc.done and sc.log[0]["rejections"] == ["no_model"]
    with pytest.raises(ValueError):
        make_roleplay("nope", course)


def test_lesson_integration(tmp_path):
    course = load_course(EN)
    model = ScriptedModel([good("Welcome! Here is the menu."), good("Tea is good."),
                           good("It is forty rupees."), good("Anything else?"),
                           good("Thank you. Goodbye!", done=True)])
    lesson = Lesson(course, "ordering_food", "l1", tmp_path / "learners.json",
                    tmp_path / "log.jsonl", roleplay=LLMRoleplay(course, model))
    view = lesson.start()
    seen_roleplay, turns = False, 0
    while view.expects not in ("done", "held"):
        turns += 1
        assert turns < 300
        if view.stage == "roleplay":
            seen_roleplay = True
            assert view.expects == "answer" and view.prompt
            view = lesson.submit("I like tea")
        elif view.expects == "continue":
            view = lesson.continue_()
        elif view.follow_up:
            view = lesson.submit("ok")
        else:
            item = lesson.scenario.item_by_id(view.item_id)
            view = lesson.submit(item.model_answer.replace("{name}", "Ravi"))
    assert seen_roleplay
    rec = next(r for r in lesson.records if r.stage == "roleplay")
    assert rec.status == "completed" and rec.detail["turns"] == 9
