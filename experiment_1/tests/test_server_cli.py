"""Roleplay through the server pipeline, and the learner CLI end to end. Offline only:
stub judge, ScriptedModel partner and generator."""

import io
from pathlib import Path

from guided_practice.course import load_course
from pedagogy_server.cli import CLI, main
from pedagogy_server.generator import ScenarioGenerator, learner_course
from pedagogy_server.judge import StubJudge
from pedagogy_server.llm import ScriptedModel
from pedagogy_server.pipeline import DecisionLogger, PedagogyServer
from pedagogy_server.roleplay_llm import LLMRoleplay
from tests.test_server_generator import GOOD, GOOD_PLAN

COURSE = load_course("english_for_hindi_a1")
FILL = {"{name}": "Ravi", "{place}": "Dehradun", "{job}": "student", "{interest}": "cricket"}


def answer(item):
    text = item.model_answer
    for slot, value in FILL.items():
        text = text.replace(slot, value)
    return text


def line(text, done=False):
    return {"partner_line": text, "done": done}


def run_lesson(srv, roleplay_lines=("Hello there.", "Yes, here.", "Okay.", "Thanks, bye.")):
    reply = srv.start()
    said = iter(roleplay_lines)
    for _ in range(100):
        if srv.done:
            return reply
        frame = srv.frame()
        if frame.expects == "continue":
            reply = srv.handle("")
        elif frame.stage == "roleplay":
            reply = srv.handle(next(said))
        elif frame.awaiting_follow_up:
            reply = srv.handle("fine thanks")
        else:
            reply = srv.handle(answer(srv.lesson.guided_session.current_item))
    raise AssertionError("lesson did not end")


def test_the_llm_partner_runs_inside_the_pipeline_and_every_line_is_logged(tmp_path):
    model = ScriptedModel([line("Welcome! Here is the menu."), line("Of course."),
                           line("Say: Can I have the menu, please?"),   # a correction: rejected
                           line("It is forty rupees."), line("Goodbye!", done=True)])
    srv = PedagogyServer(COURSE, "ordering_food", "l1", tmp_path, StubJudge(COURSE),
                         roleplay=LLMRoleplay(COURSE, model))
    reply = run_lesson(srv)
    assert reply.expects == "done"
    log = DecisionLogger.read(Path(tmp_path) / "decisions.jsonl")
    partner = [r["data"]["partner"] for r in log if r["kind"] == "turn" and r["data"].get("partner")]
    assert [p["source"] for p in partner][:3] == ["llm", "llm", "fallback"]
    assert "correction_marker" in partner[2]["rejections"]
    roleplay = next(r for r in log if r["kind"] == "policy" and r["stage"] == "roleplay")
    assert roleplay["data"]["stage_status"] == "completed"
    assert srv.roleplay.session.learner_turns == 4


def test_cli_plans_builds_and_runs_a_generated_lesson(tmp_path):
    course = learner_course(COURSE, tmp_path, "me")
    generator = ScenarioGenerator(ScriptedModel([GOOD_PLAN]), ScriptedModel([GOOD]), clock=lambda: 1.0)
    answers = iter(["I want to buy vegetables", "1", "", "", "",
                    "i want two kilo tomato please",          # the listed L1-transfer error
                    "i want two kilos of tomatoes please", "ok",
                    "how much are these", "ok",
                    "thank you very much",
                    "hello", "how much are these", "thanks", "bye"])
    out = io.StringIO()
    cli = CLI(course, "me", tmp_path, StubJudge(course), generator, "scripted",
              ask=lambda prompt: next(answers), out=out, show_decisions=True)
    scenario_id = cli.choose()
    assert scenario_id in course.generated_ids()
    server = cli.lesson(scenario_id)
    text = out.getvalue()
    assert "1. Buying vegetables  (new lesson" in text
    assert "Ready: Buying vegetables at the market (3 things to say)." in text
    assert "[judge: surface_error]" in text
    assert server.lesson.status == "complete"
    assert "Work on next: After two, English nouns take -s." in text
    assert "Conversation: you spoke" in text


def test_cli_offline_with_an_existing_lesson_and_quit(tmp_path):
    answers = iter(["", ":quit"])
    out = io.StringIO()
    code = main(["--offline", "--state-dir", str(tmp_path), "--scenario", "ordering_food"],
                ask=lambda prompt: next(answers), out=out)
    assert code == 0
    assert "[intro]" in out.getvalue() and "Bye." in out.getvalue()


def test_cli_planner_fallback_offers_existing_lessons(tmp_path):
    answers = iter(["ordering food", "1", ":quit"])
    out = io.StringIO()
    main(["--offline", "--state-dir", str(tmp_path)], ask=lambda prompt: next(answers), out=out)
    text = out.getvalue()
    assert "Ordering food  (existing lesson" in text
    assert "[intro]" in text


def test_a_repeat_with_nothing_due_says_why_practice_is_skipped(tmp_path):
    def run():
        answers = iter(["", "", "", "can i have the menu please", "ok", "i agree", "ok",
                        "how much is it", "ok", "she is vegetarian", "ok", "a", "b", "c", "d"])
        out = io.StringIO()
        main(["--offline", "--state-dir", str(tmp_path), "--scenario", "ordering_food"],
             ask=lambda prompt: next(answers), out=out)
        return out.getvalue()
    first, second = run(), run()
    assert "[guided_practice]" in first and "Nothing in this lesson is due" not in first
    assert "[guided_practice]" not in second and "Nothing in this lesson is due" in second
