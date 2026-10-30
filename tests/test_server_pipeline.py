"""Step 6: the turn pipeline, end to end, with stub and scripted judges."""

import ast
import json
from pathlib import Path

import pytest

from guided_practice.config import PROJECT_ROOT
from guided_practice.course import load_course
from guided_practice.evaluator import RuleEvaluator, Verdict
from guided_practice.turn_log import JsonlLogger
from pedagogy_server.adapters import StubLaptopClient, StubTTS
from pedagogy_server.config import load_judge_config, load_reply_strings
from pedagogy_server.judge import ScriptedJudge, StubJudge
from pedagogy_server.pipeline import DecisionLogger, PedagogyServer

COURSE = load_course("english_for_hindi_a1")
STRINGS = load_reply_strings(COURSE.ui.code)
CFG = load_judge_config()
FILL = {"{name}": "Ravi", "{place}": "Dehradun", "{job}": "student", "{interest}": "cricket"}


def model_answer(item):
    text = item.model_answer
    for slot, value in FILL.items():
        text = text.replace(slot, value)
    return text


def server(tmp_path, judge=None, scenario="introducing_yourself"):
    return PedagogyServer(COURSE, scenario, "l1", tmp_path, judge or StubJudge(COURSE),
                          session_id="test-sess")


def drive(srv, answer, max_turns=100):
    """Continue through presentations; answer guided practice with answer(item, frame)."""
    reply = srv.start()
    for _ in range(max_turns):
        if srv.done:
            return reply
        frame = srv.frame()
        if frame.expects == "continue":
            reply = srv.handle("okay")
        elif frame.awaiting_follow_up:
            reply = srv.handle("fine thanks")
        else:
            item = srv.lesson.guided_session.current_item
            reply = srv.handle(answer(item, frame))
    raise AssertionError("lesson did not end")


def decisions(tmp_path):
    return DecisionLogger.read(Path(tmp_path) / "decisions.jsonl")


def test_full_lesson_with_the_stub_judge(tmp_path):
    srv = server(tmp_path)
    reply = drive(srv, lambda item, f: model_answer(item))
    assert reply.expects == "done"
    assert STRINGS.stage_lines["complete"] in reply.text
    log = decisions(tmp_path)
    policy = [r for r in log if r["kind"] == "policy"]
    assert [r["stage"] for r in policy] == ["intro", "model_dialogue", "useful_language",
                                            "guided_practice", "roleplay", "assessment"]
    assert all(r["reason"] for r in policy)
    judged = [r for r in log if r["kind"] == "turn" and "judge" in r["data"]]
    assert judged and all(r["data"]["judge"]["source"] == "judge" for r in judged)
    assert log[-1]["kind"] == "lesson_end"


def test_the_session_log_shows_what_the_learner_heard(tmp_path):
    srv = server(tmp_path)
    drive(srv, lambda item, f: model_answer(item))
    heard = [r["data"]["judge"]["feedback"] for r in decisions(tmp_path)
             if r["kind"] == "turn" and "judge" in r["data"]]
    scored = [r["feedback_shown"] for r in JsonlLogger.read(Path(tmp_path) / "turns.jsonl")
              if r["turn_type"] == "scored"]
    assert scored == heard


def test_rejected_judge_output_falls_back_and_is_logged(tmp_path):
    srv = server(tmp_path, judge=ScriptedJudge(["not json"] * 50))
    srv.start()
    for _ in range(3):
        srv.handle("ok")
    reply = srv.handle("my name is Ravi")
    turn = [r for r in decisions(tmp_path) if r["kind"] == "turn" and "judge" in r["data"]][0]
    assert turn["data"]["judge"]["source"] == "fallback"
    assert turn["data"]["judge"]["rejections"] == ["malformed_json"]
    assert turn["reason"] == "fallback: correct"
    assert reply.feedback == STRINGS.guided("correct", 0)


def test_low_confidence_advances_the_ladder(tmp_path):
    low = {"verdict": "correct", "error_id": None, "correction": None,
           "reply": "Great.", "confidence": 0.1}
    srv = server(tmp_path, judge=ScriptedJudge([low]))
    srv.start()
    for _ in range(3):
        srv.handle("ok")
    assert srv.frame().hint_level == 0
    srv.handle("Ravi here")                       # not an accepted form
    assert srv.frame().hint_level == 1
    turn = [r for r in decisions(tmp_path) if r["kind"] == "turn" and "judge" in r["data"]][0]
    assert turn["data"]["judge"]["source"] == "judge:low_confidence"
    assert turn["data"]["judge"]["verdict"] == "meaning_error"


def test_the_server_not_the_judge_decides_the_level(tmp_path):
    srv = server(tmp_path)
    srv.start()
    for _ in range(3):
        srv.handle("ok")
    srv.handle("the weather is nice")
    request = [r for r in decisions(tmp_path) if r["kind"] == "turn" and "judge" in r["data"]][0]["data"]["request"]
    assert request["hint_level"] == 0
    assert srv.frame().hint_level == 1             # Session climbed the ladder, not the judge


def test_a_retry_without_a_correction_does_not_claim_one_was_shown(tmp_path):
    srv = server(tmp_path)
    srv.start()
    for _ in range(3):
        srv.handle("ok")
    item = srv.lesson.guided_session.current_item
    reply = srv.handle("my name Ravi")            # level 0: the recast is held back
    assert reply.feedback == STRINGS.encourage
    assert reply.prompt == STRINGS.retry_without_correction.format(prompt=item.prompt)
    assert COURSE.ui.retry_prompt.split("{")[0].strip() not in reply.text


def test_a_retry_after_a_shown_correction_keeps_the_session_prompt(tmp_path):
    srv = server(tmp_path, judge=PromptStyleJudge(COURSE))
    srv.start()
    for _ in range(3):
        srv.handle("ok")
    item = srv.lesson.guided_session.current_item
    reply = srv.handle("my name Ravi")
    assert reply.prompt == COURSE.ui.retry_prompt.format(prompt=item.prompt)


class PromptStyleJudge(StubJudge):
    """Rule verdicts, with a prompt-style correction (never a recast) on every surface error."""

    def judge(self, request):
        out = json.loads(super().judge(request))
        if out["verdict"] == "surface_error":
            out["correction"] = "Check one small word."
        return json.dumps(out)


def test_correction_cap_and_once_per_error_hold_across_the_lesson(tmp_path):
    first_try = set()

    def answer(item, frame):
        if item.id not in first_try and item.common_errors:
            first_try.add(item.id)
            return item.common_errors[0].example
        return model_answer(item)

    srv = server(tmp_path, judge=PromptStyleJudge(COURSE))
    drive(srv, answer)
    turns = [r["data"]["judge"] for r in decisions(tmp_path)
             if r["kind"] == "turn" and "judge" in r["data"]]
    shown = [t for t in turns if t["correction_shown"]]
    capped = [t for t in turns if "correction_cap_reached" in t["guardrail_actions"]]
    assert len(shown) == CFG.corrections_per_session
    assert len(first_try) > CFG.corrections_per_session and capped
    assert capped[0]["feedback"] == STRINGS.encourage
    assert len({t["error_id"] for t in shown}) == len(shown)   # each error id once


def test_follow_up_answers_skip_the_judge(tmp_path):
    judge = StubJudge(COURSE)
    srv = server(tmp_path, judge=judge)
    srv.start()
    for _ in range(3):
        srv.handle("ok")
    item = srv.lesson.guided_session.current_item
    srv.handle(model_answer(item))
    assert srv.frame().awaiting_follow_up      # intro_name has a follow-up question
    calls = len(judge.calls)
    srv.handle("I am fine")
    assert len(judge.calls) == calls


def test_handle_needs_start_and_stops_at_the_end(tmp_path):
    srv = server(tmp_path)
    with pytest.raises(RuntimeError, match="start"):
        srv.handle("hi")
    drive(srv, lambda item, f: model_answer(item))
    with pytest.raises(RuntimeError, match="lesson is"):
        srv.handle("again")


@pytest.mark.parametrize("var, key", [("ANTHROPIC_API_KEY", "sk-ant-test-1234-not-a-real-key"),
                                      ("GEMINI_API_KEY", "AIza-test-5678-not-a-real-key")])
def test_no_api_key_reaches_the_logs(tmp_path, monkeypatch, var, key):
    monkeypatch.setenv(var, key)
    srv = server(tmp_path)
    srv.start()
    srv.handle(f"my key is {key}")                 # even if it is spoken, it is redacted
    drive_rest = srv
    while not drive_rest.done:
        f = drive_rest.frame()
        if f.expects == "continue" or f.awaiting_follow_up:
            drive_rest.handle("ok")
        else:
            drive_rest.handle(model_answer(drive_rest.lesson.guided_session.current_item))
    text = (Path(tmp_path) / "decisions.jsonl").read_text(encoding="utf-8")
    assert key not in text
    assert "[redacted]" in text


def test_laptop_client_stub_round_trip(tmp_path):
    tts = StubTTS()
    client = StubLaptopClient(server(tmp_path), tts=tts)
    client.start()
    client.speak("okay".encode("utf-8"))
    assert len(tts.spoken) == 2 and tts.spoken[0]


AUDIO_MODULES = {"pyaudio", "sounddevice", "soundfile", "librosa", "pydub", "wave",
                 "audioop", "speech_recognition", "whisper", "pyttsx3", "simpleaudio",
                 "torchaudio", "webrtcvad", "vosk", "gtts"}


@pytest.mark.parametrize("package", ["guided_practice", "pedagogy_server"])
def test_no_audio_library_is_imported(package):
    hits = []
    for path in sorted(PROJECT_ROOT.joinpath(package).glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            hits += [f"{path.name}: {n}" for n in names if n.split(".")[0] in AUDIO_MODULES]
    assert hits == []


def test_server_has_no_hard_coded_learner_strings():
    forbidden = ["Correct!", "Let's try", "Almost", "Say:", "Well done", "Nearly"]
    hits = []
    for path in PROJECT_ROOT.joinpath("pedagogy_server").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits += [f"{path.name}: {n}" for n in forbidden if n in text]
    assert hits == []


def test_no_key_literal_in_code_or_config():
    for root in ("guided_practice", "pedagogy_server", "config"):
        for path in PROJECT_ROOT.joinpath(root).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".yaml"):
                assert "sk-ant-" not in path.read_text(encoding="utf-8"), path
