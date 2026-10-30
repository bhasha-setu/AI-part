"""Step 7: the text terminal harness."""

import io

from guided_practice.course import load_course
from pedagogy_server.adapters import StubLaptopClient, StubTTS
from pedagogy_server.judge import StubJudge
from pedagogy_server.pipeline import PedagogyServer
from pedagogy_server.terminal import main, run

COURSE = load_course("english_for_hindi_a1")

LESSON = ["", "", "",
          "my name Ravi", "my name is Ravi", "fine thanks",
          "I'm from Dehradun",
          "I am a student",
          "I like cricket"]


def _client(tmp_path):
    srv = PedagogyServer(COURSE, "introducing_yourself", "l1", tmp_path, StubJudge(COURSE))
    return StubLaptopClient(srv, tts=StubTTS())


def test_terminal_runs_a_lesson_from_typed_lines(tmp_path):
    out = io.StringIO()
    client = _client(tmp_path)
    sent = run(client, LESSON, out)
    text = out.getvalue()
    assert sent <= len(LESSON)
    assert "[intro]" in text
    assert "[guided_practice]" in text
    assert client.server.done
    assert len(client.tts.spoken) == sent + 1     # every reply went through TTS


def test_terminal_quits_and_prints_the_frame(tmp_path):
    out = io.StringIO()
    client = _client(tmp_path)
    assert run(client, ["", ":frame", ":quit", "never sent"], out) == 1
    assert "'stage': 'model_dialogue'" in out.getvalue()


def test_terminal_refuses_the_real_judge_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("pedagogy_server.terminal.load_dotenv", lambda *a, **k: [])
    assert main(["--judge", "gemini"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
