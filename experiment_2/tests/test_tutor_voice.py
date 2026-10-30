"""The experiment_1 tutor as the voice pipeline's LLM stage. Offline: no network, no audio."""

import json
from pathlib import Path

import pytest
import yaml

from cascade import cli
from cascade.pipeline import CascadePipeline
from cascade.stt import TypedSTT
from cascade.tts import SilentTTS
from cascade.tutor_brain import (JSONChain, LLMUnavailable, TutorBrain, VoiceGuard,
                                 load_voice_tutor_config)

CORRECT = ["my name is Ravi", "I am from Delhi", "I am a student", "I like cricket"]
CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
VOICE = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["tutor"]["overrides"]


class RecordingTTS(SilentTTS):
    def __init__(self):
        self.spoken = []

    def speak(self, text, language, play=True):
        self.spoken.append((language, text))
        return 1.0


class Failing:
    def __init__(self, name, reason="api_error:ClientError:429"):
        self.name, self.reason, self.calls = name, reason, 0

    def generate_json(self, *, system, user, schema):
        self.calls += 1
        raise LLMUnavailable(self.reason)


class Fixed:
    def __init__(self, name, out):
        self.name, self.out = name, out

    def generate_json(self, *, system, user, schema):
        return json.dumps(self.out, ensure_ascii=False)


@pytest.fixture
def cfg():
    return load_voice_tutor_config(VOICE)


def brain(cfg, model, tmp_path, stage=2):
    return TutorBrain(cfg, model, course_id="english_for_hindi_a1", scenario_id=None,
                      learner_id="t", state_dir=tmp_path, stage=stage)


def test_voice_config_writes_hindi_in_devanagari(cfg):
    assert cfg.hindi_script == "devanagari" and cfg.thinking_level == "minimal"
    assert "Write Hindi in Devanagari" in cfg.system_prompt
    assert "Roman letters" not in cfg.system_prompt
    assert any("ऀ" <= ch <= "ॿ" for ch in cfg.hi("intro_hi"))


def test_offline_lesson_runs_through_the_pipeline_and_saves(cfg, tmp_path):
    tts = RecordingTTS()
    tutor = brain(cfg, None, tmp_path)
    pipe = CascadePipeline(TypedSTT(), tutor, tts, stop_words={"quit"})
    greeting = pipe.greet()
    assert greeting.reply and tts.spoken            # the tutor speaks first
    assert greeting.tutor["phase"] == "intro"
    first = pipe.turn("मेरा नाम रवि है")              # Hindi (as Whisper writes it): bridged
    assert first.tutor["source"] == "fallback" and not first.done
    for answer in CORRECT:
        pipe.turn(answer)
    result = None
    for _ in range(30):
        result = pipe.turn("ok")
        if result.done:
            break
    assert result.done
    report = tutor.finish()
    assert any("From memory" in line for line in report)
    assert (tmp_path / "profiles.json").exists() and (tmp_path / "learners.json").exists()
    assert (tmp_path / "tutor_log.jsonl").exists()


def test_spoken_quit_is_not_graded(cfg, tmp_path):
    tutor = brain(cfg, None, tmp_path)
    pipe = CascadePipeline(TypedSTT(), tutor, SilentTTS(), stop_words={"quit"})
    pipe.greet()
    result = pipe.turn("Quit.")
    assert result.stopped and not result.reply
    assert tutor.session.record.answers == []


def test_every_key_failing_falls_back_to_rules_without_crashing(cfg, tmp_path):
    a, b = Failing("gemini[key1]"), Failing("gemini[key2]")
    tutor = brain(cfg, JSONChain([a, b]), tmp_path)
    pipe = CascadePipeline(TypedSTT(), tutor, RecordingTTS())
    assert pipe.greet().reply
    result = pipe.turn("my name Ravi")
    assert result.reply and result.tutor["source"] == "fallback" and result.llm == "rules"
    # Both were benched by the greeting; with every key benched they are tried anyway.
    assert a.calls == 2 and b.calls == 2


def test_chain_uses_next_key_and_reports_it(cfg, tmp_path):
    intro = {"say": [{"lang": "en", "text": "Hi! Today: introduce yourself."},
                     {"lang": "hi", "text": "आज हम परिचय देना सीखेंगे।"},
                     {"lang": "en", "text": "Someone asks your name. Say it."}]}
    tutor = brain(cfg, JSONChain([Failing("gemini[key1]"), Fixed("gemini[key2]", intro)]), tmp_path)
    tts = RecordingTTS()
    greeting = CascadePipeline(TypedSTT(), tutor, tts).greet()
    assert greeting.llm == "gemini[key2]" and greeting.tutor["source"] == "llm"
    assert [lang for lang, _ in tts.spoken] == ["en", "en", "hi", "en", "en"]   # one per sentence


def test_max_tokens_does_not_bench_the_key():
    now = [0.0]
    a = Failing("k1", reason="max_tokens")
    chain = JSONChain([a, Fixed("k2", {"say": []})], clock=lambda: now[0])
    chain.generate_json(system="", user="", schema={})
    chain.generate_json(system="", user="", schema={})
    assert a.calls == 2


def test_cli_tutor_text_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "EXPERIMENT_ROOT", tmp_path)
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    answers = iter(CORRECT + ["ok"] * 30)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.main(["--tutor", "--text", "--offline", "--no-tts", "--config", str(cfg_path)]) == 0
    assert (tmp_path / "runs" / "tutor_state" / "profiles.json").exists()
    [log] = (tmp_path / "runs").glob("*.jsonl")
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["tutor"]["phase"] == "practice" and records[-1]["done"]


# --- voice rules: richer answers, transcript artefacts, conversation length -------------------

def test_voice_rules_and_conversation_length_are_loaded(cfg):
    assert "speech-recognition transcript" in cfg.system_prompt
    assert "Never ask the learner to make an answer shorter" in cfg.system_prompt
    assert "Write Hindi in Devanagari" in cfg.system_prompt   # the base prompt is kept
    assert (cfg.converse_min_turns, cfg.converse_max_turns) == (3, 12)


def test_voice_guard_drops_transcript_artefacts_only():
    out = {"say": [{"lang": "en", "text": "Nice!"}], "learner_language": "en",
           "wants_to_end": False,
           "errors": [{"said": "Dayadun Utrakan", "fixed": "Dehradun, Uttarakhand",
                       "type": "spelling", "importance": 1},
                      {"said": "I study in CSE course", "fixed": "I study CSE",
                       "type": "preposition", "importance": 2}]}
    guard = VoiceGuard(Fixed("k", out))
    converse_schema = {"properties": {"errors": {}, "say": {}}}
    kept = json.loads(guard.generate_json(system="", user="", schema=converse_schema))["errors"]
    assert [e["type"] for e in kept] == ["preposition"]
    assert guard.dropped[0]["said"] == "Dayadun Utrakan"
    practice_schema = {"properties": {"verdict": {}, "say": {}}}
    assert json.loads(guard.generate_json(system="", user="", schema=practice_schema)) == out


class PhaseModel:
    """Answers each phase like a cooperative model: practice correct, conversation keeps
    going (wants_to_end false) and reports one spelling slip plus one real mistake."""
    name = "phase-model"

    def __init__(self):
        self.converse_calls = 0

    def generate_json(self, *, system, user, schema):
        req = json.loads(user)
        if req["phase"] == "intro":
            return json.dumps({"say": [{"lang": "en", "text": "Hi! Say your name."}]})
        if req["phase"] == "practice":
            return json.dumps({"learner_language": "en", "verdict": "correct",
                               "say": [{"lang": "en", "text": "Great. Next one."}]})
        if req["phase"] == "conversation":
            self.converse_calls += 1
            return json.dumps({"learner_language": "en", "wants_to_end": False,
                               "say": [{"lang": "en", "text": "Tell me more about that."}],
                               "errors": [{"said": "Dayadun", "fixed": "Dehradun",
                                           "type": "spelling", "importance": 3},
                                          {"said": "I am student", "fixed": "I am a student",
                                           "type": "article", "importance": 2}]})
        return json.dumps({"fixed_it": True, "say": [{"lang": "en", "text": "Well done."}]})


def test_engaged_conversation_goes_past_five_turns_and_reviews_no_artefacts(cfg, tmp_path):
    model = PhaseModel()
    tutor = brain(cfg, VoiceGuard(model), tmp_path)
    pipe = CascadePipeline(TypedSTT(), tutor, SilentTTS())
    pipe.greet()
    for answer in CORRECT:
        pipe.turn(answer)
    assert tutor.session.phase == "converse"
    for _ in range(cfg.converse_max_turns):
        pipe.turn("I live in Dayadun and I am student, I like building AI projects.")
        if tutor.session.phase != "converse":
            break
    assert model.converse_calls == cfg.converse_max_turns    # 12, not the old 5
    assert all(e["type"] != "spelling" for e in tutor.session.converse_errors)
    assert {e["type"] for e in tutor.session.review_queue} == {"article"}
