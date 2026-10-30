"""The tutor (tutor/) as the voice pipeline's LLM stage. Offline: no network, no audio."""

import json
from pathlib import Path

import pytest
import yaml

from cascade import cli
from cascade.pipeline import CascadePipeline
from cascade.stt import TypedSTT
from cascade.tts import SilentTTS
from cascade.tutor_brain import (JSONChain, LLMUnavailable, TutorBrain, VoiceGuard,
                                 load_voice_tutor_config, use_tutor_packages)

use_tutor_packages()
from guided_practice.course import load_course  # noqa: E402

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


COURSE = load_course("english_for_hindi_a1")


def brain(cfg, model, tmp_path, stage=2):
    return TutorBrain(cfg, model, scenario=COURSE.load_scenario("introducing_yourself"),
                      course=COURSE, learner_id="t", state_dir=tmp_path, hindi_help=stage)


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
    root = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "curriculum").symlink_to(root / "curriculum")
    answers = iter(CORRECT + ["ok"] * 30)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.main(["--tutor", "--text", "--offline", "--no-tts", "--user", "T1",
                     "--lesson", "introducing_yourself", "--config", str(root / "config.yaml")]) == 0
    learner = tmp_path / "runs" / "learners" / "t1"
    assert (learner / "profiles.json").exists() and (learner / "learners.json").exists()
    progress = json.loads((learner / "progress.json").read_text())
    assert progress["lessons"]["introducing_yourself"]["attempts"]
    [log] = (tmp_path / "runs").glob("*.jsonl")
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["greeting"] and records[0]["tutor"]["phase"] == "intro"
    assert records[1]["tutor"]["phase"] == "practice" and records[-1]["done"]
    assert {r["lesson"] for r in records} == {"introducing_yourself"}


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


# --- one male speaker, आप throughout ----------------------------------------------------------

def test_female_partner_is_played_by_a_man_and_roles_are_kept(cfg, tmp_path):
    from cascade.tutor_brain import TutorFactory, male_partner

    intro = COURSE.load_scenario("introducing_yourself")          # partner: Asha
    male = male_partner(intro)
    assert male.model_dialogue[0].speaker == "Arjun"
    assert all("Asha" not in t.text and t.speaker != "Asha" for t in male.model_dialogue)
    assert male.model_dialogue[1].speaker == "Ravi"                 # the learner's line is kept
    assert intro.model_dialogue[0].speaker == "Asha"                # the original is untouched
    food = COURSE.load_scenario("ordering_food")                    # partner: Waiter
    assert male_partner(food) is food
    factory = TutorFactory(yaml.safe_load(CONFIG.read_text())["tutor"], {"providers": []}, offline=True)
    lesson = factory.lesson(intro, COURSE, learner_id="t", state_dir=tmp_path)
    assert lesson.session.partner == "Arjun"


def test_hindi_lines_use_aap_and_prompt_asks_for_one_male_speaker(cfg):
    for key in ("hint_hi", "bridge_hi", "review_prompt_hi"):
        assert any(w in cfg.hi(key) for w in ("कीजिए", "बोलिए", "देखिए")), key
        assert not cfg.hi(key).endswith(("करो।", "बोलो।", "देखो।")), key
    assert cfg.hi("intro_hi") == "आज हम यह बोलना सीखेंगे। ग़लती हो तो कोई बात नहीं।"   # untouched lines kept
    assert "masculine first-person forms" in cfg.system_prompt
    assert "always address the learner as आप" in cfg.system_prompt


# --- hedging: a stalled key doesn't hold up the reply ----------------------------------------

class Slow:
    def __init__(self, name, seconds, out=None, fail=None):
        self.name, self.seconds, self.out, self.fail, self.calls = name, seconds, out, fail, 0

    def generate_json(self, *, system, user, schema):
        import time

        self.calls += 1
        time.sleep(self.seconds)
        if self.fail:
            raise LLMUnavailable(self.fail)
        return json.dumps(self.out)


def test_hedge_starts_the_next_key_when_the_first_stalls():
    import time

    chain = JSONChain([Slow("k1", 2.0, {"who": 1}), Slow("k2", 0.05, {"who": 2})],
                      hedge_after_seconds=0.1)
    start = time.perf_counter()
    assert json.loads(chain.generate_json(system="", user="", schema={})) == {"who": 2}
    assert time.perf_counter() - start < 1.0
    assert chain.last_used == "k2" and chain.hedged == 1


def test_hedge_waits_for_a_fast_first_key_and_moves_on_after_a_fast_failure():
    second = Slow("k2", 0.0, {"who": 2})
    chain = JSONChain([Slow("k1", 0.01, {"who": 1}), second], hedge_after_seconds=1.0)
    assert json.loads(chain.generate_json(system="", user="", schema={})) == {"who": 1}
    assert second.calls == 0 and chain.hedged == 0
    chain = JSONChain([Slow("k1", 0.0, fail="api_error:503"), Slow("k2", 0.0, {"who": 2})],
                      hedge_after_seconds=5.0)
    assert json.loads(chain.generate_json(system="", user="", schema={})) == {"who": 2}
    with pytest.raises(LLMUnavailable):
        JSONChain([Slow("a", 0, fail="x"), Slow("b", 0, fail="y")],
                  hedge_after_seconds=0.1).generate_json(system="", user="", schema={})


# --- tips: a better way to say the learner's sentence --------------------------------------

from cascade.tutor_brain import check_tip  # noqa: E402

CRICKET = "I like to watch cricket and also play it."


@pytest.mark.parametrize("tip, text, kept, why", [
    ({"said": CRICKET, "better": "I like watching and playing cricket.", "type": "natural"},
     CRICKET, True, ""),
    ({"said": "i am from delhi", "better": "I am from Delhi.", "type": "grammar"},
     "i am from delhi", False, "same_words"),
    ({"said": "Dayadun", "better": "Dehradun", "type": "spelling"}, "I live in Dayadun",
     False, "transcript_artefact"),
    ({"said": "x", "better": "I am from Delhi."}, "मैं दिल्ली से हूँ", False, "learner_spoke_hindi"),
    ({"said": "x", "better": "मैं दिल्ली से हूँ"}, "I from Delhi", False, "not_english_or_too_long"),
])
def test_check_tip(tip, text, kept, why):
    got, dropped = check_tip(tip, text)
    assert (got is not None, dropped) == (kept, why)


class TipModel(PhaseModel):
    """PhaseModel that also fills the tip field whenever the schema offers it."""

    def __init__(self, verdict="correct", better="I am a student."):
        super().__init__()
        self.verdict, self.better, self.systems = verdict, better, []

    def generate_json(self, *, system, user, schema):
        out = json.loads(super().generate_json(system=system, user=user, schema=schema))
        if json.loads(user)["phase"] == "practice":
            out["verdict"] = self.verdict
        if "tip" in schema["properties"]:
            self.systems.append(system)
            out["tip"] = {"said": "I am student", "better": self.better,
                          "type": "grammar", "why": "Use 'a' before a job."}
        return json.dumps(out)


def run_turns(tutor, answers):
    tts = RecordingTTS()
    pipe = CascadePipeline(TypedSTT(), tutor, tts)
    pipe.greet()
    return [pipe.turn(a) for a in answers], tts


def test_tip_is_spoken_before_the_reply_and_survives_the_tutor_checks(cfg, tmp_path):
    model = TipModel()
    tutor = brain(cfg, VoiceGuard(model, tips=True), tmp_path)
    results, tts = run_turns(tutor, CORRECT + ["I am student and I like AI."])
    convo = results[-1]
    assert convo.tutor["phase"] == "conversation" and convo.tutor["source"] == "llm"
    assert convo.reply[0] == 'Tip: you can say, "I am a student."'
    assert convo.reply[1] == "Tell me more about that."          # the roleplay goes on
    assert convo.tutor["tip"]["why"] == "Use 'a' before a job."
    assert results[0].reply[0].startswith("Tip:")                 # practice, correct answer
    assert "Tips (the \"tip\" field)" in model.systems[0]
    assert "I am a student." not in tutor.session.history[-1]["text"]   # not in the role's line


def test_after_a_wrong_practice_answer_a_tip_never_gives_the_target_away(cfg, tmp_path):
    # the target is "My name is {name}": a tip saying it would skip the hint ladder
    tutor = brain(cfg, VoiceGuard(TipModel("form_error", "My name is Ravi."), tips=True), tmp_path)
    results, _ = run_turns(tutor, ["my name Ravi"])
    assert results[0].tutor["verdict"] == "form_error"
    assert results[0].tutor["tip_dropped"] == "gives_answer_away"
    assert not any(s.startswith("Tip:") for s in results[0].reply)
    # a tip on something else they said is still given
    tutor = brain(cfg, VoiceGuard(TipModel("form_error", "I have lived there since 2010."),
                                  tips=True), tmp_path)
    results, _ = run_turns(tutor, ["my name Ravi, I am live there since 2010"])
    assert results[0].reply[0] == 'Tip: you can say, "I have lived there since 2010."'


def test_no_tip_when_switched_off(cfg, tmp_path):
    model = TipModel()
    tutor = brain(cfg, VoiceGuard(model, tips=False), tmp_path)
    results, _ = run_turns(tutor, CORRECT[:1])
    assert not results[0].reply[0].startswith("Tip:") and model.systems == []


def test_a_listed_mistake_becomes_the_tip_when_the_model_leaves_tip_empty():
    out = {"say": [{"lang": "en", "text": "What do you do there?"}], "learner_language": "en",
           "wants_to_end": False,
           "errors": [{"said": "Dayadun", "fixed": "Dehradun", "type": "spelling", "importance": 3},
                      {"said": "I am live there", "fixed": "I have lived there",
                       "type": "grammar", "why": "past to now", "importance": 2}]}
    guard = VoiceGuard(Fixed("k", out), tips=True)
    guard.generate_json(system="", user="", schema={"properties": {"errors": {}, "say": {}}})
    assert guard.last_tip == {"said": "I am live there", "better": "I have lived there",
                              "type": "grammar", "why": "past to now"}   # not the spelling one
