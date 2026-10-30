"""Step 1: typed data models and the server config they read."""

import dataclasses
import json

from guided_practice.config import ConfigError
from pedagogy_server.config import LEVELS, VERDICT_KEYS, load_judge_config, load_reply_strings
from pedagogy_server.models import (
    JUDGE_RESPONSE_SCHEMA,
    VERDICTS,
    Decision,
    Frame,
    JudgeRequest,
    JudgeResponse,
    KnownError,
    LogRecord,
    Reply,
    ServerSession,
    Turn,
)


def _request(**overrides):
    base = dict(
        target_language="en", learner_language="hi", cefr_level="A1", reply_language="en",
        scenario_id="introducing_yourself", scenario_title="Introducing yourself",
        item_id="intro_name", prompt="Someone asks your name. Say it.",
        target_form="My name is {name}.", accepted_forms=("my name is {name}",),
        known_errors=(KnownError("missing_copula", "my name ravi", "my name is ravi"),),
        hint_level=0, shown_to_learner=("Someone asks your name. Say it.",),
        transcript_raw="My name Ravi", transcript_normalised="my name ravi",
        rubric="r", constraints=("c",),
    )
    base.update(overrides)
    return JudgeRequest(**base)


def test_judge_request_carries_no_policy_internals():
    names = {f.name for f in dataclasses.fields(JudgeRequest)}
    forbidden = {"max_retries", "retries", "max_stage_attempts", "attempt", "box", "due",
                 "policy", "stage", "next_item", "corrections_used", "end_session"}
    assert names & forbidden == set()


def test_judge_request_payload_is_json():
    payload = _request().to_payload()
    text = json.dumps(payload, ensure_ascii=False)
    assert json.loads(text)["known_errors"][0]["id"] == "missing_copula"
    assert payload["hint_level"] == 0


def test_response_schema_matches_the_dataclass():
    fields = {f.name for f in dataclasses.fields(JudgeResponse)}
    assert set(JUDGE_RESPONSE_SCHEMA["required"]) == fields
    assert JUDGE_RESPONSE_SCHEMA["additionalProperties"] is False
    assert JUDGE_RESPONSE_SCHEMA["properties"]["verdict"]["enum"] == list(VERDICTS)


def test_server_session_counts_corrections_per_error():
    s = ServerSession("s1", "l1", "c", "sc")
    s.record_correction("intro_name", "missing_copula", "fix")
    s.record_correction("intro_work", None, "fix2")
    assert s.corrections_used == 2
    assert s.corrected_error_ids["missing_copula"] == 1
    assert s.corrections_shown["intro_name"] == ["fix"]


def test_records_serialise():
    frame = Frame(stage="guided_practice", expects="answer", item_id="intro_name", hint_level=0)
    reply = Reply(text="Correct!", stage="guided_practice", expects="answer")
    turn = Turn(index=1, frame=frame, transcript_raw="x", reply=reply)
    record = LogRecord(timestamp="t", session_id="s", turn_index=1, kind="turn",
                       stage="guided_practice", reason="r", data={"turn": dataclasses.asdict(turn)})
    assert json.loads(json.dumps(dataclasses.asdict(record)))["data"]["turn"]["index"] == 1
    assert Decision("advance", "because").reason == "because"


def test_judge_config_loads_and_is_draft():
    cfg = load_judge_config()
    assert cfg.status == "DRAFT"
    assert 2 <= cfg.corrections_per_session <= 3
    assert cfg.corrections_per_error_id == 1
    assert cfg.reply_max_sentences == 2


def test_judge_config_rejects_an_unsourced_threshold(tmp_path):
    path = tmp_path / "judge.yaml"
    path.write_text("status: DRAFT\nmin_confidence: 0.6\n", encoding="utf-8")
    try:
        load_judge_config(path)
    except ConfigError as err:
        assert "bare number" in str(err)
    else:
        raise AssertionError("expected ConfigError")


def test_reply_strings_have_a_line_per_verdict_and_level(course):
    strings = load_reply_strings(course.ui.code)
    for verdict in VERDICT_KEYS:
        for level in range(LEVELS):
            assert strings.guided(verdict, level)
    for stage in ("intro", "model_dialogue", "useful_language", "roleplay", "assessment",
                  "complete", "held"):
        assert strings.stage_lines[stage]
