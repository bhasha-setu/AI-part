"""Step 4: the LLMJudge interface, the deterministic stub, and the real client.

The real client is exercised with a fake SDK client only. No test calls the API.
"""

import json
from types import SimpleNamespace

import pytest

from guided_practice.course import load_course
from pedagogy_server.config import load_judge_config, load_reply_strings
from pedagogy_server.judge import (
    FALLBACK_BETA,
    AnthropicJudge,
    JudgeConfigError,
    JudgeUnavailable,
    ScriptedJudge,
    StubJudge,
    make_judge,
)
from pedagogy_server.models import JUDGE_RESPONSE_SCHEMA, ServerSession
from pedagogy_server.turn_judge import TurnJudge
from pedagogy_server.validation import validate_judge_output

COURSE = load_course("english_for_hindi_a1")
SCENARIO = COURSE.load_scenario("introducing_yourself")
CFG = load_judge_config()
STRINGS = load_reply_strings(COURSE.ui.code)
FAKE_KEY = "sk-ant-test-0000-not-a-real-key"


def request_for(transcript, item_id="intro_name", level=0):
    tj = TurnJudge(COURSE, judge=None)
    item = SCENARIO.item_by_id(item_id)
    from guided_practice.normalise import normalise
    norm = normalise(transcript, COURSE.lang)
    return tj.build_request(SCENARIO, item, level, transcript, norm), item, norm


# --- stub ---------------------------------------------------------------------

def test_stub_is_deterministic():
    req, _, _ = request_for("my name ravi")
    assert StubJudge(COURSE).judge(req) == StubJudge(COURSE).judge(req)


@pytest.mark.parametrize("transcript,verdict", [
    ("my name is Ravi", "correct"),
    ("my name Ravi", "surface_error"),
    ("", "off_topic"),
    ("the weather is nice", "meaning_error"),
])
def test_stub_output_is_valid_and_follows_the_rules(transcript, verdict):
    req, item, norm = request_for(transcript)
    raw = StubJudge(COURSE).judge(req)
    result = validate_judge_output(raw, item=item, level=0, norm=norm, ui=COURSE.ui,
                                   strings=STRINGS, cfg=CFG)
    assert result.ok, result.rejections
    assert result.response.verdict == verdict


def test_stub_withholds_the_recast_until_the_level_allows_it():
    req0, _, _ = request_for("my name Ravi", level=0)
    req2, _, _ = request_for("my name Ravi", level=2)
    assert json.loads(StubJudge(COURSE).judge(req0))["correction"] is None
    assert json.loads(StubJudge(COURSE).judge(req2))["correction"] == "Say: My name is Ravi."


def test_scripted_judge_replays_outputs_and_raises():
    judge = ScriptedJudge([{"verdict": "correct"}, JudgeUnavailable("timeout")])
    req, _, _ = request_for("x")
    assert json.loads(judge.judge(req)) == {"verdict": "correct"}
    with pytest.raises(JudgeUnavailable, match="timeout"):
        judge.judge(req)
    with pytest.raises(JudgeUnavailable, match="script_exhausted"):
        judge.judge(req)


def test_make_judge():
    assert make_judge("stub", COURSE).name == "stub"
    with pytest.raises(ValueError):
        make_judge("gpt", COURSE)


# --- real client, with a fake SDK client ----------------------------------------

class _Messages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.params = response, error, None

    def create(self, **params):
        self.params = params
        if self.error:
            raise self.error
        return self.response


def _client(text='{"ok": 1}', stop="end_turn", error=None):
    response = SimpleNamespace(stop_reason=stop,
                               content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(beta=SimpleNamespace(messages=_Messages(response, error)))


def test_real_judge_needs_the_env_key():
    with pytest.raises(JudgeConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicJudge(CFG, env={})


def test_real_judge_request_shape():
    client = _client()
    judge = AnthropicJudge(CFG, client=client)
    req, _, _ = request_for("my name ravi")
    assert judge.judge(req) == '{"ok": 1}'
    params = client.beta.messages.params
    assert params["model"] == CFG.model
    assert params["output_config"]["format"] == {"type": "json_schema", "schema": JUDGE_RESPONSE_SCHEMA}
    assert params["output_config"]["effort"] == CFG.effort
    assert params["betas"] == [FALLBACK_BETA] and params["fallbacks"] == "default"
    payload = json.loads(params["messages"][0]["content"])
    assert payload["item_id"] == "intro_name" and payload["hint_level"] == 0
    assert "max_retries_per_level" not in params["messages"][0]["content"]


def test_real_judge_maps_failures_to_unavailable():
    req, _, _ = request_for("x")
    with pytest.raises(JudgeUnavailable, match="refusal"):
        AnthropicJudge(CFG, client=_client(stop="refusal")).judge(req)
    with pytest.raises(JudgeUnavailable, match="max_tokens"):
        AnthropicJudge(CFG, client=_client(stop="max_tokens")).judge(req)
    with pytest.raises(JudgeUnavailable, match="api_error:TimeoutError"):
        AnthropicJudge(CFG, client=_client(error=TimeoutError("slow"))).judge(req)


def test_real_judge_does_not_hide_programming_errors():
    req, _, _ = request_for("x")
    with pytest.raises(TypeError):
        AnthropicJudge(CFG, client=_client(error=TypeError("unexpected keyword"))).judge(req)


def test_real_judge_never_holds_or_shows_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    judge = AnthropicJudge(CFG, client=_client())
    req, _, _ = request_for("x")
    judge.judge(req)
    assert FAKE_KEY not in repr(judge)
    assert FAKE_KEY not in json.dumps(judge.build_params(req))
    assert FAKE_KEY not in str(vars(judge))


def test_judging_a_turn_uses_the_fallback_when_the_judge_fails():
    state = ServerSession("s", "l", COURSE.course_id, SCENARIO.id)
    tj = TurnJudge(COURSE, ScriptedJudge([JudgeUnavailable("timeout")]))
    turn = tj.judge_turn(SCENARIO, SCENARIO.item_by_id("intro_name"), 0, "my name is Ravi", state)
    assert turn.source == "fallback"
    assert turn.rejections == ("judge_unavailable:timeout",)
    assert turn.evaluation.verdict == "correct"
