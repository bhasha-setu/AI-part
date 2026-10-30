"""Offline tests: no mic, no network, no Whisper model, no audio playback."""

import json
from pathlib import Path

import httpx
import pytest

from cascade import cli
from cascade.llm import (EchoLLM, GeminiChat, LLMUnavailable, OpenAICompatChat, ProviderChain,
                         build_chain, sentences_from_chunks, split_by_script)
from cascade.pipeline import CascadePipeline
from cascade.stt import TypedSTT, pick_language
from cascade.tts import ElevenLabsTTS, SilentTTS
from cascade.types import Segment


class RecordingTTS(SilentTTS):
    def __init__(self, splits=True):
        self.spoken = []
        self.splits_by_script = splits

    def speak(self, text, language, play=True):
        self.spoken.append((language, text))
        return 5.0


class FailingLLM:
    def __init__(self, name="failing", reason="api_error:ClientError:429", after=()):
        self.name, self.reason, self.after, self.calls = name, reason, list(after), 0

    def stream(self, history, text):
        self.calls += 1
        yield from self.after
        raise LLMUnavailable(self.reason)


class ScriptedLLM:
    def __init__(self, *replies, name="scripted"):
        self.name = name
        self.replies = list(replies)
        self.seen_history = []

    def stream(self, history, text):
        self.seen_history.append(list(history))
        yield from sentences_from_chunks([self.replies.pop(0)])


# --- text helpers ------------------------------------------------------------------------

def test_split_english_only():
    assert split_by_script("Hello! How are you?") == [Segment("en", "Hello! How are you?")]


def test_split_mixed_keeps_order_and_punctuation():
    segs = split_by_script('In English, "मेरा नाम रवि है" is "My name is Ravi".')
    assert [s.language for s in segs] == ["en", "hi", "en"]
    assert segs[1].text.startswith("मेरा नाम रवि है")


def test_split_danda_stays_with_hindi():
    assert split_by_script("मुझे चाय पसंद है। Nice!") == [Segment("hi", "मुझे चाय पसंद है।"),
                                                           Segment("en", "Nice!")]


def test_split_drops_empty():
    assert split_by_script("   ") == []


def test_sentences_from_chunks_rejoins_split_chunks():
    chunks = ["Hello Ra", "vi. Nice to", " meet you! मुझे चाय पसंद है। What", " about you?"]
    assert list(sentences_from_chunks(chunks)) == [
        "Hello Ravi.", "Nice to meet you!", "मुझे चाय पसंद है।", "What about you?"]


def test_sentences_keep_decimals_together():
    assert list(sentences_from_chunks(["It costs 2.5 rupees. Ok"])) == ["It costs 2.5 rupees.", "Ok"]


# --- pipeline ---------------------------------------------------------------------------

def test_turn_speaks_each_sentence_by_language():
    tts = RecordingTTS()
    pipe = CascadePipeline(TypedSTT(), ScriptedLLM("Nice! हाँ। Where are you from?"), tts)
    result = pipe.turn("my name is Ravi")
    assert result.reply == ["Nice!", "हाँ।", "Where are you from?"]
    assert [lang for lang, _ in tts.spoken] == ["en", "hi", "en"]
    assert result.timing.first_audio_ms is not None
    assert result.timing.first_audio_ms >= result.timing.stt_ms + result.timing.llm_first_ms
    assert result.tts_chars == sum(len(t) for _, t in tts.spoken)


def test_multilingual_tts_gets_whole_sentences():
    tts = RecordingTTS(splits=False)
    CascadePipeline(TypedSTT(), ScriptedLLM("You can say मुझे चाय पसंद है, I like tea."), tts).turn("x")
    assert tts.spoken == [("mixed", "You can say मुझे चाय पसंद है, I like tea.")]


def test_history_grows_and_is_capped():
    llm = ScriptedLLM(*["ok."] * 5)
    pipe = CascadePipeline(TypedSTT(), llm, SilentTTS(), max_history_turns=2)
    for i in range(5):
        pipe.turn(f"turn {i}")
    assert llm.seen_history[1] == [{"role": "user", "text": "turn 0"}, {"role": "model", "text": "ok."}]
    assert len(llm.seen_history[-1]) == 4


def test_llm_failure_speaks_fallback_and_skips_history():
    tts = RecordingTTS()
    pipe = CascadePipeline(TypedSTT(), FailingLLM(), tts, fallback_reply="Please say that again.")
    result = pipe.turn("hello")
    assert result.fallback_used and result.llm_error == "api_error:ClientError:429"
    assert tts.spoken == [("en", "Please say that again.")] and pipe.history == []


def test_failure_mid_reply_keeps_spoken_part():
    tts = RecordingTTS()
    pipe = CascadePipeline(TypedSTT(), FailingLLM(after=["Nice to meet you."]), tts)
    result = pipe.turn("hello")
    assert not result.fallback_used and result.llm_error
    assert pipe.history[-1] == {"role": "model", "text": "Nice to meet you."}


def test_silence_skips_llm():
    llm = ScriptedLLM()
    result = CascadePipeline(TypedSTT(), llm, SilentTTS(), no_speech_reply="I didn't hear you.").turn("")
    assert llm.seen_history == [] and result.reply_text == "I didn't hear you."


# --- provider chain -----------------------------------------------------------------------

def test_chain_falls_through_and_benches_failed_provider():
    now = [0.0]
    first = FailingLLM(name="gemini[key1]")
    chain = ProviderChain([first, ScriptedLLM("Hi.", "Again.", "Third.", name="nvidia")],
                          cooldown_seconds=60, clock=lambda: now[0])
    assert list(chain.stream([], "x")) == ["Hi."] and chain.last_used == "nvidia"
    list(chain.stream([], "x"))
    assert first.calls == 1            # benched: not retried within the cool-down
    now[0] = 61
    assert list(chain.stream([], "x")) == ["Third."]
    assert first.calls == 2            # cool-down over: tried first again


def test_chain_does_not_switch_mid_reply():
    second = ScriptedLLM("Other.", name="b")
    chain = ProviderChain([FailingLLM(name="a", after=["Started."]), second])
    with pytest.raises(LLMUnavailable):
        list(chain.stream([], "x"))
    assert second.seen_history == []


def test_build_chain_reads_numbered_keys():
    cfg = {"system_prompt": "s", "providers": [
        {"type": "gemini", "key_env": "G", "model": "m"},
        {"type": "openai_compat", "key_env": "N", "model": "x", "base_url": "http://n", "label": "nvidia"}]}
    chain = build_chain(cfg, env={"G": "a", "G_2": "b", "N": "c"})
    assert [b.name for b in chain.backends] == ["gemini:m[key1]", "gemini:m[key2]", "nvidia[key1]:x"]
    with pytest.raises(RuntimeError, match="no LLM key"):
        build_chain(cfg, env={})


# --- backends with fake transports -----------------------------------------------------------

def test_stt_language_restricted_to_allowed():
    assert pick_language({"ur": 0.6, "hi": 0.3, "en": 0.1}, ["en", "hi"]) == ("hi", 0.3)
    assert pick_language({"fr": 0.9}, ["en", "hi"]) == ("fr", 0.9)   # none allowed: best overall


def test_gemini_streams_sentences_and_maps_errors():
    class Chunk:
        def __init__(self, text):
            self.text, self.candidates = text, []

    class Client:
        class models:
            @staticmethod
            def generate_content_stream(**kw):
                assert [c["role"] for c in kw["contents"]] == ["user", "model", "user"]
                return iter([Chunk("Hello **there**. How"), Chunk(" are you?")])

    llm = GeminiChat("m", "sys", api_key="k", client=Client())
    out = list(llm.stream([{"role": "user", "text": "a"}, {"role": "model", "text": "b"}], "c"))
    assert out == ["Hello there.", "How are you?"]

    class Boom:
        class models:
            @staticmethod
            def generate_content_stream(**kw):
                raise ConnectionError("down")

    with pytest.raises(LLMUnavailable, match="ConnectionError"):
        list(GeminiChat("m", "sys", api_key="k", client=Boom()).stream([], "hi"))


def _sse(*pieces, finish=None):
    lines = [json.dumps({"choices": [{"delta": {"content": p}}]}) for p in pieces]
    if finish:
        lines.append(json.dumps({"choices": [{"delta": {}, "finish_reason": finish}]}))
    return "".join(f"data: {l}\n\n" for l in lines) + "data: [DONE]\n\n"


def test_openai_compat_streams_sse():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=_sse("Hi Ra", "vi. Where", " from?"))

    llm = OpenAICompatChat("x", "sys", api_key="k", base_url="http://n/v1",
                           extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                           transport=httpx.MockTransport(handler))
    assert list(llm.stream([{"role": "model", "text": "b"}], "c")) == ["Hi Ravi.", "Where from?"]
    assert seen["auth"] == "Bearer k"
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "assistant", "user"]
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_compat_429_and_truncation_are_unavailable():
    rate_limited = OpenAICompatChat("x", "s", api_key="k", base_url="http://n",
                                    transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(LLMUnavailable, match="http_429"):
        list(rate_limited.stream([], "c"))
    cut = OpenAICompatChat("x", "s", api_key="k", base_url="http://n", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text=_sse("Half a sent", finish="length"))))
    with pytest.raises(LLMUnavailable, match="max_tokens"):
        list(cut.stream([], "c"))


def test_elevenlabs_request_without_playback():
    seen = {}

    def handler(request):
        seen.update(path=request.url.path, fmt=request.url.params["output_format"],
                    key=request.headers["xi-api-key"], body=json.loads(request.content))
        return httpx.Response(200, content=b"\x00\x01" * 100)

    tts = ElevenLabsTTS("k", "voice1", transport=httpx.MockTransport(handler))
    assert tts.speak("नमस्ते, hello", "mixed", play=False) is not None
    assert seen == {"path": "/v1/text-to-speech/voice1/stream", "fmt": "pcm_16000", "key": "k",
                    "body": {"text": "नमस्ते, hello", "model_id": "eleven_flash_v2_5"}}


# --- CLI end to end (typed input, echo LLM, no audio) -------------------------------------------

def test_cli_text_offline_writes_log(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    cfg = Path(__file__).resolve().parent.parent / "config.yaml"
    answers = iter(["my name is Ravi", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.main(["--chat", "--text", "--offline", "--no-tts", "--config", str(cfg)]) == 0
    [log] = (tmp_path / "runs").glob("*.jsonl")
    record = json.loads(log.read_text().splitlines()[0])
    assert record["reply_text"] == "You said: my name is Ravi"
    assert record["llm"] == "echo" and "first_audio_ms" in record["timing"]


def test_vad_returns_nothing_for_silence_and_noise():
    import numpy as np

    from cascade.stt import trim_to_speech

    noise = (np.random.default_rng(0).standard_normal(32000) * 0.01).astype(np.float32)
    assert len(trim_to_speech(np.zeros(32000, np.float32))) == 0
    assert len(trim_to_speech(noise)) == 0


def test_elevenlabs_voice_id_from_env_with_config_fallback():
    from cascade.tts import load_tts

    cfg = {"engine": "elevenlabs", "elevenlabs": {"voice_id_env": "VOICE", "voice_id": "fallback"}}
    cfg = dict(cfg, voices={"en": "Rishi", "hi": "Lekha"}, rate=175)
    assert load_tts(cfg, {"ELEVENLABS_API_KEY": "k", "VOICE": " chosen "}).primary.voice_id == "chosen"
    assert load_tts(cfg, {"ELEVENLABS_API_KEY": "k"}).primary.voice_id == "fallback"


def test_tts_falls_back_once_and_splits_for_backup():
    from cascade.tts import FallbackTTS

    class Broken:
        name, splits_by_script, calls = "cloud", False, 0

        def speak(self, text, language, play=True):
            Broken.calls += 1
            raise RuntimeError("ElevenLabs HTTP 402: paid_plan_required")

    backup, warnings = RecordingTTS(), []
    tts = FallbackTTS(Broken(), backup, warn=warnings.append)
    pipe = CascadePipeline(TypedSTT(), ScriptedLLM("Nice! मुझे चाय पसंद है। Bye."), tts)
    result = pipe.turn("hi")
    assert backup.spoken == [("en", "Nice!"), ("hi", "मुझे चाय पसंद है।"), ("en", "Bye.")]
    assert Broken.calls == 1 and len(warnings) == 1 and "402" in warnings[0]
    assert result.timing.first_audio_ms is not None and "fallback" in tts.name


def test_english_is_kept_unless_hindi_clearly_wins():
    kw = {"prefer": "en", "switch_share": 0.7, "aliases": {"ur": "hi"}}
    # accented English: unsure, Hindi slightly ahead -> English
    assert pick_language({"en": 0.20, "hi": 0.26, "ur": 0.05}, ["en", "hi"], **kw)[0] == "en"
    # spoken Hindi heard as Urdu: Hindi clearly wins once Urdu counts as Hindi
    assert pick_language({"ur": 0.6, "hi": 0.3, "en": 0.1}, ["en", "hi"], **kw) == ("hi", pytest.approx(0.9))
    assert pick_language({"en": 0.9, "hi": 0.05}, ["en", "hi"], **kw)[0] == "en"
