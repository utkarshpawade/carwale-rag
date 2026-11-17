"""Unit tests: Groq client bookkeeping (no network) - header parsing, token budget, cooldowns, candidate selection."""
import time
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm.client import CallInfo, LLMClient, LLMUnavailable, _duration, _est_tokens, _model_kwargs, _status_reason


def make_client(**kw) -> LLMClient:
    s = Settings(groq_api_key="test-key-not-real", planner_models="a,b,c", answer_models="x", summary_models="x", **kw)
    return LLMClient(s)


@pytest.mark.parametrize("raw,expected", [
    ("7.66s", 7.66), ("2m59.56s", 179.56), ("120ms", 0.12), ("1h2m", 3720.0), ("500ms", 0.5), ("0s", 0.0),
    ("", None), (None, None), ("soon", None), ("1.5", None),
])
def test_duration(raw, expected):
    got = _duration(raw)
    assert got == pytest.approx(expected) if expected is not None else got is None


def test_est_tokens():
    assert _est_tokens([{"content": "a" * 400}]) == 140
    assert _est_tokens([{"content": None}, {}], "x" * 40) == 10 + 80
    assert _est_tokens([]) == 0


def test_status_reason():
    e = SimpleNamespace(status_code=500, body={"error": {"code": "server_error"}})
    assert _status_reason("m", e) == "m:http_500:server_error"
    assert _status_reason("m", SimpleNamespace(status_code=503, body={"code": "over_capacity"})) == "m:http_503:over_capacity"
    assert _status_reason("m", SimpleNamespace(status_code=400, body="text")) == "m:http_400"
    assert _status_reason("m", Exception()) == "m:http_?"


def test_model_kwargs():
    assert _model_kwargs("openai/gpt-oss-20b") == {"reasoning_effort": "low", "include_reasoning": False}
    assert _model_kwargs("qwen/qwen3.8-27b") == {"reasoning_effort": "none"}
    assert _model_kwargs("llama-3.1-8b-instant") == {}


def test_record_headers_and_fits():
    c = make_client()
    c._record_headers("a", {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "10s"})
    remaining, reset_at = c.budget["a"]
    assert remaining == 100 and reset_at == pytest.approx(time.time() + 10, abs=1)
    ok, wait = c._fits("a", 200)
    assert not ok and 8 < wait <= 10
    assert c._fits("a", 50) == (True, 0.0)
    assert c._fits("unknown", 10**8) == (True, 0.0)
    c.budget["a"] = (0, time.time() - 1)  # reset already passed
    assert c._fits("a", 10**6)[0]


@pytest.mark.parametrize("headers", [{}, {"x-ratelimit-remaining-tokens": "lots"}, {"x-ratelimit-remaining-tokens": None}])
def test_record_headers_ignores_missing_or_garbage(headers):
    c = make_client()
    c._record_headers("a", headers)
    assert "a" not in c.budget


def test_record_headers_defaults_reset_to_a_minute():
    c = make_client()
    c._record_headers("a", {"x-ratelimit-remaining-tokens": "5"})
    assert c.budget["a"][1] == pytest.approx(time.time() + 60, abs=1)


def test_cooldown_is_bounded():
    c = make_client()
    c._cool("a", 0.01)
    assert c.cooldown_until["a"] == pytest.approx(time.time() + 1, abs=0.2)
    c._cool("a", 10_000)
    assert c.cooldown_until["a"] == pytest.approx(time.time() + 120, abs=0.5)


def test_candidates_skip_cooling_and_over_budget_models():
    c = make_client()
    c._cool("a", 30)
    c.budget["b"] = (10, time.time() + 5)
    info = CallInfo("planner")
    models, wait = c._candidates("planner", 1000, info)
    assert models == ["c"] and info.skipped == ["a:cooling", "b:no_token_budget"] and 3 < wait <= 5
    models, wait = make_client()._candidates("planner", 10, CallInfo("planner"))
    assert models == ["a", "b", "c"] and wait == 10.0


def test_on_429_reads_retry_after_then_reset_header_then_defaults():
    c = make_client()
    e = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "3"}))
    assert c._on_429("a", e) == 3.0 and c.cooldown_until["a"] > time.time() + 2
    e = SimpleNamespace(response=SimpleNamespace(headers={"x-ratelimit-reset-tokens": "1m"}))
    assert c._on_429("b", e) == 60.0
    assert c._on_429("c", SimpleNamespace(response=None)) == 10.0


def test_forced_fail_parsing_and_call_info():
    c = make_client(force_fail_models=" a , ,b ")
    assert c.forced_fail == {"a", "b"}
    info = CallInfo("answer", model="x", prompt_tokens=3, completion_tokens=4, latency_ms=5, attempts=["y:timeout"])
    assert info.as_dict() == {"role": "answer", "model": "x", "prompt_tokens": 3, "completion_tokens": 4,
                              "latency_ms": 5, "fallbacks": ["y:timeout"], "skipped": []}


def test_settings_chain_parsing():
    s = Settings(groq_api_key="k", planner_models=" a, ,b ,", cors_origins="http://x, ,http://y")
    assert s.chain("planner") == ["a", "b"] and s.cors_list == ["http://x", "http://y"]
    with pytest.raises(KeyError):
        s.chain("nope")


def test_llm_unavailable_carries_retry_and_auth():
    e = LLMUnavailable("x", 4.0, auth=True)
    assert e.retry_after == 4.0 and e.auth and str(e) == "x"
