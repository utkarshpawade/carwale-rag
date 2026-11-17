"""Contract tests: LLMClient against the fake Groq API through the real groq SDK (no DB, no network)."""
import pytest

from app.config import Settings
from app.llm.client import CallInfo, LLMClient, LLMUnavailable
from app.rag.schemas import PLAN_EXAMPLE, PLAN_SCHEMA, QueryPlan
from tests.fakes.fake_groq import FakeGroq

pytestmark = pytest.mark.anyio

PLANNER_MSGS = [{"role": "system", "content": "planner"},
                {"role": "user", "content": "CONTEXT\n(new conversation)\n\nLATEST USER MESSAGE: Creta vs Seltos"}]
ANSWER_MSGS = [{"role": "system", "content": "You are CarWale Assistant, a friendly expert car advisor"},
               {"role": "user", "content": "## CAR DATA\n#### Hyundai Creta (u)\n- Price (ex-showroom): ₹11 L\n\n## QUESTION\nPrice?"}]


@pytest.fixture
def fake():
    return FakeGroq()


@pytest.fixture
def llm(fake):
    c = LLMClient(Settings(groq_api_key="test-key-not-real"))
    c.client = fake.client()
    return c


async def plan(llm):
    return await llm.json("planner", PLANNER_MSGS, QueryPlan, PLAN_SCHEMA, example=PLAN_EXAMPLE)


async def collect(llm, info=None):
    info = info or CallInfo("answer")
    return "".join([d async for d in llm.stream("answer", ANSWER_MSGS, 300, info)]), info


async def test_json_success_records_model_usage_and_budget(llm):
    p, info = await plan(llm)
    assert p.intent == "compare" and [c.mention for c in p.cars] == ["Hyundai Creta", "Kia Seltos"]
    assert info.model == "qwen/qwen3.8-27b" and info.prompt_tokens > 0 and info.completion_tokens > 0 and info.attempts == []
    assert llm.budget["qwen/qwen3.8-27b"][0] == 50_000


async def test_json_uses_json_object_mode_with_the_example(llm, fake):
    await plan(llm)
    assert fake.calls[0]["kind"] == "planner" and fake.calls[0]["max_tokens"] == 700


async def test_json_falls_back_on_429_and_cools_the_model(llm, fake):
    fake.add_rule("status", model="qwen/qwen3.8-27b", status=429, retry_after="7")
    p, info = await plan(llm)
    assert info.model == "openai/gpt-oss-20b" and info.attempts == ["qwen/qwen3.8-27b:rate_limited"]
    _, info2 = await plan(llm)
    assert info2.skipped == ["qwen/qwen3.8-27b:cooling"] and info2.model == "openai/gpt-oss-20b"


async def test_json_raises_with_retry_hint_when_every_model_is_limited(llm, fake):
    fake.add_rule("status", status=429, retry_after="9")
    with pytest.raises(LLMUnavailable) as e:
        await plan(llm)
    assert e.value.retry_after == 9.0 and not e.value.auth


async def test_json_auth_error_is_flagged(llm, fake):
    fake.add_rule("status", status=401)
    with pytest.raises(LLMUnavailable) as e:
        await plan(llm)
    assert e.value.auth and len(fake.calls) == 1


async def test_missing_api_key_is_an_auth_error(fake):
    c = LLMClient(Settings(groq_api_key=""))  # the client sends "missing"
    c.client = fake.client(api_key="missing")
    with pytest.raises(LLMUnavailable) as e:
        await plan(c)
    assert e.value.auth


async def test_json_invalid_output_gets_one_repair_try_per_model(llm, fake):
    fake.add_rule("invalid_json", model="qwen/qwen3.8-27b")
    p, info = await plan(llm)
    assert info.model == "openai/gpt-oss-20b" and info.attempts == ["qwen/qwen3.8-27b:invalid_json"] * 2


async def test_json_skips_a_model_without_token_budget(llm, fake):
    fake.configure({"remaining_tokens": 10})  # every response now reports almost no budget left
    await plan(llm)
    _, info = await plan(llm)
    assert "qwen/qwen3.8-27b:no_token_budget" in info.skipped


async def test_json_timeout_and_5xx_move_to_the_next_model(llm, fake):
    fake.add_rule("timeout", model="qwen/qwen3.8-27b")
    fake.add_rule("status", model="openai/gpt-oss-20b", status=503)
    p, info = await plan(llm)
    assert info.model == "openai/gpt-oss-120b"
    assert info.attempts == ["qwen/qwen3.8-27b:timeout", "openai/gpt-oss-20b:http_503:server_error"]


async def test_stream_yields_text_and_usage(llm):
    text, info = await collect(llm)
    assert text.startswith("Here is what CarWale's data says about **Hyundai Creta**")
    assert info.model == "openai/gpt-oss-120b" and info.completion_tokens > 0 and info.latency_ms >= 0


async def test_stream_falls_back_only_before_the_first_token(llm, fake):
    fake.add_rule("status", model="openai/gpt-oss-120b", status=500)
    text, info = await collect(llm)
    assert text and info.model == "openai/gpt-oss-20b" and info.attempts == ["openai/gpt-oss-120b:http_500:server_error"]


async def test_stream_forced_fail(fake):
    c = LLMClient(Settings(groq_api_key="test-key-not-real", force_fail_models="openai/gpt-oss-120b"))
    c.client = fake.client()
    text, info = await collect(c)
    assert info.attempts == ["openai/gpt-oss-120b:forced_fail"] and info.model == "openai/gpt-oss-20b"


@pytest.mark.parametrize("action", ["cut", "stream_error"])
async def test_stream_failure_after_tokens_raises_llm_unavailable(llm, fake, action):
    # B-09: the raw httpx.ReadError / groq.APIError escaped, bypassing the "answer cut short" handling
    fake.add_rule(action, kind="answer", after_chunks=2)
    got = []
    with pytest.raises(LLMUnavailable):
        async for d in llm.stream("answer", ANSWER_MSGS, 300, CallInfo("answer")):
            got.append(d)
    assert got  # some text had already been delivered


async def test_text_helper(llm):
    text, info = await llm.text("summary", [{"role": "system", "content": "Summarize this car-shopping conversation"},
                                            {"role": "user", "content": "x"}])
    assert text.startswith("Summary:") and info.role == "summary"
