"""Unit tests: API-layer helpers - rate limiter, SSE framing, client IP, request validation."""
import datetime as dt
import json
import uuid

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from app.api import chat as chat_api
from app.api.chat import ChatIn, RateLimiter, client_ip, sse
from app.api.routes import FeedbackIn, _uuid


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(chat_api.time, "time", c)
    return c


# ---------------------------------------------------------------- rate limiter
def test_rate_limiter_per_minute(clock):
    rl = RateLimiter()
    assert rl.allow("ip1", 2, 100) == (True, 0) and rl.allow("ip1", 2, 100) == (True, 0)
    ok, wait = rl.allow("ip1", 2, 100)
    assert not ok and 1 <= wait <= 61
    assert rl.allow("ip2", 2, 100)[0]  # other clients are unaffected
    clock.t += 61
    assert rl.allow("ip1", 2, 100)[0]  # window slid


def test_rate_limiter_per_day(clock):
    rl = RateLimiter()
    for _ in range(3):
        assert rl.allow("ip", 100, 3)[0]
        clock.t += 61
    ok, wait = rl.allow("ip", 100, 3)
    assert not ok and 86_000 < wait <= 86_401
    clock.t += 86_401
    assert rl.allow("ip", 100, 3)[0]


def test_rejected_requests_do_not_consume_quota(clock):
    rl = RateLimiter()
    rl.allow("ip", 1, 100)
    for _ in range(50):
        assert not rl.allow("ip", 1, 100)[0]
    assert len(rl.hits["ip"]) == 1


def test_rate_limiter_forgets_idle_clients(clock):
    # B-13: one dict entry per client IP was kept forever; with a spoofable X-Forwarded-For an attacker can grow the
    # dict without bound (memory exhaustion on the 512 MB instance). Entries idle for a day must be dropped.
    rl = RateLimiter()
    for i in range(5_000):
        rl.allow(f"10.0.{i // 256}.{i % 256}", 12, 300)
    clock.t += 2 * 86_400
    rl.allow("fresh-client", 12, 300)
    assert len(rl.hits) < 100


def test_rate_limiter_caps_tracked_clients(clock):
    # B-13: idle sweeping alone cannot stop a flood of fresh (spoofed) client keys inside one day; cap the table
    rl = RateLimiter(max_keys=100)
    for i in range(1_000):
        rl.allow(f"spoofed-{i}", 12, 300)
    assert len(rl.hits) <= 100
    assert rl.allow("spoofed-999", 1, 300)[0] is False  # the most recent clients are still tracked and limited


# ---------------------------------------------------------------- SSE framing
def test_sse_frames_one_event_per_message():
    out = sse("token", {"text": "a\nb", "price": "₹5 L"})
    assert out == 'event: token\ndata: {"text": "a\\nb", "price": "₹5 L"}\n\n'
    assert out.count("\n\n") == 1  # newlines inside data are escaped, so framing can't break


def test_sse_serialises_non_json_types():
    u = uuid.UUID("12345678-1234-5678-1234-567812345678")
    payload = json.loads(sse("done", {"id": u, "at": dt.datetime(2026, 11, 17, 10, 0)}).split("data: ", 1)[1])
    assert payload == {"id": str(u), "at": "2026-11-17 10:00:00"}


# ---------------------------------------------------------------- client IP
def _req(headers=None, client=("9.9.9.9", 1234)):
    scope = {"type": "http", "method": "POST", "path": "/api/chat", "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]}
    if client:
        scope["client"] = client
    return Request(scope)


def test_client_ip_uses_socket_address_without_proxy_header():
    assert client_ip(_req()) == "9.9.9.9"
    assert client_ip(_req(client=None)) == "unknown"


def test_client_ip_first_forwarded_entry():
    assert client_ip(_req({"X-Forwarded-For": "1.2.3.4, 10.0.0.1"})) == "1.2.3.4"


# ---------------------------------------------------------------- request models
@pytest.mark.parametrize("user_id,valid", [
    ("abcdef", True), ("a" * 64, True), ("uuid-with-dashes_ok", True), (str(uuid.uuid4()), True),
    ("abcde", False), ("a" * 65, False), ("bad id!", False), ("ユーザー123456", False), ("", False), ("abc\x00def", False),
])
def test_chat_user_id_validation(user_id, valid):
    if valid:
        assert ChatIn(user_id=user_id, message="hi").user_id == user_id
    else:
        with pytest.raises(ValidationError):
            ChatIn(user_id=user_id, message="hi")


@pytest.mark.parametrize("message,valid", [("x", True), ("x" * 4000, True), ("", False), ("x" * 4001, False), (123, False),
                                           (None, False)])
def test_chat_message_validation(message, valid):
    if valid:
        ChatIn(user_id="abcdef", message=message)
    else:
        with pytest.raises(ValidationError):
            ChatIn(user_id="abcdef", message=message)


@pytest.mark.parametrize("message", ["   ", "\n\t \n", "\x00", " \x00 \x00 "])
def test_chat_rejects_blank_messages(message):
    # B-02: whitespace-only (or NUL-only) messages passed validation, then ran the whole pipeline on an empty string
    with pytest.raises(ValidationError):
        ChatIn(user_id="abcdef", message=message)


def test_chat_strips_nul_bytes():
    # B-02: PostgreSQL TEXT cannot store NUL; a stray \x00 made the turn fail with a generic error
    assert ChatIn(user_id="abcdef", message="creta\x00 price").message == "creta price"


@pytest.mark.parametrize("payload,valid", [
    ({"message_id": 1, "rating": 1}, True), ({"message_id": 1, "rating": -1}, True), ({"message_id": 1, "rating": 0}, True),
    ({"message_id": 1, "rating": "1"}, True), ({"message_id": 1, "rating": 2}, False), ({"message_id": 1, "rating": -2}, False),
    ({"message_id": "abc", "rating": 1}, False), ({"rating": 1}, False), ({"message_id": 1}, False),
    ({"message_id": 1, "rating": 1, "comment": "x" * 1000}, True), ({"message_id": 1, "rating": 1, "comment": "x" * 1001}, False),
])
def test_feedback_validation(payload, valid):
    if valid:
        FeedbackIn(**payload)
    else:
        with pytest.raises(ValidationError):
            FeedbackIn(**payload)


@pytest.mark.parametrize("message_id", [0, -5, 2**63, 10**30])
def test_feedback_message_id_must_be_a_real_bigint_id(message_id):
    # B-14: ids outside BIGSERIAL's range reached Postgres and came back as an HTTP 500
    with pytest.raises(ValidationError):
        FeedbackIn(message_id=message_id, rating=1)


def test_feedback_comment_nul_bytes_are_stripped():
    assert FeedbackIn(message_id=1, rating=1, comment="ok\x00").comment == "ok"


def test_uuid_helper():
    u = "12345678-1234-5678-1234-567812345678"
    assert _uuid(u.upper()) == u and _uuid("{" + u + "}") == u and _uuid(u.replace("-", "")) == u
    for bad in ("not-a-uuid", "", "1234"):
        with pytest.raises(HTTPException) as e:
            _uuid(bad)
        assert e.value.status_code == 400
