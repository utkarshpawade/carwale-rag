"""Integration fixtures: real Postgres (the *_test clone), the real FastAPI app in-process, the real groq SDK
talking to the fake Groq server (tests/fakes/fake_groq.py), and the real fastembed embedder."""
from __future__ import annotations

import asyncio
import json
import os
import uuid

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row

TEST_DB = os.environ["DATABASE_URL"]  # set by tests/conftest.py (TEST_DATABASE_URL or the local *_test default)
DB_NAME = TEST_DB.rsplit("/", 1)[-1].split("?")[0]
if not DB_NAME.endswith("_test"):
    raise RuntimeError(f"refusing to run integration tests against '{DB_NAME}': the database name must end in _test")


def _db_ok() -> str | None:
    try:
        with psycopg.connect(TEST_DB, connect_timeout=3) as c:
            n = c.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            return None if n else "test database has no catalog rows"
    except Exception as e:  # noqa: BLE001
        return f"test database unavailable: {type(e).__name__}"


_SKIP = _db_ok()


def pytest_collection_modifyitems(config, items):
    if _SKIP:
        for it in items:
            if "integration" in str(it.fspath):
                it.add_marker(pytest.mark.skip(reason=_SKIP))


@pytest.fixture(scope="session")
def catalog():
    with psycopg.connect(TEST_DB) as c:
        c.row_factory = dict_row
        models = c.execute("SELECT id, brand, brand_slug, name, slug, full_name, status, user_rating_count, is_ev FROM models").fetchall()
        stats = c.execute("""
            SELECT (SELECT COUNT(*) FROM models WHERE status = 'launched') AS models,
                   (SELECT COUNT(*) FROM models WHERE status = 'upcoming') AS upcoming,
                   (SELECT COUNT(*) FROM variants v JOIN models m ON m.id = v.model_id WHERE m.status = 'launched') AS variants,
                   (SELECT COUNT(*) FROM chunks) AS chunks,
                   (SELECT value FROM meta WHERE key = 'data_snapshot_date') AS snapshot""").fetchone()
    from app.rag.entities import CarResolver
    return CarResolver(models), dict(stats)


@pytest.fixture(scope="session")
def embedder():
    from app.config import get_settings
    from app.embeddings import Embedder
    return Embedder(get_settings().embed_model)


@pytest.fixture(scope="session")
async def db_pool(anyio_backend):
    from app import db
    await db.init_pool(TEST_DB)
    yield db
    await db.close_pool()


@pytest.fixture
def sync_db():
    """Direct SQL for assertions and setup (autocommit)."""
    with psycopg.connect(TEST_DB, autocommit=True) as c:
        c.row_factory = dict_row
        yield c


@pytest.fixture(autouse=True)
def _clean_shared_state(request):
    """Answer cache and rate-limiter buckets are global: reset them so tests cannot influence each other."""
    if "integration" not in str(request.fspath) or _SKIP:
        yield
        return
    with psycopg.connect(TEST_DB, autocommit=True) as c:
        c.execute("TRUNCATE answer_cache")
    from app.api import chat as chat_api
    chat_api.limiter.hits.clear()
    yield
    chat_api.limiter.hits.clear()


@pytest.fixture
def settings():
    from app.config import Settings
    return Settings(groq_api_key="test-key-not-real", database_url=TEST_DB, rate_limit_per_min=100_000,
                    rate_limit_per_day=1_000_000)


@pytest.fixture
def fake_llm():
    from tests.fakes.fake_groq import FakeGroq
    return FakeGroq()


@pytest.fixture
def make_pipeline(settings, fake_llm, catalog, embedder, db_pool):
    from app.llm.client import LLMClient
    from app.rag.orchestrator import Pipeline

    resolver, stats = catalog

    def build(s=None, fake=None):
        s = s or settings
        llm = LLMClient(s)
        llm.client = (fake or fake_llm).client()
        return Pipeline(s, llm, resolver, embedder, dict(stats))
    return build


@pytest.fixture
def pipeline(make_pipeline):
    return make_pipeline()


@pytest.fixture
async def client(pipeline):
    from app.main import app
    app.state.pipeline, app.state.stats = pipeline, pipeline.stats
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False, client=("127.0.0.1", 50000))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=60) as c:
        yield c
    if pipeline._bg:  # background summaries must not outlive the test
        await asyncio.gather(*list(pipeline._bg), return_exceptions=True)


@pytest.fixture
def user_id():
    return f"it-{uuid.uuid4().hex[:16]}"


def parse_sse(text: str) -> list[tuple[str, object]]:
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        ev, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((ev, json.loads(data) if data else None))
    return events


class Turn:
    def __init__(self, status: int, events: list[tuple[str, object]], raw: str):
        self.status, self.events, self.raw = status, events, raw

    def of(self, name: str) -> list:
        return [d for e, d in self.events if e == name]

    def one(self, name: str):
        got = self.of(name)
        assert len(got) >= 1, f"no '{name}' event in {[e for e, _ in self.events]}"
        return got[-1]

    @property
    def names(self) -> list[str]:
        return [e for e, _ in self.events]

    @property
    def text(self) -> str:
        return "".join(d["text"] for d in self.of("token"))

    @property
    def conversation_id(self) -> str | None:
        c = self.of("conversation")
        return c[0]["conversation_id"] if c else None


@pytest.fixture
def chat(client):
    async def send(user_id: str, message: str, conversation_id: str | None = None, headers: dict | None = None) -> Turn:
        body = {"user_id": user_id, "message": message}
        if conversation_id:
            body["conversation_id"] = conversation_id
        r = await client.post("/api/chat", json=body, headers=headers or {})
        events = parse_sse(r.text) if r.headers.get("content-type", "").startswith("text/event-stream") else []
        return Turn(r.status_code, events, r.text)
    return send
