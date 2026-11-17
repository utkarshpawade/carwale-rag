"""Integration tests: every non-chat endpoint against the real app and the test database."""
import uuid

import pytest

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- service / health / meta
async def test_root_health_meta_and_docs(client, catalog):
    _, stats = catalog
    r = await client.get("/")
    assert r.status_code == 200 and r.json() == {"service": "CarWale RAG Assistant API", "docs": "/docs", "health": "/api/health"}
    h = (await client.get("/api/health")).json()
    assert h == {"status": "ok", "models": stats["models"], "variants": stats["variants"], "chunks": stats["chunks"],
                 "data_snapshot": stats["snapshot"]}
    assert h["models"] > 200 and h["variants"] > 2000 and h["chunks"] > 20000
    meta = (await client.get("/api/meta")).json()
    assert len(meta["starters"]) == 6 and meta["stats"]["upcoming"] == stats["upcoming"] and meta["fallback_starters"]
    assert (await client.get("/openapi.json")).status_code == 200


async def test_unknown_route_and_wrong_method(client):
    assert (await client.get("/api/nope")).status_code == 404
    assert (await client.get("/api/chat")).status_code == 405
    assert (await client.put("/api/feedback", json={})).status_code == 405


# ---------------------------------------------------------------- conversations: list / get / delete
async def test_conversation_lifecycle(client, chat, user_id, sync_db):
    assert (await client.get("/api/conversations", params={"user_id": user_id})).json() == {"conversations": []}
    turn = await chat(user_id, "Creta vs Seltos")
    cid = turn.conversation_id
    assert turn.status == 200 and cid

    convs = (await client.get("/api/conversations", params={"user_id": user_id})).json()["conversations"]
    assert [c["id"] for c in convs] == [cid] and convs[0]["title"] == "Creta vs Seltos" and convs[0]["n_messages"] == 2

    full = (await client.get(f"/api/conversations/{cid}", params={"user_id": user_id})).json()
    assert full["conversation_id"] == cid and [m["role"] for m in full["messages"]] == ["user", "assistant"]
    a = full["messages"][1]
    assert a["content"] == turn.text and a["cars"] and a["sources"] and a["followups"] and a["trace"]["route"] == "compare"
    assert isinstance(a["id"], int) and a["created_at"]

    r = await client.delete(f"/api/conversations/{cid}", params={"user_id": user_id})
    assert r.status_code == 200 and r.json() == {"deleted": True}
    for table in ("conversations", "messages", "conversation_state"):
        col = "id" if table == "conversations" else "conversation_id"
        assert sync_db.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE {col} = %s", (cid,)).fetchone()["n"] == 0, table
    assert (await client.delete(f"/api/conversations/{cid}", params={"user_id": user_id})).status_code == 404
    assert (await client.get(f"/api/conversations/{cid}", params={"user_id": user_id})).status_code == 404


async def test_other_users_cannot_read_or_delete_a_conversation(client, chat, user_id):
    cid = (await chat(user_id, "Is the Tata Punch safe?")).conversation_id
    intruder = f"intruder-{uuid.uuid4().hex[:8]}"
    r = await client.get(f"/api/conversations/{cid}", params={"user_id": intruder})
    assert r.status_code == 403 and "Punch" not in r.text
    r = await client.delete(f"/api/conversations/{cid}", params={"user_id": intruder})
    assert r.status_code == 404  # no hint that the id exists
    assert (await client.get(f"/api/conversations/{cid}", params={"user_id": user_id})).status_code == 200  # still there
    assert (await client.get("/api/conversations", params={"user_id": intruder})).json() == {"conversations": []}


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "1234", "%27%20OR%201%3D1--"])
async def test_conversation_ids_are_validated(client, bad_id):
    assert (await client.get(f"/api/conversations/{bad_id}", params={"user_id": "someone-1"})).status_code == 400
    assert (await client.delete(f"/api/conversations/{bad_id}", params={"user_id": "someone-1"})).status_code == 400


async def test_user_id_is_required(client):
    cid = str(uuid.uuid4())
    assert (await client.get("/api/conversations")).status_code == 422
    assert (await client.get(f"/api/conversations/{cid}")).status_code == 422
    assert (await client.delete(f"/api/conversations/{cid}")).status_code == 422


async def test_list_is_newest_first_and_capped_at_30(client, user_id, sync_db):
    ids = [str(uuid.uuid4()) for _ in range(31)]
    with sync_db.transaction():
        for i, cid in enumerate(ids):
            sync_db.execute("INSERT INTO conversations (id, user_id, title, updated_at) VALUES (%s, %s, %s, now() - %s * interval '1 minute')",
                            (cid, user_id, f"chat {i}", i))
    convs = (await client.get("/api/conversations", params={"user_id": user_id})).json()["conversations"]
    assert len(convs) == 30 and [c["id"] for c in convs] == ids[:30] and convs[0]["n_messages"] == 0


async def test_list_conversations_with_hostile_user_id_is_harmless(client, sync_db):
    before = sync_db.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
    for uid in ("' OR '1'='1", "x' ; DROP TABLE conversations;--", "%", "_", "🚗" * 50, "a" * 5000):
        r = await client.get("/api/conversations", params={"user_id": uid})
        assert r.status_code == 200 and r.json() == {"conversations": []}
    assert sync_db.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"] == before


# ---------------------------------------------------------------- feedback
async def test_feedback_is_stored(client, sync_db):
    r = await client.post("/api/feedback", json={"message_id": 123456789, "rating": -1, "comment": "Wrong price 🚗"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    row = sync_db.execute("SELECT rating, comment FROM feedback WHERE message_id = 123456789 ORDER BY id DESC LIMIT 1").fetchone()
    assert row == {"rating": -1, "comment": "Wrong price 🚗"}


@pytest.mark.parametrize("payload", [
    {"message_id": 1, "rating": 5}, {"message_id": 1, "rating": "good"}, {"message_id": "x", "rating": 1}, {"rating": 1},
    {"message_id": 1, "rating": 1, "comment": "x" * 1001}, [], "not json",
])
async def test_feedback_validation(client, payload):
    r = await client.post("/api/feedback", json=payload)
    assert r.status_code == 422


@pytest.mark.parametrize("payload", [{"message_id": 2**63, "rating": 1}, {"message_id": 0, "rating": 1},
                                     {"message_id": 1, "rating": 1, "comment": "nul\x00byte"}])
async def test_feedback_never_returns_500(client, payload):
    # B-14: out-of-range ids and NUL bytes reached Postgres and failed with an unhandled 500
    r = await client.post("/api/feedback", json=payload)
    assert r.status_code in (200, 422), r.text


# ---------------------------------------------------------------- CORS
@pytest.mark.parametrize("origin,allowed", [
    ("http://localhost:5173", True), ("http://127.0.0.1:5173", True), ("https://carwale-rag.vercel.app", True),
    ("https://evil.example.com", False), ("http://carwale-rag.vercel.app", False),
])
async def test_cors_preflight(client, origin, allowed):
    r = await client.options("/api/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                   "Access-Control-Request-Headers": "content-type"})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed
    assert "access-control-allow-credentials" not in r.headers  # no cookies -> classic CSRF does not apply


async def test_cors_allows_any_vercel_app(client):
    # Security note S-03 (documented, not a bug fix): the regex trusts every *.vercel.app site, including ones anyone can deploy
    r = await client.options("/api/chat", headers={"Origin": "https://attacker-demo.vercel.app",
                                                   "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "https://attacker-demo.vercel.app"


# ---------------------------------------------------------------- database outage
async def test_endpoints_when_the_database_is_down(client, db_pool, monkeypatch):
    from psycopg_pool import AsyncConnectionPool
    broken = AsyncConnectionPool("postgresql://postgres:postgres@127.0.0.1:1/none", min_size=1, max_size=1, open=False,
                                 timeout=1.0, kwargs={"connect_timeout": 1})
    await broken.open(wait=False)
    monkeypatch.setattr(db_pool, "_pool", broken)
    try:
        r = await client.get("/api/conversations", params={"user_id": "someone-1"})
        assert r.status_code == 500
        assert "postgres" not in r.text.lower() and "Traceback" not in r.text  # no internals leaked
        assert (await client.get("/api/health")).status_code == 200  # health is cached: it does not notice the outage
    finally:
        await broken.close()
