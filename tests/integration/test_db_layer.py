"""Integration tests: memory store, structured SQL search and hybrid retrieval against the real catalog (test DB)."""
import datetime as dt
import uuid

import pytest

from app.memory import store
from app.memory.store import ConvState
from app.rag import car_search, specs
from app.rag.hybrid_search import brand_chunks, hybrid_search, pinned_chunks, titled_chunks, trim_chunks
from app.rag.schemas import Filters

pytestmark = pytest.mark.anyio


@pytest.fixture
def conv():
    return str(uuid.uuid4()), f"db-{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------- memory store
async def test_conversation_rows(db_pool, conv, sync_db):
    cid, uid = conv
    assert await store.conversation_owner(cid) is None
    await store.ensure_conversation(cid, uid, "  first line\nsecond line " + "x" * 200)
    row = sync_db.execute("SELECT user_id, title, updated_at FROM conversations WHERE id = %s", (cid,)).fetchone()
    assert row["user_id"] == uid and len(row["title"]) == 70 and "\n" not in row["title"]
    await store.ensure_conversation(cid, "someone-else", "other title")  # idempotent: never changes owner or title
    again = sync_db.execute("SELECT user_id, title, updated_at FROM conversations WHERE id = %s", (cid,)).fetchone()
    assert again["user_id"] == uid and again["title"] == row["title"] and again["updated_at"] >= row["updated_at"]
    assert await store.conversation_owner(cid) == uid


async def test_messages_and_history_window(db_pool, conv):
    cid, uid = conv
    await store.ensure_conversation(cid, uid, "t")
    ids = [await store.add_message(cid, "user" if i % 2 == 0 else "assistant", f"m{i} ₹ 🚗", {"i": i, "at": dt.date(2026, 1, 1)})
           for i in range(7)]
    assert ids == sorted(ids) and len(set(ids)) == 7
    recent = await store.recent_messages(cid, 4)
    assert [m["content"] for m in recent] == ["m3 ₹ 🚗", "m4 ₹ 🚗", "m5 ₹ 🚗", "m6 ₹ 🚗"]  # last N, oldest first
    assert recent[0]["meta"] == {"i": 3, "at": "2026-01-01"}
    allm = await store.get_messages(cid)
    assert len(allm) == 7 and allm[0]["created_at"] is not None
    assert [m["id"] for m in await store.messages_after(cid, ids[4])] == ids[5:]
    convs = await store.list_conversations(uid)
    assert convs[0]["id"] == cid and convs[0]["n_messages"] == 7


async def test_state_round_trip_and_defaults(db_pool, conv):
    cid, uid = conv
    assert await store.load_state(cid) == ConvState()
    await store.ensure_conversation(cid, uid, "t")
    s = ConvState(summary="likes SUVs", summarized_upto=5, active_cars=["hyundai/creta", "kia/seltos"], focus_car="kia/seltos",
                  filters={"price_max": 1_500_000, "fuel_types": ["Diesel"]}, prefs={"city": "Pune", "note": "क्रेटा"})
    await store.save_state(cid, s)
    assert await store.load_state(cid) == s
    s.focus_car, s.active_cars = None, []
    await store.save_state(cid, s)  # upsert
    assert (await store.load_state(cid)).focus_car is None


async def test_delete_is_owner_only_and_cascades(db_pool, conv, sync_db):
    cid, uid = conv
    await store.ensure_conversation(cid, uid, "t")
    await store.add_message(cid, "user", "hi", {})
    await store.save_state(cid, ConvState(summary="x"))
    assert await store.delete_conversation(cid, "not-the-owner") is False
    assert await store.delete_conversation(cid, uid) is True
    assert sync_db.execute("SELECT COUNT(*) AS n FROM messages WHERE conversation_id = %s", (cid,)).fetchone()["n"] == 0
    assert sync_db.execute("SELECT COUNT(*) AS n FROM conversation_state WHERE conversation_id = %s", (cid,)).fetchone()["n"] == 0
    assert await store.delete_conversation(cid, uid) is False


async def test_answer_cache_put_get_overwrite_and_expiry(db_pool, sync_db):
    key = f"k-{uuid.uuid4().hex}"
    assert await store.cache_get(key) is None
    await store.cache_put(key, {"answer": "A", "when": dt.datetime(2026, 1, 1)})
    assert await store.cache_get(key) == {"answer": "A", "when": "2026-01-01 00:00:00"}
    await store.cache_put(key, {"answer": "B"})
    assert await store.cache_get(key) == {"answer": "B"}
    sync_db.execute("UPDATE answer_cache SET created_at = now() - interval '8 days' WHERE key = %s", (key,))
    assert await store.cache_get(key) is None  # stale entries are never served


async def test_feedback_row(db_pool, sync_db):
    await store.add_feedback(987654, 1, None)
    assert sync_db.execute("SELECT rating FROM feedback WHERE message_id = 987654").fetchone()["rating"] == 1


# ---------------------------------------------------------------- structured search
async def test_search_respects_every_hard_filter(db_pool, sync_db):
    f = Filters(price_max=2_000_000, fuel_types=["Diesel"], body_types=["SUV"], transmission="Automatic", seats_min=7)
    rows, desc, relax = await car_search.search_cars(f, "relevance", [], [], limit=8)
    assert rows and relax == []
    assert desc == ["price <= 20.00 L", "fuel: Diesel", "body: SUV", "automatic", "seats >= 7"]
    for r in rows:
        assert r["from_price"] <= 2_000_000 and "Diesel" in r["fuels"]
        n = sync_db.execute("""SELECT COUNT(*) AS n FROM variants v JOIN models m ON m.id = v.model_id
                               WHERE m.id = %s AND v.price <= 2000000 AND v.fuel_type = 'Diesel' AND v.is_automatic
                                 AND v.seats >= 7 AND m.body_type = 'SUV' AND m.status = 'launched'""", (r["id"],)).fetchone()["n"]
        assert n >= 1, r["id"]
        assert all(v["price"] <= 2_000_000 and v["fuel_type"] == "Diesel" for v in r["matching_variants"])


@pytest.mark.parametrize("sort,key,descending", [("price_asc", "from_price", False), ("price_desc", "from_price", True),
                                                 ("mileage", "best_mileage", True), ("power", "max_power", True)])
async def test_search_sorting(db_pool, sort, key, descending):
    rows, _, _ = await car_search.search_cars(Filters(price_max=1_500_000), sort, [], [], limit=8)
    vals = [r[key] for r in rows if r[key] is not None]
    assert len(vals) >= 5 and vals == sorted(vals, reverse=descending)


async def test_search_relaxes_impossible_budgets_and_says_so(db_pool):
    rows, _, relax = await car_search.search_cars(Filters(price_max=100_000), "relevance", [], [])
    assert rows == [] and relax == ["no exact matches, so the budget was stretched by 15% to 1.1 L"]
    rows, _, relax = await car_search.search_cars(Filters(price_max=340_000), "price_asc", [], [])  # cheapest car: ₹3.499 L
    assert relax and rows and all(340_000 < r["from_price"] <= 340_000 * 1.15 for r in rows)


async def test_search_restrict_brand_and_must_have_features(db_pool, sync_db):
    rows, _, _ = await car_search.search_cars(Filters(), "relevance", [], [], restrict_ids=["hyundai/creta", "kia/seltos"])
    assert {r["id"] for r in rows} == {"hyundai/creta", "kia/seltos"}
    rows, _, _ = await car_search.search_cars(Filters(), "relevance", [], ["tata"], limit=8)
    assert rows and all(r["id"].startswith("tata/") for r in rows)
    rows, _, _ = await car_search.search_cars(Filters(feature_tags=["ventilated_seats"], price_max=1_500_000), "price_asc", [], [])
    for r in rows:
        assert sync_db.execute("""SELECT bool_or('ventilated_seats' = ANY(feature_tags) AND price <= 1500000) AS ok
                                  FROM variants WHERE model_id = %s""", (r["id"],)).fetchone()["ok"], r["id"]


async def test_hybrid_search_keeps_mild_hybrids_for_contrast(db_pool):
    rows, _, _ = await car_search.search_cars(Filters(fuel_types=["Hybrid"], hybrid_types=["strong", "mild"], price_max=3_000_000),
                                              "relevance", [], [], limit=8)
    kinds = [set(r["hybrid_types"]) for r in rows]
    assert any("strong" in k for k in kinds) and sum(1 for k in kinds if not ({"strong", "plug-in"} & k)) >= 1


async def test_upcoming_search_fuel_filter(db_pool, sync_db):
    # B-01: "(fuel_types && fuels OR is_ev = false)" matched every non-EV upcoming car for a Diesel question
    rows = await car_search.search_upcoming(Filters(fuel_types=["Diesel"], status="upcoming"), [], limit=50)
    assert rows, "the catalog lists a few upcoming diesel models"
    assert all("Diesel" in (r["fuel_types"] or []) for r in rows), [r["full_name"] for r in rows if "Diesel" not in (r["fuel_types"] or [])]


async def test_upcoming_search_electric_and_brand(db_pool, sync_db):
    rows = await car_search.search_upcoming(Filters(fuel_types=["Electric"], status="upcoming"), [], limit=50)
    assert rows
    for r in rows:
        is_ev = sync_db.execute("SELECT is_ev FROM models WHERE id = %s", (r["id"],)).fetchone()["is_ev"]
        assert is_ev or "Electric" in (r["fuel_types"] or []), r["full_name"]
    rows = await car_search.search_upcoming(Filters(status="upcoming"), ["tata"], limit=50)
    assert rows and all(r["id"].startswith("tata/") and r["status"] == "upcoming" for r in rows)


# ---------------------------------------------------------------- hybrid retrieval
async def test_hybrid_search_filters_before_ranking(db_pool, embedder):
    q = "What is the real world mileage of the Creta?"
    qv = embedder.embed_query(q)
    rows = await hybrid_search(qv, q, ["mileage", "creta"], model_ids=["hyundai/creta"], sections=["mileage", "owner_reviews"], k=5)
    assert 1 <= len(rows) <= 5
    assert {r["model_id"] for r in rows} == {"hyundai/creta"} and {r["section"] for r in rows} <= {"mileage", "owner_reviews"}
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    assert any(any(v.startswith("keyword#") for v in r["via"]) for r in rows) and any(any(v.startswith("vector#") for v in r["via"]) for r in rows)


async def test_hybrid_search_per_model_cap_and_unicode_query(db_pool, embedder):
    ids = ["hyundai/creta", "kia/seltos", "tata/nexon"]
    q = "pros and cons"
    rows = await hybrid_search(embedder.embed_query(q), q, [], model_ids=ids, k=6, per_model_cap=2)
    assert len(rows) <= 6 and all(sum(r["model_id"] == m for r in rows) <= 2 for m in ids)
    q = "क्रेटा का माइलेज"  # no ASCII words: vector search only, no tsquery
    rows = await hybrid_search(embedder.embed_query(q), q, [], k=3)
    assert len(rows) == 3 and all(v.startswith("vector#") for r in rows for v in r["via"])


async def test_hybrid_search_never_returns_discontinued_models_unfiltered(db_pool, embedder, sync_db):
    q = "discontinued old model"
    rows = await hybrid_search(embedder.embed_query(q), q, ["discontinued"], k=10)
    statuses = {r["status"] for r in sync_db.execute("SELECT status FROM chunks WHERE id = ANY(%s)", ([r["id"] for r in rows],)).fetchall()}
    assert "discontinued" not in statuses


async def test_pinned_titled_trim_and_brand_chunks(db_pool, embedder, sync_db):
    pins = await pinned_chunks(["hyundai/creta", "kia/seltos"], ["mileage", "safety"])
    assert len(pins) == len({(p["model_id"], p["section"]) for p in pins}) and pins and all(p["via"] == ["pinned"] for p in pins)
    # a pair's verdict is stored under one of the two cars; the pipeline looks it up both ways, as here
    verdict = (await titled_chunks("hyundai/creta", "comparison", "vs Kia Seltos", 1) or
               await titled_chunks("kia/seltos", "comparison", "vs Hyundai Creta", 1))
    assert verdict and {"Kia Seltos", "Hyundai Creta"} <= set(verdict[0]["title"].replace("CarWale comparison: ", "").split(" vs "))
    t = sync_db.execute("SELECT title FROM chunks WHERE model_id = 'hyundai/creta' AND section = 'trim' LIMIT 1").fetchone()
    trim_name = t["title"].removesuffix(" trim")
    got = await trim_chunks("hyundai/creta", [trim_name])
    assert got and all(c["title"] == f"{trim_name} trim" for c in got)
    assert await trim_chunks("hyundai/creta", []) == []
    b = await brand_chunks("hyundai", embedder.embed_query("discontinued hyundai cars"), 2)
    assert b and all(c["id"].startswith("brand-hyundai::") and c["section"] == "brand" for c in b)


async def test_specs_loaders(db_pool):
    models = await specs.load_models(["hyundai/creta", "no/such-car"])
    assert set(models) == {"hyundai/creta"} and isinstance(models["hyundai/creta"]["data"], dict)
    vs = (await specs.load_variants(["hyundai/creta"]))["hyundai/creta"]
    prices = [v["price"] for v in vs if v["price"] is not None]
    assert prices == sorted(prices) and len(vs) > 5
    assert await specs.load_models([]) == {} and await specs.load_variants([]) == {}
