"""Unit tests: the pipeline's deterministic decisions (no DB, no LLM) on a synthetic catalog."""
import pytest

from app.llm.client import LLMUnavailable
from app.memory.store import ConvState
from app.rag.entities import CarResolver
from app.rag.orchestrator import CITY_RE, DISCONTINUED_RE, OFFER_RE, SMALLTALK, Pipeline, canon_city, merge_filters
from app.rag.schemas import Filters, QueryPlan, empty_filters

from test_unit_entities import CATALOG


@pytest.fixture(scope="module")
def p():
    pipe = Pipeline.__new__(Pipeline)
    pipe.resolver = CarResolver(CATALOG)
    return pipe


def plan(**kw) -> QueryPlan:
    return QueryPlan.model_validate({"standalone_query": "q", **kw})


# ---------------------------------------------------------------- merge_filters (refine)
def test_merge_filters_adds_and_replaces():
    old = {"price_max": 3_000_000, "fuel_types": ["Hybrid"], "hybrid_types": ["strong", "mild"]}
    merged = merge_filters(old, Filters(seats_min=7, fuel_types=["Diesel"]))
    assert merged.price_max == 3_000_000 and merged.seats_min == 7 and merged.fuel_types == ["Diesel"]
    assert merged.hybrid_types == ["strong", "mild"]


def test_merge_filters_tolerates_missing_or_stale_state():
    assert merge_filters(None, Filters(seats_min=7)).seats_min == 7
    merged = merge_filters({"price_max": "abc", "removed_field": 1, "status": "upcoming"}, empty_filters())
    assert merged.price_max is None and merged.status == "upcoming"


# ---------------------------------------------------------------- cities and cue regexes
@pytest.mark.parametrize("raw,expected", [
    ("bengaluru", "Bangalore"), ("  New   Delhi ", "Delhi"), ("navi-mumbai", "Navi Mumbai"),
    ("pimpri chinchwad", "Pimpri-Chinchwad"), ("PUNE", "Pune"), ("timbuktu", "Timbuktu"), ("", None), (None, None),
])
def test_canon_city(raw, expected):
    assert canon_city(raw) == expected


@pytest.mark.parametrize("text,city", [
    ("creta price in pune", "pune"), ("on road price navi mumbai", "navi mumbai"), ("honda city price", None),
    ("methane emissions", None), ("viagra", None), ("Bengaluru on-road", "Bengaluru"),
])
def test_city_regex(text, city):
    m = CITY_RE.search(text)
    assert (m.group(1) if m else None) == city


@pytest.mark.parametrize("text,match", [
    ("hi", True), ("Hello!!", True), ("thanks a lot", True), ("ok.", True), ("good morning", True), ("bye", True),
    ("hi, creta price?", False), ("hello creta", False), ("thanks for the creta info, now seltos?", False), ("", False),
])
def test_smalltalk(text, match):
    assert bool(SMALLTALK.match(text)) is match


def test_offer_and_discontinued_cues():
    assert OFFER_RE.search("best offers on the creta") and OFFER_RE.search("year-end offers") and OFFER_RE.search("cashback?")
    assert not OFFER_RE.search("the creta offers a sunroof")
    assert DISCONTINUED_RE.search("is the polo discontinued?") and DISCONTINUED_RE.search("no longer sold in india")
    assert not DISCONTINUED_RE.search("continued support")


# ---------------------------------------------------------------- small helpers
def test_same_family():
    assert Pipeline._same_family("tata/punch", "tata/punch-ev") and Pipeline._same_family("tata/punch-ev", "tata/punch")
    assert Pipeline._same_family("tata/punch", "tata/punch")
    assert not Pipeline._same_family("tata/punch", "tata/nexon")


def test_digest():
    d = Pipeline._digest("**Grand Vitara** | 27 kmpl |\n\n- ok", [f"Car {i}" for i in range(9)])
    assert d.startswith("[cars: Car 0, Car 1, Car 2, Car 3, Car 4, Car 5] ") and "Car 6" not in d
    assert Pipeline._digest("x" * 1000, []) == "x" * 420


def test_busy_messages():
    assert "not configured correctly" in Pipeline._busy(LLMUnavailable("k", auth=True))
    assert Pipeline._busy(LLMUnavailable("r", 12.7, rate_limited=True)).endswith("free-tier limit right now. Please try again in about 12 seconds.")
    assert Pipeline._busy(LLMUnavailable("r", rate_limited=True)).endswith("in a moment.")
    # B-03: a failure that is not a rate limit must not be reported as one
    down = Pipeline._busy(LLMUnavailable("connection refused"))
    assert "free-tier" not in down and "temporarily unavailable" in down
    assert Pipeline._retry_text(LLMUnavailable("x", rate_limited=True), 2.0) == "Groq free-tier limit reached — retrying in 2s…"
    assert "free-tier" not in Pipeline._retry_text(LLMUnavailable("x"), 3.0)


def test_best_list_matching():
    ok, p0 = Pipeline._best_list_ok, plan(intent="recommend")
    f = Filters(brands=["Tata"], transmission="Automatic")
    assert ok(p0, f, "Best Tata Automatic Cars in India") and not ok(p0, f, "Best Tata Cars")
    assert ok(p0, Filters(body_types=["MUV"]), "Best MPV cars") and ok(p0, Filters(body_types=["MUV"]), "Best MUV cars")
    assert ok(p0, Filters(fuel_types=["Electric"], price_max=2_000_000), "Best Electric Cars Under 20 Lakh")
    assert not ok(p0, Filters(), "Best cars")


# ---------------------------------------------------------------- target resolution with memory
ST = ConvState(active_cars=["hyundai/creta", "kia/seltos", "tata/nexon"], focus_car="hyundai/creta")


def test_targets_from_memory(p):
    assert p._targets(plan(intent="safety", refers_to="focus"), ST)[0] == ["hyundai/creta"]
    assert p._targets(plan(intent="compare", refers_to="all"), ST)[0] == ST.active_cars
    ids = p._targets(plan(intent="safety", refers_to="indices", active_indices=[2, 5, 0, -1]), ST)[0]
    assert ids == ["kia/seltos"]  # out-of-range indices ignored


def test_targets_explicit_car_replaces_focus_except_in_compare(p):
    assert p._targets(plan(intent="spec", refers_to="focus", cars=["Seltos"]), ST)[0] == ["kia/seltos"]
    assert p._targets(plan(intent="compare", refers_to="focus", cars=["Seltos"]), ST)[0] == ["hyundai/creta", "kia/seltos"]


def test_targets_generic_mentions_modify_the_focus_car(p):
    st = ConvState(focus_car="tata/nexon")
    ids, resolved, _, _ = p._targets(plan(intent="spec", cars=["EV version"]), st)
    assert ids == ["tata/nexon-ev"] and resolved[0]["score"] == "memory"
    assert p._targets(plan(intent="spec", cars=["the diesel one"]), ST)[0] == ["hyundai/creta"]
    assert p._targets(plan(intent="spec", cars=["EV version"]), ConvState())[0] == []  # nothing to modify


def test_targets_ambiguity_and_unresolved(p):
    ids, resolved, unresolved, notes = p._targets(plan(intent="spec", cars=["innova", "Batmobile"]), ConvState())
    assert ids == ["toyota/innova-crysta", "toyota/innova-hycross"] and unresolved == ["Batmobile"]
    assert resolved[0]["ambiguous"] == ["toyota/innova-hycross"] and "matches more than one model" in notes[0]
    assert p._targets(plan(intent="recommend", cars=["innova"]), ConvState())[0] == ["toyota/innova-crysta"]


def test_targets_fuel_hint_electric_switches_to_ev_model(p):
    ids = p._targets(plan(intent="spec", cars=[{"mention": "Creta", "fuel_hint": "Electric"}]), ConvState())[0]
    assert ids == ["hyundai/creta-electric"]


def test_targets_planner_forgot_refers_to(p):
    assert p._targets(plan(intent="spec"), ST)[0] == ["hyundai/creta"]
    assert p._targets(plan(intent="recommend"), ST)[0] == []


def test_targets_dedupes_preserving_order(p):
    assert p._targets(plan(intent="compare", cars=["Creta", "hyundai creta", "Seltos"]), ConvState())[0] == \
        ["hyundai/creta", "kia/seltos"]


# ---------------------------------------------------------------- backstops
def test_backstop_out_of_scope_with_a_car_name_is_in_scope(p):
    pl = plan(intent="out_of_scope")
    notes = p._backstops(pl, "what is the boot space of the creta")
    assert pl.intent == "spec" and [c.mention for c in pl.cars] == ["Hyundai Creta"] and "in scope" in notes[-1]
    pl = plan(intent="out_of_scope")
    p._backstops(pl, "nexon warranty")
    assert pl.intent == "service"
    pl = plan(intent="chitchat")
    p._backstops(pl, "hi")
    assert pl.intent == "chitchat"


def test_backstop_car_names_from_text_when_planner_omits_them(p):
    pl = plan(intent="price")
    assert "car names taken from the message" in " ".join(p._backstops(pl, "seltos on road cost"))
    assert [c.mention for c in pl.cars] == ["Kia Seltos"]


def test_backstop_filtered_question_without_car_becomes_search(p):
    pl = plan(intent="safety", filters={"body_types": ["SUV"], "price_max": 1_500_000})
    p._backstops(pl, "Safest SUVs under 15 lakh")
    assert pl.intent == "recommend" and pl.sort_by == "safety" and pl.filter_mode == "new"


def test_backstop_fuel_words(p):
    pl = plan(intent="recommend")
    p._backstops(pl, "electric suvs with the best range")
    assert pl.filters.fuel_types == ["Electric"] and pl.sort_by == "range"
    pl = plan(intent="recommend")
    p._backstops(pl, "hybrids for city use")
    assert pl.filters.fuel_types == ["Hybrid"] and pl.filters.hybrid_types == ["strong", "mild"]


def test_backstop_aspects_and_city(p):
    pl = plan(intent="price", cars=["Creta"], city="bengaluru")
    p._backstops(pl, "latest news, colours and discounts on the creta")
    assert {"news", "colours", "offers"} <= set(pl.aspects) and pl.city == "Bangalore"
    pl = plan(intent="price", cars=["Creta"])
    notes = p._backstops(pl, "creta on road price in pune")
    assert pl.city == "Pune" and any("city Pune" in n for n in notes)


def test_backstop_discontinued_brand_question(p):
    pl = plan(intent="recommend", filters={"brands": ["Hyundai"]})
    p._backstops(pl, "Which Hyundai cars are discontinued?")
    assert pl.intent == "general" and pl.filters.brands == []


def test_ground_cars_replaces_a_car_the_user_never_named(p):
    pl = plan(intent="price", cars=[{"mention": "Hyundai Venue", "variant_hint": "SX"}])
    notes = p._ground_cars(pl, "fortuneer price in delhi")
    assert notes and [c.mention for c in pl.cars] == ["Toyota Fortuner"] and pl.cars[0].variant_hint is None
    pl = plan(intent="recommend", cars=["Hyundai Venue"])
    assert p._ground_cars(pl, "fortuneer price") == []  # only for car-specific intents
