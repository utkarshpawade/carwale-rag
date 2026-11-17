"""Unit tests: token-budgeted context, source labels, follow-up chips, full-text query sanitising."""
import pytest

from app.rag.context import build_context, chunk_label, est_tokens
from app.rag.followups import STARTERS, short, suggest
from app.rag.hybrid_search import tsquery_from


def chunk(i, content=None, title=None, score=0.5, via=None, section="overview"):
    return {"id": f"c{i}", "model_id": "hyundai/creta", "section": section, "title": title,
            "content": content or f"Hyundai Creta | SUV | ₹11 L ex-showroom | Petrol — Overview\nbody {i} " + "x" * 200,
            "url": f"https://u/{i}", "score": score, "via": via or ["vector#1"]}


# ---------------------------------------------------------------- est_tokens / chunk_label
def test_est_tokens():
    assert est_tokens("") == 1 and est_tokens("a" * 400) == 101


@pytest.mark.parametrize("content,expected", [
    ("Hyundai Creta | SUV | ₹11 L ex-showroom | Petrol — Mileage\nbody", ("Hyundai Creta", "Mileage")),
    ("Car glossary — ADAS\nAdvanced driver assistance", ("Glossary", "ADAS")),
    ("CarWale list — Best SUVs Under 15 Lakh\n1. Creta", ("CarWale list", "Best SUVs Under 15 Lakh")),
    ("CarWale brand page — Hyundai cars in India\nbody", ("CarWale brand page", "Hyundai cars in India")),
])
def test_chunk_label(content, expected):
    # B-08: glossary chips read "GlossaryADAS" and list / brand chips repeated their title twice
    assert chunk_label(content, None) == expected


def test_chunk_label_without_header_uses_title():
    assert chunk_label("plain text", "Some title") == ("plain text", "Some title")
    assert chunk_label("", None) == ("", "")


# ---------------------------------------------------------------- build_context
def test_build_context_empty():
    assert build_context([], [], 2200) == ("", [])


def test_build_context_orders_data_then_numbered_sources():
    ctx, sources = build_context(["#### Car\n- Price: 1"], [chunk(1), chunk(2, score=None)], 2200)
    assert ctx.startswith("## CAR DATA (CarWale catalog)\n#### Car") and "## SOURCES\n[1] " in ctx and "\n\n[2] " in ctx
    assert [s["n"] for s in sources] == [1, 2]
    s1 = sources[0]
    assert (s1["car"], s1["section"], s1["url"], s1["kind"], s1["model_id"]) == \
        ("Hyundai Creta", "Overview", "https://u/1", "overview", "hyundai/creta")
    assert s1["score"] == 0.5 and s1["via"] == ["vector#1"] and sources[1]["score"] == 0.0


def test_build_context_respects_the_token_budget():
    huge_block = "- fact\n" * 5000
    ctx, sources = build_context([huge_block], [chunk(i, content="x" * 3000) for i in range(20)], 2200)
    assert ctx.split("## SOURCES")[0].rstrip().endswith("…(truncated)")
    assert est_tokens(ctx) <= 2200 + 50  # small overshoot from headers is fine
    assert 1 <= len(sources) < 20
    for line in ctx.split("## SOURCES\n", 1)[1].split("\n\n"):
        assert len(line) <= 1400 + 6  # each source is capped at 1400 chars (+ "[n] ")


def test_build_context_skips_empty_blocks_and_stops_when_budget_is_gone():
    ctx, sources = build_context(["", "real"], [chunk(1)], 30)
    assert "real" in ctx and sources == []


# ---------------------------------------------------------------- follow-ups
@pytest.mark.parametrize("name,expected", [
    ("Maruti Suzuki Swift", "Swift"), ("Mercedes-Benz GLA", "GLA"), ("Land Rover Defender", "Defender"),
    ("Toyota Urban Cruiser Hyryder", "Toyota Hyryder"), ("Hyundai Creta", "Hyundai Creta"), ("", ""),
])
def test_short(name, expected):
    assert short(name) == expected


def test_suggest_recommend():
    out = suggest("recommend", ["Hyundai Creta", "Kia Seltos"], [], False)
    assert out == ["Compare Hyundai Creta vs Kia Seltos", "Which of these is the safest?", "Which one has the best mileage?",
                   "Show only automatics"]
    assert "Show only automatics" not in suggest("recommend", ["A", "B"], [], True)
    assert suggest("recommend", ["Hyundai Creta"], [], False)[0] == "Tell me more about the Hyundai Creta"


def test_suggest_compare_and_single_car_intents():
    assert suggest("compare", ["A", "B"], [], False)[2] == "Is the B safer than the A?"
    for intent in ("spec", "price", "feature_check", "safety", "service", "review", "variant_advice", "general"):
        out = suggest(intent, ["Maruti Suzuki Swift"], ["Hyundai Grand i10 Nios"], False)
        assert len(out) == 3 and all(isinstance(x, str) and x for x in out), intent
        assert any("Swift" in x for x in out)
    assert suggest("price", ["Maruti Suzuki Swift"], ["Hyundai Grand i10 Nios"], False)[2] == "Swift vs Hyundai Grand i10 Nios"
    assert suggest("price", ["Maruti Suzuki Swift"], [], False)[2] == "Is the Swift safe?"


def test_suggest_without_cars_falls_back_to_starters():
    assert suggest("general", [], [], False) == STARTERS[:3] and suggest("chitchat", [], [], False) == STARTERS[:3]


# ---------------------------------------------------------------- tsquery sanitising
@pytest.mark.parametrize("keywords,query,expected", [
    (["boot space"], "What's the boot space of the Creta?", "boot | space | creta"),
    ([], "Creta 2026 vs Seltos", "creta | 2026 | seltos"),
    ([], "क्रेटा", None), ([], "", None), ([], "a b c", None),
    (["a'); drop table x;--"], "", "drop | table"),
    (["creta", "CRETA", "Creta"], "creta", "creta"),
])
def test_tsquery_from(keywords, query, expected):
    assert tsquery_from(keywords, query) == expected


def test_tsquery_is_limited_and_only_alphanumeric():
    q = tsquery_from([f"word{i}" for i in range(30)], "")
    assert q.count("|") == 11
    assert all(t.strip().isalnum() for t in q.split("|"))
