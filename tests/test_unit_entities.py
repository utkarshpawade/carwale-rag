"""Unit tests: car-name resolution on a small synthetic catalog (independent of the scraped data)."""
import time

import pytest

from app.rag.entities import CarResolver, is_generic, norm


def _m(mid, brand, name, rating_count=100, status="launched", is_ev=False):
    brand_slug, slug = mid.split("/")
    return {"id": mid, "brand": brand, "brand_slug": brand_slug, "name": name, "slug": slug,
            "full_name": f"{brand} {name}", "status": status, "user_rating_count": rating_count, "is_ev": is_ev}


CATALOG = [
    _m("hyundai/creta", "Hyundai", "Creta", 900), _m("hyundai/creta-electric", "Hyundai", "Creta Electric", 60, is_ev=True),
    _m("hyundai/venue", "Hyundai", "Venue", 700), _m("kia/seltos", "Kia", "Seltos", 800),
    _m("toyota/innova-crysta", "Toyota", "Innova Crysta", 400), _m("toyota/innova-hycross", "Toyota", "Innova Hycross", 300),
    _m("toyota/fortuner", "Toyota", "Fortuner", 500), _m("honda/city", "Honda", "City", 600),
    _m("maruti-suzuki/swift", "Maruti Suzuki", "Swift", 1000), _m("maruti-suzuki/grand-vitara", "Maruti Suzuki", "Grand Vitara", 450),
    _m("mahindra/xuv-7xo", "Mahindra", "XUV 7XO", 350), _m("tata/nexon", "Tata", "Nexon", 950),
    _m("tata/nexon-ev", "Tata", "Nexon EV", 200, is_ev=True), _m("tata/sierra", "Tata", "Sierra", 0, status="upcoming"),
    _m("mini/cooper-s", "Mini", "Cooper S", 20), _m("skoda/kushaq", "Skoda", "Kushaq", 150),
]


@pytest.fixture(scope="module")
def r():
    return CarResolver(CATALOG)


def top(r, mention):
    res = r.resolve(mention)
    return res[0].model_id if res else None


# ---------------------------------------------------------------- norm / is_generic
@pytest.mark.parametrize("raw,expected", [
    ("Hyundai  CRETA!!", "hyundai creta"), ("i20 N-Line", "i20 n line"), ("A&B+C", "a and b plus c"), ("", ""),
    ("   ", ""), ("\t\nCreta\n", "creta"),
])
def test_norm(raw, expected):
    assert norm(raw) == expected


@pytest.mark.parametrize("mention,generic", [
    ("", True), ("EV version", True), ("the diesel one", True), ("its automatic variant", True), ("top variant", True),
    ("creta", False), ("Creta EV", False),
])
def test_is_generic(mention, generic):
    assert is_generic(mention) is generic


# ---------------------------------------------------------------- resolve
@pytest.mark.parametrize("mention,expected", [
    ("creta", "hyundai/creta"), ("Hyundai Creta", "hyundai/creta"), ("CRETA", "hyundai/creta"), ("creta!!!", "hyundai/creta"),
    ("cretta", "hyundai/creta"),                        # typo -> fuzzy
    ("creta ev", "hyundai/creta-electric"),             # 'Electric' model also answers to 'EV'
    ("nexon electric", "tata/nexon-ev"),                # and vice versa
    ("nexon ev", "tata/nexon-ev"), ("nexon", "tata/nexon"),
    ("xuv700", "mahindra/xuv-7xo"), ("xuv 700", "mahindra/xuv-7xo"),  # manual alias for the renamed model
    ("maruti swift", "maruti-suzuki/swift"), ("suzuki swift", "maruti-suzuki/swift"),
    ("hycross", "toyota/innova-hycross"), ("toyota hycross", "toyota/innova-hycross"),
    ("fortuneer", "toyota/fortuner"), ("grand vitara", "maruti-suzuki/grand-vitara"),
    ("new creta 2026", "hyundai/creta"),                # filler words stripped
])
def test_resolve(r, mention, expected):
    assert top(r, mention) == expected


@pytest.mark.parametrize("mention", ["", "   ", "tata", "Maruti Suzuki", "EV version", "the diesel one", "xyzzy", "ab",
                                     "!!!", "car"])
def test_resolve_returns_nothing_for_non_models(r, mention):
    assert r.resolve(mention) == []


def test_ambiguous_mention_lists_alternatives(r):
    res = r.resolve("innova")
    assert res[0].model_id == "toyota/innova-crysta"  # more owner ratings ranks first
    assert res[0].ambiguous == ["toyota/innova-hycross"]


def test_brand_prefix_restricts_matches(r):
    assert top(r, "hyundai venue") == "hyundai/venue"
    assert top(r, "kia creta") == "hyundai/creta"  # wrong brand: the model name still wins rather than nothing


def test_resolve_unicode_and_hostile_input(r):
    assert top(r, "Škoda Kushaq") == "skoda/kushaq"
    assert top(r, "creta'; DROP TABLE models;--") == "hyundai/creta"
    assert r.resolve("क्रेटा") == []  # Devanagari only: nothing to match, no crash


def test_resolve_very_long_input_is_fast(r):
    t0 = time.perf_counter()
    assert top(r, "creta " + "x" * 20_000) == "hyundai/creta"
    assert time.perf_counter() - t0 < 2.0


@pytest.mark.parametrize("mention,expected", [
    ("creta hyundai", "hyundai/creta"), ("nexon tata", "tata/nexon"),
])
def test_word_order_does_not_pick_a_sibling_model(r, mention, expected):
    # B-11: the longest alias containing every query word won ties, so "creta hyundai" became the Creta Electric
    assert top(r, mention) == expected


def test_manual_alias_only_added_when_target_exists():
    res = CarResolver([m for m in CATALOG if m["id"] != "mahindra/xuv-7xo"])
    assert "xuv700" not in res.strong


# ---------------------------------------------------------------- free-text scans
def test_find_in_text(r):
    assert r.find_in_text("compare creta and seltos") == ["hyundai/creta", "kia/seltos"]
    assert r.find_in_text("honda city price") == ["honda/city"]
    assert r.find_in_text("best car to drive in the city") == []          # 'city' alone is an everyday word
    assert sorted(r.find_in_text("xuv700 vs nexon ev")) == ["mahindra/xuv-7xo", "tata/nexon-ev"]
    assert r.find_in_text("") == [] and r.find_in_text("tata cars") == []
    assert len(r.find_in_text("creta seltos nexon venue fortuner", limit=3)) == 3


def test_find_fuzzy_in_text(r):
    assert r.find_fuzzy_in_text("fortuneer price in delhi") == ["toyota/fortuner"]
    assert r.find_fuzzy_in_text("is the seltoss good") == ["kia/seltos"]
    assert r.find_fuzzy_in_text("which cars have better resale value?") == []


def test_brand_helpers(r):
    assert r.brand_slug("Maruti") == "maruti-suzuki" and r.brand_slug("unknown") is None
    assert r.brands_in_text("maruti suzuki cars under 10 lakh") == ["maruti-suzuki"]
    assert r.brands_in_text("best mini suv") == []  # 'mini' is a brand but also an everyday word
    assert r.brands_in_text("hyundai or kia?") == ["hyundai", "kia"]
    assert r.name("tata/nexon") == "Tata Nexon" and r.name("no/such") == "no/such"


def test_prefer_upcoming_when_the_user_says_upcoming():
    # B-10: 'upcoming'/'facelift'/'next' are meant to rank the upcoming model of a shared alias first
    cat = CATALOG + [_m("toyota/urban-cruiser-ev", "Toyota", "Urban Cruiser EV", 0, status="upcoming", is_ev=True),
                     _m("toyota/hyryder", "Toyota", "Urban Cruiser Hyryder", 300)]
    res = CarResolver(cat)
    assert top(res, "urban cruiser") == "toyota/hyryder"            # launched model by default
    assert top(res, "upcoming urban cruiser") == "toyota/urban-cruiser-ev"
