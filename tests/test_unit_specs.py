"""Unit tests: deterministic fact sheets, tables and feature/trim helpers (rag/specs.py)."""
import copy

import pytest

from app.rag.specs import (card, city_variant_prices, compare_table, crash_rating, detect_features, facts, facts_block,
                           feature_availability, feature_list, filter_variants, inr, mentioned_trims, num, powertrains,
                           price_range, search_table, trim_comparison, trim_feature_matrix, variants_table)


def V(name, trim, fuel="Petrol", price=1_000_000, auto=False, htype=None, cc=1497, trans="Manual", tags=(), **kw):
    return {"name": name, "trim": trim, "fuel_type": fuel, "hybrid_type": htype, "is_automatic": auto, "price": price,
            "engine_cc": cc, "transmission": trans, "feature_tags": list(tags), "power_bhp": 113.0, "torque_nm": 144.0,
            "mileage_arai": 17.4, "mileage_unit": "kmpl", "seats": 5, "airbags": 6, **kw}


VARIANTS = [
    V("E 1.5 Petrol MT", "E", price=1_100_000),
    V("EX 1.5 Petrol MT", "EX", price=1_250_000, tags=["sunroof"]),
    V("S 1.5 Petrol MT", "S", price=1_400_000, tags=["sunroof", "wireless_charger"]),
    V("SX 1.5 Diesel AT", "SX", fuel="Diesel", price=1_800_000, auto=True, trans="AT", cc=1493,
      tags=["sunroof", "wireless_charger", "ventilated_seats", "adas"]),
]

MODEL = {
    "id": "hyundai/creta", "full_name": "Hyundai Creta", "url": "https://www.carwale.com/hyundai-cars/creta/",
    "status": "launched", "body_type_raw": "Compact SUV", "price_is_estimate": False, "boot_l": 433,
    "ground_clearance_mm": 190, "ncap_rating": 5, "ncap_agency": "Bharat NCAP", "user_rating": 4.5,
    "user_rating_count": 900, "expert_rating_5": 4.2, "feature_tags": ["adas", "sunroof", "panoramic_sunroof", "esc"],
    "image_url": "https://img/creta.png", "price_min": 1_100_000, "price_max": 2_000_000,
    "data": {
        "warranty": ["Standard: 3 years / unlimited km"], "pros": ["Comfortable", "Feature-rich"], "cons": ["Pricey"],
        "city_prices": [{"city": "Mumbai", "range": "13 - 24 Lakh"}],
        "city_pages": [{"city": "Delhi",
                        "versions": [{"name": "SX 1.5 Diesel AT", "trim": "SX", "on_road": 2_100_000, "ex_showroom": 1_800_000},
                                     {"name": "E 1.5 Petrol MT", "trim": "E", "on_road": 1_250_000, "ex_showroom": 1_100_000}],
                        "breakup": {"version": "E 1.5 Petrol MT", "charges": [
                            {"name": "RTO", "group": "Taxes", "value": 110_000},
                            {"name": "Insurance", "group": "Insurance", "value": 45_000},
                            {"name": "Extended Warranty", "group": "Warranties", "value": 20_000}]}}],
        "colours": ["Abyss Black", "Atlas White"],
        "offers": {"city": "Mumbai", "items": ["Cash discount up to Rs. 20,000"], "valid": "till 30 Nov"},
    },
}


# ---------------------------------------------------------------- formatting
@pytest.mark.parametrize("v,expected", [(None, "n/a"), (0, "n/a"), (550_000, "₹5.50 L"), (10_000_000, "₹1.00 Cr"),
                                        (25_500_000, "₹2.55 Cr"), (99_000, "₹0.99 L")])
def test_inr(v, expected):
    assert inr(v) == expected


@pytest.mark.parametrize("lo,hi,expected", [(None, None, "n/a"), (0, 900_000, "n/a"), (500_000, None, "₹5.00 L"),
                                            (500_000, 500_000, "₹5.00 L"), (500_000, 900_000, "₹5.00 L – ₹9.00 L")])
def test_price_range(lo, hi, expected):
    assert price_range(lo, hi) == expected


@pytest.mark.parametrize("v,unit,nd,expected", [(None, "", 1, None), ("", "", 1, None), (0, "kmpl", 1, None), (5, "", 1, "5"),
                                                (5.0, "kmpl", 1, "5 kmpl"), (20.126, "kmpl", 2, "20.13 kmpl"), (-3, "", 1, "-3")])
def test_num(v, unit, nd, expected):
    assert num(v, unit, nd) == expected


def test_feature_list():
    assert feature_list(None) == "basic equipment" and feature_list([]) == "basic equipment"
    assert feature_list(["sunroof", "panoramic_sunroof", "adas"]) == "ADAS, panoramic sunroof"  # sunroof folded into panoramic
    assert feature_list(["4x4", "hud", "adas", "ota_updates"], limit=2) == "ADAS, OTA updates"  # HEADLINE_TAGS order
    assert feature_list(["unknown_tag"]) == "basic equipment"


# ---------------------------------------------------------------- variant filtering / grouping
def test_filter_variants_by_fuel_transmission_and_notes():
    assert [v["trim"] for v in filter_variants(VARIANTS, "Diesel", None, None)[0]] == ["SX"]
    out, notes = filter_variants(VARIANTS, "CNG", None, None)
    assert out == VARIANTS and notes == ["no CNG variant is listed"]
    assert [v["trim"] for v in filter_variants(VARIANTS, None, "Automatic", None)[0]] == ["SX"]
    out, notes = filter_variants(VARIANTS, "Diesel", "Manual", None)
    assert [v["trim"] for v in out] == ["SX"] and notes == ["no manual variant is listed"]
    out, notes = filter_variants(VARIANTS, None, None, "ZX (O)")
    assert out == VARIANTS and notes == ["variant 'ZX (O)' not found; showing all"]
    assert filter_variants([], "Diesel", "Automatic", "SX") == ([], ["no Diesel variant is listed",
                                                                     "no automatic variant is listed",
                                                                     "variant 'SX' not found; showing all"])


def test_filter_variants_hybrid_means_strong_or_plugin_not_mild():
    vs = [V("Mild", "A", htype="mild"), V("Strong", "B", fuel="Hybrid", htype="strong"), V("PHEV", "C", htype="plug-in")]
    assert [v["name"] for v in filter_variants(vs, "Hybrid", None, None)[0]] == ["Strong", "PHEV"]


@pytest.mark.parametrize("hint,expected", [("SX", ["SX"]), ("S", ["S"]), ("E", ["E"]), ("EX", ["EX"])])
def test_filter_variants_short_trim_hint_matches_that_trim_only(hint, expected):
    # B-12: the hint was matched as a substring of the variant name, so "E" matched every name containing an "e"
    # (P-e-trol, Di-e-sel, EX ...) and "S" matched "SX": the answer then showed the whole model instead of the trim
    assert [v["trim"] for v in filter_variants(VARIANTS, None, None, hint)[0]] == expected


def test_powertrains_groups_and_labels():
    vs = VARIANTS + [V("ZX e-CVT", "ZX", fuel="Hybrid", htype="strong", cc=1490, trans="e-CVT", auto=True, price=2_000_000,
                       mileage_arai=27.97, mileage_owner=22.5, mileage_city_tested=18.0, mileage_highway_tested=None),
                     V("Smart Hybrid AT", "Alpha", htype="mild", cc=1462, trans="AT", auto=True, price=1_500_000)]
    pts = powertrains(vs)
    labels = [p["label"] for p in pts]
    assert labels == ["Petrol 1497 cc Manual", "Diesel 1493 cc AT", "strong hybrid 1490 cc e-CVT", "Petrol mild-hybrid 1462 cc AT"]
    petrol = pts[0]
    assert petrol["price"] == "₹11.00 L – ₹14.00 L" and petrol["power"] == "113 bhp" and petrol["mileage"] == "17.4 kmpl"
    hyb = pts[2]
    assert hyb["mileage"] == "27.97 kmpl" and hyb["mileage_real"] == "22.5 kmpl" and hyb["tested"] == "city 18 / highway None"
    assert powertrains([]) == []


def test_powertrains_price_when_no_variant_is_priced():
    assert powertrains([V("X", "X", price=None)])[0]["price"] == "n/a"


def test_crash_rating_variants():
    assert crash_rating({"ncap_rating": 4}, []) == "4-star (NCAP)"
    assert crash_rating({}, [{"specs": {"NCAP Rating": " not tested "}}]).startswith("Not Tested")
    assert crash_rating({}, []) == "no crash-test rating listed on CarWale"


# ---------------------------------------------------------------- fact sheets
def test_facts_limited_to_the_aspects_asked():
    f = facts(MODEL, VARIANTS, {"mileage"})
    assert list(f)[:3] == ["Price (ex-showroom)", "Body type", "Powertrains"]
    assert "Mileage (ARAI claimed)" in f and "Airbags" not in f and "Warranty" not in f and "Owner rating" not in f
    assert f["Price (ex-showroom)"] == "₹11.00 L – ₹18.00 L"


def test_facts_without_aspects_or_in_compare_include_everything():
    for f in (facts(MODEL, VARIANTS, set()), facts(MODEL, VARIANTS, {"mileage"}, compare=True)):
        for key in ("Power / torque", "Seating", "Boot space", "Crash rating", "Airbags", "Safety tech", "Warranty",
                    "Owner rating", "Pros", "Cons", "Variants", "Key features (top variant)"):
            assert key in f, key
    f = facts(MODEL, VARIANTS, set())
    assert f["Crash rating"] == "5-star (Bharat NCAP)" and f["Safety tech"] == "ADAS, ESC" and f["Owner rating"] == "4.5/5 (900 ratings)"
    assert f["Variants"] == "4 versions from E 1.5 Petrol MT to SX 1.5 Diesel AT"


def test_facts_city_on_road_price_from_city_page():
    f = facts(MODEL, VARIANTS, {"price"}, city="delhi")
    assert f["On-road price (Delhi)"] == "₹12.50 L (E 1.5 Petrol MT) to ₹21.00 L (SX 1.5 Diesel AT), 2 variants"
    assert f["On-road breakup (E 1.5 Petrol MT)"] == "RTO ₹1.10 L; Insurance ₹45,000"  # warranty add-ons excluded


def test_facts_city_fallbacks():
    assert facts(MODEL, VARIANTS, {"price"}, city="Mumbai")["On-road price (Mumbai)"] == "Rs. 13 - 24 Lakh"
    assert facts(MODEL, VARIANTS, {"price"}, city="Pune")["On-road price (Pune)"] == "not listed for Pune (listed cities: Mumbai)"
    assert facts(MODEL, VARIANTS, {"on_road_price"})["On-road prices"] == "Mumbai Rs. 13 - 24 Lakh"


def test_facts_offers_and_colours():
    f = facts(MODEL, VARIANTS, {"offers", "colours"})
    assert f["Current offers (CarWale, shown for Mumbai; other cities and dealers may differ)"] == \
        "Cash discount up to Rs. 20,000 (till 30 Nov)"
    assert f["Colours (2)"] == "Abyss Black, Atlas White"
    m = copy.deepcopy(MODEL)
    m["data"]["offers"] = None
    assert facts(m, VARIANTS, {"offers"})["Current offers"] == "no offer listed on CarWale right now"


def test_facts_upcoming_model_without_variants():
    m = {**MODEL, "status": "upcoming", "launched_on": "2027-01-15", "price_is_estimate": True, "data": None}
    f = facts(m, [], {"specs", "price"})
    assert f["Price (ex-showroom)"] == "₹11.00 L – ₹20.00 L (expected)"
    assert f["Status"] == "upcoming, expected 2027-01-15" and f["Seating"] == "n/a" and f["Variants"] == "0 versions"
    assert "Dimensions (L×W×H)" not in f and "Powertrains" not in f


def test_facts_dimensions_from_top_variant():
    vs = [V("Base", "A", price=900_000), V("Top", "B", price=1_500_000, length_mm=4330, width_mm=1790, height_mm=1635,
                                           wheelbase_mm=2610, fuel_tank_l=50)]
    f = facts(MODEL, vs, {"dimensions"})
    assert f["Dimensions (L×W×H)"] == "4330×1790×1635 mm, wheelbase 2610 mm" and f["Fuel tank"] == "50 litres"


# ---------------------------------------------------------------- feature detection / availability
@pytest.mark.parametrize("text,expected", [
    ("Which Sonet variants get ventilated seats and a panoramic sunroof?", ["ventilated_seats", "panoramic_sunroof"]),
    ("does it have adaptive cruise control", ["adaptive_cruise"]),
    ("does the creta have cruise control?", ["cruise_control"]),
    ("sunroof first, then a panoramic sunroof", ["sunroof", "panoramic_sunroof"]),
    ("wireless charging and leatherette seats", ["wireless_charger", "leather_seats"]),
    ("is there an OTA update and ESC?", ["ota_updates", "esc"]),
    ("turbocharged petrol with 6 airbags", ["turbo", "6_airbags"]),
    ("360 degree camera", ["360_camera"]),
    ("", []),
])
def test_detect_features(text, expected):
    assert detect_features(text) == expected


@pytest.mark.parametrize("text", [
    "What is the price of the Toyota Fortuner?", "Toyota Urban Cruiser Hyryder mileage", "does it have hill descent control",
    "describe the escape route", "Toyota Innova Hycross on-road price in Delhi",
])
def test_detect_features_ignores_words_that_merely_contain_a_feature_name(text):
    # B-07: substring matching read "ota" inside "Toy-ota" as OTA updates, "cruise" inside "Cruiser" as cruise control
    # and "esc" inside "descent"; those phantom features were then injected into the answer context
    assert detect_features(text) == []


def test_feature_availability_by_trim():
    out = feature_availability("Hyundai Creta", VARIANTS, ["sunroof", "adas", "hud"])
    assert out.startswith("Feature availability for Hyundai Creta by trim (all 4 trims")
    assert "- sunroof: available on EX, S, SX (cheapest: EX 1.5 Petrol MT at ₹12.50 L); not on E." in out
    assert "- ADAS: available on SX (cheapest: SX 1.5 Diesel AT at ₹18.00 L); not on E, EX, S." in out
    assert "- head-up display: not available on any trim." in out
    every = feature_availability("X", [V("a", "A", tags=["esc"]), V("b", "B", tags=["esc"])], ["esc"])
    assert "; on every trim." in every


def test_mentioned_trims_skips_ambiguous_short_names_unless_hinted():
    assert mentioned_trims("is the SX worth it over the S?", VARIANTS) == ["SX"]
    assert mentioned_trims("is the SX worth it over the S?", VARIANTS, hints=["S"]) == ["SX", "S"]
    assert mentioned_trims("", VARIANTS) == [] and mentioned_trims("SX vs EX", []) == []


def test_trim_comparison_and_matrix():
    out = trim_comparison("Hyundai Creta", VARIANTS, "SX", "S")
    assert "- SX has but S lacks: ADAS, ventilated seats" in out and "- S has but SX lacks: nothing in the tracked feature list" in out
    assert "- Prices: SX ₹18.00 L; S ₹14.00 L" in out
    assert trim_comparison("Hyundai Creta", VARIANTS, "SX", "ZZ") == ""
    matrix = trim_feature_matrix("Hyundai Creta", VARIANTS)
    assert matrix.splitlines()[1] == "- E (₹11.00 L): basic equipment" and len(matrix.splitlines()) == 5


def test_city_variant_prices():
    full = city_variant_prices(MODEL, "Delhi")
    assert full.splitlines()[0] == "On-road prices in Delhi for Hyundai Creta (all variants, CarWale data):"
    assert full.splitlines()[1] == "- E 1.5 Petrol MT: ₹12.50 L on-road (ex-showroom ₹11.00 L)"  # cheapest first
    assert "trim SX" in city_variant_prices(MODEL, "delhi", trims=["SX"])
    only = city_variant_prices(MODEL, "Delhi", only=[{"name": "E 1.5 Petrol MT"}])
    assert "the variants asked about" in only and "SX 1.5" not in only
    assert city_variant_prices(MODEL, "Pune") == "" and city_variant_prices({"data": None}, "Delhi") == ""
    assert city_variant_prices(MODEL, "Delhi", trims=["ZZ"]) == ""


# ---------------------------------------------------------------- tables / cards
def test_variants_table_limits_rows():
    vs = [V(f"V{i}", f"T{i}", price=1_000_000 + i) for i in range(16)]
    lines = variants_table(vs).splitlines()
    assert len(lines) == 2 + 14 + 1 and lines[-1].startswith("| … 2 more variants")
    ev = variants_table([V("EV", "EV", fuel="Electric", mileage_arai=None, range_km=465.0)])
    assert "| 465 km |" in ev


def test_compare_table_escapes_pipes_and_fills_gaps():
    t = compare_table(["A", "B"], [{"Price": "1|2"}, {"Seats": "5"}])
    assert t.splitlines() == ["| Attribute | A | B |", "|---|---|---|", "| Price | 1/2 | n/a |", "| Seats | n/a | 5 |"]


def test_facts_block():
    assert facts_block("Car", "u", {"A": "1"}) == "#### Car (u)\n- A: 1"


def test_search_table_rows():
    rows = [{"full_name": "EV One", "from_price": 1_500_000, "to_price": 2_000_000, "hybrid_types": [], "fuels": ["Electric"],
             "best_range": 465.0, "ncap_rating": 5, "ncap_agency": "Bharat NCAP", "user_rating": 4.25, "feature_tags": ["adas"],
             "matching_variants": [{"name": "LR"}]},
            {"full_name": "Hybrid Two", "from_price": 1_200_000, "to_price": None, "hybrid_types": ["strong"], "fuels": ["Hybrid"],
             "best_mileage": 27.97, "best_mileage_basis": "ARAI claimed", "max_airbags": 6, "feature_tags": []},
            {"full_name": "Plain", "from_price": 600_000, "fuels": ["Petrol"], "feature_tags": None}]
    lines = search_table(rows).splitlines()
    assert lines[2] == "| 1 | EV One | ₹15.00 L – ₹20.00 L (from LR) | Electric | 465 km range | ADAS | 5★ Bharat NCAP | 4.2/5 |"
    assert lines[3] == "| 2 | Hybrid Two | ₹12.00 L | strong hybrid | 27.97 kmpl (ARAI claimed) | basic equipment | 6 airbags | n/a |"
    assert lines[4] == "| 3 | Plain | ₹6.00 L | Petrol | not listed | basic equipment | ? airbags | n/a |"


def test_card_prefers_search_prices():
    c = card({**MODEL, "from_price": 1_200_000, "to_price": 1_300_000})
    assert c == {"id": "hyundai/creta", "name": "Hyundai Creta", "url": MODEL["url"], "image": "https://img/creta.png",
                 "price": "₹12.00 L – ₹13.00 L", "body": "Compact SUV", "rating": 4.5, "status": "launched", "ncap": 5}
    assert card(MODEL)["price"] == "₹11.00 L – ₹20.00 L"
