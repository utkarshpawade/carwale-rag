"""Unit tests: offline ingestion helpers (normalisation, parsing of CarWale's embedded JSON, chunking, loading).

All inputs are small synthetic fixtures; nothing here touches the network or the database.
"""
import gzip
import hashlib
import json

import pytest

from ingestion import build_chunks as bc
from ingestion import common
from ingestion import parse as P
from ingestion.discover_more import review_urls
from ingestion.load_db import upsert_sql
from ingestion.normalize import (canon_body, canon_fuel, canon_transmission, hybrid_type, lakh_text_to_inr, num,
                                 parse_date, parse_ncap, positive, to_int, tree_tags, variant_tags, yes)


# ---------------------------------------------------------------- numbers
@pytest.mark.parametrize("raw,expected", [(None, None), (5, 5.0), (2.5, 2.5), ("1,234.5 cc", 1234.5), ("-3 deg", -3.0),
                                          ("abc", None), ("", None), ("Rs. 12", 12.0)])
def test_num(raw, expected):
    assert num(raw) == expected


def test_to_int_and_positive():
    assert to_int("1,497 cc") == 1497 and to_int(None) is None and to_int("x") is None and to_int(3.6) == 4
    assert positive(0) is None and positive(-1) is None and positive(None) is None and positive(5) == 5


@pytest.mark.parametrize("text,expected", [
    ("Rs. 5.99 Lakh onwards", (599_000, 599_000)), ("Rs. 1.05 - 1.50 Crore", (10_500_000, 15_000_000)),
    ("Price not announced", (None, None)), ("Rs. 0.00 Lakh", (None, None)), ("", (None, None)),
    ("Rs. 10.5 L - 12 L", (1_050_000, 1_200_000)),
])
def test_lakh_text_to_inr_more(text, expected):
    assert lakh_text_to_inr(text) == expected


@pytest.mark.parametrize("raw,expected", [("12/28/2022 00:00:00", "2022-12-28"), ("2026-01-15T00:00:00", "2026-01-15"),
                                          ("1/5/2027", "2027-01-05"), ("  12/28/2022 00:00:00  ", "2022-12-28"),
                                          ("garbage", None), (None, None), ("13/45/2022", None), ("", None)])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


# ---------------------------------------------------------------- canonical values
@pytest.mark.parametrize("raw,expected", [
    ("Hybrid (Electric + Petrol)", "Hybrid"), ("Plug-in Hybrid (Electric + Petrol)", "Hybrid"),
    ("Mild Hybrid (Electric + Petrol)", "Petrol"), ("Electric", "Electric"), ("CNG", "CNG"), ("Petrol + CNG", "CNG"),
    ("Diesel", "Diesel"), ("Petrol", "Petrol"), ("LPG", "Lpg"), (None, None), ("", None),
])
def test_canon_fuel(raw, expected):
    assert canon_fuel(raw) == expected


@pytest.mark.parametrize("fuel,text,expected", [
    ("Petrol", "1.5L SHVS", "mild"), ("Petrol", "48V mild hybrid", "mild"), ("Petrol", "e:HEV", "strong"),
    ("Electric", None, None), ("Diesel", "mHawk 2.2", None), (None, None, None),
])
def test_hybrid_type_more(fuel, text, expected):
    assert hybrid_type(fuel, text) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Automatic (DSG)", ("DCT", True)), ("Automatic (Torque Converter)", ("AT", True)), ("Automatic", ("Automatic", True)),
    ("Automatic (Dual Clutch)", ("DCT", True)), ("Sequential", ("Sequential", None)), (None, (None, None)),
])
def test_canon_transmission_more(raw, expected):
    assert canon_transmission(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Coupe SUV", "SUV"), ("Sedan", "Sedan"), ("Coupe", "Coupe"), ("Roadster", "Convertible"), ("Pickup Truck", "Pickup"),
    ("Minivan", "Van"), ("Estate", "Wagon"), ("Limousine", "Limousine"), (None, None),
])
def test_canon_body_more(raw, expected):
    assert canon_body(raw) == expected


@pytest.mark.parametrize("raw,expected", [("4 Star (Global NCAP)", (4, "Global NCAP")), ("3 star", (3, None)),
                                          (None, (None, None)), ("Five stars", (None, None))])
def test_parse_ncap_more(raw, expected):
    assert parse_ncap(raw) == expected


@pytest.mark.parametrize("raw,expected", [(None, False), ("Yes", True), ("No", False), (" no ", False), ("0", False),
                                          ("N/A", False), ("-", False), ("", False), ("Optional", True), (1, True), (0, False)])
def test_yes(raw, expected):
    assert yes(raw) is expected


def test_variant_tags_more():
    specs = {"Air Conditioner Control": "Automatic Dual Zone", "Apple CarPlay": "Wireless", "Drivetrain": "4WD",
             "Seat Upholstery": "Leatherette", "Seat: Forward / Back": "Electrically adjustable", "Headlights": "LED",
             "Infotainment (Front)": "10.25 inch touchscreen", "Parking Sensors": "Front & Rear", "Ventilated Seats": "No",
             "Tyre Pressure Monitoring System": "Yes", "Hill Hold Control": "Yes"}
    tags = set(variant_tags(specs, 2))
    assert {"auto_climate", "dual_zone_climate", "apple_carplay_android_auto", "wireless_carplay", "4x4", "leather_seats",
            "powered_seats", "led_headlights", "touchscreen", "front_parking_sensors", "tpms", "hill_hold"} <= tags
    assert "ventilated_seats" not in tags and "6_airbags" not in tags
    assert variant_tags({}, None) == []


def test_tree_tags():
    groups = {"Connectivity > Mobile App Features": [{"name": "Remote AC on/off", "value": "Yes"},
                                                      {"name": "Find My Car", "value": "Yes"}],
              "Safety > ADAS": [{"name": "ADAS", "value": "Yes"}, {"name": "Automatic Emergency Braking", "value": "Yes"},
                                {"name": "Lane Departure Warning", "value": "Lane Keep Assist"}],
              "Others": [{"name": "Over The Air (OTA) Updates", "value": "Yes"},
                         {"name": "Electronic Parking Brake", "value": "Yes"}, {"name": "Sunroof", "value": "No"}]}
    assert tree_tags(groups) == ["adas", "aeb", "connected_car", "electric_parking_brake", "lane_keep_assist", "ota_updates"]
    one_app = {"Connectivity > Mobile App Features": [{"name": "Remote AC on/off", "value": "Yes"}]}
    assert tree_tags(one_app) == [] and tree_tags({}) == []


# ---------------------------------------------------------------- HTML / JSON state helpers
def test_cache_path_is_deterministic():
    p1, p2 = common.cache_path("https://a/"), common.cache_path("https://a/")
    assert p1 == p2 and p1.name == hashlib.sha1(b"https://a/").hexdigest() + ".html.gz"
    assert common.cache_path("https://b/") != p1


def test_extract_state():
    tpl = '<html><script type="application/json" id="initial-state">{}</script></html>'
    assert common.extract_state(tpl.replace("{}", '{"a": 1}')) == {"a": 1}
    assert common.extract_state(tpl.replace("{}", "{bad json")) is None
    assert common.extract_state("<html>no state</html>") is None


@pytest.mark.parametrize("raw,expected", [
    ("<p>Hello&nbsp;<b>World</b></p><ul><li>One</li><li>Two</li></ul>", "Hello World\n- One\n- Two"),
    (None, ""), ("", ""), (123, "123"), ("a  \t b", "a b"), ("Tom &amp; Jerry", "Tom & Jerry"), ("it�s", "it's"),
    ("<h2>Title</h2>Body<br/>Line", "Title\nBody\nLine"),
])
def test_clean_text(raw, expected):
    assert common.clean_text(raw) == expected


def test_list_items():
    assert common.list_items("<ul><li>A</li><li> <b>B</b> </li><li></li></ul>") == ["A", "B"]
    assert common.list_items("line1<br>- line2") == ["line1", "line2"]
    assert common.list_items(None) == []


def test_jsonl_round_trip(tmp_path):
    rows = [{"name": "क्रेटा", "price": 1}, {"emoji": "🚗"}]
    path = tmp_path / "sub" / "x.jsonl"
    assert common.write_jsonl(path, rows) == 2
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")  # blank lines are skipped
    assert common.read_jsonl(path) == rows


def test_read_cached_missing_valid_and_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RAW", tmp_path)
    url = "https://www.carwale.com/x/"
    assert common.read_cached(url) is None
    with gzip.open(common.cache_path(url), "wt", encoding="utf-8") as f:
        f.write("<html>ok</html>")
    assert common.read_cached(url) == "<html>ok</html>"
    good = common.cache_path(url).read_bytes()
    common.cache_path(url).write_bytes(good[: len(good) // 2])  # a page the fetcher is still writing
    assert common.read_cached(url) is None


# ---------------------------------------------------------------- chunking
def test_chunk_price_and_header_formatting():
    m = {"id": "x/y", "full_name": "X Y", "status": "upcoming", "body_type_raw": "SUV", "price_min": 500_000,
         "price_max": 500_000, "price_is_estimate": True, "fuel_types": ["Electric"]}
    assert bc.price_range({"price_min": None}) == "price not announced"
    assert bc.price_range(m) == "₹5.00 L ex-showroom (expected)"
    assert bc.header(m, "Overview") == "X Y | SUV | ₹5.00 L ex-showroom (expected) | Electric | UPCOMING — Overview"
    assert bc.fmt_num(None, "x") is None and bc.fmt_num(5.0, "kmpl") == "5 kmpl" and bc.fmt_num(5.5, "kmpl") == "5.5 kmpl"
    assert bc.rs(None) == "n/a" and bc.rs(15_929) == "₹15,929" and bc.rs(150_000) == "₹1.50 L"


def test_split_text_edge_cases():
    assert bc.split_text("") == [] and bc.split_text("   \n ") == [] and bc.split_text("short") == ["short"]
    long_line = "word " * 1000  # 5,000 chars, no sentence punctuation
    parts = bc.split_text(long_line, 500)
    assert all(len(p) <= 500 for p in parts) and "".join(parts).replace(" ", "") == long_line.replace(" ", "")
    sentences = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = bc.split_text(sentences, 300)
    assert all(len(p) <= 300 for p in parts) and parts[0].startswith("Sentence number 0")


def test_chunker_ids_headers_and_hashes():
    ch = bc.Chunker()
    m = {"id": "hyundai/creta", "full_name": "Hyundai Creta", "status": "launched", "body_type_raw": "SUV",
         "price_min": 1_100_000, "price_max": 2_000_000, "fuel_types": ["Petrol"], "url": "https://u/", "brand": "Hyundai",
         "body_type": "SUV"}
    ch.add(m, "faq", "FAQs", "Q: a?\nA: b\n" * 200)  # long body -> several pieces
    ch.add(None, "glossary", "ADAS", "Advanced driver assistance systems.")
    ids = [r["id"] for r in ch.rows]
    assert ids[0] == "hyundai/creta::faq::1" and ids[1] == "hyundai/creta::faq::2" and ids[-1] == "glossary::glossary::1"
    first, gl = ch.rows[0], ch.rows[-1]
    assert first["content"].startswith("Hyundai Creta | SUV | ₹11.00 L – ₹20.00 L ex-showroom | Petrol — FAQs\n")
    assert first["content_hash"] == hashlib.sha1(first["content"].encode()).hexdigest() and first["url"] == "https://u/"
    assert gl["content"] == "Car glossary — ADAS\nAdvanced driver assistance systems." and gl["model_id"] is None
    assert gl["fuel_types"] == [] and gl["url"] is None


# ---------------------------------------------------------------- CarWale page parsing
def test_spec_items_to_dict_keeps_richer_duplicate_and_units():
    items = [{"itemName": "Displacement", "value": "1497", "unitType": "cc"}, {"itemName": "Airbags", "value": "6"},
             {"itemName": "Airbags", "value": "6 airbags (front, side, curtain)"}, {"itemName": "Airbags", "value": "6"},
             {"itemName": "", "value": "x"}, {"itemName": "Mileage (ARAI)", "value": "17.4 kmpl", "unitType": "kmpl"},
             {"itemName": "Empty", "value": ""}, {"itemName": "Html", "value": "<b>Yes</b>"}]
    disp, raw = P.spec_items_to_dict(items)
    assert disp == {"Displacement": "1497 cc", "Airbags": "6 airbags (front, side, curtain)", "Mileage (ARAI)": "17.4 kmpl",
                    "Html": "Yes"}
    assert raw["Displacement"] == ("1497", "cc")
    assert P.spec_items_to_dict(None) == ({}, {})


def test_numeric_fields_ice_and_ev():
    _, raw = P.spec_items_to_dict([{"itemName": "Fuel Type", "value": "Petrol"}, {"itemName": "Displacement", "value": "1497", "unitType": "cc"},
                                   {"itemName": "Mileage (ARAI)", "value": "17.4", "unitType": "kmpl"},
                                   {"itemName": "Airbags", "value": "6 airbags"}, {"itemName": "NCAP Rating", "value": "5 Star (Bharat NCAP)"}])
    nf = P.numeric_fields(raw)
    assert (nf["engine_cc"], nf["mileage_arai"], nf["mileage_unit"], nf["airbags"], nf["ncap_rating"], nf["range_km"]) == \
        (1497, 17.4, "kmpl", 6, 5, None)
    _, raw = P.spec_items_to_dict([{"itemName": "Fuel Type", "value": "Electric"}, {"itemName": "Mileage (ARAI)", "value": "8"},
                                   {"itemName": "MIDC Certified Range (km)", "value": "465", "unitType": "km"},
                                   {"itemName": "Battery Capacity", "value": "45", "unitType": "kWh"}])
    nf = P.numeric_fields(raw)
    assert nf["mileage_arai"] is None and nf["range_km"] == 465.0 and nf["battery_kwh"] == 45.0


def test_feature_tree_and_trim_highlights():
    tp = {"specsFeaturesMaster": [{"name": "Safety", "subCategories": [
        {"name": "Airbags", "items": [{"itemName": "Airbags", "value": "6", "unitType": ""}, {"itemName": "", "value": "x"}],
         "highLights": {"lowerTrimName": "S", "addedInCurrent": [{"itemName": "ADAS", "value": "Yes"},
                                                                 {"itemName": "Speakers", "value": "8"}]}},
        {"name": "Empty", "items": []}]}]}
    assert P.feature_tree(tp) == {"Safety > Airbags": [{"name": "Airbags", "value": "6"}]}
    hl = P.trim_highlights(tp)
    assert hl["lower_trim"] == "S" and hl["added"] == ["ADAS", "Speakers: 8"] and hl["upper_trim"] is None
    assert P.trim_highlights({}) == {}


@pytest.mark.parametrize("md,expected", [
    ({"discontinuedOn": "1/1/2020"}, "discontinued"), ({"priceOverview": {"priceStatus": 4}}, "upcoming"),
    ({"priceOverview": {"priceLabel": "Estimated Price"}}, "upcoming"), ({"launchedOn": "12/31/2999 00:00:00"}, "upcoming"),
    ({"launchedOn": "1/1/2020 00:00:00", "priceOverview": {"priceLabel": "Ex-Showroom"}}, "launched"), ({}, "launched"),
])
def test_model_status(md, expected):
    assert P.model_status(md) == expected


def test_body_type_sources():
    assert P.body_type({"extendedH1SeoText": {"engineAndVariantOptions": {"bodyTypeText": " Compact SUV "}}}) == "Compact SUV"
    assert P.body_type({"metaTags": {"schema": '{"bodyType": "SUVs"}'}}) == "SUV"
    assert P.body_type({"metaTags": {"schema": '{"bodyType": "Limos"}'}}) == "Limos"
    assert P.body_type({}) is None


def test_owner_reviews_dedupes_and_marks_critical():
    r = {"id": 1, "title": "<b>Good</b>", "description": "x" * 2000, "userRating": 4,
         "reviewParameters": [{"heading": "Comfort", "answerText": "5"}, {"heading": "", "answer": 1}]}
    out = P.owner_reviews({"criticalUserReviews": [{**r, "id": 2, "userRating": 1}], "userReviews": [r, r]})
    assert [x["id"] for x in out] == [2, 1] and out[0]["kind"] == "critical" and out[1]["kind"] == "general"
    assert out[1]["title"] == "Good" and len(out[1]["text"]) == 1500 and out[1]["params"] == {"Comfort": "5"}


def test_effective_mileage_priority():
    v = {"fuel_type": "Petrol", "mileage_arai": None, "mileage_expected": 15.0, "mileage_owner": 13.0}
    P.set_effective_mileage(v)
    assert (v["mileage_effective"], v["mileage_basis"]) == (15.0, "expected")
    v = {"fuel_type": "Electric", "range_km": None, "range_owner_km": 380.0}
    P.set_effective_mileage(v)
    assert (v["mileage_effective"], v["mileage_basis"]) == (380.0, "range_km")
    v = {"fuel_type": "Diesel"}
    P.set_effective_mileage(v)
    assert (v["mileage_effective"], v["mileage_basis"]) == (None, None)


def test_fill_from_mileage_table_matches_fuel_engine_and_gearbox():
    table = [{"fuelType": "Petrol", "displacement": "1497 cc", "transmission": "Automatic (CVT)", "arai": 17.7},
             {"fuelType": "Petrol", "displacement": "1497 cc", "transmission": "Manual", "arai": 17.4,
              "userReportedArai": 15.2, "expertTested": 14.0, "mileageUnit": "kmpl"},
             {"fuelType": "Diesel", "displacement": "1493", "transmission": "Manual", "arai": 21.8},
             {"fuelType": "Petrol", "displacement": "1197", "transmission": "Manual", "arai": 20.0}]
    v = {"fuel_type": "Petrol", "engine_cc": 1497, "transmission": "Manual", "is_automatic": False}
    P.fill_from_mileage_table(v, table)
    assert (v["mileage_arai"], v["mileage_source"], v["mileage_owner"], v["mileage_city_tested"]) == (17.4, "model_table", 15.2, 14.0)
    ev = {"fuel_type": "Electric", "transmission": "Automatic", "is_automatic": True}
    P.fill_from_mileage_table(ev, [{"fuelType": "Electric", "transmission": "Automatic", "arai": 465}])
    assert ev["range_km"] == 465
    untouched = {"fuel_type": "CNG", "engine_cc": 1197}
    P.fill_from_mileage_table(untouched, table)
    assert "mileage_arai" not in untouched
    P.fill_from_mileage_table(untouched, [])


def test_merge_reviews_cap_and_critical_quota():
    crit = [{"id": f"c{i}", "text": "bad", "kind": "critical"} for i in range(30)]
    gen = [{"id": f"g{i}", "text": "good", "kind": "general"} for i in range(100)]
    out = P.merge_reviews(crit + [{"id": "c0", "text": "dup", "kind": "critical"}], gen + [{"id": "e", "text": "", "kind": "general"}], cap=80)
    assert len(out) == 80 and sum(r["kind"] == "critical" for r in out) == 20 and len({r["id"] for r in out}) == 80


def test_finalize_model_aggregates_variants():
    model = {"price_est_min": 1, "price_est_max": 2, "body_type_raw": "Compact SUV", "key_specs": {"Mileage": "17 kmpl"}}
    vs = [{"price": 1_000_000, "fuel_type": "Petrol", "hybrid_type": None, "transmission": "Manual", "is_automatic": False,
           "seats": 5, "ncap_rating": 5, "ncap_agency": "Bharat NCAP", "airbags": 6, "mileage_arai": 17.4, "range_km": None,
           "power_bhp": 113.0, "boot_l": 433, "ground_clearance_mm": 190, "feature_tags": ["sunroof"], "mileage_effective": 17.4},
          {"price": 1_800_000, "fuel_type": "Diesel", "hybrid_type": None, "transmission": "AT", "is_automatic": True,
           "seats": 7, "ncap_rating": None, "airbags": 2, "mileage_arai": 21.8, "power_bhp": 114.0, "feature_tags": ["adas"],
           "mileage_effective": 21.8}]
    tree = {"Warranty > Standard": [{"name": "Years", "value": "3"}], "Safety > ADAS": [{"name": "ADAS", "value": "Yes"}]}
    m = P.finalize_model(model, vs, tree, "SX(O)")
    assert (m["price_min"], m["price_max"], m["price_is_estimate"], m["body_type"]) == (1_000_000, 1_800_000, False, "SUV")
    assert m["fuel_types"] == ["Diesel", "Petrol"] and m["has_automatic"] and m["seats"] == [5, 7]
    assert (m["ncap_rating"], m["ncap_agency"], m["airbags_max"], m["mileage_best"]) == (5, "Bharat NCAP", 6, 21.8)
    assert m["feature_tags"] == ["adas", "sunroof"] and m["warranty"] == ["Years: 3"] and m["mileage_claimed_text"] == "17 kmpl"
    empty = P.finalize_model({"price_est_min": 5, "price_est_max": 6}, [], {}, None)
    assert (empty["price_min"], empty["price_max"], empty["price_is_estimate"], empty["airbags_max"]) == (5, 6, True, None)


def test_norm_city():
    assert P.norm_city("Navi Mumbai") == "navimumbai" and P.norm_city(None) == ""


# ---------------------------------------------------------------- loading / discovery
def test_upsert_sql():
    assert upsert_sql("models", ["id", "a", "b"], "id", ["data"]) == (
        "INSERT INTO models (id, a, b, data) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
        "a = EXCLUDED.a, b = EXCLUDED.b, data = EXCLUDED.data")


@pytest.mark.parametrize("count,pages", [(None, 0), (0, 0), (5, 1), (15, 2), (25, 3), (500, 3)])
def test_review_urls(count, pages):
    urls = review_urls({"url": "https://u/", "user_review_count": count})
    assert len(urls) == pages and (not urls or urls[0] == "https://u/reviews/")
    if pages > 1:
        assert urls[1] == "https://u/reviews/page/2/"


def test_extract_state_round_trip_on_a_realistic_page():
    state = {"modelPage": {"modelDetails": {"makeName": "Hyundai", "modelName": "Creta"}}}
    html = f'<html><head></head><body><script type="application/json" id="initial-state">{json.dumps(state)}</script></body></html>'
    assert common.extract_state(html) == state
