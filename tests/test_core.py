"""Unit tests for the deterministic parts of the pipeline (no network, no LLM)."""
import json
from pathlib import Path

import pytest

from app.rag.car_search import _where
from app.rag.hybrid_search import tsquery_from
from app.rag.orchestrator import Pipeline, merge_filters
from app.rag.schemas import PLAN_SCHEMA, Filters, empty_filters
from ingestion.normalize import (canon_body, canon_fuel, canon_transmission, hybrid_type, lakh_text_to_inr,
                                 parse_ncap, variant_tags)

MODELS = Path(__file__).resolve().parent.parent / "data" / "processed" / "models.jsonl"


# ---------------------------------------------------------------- normalisation
@pytest.mark.parametrize("text,expected", [
    ("Rs. 20.00 - 33.00 Lakh", (2_000_000, 3_300_000)),
    ("Rs. 1.20 Crore", (12_000_000, 12_000_000)),
    ("Rs. 95 Lakh - 1.2 Crore", (9_500_000, 12_000_000)),
    ("Rs.  19.53 - 32.95 Lakh", (1_953_000, 3_295_000)),
    (None, (None, None)),
])
def test_lakh_parsing(text, expected):
    assert lakh_text_to_inr(text) == expected


def test_hybrid_type():
    assert hybrid_type("Hybrid (Electric + Petrol)") == "strong"
    assert hybrid_type("Petrol", "K15C Smart Hybrid") == "mild"
    assert hybrid_type("Plug-in Hybrid (Electric + Petrol)") == "plug-in"
    assert hybrid_type("Petrol", "1.2L Turbocharged Revotron Engine") is None
    assert canon_fuel("Hybrid (Electric + Petrol)") == "Hybrid"
    assert canon_fuel("Mild Hybrid (Electric + Petrol)") == "Petrol"


@pytest.mark.parametrize("raw,expected", [
    ("Automatic (e-CVT)", ("e-CVT", True)), ("Automatic (CVT)", ("CVT", True)), ("Automatic (AMT)", ("AMT", True)),
    ("Automatic (DCT)", ("DCT", True)), ("Automatic (TC)", ("AT", True)), ("Clutchless Manual (iMT)", ("iMT", False)),
    ("Manual", ("Manual", False)),
])
def test_transmission(raw, expected):
    assert canon_transmission(raw) == expected


def test_ncap_body_tags():
    assert parse_ncap("5 Star (Bharat NCAP)") == (5, "Bharat NCAP")
    assert parse_ncap("Not Tested") == (None, None)
    assert canon_body("Sub-Compact SUV") == "SUV" and canon_body("MUV/MPV") == "MUV" and canon_body("Hatchback") == "Hatchback"
    tags = variant_tags({"Sunroof Type": "Panoramic", "Cruise Control": "Adaptive", "Parking Assist": "360 Degree Camera",
                         "Ventilated Seats": "No"}, 6)
    assert {"sunroof", "panoramic_sunroof", "adaptive_cruise", "adas", "360_camera", "6_airbags"} <= set(tags)
    assert "ventilated_seats" not in tags


# ---------------------------------------------------------------- entity resolution
@pytest.fixture(scope="module")
def resolver():
    if not MODELS.exists():
        pytest.skip("run ingestion first")
    from app.rag.entities import CarResolver
    return CarResolver([json.loads(l) for l in MODELS.open(encoding="utf-8")])


@pytest.mark.parametrize("mention,expected", [
    ("cretta", "hyundai/creta"), ("xuv700", "mahindra/xuv-7xo"), ("nexon ev", "tata/nexon-ev"), ("nexon", "tata/nexon"),
    ("model y", "tesla/model-y"), ("fortuneer", "toyota/fortuner"), ("hycross", "toyota/innova-hycross"),
    ("creta ev", "hyundai/creta-electric"), ("maruti swift", "maruti-suzuki/swift"),
])
def test_resolver(resolver, mention, expected):
    assert resolver.resolve(mention)[0].model_id == expected


def test_resolver_ambiguous_and_brand_only(resolver):
    top = resolver.resolve("innova")[0]
    assert "toyota/innova-crysta" in [top.model_id] + top.ambiguous
    assert resolver.resolve("tata") == []


@pytest.mark.parametrize("mention", ["EV version", "the diesel one", "its automatic variant", "top variant"])
def test_generic_mentions_never_match_a_car(resolver, mention):
    # regression: WRatio matched "EV version" to Maserati GranCabrio Folgore EV
    assert resolver.resolve(mention) == []


@pytest.mark.parametrize("mention,expected", [
    ("maruti grand vitara strong hybrid", "maruti-suzuki/grand-vitara"), ("toyota hycross", "toyota/innova-hycross"),
    ("tata punch ev", "tata/punch-ev"), ("i20 n line", "hyundai/i20-n-line"), ("kia carens clavis ev", "kia/carens-clavis-ev"),
])
def test_resolver_longer_mentions(resolver, mention, expected):
    assert resolver.resolve(mention)[0].model_id == expected


# ---------------------------------------------------------------- SQL safety + filters
def test_where_is_parameterized_and_whitelisted():
    f = empty_filters()
    f.price_max, f.fuel_types, f.feature_tags = 3_000_000, ["Hybrid"], ["sunroof", "x'); DROP TABLE models;--"]
    where, params, desc = _where(f, ["toyota"], None)
    sql = " AND ".join(where)
    assert "3000000" not in sql and params["pmax"] == 3_000_000
    assert params["htypes"] == ["mild", "plug-in", "strong"]
    assert params["vtags"] == ["sunroof"] and "DROP" not in json.dumps(params)


def test_tsquery_sanitised():
    assert tsquery_from(["boot space"], "What's the boot space of the Creta?") == "boot | space | creta"
    assert "'" not in (tsquery_from(["a'); drop table x;--"], "") or "")


def test_merge_filters_refine():
    old = {"price_max": 3_000_000, "fuel_types": ["Hybrid"], "hybrid_types": ["strong", "mild"]}
    new = empty_filters()
    new.seats_min = 7
    merged = merge_filters(old, new)
    assert merged.price_max == 3_000_000 and merged.seats_min == 7 and merged.fuel_types == ["Hybrid"]


def test_plan_schema_is_strict():
    def walk(node):
        if isinstance(node, dict):
            if "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(PLAN_SCHEMA)
    Filters.model_validate(empty_filters().model_dump())


def test_split_text_keeps_lines_and_limit():
    from ingestion.build_chunks import split_text
    text = "\n".join(f"Group {i}: " + "; ".join(f"feature {i}-{j}" for j in range(40)) for i in range(6))
    parts = split_text(text, 500)
    assert all(len(p) <= 500 for p in parts)
    assert any("\n" in p for p in parts) or len(parts) == len(text.split("\n"))
    assert "".join(parts).count("feature 5-39") == 1


def test_digest():
    d = Pipeline._digest("**Grand Vitara** is | great |", ["Maruti Suzuki Grand Vitara"])
    assert d.startswith("[cars: Maruti Suzuki Grand Vitara]") and "|" not in d.split("]", 1)[1]
