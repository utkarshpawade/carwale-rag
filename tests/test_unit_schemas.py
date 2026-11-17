"""Unit tests: QueryPlan / Filters / CarRef - the tolerant schema the LLM planner output is validated against."""
import json
import math

import pytest
from pydantic import ValidationError

from app.rag.schemas import (BODY_MAP, FEATURE_TAGS, FUEL_MAP, HYB_MAP, PLAN_EXAMPLE, PLAN_SCHEMA, SORT_MAP, TRANS_MAP,
                             CarRef, Filters, QueryPlan, _cars, _enum, _enum_list, _float, _int, _price, _str_list,
                             _tags, empty_filters, simple_plan, strict_json_schema)


# ---------------------------------------------------------------- _price: INR, lakh/crore text, bare numbers = lakh
@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), (0, None), (0.0, None), (-5, None), ("abc", None),
    (1_500_000, 1_500_000), (15, 1_500_000), (15.5, 1_550_000), ("15", 1_500_000),
    ("15 lakh", 1_500_000), ("15 lakhs", 1_500_000), ("15 lac", 1_500_000), ("20 L", 2_000_000),
    ("1.2 crore", 12_000_000), ("1.2 cr", 12_000_000), ("1.5Cr", 15_000_000), ("15,00,000", 1_500_000),
    ("₹12 lakh", 1_200_000),
])
def test_price_parsing(value, expected):
    assert _price(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("Rs. 15 lakh", 1_500_000), ("Rs.15 Lakh", 1_500_000), ("INR 8.5 lakh", 850_000), ("Rs. 1.2 Crore", 12_000_000),
])
def test_price_with_currency_prefix(value, expected):
    # B-04: the "." of "Rs." used to be read as the number -> float('.') -> the whole plan was rejected
    assert Filters.model_validate({"price_max": value}).price_max == expected


@pytest.mark.parametrize("bad", [[1_500_000], {"value": 15}, math.inf, -math.inf, math.nan])
def test_price_odd_types_never_crash_validation(bad):
    # B-05: lists/dicts/infinity raised TypeError/OverflowError, which escaped Pydantic and crashed the chat turn
    assert Filters.model_validate({"price_max": bad}).price_max is None


def test_non_finite_json_numbers_are_ignored():
    # B-05: JSON Infinity / 1e400 (both parse to float inf) must not crash; they mean nothing useful
    f = Filters.model_validate_json('{"price_max": Infinity, "seats_min": 1e400, "ncap_min": NaN, "mileage_min": 1e400}')
    assert f.price_max is None and f.seats_min is None and f.ncap_min is None and f.mileage_min is None


def test_plan_with_wrapped_value_degrades_instead_of_raising():
    # B-05: the planner occasionally wraps values in a list; the plan should still validate
    p = QueryPlan.model_validate_json('{"intent": "recommend", "filters": {"price_max": [1500000], "seats_min": [7]}}')
    assert p.intent == "recommend" and p.filters.price_max is None


# ---------------------------------------------------------------- _int / _float
@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ([], None), ({}, None), (7, 7), ("7", 7), (7.9, 7), ("1,000", 1000), ("seven", None),
    (math.nan, None),
])
def test_int_parsing(value, expected):
    assert _int(value) == expected


def test_int_infinity_is_none():
    assert _int(math.inf) is None  # B-05 (was OverflowError)


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("12.5", 12.5), ("1,234.5", 1234.5), ("abc", None), ([1], None), (3, 3.0),
])
def test_float_parsing(value, expected):
    assert _float(value) == expected


# ---------------------------------------------------------------- enum normalisers
@pytest.mark.parametrize("mapping,value,expected", [
    (FUEL_MAP, "EV", "Electric"), (FUEL_MAP, "electric", "Electric"), (FUEL_MAP, "Strong Hybrid", "Hybrid"),
    (FUEL_MAP, "  Diesel ", "Diesel"), (FUEL_MAP, "rocket", None), (FUEL_MAP, None, None), (FUEL_MAP, "", None),
    (FUEL_MAP, 123, None),
    (TRANS_MAP, "AMT", "Automatic"), (TRANS_MAP, "e-CVT", "Automatic"), (TRANS_MAP, "torque_converter", "Automatic"),
    (TRANS_MAP, "MT", "Manual"),
    (BODY_MAP, "mpv", "MUV"), (BODY_MAP, "MUV/MPV", "MUV"), (BODY_MAP, "Compact SUV", "SUV"), (BODY_MAP, "crossover", "SUV"),
    (BODY_MAP, "hatch", "Hatchback"), (BODY_MAP, "Pickup Truck", "Pickup"), (BODY_MAP, "estate", "Wagon"),
    (BODY_MAP, "minivan", "Van"), (BODY_MAP, "spaceship", None),
    (HYB_MAP, "PHEV", "plug-in"), (HYB_MAP, "plug_in", "plug-in"), (HYB_MAP, "self-charging", "strong"),
    (HYB_MAP, "full", "strong"),
    (SORT_MAP, "cheapest", "price_asc"), (SORT_MAP, "fuel_efficiency", "mileage"), (SORT_MAP, "boot", "boot_space"),
    (SORT_MAP, "PRICE_DESC", "price_desc"),
])
def test_enum_normalisation(mapping, value, expected):
    assert _enum(mapping)(value) == expected


def test_enum_default_is_used_for_unknown_values():
    assert _enum(SORT_MAP, "relevance")("random") == "relevance"
    assert _enum(SORT_MAP, "relevance")(None) == "relevance"


@pytest.mark.parametrize("value,expected", [
    (["EV", "electric", "Petrol", "rocket"], ["Electric", "Petrol"]), ("Diesel", ["Diesel"]), (None, []), ([], []),
])
def test_enum_list_dedupes_and_drops_unknown(value, expected):
    assert _enum_list(FUEL_MAP)(value) == expected


@pytest.mark.parametrize("value,expected", [
    (None, []), ("x", ["x"]), ([" a ", "", "b"], ["a", "b"]), ([1, 2], ["1", "2"]),
])
def test_str_list(value, expected):
    assert _str_list(value) == expected


def test_str_list_drops_nulls():
    # B-06: [null] used to become the literal string "None" (it then leaked into keywords / aspects / brands)
    assert _str_list(["mileage", None]) == ["mileage"]
    assert QueryPlan.model_validate({"aspects": [None, "safety"], "keywords": [None]}).aspects == ["safety"]


def test_tags_whitelisted_and_normalised():
    assert _tags(["Sunroof", "360 camera", "ADAS", "panoramic-sunroof", "wings", "6 airbags"]) == \
        ["sunroof", "360_camera", "adas", "panoramic_sunroof", "6_airbags"]
    assert _tags(None) == [] and _tags("sunroof") == ["sunroof"]
    assert all(t in FEATURE_TAGS for t in _tags(FEATURE_TAGS + ["x", "y"]))


def test_cars_accepts_strings_and_dicts_and_drops_junk():
    out = _cars(["Creta", {"name": "Seltos"}, {"model": "Nexon"}, {}, "", "  ", 5, None])
    assert [c["mention"] for c in out] == ["Creta", "Seltos", "Nexon"]
    assert _cars(None) == [] and _cars("Creta") == [{"mention": "Creta"}]


def test_car_ref_hints_are_normalised():
    c = CarRef.model_validate({"mention": "Creta", "fuel_hint": "diesel", "transmission_hint": "DCT"})
    assert c.fuel_hint == "Diesel" and c.transmission_hint == "Automatic" and c.variant_hint is None
    assert CarRef.model_validate({"mention": "Creta", "fuel_hint": "jet fuel"}).fuel_hint is None


# ---------------------------------------------------------------- QueryPlan as a whole
def test_empty_plan_gets_safe_defaults():
    p = QueryPlan.model_validate({})
    assert (p.intent, p.refers_to, p.filter_mode, p.sort_by, p.standalone_query) == ("general", "none", "none", "relevance", "")
    assert p.cars == [] and p.active_indices == [] and p.filters == Filters() and p.city is None and p.clarify is None


def test_all_null_plan_gets_safe_defaults():
    keys = ["standalone_query", "intent", "cars", "refers_to", "active_indices", "filters", "filter_mode", "prefer_tags",
            "sort_by", "aspects", "keywords", "city", "clarify"]
    p = QueryPlan.model_validate(dict.fromkeys(keys))
    assert p.intent == "general" and p.filters.status == "launched" and p.standalone_query == ""


@pytest.mark.parametrize("raw,expected", [
    ("comparison", "compare"), ("Specs", "spec"), ("greeting", "chitchat"), ("off_topic", "out_of_scope"),
    ("Recommendation", "recommend"), ("nonsense", "general"),
])
def test_intent_synonyms(raw, expected):
    assert QueryPlan.model_validate({"intent": raw}).intent == expected


def test_refers_to_filter_mode_and_indices():
    p = QueryPlan.model_validate({"refers_to": "INDICES", "active_indices": ["2", "x", 3.0], "filter_mode": "REFINE"})
    assert p.refers_to == "indices" and 2 in p.active_indices and 3 in p.active_indices and p.filter_mode == "refine"
    assert QueryPlan.model_validate({"active_indices": 2}).active_indices == [2]
    assert QueryPlan.model_validate({"refers_to": "something"}).refers_to == "none"


def test_status_filter_defaults_to_launched():
    assert Filters.model_validate({"status": "UPCOMING"}).status == "upcoming"
    assert Filters.model_validate({"status": "launching soon"}).status == "launched"


def test_extra_keys_ignored_and_unicode_preserved():
    p = QueryPlan.model_validate({"standalone_query": "क्रेटा का माइलेज 🚗", "unexpected": 1})
    assert p.standalone_query == "क्रेटा का माइलेज 🚗"


def test_wrong_scalar_types_raise_validation_error_not_crash():
    # these are rejected (a ValidationError lets the client ask the model to repair its JSON)
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"city": ["Delhi"]})


def test_plan_example_is_a_valid_plan():
    p = QueryPlan.model_validate_json(PLAN_EXAMPLE)
    assert p.intent == "recommend" and p.filters.price_max == 2_000_000 and p.filters.fuel_types == ["Hybrid"]
    assert p.filters.hybrid_types == ["strong", "mild"] and p.filters.body_types == ["SUV"] and p.sort_by == "mileage"
    assert set(json.loads(PLAN_EXAMPLE)) == set(QueryPlan.model_fields)  # example covers every key


def test_strict_schema_has_no_titles_or_defaults_and_does_not_mutate_model():
    def walk(node):
        if isinstance(node, dict):
            assert "title" not in node and "default" not in node
            if "properties" in node:
                assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(strict_json_schema(QueryPlan))
    walk(PLAN_SCHEMA)
    assert "title" in QueryPlan.model_json_schema()  # deep copy: the model's own schema is untouched


def test_helpers_return_fresh_objects():
    f = empty_filters()
    f.brands.append("x")
    assert empty_filters().brands == []
    p = simple_plan("hi", "chitchat")
    assert p.standalone_query == "hi" and p.intent == "chitchat"
