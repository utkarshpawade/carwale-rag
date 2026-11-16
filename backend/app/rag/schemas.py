"""QueryPlan: the validated JSON the LLM planner must return. The LLM never writes SQL - code executes this plan."""
from __future__ import annotations

import copy
from typing import Literal

from pydantic import BaseModel

Fuel = Literal["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
Body = Literal["SUV", "MUV", "Hatchback", "Sedan", "Coupe", "Convertible", "Pickup", "Van", "Wagon"]
Intent = Literal["recommend", "spec", "price", "feature_check", "compare", "safety", "service", "review",
                 "variant_advice", "general", "chitchat", "out_of_scope"]
SortBy = Literal["relevance", "price_asc", "price_desc", "mileage", "range", "power", "safety", "boot_space", "rating"]

FEATURE_TAGS = [
    "sunroof", "panoramic_sunroof", "ventilated_seats", "wireless_charger", "360_camera", "reverse_camera",
    "cruise_control", "adaptive_cruise", "adas", "blind_spot_monitor", "connected_car", "ota_updates",
    "apple_carplay_android_auto", "wireless_carplay", "auto_climate", "dual_zone_climate", "hud",
    "tpms", "esc", "hill_hold", "6_airbags", "4x4", "ambient_lighting", "leather_seats", "powered_seats",
    "led_headlights", "keyless_start", "rear_ac_vents", "turbo", "dashcam", "front_parking_sensors",
    "touchscreen", "aeb", "lane_keep_assist", "voice_commands", "electric_parking_brake",
]


class CarRef(BaseModel):
    mention: str
    variant_hint: str | None
    fuel_hint: Fuel | None
    transmission_hint: Literal["Manual", "Automatic"] | None


class Filters(BaseModel):
    price_min: int | None
    price_max: int | None
    fuel_types: list[Fuel]
    hybrid_types: list[Literal["mild", "strong", "plug-in"]]
    body_types: list[Body]
    transmission: Literal["Manual", "Automatic"] | None
    seats_min: int | None
    brands: list[str]
    feature_tags: list[str]
    ncap_min: int | None
    mileage_min: float | None
    status: Literal["launched", "upcoming"]


class QueryPlan(BaseModel):
    standalone_query: str
    intent: Intent
    cars: list[CarRef]
    refers_to: Literal["none", "focus", "all", "indices"]
    active_indices: list[int]
    filters: Filters
    filter_mode: Literal["new", "refine", "none"]
    prefer_tags: list[str]
    sort_by: SortBy
    aspects: list[str]
    keywords: list[str]
    city: str | None
    clarify: str | None


def empty_filters() -> Filters:
    return Filters(price_min=None, price_max=None, fuel_types=[], hybrid_types=[], body_types=[], transmission=None,
                   seats_min=None, brands=[], feature_tags=[], ncap_min=None, mileage_min=None, status="launched")


def simple_plan(message: str, intent: str) -> QueryPlan:
    return QueryPlan(standalone_query=message, intent=intent, cars=[], refers_to="none", active_indices=[],
                     filters=empty_filters(), filter_mode="none", prefer_tags=[], sort_by="relevance", aspects=[],
                     keywords=[], city=None, clarify=None)


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema -> Groq strict-mode schema: every property required, additionalProperties false, no titles."""
    schema = copy.deepcopy(model.model_json_schema())

    def walk(node):
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


PLAN_SCHEMA = strict_json_schema(QueryPlan)
