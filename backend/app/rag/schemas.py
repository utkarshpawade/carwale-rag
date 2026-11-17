"""QueryPlan: the validated JSON the LLM planner must return. The LLM never writes SQL - code executes this plan.

The model is deliberately *tolerant*: free-tier models omit keys, emit null for empty lists, vary enum casing
("EV", "mpv", "AMT") or give budgets in lakh ("15"). Before-validators normalise all of that, so a slightly
imperfect plan is repaired locally instead of failing (and costing a fallback call).
"""
from __future__ import annotations

import copy
import re
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field

Fuel = Literal["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
Body = Literal["SUV", "MUV", "Hatchback", "Sedan", "Coupe", "Convertible", "Pickup", "Van", "Wagon"]
Trans = Literal["Manual", "Automatic"]
Hyb = Literal["mild", "strong", "plug-in"]
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

FUEL_MAP = {"petrol": "Petrol", "gasoline": "Petrol", "diesel": "Diesel", "cng": "CNG", "electric": "Electric",
            "ev": "Electric", "bev": "Electric", "hybrid": "Hybrid", "strong hybrid": "Hybrid", "mild hybrid": "Hybrid"}
HYB_MAP = {"mild": "mild", "strong": "strong", "plug-in": "plug-in", "plugin": "plug-in", "plug in": "plug-in",
           "phev": "plug-in", "self-charging": "strong", "full": "strong"}
BODY_MAP = {**{b.lower(): b for b in ("SUV", "MUV", "Hatchback", "Sedan", "Coupe", "Convertible", "Pickup", "Van", "Wagon")},
            "mpv": "MUV", "muv/mpv": "MUV", "compact suv": "SUV", "sub-compact suv": "SUV", "crossover": "SUV",
            "hatch": "Hatchback", "pickup truck": "Pickup", "estate": "Wagon", "minivan": "Van"}
TRANS_MAP = {"manual": "Manual", "mt": "Manual", "automatic": "Automatic", "auto": "Automatic", "amt": "Automatic",
             "cvt": "Automatic", "dct": "Automatic", "at": "Automatic", "e-cvt": "Automatic", "torque converter": "Automatic"}
INTENT_MAP = {**{i: i for i in Intent.__args__}, "recommendation": "recommend", "search": "recommend",
              "comparison": "compare", "specs": "spec", "specification": "spec", "mileage": "spec",
              "features": "feature_check", "feature": "feature_check", "variant": "variant_advice", "pricing": "price",
              "reviews": "review", "maintenance": "service", "greeting": "chitchat", "off_topic": "out_of_scope"}
SORT_MAP = {**{s: s for s in SortBy.__args__}, "cheapest": "price_asc", "price": "price_asc", "fuel_efficiency": "mileage",
            "efficiency": "mileage", "performance": "power", "safest": "safety", "boot": "boot_space"}


def _key(v) -> str:
    return str(v).strip().lower().replace("_", " ") if v is not None else ""


def _enum(mapping: dict, default=None):
    def f(v):
        if v is None or v == "":
            return default
        k = _key(v)
        return mapping.get(k, mapping.get(k.replace(" ", "_"), mapping.get(str(v).strip(), default)))
    return f


def _enum_list(mapping: dict):
    def f(v):
        if v is None:
            return []
        out = []
        for x in (v if isinstance(v, list) else [v]):
            c = _enum(mapping)(x)
            if c and c not in out:
                out.append(c)
        return out
    return f


def _str_list(v):
    if v is None:
        return []
    return [str(x).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]


def _tags(v):
    return [t for t in (x.lower().replace(" ", "_").replace("-", "_") for x in _str_list(v)) if t in FEATURE_TAGS]


def _int(v):
    if v in (None, "", [], {}):
        return None
    try:
        return int(float(str(v).replace(",", "")))
    except ValueError:
        return None


def _float(v):
    if v in (None, ""):
        return None
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def _price(v):
    """INR. Accepts 1500000, '15 lakh', '1.2 crore', or a bare 15 / '15' (read as lakh)."""
    if v in (None, "", 0):
        return None
    if isinstance(v, str):
        m = re.search(r"([\d.]+)\s*(crore|cr|lakh|lac|l)?", v.lower().replace(",", ""))
        if not m:
            return None
        n = float(m.group(1))
        unit = m.group(2) or ""
        v = n * (1e7 if unit.startswith("c") else 1e5 if unit else 1)
    v = float(v)
    if 0 < v < 1000:  # "under 15" almost always means 15 lakh
        v *= 1e5
    return int(v) if v > 0 else None


def _cars(v):
    if v is None:
        return []
    out = []
    for x in (v if isinstance(v, list) else [v]):
        if isinstance(x, str) and x.strip():
            out.append({"mention": x.strip()})
        elif isinstance(x, dict) and (x.get("mention") or x.get("name") or x.get("model")):
            out.append({**x, "mention": x.get("mention") or x.get("name") or x.get("model")})
    return out


class CarRef(BaseModel):
    mention: str
    variant_hint: Optional[str] = None
    fuel_hint: Annotated[Optional[Fuel], BeforeValidator(_enum(FUEL_MAP))] = None
    transmission_hint: Annotated[Optional[Trans], BeforeValidator(_enum(TRANS_MAP))] = None


class Filters(BaseModel):
    price_min: Annotated[Optional[int], BeforeValidator(_price)] = None
    price_max: Annotated[Optional[int], BeforeValidator(_price)] = None
    fuel_types: Annotated[list[Fuel], BeforeValidator(_enum_list(FUEL_MAP))] = Field(default_factory=list)
    hybrid_types: Annotated[list[Hyb], BeforeValidator(_enum_list(HYB_MAP))] = Field(default_factory=list)
    body_types: Annotated[list[Body], BeforeValidator(_enum_list(BODY_MAP))] = Field(default_factory=list)
    transmission: Annotated[Optional[Trans], BeforeValidator(_enum(TRANS_MAP))] = None
    seats_min: Annotated[Optional[int], BeforeValidator(_int)] = None
    brands: Annotated[list[str], BeforeValidator(_str_list)] = Field(default_factory=list)
    feature_tags: Annotated[list[str], BeforeValidator(_tags)] = Field(default_factory=list)
    ncap_min: Annotated[Optional[int], BeforeValidator(_int)] = None
    mileage_min: Annotated[Optional[float], BeforeValidator(_float)] = None
    status: Annotated[Literal["launched", "upcoming"], BeforeValidator(_enum({"launched": "launched", "upcoming": "upcoming"}, "launched"))] = "launched"


class QueryPlan(BaseModel):
    standalone_query: Annotated[str, BeforeValidator(lambda v: "" if v is None else str(v))] = ""
    intent: Annotated[Intent, BeforeValidator(_enum(INTENT_MAP, "general"))] = "general"
    cars: Annotated[list[CarRef], BeforeValidator(_cars)] = Field(default_factory=list)
    refers_to: Annotated[Literal["none", "focus", "all", "indices"],
                         BeforeValidator(_enum({"none": "none", "focus": "focus", "all": "all", "indices": "indices"}, "none"))] = "none"
    active_indices: Annotated[list[int], BeforeValidator(lambda v: [i for i in map(_int, v if isinstance(v, list) else [v]) if i] if v else [])] = Field(default_factory=list)
    filters: Annotated[Filters, BeforeValidator(lambda v: v or {})] = Field(default_factory=Filters)
    filter_mode: Annotated[Literal["new", "refine", "none"],
                           BeforeValidator(_enum({"new": "new", "refine": "refine", "none": "none"}, "none"))] = "none"
    prefer_tags: Annotated[list[str], BeforeValidator(_tags)] = Field(default_factory=list)
    sort_by: Annotated[SortBy, BeforeValidator(_enum(SORT_MAP, "relevance"))] = "relevance"
    aspects: Annotated[list[str], BeforeValidator(_str_list)] = Field(default_factory=list)
    keywords: Annotated[list[str], BeforeValidator(_str_list)] = Field(default_factory=list)
    city: Optional[str] = None
    clarify: Optional[str] = None


def empty_filters() -> Filters:
    return Filters()


def simple_plan(message: str, intent: str) -> QueryPlan:
    return QueryPlan(standalone_query=message, intent=intent)


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

# One complete example output; used in JSON-object mode so every model sees the exact shape to produce.
PLAN_EXAMPLE = (
    '{"standalone_query":"Which hybrid SUVs under 20 lakh have the best mileage?","intent":"recommend","cars":[],'
    '"refers_to":"none","active_indices":[],"filters":{"price_min":null,"price_max":2000000,"fuel_types":["Hybrid"],'
    '"hybrid_types":["strong","mild"],"body_types":["SUV"],"transmission":null,"seats_min":null,"brands":[],'
    '"feature_tags":[],"ncap_min":null,"mileage_min":null,"status":"launched"},"filter_mode":"new","prefer_tags":[],'
    '"sort_by":"mileage","aspects":["mileage","price"],"keywords":["hybrid","SUV","mileage"],"city":null,"clarify":null}'
)
