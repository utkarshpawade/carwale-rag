"""Structured fact sheets and comparison tables built from the catalog (deterministic, no LLM)."""
from __future__ import annotations

import json
from collections import OrderedDict

from rapidfuzz import fuzz

from ..db import fetch
from .entities import norm

TAG_LABELS = {
    "sunroof": "sunroof", "panoramic_sunroof": "panoramic sunroof", "ventilated_seats": "ventilated seats",
    "wireless_charger": "wireless charger", "360_camera": "360° camera", "adas": "ADAS",
    "adaptive_cruise": "adaptive cruise", "connected_car": "connected-car app", "ota_updates": "OTA updates",
    "wireless_carplay": "wireless CarPlay/AA", "hud": "head-up display", "dual_zone_climate": "dual-zone AC",
    "6_airbags": "6 airbags", "4x4": "4x4/AWD", "ambient_lighting": "ambient lighting", "powered_seats": "powered seats",
    "blind_spot_monitor": "blind-spot monitor", "aeb": "auto emergency braking", "lane_keep_assist": "lane keep assist",
    "tpms": "TPMS", "esc": "ESC", "leather_seats": "leatherette/leather seats", "cruise_control": "cruise control",
}
HEADLINE_TAGS = ["adas", "connected_car", "ota_updates", "360_camera", "panoramic_sunroof", "sunroof", "ventilated_seats",
                 "wireless_carplay", "wireless_charger", "hud", "adaptive_cruise", "6_airbags", "4x4", "powered_seats"]


def inr(v) -> str:
    if not v:
        return "n/a"
    return f"₹{v / 1e7:.2f} Cr" if v >= 1e7 else f"₹{v / 1e5:.2f} L"


def price_range(lo, hi) -> str:
    if not lo:
        return "n/a"
    return inr(lo) if not hi or lo == hi else f"{inr(lo)} – {inr(hi)}"


def num(v, unit="", nd=1) -> str | None:
    if v in (None, "", 0):
        return None
    if isinstance(v, float):
        v = int(v) if v.is_integer() else round(v, nd)
    return f"{v}{(' ' + unit) if unit else ''}"


def feature_list(tags: list[str] | None, limit: int = 8) -> str:
    tags = set(tags or [])
    out = [TAG_LABELS.get(t, t) for t in HEADLINE_TAGS if t in tags]
    if "panoramic sunroof" in out and "sunroof" in out:
        out.remove("sunroof")
    return ", ".join(out[:limit]) or "basic equipment"


async def load_models(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    rows = await fetch("SELECT * FROM models WHERE id = ANY(%s)", (ids,))
    for r in rows:
        if isinstance(r.get("data"), str):
            r["data"] = json.loads(r["data"])
    return {r["id"]: r for r in rows}


async def load_variants(ids: list[str]) -> dict[str, list[dict]]:
    if not ids:
        return {}
    rows = await fetch("SELECT * FROM variants WHERE model_id = ANY(%s) ORDER BY price NULLS LAST", (ids,))
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["model_id"], []).append(r)
    return out


def filter_variants(vs: list[dict], fuel: str | None, trans: str | None, hint: str | None) -> tuple[list[dict], list[str]]:
    notes, out = [], vs
    if fuel:
        sel = [v for v in out if v["fuel_type"] == fuel or (fuel == "Hybrid" and v.get("hybrid_type") in ("strong", "plug-in"))]
        if sel:
            out = sel
        else:
            notes.append(f"no {fuel} variant is listed")
    if trans:
        sel = [v for v in out if bool(v.get("is_automatic")) == (trans == "Automatic")]
        if sel:
            out = sel
        else:
            notes.append(f"no {trans.lower()} variant is listed")
    if hint:
        h = norm(hint)
        sel = [v for v in out if h and (h in norm(v.get("name") or "") or fuzz.ratio(h, norm(v.get("trim") or "")) >= 85)]
        if sel:
            out = sel
        else:
            notes.append(f"variant '{hint}' not found; showing all")
    return out, notes


def powertrains(vs: list[dict]) -> list[dict]:
    groups: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for v in vs:
        key = (v.get("fuel_type"), v.get("hybrid_type"), v.get("engine_cc"), v.get("transmission"))
        groups.setdefault(key, []).append(v)
    out = []
    for (fuel, htype, cc, trans), g in groups.items():
        e = g[0]
        label = f"{htype} hybrid" if htype in ("strong", "plug-in") else fuel
        if htype == "mild":
            label = f"{fuel} mild-hybrid"
        out.append({
            "label": f"{label} {num(cc, 'cc') or ''} {trans or ''}".replace("  ", " ").strip(),
            "price": price_range(min(x["price"] for x in g if x.get("price")) if any(x.get("price") for x in g) else None,
                                 max((x["price"] or 0) for x in g)),
            "power": num(e.get("power_bhp"), "bhp"), "torque": num(e.get("torque_nm"), "Nm"),
            "mileage": num(e.get("mileage_arai"), e.get("mileage_unit") or "kmpl"),
            "mileage_real": num(e.get("mileage_owner"), e.get("mileage_unit") or "kmpl"),
            "tested": (f"city {num(e.get('mileage_city_tested'))} / highway {num(e.get('mileage_highway_tested'))}"
                       if e.get("mileage_city_tested") or e.get("mileage_highway_tested") else None),
            "range": num(e.get("range_km"), "km"), "battery": num(e.get("battery_kwh"), "kWh"),
            "dc": num(e.get("dc_charge_minutes"), "min"),
        })
    return out


def facts(m: dict, vs: list[dict], aspects: set[str], city: str | None = None, compare: bool = False) -> "OrderedDict[str, str]":
    """Ordered label -> value facts for one car, limited to the aspects asked about."""
    d = m.get("data") or {}
    every = compare or not aspects
    f: "OrderedDict[str, str]" = OrderedDict()
    prices = [v["price"] for v in vs if v.get("price")]
    est = " (expected)" if m.get("price_is_estimate") else ""
    f["Price (ex-showroom)"] = (price_range(min(prices), max(prices)) if prices else price_range(m.get("price_min"), m.get("price_max"))) + est
    if m.get("status") != "launched":
        f["Status"] = f"{m.get('status')}" + (f", expected {m['launched_on']}" if m.get("launched_on") else "")
    f["Body type"] = m.get("body_type_raw") or m.get("body_type") or "n/a"
    pts = powertrains(vs)
    if pts:
        f["Powertrains"] = "; ".join(p["label"] for p in pts[:5])
    if every or aspects & {"performance", "specs"}:
        f["Power / torque"] = "; ".join(f"{p['label']}: {p['power'] or 'n/a'}, {p['torque'] or 'n/a'}" for p in pts[:4]) or "n/a"
    if every or aspects & {"mileage"}:
        if any(p["range"] for p in pts):
            f["EV range (claimed)"] = "; ".join(f"{p['label']}: {p['range']} ({p['battery'] or '?'})" for p in pts if p["range"])
        mil = [f"{p['label']}: {p['mileage']}" for p in pts if p["mileage"]]
        f["Mileage (ARAI claimed)"] = "; ".join(mil[:4]) if mil else (d.get("mileage_claimed_text") or "not listed")
        real = [f"{p['label']}: {p['mileage_real']}" for p in pts if p["mileage_real"]]
        if real:
            f["Mileage (owner-reported)"] = "; ".join(real[:4])
        tested = [f"{p['label']}: {p['tested']}" for p in pts if p["tested"]]
        if tested:
            f["Mileage (CarWale tested)"] = "; ".join(tested[:2])
    if every or aspects & {"specs", "dimensions", "comfort"}:
        seats = sorted({v["seats"] for v in vs if v.get("seats")})
        f["Seating"] = ", ".join(map(str, seats)) or "n/a"
        top = max(vs, key=lambda v: v.get("price") or 0) if vs else {}
        dims = [top.get("length_mm"), top.get("width_mm"), top.get("height_mm")]
        if all(dims):
            f["Dimensions (L×W×H)"] = "×".join(str(x) for x in dims) + f" mm, wheelbase {top.get('wheelbase_mm') or 'n/a'} mm"
        f["Boot space"] = num(m.get("boot_l"), "litres") or "not listed"
        f["Ground clearance"] = num(m.get("ground_clearance_mm"), "mm") or "not listed"
        if top.get("fuel_tank_l"):
            f["Fuel tank"] = num(top.get("fuel_tank_l"), "litres")
    if every or aspects & {"safety"}:
        f["Crash rating"] = f"{m['ncap_rating']}-star ({m.get('ncap_agency') or 'NCAP'})" if m.get("ncap_rating") else "not crash-tested / not listed"
        ab = sorted({v["airbags"] for v in vs if v.get("airbags")})
        f["Airbags"] = ", ".join(map(str, ab)) or "n/a"
        tags = set(m.get("feature_tags") or [])
        f["Safety tech"] = ", ".join(TAG_LABELS[t] for t in ("adas", "aeb", "lane_keep_assist", "blind_spot_monitor", "360_camera", "esc", "tpms") if t in tags) or "basic"
    if every or aspects & {"features", "comfort"}:
        f["Key features (top variant)"] = feature_list(m.get("feature_tags"), 10)
    if every or aspects & {"price", "on_road_price", "variants"}:
        cps = d.get("city_prices") or []
        if city:
            hit = next((c for c in cps if norm(c.get("city") or "") == norm(city)), None)
            f[f"On-road price ({city})"] = f"Rs. {hit['range']}" if hit else f"not listed for {city} (listed cities: {', '.join(c['city'] for c in cps[:8])})"
        elif aspects & {"on_road_price"} and cps:
            f["On-road prices"] = "; ".join(f"{c['city']} Rs. {c['range']}" for c in cps[:5])
        f["Variants"] = f"{len(vs)} versions" + (f" from {vs[0]['name']} to {vs[-1]['name']}" if vs else "")
    if every or aspects & {"service"}:
        f["Warranty"] = "; ".join(d.get("warranty") or []) or "not listed"
    if every or aspects & {"reviews"}:
        if m.get("user_rating"):
            f["Owner rating"] = f"{m['user_rating']:.1f}/5 ({m.get('user_rating_count')} ratings)"
        if m.get("expert_rating_5"):
            f["CarWale expert rating"] = f"{m['expert_rating_5']}/5"
        if d.get("pros"):
            f["Pros"] = "; ".join(d["pros"][:3])
        if d.get("cons"):
            f["Cons"] = "; ".join(d["cons"][:3])
    if aspects & {"colours"} and d.get("colours"):
        f["Colours"] = ", ".join(d["colours"][:12])
    return f


def variants_table(vs: list[dict], limit: int = 14) -> str:
    lines = ["| Variant | Price | Fuel | Transmission | Mileage/Range | Notable features |", "|---|---|---|---|---|---|"]
    for v in vs[:limit]:
        econ = num(v.get("mileage_arai"), v.get("mileage_unit") or "kmpl") or num(v.get("range_km"), "km") or \
            (f"{num(v.get('mileage_effective'), v.get('mileage_unit') or 'kmpl')} ({v.get('mileage_basis')})" if v.get("mileage_effective") else "n/a")
        lines.append(f"| {v['name']} | {inr(v.get('price'))} | {v.get('fuel_type')} | {v.get('transmission')} | {econ} | {feature_list(v.get('feature_tags'), 5)} |")
    if len(vs) > limit:
        lines.append(f"| … {len(vs) - limit} more variants | | | | | |")
    return "\n".join(lines)


def facts_block(name: str, url: str, fs: "OrderedDict[str, str]") -> str:
    return f"#### {name} ({url})\n" + "\n".join(f"- {k}: {v}" for k, v in fs.items())


def compare_table(names: list[str], all_facts: list["OrderedDict[str, str]"]) -> str:
    labels = list(OrderedDict.fromkeys(k for fs in all_facts for k in fs))
    head = "| Attribute | " + " | ".join(names) + " |\n|---|" + "---|" * len(names)
    rows = [f"| {lab} | " + " | ".join(fs.get(lab, "n/a").replace("|", "/") for fs in all_facts) + " |" for lab in labels]
    return head + "\n" + "\n".join(rows)


def search_table(rows: list[dict]) -> str:
    lines = ["| # | Car | Matching price (ex-showroom) | Type | Best mileage/range | Key features | Safety | Owner rating |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        types = ", ".join(sorted(set(
            [f"{h} hybrid" for h in (r.get("hybrid_types") or [])] +
            [f for f in (r.get("fuels") or []) if f != "Hybrid"])))
        if r.get("best_range"):
            econ = f"{num(r['best_range'], 'km')} range"
        elif r.get("best_mileage"):
            econ = f"{num(r['best_mileage'], r.get('mileage_unit') or 'kmpl')} ({r.get('best_mileage_basis') or 'claimed'})"
        else:
            econ = "not listed"
        safety = f"{r['ncap_rating']}★ {r.get('ncap_agency') or ''}".strip() if r.get("ncap_rating") else f"{r.get('max_airbags') or '?'} airbags"
        rating = f"{r['user_rating']:.1f}/5" if r.get("user_rating") else "n/a"
        mv = r.get("matching_variants") or []
        cheapest = f" (from {mv[0]['name']})" if mv else ""
        lines.append(f"| {i} | {r['full_name']} | {price_range(r.get('from_price'), r.get('to_price'))}{cheapest} | {types} | {econ} | "
                     f"{feature_list(r.get('feature_tags'), 5)} | {safety} | {rating} |")
    return "\n".join(lines)


def card(m: dict) -> dict:
    """Payload for a frontend car card."""
    return {
        "id": m["id"], "name": m.get("full_name"), "url": m.get("url"), "image": m.get("image_url"),
        "price": price_range(m.get("from_price") or m.get("price_min"), m.get("to_price") or m.get("price_max")),
        "body": m.get("body_type_raw"), "rating": m.get("user_rating"), "status": m.get("status"),
        "ncap": m.get("ncap_rating"),
    }
