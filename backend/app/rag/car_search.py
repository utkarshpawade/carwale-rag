"""Structured car search: validated plan filters -> whitelisted, parameterized SQL (never LLM-written SQL)."""
from __future__ import annotations

from ..db import fetch
from .schemas import FEATURE_TAGS, Filters

# tags only known at model level (from the top variant's full feature tree)
MODEL_ONLY_TAGS = {"connected_car", "ota_updates", "aeb", "lane_keep_assist", "electric_parking_brake"}

SORTS = {
    "relevance": "score DESC, m.user_rating_count DESC NULLS LAST",
    "price_asc": "from_price ASC",
    "price_desc": "from_price DESC",
    "mileage": "best_mileage DESC NULLS LAST, from_price ASC",
    "range": "best_range DESC NULLS LAST, from_price ASC",
    "power": "max_power DESC NULLS LAST",
    "safety": "COALESCE(m.ncap_rating, 0) DESC, max_airbags DESC NULLS LAST, score DESC",
    "boot_space": "boot_l DESC NULLS LAST",
    "rating": "CASE WHEN m.user_rating_count >= 5 THEN m.user_rating END DESC NULLS LAST, score DESC",
}

SCORE = ("(SELECT COUNT(*) FROM unnest(m.feature_tags) t WHERE t = ANY(%(prefer)s)) * 2.0"
         " + LN(1 + COALESCE(m.user_rating_count, 0)) * 0.6 + COALESCE(m.user_rating, 3.5) * 0.5"
         # for 'hybrid' searches, true (strong/plug-in) hybrids outrank mild hybrids
         " + CASE WHEN BOOL_OR(v.hybrid_type IN ('strong', 'plug-in')) THEN %(hyboost)s::float ELSE 0 END")


def _where(f: Filters, brand_slugs: list[str], restrict_ids: list[str] | None) -> tuple[list[str], dict, list[str]]:
    where = ["v.price IS NOT NULL", "m.status = 'launched'"]
    p: dict = {}
    desc: list[str] = []
    if f.price_max:
        where.append("v.price <= %(pmax)s"); p["pmax"] = f.price_max; desc.append(f"price <= {f.price_max / 1e5:.2f} L")
    if f.price_min:
        where.append("v.price >= %(pmin)s"); p["pmin"] = f.price_min; desc.append(f"price >= {f.price_min / 1e5:.2f} L")
    fuels, htypes = set(f.fuel_types), set(f.hybrid_types)
    if "Hybrid" in fuels:
        fuels.discard("Hybrid")
        htypes = htypes or {"strong", "mild", "plug-in"}
    conds = []
    if fuels:
        conds.append("v.fuel_type = ANY(%(fuels)s)"); p["fuels"] = sorted(fuels)
    if htypes:
        conds.append("v.hybrid_type = ANY(%(htypes)s)"); p["htypes"] = sorted(htypes)
    if conds:
        where.append("(" + " OR ".join(conds) + ")")
        desc.append("fuel: " + ", ".join(sorted(fuels) + [f"{h} hybrid" for h in sorted(htypes)]))
    if f.body_types:
        where.append("m.body_type = ANY(%(bodies)s)"); p["bodies"] = list(f.body_types); desc.append("body: " + "/".join(f.body_types))
    if f.transmission == "Automatic":
        where.append("v.is_automatic IS TRUE"); desc.append("automatic")
    elif f.transmission == "Manual":
        where.append("v.is_automatic IS FALSE"); desc.append("manual")
    if f.seats_min:
        where.append("v.seats >= %(seats)s"); p["seats"] = f.seats_min; desc.append(f"seats >= {f.seats_min}")
    if brand_slugs:
        where.append("m.brand_slug = ANY(%(brands)s)"); p["brands"] = brand_slugs; desc.append("brand: " + ", ".join(brand_slugs))
    tags = [t for t in f.feature_tags if t in FEATURE_TAGS]
    vt, mt = [t for t in tags if t not in MODEL_ONLY_TAGS], [t for t in tags if t in MODEL_ONLY_TAGS]
    if vt:
        where.append("v.feature_tags @> %(vtags)s"); p["vtags"] = vt
    if mt:
        where.append("m.feature_tags @> %(mtags)s"); p["mtags"] = mt
    if tags:
        desc.append("must have: " + ", ".join(tags))
    if f.ncap_min:
        where.append("COALESCE(v.ncap_rating, m.ncap_rating, 0) >= %(ncap)s"); p["ncap"] = f.ncap_min; desc.append(f"NCAP >= {f.ncap_min}")
    if f.mileage_min:
        where.append("v.mileage_effective >= %(mmin)s"); p["mmin"] = f.mileage_min; desc.append(f"mileage >= {f.mileage_min}")
    if restrict_ids:
        where.append("m.id = ANY(%(ids)s)"); p["ids"] = restrict_ids; desc.append(f"within {len(restrict_ids)} cars from the conversation")
    return where, p, desc


async def _run(f: Filters, sort_by: str, prefer: list[str], brand_slugs: list[str], restrict_ids, limit: int):
    where, p, desc = _where(f, brand_slugs, restrict_ids)
    hybrid_search = bool(f.hybrid_types) or "Hybrid" in f.fuel_types
    p.update({"prefer": [t for t in prefer if t in FEATURE_TAGS], "limit": limit, "hyboost": 2.0 if hybrid_search else 0.0})
    sql = f"""
    SELECT m.id, m.full_name, m.brand, m.body_type_raw, m.url, m.image_url, m.user_rating, m.user_rating_count,
           m.ncap_rating, m.ncap_agency, m.status, m.feature_tags, m.price_min, m.price_max,
           MIN(v.price) AS from_price, MAX(v.price) AS to_price, COUNT(*) AS n_variants,
           MAX(v.mileage_effective) FILTER (WHERE v.fuel_type <> 'Electric') AS best_mileage,
           MAX(v.range_km) AS best_range, MAX(v.power_bhp) AS max_power, MAX(v.airbags) AS max_airbags,
           MAX(v.boot_l) AS boot_l,
           ARRAY_AGG(DISTINCT v.fuel_type) AS fuels,
           ARRAY_AGG(DISTINCT v.transmission) AS transmissions,
           ARRAY_REMOVE(ARRAY_AGG(DISTINCT v.hybrid_type), NULL) AS hybrid_types,
           ({SCORE}) AS score
    FROM variants v JOIN models m ON m.id = v.model_id
    WHERE {' AND '.join(where)}
    GROUP BY m.id
    ORDER BY {SORTS.get(sort_by, SORTS['relevance'])}
    LIMIT %(limit)s"""
    rows = await fetch(sql, p)
    if rows:  # cheapest / most efficient matching variants per model
        p2 = dict(p)
        p2["top_ids"] = [r["id"] for r in rows]
        vrows = await fetch(f"""
            SELECT v.model_id, v.name, v.price, v.fuel_type, v.hybrid_type, v.transmission, v.mileage_effective,
                   v.mileage_basis, v.mileage_unit, v.range_km
            FROM variants v JOIN models m ON m.id = v.model_id
            WHERE {' AND '.join(where)} AND m.id = ANY(%(top_ids)s)
            ORDER BY v.price""", p2)
        by = {}
        for v in vrows:
            by.setdefault(v["model_id"], []).append(v)
        for r in rows:
            vs = by.get(r["id"], [])
            r["matching_variants"] = vs[:3]
            eff = [v for v in vs if v.get("mileage_effective") and v.get("fuel_type") != "Electric"]
            if eff:
                b = max(eff, key=lambda v: v["mileage_effective"])
                r["best_mileage_basis"], r["mileage_unit"] = b["mileage_basis"], b["mileage_unit"]
    return rows, desc


async def search_cars(f: Filters, sort_by: str, prefer: list[str], brand_slugs: list[str],
                      restrict_ids: list[str] | None = None, limit: int = 6) -> tuple[list[dict], list[str], list[str]]:
    """-> (rows, human-readable filter description, relaxation notes)."""
    rows, desc = await _run(f, sort_by, prefer, brand_slugs, restrict_ids, limit)
    notes: list[str] = []
    htypes = set(f.hybrid_types) | ({"strong", "mild", "plug-in"} if "Hybrid" in f.fuel_types and not f.hybrid_types else set())
    if rows and not restrict_ids and {"mild", "strong"} <= htypes and len(rows) >= limit - 1:
        # diversity: keep two mild-hybrid slots so the answer can contrast mild vs strong hybrids
        if sum(1 for r in rows if not ({"strong", "plug-in"} & set(r.get("hybrid_types") or []))) < 2:
            mild = f.model_copy(update={"hybrid_types": ["mild"], "fuel_types": []})
            extra, _ = await _run(mild, sort_by, prefer, brand_slugs, None, 4)
            extra = [r for r in extra if r["id"] not in {x["id"] for x in rows}][:2]
            if extra:
                rows = rows[: limit - len(extra)] + extra
    if rows or restrict_ids:
        return rows, desc, notes
    relaxed = f.model_copy(deep=True)
    steps = []
    if relaxed.price_max:
        steps.append(("price_max", f"no exact matches, so the budget was stretched by 15% to {relaxed.price_max * 1.15 / 1e5:.1f} L"))
    if relaxed.feature_tags:
        steps.append(("feature_tags", "required features were treated as preferences"))
    if relaxed.seats_min or relaxed.ncap_min or relaxed.mileage_min:
        steps.append(("minimums", "minimum seats/safety/mileage limits were relaxed"))
    for what, note in steps:
        if what == "price_max":
            relaxed.price_max = int(relaxed.price_max * 1.15)
        elif what == "feature_tags":
            prefer = list(prefer) + list(relaxed.feature_tags)
            relaxed.feature_tags = []
        else:
            relaxed.seats_min = relaxed.ncap_min = relaxed.mileage_min = None
        notes.append(note)
        rows, desc = await _run(relaxed, sort_by, prefer, brand_slugs, restrict_ids, limit)
        if rows:
            break
    return rows, desc, notes


async def search_upcoming(f: Filters, brand_slugs: list[str], limit: int = 8) -> list[dict]:
    where, p = ["status = 'upcoming'"], {"limit": limit}
    if f.price_max:
        where.append("(price_min IS NULL OR price_min <= %(pmax)s)"); p["pmax"] = f.price_max
    if f.body_types:
        where.append("body_type = ANY(%(bodies)s)"); p["bodies"] = list(f.body_types)
    if brand_slugs:
        where.append("brand_slug = ANY(%(brands)s)"); p["brands"] = brand_slugs
    if f.fuel_types:
        # an EV flag only widens an Electric search; "is_ev = false" used to admit every non-EV for any fuel
        where.append("(fuel_types && %(fuels)s OR (%(ev)s AND is_ev IS TRUE))")
        p["fuels"] = list(f.fuel_types)
        p["ev"] = "Electric" in f.fuel_types
    return await fetch(f"""SELECT id, full_name, brand, body_type_raw, url, image_url, price_min, price_max,
                               price_is_estimate, launched_on, fuel_types, status, user_rating, user_rating_count
                           FROM models WHERE {' AND '.join(where)}
                           ORDER BY launched_on NULLS LAST LIMIT %(limit)s""", p)
