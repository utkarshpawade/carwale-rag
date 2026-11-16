"""Parse cached CarWale model + top-trim pages into normalised model and variant records.

Usage: python -m ingestion.parse
Reads  data/urls_models.json, data/raw/*.html.gz
Writes data/processed/models.jsonl, data/processed/variants.jsonl and prints a quality report.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter

from .common import DATA, IMG_CDN, PROCESSED, clean_text, extract_state, list_items, read_cached, write_jsonl
from .normalize import (
    canon_body, canon_fuel, canon_transmission, hybrid_type, lakh_text_to_inr, num, parse_date,
    parse_ncap, positive, to_int, tree_tags, variant_tags,
)

TODAY = time.strftime("%Y-%m-%d")
BODY_LD_MAP = {"muvsmpvs": "MUV/MPV", "suvs": "SUV", "hatchbacks": "Hatchback", "sedans": "Sedan",
               "coupes": "Coupe", "convertibles": "Convertible", "pickuptrucks": "Pickup Truck",
               "minivans": "Minivan", "wagons": "Wagon"}


# --------------------------------------------------------------------------- spec helpers
def spec_items_to_dict(items: list[dict] | None) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """[{itemName, value, unitType}] -> ({name: 'value unit'}, {name: (raw_value, unit)})."""
    disp, raw = {}, {}
    for it in items or []:
        name = (it.get("itemName") or "").strip()
        val = clean_text(str(it.get("value") or "")).strip()
        unit = (it.get("unitType") or "").strip()
        if not name or not val:
            continue
        if name in disp:  # duplicated item names (e.g. Airbags detail + count): keep the richer text
            if len(val) <= len(disp[name]):
                continue
        disp[name] = f"{val} {unit}".strip() if unit and unit.lower() not in val.lower() else val
        raw[name] = (val, unit)
    return disp, raw


def numeric_fields(raw: dict[str, tuple[str, str]]) -> dict:
    v = lambda k: raw.get(k, ("", ""))[0]  # noqa: E731
    u = lambda k: raw.get(k, ("", ""))[1]  # noqa: E731
    ncap, agency = parse_ncap(v("NCAP Rating"))
    fuel_raw = v("Fuel Type") or None
    unit = u("Mileage (ARAI)") or None
    is_ev = canon_fuel(fuel_raw) == "Electric"
    return {
        "fuel_raw": fuel_raw,
        "engine_type": v("Engine Type") or None,
        "transmission_raw": v("Transmission Type") or None,
        "gears": v("No of Gears") or None,
        "drivetrain": v("Drivetrain") or None,
        "engine_cc": positive(to_int(v("Displacement"))),
        "power_bhp": positive(num(v("Max Engine Power"))),
        "torque_nm": positive(num(v("Max Engine Torque"))),
        "mileage_arai": None if is_ev else positive(num(v("Mileage (ARAI)"))),
        "mileage_unit": None if is_ev else unit,
        "mileage_owner": positive(num(v("Mileage - Owner Reported"))),
        "mileage_expected": None if is_ev else positive(num(v("Expected Mileage"))),
        "mileage_city_tested": positive(num(v("City Mileage (As tested by CarWale)"))),
        "mileage_highway_tested": positive(num(v("Highway Mileage (As tested by CarWale)"))),
        "range_km": next((positive(num(v(k))) for k in ("Expected Range", "MIDC Certified Range (km)",
                          "ARAI Certified Range (km)", "Driving Range") if positive(num(v(k)))), None) if is_ev else None,
        "range_owner_km": positive(num(v("Range - Owner Reported"))),
        "range_tested_km": positive(num(v("Range (As tested by CarWale)"))),
        "battery_kwh": positive(num(v("Battery Capacity"))),
        "dc_charge_minutes": positive(num(v("DC Fast charging time 1 - minutes"))),
        "ac_charge_time": v("Charging Time at 220V") or v("AC Regular Charging") or None,
        "seats": positive(to_int(v("Seating Capacity"))),
        "airbags": positive(to_int(v("Airbags"))),
        "airbags_text": v("Airbags") or None,
        "ncap_rating": ncap,
        "ncap_agency": agency,
        "boot_l": positive(to_int(v("Bootspace"))),
        "ground_clearance_mm": positive(to_int(v("Ground Clearance (unladen)"))),
        "kerb_weight_kg": positive(to_int(v("Kerb Weight"))),
        "fuel_tank_l": positive(to_int(v("Fuel Tank Capacity"))),
        "length_mm": positive(to_int(v("Length"))),
        "width_mm": positive(to_int(v("Width"))),
        "height_mm": positive(to_int(v("Height"))),
        "wheelbase_mm": positive(to_int(v("Wheelbase"))),
        "acceleration_0_100_s": positive(num(v("Acceleration (0-100 kmph) Claimed"))),
        "top_speed_kmph": positive(to_int(v("Top Speed"))),
        "emission_standard": v("Emission Standard") or None,
    }


def feature_tree(tp: dict) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for cat in tp.get("specsFeaturesMaster") or []:
        for sub in cat.get("subCategories") or []:
            items = []
            for it in sub.get("items") or []:
                name = (it.get("itemName") or "").strip()
                val = clean_text(str(it.get("value") or "")).strip()
                unit = (it.get("unitType") or "").strip()
                if name and val:
                    items.append({"name": name, "value": f"{val} {unit}".strip() if unit and unit not in val else val})
            if items:
                groups[f'{cat.get("name")} > {sub.get("name")}'] = items
    return groups


# --------------------------------------------------------------------------- model page
def model_status(md: dict) -> str:
    if md.get("discontinuedOn"):
        return "discontinued"
    po = md.get("priceOverview") or {}
    launched = parse_date(md.get("launchedOn"))
    if po.get("priceStatus") == 4 or "estimat" in (po.get("priceLabel") or "").lower() or (launched and launched > TODAY):
        return "upcoming"
    return "launched"


def body_type(mp: dict) -> str | None:
    txt = (((mp.get("extendedH1SeoText") or {}).get("engineAndVariantOptions") or {}).get("bodyTypeText"))
    if txt:
        return txt.strip()
    schema = ((mp.get("metaTags") or {}).get("schema")) or ""
    m = re.search(r'"bodyType"\s*:\s*"([^"]+)"', schema)
    return BODY_LD_MAP.get(m.group(1).lower(), m.group(1)) if m else None


def owner_reviews(widget: dict) -> list[dict]:
    seen, out = set(), []
    for key in ("relevantUserReviews", "criticalUserReviews", "positiveUserReviews", "userReviews"):
        for r in widget.get(key) or []:
            rid = r.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            params = {}
            for p in r.get("reviewParameters") or []:
                h = (p.get("heading") or "").strip()
                if h:
                    params[h] = p.get("answerText") or p.get("answer")
            out.append({
                "id": rid,
                "title": clean_text(r.get("title")),
                "text": clean_text(r.get("description"))[:1500],
                "rating": r.get("userRating"),
                "params": params,
                "date": r.get("entryDate"),
                "kind": "critical" if key == "criticalUserReviews" else "general",
            })
    return out


def parse_model(url: str, mp: dict) -> tuple[dict, list[dict]]:
    md = mp.get("modelDetails") or {}
    brand_slug, slug = md.get("makeMaskingName"), md.get("modelMaskingName")
    mid = f"{brand_slug}/{slug}"
    po = md.get("priceOverview") or {}
    eh = mp.get("extendedH1SeoText") or {}
    ev_opts = eh.get("engineAndVariantOptions") or {}
    take = mp.get("carWaleTake") or {}
    widget = mp.get("userReviewsWidget") or {}
    agg = widget.get("aggregateRating") or {}

    def pages(key):
        return [{"title": clean_text(p.get("pageName")), "text": clean_text(p.get("content"))}
                for p in ((mp.get(key) or {}).get("pageList") or []) if p.get("content")]

    faqs = []
    for cat in mp.get("faqsByCategory") or []:
        for f in cat.get("faqs") or []:
            faqs.append({"category": cat.get("categoryName"), "q": clean_text(f.get("question")),
                         "a": clean_text(f.get("answer"))})

    city_prices = []
    for c in mp.get("geoCityPrices") or []:
        city_prices.append({"city": c.get("name"), "range": c.get("onRoadPrice"),
                            "min": positive(to_int(c.get("unformattedOnRoadPrice"))),
                            "max": positive(to_int(c.get("onRoadMaxPrice")))})

    similar = []
    for s in mp.get("alternativeModelSpecsComparisonData") or []:
        sid = f'{s.get("makeMaskingName")}/{s.get("modelMaskingName")}'
        if sid != mid and sid not in similar:
            similar.append(sid)

    highlights = []
    for im in ((mp.get("modelTourData") or {}).get("images") or []):
        t = clean_text(im.get("uspText"))
        if t and t not in highlights:
            highlights.append(t)

    summary_page = ((mp.get("summaryOverview") or {}).get("page") or {})
    rsum = mp.get("userReviewsSummaryData") or {}

    model = {
        "id": mid,
        "cw_model_id": md.get("modelId"),
        "brand": md.get("makeName"),
        "brand_slug": brand_slug,
        "name": md.get("modelName"),
        "slug": slug,
        "full_name": f'{md.get("makeName")} {md.get("modelName")}',
        "url": url,
        "status": model_status(md),
        "launched_on": parse_date(md.get("launchedOn")),
        "discontinued_on": parse_date(md.get("discontinuedOn")),
        "body_type_raw": body_type(mp),
        "is_ev": bool(md.get("isElectricVehicle")),
        "price_text": po.get("formattedPriceRangeText") or po.get("formattedPrice"),
        "price_label": po.get("priceLabel"),
        "image_url": f'{IMG_CDN}{md["imagePath"]}' if md.get("imagePath") else None,
        "brochure_url": md.get("brochurePath") or None,
        "user_rating": positive(md.get("modelAggregateRating")),
        "user_rating_count": md.get("modelRatingCount") or 0,
        "user_review_count": md.get("modelReviewCount") or 0,
        "rating_breakdown": {p["heading"]: round(p["answer"], 2) for p in (agg.get("reviewParameters") or [])
                             if p.get("heading") and isinstance(p.get("answer"), (int, float))},
        "review_summary": [clean_text(s) for s in (rsum.get("summary") or [])],
        "review_positive_pct": rsum.get("positiveReviewPercentage"),
        "expert_score_10": positive(take.get("overallScore")),
        "expert_rating_5": positive((mp.get("summaryOverview") or {}).get("overallScore")),
        "summary": clean_text(summary_page.get("content")),
        "intro": clean_text(eh.get("initialText")),
        "synopsis": clean_text(" ".join(mp.get("synopsis", {}).get("summary") or [])),
        "key_prices": {k: v for k, v in (eh.get("keyPriceDetails") or {}).items()},
        "engine_options": ev_opts.get("engineOptionsText"),
        "transmission_options": ev_opts.get("transmissionWithGearsText") or ev_opts.get("transmissionText"),
        "variants_text": ev_opts.get("variantsText"),
        "highlights": highlights[:15],
        "verdict": clean_text(take.get("verdict")),
        "pros": list_items(take.get("pros")),
        "cons": list_items(take.get("cons")),
        "expert_review": [{"title": clean_text(p.get("pageName")), "text": clean_text(p.get("content"))}
                          for p in (take.get("reviewDescription") or []) if p.get("content")],
        "expert_opinion": {
            "Engine & Performance": pages("engineContent"),
            "Convenience & Technology": pages("featureContent"),
            "Safety": pages("safetyContent"),
            "Exterior & Interior Styling": pages("interiorContent"),
        },
        "mileage_table": mp.get("mileage") or [],
        "key_specs": {k.get("title"): ", ".join(t.get("text", "") for t in (k.get("keySpecsValue") or []))
                      for k in (mp.get("keySpecs") or []) if k.get("title")},
        "faqs": faqs,
        "city_prices": city_prices,
        "colours": [c.get("name") for c in ((mp.get("color") or {}).get("colors") or []) if c.get("name")],
        "similar": similar[:8],
        "latest_updates": [{"date": (u.get("publishedDate") or "")[:10], "title": clean_text(u.get("title"))}
                           for u in (mp.get("latestUpdateList") or [])][:6],
        "owner_reviews": owner_reviews(widget),
        "trims": [{"name": t.get("trimName"), "slug": t.get("trimMaskingName"),
                   "url": f'{url}{t.get("trimMaskingName")}/', "version_ids": t.get("versionIds") or []}
                  for t in (mp.get("trimSummary") or [])],
        "upcoming_mentions": [],
        "scraped_at": TODAY,
    }
    # estimated price for upcoming cars (no versions yet)
    est_min, est_max = lakh_text_to_inr(model["price_text"])
    model["price_est_min"], model["price_est_max"] = est_min, est_max

    base_variants = []
    for v in mp.get("versions") or []:
        vpo = v.get("priceOverview") or {}
        price = positive(vpo.get("price")) if "on-road" not in (vpo.get("priceLabel") or "").lower() else None
        base_variants.append({
            "version_id": v.get("versionId"),
            "model_id": mid,
            "name": v.get("versionName"),
            "display_name": v.get("displayName"),
            "trim": v.get("trimName"),
            "trim_slug": v.get("trimMaskingName"),
            "url": f'{url}{v.get("trimMaskingName")}/' if v.get("trimMaskingName") else url,
            "price": price,
            "launched_on": parse_date(v.get("launchedOn")),
            "_items": (v.get("specsSummary") or []) + (v.get("featureSpecs") or []),
        })
    for u in mp.get("upcomingPopularCars") or []:
        model["upcoming_mentions"].append({
            "brand": u.get("makeName"), "brand_slug": u.get("makeMaskingName"),
            "name": u.get("modelName") or u.get("rootName"), "slug": u.get("modelMaskingName") or u.get("rootMaskingName"),
            "price_text": (u.get("priceOverview") or {}).get("formattedPrice"),
            "expected_launch": parse_date(u.get("launchedOn")),
        })
    return model, base_variants


# --------------------------------------------------------------------------- trim page
def parse_trim(tp: dict) -> tuple[dict[int, list[dict]], dict, list[dict], str | None]:
    """-> ({version_id: spec items}, feature tree of the page's version, trim FAQs, tree version name)."""
    by_vid: dict[int, list[dict]] = {}
    vd = tp.get("versionDetail") or {}
    for v in [vd] + (tp.get("versions") or []) + (tp.get("otherVersions") or []):
        vid = v.get("versionId")
        if vid and vid not in by_vid:
            by_vid[vid] = (v.get("specsSummary") or []) + (v.get("featureSpecs") or [])
    faqs = [{"category": "Top variant", "q": clean_text(f.get("question")), "a": clean_text(f.get("answer"))}
            for f in (tp.get("faqs") or [])]
    return by_vid, feature_tree(tp), faqs, vd.get("versionName")


# --------------------------------------------------------------------------- assemble
def fill_from_mileage_table(v: dict, table: list[dict]) -> None:
    """Many versions omit 'Mileage (ARAI)'; the model's mileage table has it per fuel/engine/transmission."""
    is_ev = v.get("fuel_type") == "Electric"
    if not table:
        return
    best, best_score = None, -1
    for row in table:
        if canon_fuel(row.get("fuelType")) != v.get("fuel_type"):
            continue
        score = 1
        cc = to_int(row.get("displacement"))
        if cc and v.get("engine_cc"):
            if abs(cc - v["engine_cc"]) > 15:
                continue
            score += 2
        rt, _ = canon_transmission(row.get("transmission"))
        if rt and v.get("transmission"):
            if rt == v["transmission"] or (row.get("transmission") or "").lower() == (v.get("transmission_raw") or "").lower():
                score += 2
            elif v.get("is_automatic") is not None and (rt == "Manual") == (not v["is_automatic"]):
                score += 1
            else:
                continue
        if score > best_score:
            best, best_score = row, score
    if not best:
        return
    if is_ev:
        v["range_km"] = v.get("range_km") or positive(best.get("arai"))
        v["range_owner_km"] = v.get("range_owner_km") or positive(best.get("userReportedArai"))
    else:
        if not v.get("mileage_arai") and positive(best.get("arai")):
            v["mileage_arai"] = positive(best.get("arai"))
            v["mileage_source"] = "model_table"
        v["mileage_unit"] = v.get("mileage_unit") or best.get("mileageUnit") or "kmpl"
        v["mileage_owner"] = v.get("mileage_owner") or positive(best.get("userReportedArai"))
        v["mileage_expected"] = v.get("mileage_expected") or positive(best.get("expected"))
        v["mileage_city_tested"] = v.get("mileage_city_tested") or (positive(best.get("expertTested")) if not v.get("mileage_highway_tested") else None)


def set_effective_mileage(v: dict) -> None:
    """Single sortable efficiency figure with an explicit basis (claimed > expected > owner-reported)."""
    if v.get("fuel_type") == "Electric":
        v["mileage_effective"], v["mileage_basis"] = v.get("range_km") or v.get("range_owner_km"), "range_km"
        return
    for key, basis in (("mileage_arai", "ARAI claimed"), ("mileage_expected", "expected"), ("mileage_owner", "owner-reported")):
        if v.get(key):
            v["mileage_effective"], v["mileage_basis"] = v[key], basis
            return
    v["mileage_effective"], v["mileage_basis"] = None, None


def build_variant(base: dict, trim_items: list[dict] | None, model: dict) -> dict:
    disp, raw = spec_items_to_dict(base.pop("_items"))
    if trim_items:  # trim page has the fuller 86-item spec summary; it wins on conflicts
        d2, r2 = spec_items_to_dict(trim_items)
        disp.update(d2)
        raw.update(r2)
    nf = numeric_fields(raw)
    fuel = canon_fuel(nf["fuel_raw"])
    trans, is_auto = canon_transmission(nf["transmission_raw"])
    htype = hybrid_type(nf["fuel_raw"], nf["engine_type"], base.get("name"), disp.get("Engine Type"))
    if fuel == "Electric":
        htype = None
    variant = {**base, **nf, "fuel_type": fuel, "hybrid_type": htype, "transmission": trans,
               "is_automatic": is_auto, "specs": disp}
    fill_from_mileage_table(variant, model.get("mileage_table") or [])
    set_effective_mileage(variant)
    variant["feature_tags"] = variant_tags(disp, nf["airbags"])
    return variant


def finalize_model(model: dict, variants: list[dict], tree: dict, tree_version: str | None) -> dict:
    prices = [v["price"] for v in variants if v.get("price")]
    model["price_min"] = min(prices) if prices else model.get("price_est_min")
    model["price_max"] = max(prices) if prices else model.get("price_est_max")
    model["price_is_estimate"] = not prices
    model["body_type"] = canon_body(model.get("body_type_raw"))
    model["fuel_types"] = sorted({v["fuel_type"] for v in variants if v.get("fuel_type")})
    model["hybrid_types"] = sorted({v["hybrid_type"] for v in variants if v.get("hybrid_type")})
    model["transmissions"] = sorted({v["transmission"] for v in variants if v.get("transmission")})
    model["has_automatic"] = any(v.get("is_automatic") for v in variants)
    model["seats"] = sorted({v["seats"] for v in variants if v.get("seats")})
    ncaps = [(v["ncap_rating"], v["ncap_agency"]) for v in variants if v.get("ncap_rating")]
    model["ncap_rating"], model["ncap_agency"] = (max(ncaps) if ncaps else (None, None))
    model["airbags_max"] = max([v["airbags"] for v in variants if v.get("airbags")], default=None)
    model["mileage_best"] = max([v["mileage_arai"] for v in variants if v.get("mileage_arai")], default=None) or \
        max([v["mileage_effective"] for v in variants if v.get("mileage_effective") and v.get("fuel_type") != "Electric"], default=None)
    model["mileage_claimed_text"] = (model.get("key_specs") or {}).get("Mileage")
    model["range_best_km"] = max([v["range_km"] for v in variants if v.get("range_km")], default=None)
    model["power_max_bhp"] = max([v["power_bhp"] for v in variants if v.get("power_bhp")], default=None)
    model["boot_l"] = max([v["boot_l"] for v in variants if v.get("boot_l")], default=None)
    model["ground_clearance_mm"] = max([v["ground_clearance_mm"] for v in variants if v.get("ground_clearance_mm")], default=None)
    tags = set(tree_tags(tree))
    for v in variants:
        tags.update(v["feature_tags"])
    model["feature_tags"] = sorted(tags)
    model["top_variant_features"] = {"version": tree_version, "groups": tree}
    warranty = [f'{it["name"]}: {it["value"]}' for g, items in tree.items() if "warranty" in g.lower() for it in items]
    model["warranty"] = warranty
    return model


def main() -> None:
    model_urls = json.loads((DATA / "urls_models.json").read_text(encoding="utf-8"))
    models, variants, errors = [], [], []
    for url in model_urls:
        html = read_cached(url)
        if not html:
            continue
        state = extract_state(html) or {}
        mp = state.get("modelPage")
        if not mp or not mp.get("modelDetails"):
            errors.append((url, "no modelPage state"))
            continue
        try:
            model, base_vars = parse_model(url, mp)
            trim_items, tree, trim_faqs, tree_version = {}, {}, [], None
            top_trim = next((t for t in model["trims"] if (mp.get("modelDetails") or {}).get("topVariant") in t["version_ids"]), None)
            top_trim = top_trim or (model["trims"][-1] if model["trims"] else None)
            if top_trim:
                th = read_cached(top_trim["url"])
                tp = (extract_state(th) or {}).get("trimPage") if th else None
                if tp:
                    trim_items, tree, trim_faqs, tree_version = parse_trim(tp)
            model["faqs"].extend(trim_faqs)
            mvars = [build_variant(b, trim_items.get(b["version_id"]), model) for b in base_vars]
            models.append(finalize_model(model, mvars, tree, tree_version))
            variants.extend(mvars)
        except Exception as e:  # keep going; report at the end
            errors.append((url, f"{type(e).__name__}: {e}"))

    n_m = write_jsonl(PROCESSED / "models.jsonl", models)
    n_v = write_jsonl(PROCESSED / "variants.jsonl", variants)
    report(models, variants)
    print(f"\nwrote {n_m} models, {n_v} variants; errors: {len(errors)}")
    for u, e in errors[:15]:
        print("  ERR", u, e)


def report(models: list[dict], variants: list[dict]) -> None:
    print("models by status:", dict(Counter(m["status"] for m in models)))
    print("models by body type:", dict(Counter(m.get("body_type") for m in models)))
    launched = {m["id"] for m in models if m["status"] == "launched"}
    lv = [v for v in variants if v["model_id"] in launched]
    print(f"launched-model variants: {len(lv)}")

    def pct(field, rows=lv, cond=None):
        rows = [r for r in rows if cond is None or cond(r)]
        return f"{field}: {100 * sum(1 for r in rows if r.get(field) not in (None, '', [])) / max(len(rows), 1):.0f}% of {len(rows)}"

    non_ev = lambda r: r.get("fuel_type") != "Electric"  # noqa: E731
    ev = lambda r: r.get("fuel_type") == "Electric"  # noqa: E731
    for f in ("price", "fuel_type", "transmission", "seats", "engine_cc", "power_bhp", "airbags", "boot_l",
              "ground_clearance_mm", "ncap_rating", "length_mm"):
        print("  ", pct(f))
    print("  ", pct("mileage_arai", cond=non_ev), "|", pct("range_km", cond=ev), "|", pct("battery_kwh", cond=ev))
    print("fuel types:", dict(Counter(v.get("fuel_type") for v in lv)))
    print("hybrid types:", dict(Counter(v.get("hybrid_type") for v in lv)))
    print("transmissions:", dict(Counter(v.get("transmission") for v in lv)))
    tags = Counter(t for m in models if m["status"] == "launched" for t in m["feature_tags"])
    print("model feature tags:", dict(tags.most_common(40)))
    print("models with owner reviews:", sum(1 for m in models if m["owner_reviews"]),
          "| with faqs:", sum(1 for m in models if m["faqs"]),
          "| with expert review:", sum(1 for m in models if m["expert_review"]),
          "| with feature tree:", sum(1 for m in models if m["top_variant_features"]["groups"]))


if __name__ == "__main__":
    sys.exit(main())
