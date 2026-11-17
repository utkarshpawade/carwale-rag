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


def trim_highlights(tp: dict) -> dict:
    """What this trim adds over the trim below it (and what the next trim adds) - CarWale's own diff."""
    def fmt(items):
        out = []
        for i in items or []:
            name = (i.get("itemName") or "").strip()
            val = clean_text(str(i.get("value") or "")).strip()
            if name:
                out.append(name if val in ("", "Yes") else f"{name}: {val}")
        return out
    for cat in tp.get("specsFeaturesMaster") or []:
        for sub in cat.get("subCategories") or []:
            hl = sub.get("highLights") or {}
            if hl.get("addedInCurrent") or hl.get("addedInNext") or hl.get("lowerTrimName") or hl.get("upperTrimName"):
                return {"lower_trim": hl.get("lowerTrimName") or None, "upper_trim": hl.get("upperTrimName") or None,
                        "added": fmt(hl.get("addedInCurrent")), "next_adds": fmt(hl.get("addedInNext")),
                        "segment_best": fmt(hl.get("segmentBest"))}
    return {}


def review_page_reviews(model_url: str) -> list[dict]:
    """Owner reviews from the dedicated review pages (/reviews/, /reviews/page/N/)."""
    out = []
    for page in range(1, 6):
        url = f"{model_url}reviews/" if page == 1 else f"{model_url}reviews/page/{page}/"
        html = read_cached(url)
        if not html:
            break
        rl = (extract_state(html) or {}).get("userReviewListing") or {}
        for r in ((rl.get("userReviewList") or {}).get("Reviews") or []):
            params = {}
            for q in r.get("QuesAns") or []:
                h = (q.get("Heading") or "").strip()
                if h and q.get("QuestionType") in (1, 2, 6):
                    params[h] = q.get("AnswerText") or q.get("Answer")
            rating = (r.get("RatingInfo") or {}).get("UserRating")
            text = clean_text(r.get("Description"))
            tips = clean_text(r.get("TipsAdvices"))
            out.append({
                "id": r.get("Id"), "title": clean_text(r.get("Title")),
                "text": (text + (f"\nTips: {tips}" if tips else ""))[:1500],
                "rating": rating, "params": params,
                "date": r.get("EntryDate"), "kind": "critical" if rating and rating <= 2 else "general",
                "version": (r.get("VehicleInfo") or {}).get("VersionName"),
                "mileage": positive(num(r.get("Mileage"))), "city": r.get("CityName") or None,
            })
    return out


def merge_reviews(a: list[dict], b: list[dict], cap: int = 40) -> list[dict]:
    seen, out = set(), []
    for r in a + b:
        if r.get("id") in seen or not r.get("text"):
            continue
        seen.add(r.get("id"))
        out.append(r)
    critical = [r for r in out if r["kind"] == "critical"]
    rest = [r for r in out if r["kind"] != "critical"]
    return (critical[: cap // 4] + rest)[:cap]  # keep some critical voices even when capping


def load_articles() -> dict[str, list[dict]]:
    """Expert articles (long-term reports, first drives) from the stage-2 crawl, grouped by model id."""
    path = DATA / "urls_stage2.json"
    if not path.exists():
        return {}
    current = {u.split("carwale.com/")[1].rstrip("/") for u in json.loads((DATA / "urls_models.json").read_text(encoding="utf-8"))}
    by_model: dict[str, list[dict]] = {}
    for url in json.loads(path.read_text(encoding="utf-8")):
        html = read_cached(url)
        ed = (extract_state(html) or {}).get("editorialDetails") if html else None
        if not ed or not ed.get("pageList"):
            continue
        mid = f"{ed.get('makeMaskingName')}/{ed.get('modelMaskingName')}"
        older = None
        if f"{ed.get('makeMaskingName')}-cars/{ed.get('modelMaskingName')}" not in current:
            # article about an earlier generation, listed on the current model's page: attach to that model, labelled
            m = re.search(r"carwale\.com/([a-z0-9-]+)-cars/([a-z0-9-]+)/", url)
            if not m:
                continue
            mid, older = f"{m.group(1)}/{m.group(2)}", clean_text(ed.get("modelName")) or "earlier generation"
        pages = [{"title": clean_text(p.get("pageName")), "text": clean_text(p.get("content"))}
                 for p in ed["pageList"] if p.get("content")]
        by_model.setdefault(mid, []).append({
            "title": clean_text(ed.get("title")), "url": url, "date": ed.get("publishedDate"),
            "author": ed.get("authorName"), "pages": pages, "older_generation": older,
        })
    return by_model


def stage3_urls(kind: str) -> list[str]:
    path = DATA / "urls_stage3.json"
    if not path.exists():
        return []
    urls = json.loads(path.read_text(encoding="utf-8"))
    test = {"compare": lambda u: "/compare-cars/" in u,
            "best": lambda u: "/compare-cars/" not in u and "/price-in-" not in u and not u.endswith("/news/")}[kind]
    return [u for u in urls if test(u)]


def norm_city(name: str | None) -> str:
    return re.sub(r"[^a-z]", "", (name or "").lower())


def city_price_pages(model_url: str) -> list[dict]:
    """Per-variant on-road prices (+ one full breakup, nearby cities, city FAQs) from /price-in-<city>/ pages."""
    out = []
    for city in ("mumbai", "delhi"):
        html = read_cached(f"{model_url}price-in-{city}/")
        pc = (extract_state(html) or {}).get("priceInCityPage") if html else None
        if not pc:
            continue
        cname = (pc.get("priceLocation") or {}).get("cityName") or city.title()
        versions, names = [], {}
        for v in pc.get("versions") or []:
            names[v.get("versionId")] = v.get("versionName")
            po = v.get("priceOverview") or {}
            if "on-road" in (po.get("priceLabel") or "").lower() and positive(po.get("price")):
                versions.append({"name": v.get("versionName"), "trim": v.get("trimName"), "on_road": po["price"],
                                 "ex_showroom": positive(po.get("exShowRoomPrice"))})
        breakup = None
        for d in pc.get("detailedPriceList") or []:
            for b in d.get("PriceBreakupList") or []:
                if b.get("charges"):
                    breakup = {"version": names.get((d.get("PriceCriteria") or {}).get("versionId")),
                               "on_road": b.get("onRoad"), "ex_showroom": b.get("exShowRoom"),
                               "charges": [{"name": c.get("name"), "group": c.get("chargeGroupName") or "", "value": c.get("value")}
                                           for c in b["charges"] if c.get("value")]}
                    break
            if breakup:
                break
        nearby = [{"city": c.get("name"), "range": c.get("onRoadPrice"), "min": positive(to_int(c.get("unformattedOnRoadPrice"))),
                   "max": positive(to_int(c.get("onRoadMaxPrice")))} for c in pc.get("nearByCities") or [] if c.get("onRoadPrice")]
        faqs = [{"category": f"On-road price ({cname})", "q": clean_text(f.get("question")), "a": clean_text(f.get("answer"))}
                for f in pc.get("faqs") or [] if f.get("question")]
        out.append({"city": cname, "versions": versions, "breakup": breakup, "nearby": nearby, "faqs": faqs})
    return out


def news_headlines(model_url: str, limit: int = 15) -> list[dict]:
    html = read_cached(f"{model_url}news/")
    el = (extract_state(html) or {}).get("editorialListing") if html else None
    out = []
    for i in ((el or {}).get("contentData") or [])[:limit]:
        if i.get("title"):
            u = i.get("url") or ""
            out.append({"title": clean_text(i["title"]), "date": i.get("displayDate"),
                        "url": f"https://www.carwale.com{u}" if u.startswith("/") else u})
    return out


def load_comparisons() -> dict[str, list[dict]]:
    """CarWale's written verdicts for rival pairs; templated boilerplate ('CarWale brings you comparison...') is dropped."""
    slug2id = {}
    for u in json.loads((DATA / "urls_models.json").read_text(encoding="utf-8")):
        m = re.search(r"carwale\.com/([a-z0-9-]+)-cars/([a-z0-9-]+)/", u)
        if m:
            slug2id[f"{m.group(1)}-{m.group(2)}"] = f"{m.group(1)}/{m.group(2)}"
    by_model: dict[str, list[dict]] = {}
    for url in stage3_urls("compare"):
        html = read_cached(url)
        cc = (extract_state(html) or {}).get("compareCarDetails") if html else None
        if not cc:
            continue
        text = clean_text((cc.get("metaData") or {}).get("comparisionText"))
        if text.lower().startswith("carwale brings you comparison"):
            text = ""  # templated boilerplate, not an expert verdict
        a, _, b = url.rstrip("/").split("/")[-1].partition("-vs-")
        ida, idb = slug2id.get(a), slug2id.get(b)
        # CarWale's expert take on each car of the pair (same order as in the URL)
        takes = [clean_text(v.get("verdict")) for v in cc.get("verdictInfoList") or []]
        if not ida or not idb or not (text or any(takes)):
            continue
        rec = {"pair": [ida, idb], "text": text, "takes": takes[:2], "url": url,
               "road_tests": [clean_text(r.get("title")) for r in cc.get("roadTestList") or [] if r.get("title")][:4]}
        by_model.setdefault(ida, []).append(rec)  # one record per pair, stored under the first car
    return by_model


def load_best_lists() -> list[dict]:
    """CarWale's curated 'best cars ...' pages: ranked models, summary text and FAQs."""
    out = []
    for url in stage3_urls("best"):
        html = read_cached(url)
        fp = (extract_state(html) or {}).get("filterPage") if html else None
        if not fp or not fp.get("models"):
            continue
        meta = fp.get("pageMetaTags") or {}
        ranked = []
        for m in fp["models"]:
            po = m.get("priceOverview") or {}
            ranked.append({"id": f"{m.get('makeMaskingName')}/{m.get('modelMaskingName')}",
                           "name": f"{m.get('makeName')} {m.get('modelName')}",
                           "price": clean_text(po.get("formattedPriceRangeText") or po.get("formattedPrice")),
                           "rating": m.get("carRating")})
        out.append({"title": clean_text(meta.get("heading") or meta.get("title")), "url": url, "ranked": ranked,
                    "summary": "\n".join(t for t in (clean_text(meta.get("seoText")), clean_text(fp.get("seoText"))) if t),
                    "faqs": [{"q": clean_text(f.get("question")), "a": clean_text(f.get("answer"))}
                             for f in fp.get("faqs") or [] if f.get("question")]})
    return out


def main() -> None:
    model_urls = json.loads((DATA / "urls_models.json").read_text(encoding="utf-8"))
    models, variants, errors = [], [], []
    articles = load_articles()
    comparisons = load_comparisons()
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
            # every cached trim page of this model: merge version specs, keep the richest feature tree
            ordered = ([top_trim] if top_trim else []) + [t for t in reversed(model["trims"]) if t is not top_trim]
            trim_info: dict[str, dict] = {}
            for t in ordered:
                th = read_cached(t["url"])
                tp = (extract_state(th) or {}).get("trimPage") if th else None
                if not tp:
                    continue
                items, tr, faqs_t, ver = parse_trim(tp)
                trim_info[t["slug"]] = {"name": t["name"], "url": t["url"], "version": ver, "groups": tr,
                                        **trim_highlights(tp)}
                for vid, its in items.items():
                    trim_items.setdefault(vid, its)
                if len(tr) > len(tree):
                    tree, tree_version = tr, ver
                if not trim_faqs:
                    trim_faqs = faqs_t
            model["faqs"].extend(trim_faqs)
            model["trim_features"] = trim_info
            mvars = [build_variant(b, trim_items.get(b["version_id"]), model) for b in base_vars]
            for v in mvars:  # each variant gets its own trim's full-feature-tree tags (connected car, AEB, OTA...)
                ti = trim_info.get(v.get("trim_slug"))
                if ti and ti["groups"]:
                    v["feature_tags"] = sorted(set(v["feature_tags"]) | set(tree_tags(ti["groups"])))
            model["owner_reviews"] = merge_reviews(model["owner_reviews"], review_page_reviews(url))
            model["expert_articles"] = articles.get(model["id"], [])
            model["city_pages"] = city_price_pages(url)
            for cp in model["city_pages"]:
                model["faqs"].extend(cp.pop("faqs"))
                known = {norm_city(c["city"]) for c in model["city_prices"]}
                ons = [v["on_road"] for v in cp["versions"]]
                if ons and norm_city(cp["city"]) not in known:
                    model["city_prices"].append({"city": cp["city"], "range": f"{min(ons) / 1e5:.2f} - {max(ons) / 1e5:.2f} Lakh",
                                                 "min": min(ons), "max": max(ons)})
                for nb in cp["nearby"]:  # nearby-city on-road ranges widen city coverage
                    if norm_city(nb["city"]) not in {norm_city(c["city"]) for c in model["city_prices"]}:
                        model["city_prices"].append(nb)
            model["news"] = news_headlines(url)
            model["comparisons"] = comparisons.get(model["id"], [])
            models.append(finalize_model(model, mvars, tree, tree_version))
            variants.extend(mvars)
        except Exception as e:  # keep going; report at the end
            errors.append((url, f"{type(e).__name__}: {e}"))

    n_m = write_jsonl(PROCESSED / "models.jsonl", models)
    n_v = write_jsonl(PROCESSED / "variants.jsonl", variants)
    best = load_best_lists()
    n_b = write_jsonl(PROCESSED / "best_lists.jsonl", best)
    report(models, variants)
    L = [m for m in models if m["status"] == "launched"]
    print(f"city on-road pages: {sum(len(m.get('city_pages') or []) for m in L)} | cities covered (avg/model): "
          f"{sum(len(m['city_prices']) for m in L) / max(len(L), 1):.1f} | news headlines: {sum(len(m.get('news') or []) for m in L)} | "
          f"rival pairs: {sum(len(m.get('comparisons') or []) for m in L)} "
          f"(written verdicts: {sum(1 for m in L for c in m.get('comparisons') or [] if c['text'])}) | best lists: {n_b}")
    print(f"\nwrote {n_m} models, {n_v} variants; errors: {len(errors)}")
    for u, e in errors[:15]:
        print("  ERR", u, e)


def report(models: list[dict], variants: list[dict]) -> None:
    L = [m for m in models if m["status"] == "launched"]
    print(f"trim feature trees: {sum(len(m.get('trim_features') or {}) for m in L)} trims across {len(L)} launched models | "
          f"owner reviews: {sum(len(m['owner_reviews']) for m in L)} | expert articles: {sum(len(m.get('expert_articles') or []) for m in L)}")
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
