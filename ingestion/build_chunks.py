"""Turn parsed model/variant records into retrieval chunks with contextual headers + metadata.

Usage: python -m ingestion.build_chunks
Reads  data/processed/models.jsonl, variants.jsonl, ingestion/glossary.md
Writes data/processed/chunks.jsonl
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import PROCESSED, read_jsonl, write_jsonl

MAX_CHARS = 1500  # ~350-400 tokens: safely under bge-small's 512-token window
GLOSSARY = Path(__file__).with_name("glossary.md")


def inr(v: int | float | None) -> str:
    if not v:
        return "n/a"
    return f"₹{v / 1e7:.2f} Cr" if v >= 1e7 else f"₹{v / 1e5:.2f} L"


def price_range(m: dict) -> str:
    lo, hi = m.get("price_min"), m.get("price_max")
    if not lo:
        return "price not announced"
    est = " (expected)" if m.get("price_is_estimate") else ""
    return (f"{inr(lo)}" if lo == hi else f"{inr(lo)} – {inr(hi)}") + f" ex-showroom{est}"


def header(m: dict, section: str) -> str:
    fuels = ", ".join(m.get("fuel_types") or []) or "n/a"
    status = "" if m["status"] == "launched" else f" | {m['status'].upper()}"
    return f"{m['full_name']} | {m.get('body_type_raw') or 'car'} | {price_range(m)} | {fuels}{status} — {section}"


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split on line/sentence boundaries into pieces <= limit chars."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    units: list[str] = []
    for line in text.split("\n"):  # keep line structure (feature groups, bullets); split long lines by clause
        line = line.strip()
        if not line:
            continue
        if len(line) <= limit:
            units.append(line)
        else:
            for s in re.split(r"(?<=[.!?;])\s+", line):
                units += [s[i:i + limit] for i in range(0, len(s), limit)] if s else []
    parts, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > limit:
            parts.append(cur)
            cur = ""
        cur = f"{cur}\n{u}" if cur else u
    if cur:
        parts.append(cur)
    return parts


class Chunker:
    def __init__(self):
        self.rows: list[dict] = []
        self.counter = Counter()

    def add(self, m: dict | None, section: str, title: str, body: str, url: str | None = None,
            variant_id: int | None = None) -> None:
        for piece in split_text(body):
            key = (m["id"] if m else "glossary", section)
            self.counter[key] += 1
            content = (header(m, title) if m else f"Car glossary — {title}") + "\n" + piece
            self.rows.append({
                "id": f'{key[0]}::{section}::{self.counter[key]}',
                "model_id": m["id"] if m else None,
                "variant_id": variant_id,
                "section": section,
                "title": title,
                "content": content,
                "url": url or (m["url"] if m else None),
                "brand": m.get("brand") if m else None,
                "body_type": m.get("body_type") if m else None,
                "fuel_types": m.get("fuel_types") or [] if m else [],
                "price_min": m.get("price_min") if m else None,
                "price_max": m.get("price_max") if m else None,
                "status": m.get("status") if m else None,
                "content_hash": hashlib.sha1(content.encode()).hexdigest(),
            })


def fmt_num(v, unit: str) -> str | None:
    if v in (None, ""):
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v} {unit}".strip()


def model_chunks(ch: Chunker, m: dict, vars_: list[dict]) -> None:
    # 1. overview -------------------------------------------------------------------------
    facts = [
        f"Price: {price_range(m)}." + (f" {m['intro']}" if m.get("intro") else ""),
        f"Status: {m['status']}" + (f", launched on {m['launched_on']}" if m.get("launched_on") and m["status"] == "launched" else "")
        + (f", expected launch {m['launched_on']}" if m.get("launched_on") and m["status"] == "upcoming" else "") + ".",
        f"Body type: {m.get('body_type_raw') or 'n/a'}. Fuel options: {', '.join(m.get('fuel_types') or []) or 'n/a'}"
        + (f" (hybrid type: {', '.join(m['hybrid_types'])})" if m.get("hybrid_types") else "") + ".",
        f"Engine options: {m['engine_options']}." if m.get("engine_options") else "",
        f"Transmissions: {m.get('transmission_options') or ', '.join(m.get('transmissions') or [])}.",
        f"Seating: {', '.join(map(str, m.get('seats') or [])) or 'n/a'}.",
        f"Mileage (CarWale key spec): {m['mileage_claimed_text']}." if m.get("mileage_claimed_text")
        else (f"Best mileage figure: {m['mileage_best']} kmpl." if m.get("mileage_best") else ""),
        f"Best claimed range: {m['range_best_km']:.0f} km." if m.get("range_best_km") else "",
        f"Safety: {m['ncap_rating']}-star {m.get('ncap_agency') or 'NCAP'} rating" if m.get("ncap_rating") else "Safety: no NCAP rating listed",
        f", up to {m['airbags_max']} airbags." if m.get("airbags_max") else ".",
        f"Owner rating: {m['user_rating']:.1f}/5 from {m['user_rating_count']} ratings." if m.get("user_rating") else "",
        f"CarWale expert rating: {m['expert_rating_5']}/5." if m.get("expert_rating_5") else "",
        m.get("variants_text") or "",
    ]
    body = " ".join(f for f in facts if f).replace(" .", ".").replace(" ,", ",")
    if m.get("summary"):
        body = m["summary"] + "\n" + body
    ch.add(m, "overview", "Overview", body)
    if m.get("highlights"):
        ch.add(m, "highlights", "Top things to know", "\n".join(f"- {h}" for h in m["highlights"]))
    if m.get("synopsis"):
        ch.add(m, "synopsis", "Detailed overview", m["synopsis"])

    # 2. variants & prices -----------------------------------------------------------------
    if vars_:
        lines = []
        for v in sorted(vars_, key=lambda x: x.get("price") or 0):
            bits = [v.get("fuel_type"), v.get("transmission"), fmt_num(v.get("seats"), "seats")]
            econ = (fmt_num(v.get("mileage_arai"), v.get("mileage_unit") or "kmpl")
                    or fmt_num(v.get("range_km"), "km range")
                    or (fmt_num(v.get("mileage_effective"), v.get("mileage_unit") or "kmpl") and
                        f"{fmt_num(v.get('mileage_effective'), v.get('mileage_unit') or 'kmpl')} ({v.get('mileage_basis')})"))
            lines.append(f"- {v['name']}: {inr(v.get('price'))} | {' | '.join(b for b in bits if b)}" + (f" | {econ}" if econ else ""))
        ch.add(m, "variants", "Variants and ex-showroom prices", "\n".join(lines))
        kp = m.get("key_prices") or {}
        kp_lines = [f"- {d.get('name')}: {d.get('minPrice')} – {d.get('maxPrice')}" for d in kp.values() if isinstance(d, dict) and d.get("name")]
        if kp_lines:
            ch.add(m, "variants", "Price summary", "\n".join(kp_lines))

    # 3. per-trim chunks ------------------------------------------------------------------
    by_trim = defaultdict(list)
    for v in vars_:
        by_trim[v.get("trim") or v["name"]].append(v)
    for trim, tv in by_trim.items():
        tv.sort(key=lambda x: x.get("price") or 0)
        lines = []
        for v in tv:
            spec_bits = [
                fmt_num(v.get("engine_cc"), "cc"), fmt_num(v.get("power_bhp"), "bhp"), fmt_num(v.get("torque_nm"), "Nm"),
                fmt_num(v.get("mileage_arai"), v.get("mileage_unit") or "kmpl") and f"ARAI {fmt_num(v.get('mileage_arai'), v.get('mileage_unit') or 'kmpl')}",
                fmt_num(v.get("range_km"), "km claimed range"), fmt_num(v.get("battery_kwh"), "kWh battery"),
                fmt_num(v.get("airbags"), "airbags"),
            ]
            lines.append(f"- {v['name']} ({inr(v.get('price'))}): {', '.join(b for b in spec_bits if b)}")
        common = set.intersection(*(set(v["feature_tags"]) for v in tv)) if tv else set()
        feat = ", ".join(sorted(common)).replace("_", " ")
        ti = (m.get("trim_features") or {}).get(tv[0].get("trim_slug")) or {}
        extra = []
        if ti.get("added"):
            extra.append(f"What {trim} adds over {ti.get('lower_trim') or 'the trim below'}: {', '.join(ti['added'])}.")
        if ti.get("next_adds"):
            extra.append(f"Upgrading to {ti.get('upper_trim') or 'the next trim'} adds: {', '.join(ti['next_adds'])}.")
        if ti.get("segment_best"):
            extra.append(f"Segment-best in this trim: {', '.join(ti['segment_best'])}.")
        body = "\n".join(lines) + (f"\nKey features in this trim: {feat}." if feat else "") + \
            ("\n" + "\n".join(extra) if extra else "")
        ch.add(m, "trim", f"{trim} trim", body, url=tv[0].get("url"))
        groups = ti.get("groups") or {}
        if groups and ti.get("version") != (m.get("top_variant_features") or {}).get("version"):
            feats = []
            for g, items in groups.items():
                yes_items = [i["name"] + ("" if i["value"] == "Yes" else f": {i['value']}") for i in items if i["value"] != "No"]
                if yes_items and not g.lower().startswith("specification"):
                    feats.append(f"{g.split('>')[-1].strip()}: {'; '.join(yes_items)}")
            if feats:
                ch.add(m, "trim_features", f"{trim} trim features ({ti.get('version')})", "\n".join(feats), url=ti.get("url"))

    # 4. pros/cons + expert opinion + expert review ---------------------------------------
    if m.get("pros") or m.get("cons") or m.get("verdict"):
        body = (f"Verdict: {m['verdict']}\n" if m.get("verdict") else "")
        body += "Pros:\n" + "\n".join(f"- {p}" for p in m.get("pros") or []) + "\nCons:\n" + "\n".join(f"- {c}" for c in m.get("cons") or [])
        ch.add(m, "pros_cons", "Expert verdict, pros and cons", body)
    for area, pages in (m.get("expert_opinion") or {}).items():
        for p in pages:
            ch.add(m, "expert_opinion", f"Expert opinion: {area} ({p['title']})", p["text"])
    for p in m.get("expert_review") or []:
        ch.add(m, "expert_review", f"Expert review: {p['title']}", p["text"])
    for a in m.get("expert_articles") or []:  # long-term reports, first drives, service experiences
        date = ", ".join((a.get("date") or "").split(",")[:2]).strip()
        gen = f" [about the earlier generation: {a['older_generation']}]" if a.get("older_generation") else ""
        for p in a["pages"]:
            sub = f" — {p['title']}" if p["title"] and p["title"].lower() not in a["title"].lower() else ""
            ch.add(m, "expert_article", f"CarWale article: {a['title']}{sub} ({date}){gen}", p["text"], url=a["url"])

    # 5. mileage ---------------------------------------------------------------------------
    rows = []
    for r in m.get("mileage_table") or []:
        unit = r.get("mileageUnit") or "kmpl"
        label = "range" if unit == "km" else "mileage"
        bits = [f"claimed (ARAI/MIDC) {label} {r['arai']} {unit}" if r.get("arai") else None,
                f"owner-reported {r['userReportedArai']} {unit}" if r.get("userReportedArai") else None,
                f"CarWale tested {r['expertTested']} {unit}" if r.get("expertTested") else None]
        rows.append(f"- {r.get('fuelType')} {r.get('displacement') or ''}cc {r.get('transmission')}: "
                    .replace(" cc", "").replace("  ", " ") + "; ".join(b for b in bits if b))
    if m.get("mileage_claimed_text"):
        rows.insert(0, f"- Claimed mileage/range (CarWale key spec): {m['mileage_claimed_text']}")
    if rows and not any(r.get("arai") for r in m.get("mileage_table") or []):
        rows.append("- Note: CarWale does not list a per-variant ARAI figure for this model; owner-reported figures are real-world averages.")
    tested = [v for v in vars_ if v.get("mileage_city_tested") or v.get("mileage_highway_tested")]
    seen_tests = set()
    for v in tested:
        key = (v.get("fuel_type"), v.get("transmission"), v.get("mileage_city_tested"), v.get("mileage_highway_tested"))
        if key in seen_tests or len(seen_tests) >= 4:
            continue
        seen_tests.add(key)
        rows.append(f"- CarWale real-world test ({v.get('fuel_type')} {v.get('transmission')}, e.g. {v['name']}): "
                    f"city {v.get('mileage_city_tested') or 'n/a'}, highway {v.get('mileage_highway_tested') or 'n/a'} {v.get('mileage_unit') or 'kmpl'}")
    evs = [v for v in vars_ if v.get("range_km")]
    for v in evs[:6]:
        rows.append(f"- {v['name']}: claimed range {v['range_km']:.0f} km, battery {v.get('battery_kwh') or 'n/a'} kWh"
                    + (f", DC fast charge {v['dc_charge_minutes']:.0f} min" if v.get("dc_charge_minutes") else ""))
    if rows:
        ch.add(m, "mileage", "Mileage / range (claimed vs real-world)", "\n".join(rows))

    # 6. safety ----------------------------------------------------------------------------
    tree = (m.get("top_variant_features") or {}).get("groups") or {}
    safety_lines = []
    if m.get("ncap_rating"):
        safety_lines.append(f"Crash-test rating: {m['ncap_rating']} stars ({m.get('ncap_agency') or 'NCAP'}).")
    else:
        safety_lines.append("No NCAP crash-test rating is listed on CarWale.")
    ab = sorted({v["airbags"] for v in vars_ if v.get("airbags")})
    if ab:
        safety_lines.append(f"Airbags across variants: {', '.join(map(str, ab))}.")
    for g, items in tree.items():
        if g.lower().startswith("safety"):
            yes_items = [f"{i['name']}" + (f" ({i['value']})" if i["value"] not in ("Yes",) else "") for i in items if i["value"] not in ("No",)]
            no_items = [i["name"] for i in items if i["value"] == "No"]
            if yes_items:
                safety_lines.append(f"{g.split('>')[-1].strip()} on {(m.get('top_variant_features') or {}).get('version') or 'top variant'}: {'; '.join(yes_items)}.")
            if no_items:
                safety_lines.append(f"Not available: {', '.join(no_items)}.")
    ch.add(m, "safety", "Safety", "\n".join(safety_lines))

    # 7. top-variant features & specs -------------------------------------------------------
    top_name = (m.get("top_variant_features") or {}).get("version") or "top variant"
    feat_blocks, spec_blocks = [], []
    for g, items in tree.items():
        if g.lower().startswith("safety"):
            continue
        yes_items = [i["name"] + ("" if i["value"] == "Yes" else f": {i['value']}") for i in items if i["value"] != "No"]
        no_items = [i["name"] for i in items if i["value"] == "No"]
        line = f"{g.split('>')[-1].strip()}: {'; '.join(yes_items)}" + (f". Not available: {', '.join(no_items)}" if no_items else "")
        (spec_blocks if g.lower().startswith("specification") else feat_blocks).append(line)
    if feat_blocks:
        ch.add(m, "features", f"Features of {top_name}", "\n".join(feat_blocks))
    if spec_blocks:
        ch.add(m, "specs", f"Specifications of {top_name}", "\n".join(spec_blocks))

    # 8. service & ownership ---------------------------------------------------------------
    after_sales = [r["params"].get("After Sales Cost and Experience") for r in m.get("owner_reviews") or []]
    after_sales = [a for a in after_sales if a]
    svc = [f"Warranty: {'; '.join(m['warranty'])}." if m.get("warranty") else "Warranty details are not listed on CarWale for this model."]
    if after_sales:
        svc.append(f"Owners' rating of after-sales cost and experience (sample of {len(after_sales)} reviews): {', '.join(after_sales)}.")
    app = [i["name"] for g, items in tree.items() for i in items if "service" in i["name"].lower() and i["value"] != "No"]
    if app:
        svc.append(f"Service-related features: {', '.join(app)}.")
    reports = [a["title"] for a in m.get("expert_articles") or []
               if re.search(r"service|long.?term|ownership|maintenance|km report", a["title"], re.I)]
    if reports:
        svc.append(f"CarWale long-term ownership reports for this model: {'; '.join(reports[:5])}. "
                   "These describe real service visits and running experience.")
    svc.append("CarWale does not publish a scheduled service-cost table for this model; "
               "check with an authorised dealer for exact service costs.")
    ch.add(m, "service", "Service, warranty and ownership", " ".join(svc))

    # 9. FAQs, city prices, owner reviews, updates ----------------------------------------
    for f in m.get("faqs") or []:
        if f.get("q") and f.get("a"):
            ch.add(m, "faq", f"FAQ ({f.get('category')})", f"Q: {f['q']}\nA: {f['a']}")
    if m.get("city_prices"):
        ch.add(m, "city_prices", "On-road prices by city",
               "\n".join(f"- {c['city']}: Rs. {c['range']} (on-road)" for c in m["city_prices"] if c.get("range")))
    if m.get("rating_breakdown") or m.get("review_summary"):
        rb = ", ".join(f"{k} {v}/5" for k, v in (m.get("rating_breakdown") or {}).items())
        body = (f"Owner ratings: overall {m.get('user_rating') or 'n/a'}/5 ({m.get('user_rating_count')} ratings). {rb}.\n"
                + (f"{m['review_positive_pct']}% of reviews are positive.\n" if m.get("review_positive_pct") else "")
                + " ".join(m.get("review_summary") or []))
        ch.add(m, "owner_reviews", "Owner review summary", body)
    for r in m.get("owner_reviews") or []:
        params = ", ".join(f"{k}: {v}" for k, v in r["params"].items() if v not in (None, ""))
        meta = ", ".join(x for x in (r.get("version"), f"reports {r['mileage']:g} kmpl" if r.get("mileage") else None,
                                     r.get("city")) if x)
        body = (f"{'Critical owner review' if r['kind'] == 'critical' else 'Owner review'} ({r.get('rating') or 'n/a'}/5, "
                f"{r.get('date') or ''}{'; ' + meta if meta else ''}): {r['title']}\n{r['text']}\nRatings: {params}")
        ch.add(m, "owner_reviews", "Owner review", body[: MAX_CHARS])
    if m.get("latest_updates"):
        ch.add(m, "news", "Latest updates", "\n".join(f"- {u['date']}: {u['title']}" for u in m["latest_updates"]))


def glossary_chunks(ch: Chunker) -> None:
    if not GLOSSARY.exists():
        return
    for block in re.split(r"\n(?=## )", GLOSSARY.read_text(encoding="utf-8")):
        block = block.strip()
        if not block.startswith("## "):
            continue
        title, _, body = block[3:].partition("\n")
        ch.add(None, "glossary", title.strip(), body.strip())


def main() -> None:
    models = read_jsonl(PROCESSED / "models.jsonl")
    variants = read_jsonl(PROCESSED / "variants.jsonl")
    by_model = defaultdict(list)
    for v in variants:
        by_model[v["model_id"]].append(v)
    ch = Chunker()
    for m in models:
        if m["status"] == "discontinued":
            continue
        model_chunks(ch, m, by_model.get(m["id"], []))
    glossary_chunks(ch)
    n = write_jsonl(PROCESSED / "chunks.jsonl", ch.rows)
    print(f"wrote {n} chunks")
    print("by section:", dict(Counter(r["section"] for r in ch.rows).most_common()))
    lens = sorted(len(r["content"]) for r in ch.rows)
    print(f"chars: median {lens[len(lens) // 2]}, p95 {lens[int(len(lens) * .95)]}, max {lens[-1]}")


if __name__ == "__main__":
    main()
