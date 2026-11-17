"""Second-pass discovery for richer data.

  python -m ingestion.discover_more 1   -> data/urls_stage1.json : every trim page + owner-review pages (1-3 per model)
  python -m ingestion.discover_more 2   -> data/urls_stage2.json : expert articles (long-term reports, first drives, ...)
                                           listed on the cached review pages / model pages

Only robots.txt-allowed paths; already-cached pages are skipped by the fetcher.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.robotparser

import httpx

from .common import BASE, DATA, HEADERS, PROCESSED, extract_state, read_cached

MAX_REVIEW_PAGES = 3


def robots() -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(httpx.get(f"{BASE}/robots.txt", headers=HEADERS, timeout=30).text.splitlines())
    return rp


def launched_models() -> list[dict]:
    return [m for m in map(json.loads, (PROCESSED / "models.jsonl").open(encoding="utf-8")) if m["status"] == "launched"]


def review_urls(m: dict) -> list[str]:
    n = m.get("user_review_count") or 0
    if n <= 0:
        return []
    pages = min(MAX_REVIEW_PAGES, 1 + (n > 10) + (n > 20))
    base = f"{m['url']}reviews/"
    return [base] + [f"{base}page/{p}/" for p in range(2, pages + 1)]


def stage1() -> list[str]:
    urls = []
    for m in launched_models():
        urls += [t["url"] for t in m["trims"]]
        urls += review_urls(m)
    return urls


def stage2() -> list[str]:
    urls = []
    for m in launched_models():
        main_review = None
        html = read_cached(m["url"])
        mp = (extract_state(html) or {}).get("modelPage") or {} if html else {}
        main_review = ((mp.get("carWaleTake") or {}).get("reviewArticlePath") or "").strip()
        listed = [e.get("url") for e in (mp.get("expertReviews") or [])]
        rhtml = read_cached(f"{m['url']}reviews/")
        rl = (extract_state(rhtml) or {}).get("userReviewListing") or {} if rhtml else {}
        listed += [e.get("url") for e in (rl.get("expertReviewListing") or [])]
        for path in dict.fromkeys(p for p in listed if p):
            full = path if path.startswith("http") else BASE + path
            if main_review and main_review.rstrip("/") in full:
                continue  # already captured as the model page's expert review
            urls.append(full)
    return urls


CITY_PAGES = ("mumbai", "delhi")
LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")


def sitemap(name: str) -> list[str]:
    xml = httpx.get(f"{BASE}/sitemap/{name}", headers=HEADERS, timeout=60).text
    return [u for u in LOC_RE.findall(xml) if "/hi/" not in u]


def rival_compare_urls(models: list[dict], rivals_per_model: int = 4) -> list[str]:
    """CarWale comparison pages for each model's own listed alternatives (popular pairs carry written verdicts)."""
    allm = [json.loads(l) for l in (PROCESSED / "models.jsonl").open(encoding="utf-8")]
    slug2id = {f"{m['brand_slug']}-{m['slug']}": m["id"] for m in allm}
    pairs = {}
    for u in sitemap("cw-compare.xml"):
        s = u.rstrip("/").split("/")[-1]
        if "-vs-" in s:
            a, b = s.split("-vs-", 1)
            if a in slug2id and b in slug2id:
                pairs[frozenset((slug2id[a], slug2id[b]))] = u
    launched = {m["id"] for m in models}
    out = []
    for m in sorted(models, key=lambda m: -(m.get("user_rating_count") or 0)):
        for r in (m.get("similar") or [])[:rivals_per_model]:
            key = frozenset((m["id"], r))
            if r in launched and key in pairs and pairs[key] not in out:
                out.append(pairs[key])
    return out


def stage3() -> list[str]:
    models = launched_models()
    urls = [f"{m['url']}price-in-{c}/" for m in models for c in CITY_PAGES]  # per-variant on-road breakup
    urls += [f"{m['url']}news/" for m in models]                              # latest headlines
    urls += sitemap("cw-long-tail.xml")                                       # CarWale "best cars ..." lists
    urls += rival_compare_urls(models)                                        # rival comparisons
    return urls


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    rp = robots()
    urls = {"1": stage1, "2": stage2, "3": stage3}[stage]()
    urls = [u for u in dict.fromkeys(urls) if rp.can_fetch(HEADERS["User-Agent"], u)]
    out = DATA / f"urls_stage{stage}.json"
    out.write_text(json.dumps(urls, indent=1), encoding="utf-8")
    print(f"stage {stage}: {len(urls)} urls -> {out}")


if __name__ == "__main__":
    main()
