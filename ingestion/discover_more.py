"""Second-pass discovery for richer data.

  python -m ingestion.discover_more 1   -> data/urls_stage1.json : every trim page + owner-review pages (1-3 per model)
  python -m ingestion.discover_more 2   -> data/urls_stage2.json : expert articles (long-term reports, first drives, ...)
                                           listed on the cached review pages / model pages

Only robots.txt-allowed paths; already-cached pages are skipped by the fetcher.
"""
from __future__ import annotations

import json
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


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    rp = robots()
    urls = stage1() if stage == "1" else stage2()
    urls = [u for u in dict.fromkeys(urls) if rp.can_fetch(HEADERS["User-Agent"], u)]
    out = DATA / f"urls_stage{stage}.json"
    out.write_text(json.dumps(urls, indent=1), encoding="utf-8")
    print(f"stage {stage}: {len(urls)} urls -> {out}")


if __name__ == "__main__":
    main()
