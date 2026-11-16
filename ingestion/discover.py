"""Discover CarWale model page URLs from the public sitemap (English pages only).

Usage: python -m ingestion.discover
Writes data/urls_models.json
"""
from __future__ import annotations

import json
import re
import urllib.robotparser

import httpx

from .common import BASE, DATA, HEADERS, MODEL_SITEMAP, MODEL_URL_RE

LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")


def robots() -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser()
    with httpx.Client(headers=HEADERS, timeout=30) as c:
        rp.parse(c.get(f"{BASE}/robots.txt").text.splitlines())
    return rp


def main() -> None:
    rp = robots()
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as c:
        xml = c.get(MODEL_SITEMAP).text
    locs = LOC_RE.findall(xml)
    urls, seen = [], set()
    for u in locs:
        if "/hi/" in u or not MODEL_URL_RE.match(u) or u in seen:
            continue
        if not rp.can_fetch(HEADERS["User-Agent"], u):
            continue
        seen.add(u)
        urls.append(u)
    out = DATA / "urls_models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(urls, indent=1), encoding="utf-8")
    brands = sorted({MODEL_URL_RE.match(u).group(1) for u in urls})
    print(f"sitemap <loc> entries: {len(locs)} | English model URLs: {len(urls)} | brands: {len(brands)}")
    print("brands:", ", ".join(brands))


if __name__ == "__main__":
    main()
