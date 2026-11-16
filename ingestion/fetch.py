"""Polite, resumable page fetcher with a gzip disk cache.

Usage:
  python -m ingestion.fetch models   # fetch every URL in data/urls_models.json
  python -m ingestion.fetch trims    # derive one top-trim URL per cached model page, then fetch

Etiquette: 2 concurrent requests, 0.6-1.2 s jittered pause after each request (~2 req/s overall),
exponential backoff on 429/5xx, robots.txt-allowed paths only, every page fetched once (cached).
"""
from __future__ import annotations

import asyncio
import gzip
import json
import random
import sys
import time

import httpx

from .common import DATA, HEADERS, MANIFEST, RAW, cache_path, extract_state, read_cached

CONCURRENCY = 2
MIN_DELAY, MAX_DELAY = 0.6, 1.2
RETRY_STATUS = {429, 500, 502, 503, 504}


def log_manifest(row: dict) -> None:
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


async def fetch_all(urls: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [u for u in urls if not cache_path(u).exists()]
    print(f"{len(urls)} urls, {len(urls) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    t0 = time.time()

    async with httpx.AsyncClient(headers=HEADERS, timeout=40, follow_redirects=True) as client:

        async def one(url: str) -> None:
            nonlocal done
            async with sem:
                status, final_url, size = None, url, 0
                for attempt in range(4):
                    try:
                        r = await client.get(url)
                        status, final_url = r.status_code, str(r.url)
                        if r.status_code == 200:
                            data = r.text
                            size = len(data)
                            with gzip.open(cache_path(url), "wt", encoding="utf-8") as f:
                                f.write(data)
                            break
                        if r.status_code not in RETRY_STATUS:
                            break
                        wait = float(r.headers.get("retry-after", 0) or 0) or (2 ** attempt) * 3
                        await asyncio.sleep(min(wait, 60))
                    except httpx.TransportError as e:
                        status = f"error:{type(e).__name__}"
                        await asyncio.sleep((2 ** attempt) * 3)
                    finally:
                        await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                log_manifest({"url": url, "status": status, "final_url": final_url, "bytes": size,
                              "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
                done += 1
                if done % 25 == 0 or done == len(todo):
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"  {done}/{len(todo)} fetched ({rate:.2f}/s) last={status} {url}", flush=True)

        await asyncio.gather(*(one(u) for u in todo))


def top_trim_urls(model_urls: list[str]) -> list[str]:
    """For each cached model page, pick the trim that contains the top variant (richest feature tree)."""
    out = []
    for mu in model_urls:
        html = read_cached(mu)
        state = extract_state(html) if html else None
        mp = (state or {}).get("modelPage") or {}
        trims = mp.get("trimSummary") or []
        if not trims:
            continue
        top_vid = (mp.get("modelDetails") or {}).get("topVariant")
        chosen = next((t for t in trims if top_vid in (t.get("versionIds") or [])), None)
        chosen = chosen or max(trims, key=lambda t: t.get("sortOrder", 0))
        slug = chosen.get("trimMaskingName")
        if slug:
            out.append(f"{mu}{slug}/")
    return out


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "models"
    model_urls = json.loads((DATA / "urls_models.json").read_text(encoding="utf-8"))
    if what == "models":
        urls = model_urls
    elif what == "trims":
        urls = top_trim_urls(model_urls)
        (DATA / "urls_trims.json").write_text(json.dumps(urls, indent=1), encoding="utf-8")
    else:
        raise SystemExit("usage: python -m ingestion.fetch [models|trims]")
    asyncio.run(fetch_all(urls))


if __name__ == "__main__":
    main()
