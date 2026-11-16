"""Shared paths, HTTP settings and HTML/state helpers for the ingestion pipeline."""
from __future__ import annotations

import gzip
import hashlib
import html as html_lib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MANIFEST = RAW / "manifest.jsonl"

BASE = "https://www.carwale.com"
SITEMAP_INDEX = f"{BASE}/sitemap/cw-index.xml"
MODEL_SITEMAP = f"{BASE}/sitemap/cw-model.xml"
IMG_CDN = "https://imgd.aeplcdn.com/664x374"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}

MODEL_URL_RE = re.compile(r"^https://www\.carwale\.com/([a-z0-9-]+)-cars/([a-z0-9-]+)/$")
_STATE_RE = re.compile(r'<script type="application/json" id="initial-state">(.*?)</script>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def cache_path(url: str) -> Path:
    return RAW / (hashlib.sha1(url.encode()).hexdigest() + ".html.gz")


def read_cached(url: str) -> str | None:
    p = cache_path(url)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return f.read()


def extract_state(html: str) -> dict | None:
    m = _STATE_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def clean_text(s: str | None) -> str:
    """HTML fragment -> readable plain text (keeps list items and paragraphs as lines)."""
    if not s:
        return ""
    s = str(s)
    s = re.sub(r"(?i)<\s*(br|/p|/li|/h[1-6]|/div|/tr)\s*/?>", "\n", s)
    s = re.sub(r"(?i)<\s*li[^>]*>", "- ", s)
    s = _TAG_RE.sub("", s)
    s = html_lib.unescape(s)
    s = s.replace("�", "'").replace("\xa0", " ")
    s = _WS_RE.sub(" ", s)
    lines = [ln.strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def list_items(s: str | None) -> list[str]:
    """Extract <li> items from an HTML fragment (fallback: non-empty lines)."""
    if not s:
        return []
    items = re.findall(r"(?is)<li[^>]*>(.*?)</li>", str(s))
    if items:
        return [t for t in (clean_text(i) for i in items) if t]
    return [ln.lstrip("- ").strip() for ln in clean_text(s).split("\n") if ln.strip()]


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
