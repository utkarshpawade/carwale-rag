"""Hybrid retrieval in one SQL query: pgvector cosine KNN + Postgres full-text search, fused with RRF.

Metadata filters (model_id, section) are applied *before* ranking, so a question about one car only
ranks that car's chunks. RRF score = sum over lists of 1 / (60 + rank).
"""
from __future__ import annotations

import re

import numpy as np

from ..db import fetch

STOP = {"what", "which", "is", "the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "does", "do", "it", "its",
        "me", "show", "tell", "about", "how", "much", "are", "with", "car", "cars", "vs", "versus", "best", "good",
        "get", "has", "have", "i", "my", "should", "can", "you", "please", "under", "lakh", "lakhs", "rs", "this",
        "that", "these", "those", "there", "any", "between", "than", "more", "less", "give"}


def tsquery_from(keywords: list[str], query: str, limit: int = 12) -> str | None:
    words: list[str] = []
    for text in list(keywords) + [query]:
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if len(w) >= 2 and w not in STOP and w not in words:
                words.append(w)
    return " | ".join(words[:limit]) or None


async def pinned_chunks(model_ids: list[str], sections: list[str]) -> list[dict]:
    """The canonical summary chunk per (car, aspect) - e.g. the 'mileage' chunk for a mileage question."""
    if not model_ids or not sections:
        return []
    return await fetch("""
        SELECT DISTINCT ON (model_id, section) id, model_id, section, title, content, url,
               1.0::float AS score, ARRAY['pinned'] AS via
        FROM chunks WHERE model_id = ANY(%s) AND section = ANY(%s)
        ORDER BY model_id, section, id""", (model_ids, sections))


async def brand_chunks(brand_slug: str, qvec, limit: int = 2) -> list[dict]:
    """A brand's own chunks (overview, discontinued models, FAQs), closest to the question first."""
    return await fetch("""
        SELECT id, model_id, section, title, content, url, 1.0::float AS score, ARRAY['pinned-brand'] AS via
        FROM chunks WHERE section = 'brand' AND id LIKE %s
        ORDER BY embedding <=> %s LIMIT %s""", (f"brand-{brand_slug}::%", np.asarray(qvec, dtype=np.float32), limit))


async def titled_chunks(model_id: str, section: str, title_like: str, limit: int = 2) -> list[dict]:
    """Chunks of one section whose title matches, e.g. the 'X vs Y' verdict or 'On-road price in Mumbai'."""
    return await fetch("""
        SELECT id, model_id, section, title, content, url, 1.0::float AS score, ARRAY['pinned'] AS via
        FROM chunks WHERE model_id = %s AND section = %s AND title ILIKE %s ORDER BY id LIMIT %s""",
                       (model_id, section, f"%{title_like}%", limit))


async def trim_chunks(model_id: str, trims: list[str]) -> list[dict]:
    """The per-trim chunks (versions, key features, CarWale's 'adds over' diff) for trims the user named."""
    if not trims:
        return []
    return await fetch("""
        SELECT id, model_id, section, title, content, url, 1.0::float AS score, ARRAY['pinned-trim'] AS via
        FROM chunks WHERE model_id = %s AND section = 'trim' AND title = ANY(%s) ORDER BY id""",
                       (model_id, [f"{t} trim" for t in trims]))


async def hybrid_search(qvec, query: str, keywords: list[str], model_ids: list[str] | None = None,
                        sections: list[str] | None = None, k: int = 6, per_model_cap: int | None = None) -> list[dict]:
    filt, p = [], {"qv": np.asarray(qvec, dtype=np.float32), "k": k * 3 if per_model_cap else k}
    if model_ids:
        filt.append("model_id = ANY(%(mids)s)"); p["mids"] = model_ids
    else:
        filt.append("(status IS NULL OR status <> 'discontinued')")
    if sections:
        filt.append("section = ANY(%(secs)s)"); p["secs"] = sections
    where = " AND ".join(filt)
    tsq = tsquery_from(keywords, query)
    kw_cte, kw_union = "", ""
    if tsq:
        p["tsq"] = tsq
        kw_cte = f""",
    kw AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
        FROM chunks, to_tsquery('english', %(tsq)s) q
        WHERE tsv @@ q AND {where}
        ORDER BY ts_rank_cd(tsv, q) DESC LIMIT 30)"""
        kw_union = " UNION ALL SELECT id, r, 'keyword' AS src FROM kw"
    sql = f"""
    WITH vec AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(qv)s) AS r
        FROM chunks WHERE {where}
        ORDER BY embedding <=> %(qv)s LIMIT 30){kw_cte},
    fused AS (
        SELECT id, SUM(1.0 / (60 + r)) AS score, ARRAY_AGG(src || '#' || r) AS via
        FROM (SELECT id, r, 'vector' AS src FROM vec{kw_union}) u GROUP BY id)
    SELECT c.id, c.model_id, c.section, c.title, c.content, c.url, f.score::float AS score, f.via
    FROM fused f JOIN chunks c USING (id)
    ORDER BY f.score DESC LIMIT %(k)s"""
    rows = await fetch(sql, p)
    if per_model_cap:
        seen: dict[str | None, int] = {}
        capped = []
        for r in rows:
            n = seen.get(r["model_id"], 0)
            if n < per_model_cap:
                capped.append(r)
                seen[r["model_id"]] = n + 1
        rows = capped[:k]
    return rows
