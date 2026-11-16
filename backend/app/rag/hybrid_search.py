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
