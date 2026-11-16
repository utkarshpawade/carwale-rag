"""Build the long-lived singletons (DB pool, resolver, embedder, LLM client, pipeline)."""
from __future__ import annotations

import asyncio
import logging

from . import db
from .config import Settings
from .embeddings import Embedder
from .llm.client import LLMClient
from .rag.entities import CarResolver
from .rag.orchestrator import Pipeline

log = logging.getLogger("carwale")


async def build_pipeline(s: Settings) -> Pipeline:
    await db.init_pool(s.database_url)
    models = await db.fetch("SELECT id, brand, brand_slug, name, slug, full_name, status, user_rating_count FROM models")
    resolver = CarResolver(models)
    stats = await db.fetchrow("""
        SELECT (SELECT COUNT(*) FROM models WHERE status = 'launched') AS models,
               (SELECT COUNT(*) FROM models WHERE status = 'upcoming') AS upcoming,
               (SELECT COUNT(*) FROM variants v JOIN models m ON m.id = v.model_id WHERE m.status = 'launched') AS variants,
               (SELECT COUNT(*) FROM chunks) AS chunks,
               (SELECT value FROM meta WHERE key = 'data_snapshot_date') AS snapshot""") or {}
    embedder = await asyncio.to_thread(Embedder, s.embed_model, s.embed_cache_dir, s.embed_threads)
    await asyncio.to_thread(embedder.embed_query, "warm up")
    log.info("pipeline ready: %s", dict(stats))
    return Pipeline(s, LLMClient(s), resolver, embedder, dict(stats))
