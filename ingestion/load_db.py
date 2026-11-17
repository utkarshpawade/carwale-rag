"""Create the schema and load models, variants and embedded chunks into Postgres (local or Neon).

Usage: python -m ingestion.load_db            (uses DATABASE_URL from .env / env; use Neon's *direct* URL)
Incremental: chunks are re-embedded only when their content_hash changed; stale rows are deleted.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.embeddings import Embedder  # noqa: E402

from .common import PROCESSED, read_jsonl  # noqa: E402

SCHEMA = ROOT / "backend" / "app" / "db" / "schema.sql"

MODEL_COLS = ["id", "cw_model_id", "brand", "brand_slug", "name", "slug", "full_name", "url", "status", "launched_on",
              "body_type", "body_type_raw", "is_ev", "price_min", "price_max", "price_is_estimate", "fuel_types",
              "hybrid_types", "transmissions", "has_automatic", "seats", "ncap_rating", "ncap_agency", "airbags_max",
              "mileage_best", "range_best_km", "power_max_bhp", "boot_l", "ground_clearance_mm", "user_rating",
              "user_rating_count", "expert_rating_5", "feature_tags", "image_url"]
# bulky text lives in chunks; keep the structured remainder in models.data
DATA_DROP = {"owner_reviews", "expert_review", "synopsis", "expert_opinion", "faqs", "upcoming_mentions",
             "trim_features", "expert_articles", "news", "comparisons", "news_articles"}
VARIANT_COLS = ["version_id", "model_id", "name", "display_name", "trim", "url", "price", "fuel_type", "hybrid_type",
                "transmission", "is_automatic", "engine_cc", "power_bhp", "torque_nm", "mileage_arai", "mileage_unit",
                "mileage_owner", "mileage_expected", "mileage_effective", "mileage_basis", "mileage_city_tested", "mileage_highway_tested", "range_km", "battery_kwh",
                "dc_charge_minutes", "seats", "airbags", "ncap_rating", "ncap_agency", "boot_l", "ground_clearance_mm",
                "length_mm", "width_mm", "height_mm", "wheelbase_mm", "fuel_tank_l", "kerb_weight_kg", "drivetrain",
                "feature_tags"]
CHUNK_COLS = ["id", "model_id", "variant_id", "section", "title", "content", "url", "brand", "body_type", "fuel_types",
              "price_min", "price_max", "status", "content_hash"]


def upsert_sql(table: str, cols: list[str], key: str, extra: list[str] = ()) -> str:
    allc = cols + list(extra)
    ph = ", ".join(["%s"] * len(allc))
    upd = ", ".join(f"{c} = EXCLUDED.{c}" for c in allc if c != key)
    return f"INSERT INTO {table} ({', '.join(allc)}) VALUES ({ph}) ON CONFLICT ({key}) DO UPDATE SET {upd}"


def load_embedding_source() -> dict:
    """Optionally reuse vectors already computed in another database (EMBED_SOURCE_URL), keyed by content hash."""
    src = os.environ.get("EMBED_SOURCE_URL")
    if not src:
        return {}
    with psycopg.connect(src, connect_timeout=30) as sconn:
        register_vector(sconn)
        rows = sconn.execute("SELECT content_hash, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
    print(f"reusing {len(rows)} embeddings from EMBED_SOURCE_URL", flush=True)
    # pgvector >= 0.5 returns Vector objects; convert to float32 numpy arrays for re-insertion
    return {h: np.asarray(e.to_numpy() if hasattr(e, "to_numpy") else e, dtype=np.float32) for h, e in rows}


def main() -> None:
    url = os.environ.get("DATABASE_URL") or get_settings().database_url
    models = read_jsonl(PROCESSED / "models.jsonl")
    variants = read_jsonl(PROCESSED / "variants.jsonl")
    chunks = read_jsonl(PROCESSED / "chunks.jsonl")
    keep_ids = {m["id"] for m in models}
    t0 = time.time()
    print(f"connecting to {url.split('@')[-1]} ...", flush=True)
    with psycopg.connect(url, autocommit=False, connect_timeout=30) as conn:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))
        register_vector(conn)
        with conn.cursor() as cur:
            cur.executemany(
                upsert_sql("models", MODEL_COLS, "id", ["data"]),
                [[m.get(c) for c in MODEL_COLS] + [json.dumps({k: v for k, v in m.items() if k not in DATA_DROP})]
                 for m in models],
            )
            cur.execute("DELETE FROM models WHERE NOT (id = ANY(%s))", (list(keep_ids),))
            vids = [v["version_id"] for v in variants]
            cur.executemany(
                upsert_sql("variants", VARIANT_COLS, "version_id", ["specs"]),
                [[v.get(c) for c in VARIANT_COLS] + [json.dumps(v.get("specs") or {})] for v in variants],
            )
            cur.execute("DELETE FROM variants WHERE NOT (version_id = ANY(%s))", (vids,))
            conn.commit()
            print(f"models {len(models)}, variants {len(variants)} upserted ({time.time() - t0:.1f}s)", flush=True)

            cur.execute("SELECT id, content_hash FROM chunks WHERE embedding IS NOT NULL")
            existing = dict(cur.fetchall())
            todo = [c for c in chunks if existing.get(c["id"]) != c["content_hash"]]
            todo.sort(key=lambda c: len(c["content"]))  # similar lengths per batch -> less padding, faster embedding
            print(f"chunks: {len(chunks)} total, {len(todo)} new/changed -> embedding ...", flush=True)
            reuse = load_embedding_source()  # content_hash -> vector from another DB (e.g. local -> Neon)
            if todo:
                s = get_settings()
                emb = None
                B = 256
                for i in range(0, len(todo), B):
                    batch = todo[i:i + B]
                    missing = [c for c in batch if c["content_hash"] not in reuse]
                    if missing:
                        emb = emb or Embedder(s.embed_model, s.embed_cache_dir)
                        for c, v in zip(missing, emb.embed_documents([c["content"] for c in missing])):
                            reuse[c["content_hash"]] = np.asarray(v, dtype=np.float32)
                    vecs = [reuse[c["content_hash"]] for c in batch]
                    cur.executemany(
                        upsert_sql("chunks", CHUNK_COLS, "id", ["embedding"]),
                        [[c.get(k) for k in CHUNK_COLS] + [vec] for c, vec in zip(batch, vecs)],
                    )
                    conn.commit()
                    print(f"  embedded+upserted {min(i + B, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
            cur.execute("DELETE FROM chunks WHERE NOT (id = ANY(%s))", ([c["id"] for c in chunks],))
            snapshot = max((m.get("scraped_at") or "" for m in models), default="")
            cur.execute("INSERT INTO meta (key, value) VALUES ('data_snapshot_date', %s) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (snapshot,))
            conn.commit()
            cur.execute("ANALYZE models; ANALYZE variants; ANALYZE chunks;")
            conn.commit()
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
