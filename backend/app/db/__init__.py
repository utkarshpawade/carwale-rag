"""Async Postgres pool (psycopg 3) with pgvector support. Works with Neon's pooled (PgBouncer) endpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_pool: AsyncConnectionPool | None = None


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)


async def init_pool(url: str) -> None:
    global _pool
    _pool = AsyncConnectionPool(
        url,
        min_size=1,
        max_size=5,
        open=False,
        configure=_configure,
        # prepare_threshold=None: no server-side prepared statements (PgBouncer transaction mode)
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
        check=AsyncConnectionPool.check_connection,
    )
    await _pool.open(wait=True, timeout=60)


async def close_pool() -> None:
    if _pool:
        await _pool.close()


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised")
    return _pool


async def fetch(sql: str, params: Any = None) -> list[dict]:
    async with pool().connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall() if cur.description else []


async def fetchrow(sql: str, params: Any = None) -> dict | None:
    rows = await fetch(sql, params)
    return rows[0] if rows else None


async def execute(sql: str, params: Any = None) -> None:
    async with pool().connection() as conn:
        await conn.execute(sql, params)
