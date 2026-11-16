"""Conversation memory in Postgres: persistent history + working state (summary, active cars, filters, prefs)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..db import execute, fetch, fetchrow


@dataclass
class ConvState:
    summary: str = ""
    summarized_upto: int = 0
    active_cars: list[str] = field(default_factory=list)
    focus_car: str | None = None
    filters: dict = field(default_factory=dict)
    prefs: dict = field(default_factory=dict)


def _j(v, default):
    if v is None:
        return default
    return json.loads(v) if isinstance(v, str) else v


async def ensure_conversation(conv_id: str, user_id: str, first_message: str) -> None:
    title = first_message.strip().replace("\n", " ")[:70]
    await execute("""INSERT INTO conversations (id, user_id, title) VALUES (%s, %s, %s)
                     ON CONFLICT (id) DO UPDATE SET updated_at = now()""", (conv_id, user_id, title))


async def conversation_owner(conv_id: str) -> str | None:
    row = await fetchrow("SELECT user_id FROM conversations WHERE id = %s", (conv_id,))
    return row["user_id"] if row else None


async def add_message(conv_id: str, role: str, content: str, meta: dict) -> int:
    row = await fetchrow("INSERT INTO messages (conversation_id, role, content, meta) VALUES (%s, %s, %s, %s) RETURNING id",
                         (conv_id, role, content, json.dumps(meta, default=str)))
    return row["id"]


async def recent_messages(conv_id: str, limit: int) -> list[dict]:
    rows = await fetch("SELECT id, role, content, meta FROM messages WHERE conversation_id = %s ORDER BY id DESC LIMIT %s",
                       (conv_id, limit))
    for r in rows:
        r["meta"] = _j(r["meta"], {})
    return list(reversed(rows))


async def load_state(conv_id: str) -> ConvState:
    row = await fetchrow("SELECT * FROM conversation_state WHERE conversation_id = %s", (conv_id,))
    if not row:
        return ConvState()
    return ConvState(summary=row["summary"] or "", summarized_upto=row["summarized_upto"] or 0,
                     active_cars=_j(row["active_cars"], []), focus_car=row["focus_car"],
                     filters=_j(row["filters"], {}), prefs=_j(row["prefs"], {}))


async def save_state(conv_id: str, s: ConvState) -> None:
    await execute("""
        INSERT INTO conversation_state (conversation_id, summary, summarized_upto, active_cars, focus_car, filters, prefs, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (conversation_id) DO UPDATE SET summary = EXCLUDED.summary, summarized_upto = EXCLUDED.summarized_upto,
            active_cars = EXCLUDED.active_cars, focus_car = EXCLUDED.focus_car, filters = EXCLUDED.filters,
            prefs = EXCLUDED.prefs, updated_at = now()""",
        (conv_id, s.summary, s.summarized_upto, json.dumps(s.active_cars), s.focus_car, json.dumps(s.filters),
         json.dumps(s.prefs)))


async def list_conversations(user_id: str, limit: int = 30) -> list[dict]:
    return await fetch("""SELECT c.id::text AS id, c.title, c.updated_at,
                                 (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n_messages
                          FROM conversations c WHERE c.user_id = %s ORDER BY c.updated_at DESC LIMIT %s""", (user_id, limit))


async def get_messages(conv_id: str) -> list[dict]:
    rows = await fetch("SELECT id, role, content, meta, created_at FROM messages WHERE conversation_id = %s ORDER BY id",
                       (conv_id,))
    for r in rows:
        r["meta"] = _j(r["meta"], {})
    return rows


async def delete_conversation(conv_id: str, user_id: str) -> bool:
    row = await fetchrow("DELETE FROM conversations WHERE id = %s AND user_id = %s RETURNING id", (conv_id, user_id))
    return row is not None


async def messages_after(conv_id: str, after_id: int) -> list[dict]:
    return await fetch("SELECT id, role, content FROM messages WHERE conversation_id = %s AND id > %s ORDER BY id",
                       (conv_id, after_id))


async def cache_get(key: str) -> dict | None:
    row = await fetchrow("SELECT payload FROM answer_cache WHERE key = %s AND created_at > now() - interval '7 days'", (key,))
    return _j(row["payload"], None) if row else None


async def cache_put(key: str, payload: dict) -> None:
    await execute("""INSERT INTO answer_cache (key, payload) VALUES (%s, %s)
                     ON CONFLICT (key) DO UPDATE SET payload = EXCLUDED.payload, created_at = now()""",
                  (key, json.dumps(payload, default=str)))


async def add_feedback(message_id: int, rating: int, comment: str | None) -> None:
    await execute("INSERT INTO feedback (message_id, rating, comment) VALUES (%s, %s, %s)", (message_id, rating, comment))
