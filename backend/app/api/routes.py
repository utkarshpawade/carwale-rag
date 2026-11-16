"""Health, metadata, conversation history and feedback endpoints."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..memory import store
from ..rag.followups import STARTERS

router = APIRouter()


def _uuid(v: str) -> str:
    try:
        return str(uuid.UUID(v))
    except ValueError:
        raise HTTPException(400, "invalid conversation id")


@router.get("/api/health")
async def health(request: Request):
    stats = getattr(request.app.state, "stats", {}) or {}
    return {"status": "ok", "models": stats.get("models"), "variants": stats.get("variants"),
            "chunks": stats.get("chunks"), "data_snapshot": stats.get("snapshot")}


@router.get("/api/meta")
async def meta(request: Request):
    stats = getattr(request.app.state, "stats", {}) or {}
    return {"stats": stats, "starters": ["Smart hybrid cars under 30 lakhs", "Creta vs Seltos — which is better?",
                                         "Mileage of Maruti Swift", "Safest SUVs under 15 lakh",
                                         "Does the Tata Nexon have a sunroof?", "Warranty and service of Hyundai Venue"],
            "fallback_starters": STARTERS}


@router.get("/api/conversations")
async def conversations(user_id: str):
    return {"conversations": await store.list_conversations(user_id)}


@router.get("/api/conversations/{conv_id}")
async def conversation(conv_id: str, user_id: str):
    conv_id = _uuid(conv_id)
    owner = await store.conversation_owner(conv_id)
    if owner is None:
        raise HTTPException(404, "not found")
    if owner != user_id:
        raise HTTPException(403, "forbidden")
    msgs = await store.get_messages(conv_id)
    return {"conversation_id": conv_id, "messages": [
        {"id": m["id"], "role": m["role"], "content": m["content"], "created_at": m["created_at"],
         "cars": m["meta"].get("cars"), "sources": m["meta"].get("sources"), "followups": m["meta"].get("followups"),
         "trace": m["meta"].get("trace")} for m in msgs]}


@router.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, user_id: str):
    if not await store.delete_conversation(_uuid(conv_id), user_id):
        raise HTTPException(404, "not found")
    return {"deleted": True}


class FeedbackIn(BaseModel):
    message_id: int
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=1000)


@router.post("/api/feedback")
async def feedback(body: FeedbackIn):
    await store.add_feedback(body.message_id, body.rating, body.comment)
    return {"ok": True}
