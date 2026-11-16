"""POST /api/chat -> Server-Sent Events stream of the RAG pipeline."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import get_settings
from ..memory import store

router = APIRouter()
log = logging.getLogger("carwale.chat")


class ChatIn(BaseModel):
    user_id: str = Field(min_length=6, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


class RateLimiter:
    """In-memory sliding window per client IP (protects the shared Groq free-tier quota)."""

    def __init__(self):
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str, per_min: int, per_day: int) -> tuple[bool, int]:
        now = time.time()
        dq = self.hits[key]
        while dq and now - dq[0] > 86400:
            dq.popleft()
        recent = [t for t in dq if now - t < 60]
        if len(recent) >= per_min:
            return False, int(60 - (now - recent[0])) + 1
        if len(dq) >= per_day:
            return False, int(86400 - (now - dq[0])) + 1
        dq.append(now)
        return True, 0


limiter = RateLimiter()


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


@router.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    s = get_settings()
    ok, wait = limiter.allow(client_ip(request), s.rate_limit_per_min, s.rate_limit_per_day)
    if not ok:
        return JSONResponse({"error": f"Too many requests. Please retry in {wait} seconds."}, status_code=429)
    message = body.message.strip()[: s.max_input_chars]
    conv_id = body.conversation_id
    if conv_id:
        try:
            conv_id = str(uuid.UUID(conv_id))
        except ValueError:
            return JSONResponse({"error": "invalid conversation_id"}, status_code=400)
        owner = await store.conversation_owner(conv_id)
        if owner and owner != body.user_id:
            return JSONResponse({"error": "conversation belongs to another user"}, status_code=403)
    else:
        conv_id = str(uuid.uuid4())
    pipeline = request.app.state.pipeline

    async def gen():
        yield sse("conversation", {"conversation_id": conv_id})
        try:
            async for ev in pipeline.run(conv_id, body.user_id, message):
                yield sse(ev.type, ev.data)
        except Exception:
            log.exception("chat turn failed")
            yield sse("error", {"message": "Something went wrong while answering. Please try again."})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
