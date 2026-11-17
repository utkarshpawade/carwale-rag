"""POST /api/chat -> Server-Sent Events stream of the RAG pipeline."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict, deque

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..config import get_settings
from ..memory import store

router = APIRouter()
log = logging.getLogger("carwale.chat")


class ChatIn(BaseModel):
    user_id: str = Field(min_length=6, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.replace("\x00", "")  # PostgreSQL TEXT cannot store NUL bytes
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class RateLimiter:
    """In-memory sliding window per client IP (protects the shared Groq free-tier quota).

    Memory is bounded: clients idle for a day are swept every few minutes, and at most `max_keys` clients are
    tracked (the least recently seen are dropped first)."""

    SWEEP_EVERY_S = 600

    def __init__(self, max_keys: int = 100_000):
        self.hits: OrderedDict[str, deque] = OrderedDict()
        self.max_keys = max_keys
        self._last_sweep = time.time()

    def _sweep(self, now: float) -> None:
        for k in [k for k, dq in self.hits.items() if not dq or now - dq[-1] > 86400]:
            del self.hits[k]
        self._last_sweep = now

    def allow(self, key: str, per_min: int, per_day: int) -> tuple[bool, int]:
        now = time.time()
        if now - self._last_sweep > self.SWEEP_EVERY_S:
            self._sweep(now)
        dq = self.hits.get(key)
        if dq is None:
            dq = self.hits[key] = deque()
            while len(self.hits) > self.max_keys:
                self.hits.popitem(last=False)
        self.hits.move_to_end(key)
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
