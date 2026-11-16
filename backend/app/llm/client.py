"""Groq client wrapper: per-role model fallback chains, 429/retry-after handling, usage accounting.

Free-tier limits are per model (TPM is the binding one), so spreading load across a chain of models
multiplies capacity. A model that returns 429 is put on cooldown for its retry-after window.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, TypeVar

import groq
from pydantic import BaseModel, ValidationError

from ..config import Settings

T = TypeVar("T", bound=BaseModel)

STRICT_SCHEMA_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "meta-models/muse-spark-open")


class LLMUnavailable(Exception):
    def __init__(self, msg: str, retry_after: float | None = None, auth: bool = False):
        super().__init__(msg)
        self.retry_after = retry_after
        self.auth = auth


@dataclass
class CallInfo:
    role: str
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    attempts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"role": self.role, "model": self.model, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "latency_ms": self.latency_ms,
                "fallbacks": self.attempts}


def _model_kwargs(model: str) -> dict:
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False}
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}
    return {}


def _retry_after(e: Exception) -> float:
    try:
        return float(e.response.headers.get("retry-after", 10))  # type: ignore[attr-defined]
    except Exception:
        return 10.0


class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = groq.AsyncGroq(api_key=settings.groq_api_key or "missing", max_retries=0, timeout=25.0)
        self.cooldown_until: dict[str, float] = {}
        self.forced_fail = {m.strip() for m in settings.force_fail_models.split(",") if m.strip()}

    def _available(self, role: str) -> list[str]:
        now = time.time()
        return [m for m in self.s.chain(role) if self.cooldown_until.get(m, 0) <= now]

    def _cool(self, model: str, seconds: float) -> None:
        self.cooldown_until[model] = time.time() + min(max(seconds, 2.0), 120.0)

    async def json(self, role: str, messages: list[dict], schema_model: type[T], json_schema: dict,
                   max_tokens: int = 700) -> tuple[T, CallInfo]:
        """Structured output: strict json_schema where supported, else json_object + Pydantic validation."""
        info = CallInfo(role)
        t0 = time.time()
        last_retry = None
        for model in self._available(role):
            info.attempts.append(model)
            if model in self.forced_fail:
                self._cool(model, 30)
                continue
            if model in STRICT_SCHEMA_MODELS:
                rf = {"type": "json_schema", "json_schema": {"name": schema_model.__name__, "strict": True, "schema": json_schema}}
                msgs = messages
            else:
                rf = {"type": "json_object"}
                msgs = messages + [{"role": "system", "content":
                                    "Respond with a single JSON object that matches this JSON schema exactly:\n"
                                    + json.dumps(json_schema, separators=(",", ":"))}]
            for attempt in range(2):  # one repair retry for non-strict models
                try:
                    resp = await self.client.chat.completions.create(
                        model=model, messages=msgs, temperature=0, max_completion_tokens=max_tokens,
                        response_format=rf, **_model_kwargs(model))
                    text = resp.choices[0].message.content or ""
                    obj = schema_model.model_validate_json(text)
                    info.model = model
                    if resp.usage:
                        info.prompt_tokens, info.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
                    info.latency_ms = int((time.time() - t0) * 1000)
                    return obj, info
                except groq.RateLimitError as e:
                    last_retry = _retry_after(e)
                    self._cool(model, last_retry)
                    break
                except (ValidationError, json.JSONDecodeError) as e:
                    if rf["type"] == "json_schema" or attempt == 1:
                        break
                    msgs = msgs + [{"role": "user", "content": f"Your JSON was invalid ({str(e)[:200]}). Return corrected JSON only."}]
                except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
                    raise LLMUnavailable(f"Groq API key missing or invalid: {e}", auth=True)
                except (groq.APIStatusError, groq.APIConnectionError, groq.APITimeoutError):
                    break
        raise LLMUnavailable(f"no {role} model available", last_retry)

    async def stream(self, role: str, messages: list[dict], max_tokens: int, info: CallInfo) -> AsyncIterator[str]:
        """Stream text deltas; falls back to the next model only if nothing has been emitted yet."""
        t0 = time.time()
        last_retry = None
        for model in self._available(role):
            info.attempts.append(model)
            if model in self.forced_fail:
                self._cool(model, 30)
                continue
            emitted = False
            try:
                stream = await self.client.chat.completions.create(
                    model=model, messages=messages, temperature=0.3, max_completion_tokens=max_tokens,
                    stream=True, **_model_kwargs(model))
                async for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            emitted = True
                            yield delta
                    xg = getattr(chunk, "x_groq", None)
                    usage = getattr(xg, "usage", None) if xg else None
                    if usage:
                        info.prompt_tokens, info.completion_tokens = usage.prompt_tokens, usage.completion_tokens
                info.model = model
                info.latency_ms = int((time.time() - t0) * 1000)
                return
            except groq.RateLimitError as e:
                last_retry = _retry_after(e)
                self._cool(model, last_retry)
                if emitted:
                    raise LLMUnavailable("rate limited mid-stream", last_retry)
            except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
                raise LLMUnavailable(f"Groq API key missing or invalid: {e}", auth=True)
            except (groq.APIStatusError, groq.APIConnectionError, groq.APITimeoutError):
                if emitted:
                    raise LLMUnavailable("stream interrupted")
        raise LLMUnavailable(f"no {role} model available", last_retry)

    async def text(self, role: str, messages: list[dict], max_tokens: int = 300) -> tuple[str, CallInfo]:
        info = CallInfo(role)
        parts = [p async for p in self.stream(role, messages, max_tokens, info)]
        return "".join(parts), info


async def gather_safely(*aws):
    return await asyncio.gather(*aws, return_exceptions=True)
