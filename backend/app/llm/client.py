"""Groq client wrapper: per-role model fallback chains, client-side token budgeting, 429 handling.

Free-tier limits are per model and per minute (verified from response headers: 8K TPM, 1K RPD for each
gpt-oss/qwen model). Groq reserves prompt + max_completion_tokens up front, so we:
  * keep max_completion_tokens tight,
  * read x-ratelimit-remaining-tokens / x-ratelimit-reset-tokens after every call and skip a model that
    cannot fit the next request (instead of eating a 429),
  * put a model on cooldown for its retry-after window when it does return 429,
  * spread roles across different models (planner on qwen, answers on gpt-oss-120b).
"""
from __future__ import annotations

import inspect
import json
import re
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
    attempts: list[str] = field(default_factory=list)  # models actually tried and why they failed, in order
    skipped: list[str] = field(default_factory=list)   # models skipped up front (cooldown / no token budget)

    def as_dict(self) -> dict:
        return {"role": self.role, "model": self.model, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "latency_ms": self.latency_ms,
                "fallbacks": self.attempts, "skipped": self.skipped}


def _status_reason(model: str, e: Exception) -> str:
    code = None
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        code = (body.get("error") or {}).get("code") if isinstance(body.get("error"), dict) else body.get("code")
    return f"{model}:http_{getattr(e, 'status_code', '?')}" + (f":{code}" if code else "")


def _model_kwargs(model: str) -> dict:
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False}
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}
    return {}


def _duration(s: str | None) -> float | None:
    """Groq reset format: '7.66s', '2m59.56s', '120ms', '1h2m'."""
    if not s:
        return None
    total, found = 0.0, False
    for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", s):
        found = True
        total += float(value) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None


def _est_tokens(messages: list[dict], extra: str = "") -> int:
    return (sum(len(m.get("content") or "") for m in messages) + len(extra)) // 4 + 40 * len(messages)


async def _parse(raw):
    out = raw.parse()
    return await out if inspect.isawaitable(out) else out


class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = groq.AsyncGroq(api_key=settings.groq_api_key or "missing", max_retries=0, timeout=25.0)
        self.cooldown_until: dict[str, float] = {}
        self.budget: dict[str, tuple[int, float]] = {}  # model -> (remaining tokens, reset epoch)
        self.forced_fail = {m.strip() for m in settings.force_fail_models.split(",") if m.strip()}

    # ---------------------------------------------------------------- bookkeeping
    def _record_headers(self, model: str, headers) -> None:
        try:
            remaining = int(headers.get("x-ratelimit-remaining-tokens"))
            reset = _duration(headers.get("x-ratelimit-reset-tokens")) or 60.0
            self.budget[model] = (remaining, time.time() + reset)
        except (TypeError, ValueError):
            pass

    def _fits(self, model: str, need: int) -> tuple[bool, float]:
        remaining, reset_at = self.budget.get(model, (10**9, 0.0))
        now = time.time()
        if now >= reset_at or remaining >= need:
            return True, 0.0
        return False, reset_at - now

    def _cool(self, model: str, seconds: float) -> None:
        self.cooldown_until[model] = time.time() + min(max(seconds, 1.0), 120.0)

    def _candidates(self, role: str, need: int, info: CallInfo) -> tuple[list[str], float]:
        """Models that can take this request now, plus the shortest wait until another one frees up."""
        now, ok, waits = time.time(), [], []
        for m in self.s.chain(role):
            if self.cooldown_until.get(m, 0) > now:
                waits.append(self.cooldown_until[m] - now)
                info.skipped.append(f"{m}:cooling")
                continue
            fits, wait = self._fits(m, need)
            if not fits:
                waits.append(wait)
                info.skipped.append(f"{m}:no_token_budget")
                continue
            ok.append(m)
        return ok, (min(waits) if waits else 10.0)

    # ---------------------------------------------------------------- structured output
    async def json(self, role: str, messages: list[dict], schema_model: type[T], json_schema: dict,
                   max_tokens: int = 700, timeout: float = 8.0) -> tuple[T, CallInfo]:
        """Strict json_schema where supported, else json_object + Pydantic validation (+1 repair retry)."""
        info = CallInfo(role)
        t0 = time.time()
        schema_txt = json.dumps(json_schema, separators=(",", ":"))
        need = _est_tokens(messages, schema_txt) + max_tokens
        models, wait = self._candidates(role, need, info)
        last_retry = wait
        for model in models:
            if model in self.forced_fail:
                self._cool(model, 30)
                info.attempts.append(f"{model}:forced_fail")
                continue
            if model in STRICT_SCHEMA_MODELS:
                rf = {"type": "json_schema", "json_schema": {"name": schema_model.__name__, "strict": True, "schema": json_schema}}
                msgs = messages
            else:
                rf = {"type": "json_object"}
                msgs = messages + [{"role": "system", "content": "Respond with one JSON object matching this JSON schema:\n" + schema_txt}]
            for attempt in range(2):
                try:
                    raw = await self.client.chat.completions.with_raw_response.create(
                        model=model, messages=msgs, temperature=0, max_completion_tokens=max_tokens,
                        response_format=rf, timeout=timeout, **_model_kwargs(model))
                    self._record_headers(model, raw.headers)
                    resp = await _parse(raw)
                    obj = schema_model.model_validate_json(resp.choices[0].message.content or "")
                    info.model = model
                    if resp.usage:
                        info.prompt_tokens, info.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
                    info.latency_ms = int((time.time() - t0) * 1000)
                    return obj, info
                except groq.RateLimitError as e:
                    last_retry = self._on_429(model, e)
                    info.attempts.append(f"{model}:rate_limited")
                    break
                except (ValidationError, json.JSONDecodeError) as e:
                    info.attempts.append(f"{model}:invalid_json")
                    if rf["type"] == "json_schema" or attempt == 1:
                        break
                    msgs = msgs + [{"role": "user", "content": f"Your JSON was invalid ({str(e)[:200]}). Return corrected JSON only."}]
                except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
                    raise LLMUnavailable(f"Groq API key missing or invalid: {e}", auth=True)
                except groq.APIStatusError as e:
                    info.attempts.append(_status_reason(model, e))
                    break
                except (groq.APIConnectionError, groq.APITimeoutError):
                    info.attempts.append(f"{model}:timeout")
                    break
        raise LLMUnavailable(f"no {role} model available ({', '.join(info.attempts)})", last_retry)

    # ---------------------------------------------------------------- streaming text
    async def stream(self, role: str, messages: list[dict], max_tokens: int, info: CallInfo,
                     timeout: float = 12.0) -> AsyncIterator[str]:
        """Stream text deltas; falls back to the next model only if nothing has been emitted yet."""
        t0 = time.time()
        need = _est_tokens(messages) + max_tokens
        models, wait = self._candidates(role, need, info)
        last_retry = wait
        for model in models:
            if model in self.forced_fail:
                self._cool(model, 30)
                info.attempts.append(f"{model}:forced_fail")
                continue
            emitted = False
            try:
                raw = await self.client.chat.completions.with_raw_response.create(
                    model=model, messages=messages, temperature=0.3, max_completion_tokens=max_tokens,
                    stream=True, timeout=timeout, **_model_kwargs(model))
                self._record_headers(model, raw.headers)
                stream = await _parse(raw)
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
                last_retry = self._on_429(model, e)
                info.attempts.append(f"{model}:rate_limited")
                if emitted:
                    raise LLMUnavailable("rate limited mid-stream", last_retry)
            except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
                raise LLMUnavailable(f"Groq API key missing or invalid: {e}", auth=True)
            except (groq.APIStatusError, groq.APIConnectionError, groq.APITimeoutError) as e:
                info.attempts.append(_status_reason(model, e) if isinstance(e, groq.APIStatusError) else f"{model}:timeout")
                if emitted:
                    raise LLMUnavailable("stream interrupted")
        raise LLMUnavailable(f"no {role} model available ({', '.join(info.attempts)})", last_retry)

    def _on_429(self, model: str, e: Exception) -> float:
        headers = getattr(getattr(e, "response", None), "headers", {}) or {}
        wait = None
        try:
            wait = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            wait = _duration(headers.get("x-ratelimit-reset-tokens"))
        wait = wait or 10.0
        self._cool(model, wait)
        return wait

    async def text(self, role: str, messages: list[dict], max_tokens: int = 300) -> tuple[str, CallInfo]:
        info = CallInfo(role)
        parts = [p async for p in self.stream(role, messages, max_tokens, info)]
        return "".join(parts), info
