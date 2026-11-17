"""Fake Groq (OpenAI-compatible) chat-completions API for tests. It never calls the real Groq API.

Two ways to use it:
  * in-process (pytest): ``FakeGroq().client()`` returns a real ``groq.AsyncGroq`` whose HTTP calls go to a custom
    httpx transport (no sockets). The real SDK parses the responses, so status codes, headers, SSE streaming and the
    SDK's error mapping are all exercised.
  * as an HTTP server (Playwright E2E): ``python -m tests.fakes.fake_groq --port 18081`` and point the backend at it
    with ``GROQ_BASE_URL=http://127.0.0.1:18081``. Tests steer it through ``/__control/*`` endpoints.

Behaviour is deterministic:
  * planner requests (``response_format`` set)  -> a rule-based QueryPlan JSON (overridable per message)
  * answer requests (ANSWER system prompt)       -> a markdown answer built from the CAR DATA / SOURCES context
  * chit-chat and summary requests               -> short fixed texts
  * failure rules: HTTP status (429 + retry-after, 401, 500, 503...), timeout, connection error, a stream cut after
    N chunks, an in-stream error event, invalid JSON, an empty answer, slow tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

PLAN_KEYS_DEFAULT: dict[str, Any] = {
    "standalone_query": "", "intent": "general", "cars": [], "refers_to": "none", "active_indices": [],
    "filters": {"price_min": None, "price_max": None, "fuel_types": [], "hybrid_types": [], "body_types": [],
                "transmission": None, "seats_min": None, "brands": [], "feature_tags": [], "ncap_min": None,
                "mileage_min": None, "status": "launched"},
    "filter_mode": "none", "prefer_tags": [], "sort_by": "relevance", "aspects": [], "keywords": [], "city": None,
    "clarify": None,
}

# (regex on the lower-cased message, mention the planner emits) - longest names first
CAR_WORDS = [
    (r"grand vitara", "Maruti Suzuki Grand Vitara"), (r"hyryder", "Toyota Urban Cruiser Hyryder"),
    (r"innova hycross|hycross", "Toyota Innova Hycross"), (r"innova crysta|crysta", "Toyota Innova Crysta"),
    (r"fortuner|fortuneer", "Toyota Fortuner"), (r"nexon ev", "Tata Nexon EV"), (r"nexon", "Tata Nexon"),
    (r"punch", "Tata Punch"), (r"cretta|creta", "Hyundai Creta"), (r"seltos", "Kia Seltos"), (r"sonet", "Kia Sonet"),
    (r"venue", "Hyundai Venue"), (r"swift", "Maruti Suzuki Swift"), (r"brezza", "Maruti Suzuki Brezza"),
    (r"xuv ?700|xuv 7xo", "Mahindra XUV 7XO"), (r"hector", "MG Hector"), (r"honda city", "Honda City"),
    (r"batmobile", "Batmobile"),
]
CITIES = ["mumbai", "delhi", "bangalore", "bengaluru", "pune", "chennai", "hyderabad", "kolkata", "jaipur"]
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
STOP = {"the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "is", "are", "what", "which", "does", "do", "it",
        "its", "me", "show", "tell", "about", "how", "much", "with", "car", "cars", "vs", "best", "under", "lakh",
        "lakhs", "only", "one", "has", "have", "get", "this", "that", "these", "them"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def _price(num: str, unit: str | None) -> int:
    v = float(num)
    return int(v * (1e7 if (unit or "").startswith("c") else 1e5))


def heuristic_plan(message: str, context: str) -> dict:
    """A small rule-based stand-in for the LLM planner: good enough for realistic end-to-end flows."""
    m = message.lower()
    plan = json.loads(json.dumps(PLAN_KEYS_DEFAULT))
    f = plan["filters"]
    cars = []
    for rx, mention in CAR_WORDS:
        if re.search(rf"\b(?:{rx})\b", m) and mention not in cars:
            cars.append(mention)
    plan["cars"] = [{"mention": c, "variant_hint": None, "fuel_hint": None, "transmission_hint": None} for c in cars]
    has_active = "Active cars from the last answer" in context
    focus = re.search(r"Focus car: (.+)", context)

    # ---- references to earlier turns
    ords = [ORDINALS[w] for w in re.findall(r"\b(first|1st|second|2nd|third|3rd|fourth|4th)\b", m)]
    if ords and has_active:
        plan["refers_to"], plan["active_indices"] = "indices", ords
    elif has_active and re.search(r"\b(these|them|those|which one|which of|both|all of)\b", m):
        plan["refers_to"] = "all"
    elif focus and not cars and re.search(r"\b(it|its|it's|this car|that car|the car)\b", m):
        plan["refers_to"] = "focus"

    # ---- filters
    if mm := re.search(r"(?:under|below|less than|up ?to|within|max)\s*(?:rs\.?|₹)?\s*(\d+(?:\.\d+)?)\s*(lakhs?|lac|l\b|crores?|cr\b)?", m):
        f["price_max"] = _price(mm.group(1), mm.group(2))
    if mm := re.search(r"(?:above|over|more than|starting)\s*(?:rs\.?|₹)?\s*(\d+(?:\.\d+)?)\s*(lakhs?|lac|crores?|cr\b)", m):
        f["price_min"] = _price(mm.group(1), mm.group(2))
    for word, fuel in (("petrol", "Petrol"), ("diesel", "Diesel"), ("cng", "CNG"), ("electric", "Electric"),
                       (r"evs?", "Electric"), ("hybrids?", "Hybrid")):
        if re.search(rf"\b{word}\b", m) and fuel not in f["fuel_types"] and not cars:
            f["fuel_types"].append(fuel)
    if "Hybrid" in f["fuel_types"]:
        f["hybrid_types"] = ["strong"] if "strong hybrid" in m else ["strong", "mild"]
    for word, body in ((r"suvs?", "SUV"), (r"sedans?", "Sedan"), (r"hatchbacks?", "Hatchback"), (r"mpvs?|muvs?", "MUV")):
        if re.search(rf"\b{word}\b", m) and not cars:
            f["body_types"].append(body)
    if re.search(r"\b(automatics?|amt|cvt|dct)\b", m):
        f["transmission"] = "Automatic"
    elif re.search(r"\bmanuals?\b", m):
        f["transmission"] = "Manual"
    if mm := re.search(r"\b(\d)\s*-?\s*seat(?:er|ers|s)?\b", m):
        f["seats_min"] = int(mm.group(1))
    for word, tag in (("sunroof", "sunroof"), ("ventilated", "ventilated_seats"), (r"360", "360_camera"),
                      ("adas", "adas"), ("wireless charg", "wireless_charger")):
        if re.search(rf"\b{word}", m) and re.search(r"\bwith\b", m) and not cars:
            f["feature_tags"].append(tag)
    if re.search(r"\b(upcoming|launching soon|new launches)\b", m):
        f["status"] = "upcoming"
    if re.search(r"\bsmart|tech\b", m):
        plan["prefer_tags"] = ["connected_car", "adas", "ota_updates", "wireless_carplay"]

    # ---- sort
    for rx, sort in ((r"mileage|fuel efficient|economical", "mileage"), (r"cheapest|affordable|lowest price", "price_asc"),
                     (r"safest", "safety"), (r"powerful|fastest|sporty", "power"), (r"best rated", "rating")):
        if re.search(rx, m):
            plan["sort_by"] = sort
            break
    if plan["sort_by"] == "relevance" and "Electric" in f["fuel_types"] and "range" in m:
        plan["sort_by"] = "range"

    # ---- intent
    has_filters = any(f[k] for k in ("price_max", "price_min", "fuel_types", "body_types", "transmission",
                                     "seats_min", "feature_tags")) or f["status"] == "upcoming"
    if re.search(r"\b(cricket|weather|recipe|movie|football|bitcoin|stock market|world cup)\b", m) and not cars:
        intent = "out_of_scope"
    elif re.search(r"\bvs\.?\b|versus|compare|better than", m) and (len(cars) >= 2 or plan["refers_to"] == "all"):
        intent = "compare"
    elif not cars and re.search(r"\b(what is|what are|explain|difference between|meaning of)\b", m):
        intent = "general"
    elif not cars and (has_filters or plan["refers_to"] in ("all", "indices")
                       or re.search(r"\b(best|recommend|suggest|cheapest|safest|which cars|only|show)\b", m)):
        intent = "recommend"
        plan["filter_mode"] = "refine" if re.search(r"\b(only|make it|instead|refine)\b", m) and has_active else "new"
    elif re.search(r"\b(safe|safety|ncap|airbags?|crash)\b", m):
        intent = "safety"
    elif re.search(r"\b(service|maintenance|warranty|ownership cost)\b", m):
        intent = "service"
    elif re.search(r"which variant|value for money|worth it over|adds? over", m):
        intent = "variant_advice"
    elif re.search(r"\b(does|do)\b.*\b(have|get|come with)\b|which variants?|sunroof|ventilated|features?", m):
        intent = "feature_check"
    elif re.search(r"\b(price|cost|on-road|on road|emi|how much)\b", m):
        intent = "price"
    elif re.search(r"\b(mileage|range|boot|engine|power|bhp|torque|dimensions?|ground clearance|seating|colou?rs?|charging)\b", m):
        intent = "spec"
    elif cars or plan["refers_to"] != "none":
        intent = "review"
    else:
        intent = "general"
    plan["intent"] = intent

    # ---- aspects, keywords, city, rewrite
    for rx, aspect in ((r"mileage|fuel efficien", "mileage"), (r"safe|ncap|airbag", "safety"), (r"on-road|on road", "on_road_price"),
                       (r"\bprice|cost", "price"), (r"feature|sunroof|ventilated|adas", "features"),
                       (r"service|warranty|maintenance", "service"), (r"owners|review|pros|cons", "reviews"),
                       (r"variant", "variants"), (r"boot|dimension|length|width", "dimensions"), (r"colou?r", "colours")):
        if re.search(rx, m) and aspect not in plan["aspects"]:
            plan["aspects"].append(aspect)
    plan["keywords"] = [w for w in _norm(message).split() if w not in STOP and len(w) > 2][:8]
    plan["city"] = next((c.title() for c in CITIES if re.search(rf"\b{c}\b", m)), None)
    rewrite = message
    if plan["refers_to"] == "focus" and focus:
        rewrite = re.sub(r"\b(it|its|this car|that car|the car)\b", focus.group(1).strip(), message, flags=re.I)
    plan["standalone_query"] = rewrite
    return plan


def fake_answer(context_and_question: str) -> str:
    """A grounded-looking markdown answer built only from the context the pipeline sent (no outside facts)."""
    q_lines = context_and_question.split("## QUESTION", 1)[-1].strip().splitlines() if "## QUESTION" in context_and_question else []
    q = q_lines[0] if q_lines else ""
    names = re.findall(r"^#### (.+?) \(", context_and_question, flags=re.M)
    if not names and (hdr := re.search(r"^\| Attribute \| (.+) \|$", context_and_question, flags=re.M)):
        names = [n.strip() for n in hdr.group(1).split("|")]
    if not names:
        names = re.findall(r"^\| \d+ \| ([^|]+?) \|", context_and_question, flags=re.M)[:4]
    facts = re.findall(r"^- ([A-Z][^:\n]{2,40}): (.{1,90})$", context_and_question, flags=re.M)[:4]
    has_sources = "## SOURCES" in context_and_question
    cite = " [1]" if has_sources else ""
    lead = ", ".join(f"**{n}**" for n in names[:4]) or "your question"
    lines = [f"Here is what CarWale's data says about {lead}{cite}.", ""]
    lines += [f"- {k}: {v}" for k, v in facts]
    if len(names) >= 2:
        lines += ["", "| Car | Why it fits |", "|---|---|"] + [f"| {n} | Matches what you asked |" for n in names[:4]]
    lines += ["", f"_Test answer for: {q}_"]
    return "\n".join(lines)


@dataclass
class Rule:
    """One failure rule. model / kind '*' match everything; times=None means forever."""
    action: str  # status | timeout | connect | cut | stream_error | invalid_json | empty | hang
    model: str = "*"
    kind: str = "*"  # planner | answer | chitchat | summary
    status: int = 500
    retry_after: str | None = None
    after_chunks: int = 2
    times: int | None = None
    code: str | None = None

    def matches(self, model: str, kind: str) -> bool:
        return self.model in ("*", model) and self.kind in ("*", kind)


@dataclass
class Reply:
    status: int = 200
    headers: dict = field(default_factory=dict)
    json: dict | None = None
    chunks: list[str] | None = None  # SSE payloads ("data: ...")
    cut_after: int | None = None  # raise a network error after N chunks
    raise_before: Exception | None = None
    delay_s: float = 0.0  # pause between streamed chunks
    hang_s: float = 0.0  # pause before responding


class FakeGroq:
    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ control
    def reset(self) -> None:
        self.rules: list[Rule] = []
        self.plans: dict[str, dict] = {}  # normalised message -> plan fields to override
        self.raw_plans: dict[str, str] = {}  # normalised message -> raw planner output (e.g. invalid JSON)
        self.answer_override: str | None = None
        self.token_delay_s = 0.0
        self.chunk_size = 16
        self.calls: list[dict] = []
        self.remaining_tokens = 50_000

    def configure(self, cfg: dict) -> None:
        if "rules" in cfg:
            self.rules = [Rule(**r) for r in cfg["rules"]]
        if "plans" in cfg:
            self.plans = {_norm(k): v for k, v in cfg["plans"].items()}
        if "raw_plans" in cfg:
            self.raw_plans = {_norm(k): v for k, v in cfg["raw_plans"].items()}
        if "answer_override" in cfg:
            self.answer_override = cfg["answer_override"]
        if "token_delay_ms" in cfg:
            self.token_delay_s = float(cfg["token_delay_ms"]) / 1000
        if "chunk_size" in cfg:
            self.chunk_size = int(cfg["chunk_size"])
        if "remaining_tokens" in cfg:
            self.remaining_tokens = int(cfg["remaining_tokens"])

    def add_rule(self, action: str, **kw) -> Rule:
        r = Rule(action=action, **kw)
        self.rules.append(r)
        return r

    def calls_of(self, kind: str) -> list[dict]:
        return [c for c in self.calls if c["kind"] == kind]

    # ------------------------------------------------------------------ core
    @staticmethod
    def _kind(body: dict) -> str:
        if body.get("response_format"):
            return "planner"
        system = next((m.get("content") or "" for m in body.get("messages") or [] if m.get("role") == "system"), "")
        if system.startswith("Summarize this car-shopping conversation"):
            return "summary"
        if "Reply briefly (1-3 sentences)" in system:
            return "chitchat"
        return "answer"

    @staticmethod
    def _latest_user(body: dict) -> str:
        users = [m.get("content") or "" for m in body.get("messages") or [] if m.get("role") == "user"]
        for u in users:
            if "LATEST USER MESSAGE:" in u:
                return u.rsplit("LATEST USER MESSAGE:", 1)[1].strip()
        return users[-1] if users else ""

    def _rate_headers(self) -> dict:
        return {"x-ratelimit-remaining-tokens": str(self.remaining_tokens), "x-ratelimit-reset-tokens": "1s",
                "x-ratelimit-remaining-requests": "999", "x-ratelimit-reset-requests": "1m0s"}

    def respond(self, body: dict, headers: dict) -> Reply:
        model, kind = body.get("model", "?"), self._kind(body)
        latest = self._latest_user(body)
        prompt = "\n\n".join(m.get("content") or "" for m in body.get("messages") or [] if m.get("role") != "system")
        self.calls.append({"model": model, "kind": kind, "message": latest, "prompt": prompt, "stream": bool(body.get("stream")),
                           "max_tokens": body.get("max_completion_tokens"), "t": time.time()})
        auth = headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] in ("", "missing", "invalid-key"):
            return Reply(401, json={"error": {"message": "Invalid API Key", "type": "invalid_request_error",
                                              "code": "invalid_api_key"}})
        rule = next((r for r in self.rules if r.matches(model, kind)), None)
        if rule and rule.times is not None:
            rule.times -= 1
            if rule.times <= 0:
                self.rules.remove(rule)
        if rule:
            if rule.action == "status":
                h = {"retry-after": rule.retry_after} if rule.retry_after else {}
                code = rule.code or {429: "rate_limit_exceeded", 401: "invalid_api_key"}.get(rule.status, "server_error")
                return Reply(rule.status, headers=h, json={"error": {"message": f"fake {rule.status}", "type": "fake", "code": code}})
            if rule.action == "timeout":
                return Reply(raise_before=httpx.ReadTimeout("fake read timeout"), hang_s=30)
            if rule.action == "connect":
                return Reply(raise_before=httpx.ConnectError("fake connection refused"), hang_s=0)
            if rule.action == "hang":
                return Reply(hang_s=30)

        if kind == "planner":
            key = _norm(latest)
            if rule and rule.action == "invalid_json":
                content = "this is not json {"
            elif key in self.raw_plans:
                content = self.raw_plans[key]
            else:
                ctx = latest and next((m.get("content") or "" for m in body["messages"] if m.get("role") == "user"), "")
                plan = heuristic_plan(latest, ctx)
                for k, v in self.plans.get(key, {}).items():
                    if k == "filters":
                        plan["filters"].update(v)
                    else:
                        plan[k] = v
                content = json.dumps(plan)
            text = content
        elif kind == "chitchat":
            text = ("I can only help with cars. Try asking about a car's price, mileage or safety."
                    if re.search(r"cricket|weather|recipe|movie|football", latest.lower())
                    else "Hi! I'm the CarWale test assistant. Ask me about car prices, mileage, safety or comparisons.")
        elif kind == "summary":
            text = "Summary: the user is shopping for a car and compared a few models."
        else:
            text = self.answer_override if self.answer_override is not None else fake_answer(latest)
        if rule and rule.action == "empty":
            text = ""
        usage = {"prompt_tokens": sum(len(m.get("content") or "") for m in body.get("messages") or []) // 4,
                 "completion_tokens": max(1, len(text) // 4)}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        created, cid = int(time.time()), f"chatcmpl-fake-{len(self.calls)}"
        if not body.get("stream"):
            return Reply(200, headers=self._rate_headers(), json={
                "id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": usage})
        pieces = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        def chunk(delta: dict, finish=None, extra=None) -> str:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            payload.update(extra or {})
            return "data: " + json.dumps(payload) + "\n\n"

        chunks = [chunk({"role": "assistant", "content": ""})] + [chunk({"content": p}) for p in pieces]
        if rule and rule.action == "stream_error":
            chunks = chunks[: rule.after_chunks + 1] + [
                "data: " + json.dumps({"error": {"message": "fake upstream failure mid-stream", "type": "server_error"}}) + "\n\n"]
        chunks += [chunk({}, "stop", {"x_groq": {"id": cid, "usage": usage}}), "data: [DONE]\n\n"]
        return Reply(200, headers={**self._rate_headers(), "content-type": "text/event-stream"}, chunks=chunks,
                     cut_after=rule.after_chunks + 1 if rule and rule.action == "cut" else None,
                     delay_s=self.token_delay_s)

    # ------------------------------------------------------------------ in-process transport
    def transport(self) -> httpx.AsyncBaseTransport:
        return _FakeTransport(self)

    def client(self, api_key: str = "test-key-not-real"):
        import groq

        return groq.AsyncGroq(api_key=api_key, base_url="http://fake-groq.test", max_retries=0, timeout=25.0,
                              http_client=httpx.AsyncClient(transport=self.transport(), base_url="http://fake-groq.test"))

    # ------------------------------------------------------------------ HTTP server (E2E)
    def asgi_app(self):
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse
        from starlette.routing import Route

        async def completions(request: Request):
            body = await request.json()
            reply = self.respond(body, {k.lower(): v for k, v in request.headers.items()})
            if reply.hang_s:
                await asyncio.sleep(reply.hang_s)
            if reply.raise_before is not None:
                return JSONResponse({"error": {"message": str(reply.raise_before), "type": "fake"}}, status_code=503)
            if reply.chunks is None:
                return JSONResponse(reply.json, status_code=reply.status, headers=reply.headers)

            async def gen():
                for i, c in enumerate(reply.chunks):
                    if reply.cut_after is not None and i >= reply.cut_after:
                        raise RuntimeError("fake connection drop mid-stream")
                    yield c
                    if reply.delay_s:
                        await asyncio.sleep(reply.delay_s)

            hdrs = {k: v for k, v in reply.headers.items() if k != "content-type"}
            return StreamingResponse(gen(), media_type="text/event-stream", headers=hdrs)

        async def reset(_: Request):
            self.reset()
            return JSONResponse({"ok": True})

        async def configure(request: Request):
            self.configure(await request.json())
            return JSONResponse({"ok": True})

        async def calls(_: Request):
            return JSONResponse({"calls": self.calls})

        async def health(_: Request):
            return JSONResponse({"status": "ok", "fake": True})

        return Starlette(routes=[
            Route("/openai/v1/chat/completions", completions, methods=["POST"]),
            Route("/__control/reset", reset, methods=["POST"]),
            Route("/__control/configure", configure, methods=["POST"]),
            Route("/__control/calls", calls, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
        ])


class _FlakyStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[str], cut_after: int | None, delay_s: float):
        self.chunks, self.cut_after, self.delay_s = chunks, cut_after, delay_s

    async def __aiter__(self):
        for i, c in enumerate(self.chunks):
            if self.cut_after is not None and i >= self.cut_after:
                raise httpx.ReadError("fake connection reset mid-stream")
            yield c.encode()
            if self.delay_s:
                await asyncio.sleep(self.delay_s)


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, fake: FakeGroq):
        self.fake = fake

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        reply = self.fake.respond(body, {k.lower(): v for k, v in request.headers.items()})
        if reply.raise_before is not None:
            raise reply.raise_before
        if reply.hang_s:
            await asyncio.sleep(reply.hang_s)
        if reply.chunks is not None:
            return httpx.Response(200, headers=reply.headers, stream=_FlakyStream(reply.chunks, reply.cut_after, reply.delay_s),
                                  request=request)
        return httpx.Response(reply.status, headers=reply.headers, json=reply.json, request=request)


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Fake Groq API for end-to-end tests")
    ap.add_argument("--port", type=int, default=18081)
    args = ap.parse_args()
    uvicorn.run(FakeGroq().asgi_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
