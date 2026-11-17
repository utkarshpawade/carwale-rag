"""Per-turn RAG pipeline (exactly 2 LLM calls on the normal path):

  memory -> PLAN (LLM #1, strict JSON) -> resolve cars -> route -> retrieve (SQL and/or hybrid search)
         -> token-budgeted context -> ANSWER (LLM #2, streamed) -> persist history + working state
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from ..config import Settings
from ..embeddings import Embedder
from ..llm.client import CallInfo, LLMClient, LLMUnavailable
from ..llm.prompts import ANSWER_SYSTEM, CHITCHAT_SYSTEM, PLANNER_SYSTEM, SUMMARY_SYSTEM
from ..memory import store
from ..memory.store import ConvState
from . import car_search, specs
from .context import build_context
from .entities import CarResolver, is_generic, norm
from .followups import suggest
from .hybrid_search import hybrid_search, pinned_chunks
from .schemas import PLAN_EXAMPLE, PLAN_SCHEMA, CarRef, Filters, QueryPlan, empty_filters, simple_plan

CACHE_VERSION = "v8"  # bump when prompts / ranking change
SMALLTALK = re.compile(r"^\s*(hi+|hello+|hey+|hii+|namaste|hola|yo|good (morning|afternoon|evening)|thanks?( you| a lot)?|"
                       r"thank u|thx|ok(ay)?|cool|great|nice|bye|goodbye)\s*[!.?]*\s*$", re.I)
CAR_INTENTS = {"spec", "price", "feature_check", "compare", "safety", "service", "review", "variant_advice"}
INTENT_SECTIONS = {
    "spec": ["specs", "mileage", "overview", "faq", "trim", "features"],
    "price": ["variants", "city_prices", "faq", "overview", "trim"],
    "feature_check": ["features", "trim", "trim_features", "faq", "safety"],
    "safety": ["safety", "expert_opinion", "faq", "overview"],
    "service": ["service", "expert_article", "owner_reviews", "faq", "pros_cons"],
    "review": ["pros_cons", "expert_opinion", "expert_review", "expert_article", "owner_reviews", "overview"],
    "variant_advice": ["trim", "trim_features", "variants", "pros_cons", "features"],
    "compare": ["pros_cons", "expert_opinion", "overview", "mileage", "safety"],
}
DEFAULT_ASPECTS = {"spec": {"specs"}, "price": {"price", "variants"}, "feature_check": {"features", "variants"},
                   "safety": {"safety"}, "service": {"service", "reviews"}, "review": {"reviews"},
                   "variant_advice": {"variants", "features", "price"}}
ASPECT_SECTIONS = {"mileage": ["mileage", "expert_article", "owner_reviews"], "safety": ["safety"],
                   "service": ["service", "expert_article"], "features": ["features", "trim_features"],
                   "reviews": ["owner_reviews", "pros_cons"], "on_road_price": ["city_prices"],
                   "price": ["variants", "city_prices"], "variants": ["trim", "variants"], "dimensions": ["specs"],
                   "performance": ["specs", "expert_opinion"], "comfort": ["expert_opinion", "features"]}


# the dedicated summary chunk always included for an aspect / intent (fetched directly, not ranked)
ASPECT_PIN = {"mileage": "mileage", "safety": "safety", "service": "service", "on_road_price": "city_prices",
              "dimensions": "specs", "variants": "variants"}
INTENT_PIN = {"safety": "safety", "service": "service", "price": "variants", "variant_advice": "variants"}


@dataclass
class Event:
    type: str
    data: Any


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks: dict[str, int] = {}

    def mark(self, name: str) -> None:
        self.marks[name] = int((time.perf_counter() - self.t0) * 1000)


def merge_filters(old: dict, new: Filters) -> Filters:
    base = Filters.model_validate({**empty_filters().model_dump(), **(old or {})})
    upd = {k: v for k, v in new.model_dump().items() if v not in (None, [], "launched")}
    return base.model_copy(update=upd)


class Pipeline:
    def __init__(self, settings: Settings, llm: LLMClient, resolver: CarResolver, embedder: Embedder, stats: dict):
        self.s, self.llm, self.resolver, self.embedder, self.stats = settings, llm, resolver, embedder, stats
        self._bg: set[asyncio.Task] = set()

    # ------------------------------------------------------------------------------------------ planner
    def _planner_messages(self, message: str, history: list[dict], st: ConvState) -> list[dict]:
        lines = []
        if st.summary:
            lines.append(f"Summary of earlier conversation: {st.summary}")
        if history:
            lines.append("Recent turns:")
            for h in history:
                txt = h["content"] if h["role"] == "user" else (h["meta"].get("digest") or h["content"])
                lines.append(f"{'User' if h['role'] == 'user' else 'Assistant'}: {txt[:350]}")
        if st.active_cars:
            lines.append("Active cars from the last answer (numbered): " +
                         " ".join(f"{i}. {self.resolver.name(m)}" for i, m in enumerate(st.active_cars, 1)))
        if st.focus_car:
            lines.append(f"Focus car: {self.resolver.name(st.focus_car)}")
        f = {k: v for k, v in (st.filters or {}).items() if v not in (None, [], "launched")}
        if f:
            lines.append("Current search filters: " + json.dumps(f))
        if st.prefs:
            lines.append("User preferences: " + json.dumps(st.prefs))
        ctx = "CONTEXT\n" + ("\n".join(lines) if lines else "(new conversation)")
        return [{"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": f"{ctx}\n\nLATEST USER MESSAGE: {message}"}]

    # ------------------------------------------------------------------------------------------ backstops
    FUEL_WORDS = {"electric": "Electric", "ev": "Electric", "evs": "Electric", "diesel": "Diesel", "petrol": "Petrol",
                  "cng": "CNG", "hybrid": "Hybrid", "hybrids": "Hybrid"}

    def _backstops(self, plan: QueryPlan, message: str) -> list[str]:
        """Deterministic guards for the few plan mistakes the eval exposed (LLM output is still the primary plan)."""
        notes = []
        found = self.resolver.find_in_text(message)
        low = message.lower()
        if plan.intent in ("out_of_scope", "chitchat") and found and not SMALLTALK.match(message):
            plan.intent = ("service" if re.search(r"servic|mainten|warranty|ownership cost", low) else
                           "spec" if re.search(r"mileage|range|boot|engine|power|spec|dimension", low) else "review")
            plan.cars = [CarRef(mention=self.resolver.name(m), variant_hint=None, fuel_hint=None, transmission_hint=None)
                         for m in found]
            notes.append(f"Backstop: message names {', '.join(self.resolver.name(m) for m in found)}, so it is in scope.")
        elif plan.intent in CAR_INTENTS and not plan.cars and plan.refers_to == "none" and found:
            plan.cars = [CarRef(mention=self.resolver.name(m), variant_hint=None, fuel_hint=None, transmission_hint=None)
                         for m in found]
            notes.append("Backstop: car names taken from the message text.")
        f = plan.filters
        has_filters = bool(f.price_max or f.price_min or f.body_types or f.fuel_types or f.hybrid_types or f.seats_min
                           or f.transmission or f.feature_tags or f.ncap_min or f.brands)
        if plan.intent in CAR_INTENTS - {"compare"} and not plan.cars and plan.refers_to == "none" and has_filters:
            # "Safest SUVs under 15 lakh" labelled 'safety' with no car named: it is a search, keep the implied sort
            implied = {"safety": "safety", "price": "price_asc", "spec": "relevance"}.get(plan.intent, "relevance")
            if plan.sort_by == "relevance":
                plan.sort_by = implied
            notes.append(f"Backstop: '{plan.intent}' question with filters and no car named -> structured search.")
            plan.intent = "recommend"
            plan.filter_mode = plan.filter_mode if plan.filter_mode != "none" else "new"
        if plan.intent == "recommend" and not plan.cars and not plan.filters.fuel_types and not plan.filters.hybrid_types:
            fuels = sorted({v for k, v in self.FUEL_WORDS.items() if k in norm(message).split()})
            if fuels:
                plan.filters.fuel_types = fuels
                if "Hybrid" in fuels:
                    plan.filters.hybrid_types = ["strong", "mild"]
                if fuels == ["Electric"] and plan.sort_by == "relevance" and "range" in low:
                    plan.sort_by = "range"
                notes.append(f"Backstop: fuel filter {fuels} taken from the message text.")
        return notes

    # ------------------------------------------------------------------------------------------ resolve
    def _targets(self, plan: QueryPlan, st: ConvState) -> tuple[list[str], list[dict], list[str], list[str]]:
        ids: list[str] = []
        # "its EV version" names the car explicitly: the named car replaces the remembered one unless comparing
        explicit = bool(plan.cars) and plan.intent != "compare"
        if plan.refers_to == "focus" and st.focus_car and not explicit:
            ids.append(st.focus_car)
        elif plan.refers_to == "all":
            ids += st.active_cars
        elif plan.refers_to == "indices":
            ids += [st.active_cars[i - 1] for i in plan.active_indices if 1 <= i <= len(st.active_cars)]
        resolved, unresolved, notes = [], [], []
        for car in plan.cars:
            if is_generic(car.mention):
                # a modifier of the focus car: "its EV version" -> Punch EV; "the diesel one" -> focus car + fuel hint
                if st.focus_car:
                    target = st.focus_car
                    wants_ev = car.fuel_hint == "Electric" or {"ev", "electric"} & set(norm(car.mention).split())
                    if wants_ev and not self.resolver.models.get(target, {}).get("is_ev"):
                        ev = self.resolver.resolve(f"{self.resolver.name(target)} EV")
                        if ev and self.resolver.models[ev[0].model_id].get("is_ev"):
                            target = ev[0].model_id
                    resolved.append({"mention": car.mention, "model_id": target, "name": self.resolver.name(target),
                                     "score": "memory", "ambiguous": []})
                    ids.append(target)
                continue
            res = self.resolver.resolve(car.mention)
            if not res:
                unresolved.append(car.mention)
                continue
            top = res[0]
            if car.fuel_hint == "Electric" and not self.resolver.models[top.model_id].get("is_ev"):
                ev = self.resolver.resolve(f"{car.mention} EV")  # "Punch, electric" -> the separate Punch EV model
                if ev and self.resolver.models[ev[0].model_id].get("is_ev"):
                    top = ev[0]
            alts = [a for a in top.ambiguous if self.resolver.models[a].get("status") == "launched"][:1]
            resolved.append({"mention": car.mention, "model_id": top.model_id, "name": top.full_name,
                             "score": top.score, "ambiguous": alts})
            ids.append(top.model_id)
            if alts and plan.intent in CAR_INTENTS:
                ids += alts
                notes.append(f"'{car.mention}' matches more than one model ({', '.join(self.resolver.name(x) for x in [top.model_id] + alts)}); covering each.")
        if not ids and plan.intent in CAR_INTENTS and st.focus_car and not unresolved and not plan.cars:
            ids.append(st.focus_car)  # planner forgot refers_to
        return list(dict.fromkeys(ids)), resolved, unresolved, notes  # capped per route (details: 4)

    # ------------------------------------------------------------------------------------------ main turn
    async def run(self, conv_id: str, user_id: str, message: str, dry_run: bool = False) -> AsyncIterator[Event]:
        """dry_run=True stops after retrieval (used by the eval harness to save LLM tokens)."""
        timer = Timer()
        st = await store.load_state(conv_id)
        history = await store.recent_messages(conv_id, self.s.history_turns * 2)
        await store.ensure_conversation(conv_id, user_id, message)
        await store.add_message(conv_id, "user", message, {})
        yield Event("status", {"text": "Understanding your question…"})

        cache_key = None
        if not history and not SMALLTALK.match(message) and not dry_run:
            # versioned by pipeline revision + data snapshot, so re-ingestion or prompt changes never serve stale answers
            cache_key = hashlib.sha1(f"{CACHE_VERSION}|{self.stats.get('snapshot')}|{norm(message)}".encode()).hexdigest()
            cached = await store.cache_get(cache_key)
            if cached:
                async for ev in self._replay(conv_id, st, cached, timer):
                    yield ev
                return

        # ---- 1. plan
        planner_info: CallInfo | None = None
        if SMALLTALK.match(message):
            plan = simple_plan(message, "chitchat")
        else:
            plan = None
            for attempt in range(3):  # wait out short free-tier rate-limit windows instead of failing
                try:
                    plan, planner_info = await self.llm.json("planner", self._planner_messages(message, history, st),
                                                             QueryPlan, PLAN_SCHEMA, max_tokens=self.s.planner_max_tokens,
                                                             example=PLAN_EXAMPLE)
                    if not plan.standalone_query.strip():
                        plan.standalone_query = message
                    break
                except LLMUnavailable as e:
                    if e.auth or attempt == 2 or (e.retry_after or 0) > 30:
                        yield Event("error", {"message": self._busy(e)})
                        return
                    wait = min(max(e.retry_after or 3.0, 1.5), 30.0)
                    yield Event("status", {"text": f"Groq free-tier limit reached — retrying in {wait:.0f}s…"})
                    await asyncio.sleep(wait)
        timer.mark("plan")
        backstops = self._backstops(plan, message)

        # ---- 2. resolve
        target_ids, resolved, unresolved, notes = self._targets(plan, st)
        notes = backstops + notes
        trace: dict[str, Any] = {
            "rewritten_query": plan.standalone_query, "intent": plan.intent, "plan": plan.model_dump(),
            "resolved_cars": resolved, "unresolved": unresolved, "notes": list(notes),
            "memory": {"summary_used": bool(st.summary), "history_messages": len(history),
                       "active_cars_before": [self.resolver.name(m) for m in st.active_cars],
                       "focus_before": self.resolver.name(st.focus_car) if st.focus_car else None,
                       "filters_before": st.filters},
        }

        # ---- 3. chit-chat / clarify short paths
        if plan.intent in ("chitchat", "out_of_scope"):
            trace["route"] = plan.intent
            async for ev in self._simple_answer(conv_id, st, message, history, plan, trace, timer, planner_info):
                yield ev
            return
        if plan.clarify and not target_ids and plan.intent in CAR_INTENTS:
            trace["route"] = "clarify"
            async for ev in self._emit_fixed(conv_id, st, plan.clarify, plan, trace, timer, planner_info):
                yield ev
            return

        # ---- 4. route + retrieve
        qvec = await asyncio.to_thread(self.embedder.embed_query, plan.standalone_query)
        blocks: list[str] = [f"Note: {n}" for n in notes]
        chunks: list[dict] = []
        cards: list[dict] = []
        new_active, new_focus, new_filters = st.active_cars, st.focus_car, st.filters
        rivals: list[str] = []
        search_like = plan.intent == "recommend" or (plan.intent not in CAR_INTENTS | {"general"} and plan.filter_mode != "none")
        restrict = target_ids if plan.refers_to in ("all", "indices") and target_ids and plan.intent == "recommend" else None

        if search_like:
            f = merge_filters(st.filters, plan.filters) if (plan.filter_mode == "refine" or restrict) else plan.filters
            brand_slugs = [b for b in (self.resolver.brand_slug(x) for x in f.brands) if b]
            if f.status == "upcoming":
                yield Event("status", {"text": "Looking up upcoming launches…"})
                rows = await car_search.search_upcoming(f, brand_slugs)
                trace["route"] = "upcoming_search"
                blocks.append("Upcoming cars (prices are expected):\n" + "\n".join(
                    f"- {r['full_name']}: {specs.price_range(r.get('price_min'), r.get('price_max'))} expected, "
                    f"launch {r.get('launched_on') or 'TBA'}, {r.get('body_type_raw') or ''}" for r in rows))
                desc, relax = [], []
            else:
                yield Event("status", {"text": f"Searching {self.stats.get('variants', 0):,} variants across "
                                               f"{self.stats.get('models', 0)} models…"})
                rows, desc, relax = await car_search.search_cars(f, plan.sort_by, plan.prefer_tags, brand_slugs, restrict,
                                                                 limit=8)
                if not rows and restrict:
                    rows, desc, relax = await car_search.search_cars(empty_filters(), plan.sort_by, plan.prefer_tags, [], restrict)
                trace["route"] = "structured_search"
                blocks.append(f"Search: {'; '.join(desc) or 'no filters'}; sorted by {plan.sort_by}."
                              + (f" Relaxation: {'; '.join(relax)}." if relax else "")
                              + f" {len(rows)} matching models shown.")
                if rows:
                    blocks.append(specs.search_table(rows))
                else:
                    blocks.append("No cars in the CarWale catalog match these filters.")
            trace["sql_filters"], trace["relaxed"] = desc, relax
            ids = [r["id"] for r in rows]
            cards = [specs.card(r) for r in rows]
            if ids:
                chunks = await hybrid_search(qvec, plan.standalone_query, plan.keywords, model_ids=ids,
                                             sections=["overview", "pros_cons", "expert_opinion", "highlights"],
                                             k=7, per_model_cap=2)
                new_active, new_focus = ids, ids[0]
            new_filters = f.model_dump()
        elif target_ids:
            target_ids = target_ids[:4]  # fact sheets / comparison tables stay readable and within budget
            yield Event("status", {"text": "Reading " + ", ".join(self.resolver.name(i) for i in target_ids) + " data…"})
            models = await specs.load_models(target_ids)
            variants = await specs.load_variants(target_ids)
            aspects = set(plan.aspects) or DEFAULT_ASPECTS.get(plan.intent, set())
            compare = plan.intent == "compare" or len(target_ids) > 1
            hints = {self.resolver.resolve(c.mention)[0].model_id: c for c in plan.cars if self.resolver.resolve(c.mention)}
            all_facts, names = [], []
            for mid in target_ids:
                m = models.get(mid)
                if not m:
                    continue
                h = hints.get(mid)
                vs, vnotes = specs.filter_variants(variants.get(mid, []), h.fuel_hint if h else None,
                                                   h.transmission_hint if h else None, h.variant_hint if h else None)
                fs = specs.facts(m, vs, aspects, plan.city or st.prefs.get("city"), compare=compare)
                if vnotes:
                    fs["Note"] = "; ".join(vnotes)
                all_facts.append(fs)
                names.append(m["full_name"])
                cards.append(specs.card(m))
                if not compare:
                    blocks.append(specs.facts_block(m["full_name"], m["url"], fs))
                    wanted = list(dict.fromkeys(plan.filters.feature_tags + specs.detect_features(
                        " ".join([message, plan.standalone_query] + plan.keywords))))
                    if wanted and plan.intent in ("feature_check", "variant_advice", "spec", "price"):
                        blocks.append(specs.feature_availability(m["full_name"], vs, wanted))
                    elif plan.intent in ("feature_check", "variant_advice"):
                        blocks.append(specs.trim_feature_matrix(m["full_name"], vs))
                    if plan.intent in ("feature_check", "variant_advice", "price") or "variants" in aspects:
                        blocks.append(f"Variants of {m['full_name']}:\n" + specs.variants_table(vs))
                rivals += [self.resolver.name(x) for x in ((m.get("data") or {}).get("similar") or [])[:2]]
            if compare and all_facts:
                blocks.append("Comparison:\n" + specs.compare_table(names, all_facts))
            # two-tier retrieval: core sections for this intent get most slots; aspect extras (long-term
            # reports, owner reviews) get the rest, so plentiful review text cannot crowd out the core facts
            primary = list(dict.fromkeys(INTENT_SECTIONS.get(plan.intent, [])))
            extra = [s for s in dict.fromkeys(s for a in aspects for s in ASPECT_SECTIONS.get(a, [])) if s not in primary]
            k_main, k_extra = (6, 2) if compare else (4, 2)
            cap = 3 if compare else None
            if primary:
                chunks = await hybrid_search(qvec, plan.standalone_query, plan.keywords, model_ids=target_ids,
                                             sections=primary, k=k_main if extra else k_main + k_extra, per_model_cap=cap)
                if extra:
                    more = await hybrid_search(qvec, plan.standalone_query, plan.keywords, model_ids=target_ids,
                                               sections=extra, k=k_extra)
                    have = {c["id"] for c in chunks}
                    chunks += [c for c in more if c["id"] not in have]
            else:
                chunks = await hybrid_search(qvec, plan.standalone_query, plan.keywords, model_ids=target_ids,
                                             sections=extra or None, k=k_main + k_extra, per_model_cap=cap)
            pins = list(dict.fromkeys([ASPECT_PIN[a] for a in aspects if a in ASPECT_PIN] +
                                      ([INTENT_PIN[plan.intent]] if plan.intent in INTENT_PIN else [])))[:2]
            pinned = await pinned_chunks(target_ids[:3], pins)
            pinned_ids = {c["id"] for c in pinned}
            chunks = pinned + [c for c in chunks if c["id"] not in pinned_ids]
            trace["route"] = "compare" if compare else "car_details"
            trace["aspects"] = sorted(aspects)
            if not set(target_ids) <= set(st.active_cars):
                new_active = target_ids
            new_focus = target_ids[0]
        else:
            trace["route"] = "semantic_search"
            if unresolved:
                blocks.append(f"Not found in CarWale's India new-car catalog: {', '.join(unresolved)}. "
                              "Tell the user it isn't listed and suggest similar cars if the sources mention any.")
            gl = await hybrid_search(qvec, plan.standalone_query, plan.keywords, sections=["glossary"], k=2) \
                if plan.intent == "general" else []
            rest = await hybrid_search(qvec, plan.standalone_query, plan.keywords, k=6)
            seen = {c["id"] for c in gl}
            chunks = gl + [c for c in rest if c["id"] not in seen][: 6 - len(gl)]
        timer.mark("retrieve")

        context, sources = build_context(blocks, chunks, self.s.context_token_budget)
        trace["retrieved"] = sources
        trace["context_tokens_est"] = len(context) // 4
        if plan.city:
            st.prefs["city"] = plan.city
        yield Event("trace", trace)
        if cards:
            yield Event("cars", {"cars": cards[:8]})
        yield Event("sources", {"sources": sources})  # before tokens, so the UI can link [n] citations live
        if dry_run:
            names = [c["name"] for c in cards] if cards else [self.resolver.name(i) for i in target_ids]
            await store.add_message(conv_id, "assistant", "[dry run]", {"digest": f"[cars: {', '.join(names[:6])}]", "trace": trace})
            await store.save_state(conv_id, ConvState(summary=st.summary, summarized_upto=st.summarized_upto,
                                                      active_cars=new_active or [], focus_car=new_focus,
                                                      filters=new_filters or {}, prefs=st.prefs))
            trace["llm"] = [planner_info.as_dict()] if planner_info else []
            yield Event("done", {"dry_run": True, "trace": trace})
            return
        yield Event("status", {"text": "Writing the answer…"})

        # ---- 5. generate
        msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
        if st.summary:
            msgs.append({"role": "system", "content": f"Conversation summary so far: {st.summary}"})
        for h in history[-4:]:
            content = h["content"] if h["role"] == "user" else (h["meta"].get("digest") or h["content"])[:600]
            msgs.append({"role": h["role"], "content": content})
        q = plan.standalone_query + (f"\n(User's exact words: {message})" if norm(message) != norm(plan.standalone_query) else "")
        msgs.append({"role": "user", "content": f"{context}\n\n## QUESTION\n{q}"})
        answer_info = CallInfo("answer")
        parts: list[str] = []
        for attempt in range(3):
            try:
                async for delta in self.llm.stream("answer", msgs, self.s.answer_max_tokens, answer_info):
                    if not parts:
                        timer.mark("first_token")
                    parts.append(delta)
                    yield Event("token", {"text": delta})
                break
            except LLMUnavailable as e:
                if parts:
                    parts.append("\n\n_(answer cut short: the AI service is busy)_")
                    break
                if e.auth or attempt == 2 or (e.retry_after or 0) > 30:
                    yield Event("error", {"message": self._busy(e)})
                    return
                wait = min(max(e.retry_after or 3.0, 1.5), 30.0)
                yield Event("status", {"text": f"Groq free-tier limit reached — retrying in {wait:.0f}s…"})
                await asyncio.sleep(wait)
        answer = "".join(parts).strip()
        timer.mark("total")

        names = [c["name"] for c in cards] if cards else [self.resolver.name(i) for i in target_ids]
        fups = suggest(plan.intent, names, rivals, bool(plan.filters.transmission))
        trace["timings_ms"] = timer.marks
        trace["llm"] = [i.as_dict() for i in (planner_info, answer_info) if i]
        yield Event("trace", trace)  # final trace with timings + token usage
        yield Event("followups", {"followups": fups})

        # ---- 6. persist memory
        new_state = ConvState(summary=st.summary, summarized_upto=st.summarized_upto, active_cars=new_active or [],
                              focus_car=new_focus, filters=new_filters or {}, prefs=st.prefs)
        if plan.filters.price_max:
            new_state.prefs["budget_max"] = plan.filters.price_max
        digest = self._digest(answer, names)
        meta = {"plan": plan.model_dump(), "sources": sources, "cars": cards[:6], "followups": fups, "trace": trace,
                "digest": digest}
        msg_id = await store.add_message(conv_id, "assistant", answer, meta)
        await store.save_state(conv_id, new_state)
        if cache_key and plan.intent not in ("chitchat", "out_of_scope") and not plan.city:
            await store.cache_put(cache_key, {"answer": answer, "cars": cards[:6], "sources": sources, "followups": fups,
                                              "trace": trace, "active_cars": new_state.active_cars,
                                              "focus_car": new_state.focus_car, "filters": new_state.filters,
                                              "digest": digest, "plan": plan.model_dump()})
        self._spawn(self._maybe_summarize(conv_id))
        yield Event("done", {"message_id": msg_id, "conversation_id": conv_id, "timings_ms": timer.marks,
                             "models": [i.model for i in (planner_info, answer_info) if i and i.model]})

    # ------------------------------------------------------------------------------------------ helpers
    @staticmethod
    def _digest(answer: str, names: list[str]) -> str:
        text = re.sub(r"\s+", " ", re.sub(r"[|#*_`>-]+", " ", answer)).strip()
        return (f"[cars: {', '.join(names[:6])}] " if names else "") + text[:420]

    @staticmethod
    def _busy(e: LLMUnavailable) -> str:
        if getattr(e, "auth", False):
            return "The AI service is not configured correctly (Groq API key missing or invalid)."
        wait = f" in about {int(e.retry_after)} seconds" if e.retry_after else " in a moment"
        return f"The AI service is at its free-tier limit right now. Please try again{wait}."

    async def _simple_answer(self, conv_id, st, message, history, plan, trace, timer, planner_info):
        msgs = [{"role": "system", "content": CHITCHAT_SYSTEM}]
        for h in history[-2:]:
            msgs.append({"role": h["role"], "content": (h["meta"].get("digest") or h["content"])[:400]})
        msgs.append({"role": "user", "content": message})
        info = CallInfo("summary")
        parts = []
        try:
            async for d in self.llm.stream("summary", msgs, 200, info):
                parts.append(d)
                yield Event("token", {"text": d})
        except LLMUnavailable:
            fallback = "Hi! I'm CarWale Assistant. Ask me about car prices, mileage, safety, features or comparisons."
            parts = [fallback]
            yield Event("token", {"text": fallback})
        timer.mark("total")
        trace["timings_ms"] = timer.marks
        trace["llm"] = [i.as_dict() for i in (planner_info, info) if i]
        yield Event("trace", trace)
        fups = suggest("chitchat", [], [], False)
        yield Event("followups", {"followups": fups})
        msg_id = await store.add_message(conv_id, "assistant", "".join(parts), {"plan": plan.model_dump(), "trace": trace,
                                                                             "followups": fups, "digest": "".join(parts)[:200]})
        yield Event("done", {"message_id": msg_id, "conversation_id": conv_id, "timings_ms": timer.marks})

    async def _emit_fixed(self, conv_id, st, text, plan, trace, timer, planner_info):
        yield Event("token", {"text": text})
        timer.mark("total")
        trace["timings_ms"] = timer.marks
        trace["llm"] = [planner_info.as_dict()] if planner_info else []
        yield Event("trace", trace)
        msg_id = await store.add_message(conv_id, "assistant", text, {"plan": plan.model_dump(), "trace": trace, "digest": text})
        yield Event("done", {"message_id": msg_id, "conversation_id": conv_id, "timings_ms": timer.marks})

    async def _replay(self, conv_id, st, cached, timer):
        trace = dict(cached.get("trace") or {})
        trace["cache"] = "hit (answer served from cache, 0 LLM tokens)"
        yield Event("trace", trace)
        if cached.get("cars"):
            yield Event("cars", {"cars": cached["cars"]})
        text = cached["answer"]
        for i in range(0, len(text), 24):
            yield Event("token", {"text": text[i:i + 24]})
            await asyncio.sleep(0.008)
        timer.mark("total")
        yield Event("sources", {"sources": cached.get("sources") or []})
        yield Event("followups", {"followups": cached.get("followups") or []})
        msg_id = await store.add_message(conv_id, "assistant", text, {"plan": cached.get("plan"), "sources": cached.get("sources"),
                                                                     "cars": cached.get("cars"), "followups": cached.get("followups"),
                                                                     "trace": trace, "digest": cached.get("digest")})
        await store.save_state(conv_id, ConvState(summary=st.summary, summarized_upto=st.summarized_upto,
                                                  active_cars=cached.get("active_cars") or [], focus_car=cached.get("focus_car"),
                                                  filters=cached.get("filters") or {}, prefs=st.prefs))
        yield Event("done", {"message_id": msg_id, "conversation_id": conv_id, "timings_ms": timer.marks, "cached": True})

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _maybe_summarize(self, conv_id: str) -> None:
        """Rolling summary: fold messages older than the short-term buffer into conversation_state.summary."""
        try:
            st = await store.load_state(conv_id)
            msgs = await store.messages_after(conv_id, st.summarized_upto)
            keep = self.s.history_turns * 2
            older = msgs[:-keep] if len(msgs) > keep else []
            if len(older) < self.s.summarize_after_messages:
                return
            transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in older)
            prompt = (f"Existing summary: {st.summary or '(none)'}\n\nNew conversation part:\n{transcript}")
            text, _ = await self.llm.text("summary", [{"role": "system", "content": SUMMARY_SYSTEM},
                                                      {"role": "user", "content": prompt}], 220)
            st2 = await store.load_state(conv_id)
            st2.summary, st2.summarized_upto = text.strip(), older[-1]["id"]
            await store.save_state(conv_id, st2)
        except Exception:
            pass  # summarisation is best-effort; the short-term buffer still works
