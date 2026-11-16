"""Golden-set evaluation: planner intent/entity/filter accuracy, routing, retrieval hit rate, (optional) answer facts.

Usage (from repo root):
  python eval/run_eval.py            # dry run: plan + retrieval only (~2K tokens per case)
  python eval/run_eval.py --answers  # also generate answers and check must-contain facts (more tokens)
"""
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import db  # noqa: E402
from app.bootstrap import build_pipeline  # noqa: E402
from app.config import get_settings  # noqa: E402

GOLD = json.loads((Path(__file__).with_name("golden.json")).read_text(encoding="utf-8"))
PAUSE_S = 6  # stay under Groq free-tier TPM


async def run_turn(pipeline, conv, q, answers: bool):
    trace, text, t0 = {}, [], time.perf_counter()
    async for ev in pipeline.run(conv, "eval-user-0001", q, dry_run=not answers):
        if ev.type == "trace":
            trace = ev.data
        elif ev.type == "token":
            text.append(ev.data["text"])
        elif ev.type == "done" and ev.data.get("trace"):
            trace = ev.data["trace"]
        elif ev.type == "error":
            trace["error"] = ev.data["message"]
    return trace, "".join(text), int((time.perf_counter() - t0) * 1000)


def check(case, trace, answer, answers: bool) -> list[str]:
    fails = []
    plan = trace.get("plan") or {}
    if trace.get("error"):
        return [f"error: {trace['error']}"]
    if case.get("intent") and trace.get("intent") not in case["intent"]:
        fails.append(f"intent {trace.get('intent')} not in {case['intent']}")
    resolved = [r["model_id"] for r in trace.get("resolved_cars") or []]
    retrieved_models = {s.get("model_id") for s in trace.get("retrieved") or []}
    targets = set(resolved) | retrieved_models
    for t in case.get("targets", []):
        if t not in targets:
            fails.append(f"target {t} missing (resolved={resolved})")
    if case.get("targets_any") and not set(case["targets_any"]) & targets:
        fails.append(f"none of {case['targets_any']} in {sorted(targets)}")
    for k, v in (case.get("filters") or {}).items():
        if (plan.get("filters") or {}).get(k) != v:
            fails.append(f"filter {k}={plan.get('filters', {}).get(k)} != {v}")
    if case.get("fuel_any") and not set(case["fuel_any"]) & set((plan.get("filters") or {}).get("fuel_types") or []):
        fails.append(f"fuel_types {plan.get('filters', {}).get('fuel_types')} lacks {case['fuel_any']}")
    if case.get("body_any") and not set(case["body_any"]) & set((plan.get("filters") or {}).get("body_types") or []):
        fails.append(f"body_types {plan.get('filters', {}).get('body_types')} lacks {case['body_any']}")
    if case.get("sort") and plan.get("sort_by") != case["sort"]:
        fails.append(f"sort {plan.get('sort_by')} != {case['sort']}")
    if case.get("route") and trace.get("route") != case["route"]:
        fails.append(f"route {trace.get('route')} != {case['route']}")
    if case.get("refers") and plan.get("refers_to") not in case["refers"]:
        fails.append(f"refers_to {plan.get('refers_to')} not in {case['refers']}")
    if case.get("filter_mode") and plan.get("filter_mode") != case["filter_mode"]:
        fails.append(f"filter_mode {plan.get('filter_mode')} != {case['filter_mode']}")
    kinds = {s.get("kind") for s in trace.get("retrieved") or []}
    if case.get("sections_any") and not set(case["sections_any"]) & kinds:
        fails.append(f"sections {sorted(k for k in kinds if k)} lack {case['sections_any']}")
    if case.get("cars_any"):
        cars = set(resolved) | retrieved_models
        if not set(case["cars_any"]) & cars:
            fails.append("expected cars not retrieved")
    if answers:
        for frag in case.get("answer_has", []):
            if frag.lower() not in answer.lower():
                fails.append(f"answer lacks '{frag}'")
    return fails


async def main():
    answers = "--answers" in sys.argv
    pipeline = await build_pipeline(get_settings())
    results, lat = [], []
    print(f"mode: {'full answers' if answers else 'dry run (plan + retrieval)'}\n")
    for case in GOLD["single_turn"]:
        trace, answer, ms = await run_turn(pipeline, str(uuid.uuid4()), case["q"], answers)
        fails = check(case, trace, answer, answers)
        results.append(not fails)
        lat.append(ms)
        print(f"{'PASS' if not fails else 'FAIL'}  {ms:5d} ms  {case['q'][:60]:60s} intent={trace.get('intent')} route={trace.get('route')}")
        for f in fails:
            print(f"        - {f}")
        await asyncio.sleep(PAUSE_S)
    for conv in GOLD["multi_turn"]:
        cid = str(uuid.uuid4())
        print(f"\n# conversation: {conv['name']}")
        for case in conv["turns"]:
            trace, answer, ms = await run_turn(pipeline, cid, case["q"], answers)
            fails = check(case, trace, answer, answers)
            results.append(not fails)
            lat.append(ms)
            print(f"{'PASS' if not fails else 'FAIL'}  {ms:5d} ms  {case['q'][:60]:60s} rewritten={trace.get('rewritten_query')!r}")
            for f in fails:
                print(f"        - {f}")
            await asyncio.sleep(PAUSE_S)
    lat.sort()
    print(f"\nRESULT: {sum(results)}/{len(results)} passed ({100 * sum(results) / len(results):.0f}%) | "
          f"latency p50 {lat[len(lat) // 2]} ms, p95 {lat[int(len(lat) * .95) - 1]} ms")
    await db.close_pool()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None)
