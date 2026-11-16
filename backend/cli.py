"""Terminal harness: run one conversation through the full pipeline and print the trace + streamed answer.

Usage: python cli.py "smart hybrid cars under 30 lakhs" "which one has the best mileage?" "what about its safety?"
"""
import asyncio
import json
import sys
import uuid

from app import db
from app.bootstrap import build_pipeline
from app.config import get_settings


async def main(messages: list[str]) -> None:
    pipeline = await build_pipeline(get_settings())
    conv, user = str(uuid.uuid4()), "cli-user-0001"
    for msg in messages:
        print(f"\n{'=' * 100}\nUSER: {msg}")
        async for ev in pipeline.run(conv, user, msg):
            if ev.type == "token":
                print(ev.data["text"], end="", flush=True)
            elif ev.type == "trace":
                t = ev.data
                p = t.get("plan", {})
                print(f"[trace] intent={t.get('intent')} route={t.get('route')} rewritten={t.get('rewritten_query')!r}")
                print(f"        cars={[(r['mention'], r['model_id']) for r in t.get('resolved_cars', [])]} "
                      f"refers_to={p.get('refers_to')} idx={p.get('active_indices')} unresolved={t.get('unresolved')}")
                f = {k: v for k, v in (p.get('filters') or {}).items() if v not in (None, [], 'launched')}
                print(f"        filters={f} mode={p.get('filter_mode')} sort={p.get('sort_by')} prefer={p.get('prefer_tags')}")
                print(f"        sql={t.get('sql_filters')} relaxed={t.get('relaxed')} notes={t.get('notes')}")
                print(f"        sources={[(s['n'], s['car'], s['kind']) for s in t.get('retrieved', [])]}")
                if t.get("cache"):
                    print("        CACHE:", t["cache"])
                print("-" * 100)
            elif ev.type == "cars":
                print("[cars]", [c["name"] for c in ev.data["cars"]])
            elif ev.type in ("error", "followups"):
                print(f"\n[{ev.type}]", json.dumps(ev.data, ensure_ascii=False))
            elif ev.type == "done":
                print("[done]", ev.data.get("timings_ms"), ev.data.get("models"))
    await asyncio.sleep(0.5)
    await db.close_pool()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(sys.argv[1:] or ["smart hybrid cars under 30 lakhs"]), loop_factory=asyncio.SelectorEventLoop
                if sys.platform == "win32" else None)
