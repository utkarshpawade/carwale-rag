"""Token-budgeted context: structured CAR DATA first (dense facts), then numbered SOURCES for citations."""
from __future__ import annotations


def est_tokens(s: str) -> int:
    return len(s) // 4 + 1


def chunk_label(content: str, title: str | None) -> tuple[str, str]:
    first = content.split("\n", 1)[0]
    if " | " not in first and " — " in first:  # free chunks: "Car glossary — ADAS", "CarWale list — Best SUVs ..."
        prefix, sec = first.split(" — ", 1)
        return ("Glossary" if prefix == "Car glossary" else prefix), sec
    car = first.split(" | ")[0]
    sec = first.rsplit(" — ", 1)[-1] if " — " in first else (title or "")
    return car, sec


def build_context(blocks: list[str], chunks: list[dict], budget: int) -> tuple[str, list[dict]]:
    parts: list[str] = []
    used = 0
    if blocks:
        data = "## CAR DATA (CarWale catalog)\n" + "\n\n".join(b for b in blocks if b)
        max_data = int(budget * 0.6)
        if est_tokens(data) > max_data:
            data = data[: max_data * 4] + "\n…(truncated)"
        parts.append(data)
        used += est_tokens(data)
    sources: list[dict] = []
    src_lines: list[str] = []
    for ch in chunks:
        remaining = budget - used
        if remaining < 90:
            break
        text = ch["content"][: min(len(ch["content"]), remaining * 4, 1400)]
        n = len(sources) + 1
        src_lines.append(f"[{n}] {text}")
        car, sec = chunk_label(ch["content"], ch.get("title"))
        sources.append({"n": n, "car": car, "section": sec, "url": ch.get("url"), "model_id": ch.get("model_id"),
                        "kind": ch.get("section"), "score": round(float(ch.get("score") or 0), 4), "via": ch.get("via")})
        used += est_tokens(text)
    if src_lines:
        parts.append("## SOURCES\n" + "\n\n".join(src_lines))
    return "\n\n".join(parts), sources
