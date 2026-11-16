"""Deterministic follow-up suggestions (zero LLM tokens)."""
from __future__ import annotations

STARTERS = ["Smart hybrid cars under 30 lakhs", "Creta vs Seltos", "Safest SUVs under 15 lakh",
            "Best mileage cars under 10 lakh"]


def short(name: str) -> str:
    for brand in ("Maruti Suzuki ", "Mercedes-Benz ", "Land Rover ", "Toyota Urban Cruiser "):
        if name.startswith(brand):
            return name[len(brand):] if brand != "Toyota Urban Cruiser " else name.replace("Toyota Urban Cruiser ", "Toyota ")
    return name


def suggest(intent: str, names: list[str], rivals: list[str], has_transmission_filter: bool) -> list[str]:
    n = [short(x) for x in names]
    r = [short(x) for x in rivals]
    if intent == "recommend" and len(n) >= 2:
        out = [f"Compare {n[0]} vs {n[1]}", "Which of these is the safest?", "Which one has the best mileage?"]
        if not has_transmission_filter:
            out.append("Show only automatics")
        return out[:4]
    if intent == "recommend" and n:
        return [f"Tell me more about the {n[0]}", f"{n[0]} on-road price in Delhi", "Show cheaper alternatives"]
    if intent == "compare" and len(n) >= 2:
        return ["Which is better for highway driving?", f"Which {n[0]} variant is value for money?",
                f"Is the {n[1]} safer than the {n[0]}?"]
    if n:
        car = n[0]
        pool = {
            "spec": [f"Is the {car} safe?", f"What do owners say about the {car}?", f"Which {car} variant should I buy?"],
            "price": [f"Which {car} variant is value for money?", f"Mileage of the {car}", f"{car} vs {r[0]}" if r else f"Is the {car} safe?"],
            "feature_check": [f"Cheapest {car} variant with a sunroof", f"Safety rating of the {car}", f"{car} on-road price in Mumbai"],
            "safety": [f"Mileage of the {car}", f"{car} vs {r[0]}" if r else f"Pros and cons of the {car}", f"Which {car} variant has ADAS?"],
            "service": [f"Pros and cons of the {car}", f"Real-world mileage of the {car}", f"{car} warranty details"],
            "review": [f"Which {car} variant should I buy?", f"{car} vs {r[0]}" if r else f"Mileage of the {car}", f"Is the {car} safe?"],
            "variant_advice": [f"Does the {car} get a sunroof?", f"{car} on-road price in Delhi", f"Mileage of the {car}"],
        }
        return pool.get(intent, [f"Is the {car} safe?", f"Mileage of the {car}", f"{car} vs {r[0]}" if r else f"Pros and cons of the {car}"])
    return STARTERS[:3]
