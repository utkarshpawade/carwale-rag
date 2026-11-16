"""Resolve free-text car mentions ('cretta', 'hycross', 'xuv700', 'nexon ev') to catalog model ids.

Strong aliases (full name, model name, slug) beat weak ones (first/last token, prefixes); fuzzy matching
(rapidfuzz WRatio) handles typos. Ambiguous mentions ('innova') return every close candidate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

BRAND_ALIASES = {
    "maruti-suzuki": ["maruti", "suzuki", "maruti suzuki"],
    "mercedes-benz": ["mercedes", "benz", "merc", "mercedes benz"],
    "volkswagen": ["vw"],
    "land-rover": ["land rover", "landrover"],
    "rolls-royce": ["rolls royce", "rolls"],
    "aston-martin": ["aston martin", "aston"],
    "force-motors": ["force"],
    "mg": ["morris garages"],
}
# old names / nicknames -> model id (only applied when the target exists)
MANUAL_ALIASES = {
    "xuv700": "mahindra/xuv-7xo", "xuv 700": "mahindra/xuv-7xo", "xuv7xo": "mahindra/xuv-7xo",
    "xuv300": "mahindra/xuv-3xo", "xuv 300": "mahindra/xuv-3xo",
    "swift dzire": "maruti-suzuki/dzire", "grand i10": "hyundai/grand-i10-nios", "i10": "hyundai/grand-i10-nios",
    "city hybrid": "honda/city", "city ehev": "honda/city", "city e hev": "honda/city",
    "g wagon": "mercedes-benz/g-class", "g wagen": "mercedes-benz/g-class",
}
FILLER = {"new", "the", "car", "cars", "model", "latest", "2025", "2026", "2027", "suv", "india", "price", "a"}
GENERIC_TOKENS = {"grand", "range", "new", "city", "sport", "series", "class", "line", "plus", "max", "pro",
                  "electric", "ev", "lwb", "coupe", "next", "gen", "facelift", "edition", "urban", "cruiser"}


def norm(s: str) -> str:
    s = s.lower().replace("&", " and ").replace("+", " plus ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Resolved:
    model_id: str
    full_name: str
    score: float
    alias: str
    ambiguous: list[str] = field(default_factory=list)


class CarResolver:
    def __init__(self, models: list[dict]):
        """models: rows with id, brand, brand_slug, name, slug, full_name, status, user_rating_count."""
        self.models = {m["id"]: m for m in models}
        self.strong: dict[str, set[str]] = {}
        self.weak: dict[str, set[str]] = {}
        self.brand_words: dict[str, str] = {}
        for m in models:
            for a in self._strong_aliases(m):
                self.strong.setdefault(a, set()).add(m["id"])
            for a in self._weak_aliases(m):
                self.weak.setdefault(a, set()).add(m["id"])
            self.brand_words[norm(m["brand"])] = m["brand_slug"]
            for b in BRAND_ALIASES.get(m["brand_slug"], []):
                self.brand_words[norm(b)] = m["brand_slug"]
        for alias, mid in MANUAL_ALIASES.items():
            if mid in self.models:
                self.strong.setdefault(norm(alias), set()).add(mid)
        for a in list(self.weak):  # a strong alias always wins over a weak one
            if a in self.strong:
                del self.weak[a]
        self.keys = list(self.strong) + list(self.weak)

    @staticmethod
    def _strong_aliases(m: dict) -> set[str]:
        name, brand, slug = norm(m["name"]), norm(m["brand"]), norm(m["slug"])
        out = {name, slug, f"{brand} {name}", name.replace(" ", ""), slug.replace(" ", "")}
        out |= {f"{norm(b)} {name}" for b in BRAND_ALIASES.get(m["brand_slug"], [])}
        if " electric" in f" {name}":
            out.add(name.replace("electric", "ev"))
        if name.endswith(" ev"):
            out.add(name[:-3] + " electric")
        return {a for a in out if a}

    @staticmethod
    def _weak_aliases(m: dict) -> set[str]:
        toks = norm(m["name"]).split()
        out = set()
        if len(toks) >= 2:
            if len(toks[-1]) >= 4 and toks[-1] not in GENERIC_TOKENS:
                out.add(toks[-1])
            if len(toks[0]) >= 4 and toks[0] not in GENERIC_TOKENS:
                out.add(toks[0])
        if len(toks) >= 3:
            out.add(" ".join(toks[:2]))
            out.add("".join(toks[:2]))
        if toks and toks[-1] == "lwb":
            out.add(" ".join(toks[:-1]))
        return out

    def _rank(self, ids, score: float, alias: str, prefer_upcoming: bool) -> list[Resolved]:
        def key(mid):
            m = self.models[mid]
            status_rank = {"launched": 0, "upcoming": 1, "discontinued": 2}.get(m.get("status"), 3)
            if prefer_upcoming:
                status_rank = {"upcoming": 0, "launched": 1}.get(m.get("status"), 3)
            return (status_rank, -(m.get("user_rating_count") or 0))
        ordered = sorted(ids, key=key)
        return [Resolved(mid, self.models[mid]["full_name"], score, alias, [x for x in ordered if x != mid])
                for mid in ordered[:1]] + [Resolved(mid, self.models[mid]["full_name"], score, alias) for mid in ordered[1:]]

    def resolve(self, mention: str) -> list[Resolved]:
        """Best match first. If ambiguous, the first result lists the alternatives in .ambiguous."""
        q_full = norm(mention)
        if not q_full:
            return []
        prefer_upcoming = any(w in q_full.split() for w in ("upcoming", "facelift", "next", "nextgen"))
        strip = lambda s: " ".join(w for w in s.split() if w not in FILLER)  # noqa: E731
        brand_slug, rest = None, q_full
        for bw, bs in sorted(self.brand_words.items(), key=lambda kv: -len(kv[0])):
            if q_full == bw or q_full.startswith(bw + " "):
                brand_slug, rest = bs, q_full[len(bw):].strip()
                break
        if brand_slug and not strip(rest):
            return []  # brand only, not a model
        cands = list(dict.fromkeys(c for c in (q_full, strip(q_full), rest, strip(rest)) if c))
        for c in cands:  # exact alias hits: full mention first, so 'model y' is not reduced to 'y'
            for table, score in ((self.strong, 100.0), (self.weak, 95.0)):
                ids = table.get(c)
                if ids:
                    ids = [i for i in ids if not brand_slug or self.models[i]["brand_slug"] == brand_slug] or list(ids)
                    return self._rank(ids, score, c, prefer_upcoming)
        q = strip(q_full) or q_full
        if len(q.replace(" ", "")) < 3:
            return []
        hits = process.extract(q, self.keys, scorer=fuzz.WRatio, limit=12, score_cutoff=84)
        if not hits:
            return []
        best: dict[str, tuple[float, str]] = {}
        for alias, score, _ in hits:
            for mid in self.strong.get(alias, set()) | self.weak.get(alias, set()):
                if brand_slug and self.models[mid]["brand_slug"] != brand_slug:
                    continue
                adj = score - (3 if alias in self.weak else 0) + min(len(alias), 20) * 0.05
                if mid not in best or adj > best[mid][0]:
                    best[mid] = (adj, alias)
        if not best:
            return []
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], {"launched": 0, "upcoming": 1}.get(self.models[kv[0]].get("status"), 2)))
        top_id, (top_score, top_alias) = ranked[0]
        close = [mid for mid, (s, _) in ranked[1:] if top_score - s <= 1.5]
        out = [Resolved(top_id, self.models[top_id]["full_name"], round(top_score, 1), top_alias, close)]
        out += [Resolved(mid, self.models[mid]["full_name"], round(s, 1), a) for mid, (s, a) in ranked[1:4]]
        return out

    def brand_slug(self, text: str) -> str | None:
        return self.brand_words.get(norm(text))

    def name(self, mid: str) -> str:
        m = self.models.get(mid)
        return m["full_name"] if m else mid
