"""Five-signal ranker: similarity + recency + kind-boost + salience + lexical.

The lexical term exists because small embedders are anisotropic: generic
memories score ~0.6 cosine against *any* query and act as attractors,
while a memory literally containing the query's rare token ("SPC") can
rank below them. Distinctive-token overlap cuts through that.

Each ranked result carries a `Why` trace that explains how its score was
composed. Surfacing the *why* is one of Hippocamp's wedges over single-
bucket vector stores.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable

from hippocamp.types import RawRow, RecallResult, Why

W_SIM = 1.0
W_REC = 0.3
W_KIND = 0.2
W_SAL = 0.4
W_LEX = 0.5

# Tokens too common to signal anything — kept deliberately small.
STOPWORDS = frozenset(
    "the and for with that this from have has had are was were what when "
    "where which who how you your our their they them its can could should "
    "would will been being about into over under not but all any some out "
    "get got does did just like more most very much".split()
)


def tokenize(text: str) -> set[str]:
    """Distinctive lowercase tokens (len ≥ 3, non-stopword) for lexical
    matching — shared by recall's retrieval and the ranker."""
    import re
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 3 and t not in STOPWORDS
    }

KIND_BOOSTS: dict[str, float] = {
    "preference": 1.2,
    "fact": 1.1,
    "reflection": 1.05,
    "episode": 1.0,
}

RECENCY_HALFLIFE_DAYS = 30.0


def rank(
    rows: Iterable[RawRow],
    *,
    query_emb: list[float],
    context: dict[str, Any] | None,
    now: datetime,
    query_tokens: set[str] | None = None,
) -> list[RecallResult]:
    results: list[RecallResult] = []
    tokens = query_tokens or set()
    for row in rows:
        sim = _cosine(query_emb, row.embedding)
        rec = _recency_decay(row.created_at, now)
        kb = KIND_BOOSTS.get(row.kind, 1.0)
        sal = _salience_factor(row.salience)
        lex = 0.0
        if tokens:
            text_tokens = tokenize(row.text)
            lex = len(tokens & text_tokens) / len(tokens)

        total = (W_SIM * sim + W_REC * rec + W_KIND * (kb - 1.0)
                 + W_SAL * sal + W_LEX * lex)

        why = Why(
            similarity=sim,
            recency=rec,
            kind_boost=kb,
            salience=sal,
            lexical=lex,
            total=total,
            note=_compose_note(sim, rec, kb, sal),
        )

        results.append(
            RecallResult(
                id=row.id,
                kind=row.kind,
                text=row.text,
                score=total,
                why=why,
                created_at=row.created_at,
                metadata=row.metadata,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n])) or 1e-9
    nb = math.sqrt(sum(x * x for x in b[:n])) or 1e-9
    return dot / (na * nb)


def _recency_decay(created_at: datetime, now: datetime) -> float:
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)


SALIENCE_CAP = 0.2


def _salience_factor(salience: float) -> float:
    return min(math.log1p(max(0.0, salience)), SALIENCE_CAP)


def _compose_note(sim: float, rec: float, kb: float, sal: float) -> str:
    parts = [f"sim={sim:.2f}"]
    if rec < 0.95:
        parts.append(f"recency×{rec:.2f}")
    if kb != 1.0:
        parts.append(f"kind×{kb:.2f}")
    if sal > 0:
        parts.append(f"salience×{1 + sal:.2f}")
    return ", ".join(parts)
