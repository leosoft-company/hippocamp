"""Ranker: similarity first, then small nudges for wording, kind and age.

Similarity decides. The other signals only reorder memories that are about
equally relevant — they must never lift an unrelated memory over a relevant
one. In particular, age: a fact, preference or reflection stays true until
it is replaced, so it does not fade. Only episodes ("what happened") lose
ground as they get older, and only a little.

The weights below are checked by evals/recall_bench.py; change them there
first (see docs/fitness-plan.md).

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
W_LEX = 0.10         # share of the query's distinctive words found in the memory
W_KIND = 1.0         # × (boost − 1): preference +0.07, fact +0.05, reflection +0.025
W_REC_EPISODE = 0.06  # episodes: newer wins among similar ones (30-day half-life)
W_REC_DURABLE = 0.03  # facts etc.: a tie-break only, so the newer of two
                      # conflicting statements comes first (1-year half-life)
W_SAL = 0.0          # usage is no longer a ranking signal: reads don't write

# A result is a "strong" match when its similarity clears the embedder's bar
# (Embedder.strong_similarity, else this) or it contains every distinctive
# word of a short query (an exact identifier such as a plate or a codename).
DEFAULT_STRONG_SIMILARITY = 0.5

# Two episodes this alike are the same routine entry said twice; only the
# better one competes for the top places (the other moves to the tail).
NEAR_DUPLICATE_EPISODE = 0.90

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
    # Durable knowledge edges out journal entries that merely mention the
    # same topic ("prefers trains" over "asked for train times").
    "preference": 1.07,
    "fact": 1.05,
    "reflection": 1.025,
    "episode": 1.0,
}

RECENCY_HALFLIFE_DAYS = 30.0
DURABLE_HALFLIFE_DAYS = 365.0


def rank(
    rows: Iterable[RawRow],
    *,
    query_emb: list[float],
    context: dict[str, Any] | None,
    now: datetime,
    query_tokens: set[str] | None = None,
    strong_similarity: float | None = None,
) -> list[RecallResult]:
    results: list[RecallResult] = []
    tokens = query_tokens or set()
    bar = DEFAULT_STRONG_SIMILARITY if strong_similarity is None else strong_similarity
    for row in rows:
        sim = _cosine(query_emb, row.embedding)
        episode = row.kind == "episode"
        rec = _recency_decay(
            row.created_at, now,
            RECENCY_HALFLIFE_DAYS if episode else DURABLE_HALFLIFE_DAYS,
        )
        kb = KIND_BOOSTS.get(row.kind, 1.0)
        if row.kind == "preference":
            # A weakly held preference gets less of the preference boost.
            kb = 1.0 + (kb - 1.0) * _strength(row.metadata)
        sal = 0.0
        lex = 0.0
        if tokens:
            text_tokens = tokenize(row.text)
            lex = len(tokens & text_tokens) / len(tokens)

        total = (W_SIM * sim
                 + (W_REC_EPISODE if episode else W_REC_DURABLE) * rec
                 + W_KIND * (kb - 1.0) + W_LEX * lex)
        strong = sim >= bar or (lex == 1.0 and 0 < len(tokens) <= 3)

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
                match="strong" if strong else "weak",
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


try:  # optional: present whenever a real embedder is installed
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if _np is not None and len(a) == len(b) and len(a) > 64:
        va, vb = _np.asarray(a, dtype=_np.float32), _np.asarray(b, dtype=_np.float32)
        na, nb = float(_np.linalg.norm(va)) or 1e-9, float(_np.linalg.norm(vb)) or 1e-9
        return float(va @ vb) / (na * nb)
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n])) or 1e-9
    nb = math.sqrt(sum(x * x for x in b[:n])) or 1e-9
    return dot / (na * nb)


def _recency_decay(
    created_at: datetime, now: datetime, halflife_days: float = RECENCY_HALFLIFE_DAYS
) -> float:
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return 0.5 ** (age_days / halflife_days)


def _strength(metadata: dict) -> float:
    try:
        return min(1.0, max(0.0, float(metadata.get("strength", 1.0))))
    except (TypeError, ValueError):
        return 1.0


SALIENCE_CAP = 0.2


def _salience_factor(salience: float) -> float:
    return min(math.log1p(max(0.0, salience)), SALIENCE_CAP)


def _compose_note(sim: float, rec: float, kb: float, sal: float) -> str:
    parts = [f"sim={sim:.2f}"]
    if rec < 0.95:
        parts.append(f"age×{rec:.2f}")
    if kb != 1.0:
        parts.append(f"kind×{kb:.2f}")
    if sal > 0:
        parts.append(f"salience×{1 + sal:.2f}")
    return ", ".join(parts)
