"""Offline ranking explorer. Reads the store left by recall_bench.py and
replays ranking with numpy so weight changes can be tried in seconds.
Tune on the dev questions only; holdout is for recall_bench.py to report.

    python evals/tune.py
"""
from __future__ import annotations
import itertools, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dataset as D
from common import CachedEmbedder, ROOT, open_mem
from hippocamp.ranker import tokenize

emb = CachedEmbedder()
mem = open_mem(ROOT / "runs" / "recall_bench" / "store", emb)
rows = mem._store._conn.execute(
    "SELECT id, kind, text, created_at, embedding, metadata FROM memories WHERE superseded_at IS NULL").fetchall()
ID = [r[0] for r in rows]; KIND = np.array([r[1] for r in rows]); TEXT = [r[2] for r in rows]
M = np.array([json.loads(r[4]) for r in rows], dtype=np.float32)
now = datetime.now(timezone.utc)
AGE = np.array([(now - datetime.fromisoformat(r[3])).total_seconds() / 86400 for r in rows])
TOK = [tokenize(t) for t in TEXT]
EP = KIND == "episode"
key_of = {}
for kind, table in (("fact", D.FACTS), ("preference", D.PREFS), ("episode", D.EPISODES)):
    for k, v in table.items():
        key_of[v[0]] = k
KEY = [key_of.get(t) for t in TEXT]
SIGNAL = np.array([k is not None for k in KEY])


def feats(q):
    v = np.array(emb.embed_query(q), dtype=np.float32)
    sim = M @ v
    qt = tokenize(q)
    lex = np.array([len(qt & t) / len(qt) if qt else 0.0 for t in TOK])
    return sim, lex, len(qt)

F = {q: feats(q) for q, _, _ in D.QUERIES + D.HOLDOUT_QUERIES}
FN = {q: feats(q) for q in D.NEGATIVE_QUERIES + D.HOLDOUT_NEGATIVES}


def ranked(sim, lex, p, mask, k=40, limit=10):
    kb = np.where(KIND == "preference", p["pref"], np.where(KIND == "fact", p["fact"], np.where(EP, 0.0, p["refl"])))
    rec = np.where(EP, p["rec_ep"] * 0.5 ** (AGE / 30), 0.03 * 0.5 ** (AGE / 365))
    score = sim + p["lex"] * lex + kb + rec
    score = np.where(mask, score, -9)
    # candidate cut: nearest k durable + nearest k episodes (+ lexical hits), as the store does
    cand = np.zeros(len(ID), bool)
    for sel in (EP & mask, ~EP & mask):
        idx = np.where(sel)[0]
        cand[idx[np.argsort(-sim[idx])[:k]]] = True
    cand |= (lex > 0) & mask
    order = [i for i in np.argsort(-score) if cand[i]][:30]
    out = []
    for i in order:
        if p["div"] < 1 and any(float(M[i] @ M[j]) >= p["div"] for j in out if KIND[i] == KIND[j] == "episode"):
            continue
        out.append(i)
        if len(out) == limit: break
    return out


def evaluate(p, queries, mask):
    h1 = h3 = 0
    for q, keys, _ in queries:
        sim, lex, _ = F[q]
        got = [KEY[i] for i in ranked(sim, lex, p, mask)]
        pos = next((n for n, g in enumerate(got) if g in keys), None)
        h1 += pos == 0; h3 += pos is not None and pos < 3
    return h1 / len(queries), h3 / len(queries)


if __name__ == "__main__":
    ALL = np.ones(len(ID), bool)
    base = dict(lex=0.15, pref=0.04, fact=0.02, refl=0.01, rec_ep=0.06, div=1.0)
    print("current   small", evaluate(base, D.QUERIES, SIGNAL), "large", evaluate(base, D.QUERIES, ALL))
    best = []
    for lex, dur, rec, div in itertools.product([0.0, 0.1, 0.15, 0.25], [0.02, 0.05, 0.08, 0.12, 0.16], [0.0, 0.03, 0.06], [1.0, 0.95, 0.9, 0.85]):
        p = dict(lex=lex, pref=dur + 0.02, fact=dur, refl=dur / 2, rec_ep=rec, div=div)
        s, l = evaluate(p, D.QUERIES, SIGNAL), evaluate(p, D.QUERIES, ALL)
        best.append((l[1] + s[1] + 0.5 * (l[0] + s[0]), p, s, l))
    best.sort(key=lambda x: -x[0])
    for sc, p, s, l in best[:8]:
        print(f"{sc:.3f} small {s[0]:.2f}/{s[1]:.2f} large {l[0]:.2f}/{l[1]:.2f}  {p}")
    # strong-match bar: similarity of the right answer vs best similarity for unanswerable questions
    right = []
    for q, keys, _ in D.QUERIES:
        sim = F[q][0]; right.append(max(sim[i] for i, k in enumerate(KEY) if k in keys))
    negs = [float(FN[q][0].max()) for q in D.NEGATIVE_QUERIES]
    print("sim of right answer (dev): ", np.round(np.percentile(right, [0, 10, 25, 50, 75]), 2))
    print("top sim, unanswerable (dev):", np.round(np.percentile(negs, [25, 50, 75, 90, 100]), 2))
    for bar in (0.35, 0.4, 0.45, 0.5, 0.55):
        print(f"  bar {bar}: answerable strong {np.mean(np.array(right) >= bar):.2f} | unanswerable weak {np.mean(np.array(negs) < bar):.2f}")
