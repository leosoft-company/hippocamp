"""Recall benchmark on the synthetic household.

    python evals/recall_bench.py            # full run, checks thresholds.json
    python evals/recall_bench.py --quick    # small + 600-noise stores only

Every question has a known right answer. The same questions are asked of a
73-memory store and of the same store buried under journal-style noise.
Exit code is 1 if any threshold in thresholds.json is missed.
"""
from __future__ import annotations
import json, statistics, sys, time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dataset as D
from common import CachedEmbedder, ROOT, clock, fresh_dir, open_mem

CATS = ["direct", "paraphrase", "rare_token", "entity", "updated", "preference", "episodic", "inference", "ALL"]


def build(mem):
    """Write the household in chronological order. Returns key -> memory id."""
    ids, items = {}, []
    for k, (t, d, sup) in D.FACTS.items(): items.append((d, "fact", k, t, sup))
    for k, (t, d, s) in D.PREFS.items(): items.append((d, "pref", k, t, s))
    for k, (t, d) in D.EPISODES.items(): items.append((d, "ep", k, t, None))
    for d, kind, k, t, extra in sorted(items, key=lambda x: -x[0]):
        with clock(d):
            if kind == "fact":
                ids[k] = mem.assert_fact(t, supersedes=[ids[extra]] if extra else None)
            elif kind == "pref":
                ids[k] = mem.assert_preference(t, strength=extra)
            else:
                ids[k] = mem.observe(t, at=datetime.now(timezone.utc) - timedelta(days=d))
    return ids


def add_noise(mem, rows):
    for text, d in rows:
        mem.observe(text, at=datetime.now(timezone.utc) - timedelta(days=d))


def score(rank_lists, queries, ids):
    per = defaultdict(lambda: {"n": 0, "h1": 0, "h3": 0, "h10": 0})
    misses = []
    for (q, keys, cat), got in zip(queries, rank_lists):
        want = {ids[k] for k in keys}
        pos = next((i for i, g in enumerate(got) if g in want), None)
        for c in (cat, "ALL"):
            s = per[c]; s["n"] += 1
            if pos is not None:
                s["h1"] += pos < 1; s["h3"] += pos < 3; s["h10"] += pos < 10
        if pos is None or pos >= 3:
            misses.append((cat, q, pos))
    return ({c: {"n": s["n"], "hit@1": s["h1"] / s["n"], "hit@3": s["h3"] / s["n"], "hit@10": s["h10"] / s["n"]}
             for c, s in per.items()}, misses)


def measure(mem, ids):
    out, text_of = {}, {}
    sets = {"dev": D.QUERIES, "holdout": D.HOLDOUT_QUERIES}
    lat = []
    for name, qs in sets.items():
        lists, strong = [], 0
        for q, _, _ in qs:
            t = time.time(); hits = mem.recall(q, limit=10); lat.append(time.time() - t)
            lists.append([h.id for h in hits])
            for h in hits: text_of[h.id] = f"[{h.kind}] {h.text[:60]}"
            strong += bool(hits) and getattr(hits[0], "match", None) == "strong"
        table, misses = score(lists, qs, ids)
        again = [[h.id for h in mem.recall(q, limit=10)] for q, _, _ in qs]
        out[name] = {"table": table, "strong": strong / len(qs),
                     "repeat_identical": sum(a == b for a, b in zip(lists, again)) / len(qs),
                     "misses": [{"cat": c, "q": q, "rank": p, "top": text_of.get(lists[[x[0] for x in qs].index(q)][0])}
                                for c, q, p in misses]}
    both = D.QUERIES + D.HOLDOUT_QUERIES
    lists = [[h.id for h in mem.recall(q, limit=10)] for q, _, _ in both]
    out["all"] = {"table": score(lists, both, ids)[0]}
    neg = D.NEGATIVE_QUERIES + D.HOLDOUT_NEGATIVES
    weak = sum((lambda h: not h or getattr(h[0], "match", None) == "weak")(mem.recall(q, limit=3)) for q in neg)
    out["unanswerable_marked_weak"] = weak / len(neg)
    out["answerable_marked_strong"] = (out["dev"]["strong"] * len(D.QUERIES) + out["holdout"]["strong"] * len(D.HOLDOUT_QUERIES)) / len(both)
    out["repeat_identical"] = min(out["dev"]["repeat_identical"], out["holdout"]["repeat_identical"])
    # kind filter: every list should be full (48 facts+preferences exist)
    fp = [x for x in both if x[2] != "episodic"]
    out["kind_filter_full_lists"] = sum(len(mem.recall(q, kinds=["fact", "preference"], limit=10)) == 10 for q, _, _ in fp) / len(fp)
    stale = {}
    for q, new, old in (("Where do we live?", "home_new", "home_old"), ("Do I like aisle or window seats?", "seat_new", "seat_old")):
        got = [h.id for h in mem.recall(q, limit=10)]
        stale[q] = {"current": got.index(ids[new]) + 1 if ids[new] in got else None,
                    "stale": got.index(ids[old]) + 1 if ids[old] in got else None}
    out["stale"] = stale
    out["recall_ms_median"] = round(statistics.median(lat) * 1000, 1)
    return out


def show(label, m):
    print(f"\n=== {label}: {m['memories']} memories | recall median {m['recall_ms_median']} ms")
    print(f"{'category':11s}  n | dev  first top3 top10 | holdout first top3")
    for c in CATS:
        d = m["dev"]["table"].get(c); h = m["holdout"]["table"].get(c)
        print(f"{c:11s} {d['n']:2d} |      {d['hit@1']:.2f}  {d['hit@3']:.2f} {d['hit@10']:.2f} |         {h['hit@1']:.2f}  {h['hit@3']:.2f}")
    print(f"answerable marked strong {m['answerable_marked_strong']:.2f} | unanswerable marked weak {m['unanswerable_marked_weak']:.2f} | "
          f"kind-filter full lists {m['kind_filter_full_lists']:.2f} | repeat identical {m['repeat_identical']:.2f}")
    print("stale:", m["stale"])
    for s in ("dev", "holdout"):
        for x in m[s]["misses"]:
            print(f"  miss[{s}/{x['cat']}] rank={x['rank']} {x['q']!r} -> {x['top']}")


def check(results, thr):
    fails = []
    def need(label, got, want):
        if got + 1e-9 < want: fails.append(f"{label}: {got:.2f} < {want:.2f}")
    for size in ("small", "large"):
        if size not in results: continue
        for key, want in thr.get(size, {}).items():
            cat, metric = key.split(".")
            need(f"{size} {key}", results[size]["all"]["table"][cat][metric], want)
        for k in ("answerable_marked_strong", "kind_filter_full_lists", "repeat_identical"):
            need(f"{size} {k}", results[size][k], thr[k])
    if "large" in results:
        for key, want in thr.get("holdout_large", {}).items():
            cat, metric = key.split(".")
            need(f"holdout large {key}", results["large"]["holdout"]["table"][cat][metric], want)
    return fails


def run(quick=False, verbose=True):
    emb = CachedEmbedder()
    mem = open_mem(fresh_dir("recall_bench") / "store", emb)
    ids = build(mem)
    results = {}
    noise = D.noise(3000)
    plan = [("small", 0), ("medium", 600)] if quick else [("small", 0), ("large", 3000)]
    added = 0
    for label, n in plan:
        add_noise(mem, noise[added:n]); added = n
        inv = mem.inspect()
        m = measure(mem, ids)
        m["memories"] = inv.episodes + inv.facts + inv.preferences + inv.reflections
        results[label] = m
        if verbose: show(label, m)
    emb.save(); mem.close()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "recall_latest.json").write_text(json.dumps(results, indent=1))
    fails = check(results, json.loads((ROOT / "thresholds.json").read_text()))
    return results, fails


if __name__ == "__main__":
    _, fails = run(quick="--quick" in sys.argv)
    thr = json.loads((ROOT / "thresholds.json").read_text())
    for k, want in thr.get("_open", {}).items():
        if not k.startswith("_"):
            print(f"\nOPEN TARGET (not gating) {k}: " + ", ".join(f"{s} {r[k]:.2f}" for s, r in _.items()) + f" (target {want:.2f})")
    print("\nTHRESHOLDS:", "all met" if not fails else f"{len(fails)} missed")
    for f in fails: print("  ✗", f)
    sys.exit(1 if fails else 0)
