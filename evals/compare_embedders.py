"""Compare embedding models on the synthetic household, similarity only.

    python evals/compare_embedders.py

Qwen3 comes from the benchmark's cache; EmbeddingGemma is asked from a local
OpenAI-compatible server (LM Studio on 127.0.0.1:1234). Nothing leaves the
machine, and only the made-up dataset is embedded.
"""
from __future__ import annotations
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dataset as D
from common import CachedEmbedder

texts, keys = [], []
for table in (D.FACTS, D.PREFS, D.EPISODES):
    for k, v in table.items():
        if k in ("job_old",): continue      # superseded in the benchmark
        texts.append(v[0]); keys.append(k)
n_signal = len(texts)
for t, _ in D.noise(3000):
    texts.append(t); keys.append(None)
QS = D.QUERIES + D.HOLDOUT_QUERIES
NEG = D.NEGATIVE_QUERIES + D.HOLDOUT_NEGATIVES


def lmstudio(model, items, prefix):
    out = []
    for i in range(0, len(items), 64):
        req = urllib.request.Request("http://127.0.0.1:1234/v1/embeddings", data=json.dumps(
            {"model": model, "input": [prefix + t for t in items[i:i + 64]]}).encode(), headers={"Content-Type": "application/json"})
        out += [d["embedding"] for d in json.loads(urllib.request.urlopen(req, timeout=300).read())["data"]]
    a = np.array(out, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def report(name, Mx, Qx, Nx):
    for label, n in (("small", n_signal), ("large", len(texts))):
        sims = Qx @ Mx[:n].T
        h1 = h3 = 0; right = []
        for row, (q, want, _) in zip(sims, QS):
            order = np.argsort(-row)[:3]
            got = [keys[i] for i in order]
            h1 += got[0] in want; h3 += any(g in want for g in got)
            right.append(max(row[i] for i in range(n_signal) if keys[i] in want))
        neg = (Nx @ Mx[:n].T).max(axis=1)
        best = max(((np.mean(np.array(right) >= b) + np.mean(neg < b)) / 2, b) for b in np.arange(0.2, 0.9, 0.01))
        print(f"{name:34s} {label}: first {h1 / len(QS):.2f} top3 {h3 / len(QS):.2f} | answerable vs unanswerable best balanced accuracy {best[0]:.2f} at {best[1]:.2f}")


emb = CachedEmbedder()
t = time.time()
Mq = np.array([emb(x) for x in texts], dtype=np.float32)
report("Qwen3-0.6B, no query instruction", Mq, np.array([emb(q) for q, _, _ in QS], dtype=np.float32), np.array([emb(q) for q in NEG], dtype=np.float32))
report("Qwen3-0.6B, query instruction", Mq, np.array([emb.embed_query(q) for q, _, _ in QS], dtype=np.float32), np.array([emb.embed_query(q) for q in NEG], dtype=np.float32))
emb.save()
for model in ("text-embedding-embeddinggemma-300m-qat",):
    try:
        t = time.time()
        Mg = lmstudio(model, texts, "title: none | text: ")
        dt = time.time() - t
        Qg = lmstudio(model, [q for q, _, _ in QS], "task: search result | query: ")
        Ng = lmstudio(model, NEG, "task: search result | query: ")
        report(f"EmbeddingGemma-300m ({Mg.shape[1]}d)", Mg, Qg, Ng)
        print(f"   embedded {len(texts)} memories in {dt:.0f}s via LM Studio")
        Qg2 = lmstudio(model, [q for q, _, _ in QS], "task: question answering | query: ")
        report("EmbeddingGemma, QA task prompt", Mg, Qg2, lmstudio(model, NEG, "task: question answering | query: "))
    except Exception as e:
        print(model, "failed:", e)
