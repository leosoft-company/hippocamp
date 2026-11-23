"""Does a local reranker fix what similarity cannot — flagging unanswerable
questions and one-step-inference questions?

    python evals/rerank_experiment.py     # after recall_bench.py has built its store

Takes the top 20 from the current ranking, re-scores each (question, memory)
pair with Qwen/Qwen3-Reranker-0.6B, and compares. Thresholds are picked on the
dev questions and reported on holdout.
"""
from __future__ import annotations
import sys, time
import numpy as np
import torch
from sentence_transformers import CrossEncoder
import tune as T
import dataset as D

INSTRUCTIONS = {
    "web-search (model default)": None,
    "assistant-memory": "Given a question put to a personal assistant, judge whether the stored memory contains information that helps answer it",
}
P = dict(lex=0.10, pref=0.07, fact=0.05, refl=0.025, rec_ep=0.06, div=0.90)
TOPN = 20


def candidates(feats, mask):
    sim, lex, _ = feats
    return T.ranked(sim, lex, P, mask, limit=TOPN)


def run(model, mask, label):
    out = {}
    for name, qs, negs in (("dev", D.QUERIES, D.NEGATIVE_QUERIES), ("holdout", D.HOLDOUT_QUERIES, D.HOLDOUT_NEGATIVES)):
        h1 = h3 = b1 = b3 = 0; top_pos, top_neg, lat = [], [], []
        for q, keys, cat in qs:
            cand = candidates(T.F[q], mask)
            t = time.time()
            s = model.predict([(q, T.TEXT[i]) for i in cand], activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
            lat.append(time.time() - t)
            order = [cand[i] for i in np.argsort(-s)]
            got = [T.KEY[i] for i in order]; base = [T.KEY[i] for i in cand]
            h1 += got[0] in keys; h3 += any(g in keys for g in got[:3])
            b1 += base[0] in keys; b3 += any(g in keys for g in base[:3])
            top_pos.append(float(s.max()))
        for q in negs:
            cand = candidates(T.FN[q], mask)
            s = model.predict([(q, T.TEXT[i]) for i in cand], activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
            top_neg.append(float(s.max()))
        out[name] = dict(n=len(qs), base=(b1 / len(qs), b3 / len(qs)), rr=(h1 / len(qs), h3 / len(qs)),
                         pos=np.array(top_pos), neg=np.array(top_neg), ms=1000 * float(np.median(lat)))
    d, h = out["dev"], out["holdout"]
    bar = max(((np.mean(d["pos"] >= b) + np.mean(d["neg"] < b)) / 2, b) for b in np.arange(0.02, 0.99, 0.01))[1]
    print(f"  {label}: rerank {TOPN} pairs in {d['ms']:.0f} ms")
    for name, r in out.items():
        print(f"    {name:8s} first/top3  before {r['base'][0]:.2f}/{r['base'][1]:.2f}  after {r['rr'][0]:.2f}/{r['rr'][1]:.2f}"
              f" | bar {bar:.2f} (set on dev): answerable kept {np.mean(r['pos'] >= bar):.2f}, unanswerable flagged {np.mean(r['neg'] < bar):.2f}")


for iname, instr in INSTRUCTIONS.items():
    kw = {} if instr is None else dict(prompts={"mem": instr}, default_prompt_name="mem")
    model = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", local_files_only=True, **kw)
    print(f"\ninstruction: {iname}")
    run(model, T.SIGNAL, "small store")
    run(model, np.ones(len(T.ID), bool), "large store")
