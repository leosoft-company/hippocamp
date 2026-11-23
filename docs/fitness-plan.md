# Making Hippocamp fit for a household assistant

Status (2026-11-23): phases 1–5 implemented; every behaviour test and every
gating benchmark threshold passes. One target is open — flagging unanswerable
questions (see "Open" at the end). Baseline numbers in brackets come from the
synthetic-family evaluation of v0.11.1.

Results on the 3,073-memory store, right memory in the top three: dev
questions 57 % → 90 %, holdout questions 47 % → 93 %. Ranked first: 46 % →
82 % (dev), 37 % → 86 % (holdout).

## Purpose

Hippocamp is the memory behind ambi for one household: several people, one
machine, one local model. For that it must:

1. return the right memory when asked, however the question is worded;
2. keep facts current when they change;
3. never lose or swap a fact about a person (allergies, birthdays);
4. consolidate episodes unattended, every night, without losing any;
5. really delete what the owner asks it to forget;
6. keep everything on the machine.

## Rules for this work

- **Evals first.** Every fix starts as a failing eval. Evals stay in the repo
  and gate later changes, so a fix cannot be quietly undone.
- **Synthetic data only.** All evals use the made-up Marlowe family in
  `evals/dataset.py`, in throwaway stores under `evals/runs/`. Nobody's real
  store is opened by the evals, by tests, or by an AI assistant working on
  this repo.
- **Tune on one set, judge on another.** Ranking weights are tuned on the
  `dev` questions; the `holdout` questions are only used to report.
- **No silent format breaks.** Existing stores must open unchanged. New
  columns are added in place; the event log stays readable.

## The evals

| Layer | Where | Needs | Runs |
|---|---|---|---|
| Behaviour tests | `tests/test_fitness.py` | nothing (hash embedder, scripted LLM) | every `pytest` run, CI |
| Recall benchmark | `evals/recall_bench.py` | the real Qwen embedder | `python evals/recall_bench.py`; fails below `evals/thresholds.json` |
| Live reflect | `evals/reflect_live.py` | a local model server | by hand before a release |

`pytest -m quality` runs the recall benchmark as a test when
`HIPPOCAMP_QUALITY=1` is set.

## Phases

Each phase lists the failing evals it must turn green. Baseline in brackets.

### 1. Recall

- Right memory in the top 3, 3,073-memory store: **≥ 90 %** [57 %]; first: **≥ 75 %** [46 %]
- Same on the 73-memory store: top 3 **≥ 95 %** [63 %]
- Reworded questions, top 3: **≥ 85 %** [14 %]; preferences: **≥ 85 %** [10 %]
- Exact identifiers and events stay at **100 %** [100 %]
- A 2-year-old fact beats fresh unrelated chatter [fails]
- Asking for 10 of a kind returns 10 when 10 exist [0–5]
- A fact saved five times comes back once [5 times]
- Unanswerable questions are flagged as weak matches, answerable ones are not [no signal]
- Asking the same question again gives the same answer [8 of 68 changed]
- Non-English memories are found from English questions [fails]

Changes: use the embedder's query instruction; recency only for episodes;
filter by kind before the candidate cut; collapse duplicates on write and on
read; a `match` label on every result; reads no longer write.

### 2. Corrections

- The tool can replace a fact or preference (`supersedes`) [parameter missing]
- A write reports similar memories it may be replacing [nothing]
- The stale value is not returned once replaced [stale value ranked first]
- Reflect replaces a changed value and never replaces a different fact
  [both kept; look-alike facts at risk]
- Empty text, unknown kinds, unknown ids and bad limits are errors [accepted]
- Preference strength affects ranking [ignored]

### 3. Reflect

- 80 pending episodes are all read [30 never read]
- Episodes older than 7 days, or arriving late from another device, are read [never]
- Works with a thinking model on 30+ episodes [fails at 29]
- A dead process's lock does not block the next run [1 hour]
- Episodes are read to 1,500 characters [300]
- Re-running does not duplicate reflections [duplicates]
- The model is shown the existing memories relevant to the batch, with ids [30 picked by a fixed query]

Changes: track which episodes have been read (a `mark_reflected` event)
instead of a clock cutoff; batches; the model names what it replaces.

### 4. Forgetting

- `purge` removes the text from the event log and the cache [text stays]
- `compact` purges everything already forgotten [empty stub]
- A forgotten memory stays forgotten after a rebuild, whatever the timestamps [comes back]
- Expiry is available from the CLI [library only]
- Store files are readable by the owner only [world-readable]

### 5. Hardening

- A half-written log line is skipped, not fatal [store cannot be rebuilt]
- A device picks up another device's new events when it opens the store [manual sync]
- The embedder loads without contacting the network when the model is cached [contacts Hugging Face]
- README matches the code

## Open

- **Flagging unanswerable questions.** Target: 70 % of questions the store
  cannot answer come back marked weak. Achieved: 28–44 %, with 96 % of
  answerable questions marked strong. No score-based signal separates the two
  (best balanced accuracy 0.63–0.79 over similarity, gaps and z-scores, for
  Qwen3 and EmbeddingGemma). It needs a reranker or a model judging
  relevance. Reported by the benchmark, not gating.
- **Embedder choice.** By similarity alone EmbeddingGemma-300m beats
  Qwen3-0.6B on this data (large store, top three: 92 % vs 82 %); see
  `evals/compare_embedders.py`. Switching needs an embedder adapter and a
  `reindex` of each store.
- **Inference questions** ("is satay sauce okay for the kids?") stay the
  weakest category (50–62 % top three). Same remedy as the first point.

## Not in this plan

Encryption at rest; storing embeddings once instead of twice; the multi-user
work in ambi, which starts after phase 3 is green.
