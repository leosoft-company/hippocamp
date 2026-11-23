"""Shared helpers for the evals. Stores live under evals/runs/ only — the
evals never open a real store."""
from __future__ import annotations
import hashlib, json, os, pickle, shutil, time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
EMB_CACHE = ROOT / ".emb_cache.pkl"
REAL_STORE = Path.home() / ".hippocamp"


def fresh_dir(name: str) -> Path:
    d = (RUNS / name).resolve()
    assert RUNS.resolve() in d.parents, "evals only write under evals/runs"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


class CachedEmbedder:
    """Real Qwen embedder with an on-disk memo so re-runs are fast.
    Keeps the real embedder's name/dim so stores commit to the same model."""

    def __init__(self, which: str | None = None):
        which = which or os.environ.get("EVAL_EMBEDDER", "qwen")
        t = time.time()
        if which == "gemma":
            from hippocamp.embedders import embeddinggemma
            self.inner = embeddinggemma()
        else:
            from hippocamp.embedders import Qwen3Embedder
            self.inner = Qwen3Embedder()
        self.load_s = time.time() - t
        self._ns = "" if which == "qwen" else which + ":"
        self.name, self.dim = self.inner.name, self.inner.dim
        for attr in ("strong_similarity", "similar_statement"):
            if hasattr(self.inner, attr):
                setattr(self, attr, getattr(self.inner, attr))
        self.memo = pickle.loads(EMB_CACHE.read_bytes()) if EMB_CACHE.exists() else {}
        self.calls = self.misses = 0
        self.raw_times: list[float] = []

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        k = self._ns + hashlib.sha1(text.encode()).hexdigest()
        v = self.memo.get(k)
        if v is None:
            self.misses += 1
            t = time.time()
            v = self.inner(text)
            self.raw_times.append(time.time() - t)
            self.memo[k] = v
        return v

    def embed_query(self, text: str) -> list[float]:
        """Query-side embedding, if the embedder distinguishes it."""
        fn = getattr(self.inner, "embed_query", None)
        if fn is None:
            return self(text)
        k = self._ns + "q:" + hashlib.sha1(text.encode()).hexdigest()
        v = self.memo.get(k)
        if v is None:
            v = self.memo[k] = fn(text)
        return v

    def save(self):
        EMB_CACHE.write_bytes(pickle.dumps(self.memo))


@contextmanager
def clock(days_ago: float):
    """Make Hippocamp believe 'now' is N days in the past (for backdated writes)."""
    import hippocamp.memory as hm
    real = hm._now
    hm._now = lambda: datetime.now(timezone.utc) - timedelta(days=days_ago)
    try:
        yield
    finally:
        hm._now = real


def open_mem(store: Path, emb, device="eval-dev-a", llm=None):
    from hippocamp.memory import Memory
    return Memory(str(store), embedder=emb, device_id=device,
                  cache_dir=str(store.parent / (store.name + "-cache")), llm=llm,
                  reranker=reranker())


_RR = []


def reranker():
    """The reranker under test, when EVAL_RERANKER=1 (loaded once)."""
    if os.environ.get("EVAL_RERANKER") != "1":
        return None
    if not _RR:
        from hippocamp.rerank import Qwen3Reranker
        _RR.append(Qwen3Reranker())
    return _RR[0]
