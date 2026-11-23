"""Optional second-stage reranker.

An embedder can tell that a memory is *about* the same thing as a question;
it cannot tell whether the memory *answers* it. A reranker reads the question
and the memory together and can. With one configured, recall re-scores its
best candidates, which does two things (measured in evals/):

* the right memory is ranked first more often;
* results that do not answer the question are labelled `match: "weak"`, so
  "nothing relevant is stored" becomes visible.

It is opt-in — a second model in memory (about 1.2 GB) and roughly 0.1 s per
recall. Enable with HIPPOCAMP_RERANKER=qwen3, or pass `reranker=` to Memory.
A reranker is any callable `(query, texts) -> scores in [0, 1]` with a
`strong_score` attribute.
"""

from __future__ import annotations

import os
from typing import Protocol

RERANKER_ENV = "HIPPOCAMP_RERANKER"


class Reranker(Protocol):
    strong_score: float

    def __call__(self, query: str, texts: list[str]) -> list[float]: ...


class Qwen3Reranker:
    """Qwen/Qwen3-Reranker-0.6B via sentence-transformers, run locally."""

    name = "Qwen3-Reranker-0.6B"
    # Calibrated on evals/ (rerank_experiment.py): at this bar ~90% of
    # unanswerable questions have no strong result. It is strict — a
    # minority of answerable questions also fall below it, so "weak" means
    # "may not answer this", not "wrong".
    strong_score = 0.03
    INSTRUCTION = (
        "Given a question put to a personal assistant, judge whether the "
        "stored memory contains information that helps answer it"
    )

    def __init__(self, model_id: str = "Qwen/Qwen3-Reranker-0.6B") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "Qwen3Reranker requires sentence-transformers. "
                "Install with: pip install 'hippocamp[embeddings]'"
            ) from e
        kw = dict(prompts={"memory": self.INSTRUCTION}, default_prompt_name="memory")
        try:  # no network once the model is on disk
            self._model = CrossEncoder(model_id, local_files_only=True, **kw)
        except Exception:
            self._model = CrossEncoder(model_id, **kw)

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        import torch

        scores = self._model.predict(
            [(query, t) for t in texts],
            activation_fn=torch.nn.Sigmoid(),
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def from_env() -> Reranker | None:
    """The reranker named by $HIPPOCAMP_RERANKER, or None when unset."""
    name = os.environ.get(RERANKER_ENV, "").strip().lower()
    if name in ("", "0", "none", "off"):
        return None
    if name in ("qwen3", "qwen", "qwen3-reranker-0.6b"):
        return Qwen3Reranker()
    raise ValueError(f"unknown reranker {name!r} in ${RERANKER_ENV} (known: qwen3)")
