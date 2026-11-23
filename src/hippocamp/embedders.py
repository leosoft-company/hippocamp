"""Embedders turn text into fixed-size float vectors.

`HashEmbedder` is a deterministic stub for tests and CI.
`FastembedBgeEmbedder` runs BGE-small via fastembed (ONNX Runtime) —
the lightweight default, ~150MB install footprint.
`BgeSmallEmbedder` runs the same BGE-small weights via
sentence-transformers / PyTorch — kept for compatibility (~900MB
install).
`Qwen3Embedder` runs a larger, higher-quality local model via
sentence-transformers.
`default_embedder()` picks the best available — Qwen3 if installed,
else fastembed-BGE, else sentence-transformers-BGE, else the stub
with a warning.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def __call__(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic sha256 → 32-dim float vector.

    Not semantically meaningful. Use only for tests and CI.
    """

    name = "hash-32"
    dim = 32

    def __call__(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b - 128) / 128.0 for b in h]


class FastembedBgeEmbedder:
    """ONNX-Runtime BGE-small-en-v1.5 via fastembed.

    Numerically equivalent to BgeSmallEmbedder (same BAAI weights), but
    avoids pulling in PyTorch / transformers / sklearn. Install footprint
    is ~150MB instead of ~900MB. First call downloads the ONNX model
    (~130MB) to fastembed's cache. Vectors come back L2-normalized.
    """

    name = "bge-small-en-v1.5"
    dim = 384

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "FastembedBgeEmbedder requires fastembed. "
                "Install with: pip install 'hippocamp[embeddings-lite]'"
            ) from e
        self._model = TextEmbedding(model_name=model_id)

    def __call__(self, text: str) -> list[float]:
        vec = next(self._model.embed([text]))
        return vec.tolist()


class BgeSmallEmbedder:
    """Local sentence-transformers embedder (BAAI/bge-small-en-v1.5).

    Produces 384-dim normalized vectors. First call downloads the model
    (~130MB) to the HuggingFace cache. Runs on CPU by default. Same
    weights and output as FastembedBgeEmbedder; only kept for users who
    already have sentence-transformers installed (e.g. for Qwen3) and
    want a single backend.
    """

    name = "bge-small-en-v1.5"
    dim = 384

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "BgeSmallEmbedder requires sentence-transformers. "
                "Install with: pip install 'hippocamp[embeddings]'. "
                "For a lighter install, use FastembedBgeEmbedder via "
                "pip install 'hippocamp[embeddings-lite]'."
            ) from e
        self._model = SentenceTransformer(model_id)

    def __call__(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()


class Qwen3Embedder:
    """Local sentence-transformers embedder (Qwen/Qwen3-Embedding-0.6B).

    Produces 1024-dim normalized vectors. First call downloads the model
    (~1.2GB) to the HuggingFace cache. Higher recall quality than BGE-small
    on harder retrieval cases (vocabulary mismatch, inferential matching).
    """

    name = "Qwen3-Embedding-0.6B"
    dim = 1024
    # Calibrated on evals/ (see evals/tune.py). Query-to-memory similarity
    # of a correct answer is rarely below this. It is a floor, not a judge:
    # similarity alone cannot tell "about the same topic" from "answers it".
    strong_similarity = 0.35
    similar_statement = 0.60   # statement-to-statement: worth showing as "similar"

    def __init__(self, model_id: str = "Qwen/Qwen3-Embedding-0.6B") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Qwen3Embedder requires sentence-transformers. "
                "Install with: pip install 'hippocamp[embeddings]'"
            ) from e
        # Once the model is on disk, load it without touching the network.
        # Only a first run (nothing cached) is allowed to download.
        try:
            self._model = SentenceTransformer(model_id, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_id)

    def __call__(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a *question*. Qwen3 is trained with an instruction on the
        query side and none on the stored side; without it, short questions
        land far from the memories that answer them."""
        vec = self._model.encode(text, prompt_name="query", normalize_embeddings=True)
        return vec.tolist()


class LocalServerEmbedder:
    """Embeddings from an OpenAI-compatible server on this machine (LM
    Studio, llama.cpp server, Ollama's /v1).

    Loopback only: memory text is never sent to another host. `doc_prefix`
    and `query_prefix` carry the model's task prompts, for models that are
    trained with them.
    """

    def __init__(
        self,
        model: str,
        *,
        name: str,
        dim: int,
        base_url: str | None = None,
        doc_prefix: str = "",
        query_prefix: str = "",
        strong_similarity: float | None = None,
        similar_statement: float | None = None,
        timeout: int = 60,
    ) -> None:
        import os

        from hippocamp.reflect import DEFAULT_LOCAL_URL, _is_loopback

        url = (base_url or os.environ.get("HIPPOCAMP_EMBED_URL") or DEFAULT_LOCAL_URL).rstrip("/")
        if not _is_loopback(url):
            raise RuntimeError(f"embedding server must be on this machine (loopback); got {url!r}")
        self._url, self._model, self._timeout = url, model, timeout
        self.name, self.dim = name, dim
        self._doc_prefix, self._query_prefix = doc_prefix, query_prefix
        if strong_similarity is not None:
            self.strong_similarity = strong_similarity
        if similar_statement is not None:
            self.similar_statement = similar_statement

    def _embed(self, text: str) -> list[float]:
        import json
        import math
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            self._url + "/v1/embeddings",
            data=json.dumps({"model": self._model, "input": [text]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                vec = json.loads(resp.read().decode())["data"][0]["embedding"]
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
            raise RuntimeError(
                f"local embedding server at {self._url} did not answer ({e}); "
                "start it and load the embedding model"
            ) from e
        if len(vec) != self.dim:
            raise RuntimeError(f"{self._model} returned {len(vec)} dims, expected {self.dim}")
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def __call__(self, text: str) -> list[float]:
        return self._embed(self._doc_prefix + text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(self._query_prefix + text)


def embeddinggemma(
    model: str | None = None, base_url: str | None = None
) -> LocalServerEmbedder:
    """Google's EmbeddingGemma-300m served by a local server, with the task
    prompts it was trained with. `model` is the id the server lists
    (HIPPOCAMP_EMBED_MODEL, default LM Studio's name for it)."""
    import os

    return LocalServerEmbedder(
        model or os.environ.get("HIPPOCAMP_EMBED_MODEL") or "text-embedding-embeddinggemma-300m-qat",
        name="embeddinggemma-300m",
        dim=768,
        base_url=base_url,
        doc_prefix="title: none | text: ",
        query_prefix="task: search result | query: ",
        strong_similarity=0.40,
        similar_statement=0.60,
    )


def default_embedder() -> Embedder:
    """Return the best available embedder, with a warning if falling back.

    Order: Qwen3 (best recall, needs sentence-transformers) → fastembed-BGE
    (light + fast) → sentence-transformers-BGE → hash stub.
    """
    try:
        return Qwen3Embedder()
    except ImportError:
        pass
    try:
        return FastembedBgeEmbedder()
    except ImportError:
        pass
    try:
        return BgeSmallEmbedder()
    except ImportError:
        warnings.warn(
            "Falling back to HashEmbedder — recall will not be semantic. "
            "Install 'hippocamp[embeddings-lite]' for real embeddings "
            "(or '[embeddings]' for the heavier sentence-transformers stack).",
            UserWarning,
            stacklevel=2,
        )
        return HashEmbedder()
