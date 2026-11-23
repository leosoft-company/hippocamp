"""Hybrid retrieval regression — a memory literally containing the query's
rare token must beat generic "attractor" memories that embed near
everything (the SPC bug: a same-day authorization-review fact ranked
below months-old generic preferences and never reached the ranker)."""

from hippocamp import Memory
from hippocamp.embedders import HashEmbedder
from hippocamp.ranker import tokenize


class ConstEmbedder(HashEmbedder):
    """Every text embeds identically — similarity contributes nothing, so
    tests isolate the lexical/recency/kind signals."""

    name = "const-32"

    def __call__(self, text: str) -> list[float]:
        return [1.0] * self.dim


def _mem() -> Memory:
    return Memory(path=":memory:", embedder=ConstEmbedder())


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("What outstanding work do we have for SPC?") == {
        "outstanding", "work", "spc",
    }


def test_rare_token_match_outranks_generic_memories():
    m = _mem()
    for i in range(30):
        m.assert_preference(f"User prefers concise answers, variant {i}")
    m.assert_fact(
        "ZORgatron project: authorization review found three anonymous "
        "controllers; fixes go on a hardening branch after sign-off"
    )
    results = m.recall("zorgatron outstanding work", limit=3)
    assert results, "expected results"
    top = results[0]
    assert "ZORgatron" in top.text
    assert top.why.lexical > 0


def test_lexical_trace_zero_when_no_token_overlap():
    m = _mem()
    m.assert_fact("The sky appears blue due to Rayleigh scattering")
    results = m.recall("quarterly finances", limit=1)
    if results:  # ranked purely on embedding signals
        assert results[0].why.lexical == 0.0
