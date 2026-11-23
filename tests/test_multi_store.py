"""Multi-store mode of the MCP server: one process, one embedder, several
separate stores chosen by id.

Drives the same callables the MCP host calls, as test_mcp.py does for the
single-store surface.
"""

from __future__ import annotations

import inspect
import json

import pytest

from hippocamp import Memory
from hippocamp.embedders import HashEmbedder
from hippocamp.mcp_server import (
    StorePool,
    build_multi_store_tools,
    build_tools,
    parse_store_map,
)


class CountingEmbedder(HashEmbedder):
    """HashEmbedder that counts constructions, to prove the pool shares one."""

    built = 0

    def __init__(self) -> None:
        super().__init__()
        type(self).built += 1


def _pool(tmp_path, ids=("alice", "bob", "home"), embedder=None):
    stores = {sid: tmp_path / sid for sid in ids}
    return StorePool(
        stores,
        embedder=embedder or HashEmbedder(),
        cache_dir=tmp_path / "cache",
        device_id="dev_test",
    )


def _tools(tmp_path, **kw):
    pool = _pool(tmp_path, **kw)
    return pool, build_multi_store_tools(pool)


# ------------------------------------------------------------ store map


def test_store_map_parses_ids_to_absolute_paths(tmp_path):
    raw = json.dumps({"alice": str(tmp_path / "a"), "home": str(tmp_path / "h")})
    stores = parse_store_map(raw)
    assert set(stores) == {"alice", "home"}
    assert all(p.is_absolute() for p in stores.values())


def test_store_map_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    stores = parse_store_map('{"me": "~/mem"}')
    assert stores["me"] == (tmp_path / "mem").resolve()


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "{}",
        '{"../etc": "/tmp/x"}',
        '{"Has Space": "/tmp/x"}',
        '{"UPPER": "/tmp/x"}',
        '{"": "/tmp/x"}',
        '{"ok": ""}',
        '{"ok": 3}',
        '{"ok": "relative/path"}',
    ],
)
def test_store_map_refuses_bad_input(raw):
    with pytest.raises(ValueError):
        parse_store_map(raw)


def test_store_map_refuses_two_ids_for_one_directory(tmp_path):
    raw = json.dumps({"a": str(tmp_path / "s"), "b": str(tmp_path / "x" / ".." / "s")})
    with pytest.raises(ValueError, match="same directory"):
        parse_store_map(raw)


# ------------------------------------------------------------- isolation


def test_stores_never_see_each_others_memories(tmp_path):
    _, tools = _tools(tmp_path)
    a = tools["update_memory"](
        store="alice", action="assert_fact", text="alice keeps bees in the garden"
    )
    b = tools["update_memory"](
        store="bob", action="assert_fact", text="bob collects vintage maps"
    )
    assert a["store"] == "alice" and b["store"] == "bob"

    alice = tools["recall_memory"](stores=["alice"], query="bees garden maps")
    assert [h["id"] for h in alice["results"]] == [a["id"]]
    assert all(h["store"] == "alice" for h in alice["results"])

    bob = tools["recall_memory"](stores=["bob"], query="bees garden maps")
    assert [h["id"] for h in bob["results"]] == [b["id"]]

    home = tools["recall_memory"](stores=["home"], query="bees garden maps")
    assert home["results"] == []


def test_each_store_has_its_own_directory_and_cache(tmp_path):
    pool, tools = _tools(tmp_path)
    tools["update_memory"](store="alice", action="observe", text="alice event")
    tools["update_memory"](store="bob", action="observe", text="bob event")

    alice, bob = pool.get("alice"), pool.get("bob")
    assert alice.store_dir != bob.store_dir
    assert alice.cache_path != bob.cache_path
    assert alice.meta.store_id != bob.meta.store_id
    assert "alice event" in alice.events_own_file.read_text()
    assert "bob event" not in alice.events_own_file.read_text()


def test_forget_is_scoped_to_the_named_store(tmp_path):
    _, tools = _tools(tmp_path)
    a = tools["update_memory"](store="alice", action="assert_fact", text="alpha fact")
    # The id exists only in alice's store; forgetting it via bob must not
    # remove alice's memory.
    try:
        tools["update_memory"](store="bob", action="forget", id=a["id"])
    except Exception:
        pass
    r = tools["recall_memory"](stores=["alice"], query="alpha fact")
    assert [h["id"] for h in r["results"]] == [a["id"]]


# ---------------------------------------------------------- unknown ids


@pytest.mark.parametrize("bad", ["carol", "", "../alice", "ALICE", "alice/../bob"])
def test_unknown_store_id_is_refused_everywhere(tmp_path, bad):
    _, tools = _tools(tmp_path)
    with pytest.raises(ValueError, match="unknown store id"):
        tools["recall_memory"](stores=[bad], query="x")
    with pytest.raises(ValueError, match="unknown store id"):
        tools["update_memory"](store=bad, action="observe", text="x")
    with pytest.raises(ValueError, match="unknown store id"):
        tools["reflect_memory"](store=bad)
    assert not (tmp_path / "carol").exists()


def test_unknown_id_error_does_not_list_valid_ids(tmp_path):
    _, tools = _tools(tmp_path)
    with pytest.raises(ValueError) as e:
        tools["recall_memory"](stores=["carol"], query="x")
    assert "alice" not in str(e.value) and "bob" not in str(e.value)


def test_one_unknown_id_refuses_the_whole_recall(tmp_path):
    """A mixed list must not return the known store's results."""
    _, tools = _tools(tmp_path)
    tools["update_memory"](store="alice", action="assert_fact", text="alpha fact")
    with pytest.raises(ValueError, match="unknown store id"):
        tools["recall_memory"](stores=["alice", "carol"], query="alpha")


def test_recall_needs_at_least_one_store(tmp_path):
    _, tools = _tools(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        tools["recall_memory"](stores=[], query="x")


def test_store_arguments_are_required(tmp_path):
    _, tools = _tools(tmp_path)
    for name, arg in (
        ("recall_memory", "stores"),
        ("update_memory", "store"),
        ("reflect_memory", "store"),
    ):
        param = inspect.signature(tools[name]).parameters[arg]
        assert param.default is inspect.Parameter.empty, name
    with pytest.raises(TypeError):
        tools["update_memory"](action="observe", text="x")


# --------------------------------------------------------------- merging


def test_recall_across_stores_merges_by_score_and_labels(tmp_path):
    _, tools = _tools(tmp_path)
    a = tools["update_memory"](
        store="alice", action="assert_fact", text="alice likes green tea"
    )
    h = tools["update_memory"](
        store="home", action="assert_fact", text="the household drinks green tea"
    )
    tools["update_memory"](store="bob", action="assert_fact", text="bob likes green tea")

    r = tools["recall_memory"](stores=["alice", "home"], query="green tea")
    got = {(x["store"], x["id"]) for x in r["results"]}
    assert got == {("alice", a["id"]), ("home", h["id"])}
    scores = [x["score"] for x in r["results"]]
    assert scores == sorted(scores, reverse=True)
    assert all("why" in x for x in r["results"])


def test_merged_recall_respects_limit(tmp_path):
    _, tools = _tools(tmp_path)
    for i in range(4):
        tools["update_memory"](store="alice", action="observe", text=f"tea note {i}")
        tools["update_memory"](store="home", action="observe", text=f"tea memo {i}")
    r = tools["recall_memory"](stores=["alice", "home"], query="tea", limit=3)
    assert len(r["results"]) == 3


def test_duplicate_store_ids_are_searched_once(tmp_path):
    _, tools = _tools(tmp_path)
    a = tools["update_memory"](store="alice", action="assert_fact", text="alpha fact")
    r = tools["recall_memory"](stores=["alice", "alice"], query="alpha fact")
    assert [x["id"] for x in r["results"]] == [a["id"]]


def test_recall_without_query_returns_inventory_per_store(tmp_path):
    _, tools = _tools(tmp_path)
    tools["update_memory"](store="alice", action="assert_fact", text="one")
    r = tools["recall_memory"](stores=["alice", "home"])
    assert r["stores"]["alice"]["facts"] == 1
    assert r["stores"]["home"]["facts"] == 0
    assert "bob" not in r["stores"]


# ------------------------------------------------------------- the pool


def test_pool_opens_lazily_and_reuses_one_embedder(tmp_path):
    CountingEmbedder.built = 0
    embedder = CountingEmbedder()
    pool = _pool(tmp_path, embedder=embedder)
    assert not (tmp_path / "alice").exists()  # nothing opened yet

    a, b = pool.get("alice"), pool.get("bob")
    assert a is pool.get("alice")  # opened once, kept
    assert a._embedder is embedder and b._embedder is embedder
    assert CountingEmbedder.built == 1
    assert not (tmp_path / "home").exists()  # never asked for


def test_pool_refuses_a_store_committed_to_another_embedder(tmp_path):
    class Other(HashEmbedder):
        name = "other-embedder"

    Memory(
        path=str(tmp_path / "alice"),
        embedder=Other(),
        cache_dir=tmp_path / "cache",
        device_id="dev_test",
    )
    pool = _pool(tmp_path)
    with pytest.raises(RuntimeError, match="Embedder mismatch"):
        pool.get("alice")
    pool.get("bob")  # the others still open


def test_data_survives_a_new_pool_over_the_same_directories(tmp_path):
    _, tools = _tools(tmp_path)
    a = tools["update_memory"](store="alice", action="assert_fact", text="alpha fact")
    _, again = _tools(tmp_path)
    r = again["recall_memory"](stores=["alice"], query="alpha fact")
    assert [x["id"] for x in r["results"]] == [a["id"]]


# -------------------------------------------------------------- reflect


def test_reflect_runs_on_the_named_store_only(tmp_path):
    import re
    from datetime import datetime, timedelta, timezone

    seen: list[str] = []

    class FakeLLM:
        def complete(self, prompt, *, system=None):
            seen.append(prompt)
            ids = re.findall(r"\[(ep_[a-f0-9]+)\]", prompt)
            return json.dumps(
                {
                    "facts": [{"text": "alice works at acme", "evidence": ids[:1]}],
                    "preferences": [],
                    "reflections": [],
                }
            )

    pool = _pool(tmp_path)
    tools = build_multi_store_tools(pool, llm=FakeLLM())
    for i in range(6):
        tools["update_memory"](store="alice", action="observe", text=f"alice ep {i}")
        tools["update_memory"](store="bob", action="observe", text=f"bob secret {i}")

    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    report = tools["reflect_memory"](store="alice", since=since)
    assert report["store"] == "alice"
    assert report["new_facts"] == 1
    assert report["episodes_processed"] == 6
    assert seen and all("bob secret" not in p for p in seen)
    assert pool.get("bob").inspect().facts == 0


def test_reflect_without_llm_raises(tmp_path):
    _, tools = _tools(tmp_path)
    with pytest.raises(RuntimeError, match="no LLM configured"):
        tools["reflect_memory"](store="alice")


# ------------------------------------------------- single-store unchanged


def test_single_store_tools_take_no_store_argument():
    tools = build_tools(Memory(path=":memory:", embedder=HashEmbedder()))
    for name in ("recall_memory", "update_memory", "reflect_memory"):
        params = inspect.signature(tools[name]).parameters
        assert "store" not in params and "stores" not in params


def test_multi_store_tools_keep_the_single_store_descriptions(tmp_path):
    _, multi = _tools(tmp_path)
    single = build_tools(Memory(path=":memory:", embedder=HashEmbedder()))
    for name in ("recall_memory", "update_memory", "reflect_memory"):
        assert multi[name].__doc__.startswith(single[name].__doc__)
        assert "store" in multi[name].__doc__[len(single[name].__doc__):]


# ------------------------------------------------------------------ init


def _cli_env(tmp_path, monkeypatch):
    import hippocamp.embedders as embedders

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HIPPOCAMP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(embedders, "default_embedder", HashEmbedder)


def test_init_creates_an_empty_store(tmp_path, monkeypatch, capsys):
    from hippocamp.cli import main

    _cli_env(tmp_path, monkeypatch)
    target = tmp_path / "people" / "bob" / "memory"
    assert main(["init", "--path", str(target)]) == 0
    assert (target / "meta.json").exists()
    assert "created store" in capsys.readouterr().out
    assert main(["init", "--path", str(target)]) == 0  # empty: harmless to repeat


def test_init_refuses_a_store_that_already_holds_memories(tmp_path, monkeypatch, capsys):
    from hippocamp.cli import main

    _cli_env(tmp_path, monkeypatch)
    target = tmp_path / "store"
    Memory(path=str(target)).assert_fact("something already here")
    assert main(["init", "--path", str(target)]) == 1
    assert "already exists" in capsys.readouterr().out
