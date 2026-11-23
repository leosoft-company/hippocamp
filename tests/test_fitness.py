"""Behaviour a household assistant relies on. See docs/fitness-plan.md.

Each test was written while the behaviour was still broken (v0.11.1) and
pins the fix. Hash embedder + scripted LLM only: no model, no network, and
no real store — everything lives under pytest's tmp_path.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import hippocamp.memory as hm
from hippocamp.embedders import HashEmbedder
from hippocamp.mcp_server import build_tools
from hippocamp.memory import Memory
from hippocamp.ranker import rank
from hippocamp.types import RawRow

NOW = lambda: datetime.now(timezone.utc)  # noqa: E731


def mk(tmp_path, sub="store", device="dev-a", cache="cache", llm=None, emb=None):
    d = tmp_path / sub
    return Memory(str(d), embedder=emb or HashEmbedder(), device_id=device,
                  cache_dir=str(tmp_path / cache), llm=llm), d


def active(mem):
    return {r[0] for r in mem._store._conn.execute(
        "SELECT id FROM memories WHERE superseded_at IS NULL")}


def on_disk(tmp_path, needle: str) -> list[str]:
    hits = []
    for p in tmp_path.rglob("*"):
        if p.is_file() and needle.encode() in p.read_bytes():
            hits.append(p.name)
    return hits


class backdated:
    """Pretend 'now' is N days ago for writes made inside the block."""

    def __init__(self, days): self.days = days
    def __enter__(self):
        self.real = hm._now
        hm._now = lambda: datetime.now(timezone.utc) - timedelta(days=self.days)
    def __exit__(self, *a): hm._now = self.real


class FakeLLM:
    def __init__(self, fn): self.fn, self.prompts = fn, []
    def complete(self, prompt, *, system=None):
        self.prompts.append(prompt); return self.fn(prompt)


def ep_ids(prompt):
    return re.findall(r"\[(ep_[0-9a-f]+)\]", prompt.split("EXISTING")[0])


def echo_llm():
    """One fact per episode shown, citing that episode."""
    return FakeLLM(lambda p: json.dumps({"facts": [
        {"text": f"distilled from {e}", "evidence": [e]} for e in ep_ids(p)]}))


def row(id, kind, days_old, emb, **meta):
    t = NOW() - timedelta(days=days_old)
    return RawRow(id=id, kind=kind, text=id, created_at=t, last_seen_at=t,
                  salience=0, metadata=meta, embedding=emb)


# ====================================================================== write

def test_same_fact_saved_five_times_is_kept_once(tmp_path):
    m, _ = mk(tmp_path)
    ids = {m.assert_fact(t) for t in ["Theo is allergic to peanuts", "Theo is allergic to peanuts",
                                      "theo is allergic to peanuts.", "  Theo is allergic  to peanuts ",
                                      "Theo is allergic to peanuts"]}
    assert len(ids) == 1 and m.inspect().facts == 1
    p = {m.assert_preference("likes tea"), m.assert_preference("Likes tea.")}
    assert len(p) == 1 and m.inspect().preferences == 1


def test_a_forgotten_fact_can_be_saved_again(tmp_path):
    m, _ = mk(tmp_path)
    a = m.assert_fact("the shed key is under the pot"); m.forget(a)
    b = m.assert_fact("the shed key is under the pot")
    assert b != a and b in active(m)


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_is_refused(tmp_path, text):
    m, _ = mk(tmp_path); t = build_tools(m)
    for action in ("observe", "assert_fact", "assert_preference"):
        with pytest.raises(ValueError):
            t["update_memory"](action=action, text=text)
    assert sum(m.inspect().model_dump(exclude={"bytes_on_disk", "last_reflect"}).values()) == 0


def test_tool_can_replace_a_fact_and_a_preference(tmp_path):
    m, _ = mk(tmp_path); t = build_tools(m)
    assert "supersedes" in inspect.signature(t["update_memory"]).parameters
    old = t["update_memory"](action="assert_fact", text="works at Halvorsen")["id"]
    new = t["update_memory"](action="assert_fact", text="works at Brightwater", supersedes=[old])["id"]
    p_old = t["update_memory"](action="assert_preference", text="prefers aisle seats")["id"]
    p_new = t["update_memory"](action="assert_preference", text="prefers window seats", supersedes=[p_old])["id"]
    assert active(m) == {new, p_new}


def test_replacing_an_unknown_id_is_refused(tmp_path):
    m, _ = mk(tmp_path)
    with pytest.raises(ValueError):
        m.assert_fact("works at Brightwater", supersedes=["fa_nope"])
    assert m.inspect().facts == 0


def test_tool_text_only_promises_parameters_that_exist(tmp_path):
    m, _ = mk(tmp_path); t = build_tools(m)
    doc = t["update_memory"].__doc__
    params = inspect.signature(t["update_memory"]).parameters
    for word in ("supersedes", "strength", "evidence"):
        if word in doc:
            assert word in params


def test_write_reports_similar_existing_memories(tmp_path):
    """So the model can tell 'new' from 'this replaces that'."""
    class Same(HashEmbedder):  # every text looks alike
        def __call__(self, text): return [1.0] + [0.0] * 31
    m, _ = mk(tmp_path, emb=Same()); t = build_tools(m)
    old = t["update_memory"](action="assert_fact", text="lives in Leith")["id"]
    out = t["update_memory"](action="assert_fact", text="lives in Dunbar")
    assert [s["id"] for s in out["similar"]] == [old]
    assert out["similar"][0]["text"] == "lives in Leith"


def test_strength_is_clamped_and_used(tmp_path):
    m, _ = mk(tmp_path)
    pid = m.assert_preference("likes it hot", strength=7)
    meta = json.loads(m._store._conn.execute("SELECT metadata FROM memories WHERE id=?", (pid,)).fetchone()[0])
    assert meta["strength"] == 1.0
    e = [1.0, 0.0]
    got = rank([row("weak", "preference", 0, e, strength=0.1), row("strong", "preference", 0, e, strength=1.0)],
               query_emb=e, context=None, now=NOW())
    assert [r.id for r in got] == ["strong", "weak"] and got[0].score > got[1].score


# ===================================================================== recall

def test_unknown_kind_is_an_error_not_an_empty_list(tmp_path):
    m, _ = mk(tmp_path); m.assert_fact("boat is called Petrel")
    with pytest.raises(ValueError):
        m.recall("boat", kinds=["facts"])


def test_limit_edge_cases(tmp_path):
    m, _ = mk(tmp_path)
    for i in range(30): m.assert_fact(f"fact about boats number {i}")
    assert m.recall("boats", limit=0) == []
    assert m.recall("boats", limit=-1) == []
    assert len(m.recall("boats", limit=1000)) == 30


def test_kind_filter_returns_a_full_list(tmp_path):
    m, _ = mk(tmp_path)
    assert m._store.vec_enabled, "this test needs the sqlite-vec path"
    for i in range(15): m.assert_preference(f"preference number {i}")
    for i in range(400): m.observe(f"episode number {i}")
    f = m.assert_fact("old value"); m.assert_fact("new value", supersedes=[f])
    assert len(m.recall("preference", kinds=["preference"], limit=10)) == 10
    assert len(m.recall("anything", kinds=["fact"], limit=10)) == 1


def test_recall_does_not_change_the_store(tmp_path):
    m, _ = mk(tmp_path)
    for i in range(20): m.assert_fact(f"fact {i}"); m.observe(f"episode {i}")
    snap = lambda: m._store._conn.execute("SELECT id, salience, last_seen_at FROM memories ORDER BY id").fetchall()  # noqa: E731
    before = snap()
    first = [h.id for h in m.recall("fact", limit=10)]
    for _ in range(5): m.recall("fact", limit=10); m.recall("episode", limit=10)
    assert [h.id for h in m.recall("fact", limit=10)] == first
    assert snap() == before


def test_an_old_fact_is_not_buried_by_fresh_chatter():
    q = [1.0, 0.0]
    fact = row("allergy", "fact", 730, [0.6, 0.8])        # cosine 0.60, two years old
    pref = row("terse", "preference", 400, [0.6, 0.8])
    chat = row("eiffel", "episode", 0, [0.4, 0.9165])     # cosine 0.40, just now
    got = [r.id for r in rank([chat, fact, pref], query_emb=q, context=None, now=NOW())]
    assert got.index("allergy") < got.index("eiffel") and got.index("terse") < got.index("eiffel")


def test_recency_still_orders_episodes():
    q = [1.0, 0.0]; e = [0.6, 0.8]
    got = [r.id for r in rank([row("last_year", "episode", 365, e), row("today", "episode", 0, e)],
                              query_emb=q, context=None, now=NOW())]
    assert got == ["today", "last_year"]


def test_duplicates_already_in_a_store_come_back_once(tmp_path):
    m, _ = mk(tmp_path)
    for i in range(4):  # as an older version would have written them
        m._store.insert(id=f"fa_dup{i}", kind="fact", text="Theo is allergic to peanuts", created_at=NOW(),
                        metadata={}, embedding=HashEmbedder()("Theo is allergic to peanuts"))
    m.assert_fact("Isla is lactose intolerant")
    texts = [h.text for h in m.recall("Theo allergic peanuts", limit=5)]
    assert texts.count("Theo is allergic to peanuts") == 1 and len(texts) == 2


def test_every_result_says_how_good_the_match_is(tmp_path):
    m, _ = mk(tmp_path); m.assert_fact("boat is called Petrel")
    hits = m.recall("boat", limit=3)
    assert hits and all(h.match in ("strong", "weak") for h in hits)
    out = build_tools(m)["recall_memory"](query="boat")
    assert "results" in out and all("match" in r for r in out["results"])


def test_query_side_embedding_is_used_when_the_embedder_has_one(tmp_path):
    class Split(HashEmbedder):
        asked = []
        def embed_query(self, text):
            self.asked.append(text); return self(text)
    e = Split(); m, _ = mk(tmp_path, emb=e); m.assert_fact("boat is called Petrel")
    m.recall("what is the boat called")
    assert e.asked == ["what is the boat called"]


# ==================================================================== forget

def test_forgetting_an_unknown_id_is_an_error(tmp_path):
    m, d = mk(tmp_path); t = build_tools(m)
    with pytest.raises(ValueError):
        t["update_memory"](action="forget", id="fa_doesnotexist")
    assert not any((d / "events").glob("*.jsonl")) or "fa_doesnotexist" not in next((d / "events").glob("*.jsonl")).read_text()


def test_purge_removes_the_text_from_disk(tmp_path):
    m, d = mk(tmp_path)
    keep = m.assert_fact("the shed key is under the pot")
    x = m.assert_fact("the safe code is 4471-ZEBRA")
    assert on_disk(tmp_path, "4471-ZEBRA")
    m.purge(x)
    assert on_disk(tmp_path, "4471-ZEBRA") == []
    m.close(); shutil.rmtree(tmp_path / "cache"); m, _ = mk(tmp_path)
    assert active(m) == {keep} and on_disk(tmp_path, "4471-ZEBRA") == []


def test_compact_purges_what_was_forgotten_and_keeps_history(tmp_path):
    m, d = mk(tmp_path)
    gone = [m.assert_fact(f"secret-{i}-XYZZY") for i in range(3)]
    for g in gone: m.forget(g)
    old = m.assert_fact("worked at HALVORSEN-OLD"); new = m.assert_fact("works at Brightwater", supersedes=[old])
    assert on_disk(tmp_path, "XYZZY")
    rep = m.compact()
    assert rep.dropped == 3
    assert on_disk(tmp_path, "XYZZY") == []
    assert on_disk(tmp_path, "HALVORSEN-OLD"), "a replaced value is history, not a deletion request"
    assert active(m) == {new}


def test_purge_reaches_another_device_through_the_log(tmp_path):
    a, d = mk(tmp_path, device="dev-a")
    x = a.assert_fact("the safe code is 4471-ZEBRA")
    b, _ = mk(tmp_path, device="dev-b", cache="cacheB")
    assert x in active(b)
    a.purge(x); b.close()
    b, _ = mk(tmp_path, device="dev-b", cache="cacheB")
    assert x not in active(b) and on_disk(tmp_path, "4471-ZEBRA") == []


def test_store_files_are_private(tmp_path):
    m, d = mk(tmp_path); m.assert_fact("private"); m.close()
    mode = lambda p: stat.S_IMODE(p.stat().st_mode)  # noqa: E731
    assert mode(d) == 0o700 and mode(d / "events") == 0o700
    files = [d / "meta.json", *(d / "events").glob("*.jsonl"), *(tmp_path / "cache").rglob("store.db")]
    assert len(files) >= 3 and all(mode(f) == 0o600 for f in files)


def test_an_existing_loose_store_is_tightened_on_open(tmp_path):
    m, d = mk(tmp_path); m.assert_fact("private"); m.close()
    os.chmod(d, 0o755); os.chmod(next((d / "events").glob("*.jsonl")), 0o644)
    mk(tmp_path)
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    assert stat.S_IMODE(next((d / "events").glob("*.jsonl")).stat().st_mode) == 0o600


def _cli(monkeypatch, tmp_path, *argv):
    import hippocamp.embedders as he
    from hippocamp.cli import main
    monkeypatch.setattr(he, "default_embedder", lambda: HashEmbedder())
    monkeypatch.setenv("HIPPOCAMP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("hippocamp.memory.load_or_init_device_id", lambda *a, **k: "dev-a")
    return main(list(argv))


def test_cli_can_expire_forget_and_compact(tmp_path, monkeypatch, capsys):
    m, d = mk(tmp_path)
    with backdated(100):
        m.observe("old episode OLDEP"); old_fact = m.assert_fact("old fact")
    keep = m.observe("new episode"); x = m.assert_fact("to forget FORGETME"); m.close()
    assert _cli(monkeypatch, tmp_path, "expire", "--path", str(d), "--before", "30d", "--kind", "episode") == 0
    assert _cli(monkeypatch, tmp_path, "forget", x, "--path", str(d)) == 0
    assert _cli(monkeypatch, tmp_path, "forget", "fa_nope", "--path", str(d)) == 1
    m, _ = mk(tmp_path); assert active(m) == {old_fact, keep}; m.close()
    assert _cli(monkeypatch, tmp_path, "compact", "--path", str(d)) == 0
    assert on_disk(tmp_path, "FORGETME") == [] and on_disk(tmp_path, "OLDEP") == []


def test_cli_reflect_all_covers_every_store_in_the_map(tmp_path, monkeypatch, capsys):
    """The nightly job must consolidate each person's store, not just the default."""
    import hippocamp.cli as cli
    stores = {}
    for name in ("dana", "riley", "home"):
        m, d = mk(tmp_path, sub=name)
        for i in range(6): m.observe(f"{name} episode {i}")
        m.close(); stores[name] = str(d)
    m, d = mk(tmp_path, sub="quiet"); m.close(); stores["quiet"] = str(d)      # nothing to do
    monkeypatch.setenv("HIPPOCAMP_STORES", json.dumps(stores))
    llms = []
    monkeypatch.setattr(cli, "_resolve_llm", lambda name: llms.append(echo_llm()) or llms[-1])
    assert _cli(monkeypatch, tmp_path, "reflect", "--all") == 0
    out = json.loads(capsys.readouterr().out)
    assert {k: v["episodes_processed"] for k, v in out.items()} == {"dana": 6, "riley": 6, "home": 6, "quiet": 0}
    for name in ("dana", "riley", "home"):
        m, _ = mk(tmp_path, sub=name)
        texts = [r[0] for r in m._store._conn.execute("SELECT text FROM memories WHERE kind='episode'")]
        assert m.inspect().facts == 6 and all(t.startswith(name) for t in texts)
    # each store got its own model conversation: no episode text crossed over
    for llm in llms:
        owners = {re.search(r"\] \S+ (\w+) episode", line).group(1) for p in llm.prompts for line in p.splitlines() if " episode " in line}
        assert len(owners) <= 1


def test_cli_reflect_all_reports_a_failing_store_and_continues(tmp_path, monkeypatch, capsys):
    import hippocamp.cli as cli
    stores = {}
    for name in ("dana", "riley"):
        m, d = mk(tmp_path, sub=name)
        for i in range(6): m.observe(f"{name} episode {i}")
        m.close(); stores[name] = str(d)
    monkeypatch.setenv("HIPPOCAMP_STORES", json.dumps(stores))
    def llm_for(name):
        return FakeLLM(lambda p: "not json" if "dana episode" in p else json.dumps(
            {"facts": [{"text": f"from {e}", "evidence": [e]} for e in ep_ids(p)]}))
    monkeypatch.setattr(cli, "_resolve_llm", llm_for)
    assert _cli(monkeypatch, tmp_path, "reflect", "--all") == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" in out["dana"] and out["riley"]["episodes_processed"] == 6


# ============================================================= replay / sync

def rebuild(tmp_path, m, **kw):
    m.close(); shutil.rmtree(tmp_path / kw.get("cache", "cache")); return mk(tmp_path, **kw)[0]


def test_forget_survives_a_rebuild_even_when_dated_before_the_memory(tmp_path):
    m, _ = mk(tmp_path)
    x = m.observe("dentist appointment", at=NOW() + timedelta(days=3))
    m.forget(x)
    assert x not in active(rebuild(tmp_path, m))


def test_forget_from_a_device_with_a_slow_clock_survives_a_rebuild(tmp_path):
    a, d = mk(tmp_path, device="dev-a"); x = a.assert_fact("secret plan")
    b, _ = mk(tmp_path, device="dev-b", cache="cacheB")
    with backdated(0.01): b.forget(x)
    assert x not in active(rebuild(tmp_path, a, device="dev-a"))


def test_replace_from_a_device_with_a_slow_clock_survives_a_rebuild(tmp_path):
    a, d = mk(tmp_path, device="dev-a"); old = a.assert_fact("lives in Leith")
    b, _ = mk(tmp_path, device="dev-b", cache="cacheB")
    with backdated(0.01): new = b.assert_fact("lives in Dunbar", supersedes=[old])
    assert active(rebuild(tmp_path, a, device="dev-a")) == {new}


def test_a_half_written_log_line_is_skipped(tmp_path):
    m, d = mk(tmp_path)
    ids = {m.assert_fact(f"fact {i}") for i in range(10)}; m.close()
    f = next((d / "events").glob("*.jsonl")); f.write_text(f.read_text() + '{"id": "ev_trunc", "ts": "2026-11-2')
    shutil.rmtree(tmp_path / "cache")
    m, _ = mk(tmp_path)
    assert active(m) == ids
    extra = m.assert_fact("written after the damage")     # and the log is still appendable
    assert active(rebuild(tmp_path, m)) == ids | {extra}


def test_opening_a_store_picks_up_another_devices_new_events(tmp_path):
    a, d = mk(tmp_path, device="dev-a")
    for i in range(3): a.assert_fact(f"A fact {i}")
    b, _ = mk(tmp_path, device="dev-b", cache="cacheB"); assert len(active(b)) == 3; b.close()
    late = a.assert_fact("A later fact"); a.forget(next(iter(active(a) - {late})))
    b, _ = mk(tmp_path, device="dev-b", cache="cacheB")
    assert active(b) == active(a) and len(active(b)) == 3


def test_rebuild_matches_the_live_state(tmp_path):
    m, _ = mk(tmp_path)
    a = m.assert_fact("a"); b = m.assert_fact("b", supersedes=[a]); c = m.observe("c"); p = m.assert_preference("p")
    r = m.assert_reflection("r"); m.forget(c)
    with backdated(50): m.observe("old")
    m.expire_before("30d")
    before = active(m)
    assert active(rebuild(tmp_path, m)) == before == {b, p, r}


def test_a_cache_written_by_v0_11_is_adopted_without_a_rebuild(tmp_path):
    """Upgrade path: rows stay as they are (no re-embedding), and what was
    forgotten under the old version can still be purged."""
    m, d = mk(tmp_path)
    keep = m.assert_fact("keep me"); gone = m.assert_fact("forget me OLDSECRET"); m.forget(gone)
    c = m._store._conn                                   # make the cache look like v0.11's
    c.execute("DELETE FROM kv"); c.execute("DELETE FROM tombstones")
    c.execute("UPDATE memories SET forgotten_at = NULL, norm = NULL")
    m.close()

    class Counting(HashEmbedder):
        calls = 0
        def __call__(self, text):
            Counting.calls += 1; return super().__call__(text)
    m, _ = mk(tmp_path, emb=Counting())
    assert Counting.calls == 0, "adopting an existing cache must not re-embed it"
    assert active(m) == {keep}
    assert m.assert_fact("keep me") == keep              # duplicate check works on old rows
    assert m.compact().dropped == 1 and on_disk(tmp_path, "OLDSECRET") == []


def test_two_processes_writing_one_store_lose_nothing(tmp_path):
    m, d = mk(tmp_path); m.assert_fact("seed"); m.close()
    code = (
        "import sys; from hippocamp.embedders import HashEmbedder; from hippocamp.memory import Memory\n"
        "m = Memory(sys.argv[1], embedder=HashEmbedder(), device_id='dev-a', cache_dir=sys.argv[2])\n"
        "[m.assert_fact(f'{sys.argv[3]} fact {i}') for i in range(150)]\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(hm.__file__).parent.parent))
    ps = [subprocess.Popen([sys.executable, "-c", code, str(d), str(tmp_path / "cache"), tag], env=env,
                           stderr=subprocess.PIPE) for tag in ("P1", "P2")]
    assert [p.wait() for p in ps] == [0, 0], [p.stderr.read()[-300:] for p in ps]

    class Counting(HashEmbedder):
        calls = 0
        def __call__(self, text):
            Counting.calls += 1; return super().__call__(text)
    m, _ = mk(tmp_path, emb=Counting())
    assert len(active(m)) == 301
    assert Counting.calls == 0, "a clean shutdown of two writers must not force a rebuild"


# =================================================================== reflect

def test_reflect_reads_every_pending_episode(tmp_path):
    llm = echo_llm(); m, _ = mk(tmp_path, llm=llm)
    eps = [m.observe(f"episode {i}", at=NOW() - timedelta(hours=80 - i)) for i in range(80)]
    rep = m.reflect()
    assert {e for p in llm.prompts for e in ep_ids(p)} == set(eps)
    assert rep.episodes_processed == 80 and len(llm.prompts) > 1
    assert m.reflect().episodes_processed == 0


def test_first_reflect_covers_old_episodes(tmp_path):
    llm = echo_llm(); m, _ = mk(tmp_path, llm=llm)
    old = [m.observe(f"old {i}", at=NOW() - timedelta(days=20 + i)) for i in range(12)]
    m.reflect()
    assert set(old) <= {e for p in llm.prompts for e in ep_ids(p)}


def test_late_arriving_episodes_are_reflected(tmp_path):
    llm = echo_llm(); m, _ = mk(tmp_path, llm=llm)
    for i in range(6): m.observe(f"today {i}")
    m.reflect()
    late = [m.observe(f"arrived late {i}", at=NOW() - timedelta(days=2, hours=i)) for i in range(8)]
    m.reflect()
    assert set(late) <= {e for p in llm.prompts for e in ep_ids(p)}


def test_a_few_fresh_episodes_wait_but_do_not_wait_forever(tmp_path):
    llm = echo_llm(); m, _ = mk(tmp_path, llm=llm)
    fresh = [m.observe(f"fresh {i}") for i in range(3)]
    assert m.reflect().episodes_processed == 0 and llm.prompts == []
    stale = m.observe("left over from last week", at=NOW() - timedelta(days=3))
    m.reflect()
    assert {e for p in llm.prompts for e in ep_ids(p)} == set(fresh) | {stale}


def test_what_was_reflected_survives_a_rebuild(tmp_path):
    llm = echo_llm(); m, _ = mk(tmp_path, llm=llm)
    for i in range(6): m.observe(f"episode {i}")
    m.reflect(); n = len(llm.prompts)
    m = rebuild(tmp_path, m, llm=llm)
    assert m.reflect().episodes_processed == 0 and len(llm.prompts) == n


def test_a_store_reflected_by_an_older_version_does_not_start_over(tmp_path):
    llm = echo_llm(); m, d = mk(tmp_path, llm=llm)
    done = [m.observe(f"already reflected {i}", at=NOW() - timedelta(days=2, hours=i)) for i in range(10)]
    m.set_last_reflect_at(NOW() - timedelta(days=1))          # what v0.11 left behind
    new = [m.observe(f"new {i}") for i in range(6)]
    m.reflect()
    assert {e for p in llm.prompts for e in ep_ids(p)} == set(new)


def test_a_dead_process_does_not_hold_the_reflect_lock(tmp_path):
    llm = echo_llm(); m, d = mk(tmp_path, llm=llm)
    for i in range(6): m.observe(f"e{i}")
    dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
    (d / ".reflect.lock").write_text(f"{dead.pid} {NOW().isoformat()}\n")
    assert m.reflect().episodes_processed == 6
    (d / ".reflect.lock").write_text(f"{os.getpid()} {NOW().isoformat()}\n")   # a live holder still blocks
    for i in range(6): m.observe(f"f{i}")
    with pytest.raises(RuntimeError):
        m.reflect()


def test_long_episodes_reach_the_model(tmp_path):
    llm = echo_llm(); m, _ = mk(tmp_path, llm=llm)
    m.observe("Chatted about the weekend. " * 40 + "IMPORTANT: Dana is allergic to penicillin.")
    for i in range(5): m.observe(f"e{i}")
    m.reflect()
    assert "penicillin" in llm.prompts[0]


def test_rerunning_reflect_does_not_duplicate_anything(tmp_path):
    fixed = lambda p: json.dumps({                                              # noqa: E731
        "facts": [{"text": "User works at Brightwater Consulting.", "evidence": ep_ids(p)[:1]}],
        "preferences": [{"text": "User prefers trains over flights.", "evidence": ep_ids(p)[:1]}],
        "reflections": [{"text": "User plans family logistics carefully.", "evidence": ep_ids(p)[:2]}]})
    m, _ = mk(tmp_path, llm=FakeLLM(fixed))
    for i in range(6): m.observe(f"episode {i}")
    since = (NOW() - timedelta(days=1)).isoformat()
    for _ in range(3): m.reflect(since=since)
    inv = m.inspect()
    assert (inv.facts, inv.preferences, inv.reflections) == (1, 1, 1)


def test_the_model_sees_existing_memories_with_their_ids(tmp_path):
    llm = echo_llm(); m, _ = mk(tmp_path, llm=llm)
    f = m.assert_fact("works at Halvorsen"); p = m.assert_preference("prefers aisle seats")
    for i in range(6): m.observe(f"e{i}")
    m.reflect()
    existing = llm.prompts[0].split("EXISTING")[1]
    assert f"[{f}]" in existing and f"[{p}]" in existing


def test_reflect_replaces_what_the_model_says_it_replaces(tmp_path):
    state = {}
    def reply(p):
        return json.dumps({"facts": [{"text": "User works at Forth Bridge Engineering.", "evidence": ep_ids(p)[:1], "replaces": [state["job"]]}],
                           "preferences": [{"text": "No meetings before 10am.", "evidence": ep_ids(p)[:1], "replaces": [state["meet"]]}]})
    m, _ = mk(tmp_path, llm=FakeLLM(reply))
    state["job"] = m.assert_fact("User works at Brightwater."); state["meet"] = m.assert_preference("No meetings before 9:30am.")
    keep = m.assert_fact("User's son is allergic to peanuts.")
    for i in range(6): m.observe(f"e{i}")
    rep = m.reflect()
    texts = {r[0] for r in m._store._conn.execute("SELECT text FROM memories WHERE superseded_at IS NULL AND kind != 'episode'")}
    assert texts == {"User works at Forth Bridge Engineering.", "No meetings before 10am.", "User's son is allergic to peanuts."}
    assert rep.superseded == 2 and keep in active(m)


def test_reflect_never_replaces_on_resemblance_alone(tmp_path):
    """v0.11 replaced any fact with cosine >= 0.85 — a son's allergy could
    overwrite a daughter's. Only an id the model was shown may be replaced."""
    class Same(HashEmbedder):
        def __call__(self, text): return [1.0] + [0.0] * 31
    reply = lambda p: json.dumps({"facts": [                                     # noqa: E731
        {"text": "User's daughter is allergic to peanuts.", "evidence": ep_ids(p)[:1]},
        {"text": "Made-up replacement.", "evidence": ep_ids(p)[:1], "replaces": ["fa_never_shown"]}]})
    m, _ = mk(tmp_path, llm=FakeLLM(reply), emb=Same())
    son = m.assert_fact("User's son is allergic to peanuts.")
    for i in range(6): m.observe(f"e{i}")
    rep = m.reflect()
    assert son in active(m) and rep.superseded == 0 and m.inspect().facts == 3


def test_a_bad_model_reply_changes_nothing_and_can_be_retried(tmp_path):
    llm = FakeLLM(lambda p: "Sure! No JSON today."); m, d = mk(tmp_path, llm=llm)
    eps = [m.observe(f"e{i}") for i in range(6)]
    with pytest.raises(RuntimeError):
        m.reflect()
    assert m.inspect().facts == 0 and not (d / ".reflect.lock").exists()
    good = echo_llm(); m._llm = good
    assert m.reflect().episodes_processed == 6 and set(ep_ids(good.prompts[0])) == set(eps)


def test_a_reply_cut_off_by_the_token_limit_is_retried_in_smaller_batches(tmp_path):
    from hippocamp.reflect import ReplyTruncated
    def reply(p):
        ids = ep_ids(p)
        if len(ids) > 7:
            raise ReplyTruncated("cut off")
        return json.dumps({"facts": [{"text": f"distilled from {e}", "evidence": [e]} for e in ids]})
    llm = FakeLLM(reply); m, _ = mk(tmp_path, llm=llm)
    eps = [m.observe(f"episode {i}") for i in range(25)]
    rep = m.reflect()
    assert rep.episodes_processed == 25 and m.inspect().facts == 25
    assert m.reflect().episodes_processed == 0


def test_local_adapter_leaves_room_for_a_thinking_model():
    from hippocamp.reflect import local_llm
    assert inspect.signature(local_llm).parameters["max_tokens"].default >= 16000


# ================================================================= hardening

def test_cached_model_loads_without_the_network(monkeypatch):
    import types
    calls = []
    class ST:
        def __init__(self, model_id, **kw): calls.append(kw)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=ST))
    from hippocamp.embedders import Qwen3Embedder
    Qwen3Embedder()
    assert calls[0].get("local_files_only") is True and len(calls) == 1


def test_local_server_embedder_is_loopback_only():
    from hippocamp.embedders import LocalServerEmbedder, embeddinggemma
    with pytest.raises(RuntimeError, match="loopback"):
        LocalServerEmbedder("m", name="m", dim=4, base_url="http://example.com:1234")
    e = embeddinggemma(base_url="http://127.0.0.1:9")
    assert (e.name, e.dim) == ("embeddinggemma-300m", 768)
    with pytest.raises(RuntimeError, match="did not answer"):
        e("anything")


# ================================================================== reranker

class FakeReranker:
    """Says a memory answers the question only if it contains the word 'answer'."""
    strong_score = 0.5
    def __init__(self): self.calls = []
    def __call__(self, query, texts):
        self.calls.append((query, list(texts)))
        return [0.9 if "answer" in t else 0.01 for t in texts]


def test_a_reranker_reorders_and_labels_results(tmp_path):
    rr = FakeReranker()
    d = tmp_path / "store"
    m = Memory(str(d), embedder=HashEmbedder(), device_id="dev-a", cache_dir=str(tmp_path / "cache"), reranker=rr)
    for i in range(12): m.assert_fact(f"unrelated fact {i}")
    right = m.assert_fact("this one is the answer")
    hits = m.recall("anything at all", limit=5)
    assert hits[0].id == right and hits[0].match == "strong" and hits[0].why.rerank == 0.9
    assert all(h.match == "weak" for h in hits[1:])
    assert len(rr.calls) == 1 and len(rr.calls[0][1]) == 13     # one batch, all candidates up to the cap
    again = m.recall("anything at all", limit=5)
    assert [h.id for h in again] == [h.id for h in hits]


def test_with_a_reranker_nothing_relevant_means_all_weak(tmp_path):
    rr = FakeReranker()
    m = Memory(str(tmp_path / "s"), embedder=HashEmbedder(), device_id="dev-a", cache_dir=str(tmp_path / "c"), reranker=rr)
    for i in range(5): m.assert_fact(f"unrelated fact {i}")
    out = build_tools(m)["recall_memory"](query="what is my blood type")
    assert all(r["match"] == "weak" for r in out["results"]) and "note" in out


def test_without_a_reranker_nothing_changes(tmp_path):
    m, _ = mk(tmp_path); m.assert_fact("boat is called Petrel")
    assert m.recall("boat")[0].why.rerank is None


def test_reranker_is_opt_in_by_environment(monkeypatch):
    from hippocamp import rerank
    monkeypatch.delenv("HIPPOCAMP_RERANKER", raising=False)
    assert rerank.from_env() is None
    monkeypatch.setenv("HIPPOCAMP_RERANKER", "nonsense")
    with pytest.raises(ValueError):
        rerank.from_env()
