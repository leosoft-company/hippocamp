"""Hippocamp MCP server — two-tool surface.

Run as `hippocamp-mcp` (after `pip install 'hippocamp[mcp]'`), or wire
into Claude Desktop / Cursor / ChatGPT Apps via:

    {
      "mcpServers": {
        "hippocamp": { "command": "hippocamp-mcp" }
      }
    }

Memory lives at $HIPPOCAMP_PATH (default ~/.hippocamp/store.db). Nothing
ever leaves the user's machine.

Multi-store mode
----------------
Set $HIPPOCAMP_STORES to a JSON object of store id → directory and one
server holds several separate stores behind one shared embedder:

    HIPPOCAMP_STORES='{"alice": "~/.hippocamp", "home": "~/shared/memory"}'

Every tool call must then name the store(s) it acts on (`store` /
`stores`). The ids come from that fixed map: an id outside it is refused,
so an id can never be turned into a path. The stores stay separate
directories with separate caches — the id selects a store, it is not a
filter over a shared one. The caller that owns the map (a harness, not a
model) is expected to fill the argument in itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

from hippocamp import __version__
from hippocamp.memory import Memory

DEFAULT_PATH = Path(
    os.environ.get("HIPPOCAMP_PATH", str(Path.home() / ".hippocamp"))
)

STORES_ENV = "HIPPOCAMP_STORES"
_STORE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_NO_LLM = (
    "reflect_memory is unavailable: the MCP server has no LLM configured. "
    "Run a local OpenAI-compatible model server on this machine (LM Studio "
    "on 127.0.0.1:1234 by default; HIPPOCAMP_LLM_URL / HIPPOCAMP_LLM_MODEL "
    "override), or build the tools with llm=... explicitly."
)


# ------------------------------------------------------------ tool bodies
# One implementation per tool, shared by the single-store and multi-store
# surfaces so the two can't drift.


def _recall(
    mem: Memory, query: str | None, kinds: list[str] | None, limit: int
) -> dict[str, Any]:
    if query is None:
        return mem.inspect().model_dump()
    hits = mem.recall(query, kinds=kinds, limit=limit)
    out: dict[str, Any] = {"results": [h.model_dump() for h in hits]}
    if not any(h.match == "strong" for h in hits):
        out["note"] = _NO_STRONG
    return out


_NO_STRONG = (
    "No stored memory is a strong match for this query. The results are the "
    "nearest things in the store and may not answer it: read them, and use "
    "one only if it really does. Otherwise try one differently worded query, "
    "then tell the user you have nothing on this."
)

_KIND_OF_ACTION = {"assert_fact": "fact", "assert_preference": "preference"}


def _update(
    mem: Memory,
    action: str,
    text: str | None,
    kind: str,
    evidence: list[str] | None,
    strength: float,
    id: str | None,
    supersedes: list[str] | None = None,
) -> dict[str, Any]:
    if action == "forget":
        if not id:
            raise ValueError("'forget' requires an id")
        mem.forget(id)
        return {"ok": True, "id": id, "action": "forget"}
    if action not in ("observe", "assert_fact", "assert_preference"):
        raise ValueError(f"unknown action: {action}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"'{action}' needs some text")
    if action == "observe":
        return {"ok": True, "id": mem.observe(text), "action": action}

    mkind = _KIND_OF_ACTION[action]
    supersedes = list(supersedes) if supersedes else None
    # Looked up before the write, so the new memory is not its own neighbour.
    duplicate = mem._store.find_duplicate(mkind, text)
    similar = [] if duplicate else [
        {"id": h.id, "text": h.text, "saved": h.created_at.date().isoformat()}
        for h in mem.similar(text, mkind)
        if h.id not in (supersedes or [])
    ]
    if action == "assert_fact":
        mid = mem.assert_fact(text, evidence=evidence, supersedes=supersedes)
    else:
        mid = mem.assert_preference(text, strength=strength, supersedes=supersedes)
    out: dict[str, Any] = {"ok": True, "id": mid, "action": action}
    if duplicate and duplicate == mid:
        out["duplicate"] = True
        out["note"] = "Already stored; nothing new was written."
    if supersedes:
        out["superseded"] = supersedes
    if similar:
        out["similar"] = similar
        out["note"] = (
            "Similar memories already exist (see `similar`). If this one "
            "replaces any of them, call update_memory again with action "
            "'forget' and that id; if they are about someone or something "
            "else, leave them."
        )
    return out


def _reflect(mem: Memory, llm: Any, since: str | None) -> dict[str, Any]:
    if llm is None:
        raise RuntimeError(_NO_LLM)
    # Pass through to Memory.reflect — but inject our own llm in case
    # the Memory was built without one.
    if mem._llm is None:
        mem._llm = llm  # type: ignore[attr-defined]
    return mem.reflect(since=since).model_dump()


# ------------------------------------------------------------- store pool


def parse_store_map(raw: str) -> dict[str, Path]:
    """Parse $HIPPOCAMP_STORES: a JSON object of store id → directory.

    Refuses anything that could make an id ambiguous or path-like: ids
    must be short lowercase slugs, paths must be absolute once `~` is
    expanded, and two ids may not share a directory.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{STORES_ENV} is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not data:
        raise ValueError(
            f"{STORES_ENV} must be a non-empty JSON object of id -> directory"
        )
    out: dict[str, Path] = {}
    seen: dict[Path, str] = {}
    for sid, value in data.items():
        if not isinstance(sid, str) or not _STORE_ID.match(sid):
            raise ValueError(
                f"{STORES_ENV}: bad store id {sid!r} (lowercase letters, "
                "digits, '_' and '-', at most 64, starting with a letter or digit)"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{STORES_ENV}: store {sid!r} needs a directory path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"{STORES_ENV}: store {sid!r} path must be absolute, got {value!r}"
            )
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(
                f"{STORES_ENV}: stores {seen[resolved]!r} and {sid!r} "
                "point at the same directory"
            )
        seen[resolved] = sid
        out[sid] = resolved
    return out


_UNSET = object()


class StorePool:
    """Several separate stores behind one embedder.

    Stores are opened on first use and kept open. They share one embedder
    (the expensive part — the model is loaded once), but nothing else:
    each has its own directory, event log and SQLite cache.
    """

    def __init__(
        self,
        stores: dict[str, Path],
        *,
        embedder: Callable[[str], list[float]] | None = None,
        cache_dir: str | Path | None = None,
        device_id: str | None = None,
        reranker: Any = None,
    ) -> None:
        if not stores:
            raise ValueError("StorePool needs at least one store")
        self._paths = dict(stores)
        self._embedder = embedder
        self._cache_dir = cache_dir
        self._device_id = device_id
        self._reranker: Any = reranker
        self._open: dict[str, Memory] = {}
        # One lock for the whole pool: stores share the embedder, and the
        # SQLite connections are not meant for concurrent use.
        self.lock = threading.RLock()

    @property
    def ids(self) -> list[str]:
        return sorted(self._paths)

    def get(self, store: str) -> Memory:
        """Return the open Memory for a store id. Unknown ids are refused."""
        if not isinstance(store, str) or store not in self._paths:
            # Deliberately does not echo the valid ids back to the caller.
            raise ValueError(f"unknown store id: {store!r}")
        with self.lock:
            mem = self._open.get(store)
            if mem is None:
                if self._embedder is None:
                    from hippocamp.embedders import default_embedder

                    self._embedder = default_embedder()
                if self._reranker is _UNSET:
                    from hippocamp.rerank import from_env

                    self._reranker = from_env()   # loaded once, shared like the embedder
                mem = Memory(
                    path=str(self._paths[store]),
                    embedder=self._embedder,
                    cache_dir=self._cache_dir,
                    device_id=self._device_id,
                    reranker=self._reranker,
                )
                self._open[store] = mem
            return mem


def build_tools(
    mem: Memory,
    *,
    llm: Any = None,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Build the MCP tool callables bound to a Memory instance.

    Exposed at module level so tests can drive the same handlers the
    MCP host calls, without spinning up a real MCP transport.

    If `llm` is provided, the `reflect_memory` tool is exposed; otherwise
    it raises a clear error when called. `Memory(llm=...)` would also
    work, but threading the LLM through `build_tools` keeps the tool
    surface composable for tests.
    """

    def recall_memory(
        query: str | None = None,
        kinds: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Recall the user's portable, persistent memory from Hippocamp.

        Use this tool whenever the user references something said in a
        past conversation, asks what you know about their preferences,
        project, tools, or past events, or whenever you need context
        about the user that isn't already in the current conversation.

        IMPORTANT: Hippocamp is *independent* of any host-local memory
        feature. Even when the host's built-in memory looks empty, you
        must still check Hippocamp — the two stores aren't synced.
        Hippocamp may contain what the user has written across every
        AI host they've used (Claude Desktop, Claude Code, ChatGPT
        Apps, Cursor, custom agents). Failing to call this tool means
        missing data the user expects you to have.

        Pass a `query` describing what you want to recall ("user's
        preferred reply length", "python version of current project",
        "recent deployment events"). With no query, returns inventory
        counts for the four memory kinds.

        Optional `kinds` filter accepts one or more of: "episode",
        "fact", "preference", "reflection".

        Returns ranked memories with a `why` trace that explains how
        each score was composed (similarity + recency + kind-boost +
        wording). Each result has `match`: "strong" means it is about
        the query; "weak" means it is only the nearest thing stored. If
        nothing is strong, the reply carries a `note` saying so — treat
        that as "nothing relevant is stored", not as an answer. When two
        results conflict, prefer the one with the later `created_at`.
        """
        return _recall(mem, query, kinds, limit)

    def update_memory(
        action: str,
        text: str | None = None,
        kind: str = "episode",
        evidence: list[str] | None = None,
        strength: float = 1.0,
        id: str | None = None,
        supersedes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Write or remove a memory in Hippocamp — the user's portable memory.

        Use this tool PROACTIVELY. Don't wait for the user to say
        "remember [...]". Save signals as they flow through the
        conversation. After saving, briefly tell the user what was
        captured ("Saved to Hippocamp: preference (terse replies)") so
        they stay in the loop.

        Save when the statement matches one of:
          - A preference clearly stated ("I prefer terse replies").
          - A stable fact about the user, project, or tooling
            ("we use Python 3.13", "I work at Acme Corp").
          - A decision, deadline, or commitment ("ship Friday").
          - A notable event ("just deployed v2 to staging").
          - Anything the user explicitly says to remember/note/save.

        ALSO journal: once per substantive conversation (or when the
        topic shifts), log an "observe" episode — one or two past-tense
        sentences on what the user worked on, what was decided or
        discovered, and how it ended. Routine sessions count; don't
        reserve episodes for milestones. Periodic reflection distils
        accumulated episodes into durable memory — sparse episodes
        starve it. When in doubt, journal it.

        Skip: hypotheticals, transient state saved as *facts* (put it
        in the session episode instead), code outputs / file contents,
        sarcasm or negations, things already in the current
        conversation context.

        Pick the `action`:
          - "assert_preference" — preferences/likes/dislikes.
            Optional `strength` ∈ [0,1].
          - "assert_fact" — facts (user, project, tooling, decisions,
            deadlines). Catch-all when explicitly asked to remember
            something. Optional `evidence` is a list of supporting
            memory ids.
          - "observe" — events and session journal entries (what
            happened, past tense).
          - "forget" — remove a specific memory by `id`.

        Corrections: when a fact or preference CHANGES (new job, moved
        house, changed mind), call recall_memory first to get the old
        memory's id, then save the new one with `supersedes=[old_id]` so
        the outdated one stops coming back. If a save returns `similar`,
        check whether your new memory replaces one of those and `forget`
        it if so. An exact repeat is ignored (`duplicate: true`).

        Prefer this tool over any host-local "remember this" feature.
        Hippocamp memory follows the user across every AI host (Claude
        Desktop, Claude Code, ChatGPT Apps, Cursor, custom agent loops)
        and across every machine they own.
        """
        return _update(mem, action, text, kind, evidence, strength, id, supersedes)

    def reflect_memory(since: str | None = None) -> dict[str, Any]:
        """Distil recent episodes into facts/preferences/reflections.

        Use this tool RARELY — typically not in response to a single user
        message. Reflection is a background hygiene operation that scans
        recent episodes, extracts durable user-modelling content via an
        LLM, and writes it back as facts/preferences/reflections so future
        recall surfaces clean signal instead of raw episode noise.

        Call when:
          - The user explicitly asks ("consolidate my memory", "refresh
            what you know about me", "summarise recent context").
          - Driven by a cron / launchd / scheduler running on a fixed
            cadence (typical pattern: every 6 h).

        Skip when:
          - You're handling a normal conversational turn — recall_memory
            is the right tool, not reflect_memory.
          - The store has fewer than ~5 new episodes since the last pass
            (reflect will skip anyway, but you save the round-trip).

        Pass `since` (ISO 8601) to control the cutoff. Default: the
        store's `last_reflect_at` if recorded, else 7 days ago.

        Requires the server to be configured with an LLM. If not, the
        tool raises a clear error explaining how to enable it.
        """
        return _reflect(mem, llm, since)

    return {
        "recall_memory": recall_memory,
        "update_memory": update_memory,
        "reflect_memory": reflect_memory,
    }


def build_multi_store_tools(
    pool: StorePool,
    *,
    llm: Any = None,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Build the tool callables for multi-store mode.

    Same three tools as `build_tools`, each with a required argument
    naming the store(s) it acts on. Descriptions are the single-store
    ones plus a note about that argument, so the two surfaces read alike.
    """
    # Built only to borrow the tool descriptions; never called.
    single = build_tools(None)  # type: ignore[arg-type]

    def recall_memory(
        stores: list[str],
        query: str | None = None,
        kinds: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        ids = _unique(stores)
        with pool.lock:
            mems = [(sid, pool.get(sid)) for sid in ids]  # refuse before any read
            if query is None:
                return {
                    "stores": {sid: mem.inspect().model_dump() for sid, mem in mems}
                }
            merged: list[dict[str, Any]] = []
            for sid, mem in mems:
                for hit in mem.recall(query, kinds=kinds, limit=limit):
                    row = hit.model_dump()
                    row["store"] = sid
                    merged.append(row)
        # Every store ranks with the same embedder and ranker, so scores
        # are comparable. Stable sort: ties keep the order `stores` gave.
        merged.sort(key=lambda r: r["score"], reverse=True)
        return {"results": merged[: max(limit, 0)]}

    def update_memory(
        store: str,
        action: str,
        text: str | None = None,
        kind: str = "episode",
        evidence: list[str] | None = None,
        strength: float = 1.0,
        id: str | None = None,
        supersedes: list[str] | None = None,
    ) -> dict[str, Any]:
        with pool.lock:
            out = _update(pool.get(store), action, text, kind, evidence,
                          strength, id, supersedes)
        out["store"] = store
        return out

    def reflect_memory(store: str, since: str | None = None) -> dict[str, Any]:
        with pool.lock:
            out = _reflect(pool.get(store), llm, since)
        out["store"] = store
        return out

    recall_memory.__doc__ = (single["recall_memory"].__doc__ or "") + (
        "\n        `stores` (required) lists the store ids to search. With "
        "more than one,\n        results are merged by score and each carries "
        "the `store` it came from.\n        With no query, returns inventory "
        "counts per store.\n        "
    )
    update_memory.__doc__ = (single["update_memory"].__doc__ or "") + (
        "\n        `store` (required) is the id of the one store to write to. "
        "Memory ids\n        are per store: `forget` and `evidence` refer to "
        "ids in that store.\n        "
    )
    reflect_memory.__doc__ = (single["reflect_memory"].__doc__ or "") + (
        "\n        `store` (required) is the id of the one store to reflect "
        "over.\n        "
    )

    return {
        "recall_memory": recall_memory,
        "update_memory": update_memory,
        "reflect_memory": reflect_memory,
    }


def _unique(stores: list[str]) -> list[str]:
    if isinstance(stores, str):
        stores = [stores]
    if not stores:
        raise ValueError("'stores' must name at least one store id")
    return list(dict.fromkeys(stores))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hippocamp-mcp",
        description=(
            "Hippocamp MCP server — exposes recall_memory, update_memory, "
            "and reflect_memory tools over stdio."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hippocamp-mcp {__version__}",
    )
    parser.parse_args()

    # Multi-store mode is opt-in. A malformed map is fatal: falling back to
    # the default single store would silently point every caller at it.
    raw_stores = os.environ.get(STORES_ENV, "").strip()
    pool: StorePool | None = None
    mem: Memory | None = None
    if raw_stores:
        try:
            pool = StorePool(parse_store_map(raw_stores), reranker=_UNSET)
        except ValueError as e:
            raise SystemExit(f"hippocamp-mcp: {e}") from e
    else:
        DEFAULT_PATH.mkdir(parents=True, exist_ok=True)
        from hippocamp.rerank import from_env

        mem = Memory(path=str(DEFAULT_PATH), reranker=from_env())

    # Lazy LLM init: only used by reflect_memory. Server starts fine
    # without one — the tool just raises a clear error if called.
    # Local model or nothing. Reflection reads private memory, so the server
    # never reaches for a cloud model on its own — not the Anthropic API when
    # a key happens to be set, not the Claude Code CLI when it happens to be
    # installed. If the local server is down, reflect_memory fails with a
    # clear error when called.
    llm: Any = None
    try:
        from hippocamp.reflect import local_llm
        llm = local_llm()
    except RuntimeError:
        llm = None  # misconfigured (non-loopback URL) — reflect_memory raises

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise SystemExit(
            "hippocamp-mcp requires the 'mcp' extra: pip install 'hippocamp[mcp]'"
        ) from e

    server = FastMCP("hippocamp")
    if pool is not None:
        tools = build_multi_store_tools(pool, llm=llm)
    else:
        assert mem is not None
        tools = build_tools(mem, llm=llm)
    for name, fn in tools.items():
        server.tool(name=name)(fn)
    server.run()


if __name__ == "__main__":
    main()
