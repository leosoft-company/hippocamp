"""Hippocamp's public Memory class — the single entry point for the library.

A Hippocamp store has two layers:

* **Store directory** (designed to be cloud-synced): contains `meta.json`
  and `events/<device_id>.jsonl` files — the source of truth.
* **Cache** (always local): a SQLite file that's a fast index over the
  events. Lives in the OS cache directory by default, keyed by
  `meta.store_id` so multiple stores get isolated caches automatically.

The cache can be wiped at any time and rebuilt via `replay()`. This is
why cloud sync is safe — no concurrent-write risk on derived state.
"""

from __future__ import annotations

import json
import os
import platform
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from hippocamp.events import Event, EventLog, EventOp, new_event_id
from hippocamp.meta import (
    StoreMeta,
    load_or_init_device_id,
    load_or_init_meta,
    write_meta,
)
from hippocamp.ranker import rank, tokenize
from hippocamp.store import Store, normalise
from hippocamp.types import CompactReport, Inventory, RecallResult

Kind = Literal["episode", "fact", "preference", "reflection"]
KINDS = ("episode", "fact", "preference", "reflection")


def _default_cache_dir() -> Path:
    """OS-standard base directory for local-only caches."""
    home = Path.home()
    sysname = platform.system()
    if sysname == "Darwin":
        return home / "Library" / "Caches" / "hippocamp"
    if sysname == "Windows":
        local = os.environ.get("LOCALAPPDATA")
        return Path(local) / "hippocamp" if local else home / "AppData" / "Local" / "hippocamp"
    return home / ".cache" / "hippocamp"


class Memory:
    """Local-first agent memory.

    All data lives under a single directory containing `meta.json`,
    `events.jsonl`, and `store.db`. No network calls are made unless you
    opt in by passing an `llm` (used only by `reflect()`).

    Pass `path=":memory:"` to skip the event log entirely (used by tests
    and ephemeral integrations).
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        embedder: Callable[[str], list[float]] | None = None,
        llm: Any | None = None,
        device_id: str | None = None,
        cache_dir: str | Path | None = None,
        force_embedder: bool = False,
    ) -> None:
        if embedder is None:
            from hippocamp.embedders import default_embedder

            embedder = default_embedder()
        self._embedder = embedder
        self._llm = llm

        self._meta: StoreMeta | None = None
        self._events: EventLog | None = None
        self._store_dir: Path | None = None
        self._cache_path: Path | None = None
        self._device_id: str

        embedding_dim = getattr(embedder, "dim", None)

        if str(path) == ":memory:":
            self._store = Store(":memory:", embedding_dim=embedding_dim)
            self._device_id = device_id or "in-memory"
            return

        # Resolve store dir and a candidate legacy cache path.
        raw = Path(path).expanduser()
        if raw.suffix == ".db" or (raw.exists() and raw.is_file()):
            # Legacy / explicit-file form: `path` points at store.db itself.
            store_dir = raw.parent
            legacy_cache_path: Path | None = raw
        else:
            # New form: `path` is the store directory.
            store_dir = raw
            legacy_cache_path = None

        store_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(store_dir, 0o700)  # personal data: owner only
        self._store_dir = store_dir

        embedder_name = getattr(embedder, "name", type(embedder).__name__)
        meta_path = store_dir / "meta.json"
        meta_was_new = not meta_path.exists()
        self._meta = load_or_init_meta(
            meta_path, embedder_name=embedder_name, force=force_embedder
        )
        self._device_id = device_id or load_or_init_device_id()

        # Resolve cache path: explicit arg > env > legacy file > OS default.
        self._cache_path = self._resolve_cache_path(cache_dir, legacy_cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self._cache_path.parent, 0o700)
        self._store = Store(str(self._cache_path), embedding_dim=embedding_dim)

        self._events = EventLog(store_dir / "events", self._device_id)
        self._migrate_legacy_single_file_log()
        if meta_was_new and self._events.count() == 0:
            self._bootstrap_events_from_cache()
        self._events.secure()
        self._catch_up()
        self._store.scrub()  # anything purged while catching up is gone from the file too

    def _catch_up(self) -> None:
        """Bring the cache level with the event log.

        The cache remembers how far into each device file (in bytes) it has
        applied. Opening a store therefore picks up whatever other devices
        (or a file-sync tool) added since last time, without re-embedding
        what is already there. A file that got shorter, or an empty cache,
        means start over.
        """
        assert self._events is not None
        counts = self._events.sizes()
        applied = self._store.applied()
        if applied is None:
            if self._cache_row_count() == 0:
                if counts:
                    self.replay()
                else:
                    self._store.set_applied({})
                return
            # A cache written by v0.11: its rows are current, but it has no
            # record of why inactive rows are inactive. Recover that from the
            # log (no re-embedding) so forgotten memories can be purged.
            for ev in self._events.read_all():
                if ev.op in (EventOp.FORGET, EventOp.PURGE, EventOp.MARK_REFLECTED):
                    self._apply(ev)
                elif ev.supersedes:
                    self._store.mark_superseded(ev.supersedes, by=ev.mem_id, when=ev.ts)
            self._store.set_applied(counts)
            return
        if any(counts.get(f, 0) < n for f, n in applied.items()):
            self.replay()
            return
        if counts != applied:
            for ev in self._events.read_from(applied):
                self._apply(ev)
            self._store.set_applied(counts)

    def _resolve_cache_path(
        self,
        explicit_cache_dir: str | Path | None,
        legacy_cache_path: Path | None,
    ) -> Path:
        """Decide where store.db lives. Priority: arg > env > legacy > default."""
        assert self._meta is not None  # narrowed by caller

        if explicit_cache_dir is not None:
            base = Path(explicit_cache_dir).expanduser()
            return base / self._meta.store_id / "store.db"

        env = os.environ.get("HIPPOCAMP_CACHE_DIR")
        if env:
            return Path(env).expanduser() / self._meta.store_id / "store.db"

        if legacy_cache_path is not None:
            return legacy_cache_path

        return _default_cache_dir() / self._meta.store_id / "store.db"

    def _cache_row_count(self) -> int:
        return self._store._conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]

    @property
    def meta(self) -> StoreMeta | None:
        return self._meta

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def store_dir(self) -> Path | None:
        """The cloud-syncable directory holding meta.json + events/."""
        return self._store_dir

    @property
    def cache_path(self) -> Path | None:
        """Path to the local SQLite cache (None for in-memory stores)."""
        return self._cache_path

    @property
    def events_dir(self) -> Path | None:
        return self._events.dir if self._events else None

    @property
    def events_own_file(self) -> Path | None:
        return self._events.own_file if self._events else None

    # ------------------------------------------------------------------ write

    def observe(
        self,
        text: str,
        *,
        actors: list[str] | None = None,
        at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        text = _require_text(text)
        eid = _new_id("ep")
        when = at or _now()
        self._emit(
            EventOp.OBSERVE,
            mem_id=eid,
            ts=when,
            text=text,
            actors=actors or [],
            metadata=metadata or {},
        )
        self._store.insert(
            id=eid,
            kind="episode",
            text=text,
            created_at=when,
            metadata=metadata or {},
            embedding=self._embedder(text),
            extras={"actors": actors or []},
        )
        return eid

    def assert_fact(
        self,
        text: str,
        *,
        evidence: list[str] | None = None,
        supersedes: list[str] | None = None,
    ) -> str:
        """Record a fact. Saving the same sentence again returns the existing
        id instead of a second copy. `supersedes` names active memories this
        fact replaces; they stop being returned."""
        text = _require_text(text)
        self._check_supersedes(supersedes)
        dup = self._store.find_duplicate("fact", text)
        if dup and dup not in (supersedes or []):
            if supersedes:
                self._supersede_only(supersedes, by=dup)
            return dup
        fid = _new_id("fa")
        when = _now()
        self._emit(
            EventOp.ASSERT_FACT,
            mem_id=fid,
            ts=when,
            text=text,
            evidence=evidence or [],
            supersedes=supersedes or [],
        )
        self._store.insert(
            id=fid,
            kind="fact",
            text=text,
            created_at=when,
            metadata={"evidence": evidence or [], "supersedes": supersedes or []},
            embedding=self._embedder(text),
        )
        if supersedes:
            self._store.mark_superseded(supersedes, by=fid, when=when)
        return fid

    def assert_preference(
        self,
        text: str,
        *,
        strength: float = 1.0,
        supersedes: list[str] | None = None,
    ) -> str:
        """Record a preference. `strength` (0–1) is how firmly it is held and
        nudges ranking. Duplicates and `supersedes` work as for facts."""
        text = _require_text(text)
        strength = min(1.0, max(0.0, float(strength)))
        self._check_supersedes(supersedes)
        dup = self._store.find_duplicate("preference", text)
        if dup and dup not in (supersedes or []):
            if supersedes:
                self._supersede_only(supersedes, by=dup)
            return dup
        pid = _new_id("pr")
        when = _now()
        self._emit(
            EventOp.ASSERT_PREF,
            mem_id=pid,
            ts=when,
            text=text,
            strength=strength,
            supersedes=supersedes or [],
        )
        self._store.insert(
            id=pid,
            kind="preference",
            text=text,
            created_at=when,
            metadata={"strength": strength},
            embedding=self._embedder(text),
        )
        if supersedes:
            self._store.mark_superseded(supersedes, by=pid, when=when)
        return pid

    def _check_supersedes(self, ids: list[str] | None) -> None:
        for old in ids or []:
            if self._store.state(old) is None:
                raise ValueError(f"cannot supersede {old!r}: no such memory")

    def _supersede_only(self, ids: list[str], *, by: str) -> None:
        """Retire `ids` in favour of an existing memory `by` (the replacement
        text was already stored). Logged as forgets so every device agrees."""
        for old in ids:
            if old != by and self._store.state(old) == "active":
                when = _now()
                self._emit(EventOp.FORGET, mem_id=old, ts=when,
                           metadata={"superseded_by": by})
                self._store.mark_superseded([old], by=by, when=when)

    def assert_reflection(self, text: str, *, sources: list[str] | None = None) -> str:
        text = _require_text(text)
        dup = self._store.find_duplicate("reflection", text)
        if dup:
            return dup
        rid = _new_id("re")
        when = _now()
        self._emit(
            EventOp.ASSERT_REFLECTION,
            mem_id=rid,
            ts=when,
            text=text,
            sources=sources or [],
        )
        self._store.insert(
            id=rid,
            kind="reflection",
            text=text,
            created_at=when,
            metadata={"sources": sources or []},
            embedding=self._embedder(text),
        )
        return rid

    # ------------------------------------------------------------------- read

    def recall(
        self,
        query: str | None = None,
        *,
        kinds: Iterable[Kind] | None = None,
        limit: int = 10,
        context: dict[str, Any] | None = None,
    ) -> list[RecallResult]:
        """Ranked memories for `query`. Reading never changes the store, so
        the same question gives the same answer. Each result carries
        `match`: "strong" when it is about the question, "weak" when it is
        only the nearest thing stored."""
        kind_list = list(kinds) if kinds else None
        for k in kind_list or []:
            if k not in KINDS:
                raise ValueError(f"unknown kind {k!r}; expected one of {list(KINDS)}")
        if query is None or not str(query).strip() or limit <= 0:
            return []

        query_emb = self._embed_query(query)
        query_tokens = tokenize(query)
        # Durable knowledge and journal entries are fetched separately. A
        # single nearest-N cut gets filled by look-alike episodes ("asked
        # for train times" × 30) and the one preference that answers the
        # question never reaches the ranker.
        n_cand = max(limit * 4, 40)
        groups = (
            [kind_list] if kind_list
            else [["fact", "preference", "reflection"], ["episode"]]
        )
        by_id: dict[str, Any] = {}
        for group in groups:
            for row in self._store.search(
                query_emb=query_emb, kinds=group, candidate_limit=n_cand,
                query_tokens=query_tokens,
            ):
                by_id.setdefault(row.id, row)

        ranked = rank(by_id.values(), query_emb=query_emb, context=context, now=_now(),
                      query_tokens=query_tokens,
                      strong_similarity=getattr(self._embedder, "strong_similarity", None))

        from hippocamp.ranker import NEAR_DUPLICATE_EPISODE, _cosine

        seen: set[tuple[str, str]] = set()
        results: list[RecallResult] = []
        tail: list[RecallResult] = []
        for r in ranked[: max(limit * 3, 30)]:
            # Stores written before duplicates were refused may hold the
            # same sentence several times; show it once.
            key = (r.kind, normalise(r.text))
            if key in seen:
                continue
            seen.add(key)
            if r.kind == "episode" and any(
                p.kind == "episode"
                and _cosine(by_id[r.id].embedding, by_id[p.id].embedding)
                >= NEAR_DUPLICATE_EPISODE
                for p in results
            ):
                tail.append(r)
                continue
            results.append(r)
            if len(results) >= limit:
                break
        if len(results) < limit:
            rest = tail + [r for r in ranked[max(limit * 3, 30):]
                           if (r.kind, normalise(r.text)) not in seen]
            results += rest[: limit - len(results)]
        return results

    def _embed_query(self, text: str) -> list[float]:
        fn = getattr(self._embedder, "embed_query", None)
        return fn(text) if fn is not None else self._embedder(text)

    def similar(
        self, text: str, kind: str, *, limit: int = 3, min_similarity: float | None = None
    ) -> list[RecallResult]:
        """Active memories of `kind` that read like `text` — candidates for
        "is this new, or does it replace one of these?"."""
        from hippocamp.ranker import _cosine

        emb = self._embedder(text)
        bar = min_similarity if min_similarity is not None else getattr(
            self._embedder, "similar_statement", 0.6)
        rows = self._store.search(query_emb=emb, kinds=[kind], candidate_limit=limit * 4)
        scored = sorted(((_cosine(emb, r.embedding), r) for r in rows),
                        key=lambda x: -x[0])
        hits = rank([r for s, r in scored if s >= bar][:limit], query_emb=emb,
                    context=None, now=_now())
        return sorted(hits, key=lambda h: -h.why.similarity)

    # ------------------------------------------------------------ consolidate

    def reflect(self, *, since: str | datetime | None = None):
        if self._llm is None:
            raise RuntimeError("reflect() requires an llm; pass `llm=...` to Memory()")
        from hippocamp.reflect import run_reflection

        return run_reflection(self, self._llm, since=since)

    @property
    def last_reflect_at(self) -> datetime | None:
        """When `reflect()` was last completed against this store. None if never."""
        return self._meta.last_reflect_at if self._meta is not None else None

    def set_last_reflect_at(self, when: datetime) -> None:
        """Persist the reflect cutoff to meta.json. No-op for in-memory stores."""
        if self._meta is None or self._store_dir is None:
            return
        self._meta.last_reflect_at = when
        write_meta(self._store_dir / "meta.json", self._meta)

    def set_reflect_tracking(self) -> None:
        if self._meta is None or self._store_dir is None:
            return
        self._meta.reflect_tracking = True
        write_meta(self._store_dir / "meta.json", self._meta)

    # ----------------------------------------------------------------- forget

    def forget(self, id: str) -> None:
        """Stop returning a memory. The text stays on disk until `purge` or
        `compact`, so a mistaken forget can still be found in the log."""
        if self._store.state(id) is None:
            raise ValueError(f"cannot forget {id!r}: no such memory")
        when = _now()
        self._emit(EventOp.FORGET, mem_id=id, ts=when)
        self._store.tombstone(id, when=when)

    def purge(self, id: str) -> None:
        """Delete a memory for good: its row, its vector, and its text in
        every event file in this store. Other devices do the same when they
        next read the log. Memories distilled *from* it are separate
        memories — forget those too if they repeat it."""
        when = _now()
        self._emit(EventOp.PURGE, mem_id=id, ts=when)
        self._do_purge(id, when)
        self._store.scrub()

    def _do_purge(self, id: str, when: datetime) -> None:
        self._store.purge(id, when=when)
        if self._events is not None and self._events.redact(id):
            self._events.secure()

    def expire_before(
        self, when: str | datetime, *, kinds: Iterable[Kind] | None = None
    ) -> int:
        """Forget every active memory created before `when` ("90d", "12h" or
        an ISO date), optionally only of some kinds. Returns how many."""
        cutoff = _parse_when(when)
        ids = self._store.list_active_before(cutoff, list(kinds) if kinds else None)
        for mem_id in ids:
            self.forget(mem_id)
        return len(ids)

    def compact(self) -> CompactReport:
        """Purge everything that was forgotten. Replaced facts are kept:
        they are history, not deletion requests."""
        ids = self._store.forgotten_ids()
        for mem_id in ids:
            when = _now()
            self._emit(EventOp.PURGE, mem_id=mem_id, ts=when)
            self._do_purge(mem_id, when)
        self._store.scrub()
        return CompactReport(merged=0, dropped=len(ids))

    def mark_reflected(self, episode_ids: list[str]) -> None:
        """Record that consolidation has read these episodes."""
        if not episode_ids:
            return
        when = _now()
        self._emit(EventOp.MARK_REFLECTED, mem_id="reflect", ts=when,
                   sources=list(episode_ids))
        self._store.mark_reflected(episode_ids, when=when)

    # ---------------------------------------------------------------- reindex

    def reindex(self) -> int:
        """Recompute embeddings for all active memories with the current embedder.

        Updates `meta.json`'s embedder name if it has changed (i.e. the
        caller opened with `force_embedder=True` to switch). Returns the
        number of memories re-embedded.

        Also rewrites the vec0 mirror so newly-re-embedded rows are
        immediately searchable.
        """
        if self._meta is None or self._store_dir is None:
            raise RuntimeError("reindex() requires an on-disk store")

        embedder_name = getattr(self._embedder, "name", type(self._embedder).__name__)

        if embedder_name != self._meta.embedder:
            self._meta.embedder = embedder_name
            write_meta(self._store_dir / "meta.json", self._meta)

        rows = self._store._conn.execute(
            "SELECT id, text FROM memories WHERE superseded_at IS NULL"
        ).fetchall()

        n = 0
        for row_id, text in rows:
            new_emb = self._embedder(text or "")
            self._store.update_embedding(row_id, new_emb)
            n += 1
        return n

    # ----------------------------------------------------------------- replay

    def replay(self) -> int:
        """Wipe the cache and rebuild it from `events.jsonl`. Returns event count.

        Idempotent. Used after embedder upgrades, after corrupting the cache,
        or after merging events from another device.
        """
        if self._events is None:
            raise RuntimeError("replay() requires an on-disk store")

        self._store.wipe()
        counts = self._events.sizes()
        n = 0
        for event in self._events.read_all():
            self._apply(event)
            n += 1
        self._store.set_applied(counts)
        return n

    # ---------------------------------------------------------------- inspect

    def inspect(self) -> Inventory:
        inv = self._store.inventory()
        inv.last_reflect = self.last_reflect_at
        return inv

    # --------------------------------------------------------------- lifecycle

    def close(self) -> None:
        self._store.close()

    # ---------------------------------------------------------------- internal

    def _emit(self, op: EventOp, *, mem_id: str, ts: datetime, **fields: Any) -> None:
        if self._events is None:
            return
        end = self._events.append(
            Event(
                id=new_event_id(),
                ts=ts,
                device=self._device_id,
                op=op,
                mem_id=mem_id,
                **fields,
            )
        )
        # Our own line is applied by the caller. Record "applied up to here"
        # as a high-water mark: two processes sharing this cache can race,
        # and a mark can be raised twice but never past the end of the file.
        self._store.kv_max("applied:" + self._events.own_file.name, end)

    def _migrate_legacy_single_file_log(self) -> int:
        """Split a legacy `events.jsonl` (slice-1 layout) into per-device files.

        Slice 1.6 introduced per-device event files; an older single-file
        log is split by the `device` field on each event so writes from
        each origin device land in `events/<device>.jsonl`.
        """
        if self._events is None or self._store_dir is None:
            return 0
        legacy = self._store_dir / "events.jsonl"
        if not legacy.exists():
            return 0

        events_dir = self._events.dir
        events_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        with legacy.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = Event.model_validate_json(line)
                target = events_dir / f"{ev.device}.jsonl"
                with target.open("a", encoding="utf-8") as out:
                    out.write(line + "\n")
                moved += 1
        legacy.unlink()
        return moved

    def _bootstrap_events_from_cache(self) -> int:
        """Generate events for rows already in the cache.

        Runs once when a pre-v0.3 store is opened for the first time
        under the new architecture. Without this, the cache and the
        (empty) event log would silently disagree, and `replay()`
        would lose every existing memory.

        Returns the number of events written.
        """
        import json as _json

        if self._events is None:
            return 0

        rows = self._store._conn.execute(
            "SELECT id, kind, text, created_at, superseded_at, metadata "
            "FROM memories ORDER BY created_at"
        ).fetchall()
        if not rows:
            return 0

        n = 0
        for mem_id, kind, text, created_at, superseded_at, metadata_json in rows:
            ts = datetime.fromisoformat(created_at)
            meta = _json.loads(metadata_json or "{}")

            if kind == "episode":
                self._events.append(
                    Event(
                        id=new_event_id(),
                        ts=ts,
                        device=self._device_id,
                        op=EventOp.OBSERVE,
                        mem_id=mem_id,
                        text=text or "",
                        metadata=meta,
                    )
                )
            elif kind == "fact":
                self._events.append(
                    Event(
                        id=new_event_id(),
                        ts=ts,
                        device=self._device_id,
                        op=EventOp.ASSERT_FACT,
                        mem_id=mem_id,
                        text=text or "",
                        evidence=meta.get("evidence", []) or [],
                        supersedes=meta.get("supersedes", []) or [],
                    )
                )
            elif kind == "preference":
                self._events.append(
                    Event(
                        id=new_event_id(),
                        ts=ts,
                        device=self._device_id,
                        op=EventOp.ASSERT_PREF,
                        mem_id=mem_id,
                        text=text or "",
                        strength=meta.get("strength", 1.0),
                    )
                )
            else:
                continue
            n += 1

            # If the row was already tombstoned, also emit a forget event
            # at the original tombstone time, so replay reconstructs the
            # superseded state correctly.
            if superseded_at:
                forget_ts = datetime.fromisoformat(superseded_at)
                self._events.append(
                    Event(
                        id=new_event_id(),
                        ts=forget_ts,
                        device=self._device_id,
                        op=EventOp.FORGET,
                        mem_id=mem_id,
                    )
                )
                n += 1

        return n

    def _apply(self, event: Event) -> None:
        """Apply an event to the cache only (no event emission). Used by replay."""
        if event.text is None and event.op in (
            EventOp.OBSERVE, EventOp.ASSERT_FACT, EventOp.ASSERT_PREF,
            EventOp.ASSERT_REFLECTION,
        ) and event.metadata.get("redacted"):
            return  # purged: only the blanked line is left
        if event.op == EventOp.OBSERVE:
            self._store.insert(
                id=event.mem_id,
                kind="episode",
                text=event.text or "",
                created_at=event.ts,
                metadata=event.metadata,
                embedding=self._embedder(event.text or ""),
                extras={"actors": event.actors},
            )
        elif event.op == EventOp.ASSERT_FACT:
            self._store.insert(
                id=event.mem_id,
                kind="fact",
                text=event.text or "",
                created_at=event.ts,
                metadata={
                    "evidence": event.evidence,
                    "supersedes": event.supersedes,
                },
                embedding=self._embedder(event.text or ""),
            )
            if event.supersedes:
                self._store.mark_superseded(
                    event.supersedes, by=event.mem_id, when=event.ts
                )
        elif event.op == EventOp.ASSERT_PREF:
            self._store.insert(
                id=event.mem_id,
                kind="preference",
                text=event.text or "",
                created_at=event.ts,
                metadata={"strength": 1.0 if event.strength is None else event.strength},
                embedding=self._embedder(event.text or ""),
            )
            if event.supersedes:
                self._store.mark_superseded(
                    event.supersedes, by=event.mem_id, when=event.ts
                )
        elif event.op == EventOp.ASSERT_REFLECTION:
            self._store.insert(
                id=event.mem_id,
                kind="reflection",
                text=event.text or "",
                created_at=event.ts,
                metadata={"sources": event.sources},
                embedding=self._embedder(event.text or ""),
            )
        elif event.op == EventOp.FORGET:
            if event.metadata.get("superseded_by"):
                self._store.mark_superseded(
                    [event.mem_id], by=event.metadata["superseded_by"], when=event.ts
                )
            else:
                self._store.tombstone(event.mem_id, when=event.ts)
        elif event.op == EventOp.PURGE:
            self._do_purge(event.mem_id, event.ts)
        elif event.op == EventOp.MARK_REFLECTED:
            self._store.mark_reflected(event.sources, when=event.ts)


# ---------------------------------------------------------------------- utils


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _require_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("a memory needs some text")
    return text.strip()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _parse_when(when: str | datetime) -> datetime:
    if isinstance(when, datetime):
        return when
    s = str(when).strip().lower()
    if s.endswith("d"):
        return _now() - timedelta(days=int(s[:-1]))
    if s.endswith("h"):
        return _now() - timedelta(hours=int(s[:-1]))
    return datetime.fromisoformat(s)
