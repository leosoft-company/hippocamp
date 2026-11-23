"""SQLite-backed storage for Hippocamp memories.

Persistence: every memory row lives in the ``memories`` table with its
embedding stored as a JSON array (for portability / human inspection /
event-log replay) AND, when sqlite-vec is available, mirrored into a
``memories_vec`` virtual table keyed by ``memories.rowid`` for true ANN
candidate retrieval.

Without sqlite-vec, ``Store.search`` falls back to returning all active
rows for in-Python ranking. With it, ``Store.search`` does a real
top-K KNN query and the ranker only re-scores the survivors.

Migration: an existing store that has no ``memories_vec`` table gets
one created on first open, then back-filled from ``memories``. Rows
whose embedding dimension does not match the current embedder's dim
are skipped from the vec table — they're invisible to semantic recall
until ``Memory.reindex()`` rewrites their embedding with the current
embedder.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Prefer pysqlite3 (bundles a modern SQLite built WITH enable_load_extension)
# over the stdlib sqlite3, which is built without extension support on
# some Python distributions (notably the python.org macOS installer).
# When pysqlite3 isn't installed we fall back to the stdlib — sqlite-vec
# will then only load on Pythons whose stdlib sqlite3 has extensions.
try:
    import pysqlite3 as sqlite3  # type: ignore[import-not-found]
except ImportError:
    import sqlite3  # type: ignore[no-redef]

from hippocamp.types import CompactReport, Inventory, RawRow

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    superseded_at TEXT,
    salience REAL NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding TEXT NOT NULL,
    extras TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);

-- Why a memory is inactive, kept apart from the row so it holds whatever
-- order events arrive in: a forget that is applied before the memory it
-- names (clock skew, late sync) still wins when the memory shows up.
CREATE TABLE IF NOT EXISTS tombstones (
    mem_id TEXT PRIMARY KEY,
    at TEXT NOT NULL,
    reason TEXT NOT NULL            -- 'superseded' | 'forgotten' | 'purged'
);

CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# Columns added after v0.11. Added in place so existing caches keep working.
_ADDED_COLUMNS = {
    "forgotten_at": "TEXT",     # set when the owner asked to forget (not when replaced)
    "reflected_at": "TEXT",     # episodes: set once consolidation has read them
    "norm": "TEXT",             # normalised text, for exact-duplicate checks
}

# sqlite-vec refuses k above this in a KNN query.
_VEC_MAX_K = 4096


def normalise(text: str) -> str:
    """Canonical form for duplicate checks: case, spacing and trailing
    punctuation don't make a different memory."""
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".!?;, ")


def _serialize_f32(values: list[float]) -> bytes:
    """Serialise a float list to the little-endian f32 blob sqlite-vec expects.

    We re-implement the trivial pack here so the (de)serialisation does
    not depend on importing sqlite_vec — the call sites that need this
    only run after we've already confirmed vec0 is available, but the
    pure-Python pack keeps the code path simple for tests too.
    """
    return struct.pack(f"<{len(values)}f", *values)


class Store:
    """SQLite store. Pass ``embedding_dim`` for true ANN search.

    ``embedding_dim`` is the dimensionality of the embedder the caller
    will use to insert and query. When provided AND sqlite-vec is
    loadable on this Python's sqlite3, a ``memories_vec(embedding
    FLOAT[dim])`` virtual table is created and used for KNN candidate
    retrieval. When None or unavailable, ``search`` returns all active
    rows (correct but slow at scale) and lets the ranker do everything.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_dim: int | None = None,
    ) -> None:
        self._path = str(path)
        self._embedding_dim = embedding_dim
        if self._path != ":memory:":
            # Owner-only before SQLite touches it; the -wal/-shm files
            # inherit the main file's mode.
            fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
            for p in (self._path, self._path + "-wal", self._path + "-shm"):
                if os.path.exists(p):
                    os.chmod(p, 0o600)
        self._conn = sqlite3.connect(self._path, isolation_level=None, timeout=30)
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Deleted rows are overwritten, not just unlinked — purge relies on it.
        self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.executescript(SCHEMA)
        have = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)")}
        for col, decl in _ADDED_COLUMNS.items():
            if col not in have:
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {decl}")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_norm ON memories(kind, norm)"
        )
        for mem_id, text in self._conn.execute(
            "SELECT id, text FROM memories WHERE norm IS NULL"
        ).fetchall():
            self._conn.execute(
                "UPDATE memories SET norm = ? WHERE id = ?", (normalise(text), mem_id)
            )
        self._vec_enabled = self._try_enable_vec()
        if self._vec_enabled:
            self._maybe_backfill_vec()

    # ----------------------------------------------------------------- vec0

    def _try_enable_vec(self) -> bool:
        """Try to load sqlite-vec and create the vec0 virtual table.

        Returns True iff vec search is available for this store. Reasons
        we may return False: ``sqlite_vec`` not installed, this Python's
        ``sqlite3`` built without ``enable_load_extension`` (notably the
        python.org macOS installer), or ``embedding_dim`` not provided.
        """
        if self._embedding_dim is None:
            return False
        try:
            import sqlite_vec  # type: ignore[import-not-found]
        except ImportError:
            log.info("hippocamp_vec_unavailable: sqlite_vec not installed")
            return False
        if not hasattr(self._conn, "enable_load_extension"):
            log.warning(
                "hippocamp_vec_unavailable: this Python's sqlite3 was built "
                "without enable_load_extension; falling back to naive search"
            )
            return False
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
                f"USING vec0(embedding FLOAT[{int(self._embedding_dim)}])"
            )
            return True
        except sqlite3.OperationalError:
            log.warning("hippocamp_vec_unavailable: extension load failed", exc_info=True)
            return False

    def _maybe_backfill_vec(self) -> None:
        """Populate ``memories_vec`` from ``memories`` if it's behind.

        The vec table is empty after the first ever schema apply. It's
        also legitimately empty when the dim has been bumped (e.g.
        embedder switch) and the old vec table was dropped externally.
        In both cases we rebuild from ``memories``, skipping rows whose
        embedding dim doesn't match the current one. Skipped rows remain
        in ``memories`` but stay invisible to semantic recall until
        ``Memory.reindex()`` rewrites them.
        """
        if not self._vec_enabled or self._embedding_dim is None:
            return
        vec_count = self._conn.execute(
            "SELECT COUNT(*) FROM memories_vec"
        ).fetchone()[0]
        active_count = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_at IS NULL"
        ).fetchone()[0]
        if vec_count >= active_count:
            return  # nothing to do (or vec is fully populated)

        self._conn.execute("BEGIN IMMEDIATE")  # not while another process is mid-insert
        try:
            self._backfill_vec(active_count)
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    def _backfill_vec(self, active_count: int) -> None:
        rows = self._conn.execute(
            "SELECT rowid, embedding FROM memories WHERE superseded_at IS NULL"
        ).fetchall()
        inserted = 0
        skipped_dim = 0
        for rowid, emb_json in rows:
            already = self._conn.execute(
                "SELECT 1 FROM memories_vec WHERE rowid = ? LIMIT 1", (rowid,)
            ).fetchone()
            if already:
                continue
            try:
                vec = json.loads(emb_json)
            except (TypeError, ValueError):
                skipped_dim += 1
                continue
            if not isinstance(vec, list) or len(vec) != self._embedding_dim:
                skipped_dim += 1
                continue
            self._conn.execute(
                "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_f32([float(x) for x in vec])),
            )
            inserted += 1

        log.info(
            "hippocamp_vec_backfill",
            extra={
                "inserted": inserted,
                "skipped_dim_mismatch": skipped_dim,
                "active_rows": active_count,
                "embedding_dim": self._embedding_dim,
            },
        )

    @property
    def vec_enabled(self) -> bool:
        return self._vec_enabled

    # -------------------------------------------------------------- insert

    def insert(
        self,
        *,
        id: str,
        kind: str,
        text: str,
        created_at: datetime,
        metadata: dict,
        embedding: list[float],
        extras: dict | None = None,
    ) -> None:
        # A tombstone may already be waiting for this id (events can arrive
        # out of order). Purged: never store it. Otherwise it is born inactive.
        tomb = self._conn.execute(
            "SELECT at, reason FROM tombstones WHERE mem_id = ?", (id,)
        ).fetchone()
        if tomb and tomb[1] == "purged":
            return
        # Row and vector go in together: another process opening this cache
        # mid-write must never see a row without its vector (its backfill
        # would add one and ours would then collide).
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._insert(id, kind, text, created_at, metadata, embedding, extras, tomb)
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    def _insert(self, id, kind, text, created_at, metadata, embedding, extras, tomb) -> None:
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO memories (id, kind, text, created_at, last_seen_at,
                                  metadata, embedding, extras, norm,
                                  superseded_at, forgotten_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
                kind,
                text,
                created_at.isoformat(),
                created_at.isoformat(),
                json.dumps(metadata),
                json.dumps(embedding),
                json.dumps(extras or {}),
                normalise(text),
                tomb[0] if tomb else None,
                tomb[0] if tomb and tomb[1] == "forgotten" else None,
            ),
        )
        if cur.rowcount == 0:
            return  # already present (the same event applied twice)
        if self._vec_enabled and self._embedding_dim is not None:
            if len(embedding) != self._embedding_dim:
                log.warning(
                    "hippocamp_insert_skipped_vec: embedding dim mismatch",
                    extra={
                        "expected_dim": self._embedding_dim,
                        "got_dim": len(embedding),
                        "id": id,
                    },
                )
            else:
                self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (cur.lastrowid,))
                self._conn.execute(
                    "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, _serialize_f32([float(x) for x in embedding])),
                )

    def update_embedding(self, id: str, embedding: list[float]) -> None:
        """Rewrite the embedding for ``id`` in both tables.

        Used by ``Memory.reindex()``. If the new embedding's dim does not
        match the configured vec dim, the JSON column is still rewritten
        but the vec row is removed (and stays out) so recall stays correct.
        """
        emb_json = json.dumps(embedding)
        self._conn.execute(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            (emb_json, id),
        )
        if not self._vec_enabled or self._embedding_dim is None:
            return
        rowid_row = self._conn.execute(
            "SELECT rowid FROM memories WHERE id = ?", (id,)
        ).fetchone()
        if rowid_row is None:
            return
        rowid = rowid_row[0]
        if len(embedding) != self._embedding_dim:
            self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
            return
        blob = _serialize_f32([float(x) for x in embedding])
        existing = self._conn.execute(
            "SELECT 1 FROM memories_vec WHERE rowid = ?", (rowid,)
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE memories_vec SET embedding = ? WHERE rowid = ?",
                (blob, rowid),
            )
        else:
            self._conn.execute(
                "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, blob),
            )

    # ----------------------------------------------------------- superseded

    def mark_superseded(
        self,
        ids: list[str],
        *,
        by: str,
        when: datetime | None = None,
    ) -> None:
        ts = (when or datetime.now(timezone.utc)).isoformat()
        for old_id in ids:
            self._conn.execute(
                "INSERT OR IGNORE INTO tombstones(mem_id, at, reason) VALUES (?, ?, 'superseded')",
                (old_id, ts),
            )
            self._conn.execute(
                "UPDATE memories SET superseded_at = ? WHERE id = ? AND superseded_at IS NULL",
                (ts, old_id),
            )

    # ---------------------------------------------------------------- search

    def search(
        self,
        *,
        query_emb: list[float],
        kinds: list[str] | None,
        candidate_limit: int,
        query_tokens: set[str] | None = None,
    ) -> list[RawRow]:
        """Return candidate rows for the ranker.

        With vec0 available, runs a real KNN against ``memories_vec``
        and joins back to ``memories`` for row data and the active /
        kind filters. Without vec0, returns *all* active rows in the
        requested kind subset (no LIMIT — correct but slow). The naive
        LIMIT-without-ORDER-BY of the v0.0.1 implementation was buggy:
        it biased toward the oldest insertions, so recent writes never
        reached the ranker.
        """
        if self._vec_enabled and self._embedding_dim is not None and len(
            query_emb
        ) == self._embedding_dim:
            rows = self._search_vec(query_emb, kinds, candidate_limit)
            # Hybrid retrieval: a row containing the query's distinctive
            # tokens must reach the ranker even when the embedder puts it
            # outside the KNN neighbourhood (small embedders rank generic
            # attractor memories above literal topic matches).
            if query_tokens:
                seen = {row.id for row in rows}
                rows += [
                    row for row in self._search_lexical(query_tokens, kinds)
                    if row.id not in seen
                ]
            return rows
        return self._search_naive(kinds)

    def _search_lexical(
        self, tokens: set[str], kinds: list[str] | None, limit: int = 200
    ) -> list[RawRow]:
        if not tokens:
            return []
        sql = (
            "SELECT id, kind, text, created_at, last_seen_at, salience, "
            "metadata, embedding FROM memories WHERE superseded_at IS NULL"
        )
        params: list[Any] = []
        likes = []
        for token in sorted(tokens)[:8]:  # cap the OR fan-out
            likes.append("lower(text) LIKE ?")
            params.append(f"%{token}%")
        sql += " AND (" + " OR ".join(likes) + ")"
        if kinds:
            placeholders = ",".join("?" * len(kinds))
            sql += f" AND kind IN ({placeholders})"
            params.extend(kinds)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_from_tuple(r) for r in rows]

    def _search_vec(
        self,
        query_emb: list[float],
        kinds: list[str] | None,
        candidate_limit: int,
    ) -> list[RawRow]:
        # KNN over the vec table, then join with memories for row data
        # and to apply the active + kind filters in SQL. The filters run
        # AFTER the nearest-k cut, so a rare kind (or a store with many
        # inactive rows) can come back short: widen k until we have enough,
        # and past sqlite-vec's cap fall back to scanning the kind itself.
        total = self._conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        k = max(1, int(candidate_limit))
        while True:
            rows = self._knn(query_emb, kinds, min(k, _VEC_MAX_K))
            if len(rows) >= candidate_limit or k >= total:
                return rows
            if k >= _VEC_MAX_K:
                return self._search_naive(kinds) if kinds else rows
            k *= 4

    def _knn(
        self,
        query_emb: list[float],
        kinds: list[str] | None,
        candidate_limit: int,
    ) -> list[RawRow]:
        sql = (
            "SELECT m.id, m.kind, m.text, m.created_at, m.last_seen_at, "
            "m.salience, m.metadata, m.embedding "
            "FROM memories_vec v "
            "JOIN memories m ON m.rowid = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? "
            "AND m.superseded_at IS NULL"
        )
        params: list[Any] = [
            _serialize_f32([float(x) for x in query_emb]),
            int(candidate_limit),
        ]
        if kinds:
            placeholders = ",".join("?" * len(kinds))
            sql += f" AND m.kind IN ({placeholders})"
            params.extend(kinds)
        sql += " ORDER BY v.distance"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_from_tuple(r) for r in rows]

    def _search_naive(self, kinds: list[str] | None) -> list[RawRow]:
        sql = (
            "SELECT id, kind, text, created_at, last_seen_at, salience, "
            "metadata, embedding FROM memories WHERE superseded_at IS NULL"
        )
        params: list[Any] = []
        if kinds:
            placeholders = ",".join("?" * len(kinds))
            sql += f" AND kind IN ({placeholders})"
            params.extend(kinds)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_from_tuple(r) for r in rows]

    @staticmethod
    def _row_from_tuple(r: tuple) -> RawRow:
        return RawRow(
            id=r[0],
            kind=r[1],
            text=r[2],
            created_at=datetime.fromisoformat(r[3]),
            last_seen_at=datetime.fromisoformat(r[4]),
            salience=r[5],
            metadata=json.loads(r[6]),
            embedding=json.loads(r[7]),
        )

    # ------------------------------------------------------ misc maintenance

    def state(self, id: str) -> str | None:
        """'active', 'inactive', or None if this cache has never seen the id."""
        r = self._conn.execute(
            "SELECT superseded_at FROM memories WHERE id = ?", (id,)
        ).fetchone()
        if r is None:
            return None
        return "active" if r[0] is None else "inactive"

    def kind_of(self, id: str) -> str | None:
        r = self._conn.execute("SELECT kind FROM memories WHERE id = ?", (id,)).fetchone()
        return r[0] if r else None

    def find_duplicate(self, kind: str, text: str) -> str | None:
        """Id of an active memory of this kind with the same normalised text."""
        r = self._conn.execute(
            "SELECT id FROM memories WHERE kind = ? AND norm = ? "
            "AND superseded_at IS NULL ORDER BY created_at LIMIT 1",
            (kind, normalise(text)),
        ).fetchone()
        return r[0] if r else None

    def forgotten_ids(self) -> list[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT mem_id FROM tombstones WHERE reason = 'forgotten'"
        )]

    def purge(self, id: str, *, when: datetime | None = None) -> None:
        """Remove a memory's row and vector for good and remember that it
        was purged, so a replayed or late event cannot bring it back."""
        ts = (when or datetime.now(timezone.utc)).isoformat()
        self._conn.execute(
            "INSERT INTO tombstones(mem_id, at, reason) VALUES (?, ?, 'purged') "
            "ON CONFLICT(mem_id) DO UPDATE SET reason = 'purged'",
            (id, ts),
        )
        row = self._conn.execute("SELECT rowid FROM memories WHERE id = ?", (id,)).fetchone()
        if row is None:
            return
        if self._vec_enabled:
            self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (row[0],))
        self._conn.execute("DELETE FROM memories WHERE id = ?", (id,))

    def scrub(self) -> None:
        """Make sure purged text is gone from the file, not just the tables."""
        if self._path != ":memory:":
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def mark_reflected(self, ids: list[str], *, when: datetime | None = None) -> None:
        ts = (when or datetime.now(timezone.utc)).isoformat()
        for mem_id in ids:
            self._conn.execute(
                "UPDATE memories SET reflected_at = ? WHERE id = ? AND reflected_at IS NULL",
                (ts, mem_id),
            )

    def episodes_pending(self, *, limit: int, before: datetime | None = None) -> list[RawRow]:
        """Active episodes consolidation has not read yet, oldest first."""
        sql = (
            "SELECT id, kind, text, created_at, last_seen_at, salience, "
            "metadata, embedding FROM memories "
            "WHERE kind = 'episode' AND superseded_at IS NULL AND reflected_at IS NULL"
        )
        params: list[Any] = []
        if before is not None:
            sql += " AND created_at < ?"
            params.append(before.isoformat())
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        return [self._row_from_tuple(r) for r in self._conn.execute(sql, params).fetchall()]

    def kv_get(self, key: str) -> str | None:
        r = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return r[0] if r else None

    def kv_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def kv_max(self, key: str, n: int) -> None:
        """Raise an integer value to at least `n` (atomic)."""
        self._conn.execute(
            "INSERT INTO kv(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), ?)",
            (key, str(n), n),
        )

    def applied(self) -> dict[str, int] | None:
        """Lines of each device file already applied to this cache, or None
        if the cache has never recorded that (new, or written by v0.11)."""
        if self.kv_get("applied_init") is None:
            return None
        return {
            r[0][len("applied:"):]: int(r[1])
            for r in self._conn.execute("SELECT key, value FROM kv WHERE key LIKE 'applied:%'")
        }

    def set_applied(self, counts: dict[str, int]) -> None:
        self.kv_set("applied_init", "1")
        for name, n in counts.items():
            self.kv_max("applied:" + name, n)

    def tombstone(self, id: str, *, when: datetime | None = None) -> None:
        """Mark a memory inactive (soft-delete). Preserves the row so the
        event log remains the source of truth and replay is deterministic.

        The vec entry is left in place: ``search`` filters via the join on
        ``superseded_at IS NULL`` so a tombstoned row is never returned.
        Leaving the vec row avoids a write per tombstone; ``compact`` can
        prune the vec table later if it grows.
        """
        ts = (when or datetime.now(timezone.utc)).isoformat()
        self._conn.execute(
            "INSERT INTO tombstones(mem_id, at, reason) VALUES (?, ?, 'forgotten') "
            "ON CONFLICT(mem_id) DO UPDATE SET reason = 'forgotten' "
            "WHERE tombstones.reason = 'superseded'",
            (id, ts),
        )
        self._conn.execute(
            "UPDATE memories SET superseded_at = COALESCE(superseded_at, ?), "
            "forgotten_at = COALESCE(forgotten_at, ?) WHERE id = ?",
            (ts, ts, id),
        )

    def wipe(self) -> None:
        """Drop all rows. Used by `Memory.replay()` before rebuilding."""
        self._conn.execute("DELETE FROM memories")
        self._conn.execute("DELETE FROM tombstones")
        self._conn.execute("DELETE FROM kv")
        if self._vec_enabled:
            self._conn.execute("DELETE FROM memories_vec")

    def episodes_since(self, cutoff: datetime, *, limit: int) -> list[RawRow]:
        """Active episodes created at or after `cutoff`, oldest first.

        Used by `reflect()` to find episodes that arrived since the last
        consolidation pass.
        """
        rows = self._conn.execute(
            "SELECT id, kind, text, created_at, last_seen_at, salience, "
            "metadata, embedding FROM memories "
            "WHERE kind = 'episode' AND superseded_at IS NULL "
            "AND created_at >= ? "
            "ORDER BY created_at ASC LIMIT ?",
            (cutoff.isoformat(), limit),
        ).fetchall()
        return [self._row_from_tuple(r) for r in rows]

    def list_active_before(
        self, cutoff: datetime, kinds: list[str] | None = None
    ) -> list[str]:
        """Ids of active (non-superseded) memories created before `cutoff`."""
        sql = "SELECT id FROM memories WHERE created_at < ? AND superseded_at IS NULL"
        params: list[Any] = [cutoff.isoformat()]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        return [r[0] for r in self._conn.execute(sql, params).fetchall()]

    def expire_before(self, cutoff: datetime) -> int:
        """Tombstone all active memories created before `cutoff`. Returns count."""
        ids = self.list_active_before(cutoff)
        ts = datetime.now(timezone.utc).isoformat()
        for mem_id in ids:
            self._conn.execute(
                "UPDATE memories SET superseded_at = ? WHERE id = ?",
                (ts, mem_id),
            )
        return len(ids)

    def compact(self) -> CompactReport:
        return CompactReport(merged=0, dropped=0)

    def inventory(self) -> Inventory:
        rows = self._conn.execute(
            "SELECT kind, COUNT(*) FROM memories WHERE superseded_at IS NULL "
            "GROUP BY kind"
        ).fetchall()
        counts = {r[0]: r[1] for r in rows}

        bytes_on_disk = 0
        if self._path != ":memory:":
            p = Path(self._path)
            if p.exists():
                bytes_on_disk = p.stat().st_size

        return Inventory(
            episodes=counts.get("episode", 0),
            facts=counts.get("fact", 0),
            preferences=counts.get("preference", 0),
            reflections=counts.get("reflection", 0),
            bytes_on_disk=bytes_on_disk,
            last_reflect=None,
        )

    def close(self) -> None:
        self._conn.close()
