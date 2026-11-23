"""Per-device append-only event logs — the source of truth for Hippocamp.

Each device writes only to its own file: `events/<device_id>.jsonl`. Reads
merge events from every `*.jsonl` file in `events/`, sorted by timestamp.

This eliminates concurrent-write conflicts on cloud-synced storage
(Dropbox, iCloud, Syncthing, etc.) because no file ever has more than one
writer. Pulling another device's events file in via any sync mechanism is
a no-op for everyone else's writes.

Events carry text only. Embeddings are derived locally by each device's
embedder and never written to the log.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field


log = logging.getLogger(__name__)


class EventOp(str, Enum):
    OBSERVE = "observe"
    ASSERT_FACT = "assert_fact"
    ASSERT_PREF = "assert_pref"
    ASSERT_REFLECTION = "assert_reflection"
    FORGET = "forget"
    # Added after v0.11. A reader older than that rejects these lines.
    MARK_REFLECTED = "mark_reflected"   # `sources` = episode ids consolidation has read
    PURGE = "purge"                     # `mem_id` must be removed from every copy


class Event(BaseModel):
    """A single mutation, recorded immutably in this device's event file."""

    id: str
    ts: datetime
    device: str
    op: EventOp
    mem_id: str
    text: str | None = None
    actors: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    strength: float | None = None
    supersedes: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


def new_event_id() -> str:
    return f"ev_{uuid.uuid4().hex[:16]}"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class EventLog:
    """Per-device writer + multi-file reader for events.

    `append()` writes only to this device's file (`<device_id>.jsonl`).
    `read_all()` reads every `*.jsonl` in the events dir, ordered by ts.
    """

    def __init__(self, events_dir: Path, device_id: str) -> None:
        self._dir = Path(events_dir)
        self._device_id = device_id
        self._own_file = self._dir / f"{device_id}.jsonl"

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def own_file(self) -> Path:
        return self._own_file

    @property
    def device_id(self) -> str:
        return self._device_id

    def append(self, event: Event) -> int:
        """Append to *this* device's file only. Returns the file's end
        offset after the write."""
        if not self._dir.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self._dir, 0o700)
        line = event.model_dump_json() + "\n"
        # A crash can leave a half-written last line. Start on a fresh line
        # so the damage stays confined to that one event.
        try:
            with self._own_file.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    line = "\n" + line
        except (FileNotFoundError, OSError):
            pass
        fd = os.open(self._own_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            return f.tell()

    def secure(self) -> None:
        """Owner-only permissions on the log directory and every file in it."""
        if not self._dir.exists():
            return
        os.chmod(self._dir, 0o700)
        for p in self.device_files():
            os.chmod(p, 0o600)

    @staticmethod
    def _parse(path: Path, offset: int = 0) -> Iterator[Event]:
        """Every readable event from byte `offset` on. A line that does not
        parse — a half-written tail after a power cut, an op from a newer
        version — is skipped with a warning instead of taking the store down."""
        with path.open("rb") as f:
            f.seek(offset)
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    yield Event.model_validate_json(line)
                except ValueError:
                    log.warning("hippocamp: skipped an unreadable event in %s", path.name)

    def sizes(self) -> dict[str, int]:
        """Byte size per device file. Files only ever grow (a purge blanks
        text in place, keeping every line's length), so "applied up to byte
        N" stays meaningful and a smaller file means it was replaced."""
        return {p.name: p.stat().st_size for p in self.device_files()}

    def read_from(self, start: dict[str, int]) -> list[Event]:
        """Events after byte `start[file]` in each file, ordered by (ts, id)."""
        events = [
            e for path in self.device_files()
            for e in self._parse(path, start.get(path.name, 0))
        ]
        events.sort(key=lambda e: (e.ts, e.id))
        return events

    def redact(self, mem_id: str) -> int:
        """Blank the content of every event about `mem_id`, in every device
        file held here. Each line keeps its exact byte length (padded with
        spaces), so offsets into the file stay valid, but it carries no
        text. Returns how many events were blanked."""
        n = 0
        for path in self.device_files():
            out, changed = [], False
            with path.open("rb") as f:
                for line in f:
                    body = line[:-1] if line.endswith(b"\n") else line
                    try:
                        ev = Event.model_validate_json(body) if body.strip() else None
                    except ValueError:
                        ev = None
                    if ev is not None and ev.mem_id == mem_id and ev.op not in (
                        EventOp.FORGET, EventOp.PURGE
                    ) and (ev.text or ev.actors or ev.metadata != {"redacted": True}):
                        ev.text, ev.metadata, ev.actors = None, {"redacted": True}, []
                        blank = ev.model_dump_json().encode("utf-8")
                        if len(blank) <= len(body):
                            body = blank + b" " * (len(body) - len(blank))
                        else:  # nothing worth hiding fits in less space: minimal stub
                            body = b" " * len(body)
                        line = body + (b"\n" if line.endswith(b"\n") else b"")
                        changed = True
                        n += 1
                    out.append(line)
            if changed:
                tmp = path.with_suffix(".jsonl.tmp")
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as f:
                    f.writelines(out)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
        return n

    def device_files(self) -> list[Path]:
        if not self._dir.exists():
            return []
        return sorted(self._dir.glob("*.jsonl"))

    def read_all(self) -> Iterator[Event]:
        """Yield every event across every device file, ordered by (ts, id).

        Sorting by timestamp gives a deterministic global order. Slight
        clock skew between devices is tolerated — for our access patterns
        (idempotent inserts, supersede-by-id, tombstones), small reorders
        don't change the eventual state.
        """
        events = [e for path in self.device_files() for e in self._parse(path)]
        events.sort(key=lambda e: (e.ts, e.id))
        for e in events:
            yield e

    def count(self) -> int:
        n = 0
        for path in self.device_files():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
        return n

    def count_own(self) -> int:
        if not self._own_file.exists():
            return 0
        with self._own_file.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
