# Changelog

All notable changes to Hippocamp are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (within the 0.x line, expect occasional breaking changes between minor versions).

## Unreleased

Fitness work for household use — see docs/fitness-plan.md. Existing stores open unchanged; caches are upgraded in place without re-embedding.

### Recall
- Qwen3 queries now use the model's query instruction (no reindex needed).
- Facts, preferences and reflections no longer fade with age; only episodes do, slightly.
- Durable memories and episodes are fetched separately, and the kind filter widens the search until the list is full.
- Recall no longer writes to the store (the usage/salience bump is gone), so repeated questions give identical answers.
- Every result has `match: "strong" | "weak"`; the tools add a note when nothing is strong.
- Unknown kinds raise; `limit <= 0` returns nothing; near-identical episodes no longer fill the top places.

### Writing and corrections
- Saving the same fact/preference/reflection again returns the existing id.
- `supersedes` works for preferences as well as facts, and is exposed on the `update_memory` tool.
- `update_memory` reports `similar` existing memories and `duplicate`.
- Empty text, unknown ids (forget / supersede) are errors. Preference strength is clamped to 0–1 and nudges ranking.

### Reflect
- Tracks which episodes it has read (`mark_reflected` events) instead of a clock cutoff; processes everything pending in batches.
- The model sees the nearest existing memories with ids and names what an item `replaces`. The 0.85-similarity auto-replace is gone.
- Local adapter: 16,000 output tokens; a reply cut off by the limit splits the batch and retries.
- A lock left by a dead process is taken over. Episodes are shown up to 1,500 characters.

### Forgetting and privacy
- `Memory.purge(id)` and `compact()` erase text from the event log and the cache; `hippocamp forget [--purge]`, `expire`, `compact`.
- Forgets and replacements hold whatever order events arrive in (clock skew, late sync).
- Store and cache files are owner-only (0700 / 0600).
- A cached Qwen3 model loads without contacting the network.

### Robustness
- Unreadable event-log lines are skipped with a warning instead of making the store unopenable.
- Opening a store applies events other devices added since last time.

### Compatibility
- New event ops `mark_reflected` and `purge`: a device still on 0.11 cannot read a log that contains them. Upgrade every device that shares a store.

## [Unreleased]

## [0.11.1] — 2026-05

### Fixed
- `Store.search` did `SELECT … FROM memories LIMIT ?` with no `ORDER BY`, so once a store grew past a few thousand rows the candidate set was the oldest rows by rowid and recent writes were invisible to recall. Candidate retrieval now uses true KNN against a `memories_vec` sqlite-vec virtual table mirrored from `memories`, joined back for row data and the `superseded_at IS NULL` / kind filters.
- `Store.update_embedding` keeps the vec mirror in sync so `Memory.reindex()` rebuilds make rows immediately searchable.

### Added
- One-time backfill on `Store.__init__` populates `memories_vec` from existing rows whose embedding dim matches the configured one. Rows of a different dim (e.g. after switching embedder from BGE-384 to Qwen3-1024) stay in `memories` but are skipped until `Memory.reindex()` rewrites them.
- `sqlite-vec` is a runtime dependency. Extension load is wrapped in try/except: when `sqlite_vec` isn't installed or the Python build's sqlite3 lacks `enable_load_extension` (notably the python.org macOS installer), `Store` falls back to returning all active rows for the ranker — correct, slower than vec0, still better than the buggy `LIMIT`-without-`ORDER BY` path.
- `[bundled-sqlite]` extra (Linux only — no macOS wheels) pulls in `pysqlite3-binary` for users whose stdlib sqlite3 can't load extensions. `store.py` prefers `pysqlite3 as sqlite3` when available.

## [0.11.0] — 2026-05

### Added
- `FastembedBgeEmbedder` — BGE-small-en-v1.5 via [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime). Numerically equivalent to `BgeSmallEmbedder` (same weights), but ~6× smaller install footprint and typically faster per-call on CPU.
- New `[embeddings-lite]` extra installs only `fastembed` (~150 MB) instead of the full sentence-transformers / PyTorch stack (~900 MB). Recommended default for users who only need BGE-small.

### Changed
- `default_embedder()` ordering now: `Qwen3` → `FastembedBge` → `BgeSmall` (sentence-transformers) → `Hash`. The lite path is preferred whenever it's available and Qwen3 isn't.
- README now recommends `'hippocamp[mcp,embeddings-lite]'` for the install snippet. The heavier `[embeddings]` extra remains for users who want Qwen3.

## [0.10.0] — 2026-05

Patch-level release in spirit — bumped to 0.10.0 only because 0.5.3, 0.5.4, 0.5.5, and 0.6.0 were all blocked on PyPI by earlier (deleted) uploads. No new features or breaking changes.

### Fixed
- `hippocamp --version` reported `0.3.4` regardless of installed version. `__version__` now reads from package metadata (`importlib.metadata.version`), so `pyproject.toml` is the single source of truth.
- `hippocamp-mcp --help` previously loaded the BGE embedder before printing help. Now uses `argparse` so `--help` / `--version` short-circuit before any heavy initialisation. Adds `--version` to the MCP entry point.

## [0.5.2] — 2026-05

### Changed
- Remove personal-data example strings from the directive (`instructions.py`) and the `update_memory` MCP tool description (`mcp_server.py`). Example facts now use neutral placeholders (`"I work at Acme Corp"`) instead of family-specific examples. The `push_via_rsync` docstring example now uses `user@desktop.local`.

## [0.5.1] — 2026-05

### Added
- GitHub Actions CI: pytest matrix on Python 3.11/3.12/3.13.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates.

## [0.5.0] — 2026-05

### Added
- `Memory.reflect()` — LLM-driven memory consolidation. Reads recent episodes, distils durable facts/preferences/reflections, supersedes near-duplicates. Persists `last_reflect_at` only on success so failed runs are retried.
- `reflect_memory` MCP tool, auto-exposed when the server boots with `ANTHROPIC_API_KEY` set.
- `hippocamp reflect [--since <iso-ts>]` CLI command.
- New memory kind: **reflection** (higher-order patterns spanning multiple episodes).
- Bundled Anthropic LLM adapter; any object with `complete(prompt, *, system) -> str` also works via `Memory(llm=...)`.
- Reflection lockfile (`.reflect.lock`) prevents concurrent runs across processes.

## [0.4.0]

### Changed
- Default embedder swapped to Qwen3 for better retrieval quality.
- Salience contribution in the ranker is now capped to avoid runaway boosts.

### Added
- `scripts/` — synthetic-data generation and embedder-quality eval harness.

## [0.3.4]

### Added
- `hippocamp instructions` and `hippocamp instructions --short` to print the directive for Claude Desktop / ChatGPT custom-instruction setup.

## [0.3.3]

### Changed
- `hippocamp setup claude` now configures Claude Code **and** Claude Desktop together when both are installed.

## [0.3.2]

### Added
- Proactive ambient memory in the directive: AI captures signals as they flow through conversation, no need to say "remember".

## [0.3.1]

### Added
- `hippocamp setup claude` auto-installs the Hippocamp directive block into `~/.claude/CLAUDE.md`, wrapped in idempotent markers.

## [0.3.0]

### Added
- **Event-log architecture** — every write is appended to a per-device file `events/<device_id>.jsonl`. Conflict-free across machines, since no two devices ever touch the same file.
- Cloud-synced store dir separated from local SQLite cache. Wipe the cache anytime; rebuild from events.
- `hippocamp sync` — one smart command that takes a local path, an SSH peer, or nothing (just rebuild).
- `hippocamp reindex [--embedder NAME]` for recomputing embeddings after a model swap.
- `hippocamp status` shows what's stored, where, by which devices.

## [0.1.0]

### Added
- Initial alpha. Tiered memory (episodes / facts / preferences), why-trace recall, first-class forgetting (supersedence + TTL + tombstones), SQLite backend, MCP server with `recall_memory` and `update_memory`, `hippocamp` CLI with `setup`/`inspect`.

[Unreleased]: https://github.com/Prageethcs/hippocamp/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.11.0
[0.10.0]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.10.0
[0.5.2]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.5.2
[0.5.1]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.5.1
[0.5.0]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.5.0
[0.4.0]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.4.0
[0.3.4]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.3.4
[0.3.3]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.3.3
[0.3.2]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.3.2
[0.3.1]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.3.1
[0.3.0]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.3.0
[0.1.0]: https://github.com/Prageethcs/hippocamp/releases/tag/v0.1.0
