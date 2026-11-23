"""LLM-driven memory consolidation.

`run_reflection()` reads recent episodes from a Memory, asks an LLM to
extract durable facts/preferences/reflections, and writes the output
back to the Memory. Near-duplicate facts are superseded via a
recall+similarity threshold. Memory's `last_reflect_at` is persisted on
success so subsequent runs only process new episodes.

Hippocamp does not pick an LLM — the caller supplies any object with a
`complete(prompt, *, system) -> str` method via `Memory(llm=...)`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from hippocamp.memory import Memory
    from hippocamp.types import RawRow


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class LLM(Protocol):
    """Minimal interface any LLM client must satisfy to power reflection.

    Adapters for Anthropic / OpenAI / Gemini / local models live outside
    Hippocamp; Hippocamp does not pick a model.
    """

    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


class ReplyTruncated(RuntimeError):
    """The model ran out of output tokens before finishing its answer."""


class _Item(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list)
    # Ids of EXISTING memories this item updates or contradicts.
    replaces: list[str] = Field(default_factory=list)


class ReflectionDraft(BaseModel):
    """Schema the LLM must return — validated before any writes."""

    facts: list[_Item] = Field(default_factory=list)
    preferences: list[_Item] = Field(default_factory=list)
    reflections: list[_Item] = Field(default_factory=list)


class ReflectionReport(BaseModel):
    """Summary of a reflection run, returned to the caller."""

    new_facts: int = 0
    new_preferences: int = 0
    new_reflections: int = 0
    superseded: int = 0
    episodes_processed: int = 0
    batches: int = 0
    pending: int = 0        # episodes still waiting after this run
    notes: str = ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are a memory distiller for a personal AI assistant. You receive a list
of journal episodes (what people in a household told the assistant, or did
with it) and extract durable content: stable facts, preferences, and
patterns.

OUTPUT a single JSON object with three keys:
  - facts:        list of stable facts about a person, the household, a project
  - preferences:  list of likes, dislikes, or working styles
  - reflections:  list of higher-order patterns spanning multiple episodes

Each item has the shape:
  {"text": "<the distilled statement>", "evidence": ["<episode_id>", ...],
   "replaces": ["<existing memory id>", ...]}

RULES:
- Extract only items GROUNDED in the episode content. Do not infer beyond
  what is stated in the episodes.
- Each item MUST list at least one episode id from the input as evidence.
- Say WHO each item is about, by name when the episode gives one ("Theo is
  allergic to peanuts"). Use "the user" only when no name is given. Never
  merge statements about different people into one.
- Include plans and bookings with their dates ("holiday in Bergen, 12-19
  July"), deadlines and decisions; they stay useful until they pass.
- Most episodes are routine and yield NOTHING. A question asked, a lookup,
  a conversion, a timer, a recipe, a draft, a reminder, a bug fixed: these
  are things that happened, not facts about the person. Do NOT turn them
  into "is interested in X", "asked about X", "is working on X" or "needs
  to do X". Returning empty lists is the correct answer for such a batch.
- Write a fact or preference only when an episode STATES something lasting
  about a person, the household or a project: who someone is, what they
  have, where, when, what they like, what was decided or booked.
- A recurring theme across three or more episodes may go in "reflections".
- Ignore jokes and hypotheticals.
- EXISTING_MEMORIES lists what is already stored, each with its id.
  * If an item says the same as an existing memory, SKIP it.
  * If an item UPDATES or CONTRADICTS an existing memory about the same
    person and subject (a new employer, a changed preference, a moved
    appointment), write the new item and put the old memory's id in
    "replaces". Only use ids that appear in EXISTING_MEMORIES.
  * A similar statement about a DIFFERENT person or subject replaces
    nothing.
- When two episodes in the input disagree, the later one wins.
- Output ONLY the JSON object. No prose, no markdown fences, no commentary.
"""


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------


class _ReflectLock:
    """File lock at <store_dir>/.reflect.lock.

    Holds "<pid> <iso time>". A lock whose process is gone (a crashed or
    killed run) is taken over at once; one older than STALE_AFTER is taken
    over whatever its process is doing.
    """

    STALE_AFTER = timedelta(hours=1)

    def __init__(self, store_dir: Path | None) -> None:
        self._path = (store_dir / ".reflect.lock") if store_dir else None

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True
        return True

    def acquire(self) -> None:
        if self._path is None:
            return  # in-memory store: no lock needed
        if self._path.exists():
            mtime = datetime.fromtimestamp(
                self._path.stat().st_mtime, tz=timezone.utc
            )
            age = datetime.now(timezone.utc) - mtime
            try:
                pid = int(self._path.read_text().split()[0])
            except (ValueError, IndexError, OSError):
                pid = None
            holder_gone = pid is not None and not self._alive(pid)
            if age < self.STALE_AFTER and not holder_gone:
                raise RuntimeError(
                    f"reflect() locked by another process (lock at {self._path}, "
                    f"age {age}). Wait, or delete the lock if certain it's stale."
                )
        self._path.write_text(
            f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n"
        )
        os.chmod(self._path, 0o600)

    def release(self) -> None:
        if self._path is None:
            return
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

EPISODE_CHARS = 1500      # how much of one episode the model is shown
EXISTING_PER_EPISODE = 3  # nearest stored facts/preferences fetched per episode
EXISTING_CAP = 40         # ... and at most this many per batch


def run_reflection(
    memory: "Memory",
    llm: LLM,
    *,
    since: str | datetime | None = None,
    batch_size: int = 15,
    max_batches: int = 20,
    min_episodes: int = 5,
    max_wait: timedelta = timedelta(hours=24),
    max_episodes: int | None = None,
    dedup_threshold: float | None = None,   # accepted for old callers; unused
    default_lookback: timedelta | None = None,  # accepted for old callers; unused
) -> ReflectionReport:
    """Distil episodes into facts/preferences/reflections.

    Works through every episode consolidation has not read yet, oldest
    first, in batches of `batch_size`. Each batch is one model call; when it
    succeeds its episodes are marked as read (a `mark_reflected` event), so
    nothing is read twice and nothing is skipped — however old an episode
    is, however late it arrived from another device, and however many there
    are. A failed call leaves its batch unread for the next run.

    Fewer than `min_episodes` pending are left to accumulate, unless the
    oldest has already waited `max_wait`.

    `since` re-reads everything from that time on regardless of marks
    (repair / re-run). Exact repeats are not written twice.

    For each batch the model is shown the stored facts and preferences
    nearest to its episodes, with their ids. A new item replaces an old one
    only when the model names that id — never on resemblance alone.
    """
    if max_episodes:
        batch_size = max_episodes
    store = memory._store
    lock = _ReflectLock(memory.store_dir)
    lock.acquire()
    try:
        cap = batch_size * max_batches
        if since is not None:
            cutoff = since if isinstance(since, datetime) else datetime.fromisoformat(since)
            episodes = store.episodes_since(cutoff, limit=cap)
            waited = False
        else:
            _adopt_legacy_cutoff(memory)
            episodes = store.episodes_pending(limit=cap)
            waited = bool(episodes) and (
                datetime.now(timezone.utc) - episodes[0].created_at > max_wait
            )
        if len(episodes) < min_episodes and not waited:
            return ReflectionReport(
                episodes_processed=0,
                pending=len(episodes),
                notes=(
                    f"Skipped: {len(episodes)} episodes waiting is below "
                    f"min_episodes={min_episodes}."
                ),
            )

        report = ReflectionReport()
        queue = [episodes[i:i + batch_size] for i in range(0, len(episodes), batch_size)]
        while queue:
            batch = queue.pop(0)
            try:
                _reflect_batch(memory, llm, batch, report)
            except ReplyTruncated:
                # A thinking model can burn its whole output budget on a
                # large batch. Halve it and go again rather than fail the run.
                if len(batch) < 2:
                    raise
                mid = len(batch) // 2
                queue[:0] = [batch[:mid], batch[mid:]]
                continue
            memory.mark_reflected([e.id for e in batch])
            report.episodes_processed += len(batch)
            report.batches += 1
        if since is None:
            report.pending = len(store.episodes_pending(limit=cap))
        memory.set_last_reflect_at(datetime.now(timezone.utc))
        return report
    finally:
        lock.release()


def _adopt_legacy_cutoff(memory: "Memory") -> None:
    """One-time bridge from v0.11, which only kept `last_reflect_at`: treat
    episodes older than that as already read, so an upgraded store does not
    re-read its whole history. After this, only marks count."""
    meta = memory.meta
    if meta is None or meta.reflect_tracking:
        return
    if meta.last_reflect_at is not None:
        done = memory._store.episodes_pending(limit=1_000_000, before=meta.last_reflect_at)
        ids = [e.id for e in done]
        for i in range(0, len(ids), 500):
            memory.mark_reflected(ids[i:i + 500])
    memory.set_reflect_tracking()


def _reflect_batch(memory: "Memory", llm: LLM, batch: list, report: ReflectionReport) -> None:
    existing = _relevant_existing(memory, batch)
    prompt = (
        f"EPISODES:\n{_format_episodes(batch)}\n\n"
        f"EXISTING_MEMORIES:\n{_format_existing_facts(existing)}\n"
    )
    draft = _call_and_parse(llm, prompt, retries=1)

    valid_ids = {e.id for e in batch}
    shown = {m.id: m.kind for m in existing}
    store = memory._store

    def replaced(item: _Item, kind: str) -> list[str]:
        # Only ids the model was actually shown, of the same kind, still active.
        return [i for i in dict.fromkeys(item.replaces)
                if shown.get(i) == kind and store.state(i) == "active"]

    for f in draft.facts:
        evidence = [eid for eid in f.evidence if eid in valid_ids]
        if not evidence or not f.text.strip():
            continue
        sup = replaced(f, "fact")
        is_new = store.find_duplicate("fact", f.text) is None
        memory.assert_fact(f.text, evidence=evidence, supersedes=sup or None)
        report.new_facts += is_new
        report.superseded += len(sup)

    for p in draft.preferences:
        evidence = [eid for eid in p.evidence if eid in valid_ids]
        if not evidence or not p.text.strip():
            continue
        sup = replaced(p, "preference")
        is_new = store.find_duplicate("preference", p.text) is None
        memory.assert_preference(p.text, supersedes=sup or None)
        report.new_preferences += is_new
        report.superseded += len(sup)

    for r in draft.reflections:
        sources = [eid for eid in r.evidence if eid in valid_ids]
        if not sources or not r.text.strip():
            continue
        is_new = store.find_duplicate("reflection", r.text) is None
        memory.assert_reflection(r.text, sources=sources)
        report.new_reflections += is_new


def _relevant_existing(memory: "Memory", batch: list) -> list:
    """Stored facts and preferences closest to this batch's episodes — what
    the model needs in front of it to spot a repeat or an update."""
    out: dict[str, object] = {}
    for e in batch:
        rows = memory._store.search(
            query_emb=e.embedding, kinds=["fact", "preference"],
            candidate_limit=EXISTING_PER_EPISODE,
        )
        for r in rows[:EXISTING_PER_EPISODE]:
            out.setdefault(r.id, r)
        if len(out) >= EXISTING_CAP:
            break
    return list(out.values())[:EXISTING_CAP]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_episodes(episodes: Iterable["RawRow"]) -> str:
    lines = []
    for e in episodes:
        date = e.created_at.date().isoformat()
        snippet = e.text.replace("\n", " ")[:EPISODE_CHARS]
        lines.append(f"[{e.id}] {date} {snippet}")
    return "\n".join(lines)


def _format_existing_facts(facts) -> str:
    if not facts:
        return "(none)"
    return "\n".join(f"[{f.id}] ({f.kind}) {f.text}" for f in facts)


def _extract_json(text: str) -> str:
    """Find the first {...} block in `text` (defends against prose around JSON)."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("no JSON object found in LLM output")
    return m.group(0)


def _complete(llm: LLM, prompt: str) -> str:
    """Call the model, handing it the reply schema when the adapter can use
    one (constrained output: valid JSON by construction, and a thinking
    model answers directly instead of deliberating for minutes)."""
    if getattr(llm, "accepts_schema", False):
        return llm.complete(prompt, system=SYSTEM_PROMPT, schema=DRAFT_SCHEMA)  # type: ignore[call-arg]
    return llm.complete(prompt, system=SYSTEM_PROMPT)


_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "replaces": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "evidence", "replaces"],
}
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "array", "items": _ITEM_SCHEMA}
                   for k in ("facts", "preferences", "reflections")},
    "required": ["facts", "preferences", "reflections"],
}


def _call_and_parse(llm: LLM, prompt: str, *, retries: int = 1) -> ReflectionDraft:
    """Call LLM and parse its JSON output. Retry once on failure with a stricter nudge."""
    last_err: Exception | None = None
    current_prompt = prompt
    for attempt in range(retries + 1):
        raw = _complete(llm, current_prompt)
        try:
            blob = _extract_json(raw)
            data = json.loads(blob)
            return ReflectionDraft.model_validate(data)
        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            last_err = e
            if attempt >= retries:
                raise RuntimeError(
                    f"LLM did not return valid JSON after {retries + 1} attempts: {last_err}"
                ) from last_err
            current_prompt = current_prompt + (
                "\n\n[The previous response was not valid JSON. Output ONLY a "
                "single JSON object matching the schema; no prose, no fences.]"
            )
    raise RuntimeError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Bundled local adapter (the default)
# ---------------------------------------------------------------------------

# Reflection reads private memory. By default that text stays on this
# machine: the only adapter the CLI and MCP server pick on their own is the
# local one, and it refuses any server that isn't on loopback. The cloud
# adapters below run only when asked for by name AND this is set to 1.
CLOUD_OPT_IN_ENV = "HIPPOCAMP_ALLOW_CLOUD_REFLECT"
DEFAULT_LOCAL_URL = "http://127.0.0.1:1234"  # LM Studio's server


def cloud_reflect_allowed() -> bool:
    return os.environ.get(CLOUD_OPT_IN_ENV, "").strip() == "1"


def _require_cloud_opt_in(adapter: str) -> None:
    if not cloud_reflect_allowed():
        raise RuntimeError(
            f"the {adapter} adapter would send memory text to a cloud model; "
            f"refusing. Reflection runs on a local model by default. To allow "
            f"this on purpose, set {CLOUD_OPT_IN_ENV}=1."
        )


def _is_loopback(url: str) -> bool:
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def local_llm(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int = 900,
    max_tokens: int = 16000,
    structured: bool | None = None,
    reasoning_effort: str | None = None,
) -> LLM:
    """Build an LLM that calls an OpenAI-compatible server on this machine
    (LM Studio, llama.cpp server, Ollama's /v1, mlx_lm.server).

    ``base_url`` defaults to HIPPOCAMP_LLM_URL, else http://127.0.0.1:1234.
    It must be a loopback address — anything else raises RuntimeError, so a
    mistyped or hostile setting can't turn "local" into a remote server.
    ``model`` defaults to HIPPOCAMP_LLM_MODEL, else the first non-embedding
    model the server lists. The server is first contacted on ``complete()``,
    so building the adapter works while the server is down.
    """
    import urllib.error
    import urllib.request

    url = (base_url or os.environ.get("HIPPOCAMP_LLM_URL") or DEFAULT_LOCAL_URL).rstrip("/")
    if not _is_loopback(url):
        raise RuntimeError(
            f"local reflect LLM must be on this machine (loopback); got {url!r}"
        )
    wanted = model or os.environ.get("HIPPOCAMP_LLM_MODEL") or None
    if structured is None:
        # On by default. HIPPOCAMP_LLM_STRUCTURED=0 for a server that
        # rejects response_format.
        structured = os.environ.get("HIPPOCAMP_LLM_STRUCTURED", "1").strip() != "0"
    # Distilling is extraction, not problem solving. Measured with a
    # thinking model (Gemma 4): left to think, a batch takes 1-3 minutes
    # and can spend its whole output budget before answering; with thinking
    # off and a schema it takes seconds and passes the same checks
    # (evals/reflect_live.py). Set HIPPOCAMP_LLM_REASONING=default to leave
    # the model's own setting alone, or to low/medium/high.
    effort = reasoning_effort or os.environ.get("HIPPOCAMP_LLM_REASONING", "none").strip()

    def _call(path: str, payload: dict | None = None) -> dict:
        req = urllib.request.Request(
            url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RuntimeError(
                f"local LLM server at {url} did not answer ({e}); start it "
                "(e.g. `lms server start`) and try again"
            ) from e

    class _LocalLLM:
        base_url = url
        accepts_schema = structured

        def _model(self) -> str:
            nonlocal wanted
            if wanted is None:
                ids = [m.get("id", "") for m in _call("/v1/models").get("data", [])]
                chat = [i for i in ids if i and "embed" not in i.lower()]
                if not chat:
                    raise RuntimeError(f"local LLM server at {url} lists no chat model")
                wanted = chat[0]
            return wanted

        def complete(
            self, prompt: str, *, system: str | None = None, schema: dict | None = None
        ) -> str:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": self._model(),
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            if effort and effort != "default":
                payload["reasoning_effort"] = effort
            if schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "schema": schema},
                }
            data = _call("/v1/chat/completions", payload)
            try:
                choice = data["choices"][0]
                content = choice["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"unexpected reply from local LLM server: {e}") from e
            if choice.get("finish_reason") == "length":
                # A thinking model can spend the whole budget before it
                # starts the answer. Say so instead of returning half a reply.
                raise ReplyTruncated(
                    f"local LLM reply was cut off at max_tokens={max_tokens}"
                )
            return content

    return _LocalLLM()


# ---------------------------------------------------------------------------
# Bundled Anthropic adapter (cloud — opt-in only)
# ---------------------------------------------------------------------------


def default_anthropic_llm(
    *,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 4096,
) -> LLM:
    """Build an LLM that calls Anthropic's Messages API.

    Requires `pip install 'hippocamp[llm]'` and ANTHROPIC_API_KEY in
    the environment. Raises RuntimeError if either is missing.

    Hippocamp does not require this — any object with a `.complete(...)`
    method can be passed via `Memory(llm=...)`.

    Sends memory text to Anthropic, so it refuses to run unless
    HIPPOCAMP_ALLOW_CLOUD_REFLECT=1 is set.
    """
    _require_cloud_opt_in("Anthropic API")
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "hippocamp[llm] not installed. Run: pip install 'hippocamp[llm]'"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=api_key)

    class _AnthropicLLM:
        def complete(self, prompt: str, *, system: str | None = None) -> str:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in msg.content if hasattr(block, "text")
            )

    return _AnthropicLLM()


# ---------------------------------------------------------------------------
# Bundled Claude Code CLI adapter (cloud — opt-in only)
# ---------------------------------------------------------------------------

# Fallback locations for hosts that spawn the server with a minimal PATH
# (e.g. Claude Desktop launched from Finder on macOS).
_CLAUDE_CLI_CANDIDATES = (
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
    str(Path.home() / ".local" / "bin" / "claude"),
)


def claude_cli_llm(
    *,
    model: str = "haiku",
    timeout: int = 900,
) -> LLM:
    """Build an LLM that shells out to the Claude Code CLI (`claude -p`).

    Authenticates via the local Claude Code login — no ANTHROPIC_API_KEY
    required. Raises RuntimeError if the `claude` binary can't be found.

    Sends memory text to Anthropic, so it refuses to run unless
    HIPPOCAMP_ALLOW_CLOUD_REFLECT=1 is set.
    """
    _require_cloud_opt_in("Claude Code CLI")
    binary = shutil.which("claude")
    if binary is None:
        binary = next(
            (c for c in _CLAUDE_CLI_CANDIDATES if os.path.exists(c)), None
        )
    if binary is None:
        raise RuntimeError(
            "claude CLI not found on PATH — install Claude Code or set "
            "ANTHROPIC_API_KEY to use the API adapter instead"
        )

    class _ClaudeCLILLM:
        def complete(self, prompt: str, *, system: str | None = None) -> str:
            cmd = [binary, "-p", "--model", model]
            if system:
                cmd += ["--system-prompt", system]
            result = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"claude CLI failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            return result.stdout

    return _ClaudeCLILLM()
