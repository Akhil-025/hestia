"""
core/observability.py

Cross-cutting observability for a single Hestia process. Four concerns,
deliberately kept in one small module because they all answer the same
question — "what did Hestia just do, and why?":

1. Request IDs (backlog #17)
   A short id generated once per query in ``Hestia.process_text`` and
   attached to a ``contextvars.ContextVar``. Every log record emitted while
   that query is in flight carries it (see ``RequestIdFilter``), so one
   query's full trace across ``core/``, ``modules/`` and ``api.py`` can be
   grepped in one shot:  ``grep 'req=a1b2c3d4' hestia.log``.

2. Routing log (backlog #5)
   Every intent classification + routing decision appended as one JSON
   object per line to a size-rotating file (default ``logs/routing.jsonl``).
   JSONL rather than free text because the point of keeping it is later
   analysis — confusion matrices, low-confidence review, per-intent
   accuracy — all of which want to be read by a script, not a human.

3. Decision ring buffer (backlog #3)
   The last N routing records held in memory so a "why did you route this
   there?" query can be answered conversationally without re-reading the
   log file. Bounded, so it can never grow without limit.

4. Feedback log (backlog #259)
   An explicit "that was wrong" record, written to its own JSONL file so a
   user correction is a labelled data point rather than a complaint that
   vanishes into the chat history.

None of this imports any Hestia module, so it's safe to import from
``core/``, ``modules/`` and ``main.py`` alike without creating a cycle.
Every public function is failure-tolerant: observability must never be the
reason a query fails, so file-write errors are logged at debug level and
swallowed.
"""
from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Optional

logger = logging.getLogger(__name__)

# Where the JSONL logs live, relative to the repo root. Overridable via
# env var so tests (and anyone running from a read-only checkout) can
# redirect them without editing config.
_DEFAULT_LOG_DIR = Path(os.environ.get("HESTIA_LOG_DIR", "logs"))

# Rotation: 5 MB per file, 3 backups. At ~250 bytes/record that's roughly
# 20k classifications per file — months of personal use, and small enough
# that a script can load a whole file into memory for analysis.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# How many recent decisions to keep in memory for "why did you route this".
_RING_SIZE = 50


# ---------------------------------------------------------------------------
# 1. Request IDs
# ---------------------------------------------------------------------------

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "hestia_request_id", default="-"
)


def new_request_id() -> str:
    """Generate, set as current, and return a fresh short request id."""
    rid = uuid.uuid4().hex[:8]
    _request_id.set(rid)
    return rid


def current_request_id() -> str:
    """Return the in-flight request id, or ``"-"`` outside any request."""
    return _request_id.get()


def set_request_id(rid: str) -> None:
    """Adopt an externally supplied request id (e.g. an HTTP header)."""
    _request_id.set(rid or "-")


class RequestIdFilter(logging.Filter):
    """
    Injects ``%(request_id)s`` into every record.

    A ``Filter`` rather than a custom ``Formatter`` so it composes with
    whatever format string ``main._configure_logging`` uses, and so records
    from third-party libraries (which never set the attribute themselves)
    don't blow up formatting with a ``KeyError``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id()
        return True


# ---------------------------------------------------------------------------
# JSONL writer (shared by the routing and feedback logs)
# ---------------------------------------------------------------------------

class _JsonlLog:
    """
    Append-only JSONL file with size-based rotation.

    Built on ``RotatingFileHandler`` rather than hand-rolled file writes so
    rotation, encoding and locking come from the stdlib. The handler is
    attached to a private, non-propagating logger so these records never
    leak into Hestia's console output.
    """

    def __init__(self, filename: str, logger_name: str) -> None:
        self._log = logging.getLogger(logger_name)
        self._log.propagate = False
        self._log.setLevel(logging.INFO)
        self._enabled = False
        self._lock = threading.Lock()

        # Always rebuild the handler rather than keeping whatever is
        # already attached. The logger is a process-global singleton, so a
        # handler left over from an earlier construction points at the log
        # directory that was configured *then* — which is silently the
        # wrong file after a config change or a module reload, and was
        # exactly that in the tests. Closing and replacing is idempotent
        # and always writes where the current config says.
        for existing in list(self._log.handlers):
            try:
                existing.close()
            except Exception:  # pragma: no cover - handler-specific
                pass
            self._log.removeHandler(existing)

        try:
            _DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                _DEFAULT_LOG_DIR / filename,
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(handler)
            self._enabled = True
        except OSError as exc:
            # Read-only checkout, permissions, full disk — none of which
            # should stop Hestia answering questions.
            logger.debug("Could not open %s for writing: %s", filename, exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def write(self, record: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            logger.debug("Unserialisable observability record; skipping.")
            return
        with self._lock:
            try:
                self._log.info(line)
            except Exception:  # pragma: no cover - handler-level failure
                logger.debug("Observability write failed; continuing.")


# ---------------------------------------------------------------------------
# 2 + 3 + 4. Diagnostics
# ---------------------------------------------------------------------------

class Diagnostics:
    """
    Single place Hestia records what it decided, and the single place the
    ``explain_routing`` / ``modules_status`` / ``report_mistake`` intents
    read from.

    ``main.Hestia`` constructs exactly one of these and injects it into
    ``CoreModule`` (for the three diagnostic intents) and into
    ``api.create_app`` (for ``/health/modules``). Nothing else needs it.
    """

    def __init__(self, orchestrator: Any = None) -> None:
        self._orchestrator = orchestrator
        self._ring: Deque[dict[str, Any]] = deque(maxlen=_RING_SIZE)
        self._lock = threading.Lock()
        self._routing_log = _JsonlLog("routing.jsonl", "hestia.routing")
        self._feedback_log = _JsonlLog("feedback.jsonl", "hestia.feedback")

    # -- wiring --------------------------------------------------------

    def bind_orchestrator(self, orchestrator: Any) -> None:
        """
        Attach the orchestrator after construction.

        Needed because ``Diagnostics`` is created before the orchestrator
        exists (the routing log has to be ready for the very first query),
        while ``modules_status`` needs the orchestrator to enumerate
        modules.
        """
        self._orchestrator = orchestrator

    # -- 2. routing log ------------------------------------------------

    def record_classification(
        self,
        *,
        query: str,
        intent: str,
        confidence: float,
        module: str,
        reason: str = "",
        latency_ms: float = 0.0,
        source: str = "nlu",
    ) -> dict[str, Any]:
        """
        Log one classification+routing decision and return the record.

        Called once per query from ``Hestia.process_text``. ``source``
        distinguishes a real LLM classification from a cache hit, an alias
        match, or a dry run, so analysis scripts can exclude the cheap
        paths when measuring the model itself.
        """
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request_id": current_request_id(),
            "query": (query or "")[:500],
            "intent": intent,
            "confidence": round(float(confidence or 0.0), 3),
            "module": module,
            "reason": reason,
            "latency_ms": round(float(latency_ms or 0.0), 1),
            "source": source,
        }
        with self._lock:
            self._ring.append(record)
        self._routing_log.write(record)
        return record

    # -- 3. "why did you route this there?" ----------------------------

    def last_decision(self) -> Optional[dict[str, Any]]:
        """Most recent routing record, or None if nothing routed yet."""
        with self._lock:
            return dict(self._ring[-1]) if self._ring else None

    def recent_decisions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Up to *limit* most recent routing records, newest last."""
        with self._lock:
            items = list(self._ring)
        if limit > 0:
            items = items[-limit:]
        return [dict(i) for i in items]

    def explain_last(self) -> str:
        """
        Human-readable explanation of the previous routing decision.

        Deliberately returns prose rather than a dict: this is what the
        ``explain_routing`` intent speaks back, and it's read aloud in
        voice mode as often as it's read on screen.
        """
        # [-2], not [-1]: by the time this runs, the "why did you route
        # that" query has itself been classified and recorded, so [-1] is
        # this question, not the decision being asked about.
        with self._lock:
            items = list(self._ring)
        previous = None
        for rec in reversed(items):
            if rec.get("intent") != "explain_routing":
                previous = rec
                break
        if previous is None:
            return "I haven't routed anything yet this session."

        parts = [
            f"Your last query was {previous['query']!r}.",
            f"The NLU classified it as '{previous['intent']}' "
            f"with {previous['confidence']:.0%} confidence",
        ]
        if previous.get("source") and previous["source"] != "nlu":
            parts[-1] += f" (via the {previous['source']} path)"
        parts[-1] += "."
        parts.append(f"Hecate routed it to the '{previous['module']}' module")
        if previous.get("reason"):
            parts[-1] += f" — {previous['reason']}"
        parts[-1] += "."
        if previous.get("latency_ms"):
            parts.append(f"That took {previous['latency_ms']:.0f} ms end to end.")
        parts.append(f"Request id: {previous.get('request_id', '-')}.")
        return " ".join(parts)

    # -- 4. feedback ---------------------------------------------------

    def record_feedback(self, note: str = "") -> str:
        """
        Log an explicit user correction against the previous decision.

        Returns the confirmation string the ``report_mistake`` intent
        speaks back.
        """
        target = None
        with self._lock:
            for rec in reversed(list(self._ring)):
                if rec.get("intent") != "report_mistake":
                    target = rec
                    break

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request_id": current_request_id(),
            "note": (note or "").strip()[:500],
            "query": (target or {}).get("query", ""),
            "intent": (target or {}).get("intent", ""),
            "module": (target or {}).get("module", ""),
            "confidence": (target or {}).get("confidence", 0.0),
        }
        self._feedback_log.write(entry)

        if not target:
            return (
                "Noted, though I don't have a previous query on record this "
                "session to attach it to."
            )
        return (
            f"Logged. I had {entry['query']!r} down as '{entry['intent']}' "
            f"→ {entry['module']}; that's now marked as wrong for review."
        )

    def feedback_count(self) -> int:
        """Number of feedback records on disk (0 if the log is unwritable)."""
        path = _DEFAULT_LOG_DIR / "feedback.jsonl"
        try:
            with path.open("r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        except OSError:
            return 0

    # -- module status (backlog #8) ------------------------------------

    def module_status(self) -> dict[str, dict[str, Any]]:
        """
        One place reporting every registered module's health.

        Several modules already expose ``available()`` / ``ready()``
        individually (Athena's RAG index, Iris's embedder, Pluto's market
        feed); this asks each registered module for whichever it has and
        normalises the answer, so a caller doesn't need to know which
        module named the method what.

        ``state`` is one of:
          ``ready``     — the module says it's usable right now
          ``degraded``  — the module is registered but reports not ready
          ``unknown``   — the module exposes no health method at all
          ``error``     — asking it raised
        """
        orch = self._orchestrator
        if orch is None:
            return {}

        try:
            names = list(orch.registered_modules)
        except Exception:
            logger.debug("Could not enumerate registered modules.")
            return {}

        out: dict[str, dict[str, Any]] = {}
        for name in names:
            module = getattr(orch, "_modules", {}).get(name)
            entry: dict[str, Any] = {"registered": True, "state": "unknown"}
            if module is None:
                out[name] = entry
                continue

            probe_name = None
            for candidate in ("ready", "available", "is_ready", "is_available"):
                if callable(getattr(module, candidate, None)):
                    probe_name = candidate
                    break

            if probe_name is None:
                # No health method. Registration is itself the only signal
                # we have — report that honestly rather than claiming ready.
                entry["state"] = "unknown"
                entry["probe"] = None
            else:
                try:
                    healthy = bool(getattr(module, probe_name)())
                    entry["state"] = "ready" if healthy else "degraded"
                    entry["probe"] = probe_name
                except Exception as exc:
                    entry["state"] = "error"
                    entry["probe"] = probe_name
                    entry["detail"] = str(exc)[:200]
            out[name] = entry
        return out

    def status_summary(self) -> str:
        """Prose version of ``module_status`` for the spoken/chat reply."""
        status = self.module_status()
        if not status:
            return "I can't see the module registry from here."

        by_state: dict[str, list[str]] = {}
        for name, info in sorted(status.items()):
            by_state.setdefault(info["state"], []).append(name)

        lines = [f"{len(status)} module(s) registered."]
        for state in ("ready", "degraded", "error", "unknown"):
            names = by_state.get(state)
            if names:
                lines.append(f"{state}: {', '.join(names)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Latency helper
# ---------------------------------------------------------------------------

class Timer:
    """
    Context manager yielding elapsed milliseconds.

        with Timer() as t:
            ...
        t.ms  # float

    Uses ``perf_counter`` (monotonic) so a system clock adjustment can't
    produce a negative latency in the routing log.
    """

    def __init__(self) -> None:
        self.ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
