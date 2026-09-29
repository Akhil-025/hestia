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
from core.language_detect import detect_script
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

def _read_jsonl_since(path: Path, cutoff_ts: float) -> list[dict[str, Any]]:
    """
    Read *path* as JSONL, returning records whose ``ts`` field parses to a
    timestamp at or after *cutoff_ts*.

    Shared by every log-window query in this module
    (`low_confidence_since`, `per_intent_accuracy`) so "how do we read a
    rotating JSONL log and treat malformed lines" has exactly one answer:
    skip lines that aren't valid JSON, skip records with no parseable
    ``ts``, and return an empty list rather than raising if the file
    doesn't exist yet (a brand-new install with no traffic yet).

    Reads only *path* itself, not its rotated `.1`/`.2`/`.3` backups — see
    `Diagnostics.low_confidence_since`'s docstring for why that's a
    documented approximation rather than a bug for realistic personal-use
    volume.
    """
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                try:
                    ts = datetime.fromisoformat(record["ts"]).timestamp()
                except (KeyError, TypeError, ValueError):
                    continue
                if ts >= cutoff_ts:
                    out.append(record)
    except OSError:
        pass
    return out


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
            # backlog #27: which Unicode script the query was written in
            # (see core/language_detect.py). Lets a future accuracy
            # breakdown ask "is classification worse for Devanagari input"
            # from the log alone, without re-running anything.
            "script": detect_script(query or ""),
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

    # -- nightly low-confidence review (backlog #6) ---------------------

    _REVIEW_QUEUE_FILE = "review_queue.jsonl"

    def low_confidence_since(
        self, hours: float = 24.0, threshold: float = 0.6
    ) -> list[dict[str, Any]]:
        """
        Routing records from the last *hours* with confidence below
        *threshold*, read straight from ``logs/routing.jsonl``.

        Reads the log file rather than the in-memory ring buffer
        deliberately: the ring only holds the last 50 decisions
        (_RING_SIZE), nowhere near a full day's traffic, and this is
        explicitly a "yesterday" report. Reads only the CURRENT log file,
        not its rotated backups — a day's worth of classifications is
        smaller than one rotation (5 MB) for any realistic personal-use
        volume, so this is a documented approximation, not a bug: if the
        file happens to have rotated mid-window, the oldest part of the
        window is silently missed rather than the call failing.
        """
        path = _DEFAULT_LOG_DIR / "routing.jsonl"
        cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
        return [
            record for record in _read_jsonl_since(path, cutoff)
            if float(record.get("confidence", 1.0) or 0.0) < threshold
        ]

    def write_review_queue(
        self, hours: float = 24.0, threshold: float = 0.6
    ) -> int:
        """
        Append yesterday's low-confidence classifications to
        ``logs/review_queue.jsonl`` for manual labelling, deduplicated
        against what's already queued by request id.

        Called once a day by ``core.heartbeat.HestiaHeartbeat`` (backlog
        #6) rather than exposing this only as something you'd have to
        remember to run — "surfaces them for you to manually label" only
        works if surfacing doesn't depend on remembering to ask. Returns
        the number of NEW entries appended (0 is not an error — it means
        classification was confident all day, which is the goal, not a
        failure of this method).
        """
        candidates = self.low_confidence_since(hours=hours, threshold=threshold)
        if not candidates:
            return 0

        queue_path = _DEFAULT_LOG_DIR / self._REVIEW_QUEUE_FILE
        existing_ids: set[str] = set()
        try:
            with queue_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing_ids.add(json.loads(line).get("request_id", ""))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            pass  # queue file doesn't exist yet — nothing to dedup against

        new_entries = [
            c for c in candidates if c.get("request_id") not in existing_ids
        ]
        if not new_entries:
            return 0

        try:
            _DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with queue_path.open("a", encoding="utf-8") as fh:
                for entry in new_entries:
                    fh.write(json.dumps(entry, ensure_ascii=False, default=str))
                    fh.write("\n")
        except OSError as exc:
            logger.debug("Could not write review queue: %s", exc)
            return 0

        return len(new_entries)

    def review_queue_summary(self) -> str:
        """
        Prose summary of the review queue, for the heartbeat's log line
        and for a future "review my low-confidence queue" intent to speak.
        """
        queue_path = _DEFAULT_LOG_DIR / self._REVIEW_QUEUE_FILE
        try:
            with queue_path.open("r", encoding="utf-8") as fh:
                count = sum(1 for line in fh if line.strip())
        except OSError:
            return "No low-confidence classifications queued for review."
        if count == 0:
            return "No low-confidence classifications queued for review."
        return (
            f"{count} low-confidence classification(s) queued for review "
            f"in logs/{self._REVIEW_QUEUE_FILE}."
        )

    # -- per-intent accuracy tracking (backlog #30) ---------------------

    _MIN_SAMPLES_FOR_ACCURACY = 3

    def per_intent_accuracy(
        self, days: float = 7.0, min_samples: int = _MIN_SAMPLES_FOR_ACCURACY
    ) -> dict[str, dict[str, Any]]:
        """
        Per-intent accuracy estimate over the last *days*, derived from
        two logs that already exist for other reasons: `routing.jsonl`
        supplies how many times each intent was classified at all, and
        `feedback.jsonl` supplies how many of those were explicitly
        flagged wrong via the `report_mistake` intent (backlog #259) — an
        explicit correction, not a guess at silent dissatisfaction.

        This is an ESTIMATE, not ground truth, and the name says so:
        accuracy_estimate = 1 - (flagged_wrong / total). It only reflects
        mistakes the user bothered to report, so it's a lower bound on
        the true error rate, not a measurement of it — a wrong
        classification the user never mentioned counts as correct here.
        Intents with fewer than *min_samples* total classifications in
        the window are excluded entirely rather than reported with a
        misleadingly precise-looking 0% or 100%, since a single
        classification's outcome is not a rate.

        Reads ONLY the current routing/feedback log files, same documented
        approximation as `low_confidence_since` — see that method's
        docstring for why.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400

        totals: dict[str, int] = {}
        for path_name in ("routing.jsonl",):
            for record in _read_jsonl_since(_DEFAULT_LOG_DIR / path_name, cutoff):
                intent = record.get("intent")
                if intent:
                    totals[intent] = totals.get(intent, 0) + 1

        flagged: dict[str, int] = {}
        for record in _read_jsonl_since(_DEFAULT_LOG_DIR / "feedback.jsonl", cutoff):
            intent = record.get("intent")
            if intent:
                flagged[intent] = flagged.get(intent, 0) + 1

        out: dict[str, dict[str, Any]] = {}
        for intent, total in totals.items():
            if total < min_samples:
                continue
            wrong = flagged.get(intent, 0)
            out[intent] = {
                "total": total,
                "flagged_wrong": wrong,
                "accuracy_estimate": round(1 - (wrong / total), 3),
            }
        return out

    def worst_performing_intents(
        self,
        days: float = 7.0,
        min_samples: int = _MIN_SAMPLES_FOR_ACCURACY,
        top_n: int = 5,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        The *top_n* lowest-accuracy intents over the window, worst first.

        Ties broken by sample count descending — a 70% estimate from 20
        samples is a more useful thing to look at than a 70% estimate
        from 3, even though the estimate itself is identical.
        """
        accuracy = self.per_intent_accuracy(days=days, min_samples=min_samples)
        ranked = sorted(
            accuracy.items(),
            key=lambda kv: (kv[1]["accuracy_estimate"], -kv[1]["total"]),
        )
        return ranked[:top_n]

    def weekly_accuracy_summary(self, days: float = 7.0) -> str:
        """
        Prose report for the heartbeat's weekly job and its log line —
        the worst-performing intents in the window, or an honest "not
        enough data" / "nothing flagged" when there's nothing to show.
        """
        accuracy = self.per_intent_accuracy(days=days)
        if not accuracy:
            return (
                f"Not enough classification volume in the last "
                f"{days:.0f} days to estimate per-intent accuracy."
            )

        worst = [
            (intent, stats) for intent, stats in self.worst_performing_intents(days=days)
            if stats["flagged_wrong"] > 0
        ]
        if not worst:
            return f"No mistakes reported against any intent in the last {days:.0f} days."

        lines = [f"Worst-performing intent(s) over the last {days:.0f} days:"]
        for intent, stats in worst:
            lines.append(
                f"  {intent}: ~{stats['accuracy_estimate']:.0%} "
                f"({stats['flagged_wrong']}/{stats['total']} flagged wrong)"
            )
        return "\n".join(lines)

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
        breakers = getattr(orch, "circuit_breaker_status", None) or {}
        for name in names:
            module = getattr(orch, "_modules", {}).get(name)
            entry: dict[str, Any] = {"registered": True, "state": "unknown"}
            if name in breakers:
                # A breaker only exists once a module has been dispatched
                # to at least once, so this is absent for most modules
                # most of the time — that absence is itself informative
                # (never called, or never failed) and left out rather than
                # padded with a fake "closed" default.
                entry["circuit"] = breakers[name]
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

            # An open circuit breaker means the module is currently being
            # skipped regardless of what its own health probe says (the
            # probe checks "can it work in principle"; the breaker tracks
            # "has it actually been failing on real calls") — surface the
            # more urgent signal without overwriting an existing "error".
            if entry.get("circuit", {}).get("state") == "open" and entry["state"] != "error":
                entry["state"] = "degraded"
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
