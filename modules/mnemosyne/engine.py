"""
modules/mnemosyne/engine.py

MnemosyneEngine: unified entry point for memory, goals, and summarisation.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from modules.base import BaseModule
from .config import get_config
from .db import MnemosyneDB
from .extensions import EXTENSION_INTENTS, MnemosyneExtensionsMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from .vector_store import MnemosyneVectorStore
    _CHROMA_AVAILABLE = True
except ImportError:
    logger.warning(
        "ChromaDB or sentence-transformers not available; semantic memory disabled."
    )
    _CHROMA_AVAILABLE = False

try:
    from .summariser import Summariser
    _SUMMARISER_AVAILABLE = True
except ImportError:
    logger.warning("Summariser not available; summarisation disabled.")
    _SUMMARISER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _readable(key: str) -> str:
    """Convert a snake_case key to a human-readable label."""
    return key.replace("_", " ")


def _months_delta(months: float):
    """A timedelta approximating *months* calendar months (30.44 days/month average)."""
    from datetime import timedelta
    return timedelta(days=months * 30.44)


def _normalise_key(key: str) -> str:
    """Lowercase, underscore-stripped form of a fact key, for fuzzy-matching similar keys."""
    return (key or "").lower().replace("_", " ").strip()


def _relative_day(ts: datetime) -> str:
    """'today' / 'yesterday' / 'on Tuesday' / 'on 2026-01-15' relative to now, for provenance phrasing."""
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    days = (now.date() - ts.date()).days
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if 2 <= days <= 6:
        return f"on {ts.strftime('%A')}"
    return f"on {ts.strftime('%Y-%m-%d')}"


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _ok(response: str, data: Optional[dict] = None, confidence: float = 0.9) -> dict:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _miss(response: str = "I don't have anything on that.") -> dict:
    return {"response": response, "data": {}, "confidence": 0.0}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MnemosyneEngine(MnemosyneExtensionsMixin, BaseModule):
    """
    Unified entry point for memory, goals, and summarisation.

    Responsibilities
    ----------------
    - Persist and retrieve interactions via SQLite (MnemosyneDB).
    - Provide semantic recall via an optional ChromaDB vector store.
    - Delegate periodic summarisation to an optional Summariser.
    - Expose a stable `handle` / `can_handle` interface for the dispatcher.
    """

    name = "mnemosyne"

    _INTENTS: frozenset[str] = frozenset(
        {
            "remember",
            "recall",
            "get_facts",
            "learn_fact",
            "forget_fact",
            "get_user_info",
        }
    ) | EXTENSION_INTENTS

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, hestia_llm: Any) -> None:
        self.config = get_config()
        self.db = MnemosyneDB(self.config.db_path)
        self.hestia_llm = hestia_llm
        self.vector_store: Optional[MnemosyneVectorStore] = None
        self.summariser: Optional[Summariser] = None
        # Callbacks told about every precise device-location update
        # (Chronos uses this for "remind me when I get home", backlog #82).
        self._location_listeners: list[Any] = []

        if _CHROMA_AVAILABLE:
            self.vector_store = MnemosyneVectorStore(
                self.config.chroma_dir,
                self.config.embedding_model,
            )

        if _SUMMARISER_AVAILABLE:
            self.summariser = Summariser(self, hestia_llm)

        # Knowledge graph, study cards, episodes, quizzes, and the opt-in
        # Obsidian / paper-monitor features (backlog #31-#47). See
        # extensions.py; each component degrades independently.
        self._init_extensions()

        logger.info(
            "MnemosyneEngine ready (vector_store=%s, summariser=%s)",
            self.vector_store is not None,
            self.summariser is not None,
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Route an intent to the appropriate handler.

        Returns a response dict with keys: response, data, confidence.
        Never raises; errors are caught and returned as low-confidence misses.
        """
        try:
            return self._dispatch(intent, entities, context)
        except Exception:
            logger.exception("handle() failed for intent=%s", intent)
            return _miss("Something went wrong retrieving that memory.")

    # ------------------------------------------------------------------
    # Dispatcher (private)
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, entities: dict, context: dict) -> dict:
        query: str = (
            entities.get("query")
            or entities.get("raw_query")
            or context.get("raw_query", "")
        )

        if intent in ("remember", "recall", "get_facts"):
            return self._handle_recall(query)

        if intent == "get_user_info":
            return self._handle_get_user_info(entities, query)

        if intent == "learn_fact":
            return self._handle_learn_fact(entities)

        if intent == "forget_fact":
            return self._handle_forget_fact(entities)

        if intent in EXTENSION_INTENTS:
            result = self._dispatch_extension(intent, entities, context)
            if result is not None:
                return result

        return _miss()

    # ------------------------------------------------------------------
    # Intent handlers (private)
    # ------------------------------------------------------------------

    def _handle_recall(self, query: str) -> dict:
        response = self.remember(query)
        if not response:
            response = "I don't have any memories about that yet."
        return _ok(response, confidence=0.85)

    def _handle_get_user_info(self, entities: dict, query: str) -> dict:
        # `or ""` (not just a `.get(..., "")` default) because the NLU can
        # emit the key explicitly as None (present but empty) rather than
        # omitting it — a plain default only covers the missing-key case
        # and `.strip()` on None would raise, turning this into a generic
        # "something went wrong" instead of falling through to recall.
        key: str = (entities.get("key") or "").strip()

        if key:
            value = self.db.get_fact(key)
            if value:
                self.db.touch_fact(key)
                label = "name" if key == "user_name" else _readable(key)
                row = self.db.get_fact_row(key)
                provenance = (
                    self._provenance_phrase(row.get("created_at"), row.get("source"))
                    if row else ""
                )
                return _ok(f"Your {label} is {value}.{provenance}", confidence=0.95)
            return _ok("I don't have that information yet.", confidence=0.5)

        # Fallback: semantic recall
        response = self.remember(query)
        return _ok(response or "I don't have anything on that.", confidence=0.85)

    def _handle_learn_fact(self, entities: dict) -> dict:
        key: str = (entities.get("key") or "").strip()
        value: str = (entities.get("value") or "").strip()

        if not key or not value:
            return _ok("What should I remember?", confidence=0.0)

        # Contradiction check (backlog #37) runs BEFORE writing, against
        # the fact table as it stood before this call — checking after
        # the write risks the new row itself (if its key happens to fuzzy-
        # match something) being compared against, which is meaningless.
        conflict = self.check_for_contradiction(key, value)

        result = self.learn(key, value)

        if result["deduplicated"]:
            other = _readable(result["matched_key"])
            return _ok(
                f"I already have that noted (as your {other}).", confidence=0.9,
                data={"deduplicated": True, "matched_key": result["matched_key"]},
            )

        response = f"Got it — I'll remember your {_readable(key)}."
        if conflict:
            response += (
                f" Note: this seems to differ from what you told me about "
                f"your {_readable(conflict['key'])} (\"{conflict['value']}\") — "
                f"let me know if that one should be updated instead."
            )
        return _ok(response, confidence=0.95, data={"conflict": conflict} if conflict else {})

    def _handle_forget_fact(self, entities: dict) -> dict:
        """
        Forget a remembered fact — a single one by exact key, or every
        fact matching a pattern (backlog #41, "forget everything about
        X"). Both paths are gated by the orchestrator's confirmation
        mechanism (see HestiaOrchestrator._resolve_pending): the first
        call shows what's about to be forgotten and asks for confirmation
        instead of deleting immediately — forgetting is not undoable, and
        doubly so for a bulk delete.
        """
        pattern: str = (entities.get("pattern") or entities.get("topic") or "").strip()
        if pattern:
            return self._handle_forget_matching(entities, pattern)

        key: str = (entities.get("key") or "").strip()
        if not key:
            return _ok("Which fact should I forget?", confidence=0.0)

        if not entities.get("_confirmed"):
            current = self.db.get_fact(key)
            if not current:
                return _ok(f"I don't have anything remembered for {_readable(key)}.", confidence=0.5)
            return {
                "response": (
                    f"Forget that your {_readable(key)} is \"{current}\"? "
                    "Say yes to confirm."
                ),
                "data": {"key": key, "value": current},
                "confidence": 0.9,
                "needs_confirmation": True,
                "confirm_intent": "forget_fact",
                "confirm_entities": {"key": key},
                "confirm_label": f"forget your {_readable(key)}",
            }

        self.forget(key)
        logger.info("Fact forgotten: key=%s", key)
        return _ok(f"Forgotten: {_readable(key)}.", confidence=0.9)

    def _handle_forget_matching(self, entities: dict, pattern: str) -> dict:
        matches = self.find_matching_facts(pattern)
        if not matches:
            return _ok(f"I don't have anything remembered about {pattern}.", confidence=0.5)

        if not entities.get("_confirmed"):
            listed = ", ".join(_readable(k) for k in matches[:10])
            more = f" and {len(matches) - 10} more" if len(matches) > 10 else ""
            return {
                "response": (
                    f"That would forget {len(matches)} thing(s) about {pattern}: "
                    f"{listed}{more}. Say yes to confirm."
                ),
                "data": {"pattern": pattern, "keys": matches},
                "confidence": 0.9,
                "needs_confirmation": True,
                "confirm_intent": "forget_fact",
                "confirm_entities": {"pattern": pattern},
                "confirm_label": f"forget everything about {pattern}",
            }

        removed = self.forget_matching(matches)
        logger.info("Bulk-forgot %d fact(s) matching %r: %s", len(removed), pattern, removed)
        return _ok(f"Forgotten: {len(removed)} thing(s) about {pattern}.", confidence=0.9)

    # ------------------------------------------------------------------
    # Core memory operations (public)
    # ------------------------------------------------------------------

    def push(
        self,
        user_text: str,
        hestia_response: str,
        intent: str,
        source_device: str = "hestia",
    ) -> None:
        """Persist an interaction and trigger summarisation if due."""
        if not user_text or not hestia_response:
            logger.warning("push() called with empty user_text or hestia_response; skipping.")
            return

        self.db.push_interaction(user_text, hestia_response, intent, source_device)

        if self.summariser and self.summariser.should_summarise():
            try:
                self.summariser.run()
            except Exception:
                logger.exception("Summarisation failed; continuing without it.")

    def remember(self, query: str, n: int = 5) -> str:
        """
        Semantic recall over summaries and facts.

        Returns a natural-language string or an empty string when nothing
        is found (callers decide how to phrase the fallback).
        """
        if not self.vector_store:
            return ""

        if not query or not query.strip():
            return ""

        try:
            summaries = self.vector_store.search(
                query,
                n_results=n,
                where={"type": {"$eq": "summary"}},
            )
            facts = self.vector_store.search(
                query,
                n_results=n,
                where={"type": {"$eq": "fact"}},
            )
        except Exception:
            logger.exception("Vector search failed for query=%r", query)
            return ""

        results = summaries + facts + self._recall_notes(query, n)
        if not results:
            return ""

        # Deduplicate by id, sorted by relevance score descending.
        #
        # `r["id"]` is the vector store's doc_id: for facts this is the fact
        # `key` (see `learn()` above), and for summaries it's the summary's
        # own row id (see `mnemosyne/summariser.py`). Both are non-empty
        # strings by construction — `learn()` rejects an empty key, and
        # summary ids are generated from an autoincrement primary key — so
        # `seen` correctly dedupes on a stable identifier rather than on
        # content, which two different facts/summaries could otherwise share.
        seen: set[str] = set()
        deduped = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            assert r.get("id"), f"vector_store result missing non-empty id: {r!r}"
            if r["id"] not in seen:
                deduped.append(r)
                seen.add(r["id"])

        # Vector search always returns up to `n_results` items even when
        # none of them are actually relevant to the query — for a vague
        # query like "what did we talk about yesterday?" the embedding
        # weakly matches everything in history (job offers, flights, pizza,
        # SWOT analyses), and without a floor all of it gets concatenated
        # into one answer, which reads as a hallucinated mashup even though
        # every individual line is real.
        #
        # `score` is now 1/(1 + L2_distance) (see
        # vector_store.py::_distances_to_scores) — an absolute,
        # batch-independent similarity, not a per-query min-max rescale.
        # 0.5 is a starting point (score=0.5 <=> L2 distance=1.0 between
        # embeddings), not a measured value — I don't have a way to run
        # your actual embedding model here, so log the (query, score) pairs
        # for a week of real traffic and adjust this against where genuine
        # matches vs. noise actually fall for your embedding_model config.
        _MIN_RELEVANCE = 0.5
        deduped = [r for r in deduped if r.get("score", 0) >= _MIN_RELEVANCE]

        lines: list[str] = []
        for r in deduped[:n]:
            line = self._format_result(r)
            if line:
                lines.append(line)
            # Being recalled is direct evidence a fact is still relevant
            # (backlog #45's frequency signal, and #36's decay job reads
            # this same access_count/last_accessed pair to know what NOT
            # to flag as stale).
            meta = r.get("metadata") or {}
            if meta.get("type") == "fact" and meta.get("key"):
                try:
                    self.db.touch_fact(meta["key"])
                except Exception:
                    logger.debug("touch_fact failed for key=%s", meta.get("key"))

        return " ".join(lines)

    def learn(self, key: str, value: str, source: str = "user") -> dict:
        """
        Persist a key/value fact to SQLite and the vector store.

        Returns {"deduplicated": bool, "matched_key": Optional[str]} —
        every existing caller (several modules call this as a fire-and-
        forget statement and never inspected the old None return) is
        unaffected by this becoming a dict instead of None (backlog #42).

        Deduplication (#42) only applies when *key* doesn't already exist:
        an update to an existing key is deliberate (the caller already
        knows the key) and goes through the normal upsert untouched. Only
        for a genuinely NEW key is the value checked against existing
        facts' values — if it closely matches one under a DIFFERENT key,
        that existing fact's reference count is bumped instead of writing
        a near-duplicate second copy.
        """
        if not key or not value:
            raise ValueError(f"learn() requires non-empty key and value; got key={key!r} value={value!r}")

        is_new_key = self.db.get_fact(key) is None
        if is_new_key:
            matched_key = self._find_duplicate_value(key, value)
            if matched_key:
                self.db.touch_fact(matched_key)
                logger.info(
                    "learn(): %r deduplicated against existing fact %r "
                    "(value similarity >= %.2f); no new row written.",
                    key, matched_key, self._DEDUP_VALUE_SIMILARITY,
                )
                return {"deduplicated": True, "matched_key": matched_key}

        self.db.set_fact(key, value, source)

        if self.vector_store:
            metadata = {
                "type": "fact",
                "created_at": _utc_now(),
                "key": key,
                "source": source,
            }
            try:
                self.vector_store.add(value, metadata, doc_id=key)
            except Exception:
                logger.exception("Vector store add failed for key=%s", key)

        self._graph_learn_fact(key, value)      # backlog #31: keep the graph current

        return {"deduplicated": False, "matched_key": None}

    def forget(self, key: str) -> None:
        """Remove a fact from SQLite and the vector store."""
        if not key:
            raise ValueError("forget() requires a non-empty key.")

        self.db.delete_fact(key)
        self._graph_forget_fact(key)            # graph edges + study card go with it

        if self.vector_store:
            try:
                self.vector_store.delete(key)
            except Exception:
                logger.exception("Vector store delete failed for key=%s", key)

    # ------------------------------------------------------------------
    # Summarisation helpers (public)
    # ------------------------------------------------------------------

    def trigger_summarise(self) -> None:
        """Externally trigger an immediate summarisation run."""
        if not self.summariser:
            logger.warning("trigger_summarise() called but summariser is unavailable.")
            return
        try:
            self.summariser.run()
        except Exception:
            logger.exception("trigger_summarise() failed.")

    def add_summary(
        self,
        period_start: str,
        period_end: str,
        content: str,
        topic: str,
        interaction_count: int,
    ) -> int:
        """Write a summary to SQLite and embed it in the vector store."""
        if not content or not content.strip():
            raise ValueError("add_summary() requires non-empty content.")

        summary_id = self.db.add_summary(
            period_start, period_end, content, topic, interaction_count
        )

        if self.vector_store:
            metadata = {
                "type": "summary",
                "created_at": _utc_now(),
                "topic": topic,
            }
            try:
                self.vector_store.add(content, metadata, doc_id=str(summary_id))
            except Exception:
                logger.exception("Vector store add failed for summary_id=%d", summary_id)

        return summary_id

    def get_unsummarised(self, limit: int = 50) -> list[dict]:
        return self.db.get_unsummarised(limit)

    def mark_summarised(self, ids: list[int]) -> None:
        if ids:
            self.db.mark_summarised(ids)

    # ------------------------------------------------------------------
    # Reminder helpers (public)
    # ------------------------------------------------------------------

    def add_reminder(self, text: str, due_time: str) -> None:
        if not text or not due_time:
            raise ValueError("add_reminder() requires non-empty text and due_time.")
        self.db.add_reminder(text, due_time)

    def get_due_reminders(self) -> list[dict]:
        return self.db.get_due_reminders(_utc_now())

    def mark_reminder_done(self, rid: int) -> None:
        self.db.mark_reminder_done(rid)

    # ------------------------------------------------------------------
    # Context and stats (public)
    # ------------------------------------------------------------------

    def get_context(self) -> dict:
        """
        Return a lightweight context snapshot for NLU enrichment.
        Never raises; returns an empty dict on failure.
        """
        try:
            s = self.status()
            recent = self.db.get_recent_interactions(3)
            recent_summary = [
                {"query": r["query"], "intent": r["intent"]} for r in recent
            ]
            facts = self.db.get_all_facts()
            top_facts = {f["key"]: f["value"] for f in facts[:5]}

            return {
                "mnemosyne_facts": s.get("facts", 0),
                "mnemosyne_goals": s.get("active_goals", 0),
                "mnemosyne_summaries": s.get("summaries", 0),
                "mnemosyne_recent": recent_summary,
                "mnemosyne_top_facts": top_facts,
                # Not guaranteed to appear in top_facts above (only the 5
                # most-recently-updated facts make that cut), but every god
                # that wants "near me" queries — Chronos for weather,
                # Dionysus for restaurants, Hephaestus for local search —
                # needs this reliably present, not just when it happens to
                # be the most recently touched fact.
                "device_location": self.get_device_location(),
            }
        except Exception:
            logger.exception("get_context() failed.")
            return {}

    # ------------------------------------------------------------------
    # Device location (public)
    # ------------------------------------------------------------------
    #
    # Stored as a fact under a fixed key so it reuses existing
    # set_fact/get_fact plumbing rather than needing a new table. Priority
    # for *source* of the value (browser GPS > Telegram location > IP
    # lookup) is decided by the caller — see web_ui.py / telegram_bot.py /
    # main.py's CLI fallback — this method just persists/retrieves whatever
    # it's given.

    _DEVICE_LOCATION_KEY = "device_location"

    def set_device_location(
        self, lat: float, lon: float, source: str = "unknown", label: Optional[str] = None
    ) -> None:
        """Persist the device's current coordinates."""
        payload = {
            "lat": lat,
            "lon": lon,
            "source": source,
            "label": label,
            "updated_at": _utc_now(),
        }
        self.db.set_fact(self._DEVICE_LOCATION_KEY, json.dumps(payload), source=source)
        # A coarse IP-derived fix can be kilometres off, which would make a
        # location reminder fire (or arm) spuriously - so only GPS-grade
        # sources are passed on.
        if source != "ip_geolocation":
            for listener in list(self._location_listeners):
                try:
                    listener(lat, lon, source)
                except Exception:
                    logger.exception("Location listener failed; continuing.")

    def add_location_listener(self, callback: Any) -> None:
        """Register ``callback(lat, lon, source)`` to run on each location update."""
        if callable(callback) and callback not in self._location_listeners:
            self._location_listeners.append(callback)

    def get_device_location(self) -> Optional[dict]:
        """Return the last known {"lat", "lon", "source", "label", "updated_at"}
        dict, or None if no location has ever been recorded. Never raises —
        a malformed stored value is treated as "no location" rather than
        breaking every caller of get_context()."""
        raw = self.db.get_fact(self._DEVICE_LOCATION_KEY)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("get_device_location: stored value is not valid JSON; ignoring.")
            return None

    def ensure_device_location_via_ip(self, timeout: float = 3.0) -> Optional[dict]:
        """
        Lowest-priority location source: IP-based geolocation, used only
        when nothing more precise has been recorded yet.

        Priority, per the device-location design, is:
          1. Browser GPS   (web_ui.py POST /api/location)
          2. Telegram location share (core/telegram_bot.py)
          3. IP lookup     (this method — CLI-only sessions have no sensor)

        A no-op if a location is already stored (whatever its source — this
        deliberately never overwrites a more precise GPS/Telegram fix with a
        coarser IP-based one). Best-effort: network failures are logged and
        swallowed rather than raised, since this is a convenience fallback,
        not a required startup step.
        """
        if self.get_device_location() is not None:
            return None

        try:
            import requests
            resp = requests.get("http://ip-api.com/json/", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                logger.info("ensure_device_location_via_ip: lookup unsuccessful (%s).", data.get("message"))
                return None
            lat, lon = data.get("lat"), data.get("lon")
            if lat is None or lon is None:
                return None
            label = ", ".join(p for p in (data.get("city"), data.get("country")) if p) or None
            self.set_device_location(float(lat), float(lon), source="ip_geolocation", label=label)
            logger.info("ensure_device_location_via_ip: set fallback location from IP (%s).", label)
            return self.get_device_location()
        except Exception:
            logger.info("ensure_device_location_via_ip: lookup failed; continuing without a location.", exc_info=True)
            return None

    def get_stats(self) -> dict:
        """Return aggregate statistics using SQL COUNT queries."""
        try:
            s = self.status()
            stats = self.db.get_interaction_stats()
            return {
                "total_interactions": stats["total"],
                "notes": stats["notes"],
                "facts_known": s.get("facts", 0),
                "unique_intents": stats["unique_intents"],
            }
        except Exception:
            logger.exception("get_stats() failed.")
            return {}

    def get_recent(self, limit: int = 5) -> list[dict]:
        """Return the most recent interactions (matches legacy HestiaMemory API)."""
        try:
            return self.db.get_recent_interactions(limit)
        except Exception:
            logger.exception("get_recent() failed.")
            return []

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Shim for modules that expect a preference lookup."""
        try:
            value = self.db.get_fact(key)
            return value if value is not None else default
        except Exception:
            logger.exception("get_preference() failed for key=%s", key)
            return default

    def status(self) -> dict:
        """Return live counts for facts, active goals, and summaries."""
        try:
            stats = self.db.get_memory_stats()
            return {
                "facts": stats["facts"],
                "active_goals": stats["goals"],
                "summaries": stats["summaries"],
            }
        except Exception:
            logger.exception("status() failed.")
            return {"facts": 0, "active_goals": 0, "summaries": 0}
        
    def get_top_facts_for_context(self, limit: int = 5) -> str:
        try:
            facts = self.get_top_facts_scored(limit)
        except Exception:
            logger.exception("Failed to fetch top facts")
            return ""

        if not facts:
            return ""

        return "\n".join(f"- {f['key']}: {f['value']}" for f in facts)

    # ------------------------------------------------------------------
    # Importance-weighted fact ranking (backlog #45)
    # ------------------------------------------------------------------
    #
    # Recency alone (the old ORDER BY updated_at DESC) means a fact
    # mentioned once, six months ago, that happens to have been the last
    # one touched, outranks a fact referenced constantly. Scoring blends
    # three signals, computed here in Python (not as one large SQL
    # expression) specifically so the weights are visible, unit-testable
    # constants rather than buried in a query string.

    # Tunable weights — see _score_fact for how each is combined. Kept as
    # class attributes (not module constants) so a subclass or a future
    # per-user config override could adjust them without editing this file.
    _RECENCY_HALF_LIFE_DAYS = 14.0   # a fact's recency contribution halves every ~2 weeks
    _WEIGHT_RECENCY = 0.4
    _WEIGHT_FREQUENCY = 0.3
    _WEIGHT_IMPORTANCE = 0.2
    _WEIGHT_CONFIDENCE = 0.1

    @classmethod
    def _score_fact(cls, fact: dict, now: Optional[datetime] = None) -> float:
        """
        Blend recency, access frequency, explicit importance, and
        confidence into one ranking score. Every component is normalised
        to roughly [0, 1] before weighting, so the weights above are
        directly comparable to each other rather than needing to also
        absorb unit conversions.
        """
        import math

        now = now or datetime.now(timezone.utc)

        updated_at = fact.get("updated_at")
        recency = 0.0
        if updated_at:
            try:
                ts = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
                recency = 0.5 ** (age_days / cls._RECENCY_HALF_LIFE_DAYS)
            except (TypeError, ValueError):
                recency = 0.0

        access_count = int(fact.get("access_count") or 0)
        # log1p so the 1st->2nd reference matters far more than the
        # 50th->51st — frequency should distinguish "never" from
        # "sometimes" much more than it distinguishes "a lot" from "a lot
        # more". Divided by log1p(20) so ~20 references saturates near 1.0.
        frequency = min(math.log1p(access_count) / math.log1p(20), 1.0)

        importance = max(0.0, min(1.0, float(fact.get("importance") if fact.get("importance") is not None else 0.5)))
        confidence = max(0.0, min(1.0, float(fact.get("confidence") if fact.get("confidence") is not None else 1.0)))

        return (
            cls._WEIGHT_RECENCY * recency
            + cls._WEIGHT_FREQUENCY * frequency
            + cls._WEIGHT_IMPORTANCE * importance
            + cls._WEIGHT_CONFIDENCE * confidence
        )

    def get_top_facts_scored(self, limit: int = 5) -> list[dict]:
        """Facts ranked by _score_fact, highest first — the #45 replacement for pure recency."""
        candidates = self.db.get_facts_for_scoring(limit=max(limit * 10, 50))
        now = datetime.now(timezone.utc)
        scored = sorted(candidates, key=lambda f: self._score_fact(f, now), reverse=True)
        return [{"key": f["key"], "value": f["value"]} for f in scored[:limit]]

    def set_fact_importance(self, key: str, importance: float) -> bool:
        """Explicitly weight a fact's ranking (backlog #45). Returns False if the key doesn't exist."""
        return self.db.set_fact_importance(key, importance)

    # ------------------------------------------------------------------
    # Fact expiry / decay (backlog #36)
    # ------------------------------------------------------------------
    #
    # Decay FLAGS a fact for review; it never deletes one automatically —
    # "not referenced in months" is a reasonable prompt to double-check a
    # fact, not proof it's wrong or unwanted. Deletion stays an explicit
    # forget()/forget_matching() call.

    _DEFAULT_STALE_MONTHS = 6

    def run_decay_check(self, months: float = _DEFAULT_STALE_MONTHS) -> list[dict]:
        """
        Flag facts not accessed in *months* months. Returns the newly-
        flagged facts (empty if nothing qualified or everything
        qualifying was already flagged from a previous run).
        """
        cutoff = (datetime.now(timezone.utc) - _months_delta(months)).isoformat()
        try:
            candidates = self.db.get_stale_facts(cutoff)
        except Exception:
            logger.exception("run_decay_check: could not query stale facts.")
            return []
        if not candidates:
            return []
        keys = [c["key"] for c in candidates]
        self.db.flag_stale(keys)
        logger.info("Decay check flagged %d fact(s) as stale: %s", len(keys), keys)
        return candidates

    def get_stale_facts_for_review(self, limit: int = 50) -> list[dict]:
        """Previously-flagged stale facts, for a review UI or digest."""
        try:
            return self.db.get_flagged_stale_facts(limit)
        except Exception:
            logger.exception("get_stale_facts_for_review failed.")
            return []

    # ------------------------------------------------------------------
    # Contradiction detection (backlog #37)
    # ------------------------------------------------------------------
    #
    # Deliberately keyed off KEY similarity, not value-embedding
    # similarity: two facts about genuinely different, unrelated topics
    # can have similar embeddings just for being ordinary English
    # sentences, which would make embedding-similarity-based contradiction
    # detection fire constantly on unrelated pairs. A near-identical KEY
    # ("favorite_color" vs "fav_colour") with a DIFFERENT value is a much
    # stronger, lower-false-positive signal that the user is re-stating
    # the same slot differently. Compare with _find_duplicate_value below,
    # which is the mirror case: same VALUE, different key.

    _CONTRADICTION_KEY_SIMILARITY = 0.8

    def check_for_contradiction(self, key: str, value: str) -> Optional[dict]:
        """
        If an EXISTING fact has a similarly-named key but a different
        value, return {"key": ..., "value": ...} for it — a candidate
        contradiction to surface to the user, never to block on. None if
        nothing looks like a conflict.
        """
        import difflib

        try:
            existing = self.db.get_all_facts(limit=1000)
        except Exception:
            logger.exception("check_for_contradiction: could not list facts.")
            return None

        norm_key = _normalise_key(key)
        best: Optional[dict] = None
        best_ratio = 0.0
        for fact in existing:
            other_key = fact.get("key", "")
            if other_key == key:
                continue  # same key = an update, not a contradiction
            ratio = difflib.SequenceMatcher(
                None, norm_key, _normalise_key(other_key)
            ).ratio()
            if ratio >= self._CONTRADICTION_KEY_SIMILARITY and ratio > best_ratio:
                if str(fact.get("value", "")).strip().lower() != str(value).strip().lower():
                    best, best_ratio = fact, ratio

        return {"key": best["key"], "value": best["value"]} if best else None

    # ------------------------------------------------------------------
    # Semantic deduplication on ingest (backlog #42)
    # ------------------------------------------------------------------
    #
    # Mirror case of contradiction detection above: same VALUE (high
    # embedding similarity), different key — the user restating a fact
    # they already told Hestia, under a new label. Rather than storing a
    # near-identical second copy, the existing fact's reference count is
    # bumped and no new row/embedding is written.

    _DEDUP_VALUE_SIMILARITY = 0.92

    def _find_duplicate_value(self, key: str, value: str) -> Optional[str]:
        """Return an existing DIFFERENT key whose stored value closely matches *value*, or None."""
        if not self.vector_store:
            return None
        try:
            results = self.vector_store.search(value, n_results=3, where={"type": {"$eq": "fact"}})
        except Exception:
            logger.exception("_find_duplicate_value: vector search failed.")
            return None
        for r in results:
            other_key = (r.get("metadata") or {}).get("key")
            if other_key and other_key != key and r.get("score", 0) >= self._DEDUP_VALUE_SIMILARITY:
                return other_key
        return None

    # ------------------------------------------------------------------
    # Memory provenance (backlog #50)
    # ------------------------------------------------------------------

    @staticmethod
    def _provenance_phrase(created_at: Optional[str], source: Optional[str]) -> str:
        """
        A short "(you told me this on Tuesday)" / "(inferred)" clause,
        or "" when there's nothing useful to say (no timestamp, or a
        malformed one — never raises trying to build this).
        """
        parts = []
        if created_at:
            try:
                ts = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                parts.append(f"mentioned {_relative_day(ts)}")
            except (TypeError, ValueError):
                pass
        if source and source not in ("user",):
            parts.append(source)
        return f" ({', '.join(parts)})" if parts else ""

    # ------------------------------------------------------------------
    # Dated recall (backlog #48)
    # ------------------------------------------------------------------

    def recall_on_date(self, date_str: str) -> str:
        """
        A first-class dated query — "what did I say on Tuesday" —
        distinct from semantic search (remember()), which has no notion
        of "on this specific day" at all. *date_str* must already be
        normalised to YYYY-MM-DD (callers resolve free text like
        "last Tuesday" via dateparser before calling this — see
        CoreModule/MnemosyneEngine's intent handler).
        """
        try:
            interactions = self.db.get_interactions_on_date(date_str)
            facts = self.db.get_facts_created_on_date(date_str)
        except Exception:
            logger.exception("recall_on_date failed for date=%s", date_str)
            return ""

        if not interactions and not facts:
            return ""

        lines: list[str] = []
        for f in facts:
            lines.append(f"You told me your {_readable(f['key'])} is {f['value']}.")
        if interactions:
            topics = ", ".join(
                i["query"][:60] for i in interactions[:5] if i.get("query")
            )
            if topics:
                lines.append(f"You also talked about: {topics}.")
        return " ".join(lines)

    # ------------------------------------------------------------------
    # Bulk forget (backlog #41)
    # ------------------------------------------------------------------

    def find_matching_facts(self, pattern: str) -> list[str]:
        """Fact keys containing *pattern* as a substring — candidates for bulk forget."""
        try:
            return self.db.search_fact_keys(pattern)
        except Exception:
            logger.exception("find_matching_facts failed for pattern=%s", pattern)
            return []

    def forget_matching(self, keys: list[str]) -> list[str]:
        """
        Forget every fact in *keys* (SQL row + vector store entry each).
        Returns the keys actually removed. Reuses forget() per-key rather
        than a bulk SQL statement so each deletion gets the same vector-
        store cleanup and error isolation forget() already has — one bad
        vector-store delete must not abort the rest of the batch.
        """
        removed = []
        for key in keys:
            try:
                self.forget(key)
                removed.append(key)
            except Exception:
                logger.exception("forget_matching: failed to forget key=%s", key)
        return removed

    # ------------------------------------------------------------------
    # Memory export (backlog #44)
    # ------------------------------------------------------------------

    def export_memory(self, fmt: str = "json") -> str:
        """
        A full dump of facts, goals, and summaries — independent of the
        sync API (that's for device-to-device delta sync; this is a
        point-in-time backup/portability snapshot a person can read or
        archive on its own).
        """
        facts = self.db.get_all_facts(limit=1000)
        summaries = self.db.get_recent_summaries(n=1000)
        goals = self.db.get_goals(status="active") + self.db.get_goals(status="completed")
        payload = {
            "exported_at": _utc_now(),
            "facts": facts,
            "summaries": summaries,
            "goals": goals,
        }

        if fmt == "markdown":
            lines = [f"# Hestia memory export", f"_Exported {payload['exported_at']}_", ""]
            lines.append(f"## Facts ({len(facts)})")
            for f in facts:
                lines.append(f"- **{f['key']}**: {f['value']}")
            lines.append("")
            lines.append(f"## Summaries ({len(summaries)})")
            for s in summaries:
                lines.append(f"- _{s.get('period_start', '?')}_ ({s.get('topic', 'General')}): {s.get('content', '')}")
            lines.append("")
            lines.append(f"## Goals ({len(goals)})")
            for g in goals:
                lines.append(f"- [{g.get('status', '?')}] {g.get('text', '')}")
            return "\n".join(lines)

        return json.dumps(payload, indent=2, default=str)

    # ------------------------------------------------------------------
    # Memory dashboard (backlog #49)
    # ------------------------------------------------------------------

    def get_memory_dashboard(self) -> dict:
        """
        Facts/goals/summaries/interaction counts, stale-fact count, DB
        size on disk, and embedding count — everything `get_memory_stats`
        already had, plus the size/cost signals that method didn't cover.
        """
        try:
            stats = self.db.get_memory_stats()
        except Exception:
            logger.exception("get_memory_dashboard: stats query failed.")
            stats = {}

        try:
            db_size = self.db.get_db_size_bytes()
        except Exception:
            db_size = 0

        embedding_count = None
        if self.vector_store is not None:
            try:
                embedding_count = self.vector_store.collection.count()
            except Exception:
                logger.exception("get_memory_dashboard: embedding count failed.")

        return {**stats, "db_size_bytes": db_size, "embedding_count": embedding_count}

    # ------------------------------------------------------------------
    # Weekly/monthly digest (backlog #40)
    # ------------------------------------------------------------------
    #
    # Distinct from Summariser (count-based: every N raw interactions):
    # this is a TIME-based rollup over already-generated summaries — the
    # "what happened this week" digest, not "summarise these 20 messages".
    # Called by core/heartbeat.py on a 7-day / 30-day rolling gap, the
    # same pattern as the #6/#30 heartbeat jobs.

    def generate_periodic_digest(self, period: str = "weekly") -> Optional[str]:
        """
        Roll up recent summaries into one higher-level digest via the LLM
        and store it (topic="weekly_digest"/"monthly_digest") so it shows
        up in recall/remember() like any other summary. Returns the
        digest text, or None if there was nothing to summarise or
        generation failed — logged either way, never raised.
        """
        n = 7 if period == "weekly" else 30
        try:
            recent = self.db.get_recent_summaries(n=n)
        except Exception:
            logger.exception("generate_periodic_digest: could not fetch summaries.")
            return None
        if not recent:
            logger.info("generate_periodic_digest(%s): nothing to summarise.", period)
            return None

        joined = "\n".join(
            f"- ({s.get('topic', 'General')}) {s.get('content', '')}" for s in recent
        )
        prompt = (
            f"Here are the last {len(recent)} conversation summaries. Write one "
            f"{period} digest paragraph (4-6 sentences) covering the recurring "
            f"themes and anything notable. Return only the paragraph, no preamble."
            f"\n\n{joined}"
        )
        try:
            digest = self.hestia_llm.generate(prompt).strip()
        except Exception:
            logger.exception("generate_periodic_digest: LLM call failed.")
            return None
        if not digest:
            return None

        now = _utc_now()
        self.add_summary(
            period_start=now, period_end=now, content=digest,
            topic=f"{period}_digest", interaction_count=len(recent),
        )
        # Opt-in write-back (backlog #39): a new note in the vault's Hestia folder.
        self.write_obsidian_note(
            f"{period.title()} digest {datetime.now():%Y-%m-%d}", digest,
            tags=[f"{period}-digest"],
        )
        return digest

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_result(result: dict) -> str:
        """Convert a vector-store search result into a readable sentence."""
        meta: dict = result.get("metadata", {})
        text: str = result.get("text", "").strip()

        if not text:
            return ""

        kind = meta.get("type")
        provenance = MnemosyneEngine._provenance_phrase(
            meta.get("created_at"), meta.get("source")
        )

        if kind == "fact":
            key = meta.get("key", "")
            label = _readable(key) if key else "detail"
            return f"Your {label} is {text}.{provenance}"

        if kind == "note":
            title = meta.get("title") or "an Obsidian note"
            return f"From your note {title}: {text}"

        if kind == "summary":
            topic = meta.get("topic", "")
            prefix = (
                f"Regarding {topic}: "
                if topic and topic.lower() != "general"
                else ""
            )
            return f"{prefix}{text}{provenance}"

        return text