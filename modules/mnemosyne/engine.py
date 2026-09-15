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

class MnemosyneEngine(BaseModule):
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
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, hestia_llm: Any) -> None:
        self.config = get_config()
        self.db = MnemosyneDB(self.config.db_path)
        self.hestia_llm = hestia_llm
        self.vector_store: Optional[MnemosyneVectorStore] = None
        self.summariser: Optional[Summariser] = None

        if _CHROMA_AVAILABLE:
            self.vector_store = MnemosyneVectorStore(
                self.config.chroma_dir,
                self.config.embedding_model,
            )

        if _SUMMARISER_AVAILABLE:
            self.summariser = Summariser(self, hestia_llm)

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
                label = "name" if key == "user_name" else _readable(key)
                return _ok(f"Your {label} is {value}.", confidence=0.95)
            return _ok("I don't have that information yet.", confidence=0.5)

        # Fallback: semantic recall
        response = self.remember(query)
        return _ok(response or "I don't have anything on that.", confidence=0.85)

    def _handle_learn_fact(self, entities: dict) -> dict:
        key: str = (entities.get("key") or "").strip()
        value: str = (entities.get("value") or "").strip()

        if not key or not value:
            return _ok("What should I remember?", confidence=0.0)

        self.learn(key, value)
        return _ok(f"Got it — I'll remember your {_readable(key)}.", confidence=0.95)

    def _handle_forget_fact(self, entities: dict) -> dict:
        """
        Forget a remembered fact.

        Gated by the orchestrator's confirmation mechanism (see
        HestiaOrchestrator._resolve_pending): the first call shows what's
        about to be forgotten and asks for confirmation instead of deleting
        it immediately — forgetting is not undoable, and "forget fact" is
        exactly the kind of short, easily-misheard phrase STT gets wrong.
        """
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

        results = summaries + facts
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

        return " ".join(lines)

    def learn(self, key: str, value: str, source: str = "user") -> None:
        """Persist a key/value fact to SQLite and the vector store."""
        if not key or not value:
            raise ValueError(f"learn() requires non-empty key and value; got key={key!r} value={value!r}")

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

    def forget(self, key: str) -> None:
        """Remove a fact from SQLite and the vector store."""
        if not key:
            raise ValueError("forget() requires a non-empty key.")

        self.db.delete_fact(key)

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
            facts = self.db.get_top_facts(limit)
        except Exception:
            logger.exception("Failed to fetch top facts")
            return ""

        if not facts:
            return ""

        return "\n".join(f"- {f['key']}: {f['value']}" for f in facts)

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

        if kind == "fact":
            key = meta.get("key", "")
            label = _readable(key) if key else "detail"
            return f"Your {label} is {text}."

        if kind == "summary":
            topic = meta.get("topic", "")
            prefix = (
                f"Regarding {topic}: "
                if topic and topic.lower() != "general"
                else ""
            )
            return f"{prefix}{text}"

        return text