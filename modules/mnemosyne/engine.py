"""
modules/mnemosyne/engine.py

MnemosyneEngine: unified entry point for memory, goals, and summarisation.
"""
from __future__ import annotations

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
        key: str = entities.get("key", "").strip()

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
        key: str = entities.get("key", "").strip()
        value: str = entities.get("value", "").strip()

        if not key or not value:
            return _ok("What should I remember?", confidence=0.0)

        self.learn(key, value)
        return _ok(f"Got it — I'll remember your {_readable(key)}.", confidence=0.95)

    def _handle_forget_fact(self, entities: dict) -> dict:
        key: str = entities.get("key", "").strip()
        if not key:
            return _ok("Which fact should I forget?", confidence=0.0)

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

        # Deduplicate by id, sorted by relevance score descending
        seen: set[str] = set()
        deduped = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["id"] not in seen:
                deduped.append(r)
                seen.add(r["id"])

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
            }
        except Exception:
            logger.exception("get_context() failed.")
            return {}

    def get_stats(self) -> dict:
        """Return aggregate statistics using SQL COUNT queries."""
        try:
            s = self.status()
            conn = self.db._conn

            total: int = conn.execute(
                "SELECT COUNT(*) FROM interaction_log"
            ).fetchone()[0]

            notes: int = conn.execute(
                "SELECT COUNT(*) FROM interaction_log WHERE intent = 'take_note'"
            ).fetchone()[0]

            unique_intents: int = conn.execute(
                "SELECT COUNT(DISTINCT intent) FROM interaction_log"
            ).fetchone()[0]

            return {
                "total_interactions": total,
                "notes": notes,
                "facts_known": s.get("facts", 0),
                "unique_intents": unique_intents,
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
            conn = self.db._conn
            facts: int = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            goals: int = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status = 'active'"
            ).fetchone()[0]
            summaries: int = conn.execute(
                "SELECT COUNT(*) FROM summaries"
            ).fetchone()[0]
            return {"facts": facts, "active_goals": goals, "summaries": summaries}
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