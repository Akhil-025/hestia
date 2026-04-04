"""
modules/mnemosyne/engine.py

MnemosyneEngine: unified entry point for memory, goals, and summarisation.
"""
import logging
from datetime import datetime
from typing import Optional

from .config import get_config
from .db import MnemosyneDB

logger = logging.getLogger(__name__)

try:
    from .vector_store import MnemosyneVectorStore
    chroma_available = True
except ImportError:
    logger.warning("ChromaDB or sentence-transformers not available; semantic memory disabled.")
    chroma_available = False

try:
    from .summariser import Summariser
    summariser_available = True
except ImportError:
    logger.warning("Summariser not available; summarisation disabled.")
    summariser_available = False

class MnemosyneEngine:
    def __init__(self, hestia_llm):
        self.config = get_config()
        self.db = MnemosyneDB(self.config.db_path)
        self.hestia_llm = hestia_llm
        if chroma_available:
            self.vector_store = MnemosyneVectorStore(self.config.chroma_dir, self.config.embedding_model)
        else:
            self.vector_store = None
        self.summariser = None
        if summariser_available:
            self.summariser = Summariser(self.db, self.vector_store, hestia_llm)

    def push(self, user_text: str, hestia_response: str, intent: str, source_device: str = "hestia") -> None:
        self.db.push_interaction(user_text, hestia_response, intent, source_device)
        if self.summariser and self.summariser.should_summarise():
            self.summariser.run()

    def remember(self, query: str, n: int = 5) -> str:
        if not self.vector_store:
            return ""
        # Search summaries
        summaries = self.vector_store.search(query, n_results=n, where={"type": "summary"}) if chroma_available else []
        # Search facts
        facts = self.vector_store.search(query, n_results=n, where={"type": "fact"}) if chroma_available else []
        results = summaries + facts
        if not results:
            return "I don't have any relevant memories for that."
        # Sort by score descending, deduplicate by id
        seen = set()
        sorted_results = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["id"] not in seen:
                sorted_results.append(r)
                seen.add(r["id"])
        # Format as natural language
        lines = []
        for r in sorted_results[:n]:
            meta = r["metadata"]
            text = r["text"]
            if meta.get("type") == "fact":
                key = meta.get("key", "")
                key_readable = key.replace("_", " ")
                lines.append(f"Your {key_readable} is {text}.")
            elif meta.get("type") == "summary":
                topic = meta.get("topic", "")
                prefix = f"Regarding {topic}: " if topic and topic != "General" else ""
                lines.append(f"{prefix}{text}")
            else:
                lines.append(text)

        if not lines:
            return "I don't have any relevant memories for that."
        return " ".join(lines)

    def learn(self, key: str, value: str, source: str = "user") -> None:
        self.db.set_fact(key, value, source)
        if self.vector_store:
            metadata = {"type": "fact", "created_at": datetime.utcnow().isoformat(), "key": key, "source": source}
            self.vector_store.add(value, metadata, doc_id=key)

    def forget(self, key: str) -> None:
        self.db.delete_fact(key)
        if self.vector_store:
            self.vector_store.delete(key)

    def add_goal(self, text: str, due_date=None) -> str:
        goal_id = self.db.add_goal(text, due_date)
        return f"Goal added (ID: {goal_id}): {text}"

    def get_goals(self, status: str = "active") -> str:
        goals = self.db.get_goals(status)
        if not goals:
            return "No goals found."
        lines = [f"[{g['id']}] {g['text']} (Due: {g['due_date'] or 'N/A'}) - {g['status']}" for g in goals]
        return "\n".join(lines)

    def complete_goal(self, goal_id: int) -> None:
        self.db.complete_goal(goal_id)

    def cancel_goal(self, goal_id: int) -> None:
        self.db.cancel_goal(goal_id)

    def status(self) -> dict:
        facts_count = len(self.db.get_all_facts())
        summaries_count = 0
        active_goals_count = len(self.db.get_goals("active"))
        unsummarised_count = len(self.db.get_unsummarised())
        if self.vector_store:
            try:
                summaries_count = len(self.vector_store.search("*", n_results=100, where={"type": "summary"}))
            except Exception:
                summaries_count = 0
        return {
            "facts": facts_count,
            "summaries": summaries_count,
            "active_goals": active_goals_count,
            "unsummarised_interactions": unsummarised_count,
        }
