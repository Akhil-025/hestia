"""
modules/mnemosyne/engine.py

MnemosyneEngine: unified entry point for memory, goals, and summarisation.
"""
import logging
from datetime import datetime
from typing import Optional
from modules.base import BaseModule  
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

class MnemosyneEngine(BaseModule):           
    name = "mnemosyne"

    _INTENTS = {
        "remember", "recall", "get_facts", "learn_fact",
        "forget_fact", "add_goal", "get_goals", "complete_goal",
    }
    
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

    def can_handle(self, intent: str) -> bool:    #
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:   
        query = (
            entities.get("query")
            or entities.get("raw_query")
            or context.get("raw_query", "")
        )
        if intent in ("remember", "recall"):
            response = self.remember(query)
            return {"response": response, "data": {}, "confidence": 0.85}
        elif intent == "learn_fact":
            key   = entities.get("key", "")
            value = entities.get("value", "")
            if key and value:
                self.learn(key, value)
                return {"response": f"Got it — I'll remember your {key}.", "data": {}, "confidence": 0.95}
            return {"response": "What should I remember?", "data": {}, "confidence": 0.0}
        elif intent == "forget_fact":
            key = entities.get("key", "")
            self.forget(key)
            return {"response": f"Forgotten: {key}.", "data": {}, "confidence": 0.9}
        elif intent == "add_goal":
            text = entities.get("goal") or entities.get("text", "")
            return {"response": self.add_goal(text), "data": {}, "confidence": 0.9}
        elif intent == "get_goals":
            return {"response": self.get_goals(), "data": {}, "confidence": 0.9}
        return {"response": "I don't have anything on that.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:               
        try:
            s = self.status()
            return {
                "mnemosyne_facts":   s.get("facts", 0),
                "mnemosyne_goals":   s.get("active_goals", 0),
                "mnemosyne_summaries": s.get("summaries", 0),
            }
        except Exception:
            return {}

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

    def add_interaction(self, q, r, i):
        self.memory.add_interaction(q, r, i)

    def get_recent(self, n=5):
        return self.memory.get_recent(n)

    def set_preference(self, k, v):
        self.memory.set_preference(k, v)

    def get_preference(self, k, d=None):
        return self.memory.get_preference(k, d)

    def get_all_preferences(self):
        return self.memory.get_all_preferences()
