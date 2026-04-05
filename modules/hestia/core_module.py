# modules/hestia/core_module.py

from modules.base import BaseModule
from core.ollama_client import generate
import platform, datetime, re


class CoreModule(BaseModule):
    name = "core"

    _INTENTS = {
        "save_name", "take_note", "get_notes",
        "get_history", "set_preference",
        "get_system_info", "chat"
    }

    def __init__(self, memory, ollama_cfg: dict):
        self._memory = memory
        self._ollama = ollama_cfg

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        raw = entities.get("raw_query", "")

        if intent == "chat":
            return self._chat(raw)

        if intent == "get_system_info":
            return self._sys_info()

        if intent == "save_name":
            return self._save_name(entities)

        if intent == "take_note":
            return self._take_note(entities, raw)

        if intent == "get_notes":
            return self._get_notes()

        if intent == "get_history":
            return self._get_history(entities)

        if intent == "get_user_info":
            return self._get_user_info()

        if intent == "set_preference":
            return self._set_preference(entities)

        return {"response": "", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        return {}

    # ───── handlers ─────

    def _chat(self, query: str) -> dict:
        try:
            text = generate(
                f"You are Hestia. Answer concisely in 1-2 sentences.\n\nQuestion: {query}",
                model=self._ollama.get("model", "mistral"),
                host=self._ollama.get("host", "127.0.0.1"),
                port=self._ollama.get("port", 11434),
            )
            return {"response": text, "data": {}, "confidence": 0.7}
        except Exception:
            return {"response": "I'm not sure about that.", "data": {}, "confidence": 0.3}

    def _sys_info(self) -> dict:
        return {
            "response": (
                f"Running on {platform.system()} {platform.release()}, "
                f"Python {platform.python_version()}, "
                f"time {datetime.datetime.now().strftime('%I:%M %p')}."
            ),
            "data": {},
            "confidence": 1.0,
        }

    def _save_name(self, entities: dict) -> dict:
        name = entities.get("name", "").strip().title()
        if not name:
            return {"response": "I didn't catch your name.", "data": {}, "confidence": 0.0}

        self._memory.learn("user_name", name)
        return {"response": f"Got it! I'll remember you as {name}.", "data": {}, "confidence": 0.9}

    def _take_note(self, entities: dict, raw: str) -> dict:
        note = (
            entities.get("content")
            or entities.get("text")
            or entities.get("task")
            or ""
        )

        if not note and raw:
            note = re.sub(
                r'^(take a note|note down|jot down|remember|note)\s*[:\-]?\s*',
                '',
                raw,
                flags=re.IGNORECASE
            ).strip()

        if not note:
            return {"response": "What would you like me to note down?", "data": {}, "confidence": 0.0}
        
        timestamp = datetime.datetime.utcnow().isoformat()
        note_key = f"note_{timestamp}"

        self._memory.learn(note_key, note)

        return {
            "response": "Note saved.",
            "data": {"note": note},
            "confidence": 0.95
        }
    
    def _get_notes(self) -> dict:
        rows = self._memory.db.get_by_intent("take_note", 10)
        notes = [{"query": r["query"], "response": r["response"], "intent": r["intent"]} for r in rows]

        if not notes:
            return {"response": "No notes saved yet.", "data": {}, "confidence": 0.9}

        body = "Your notes:\n" + "\n".join(f"- {n['query']}" for n in notes)

        return {"response": body, "data": {"notes": notes}, "confidence": 0.9}

    def _get_history(self, entities: dict) -> dict:
        limit = int(entities.get("limit", 5))

        rows = self._memory.db.get_recent_interactions(limit * 2)

        recent = [
            r for r in rows
            if r["intent"] not in ["take_note", "set_reminder"]
        ][:limit]

        recent = [{"query": r["query"], "response": r["response"], "intent": r["intent"]} for r in recent]

        if not recent:
            return {"response": "We haven't talked much yet.", "data": {}, "confidence": 0.9}

        body = "Here's what we talked about:\n" + "\n".join(
            f"- {r['query']}" for r in recent
        )

        return {"response": body, "data": {"history": recent}, "confidence": 0.9}


    def _set_preference(self, entities: dict) -> dict:
        key = entities.get("key", "")
        value = entities.get("value", "")

        if not value:
            return {"response": "What should I remember?", "data": {}, "confidence": 0.0}

        if not key or key == "preference":
            key = "_".join(value.strip().split()[:3]).lower()

        self._memory.learn(key, value)

        return {"response": "Got it, I'll remember that.", "data": {}, "confidence": 0.95}