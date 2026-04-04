
# core/actions.py

from core.browser_agent import HestiaBrowserAgent
import datetime as _datetime
import re
import requests
from core.memory import HestiaMemory
from core.event_bus import bus

class HestiaActions:
    """Execute structured intents from NLU using memory and external APIs."""
    

    def __init__(self, memory: HestiaMemory):
        """Store memory reference and event bus."""
        self.memory = memory
        self._google = None
        self._browser = None
    def set_browser_agent(self, agent: "HestiaBrowserAgent") -> None:
        """Inject browser agent reference."""
        self._browser = agent

    def execute(self, intent: str, entities: dict, raw_query: str = "") -> str:

        result = None

        if intent == "save_name":
            result = self._save_name(entities)

        elif intent == "take_note":
            result = self._take_note(entities, raw_query)

        elif intent == "get_notes":
            result = self._get_notes()

        elif intent == "get_history":
            result = self._get_history(entities)

        elif intent == "get_user_info":
            result = self._get_user_info(entities)

        elif intent == "set_preference":
            result = self._set_preference(entities)

        elif intent == "chat":
            result = ""

        else:
            result = None

        if result:
            bus.emit("action_done", {"intent": intent, "response": result})

        return result or "I'm not sure how to help with that yet."

    def set_google_agent(self, agent) -> None:
        """Inject Google agent reference after initialisation."""
        self._google = agent

    def _save_name(self, entities: dict) -> str:
        """Save user's name to preferences."""
        name = entities.get("name", "").strip()
        if not name:
            return "I didn't catch your name, could you repeat it?"
        name = name.title()
        self.memory.set_preference("user_name", name)
        return f"Got it! I'll remember you as {name}."


    def _take_note(self, entities: dict, raw_query: str = "") -> str:
        """Save a note to memory."""
        note_text = entities.get("content") or entities.get("text") or entities.get("task") or entities.get("note", "")

        if not note_text and raw_query:
            cleaned = re.sub(r'^(take a note|note down|jot down|remember|note)\s*[:\-]?\s*',
                             '', raw_query, flags=re.IGNORECASE).strip()
            note_text = cleaned

        if not note_text:
            return "What would you like me to note down?"

        self.memory.add_interaction(note_text, "Note saved.", "take_note")
        return "Note saved."

    def _get_notes(self) -> str:
        """Retrieve recent notes from memory."""
        notes = self.memory.get_by_intent("take_note", limit=10)
        if not notes:
            return "You don't have any notes saved yet."

        result = "Here are your recent notes:\n"
        for item in notes:
            result += f"- {item['query']}\n"
        return result.strip()

    def _get_history(self, entities: dict) -> str:
        """Retrieve recent conversation history, excluding notes and reminders."""
        limit = int(entities.get("limit", 5))
        recent = self.memory.get_recent_filtered(limit, exclude_intents=["take_note", "set_reminder"])
        if not recent:
            return "We haven't talked much yet."

        result = "Here's what we talked about:\n"
        for item in recent:
            result += f"- {item['query']}\n"
        return result.strip()

    def _get_user_info(self, entities: dict) -> str:
        """Return all stored user preferences."""
        prefs = self.memory.get_all_preferences()
        if not prefs:
            return "I don't know much about you yet."

        result = "Here's what I know about you:\n"
        for key, value in prefs.items():
            formatted_key = key.replace("_", " ").capitalize()
            result += f"- {formatted_key}: {value}\n"
        return result.strip()

    def _set_preference(self, entities: dict) -> str:
        """Store a user preference."""
        key = entities.get("key", "")
        value = entities.get("value", "")

        if not value:
            return "I didn't quite catch what you want me to remember."

        if not key or key == "preference":
            words = value.strip().split()
            key = "_".join(words[:3]).lower() if words else "preference"

        self.memory.set_preference(key, value)
        return "Got it, I'll remember that."