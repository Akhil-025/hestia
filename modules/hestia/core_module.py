# modules/hestia/core_module.py

from modules.base import BaseModule
from core.actions import HestiaActions
from core.ollama_client import generate


class CoreModule(BaseModule):
    name = "core"

    _INTENTS = {
        "save_name", "take_note", "get_notes",
        "get_history", "get_user_info", "set_preference",
        "get_system_info", "chat"
    }

    def __init__(self, actions: HestiaActions, ollama_cfg: dict):
        self.actions = actions
        self.ollama_cfg = ollama_cfg

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        raw_query = entities.get("raw_query", "")

        if intent == "chat":
            try:
                response = generate(
                    f"You are Hestia. Answer concisely.\n\nQuestion: {raw_query}",
                    model=self.ollama_cfg.get("model", "mistral"),
                    host=self.ollama_cfg.get("host", "127.0.0.1"),
                    port=self.ollama_cfg.get("port", 11434),
                )
                return {"response": response, "data": {}, "confidence": 0.7}
            except Exception:
                return {"response": "I'm not sure about that.", "data": {}, "confidence": 0.3}

        result = self.actions.execute(intent, entities, raw_query)

        return {
            "response": result or "",
            "data": {},
            "confidence": 0.9 if result else 0.0,
        }

    def get_context(self) -> dict:
        return {}