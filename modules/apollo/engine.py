# modules/apollo/engine.py

from modules.base import BaseModule

class ApolloEngine(BaseModule):
    name = "apollo"
    _INTENTS = {"log_health", "get_health_summary", "track_sleep", "log_workout", "log_mood"}

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        return {"response": "Health tracking is coming soon.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        return {}