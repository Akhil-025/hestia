# modules/pluto/engine.py

from modules.base import BaseModule

class PlutoEngine(BaseModule):
    name = "pluto"
    _INTENTS = {
        "log_expense",
        "get_budget_summary",
        "track_investment",
        "spending_report"
    }

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        return {
            "response": "Financial tracking is coming soon.",
            "data": {},
            "confidence": 0.0
        }

    def get_context(self) -> dict:
        return {}