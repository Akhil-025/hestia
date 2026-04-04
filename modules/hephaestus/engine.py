# modules/hephaestus/engine.py

from modules.base import BaseModule


class HephaestusEngine(BaseModule):
    name = "hephaestus"

    _INTENTS = {"browser_action", "search_web", "check_flight"}

    def __init__(self, browser_agent=None):
        self._browser = browser_agent  # HestiaBrowserAgent, injected by orchestrator

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        if not self._browser:
            return {"response": "Browser automation is not enabled.", "data": {}, "confidence": 0.0}

        action = entities.get("action", "").lower()
        url    = entities.get("url", "")
        query  = entities.get("query") or entities.get("topic") or entities.get("raw_query", "")
        flight = entities.get("flight_number", "")

        if intent == "check_flight" or flight:
            result = self._browser.check_flight_status(flight)
            return {"response": result, "data": {}, "confidence": 0.9}

        if url and action in ("open", "browse", "navigate", "go to", ""):
            result = self._browser.open_url(url)
            return {"response": result, "data": {}, "confidence": 0.9}

        if query:
            result = self._browser.search_web(query)
            return {"response": result, "data": {}, "confidence": 0.85}

        return {
            "response":   "I'm not sure what browser action to take.",
            "data":       {},
            "confidence": 0.0,
        }

    def get_context(self) -> dict:
        return {"hephaestus_available": self._browser is not None}