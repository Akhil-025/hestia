# modules/hecate/engine.py  

from modules.base import BaseModule


class HecateEngine(BaseModule):
    """
    Decision engine. The only component allowed to perform routing.
    Hestia calls decide() once per query. No module calls decide() on another module.
    """
    name = "hecate"

    # Moved verbatim from main.py — single source of truth for trigger matching
    _ATHENA_TRIGGERS = [
        "from my notes", "in my documents", "from my files",
        "according to my notes", "what does my", "explain from",
        "in my notes", "from my docs", "search my documents",
    ]
    _MNEMOSYNE_TRIGGERS = [
        "do you remember", "what do you know about me",
        "what are my goals", "remind me", "what did we talk about",
        "what have i told you", "my goals", "forget that",
    ]
    _IRIS_TRIGGERS = [
        "in my photos", "in my pictures", "in my images", "in my videos",
        "in my media", "in my gallery", "from my photos", "from my pictures",
        "find photo", "find image", "find video", "find picture",
        "search my photos", "analyse my photos", "ingest media",
        "ingest photos", "describe my photos",
    ]
    _ARTEMIS_KEYWORDS = {"habit", "goal", "productivity", "streak"}

    _CHRONOS_INTENTS  = {"get_time", "get_date", "get_weather", "set_reminder"}
    _HERMES_INTENTS   = {"read_email", "send_email", "list_events", "create_event"}
    _HEPHAESTUS_INTENTS = {"search_web", "browser_action", "check_flight"}

    def can_handle(self, intent: str) -> bool:
        return True  # Hecate is consulted for all routing; it does not handle content

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        raise NotImplementedError(
            "HecateEngine.handle() must not be called directly. Use decide()."
        )

    def get_context(self) -> dict:
        return {}

    def decide(self, query: str, nlu_result: dict, active_modules: list) -> dict:
        """
        Single routing decision. Returns:
            {
                "primary":    str,        # module name to dispatch to
                "secondary":  list[str],  # modules to call get_context() on first
                "confidence": float,
                "reason":     str,
            }
        """
        q          = query.lower().strip()
        intent     = nlu_result.get("intent", "chat")
        confidence = float(nlu_result.get("confidence", 0.5))

        # --- Tier 1: Hard-wired by intent class (no ambiguity) ---
        if intent in self._CHRONOS_INTENTS and "chronos" in active_modules:
            return self._route("chronos", [], 1.0, f"intent '{intent}' → chronos")

        if intent in self._HERMES_INTENTS and "hermes" in active_modules:
            return self._route("hermes", [], 1.0, f"intent '{intent}' → hermes")

        if intent in self._HEPHAESTUS_INTENTS and "hephaestus" in active_modules:
            return self._route("hephaestus", [], 1.0, f"intent '{intent}' → hephaestus")

        # --- Tier 2: Text trigger matching ---
        if "athena" in active_modules and self._match(q, self._ATHENA_TRIGGERS):
            return self._route("athena", ["mnemosyne"] if "mnemosyne" in active_modules else [], 1.0, "athena trigger")

        if "mnemosyne" in active_modules and self._match(q, self._MNEMOSYNE_TRIGGERS):
            return self._route("mnemosyne", [], 1.0, "mnemosyne trigger")

        if "iris" in active_modules and self._match(q, self._IRIS_TRIGGERS):
            return self._route("iris", [], 1.0, "iris trigger")

        # --- Tier 3: Keyword matching ---
        if (
            "artemis" in active_modules
            and any(k in q for k in self._ARTEMIS_KEYWORDS)
            and intent not in {"add_goal", "get_goals"}
        ):
            return self._route("artemis", [], 0.9, "artemis keyword match")
        
        # --- Tier X: New module routing ---

        if intent.startswith("apollo_") and "apollo" in active_modules:
            return self._route("apollo", [], 0.95, f"intent '{intent}' → apollo")

        if intent.startswith("ares_") and "ares" in active_modules:
            return self._route("ares", [], 0.95, f"intent '{intent}' → ares")

        if intent.startswith("orpheus_") and "orpheus" in active_modules:
            return self._route("orpheus", [], 0.95, f"intent '{intent}' → orpheus")

        if intent.startswith("dionysus_") and "dionysus" in active_modules:
            return self._route("dionysus", [], 0.95, f"intent '{intent}' → dionysus")

        if intent.startswith("pluto_") and "pluto" in active_modules:
            return self._route("pluto", [], 0.95, f"intent '{intent}' → pluto")
        

        # --- IRIS ROUTING FIX ---
        if intent in {
            "iris_search",
            "iris_ingest",
            "iris_analyse",
            "iris_query",
            "iris_status"
        } and "iris" in active_modules:
            return self._route("iris", [], 0.95, f"intent '{intent}' → iris")
        
        # --- MNEMOSYNE ROUTING FIX ---
        if intent in {"get_user_info", "learn_fact", "forget_fact", "add_goal", "get_goals"} \
                and "mnemosyne" in active_modules:
            return self._route("mnemosyne", [], 0.95, f"intent '{intent}' → mnemosyne")

        # --- Tier 4: High-confidence NLU non-chat intent ---
        if confidence >= 0.85 and intent != "chat":
            return self._route("core", [], confidence, f"high-confidence intent '{intent}'")

        # --- Tier 5: Low-confidence → force chat ---
        if confidence < 0.5:
            return self._route("core", [], 0.4, "low confidence → chat fallback")

        return self._route("core", [], confidence, "default core")

    @staticmethod
    def _match(text: str, triggers: list) -> bool:
        import re
        return any(re.search(r"\b" + re.escape(t) + r"\b", text) for t in triggers)

    @staticmethod
    def _route(primary: str, secondary: list, confidence: float, reason: str) -> dict:
        return {
            "primary":    primary,
            "secondary":  secondary,
            "confidence": confidence,
            "reason":     reason,
        }