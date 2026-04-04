# hecate/engine.py

class HecateEngine:
    def decide(self, query: str, nlu_result: dict, active_modules: list[str]) -> dict:
        # Decide which module(s) should handle the query
        q = query.lower().strip()
        intent = nlu_result.get("intent", "chat")
        confidence = nlu_result.get("confidence", 0.5)
        primary = "core"
        secondary = []
        reason = ""

        # Priority 1: NLU intent is web/search/browser
        if intent in {"search_web", "browser_action", "check_flight"}:
            primary = "core"
            reason = f"Intent '{intent}' is always handled by core."
            return {
                "primary": primary,
                "secondary": [],
                "confidence": 1.0,
                "reason": reason
            }

        # Priority 2: Athena triggers
        athena_patterns = [
            "from my notes", "in my documents", "from my files", "explain from",
            "in my notes", "from my docs", "search my documents"
        ]
        if "athena" in active_modules:
            for pat in athena_patterns:
                if pat in q:
                    primary = "athena"
                    reason = f"Matched Athena pattern: '{pat}'."
                    break

        # Priority 3: Mnemosyne triggers
        if primary == "core" and "mnemosyne" in active_modules:
            mnemosyne_patterns = [
                "do you remember", "what do you know about me", "what have i told you",
                "my goals", "what did we talk"
            ]
            for pat in mnemosyne_patterns:
                if pat in q:
                    primary = "mnemosyne"
                    reason = f"Matched Mnemosyne pattern: '{pat}'."
                    break


        # Artemis triggers
        if any(k in q for k in ["habit", "goal", "productivity", "streak"]):
            if "artemis" in active_modules:
                return {
                    "primary": "artemis",
                    "secondary": [],
                    "confidence": 0.9,
                    "reason": "Artemis keyword match."
                }

        # Priority 4: Iris triggers
        if primary == "core" and "iris" in active_modules:
            iris_patterns = [
                "photo", "picture", "image", "my photos", "my pictures", "find media", "ingest"
            ]
            for pat in iris_patterns:
                if pat in q:
                    primary = "iris"
                    reason = f"Matched Iris pattern: '{pat}'."
                    break

        # Priority 5: High-confidence NLU for a known non-chat intent
        if primary == "core" and confidence >= 0.85 and intent != "chat":
            reason = f"High-confidence NLU intent '{intent}' ({confidence:.2f}) routed to core."


        # Priority 6: Low-confidence → force chat
        if primary == "core" and confidence < 0.5:
            nlu_result["intent"] = "chat"
            reason = f"Low NLU confidence ({confidence:.2f}), falling back to chat."
            confidence = 0.4

        # Default
        if not reason:
            reason = f"Defaulted to core."

        # Secondary enrichment
        if primary == "core" and intent == "get_user_info" and "mnemosyne" in active_modules:
            secondary.append("mnemosyne")
        if primary == "athena" and "mnemosyne" in active_modules:
            secondary.append("mnemosyne")
        # Never add primary to secondary
        secondary = [m for m in secondary if m != primary]

        return {
            "primary": primary,
            "secondary": secondary,
            "confidence": float(confidence),
            "reason": reason
        }
