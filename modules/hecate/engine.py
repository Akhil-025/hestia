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
    # Mirrors _IRIS_TRIGGERS' ingest/search split below — without a
    # dedicated ingest trigger, Athena had no voice/chat-reachable way to
    # index new documents at all (only the private engine._ingest() method,
    # never wired to anything a user could actually say).
    _ATHENA_INGEST_TRIGGERS = [
        "ingest my documents", "ingest my notes", "ingest documents",
        "index my documents", "index my notes", "scan my documents",
        "reindex my documents", "update my document index",
    ]
    _MNEMOSYNE_TRIGGERS = [
        "do you remember", "what do you know about me",
        # NOT the bare phrase "remind me" — that also matches genuine
        # reminder-creation requests ("remind me to call mom tomorrow"),
        # which belong to Chronos's "set_reminder" intent. A bare match
        # here stole every such request the NLU classified as "chat"
        # (Chronos has no Tier-2 text-trigger fallback of its own to
        # recover it), silently turning "remind me to X" into a semantic
        # memory search that always answered "I don't have any memories
        # about that yet." instead of creating the reminder. Only match
        # the interrogative "remind me what/who/..." form, which really
        # is a recall question.
        "remind me what", "remind me who", "remind me where",
        "remind me when", "remind me why", "remind me how",
        "what have i told you", "forget that",
        # "what did we talk about" intentionally removed from here — see
        # _RECENCY_TRIGGERS below. It used to live in this list and route
        # straight to Mnemosyne's semantic vector recall, but "what did we
        # talk about yesterday/today/earlier" is a chronological request,
        # not a topical one: it wants the last N raw interactions in order,
        # not "whatever weakly matches this embedding". Bare "what did we
        # talk about" (no time anchor) still falls through Tier 2/3/4 to
        # the NLU-classified intent, which can legitimately be semantic
        # recall for a genuinely topical phrasing like "what did we talk
        # about regarding the Hestia project?".
    ]
    # Chronological phrasing — routes to Core's get_history (plain SQL read
    # of the last N interactions, no LLM/embedding involved) instead of
    # Mnemosyne's semantic recall. Checked before _MNEMOSYNE_TRIGGERS below.
    _RECENCY_TRIGGERS = [
        "what did we talk about yesterday", "what did we talk about today",
        "what did we talk about earlier", "what did we talk about recently",
        "what did we just talk about", "what did we discuss yesterday",
        "what did we discuss today", "what did we discuss earlier",
    ]
    _IRIS_TRIGGERS = [
        "in my photos", "in my pictures", "in my images", "in my videos",
        "in my media", "in my gallery", "from my photos", "from my pictures",
        "find photo", "find image", "find video", "find picture",
        "search my photos", "analyse my photos", "ingest media",
        "ingest photos", "describe my photos",
    ]
    _ARTEMIS_KEYWORDS = {"habit", "goal", "productivity", "streak", "motivate", "motivation"}

    _CHRONOS_INTENTS  = {"get_time", "get_date", "get_weather", "set_reminder"}
    # NOTE: the NLU is inconsistent about emitting these with or without the
    # module prefix (e.g. both "read_email" and "hermes_read_email" have been
    # observed for the same query, and both "search_web" and
    # "hephaestus_search_web"). Both spellings are listed here so Tier 1
    # matches regardless of which form comes back, instead of silently
    # falling through to Tier 4 -> "core" and relying on the orchestrator's
    # can_handle() recovery path to bail us out.
    _HERMES_INTENTS = {
        "read_email", "send_email", "list_events", "create_event", "delete_events",
        "hermes_read_email", "hermes_send_email", "hermes_list_events",
        "hermes_create_event", "hermes_delete_events",
    }

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

        # Normalise case defensively — the NLU has been observed to emit
        # intents in unexpected casing (e.g. "PLATFORM_ACTION") which would
        # otherwise silently fail every `intent in {...}` / `.startswith()`
        # check below and fall through to chat, even though a matching
        # module exists.
        intent = intent.strip().lower() if isinstance(intent, str) else "chat"

        # Some NLU responses wrap the real intent inside a generic envelope
        # instead of emitting it directly, e.g.
        #   {"intent": "PLATFORM_ACTION", "entities": {"action": "LOG_EXPENSE", ...}}
        # instead of {"intent": "log_expense", "entities": {...}}.
        # Unwrap it so every tier below operates on the actual intent
        # ("log_expense") rather than the meaningless envelope name.
        if intent == "platform_action":
            inner = (nlu_result.get("entities") or {}).get("action")
            if isinstance(inner, str) and inner.strip():
                intent = inner.strip().lower()

        # --- Tier 1: Hard-wired by intent class (no ambiguity) ---
        if intent in self._CHRONOS_INTENTS and "chronos" in active_modules:
            return self._route("chronos", [], 1.0, f"intent '{intent}' → chronos")

        if intent in self._HERMES_INTENTS and "hermes" in active_modules:
            return self._route("hermes", [], 1.0, f"intent '{intent}' → hermes")

        # Hephaestus has no exact-match set here (unlike Hermes/Chronos) —
        # every intent it declares is prefixed ("hephaestus_browser_action",
        # "hephaestus_scrape_page", "hephaestus_open_app", ...), so the
        # "hephaestus_" prefix check in Tier X below is sufficient and never
        # needs a manual entry added for new Hephaestus intents.
        

        # --- Tier 1.5: Direct intent routing (NEW — CRITICAL) ---

        if intent == "athena_search" and "athena" in active_modules:
            return self._route("athena", [], 1.0, "intent 'athena_search' → athena")

        if intent == "athena_ingest" and "athena" in active_modules:
            return self._route("athena", [], 1.0, "intent 'athena_ingest' → athena", intent="ingest")

        if intent in {
            "iris_search",
            "iris_ingest",
            "iris_analyse",
            "iris_query",
            "iris_status"
        } and "iris" in active_modules:
            return self._route("iris", [], 1.0, f"intent '{intent}' → iris")

        # --- Tier 2: Text trigger matching ---
        # These match on the raw query text specifically because the NLU
        # intent can't be trusted for these phrasings (often "chat"). Pass
        # an explicit `intent` override — see `_route()` docstring — so the
        # orchestrator actually dispatches to the matched module instead of
        # calling can_handle() with an intent it doesn't declare.
        #
        # Ingest check comes first: "ingest my documents" would otherwise
        # never be reachable if a broader athena-search trigger happened to
        # overlap it (same ordering Iris uses for its ingest/search split).
        if "athena" in active_modules and self._match(q, self._ATHENA_INGEST_TRIGGERS):
            return self._route("athena", [], 1.0, "athena ingest trigger", intent="ingest")

        if "athena" in active_modules and self._match(q, self._ATHENA_TRIGGERS):
            return self._route(
                "athena", ["mnemosyne"] if "mnemosyne" in active_modules else [],
                1.0, "athena trigger", intent="search",
            )

        # Recency-anchored phrasing ("...yesterday/today/earlier") wants
        # chronological history, not semantic search — route to Core's
        # get_history before the Mnemosyne trigger check below gets a
        # chance to send it to vector recall instead.
        if "core" in active_modules and self._match(q, self._RECENCY_TRIGGERS):
            return self._route("core", [], 1.0, "recency trigger → get_history", intent="get_history")

        if "mnemosyne" in active_modules and self._match(q, self._MNEMOSYNE_TRIGGERS):
            return self._route("mnemosyne", [], 1.0, "mnemosyne trigger", intent="recall")

        if "iris" in active_modules and self._match(q, self._IRIS_TRIGGERS):
            ingest_triggers = {"ingest media", "ingest photos"}
            iris_intent = "ingest" if q in ingest_triggers or any(
                t in q for t in ingest_triggers
            ) else "search"
            return self._route("iris", [], 1.0, "iris trigger", intent=iris_intent)

        # --- Tier 3: Keyword matching ---
        if (
            "artemis" in active_modules
            and any(k in q for k in self._ARTEMIS_KEYWORDS)
            and intent not in {
                "add_goal", "get_goals", "list_goals", "update_goal",
                "remove_goal", "abandon_goal", "get_at_risk_goals",
                "add_habit", "complete_habit", "list_habits", "remove_habit",
                "productivity_summary", "get_motivation",
            }
        ):
            return self._route("artemis", [], 0.9, "artemis keyword match")
        
        # --- Tier X: New module routing ---

        if intent.startswith("apollo_") and "apollo" in active_modules:
            return self._route("apollo", [], 0.95, f"intent '{intent}' → apollo")

        if intent.startswith("ares_") and "ares" in active_modules:
            return self._route("ares", [], 0.95, f"intent '{intent}' → ares")

        if intent.startswith("orpheus_") and "orpheus" in active_modules:
            return self._route("orpheus", [], 0.95, f"intent '{intent}' → orpheus")

        if intent.startswith("metis_") and "metis" in active_modules:
            return self._route("metis", [], 0.95, f"intent '{intent}' → metis")

        if intent.startswith("dionysus_") and "dionysus" in active_modules:
            return self._route("dionysus", [], 0.95, f"intent '{intent}' → dionysus")

        if intent.startswith("pluto_") and "pluto" in active_modules:
            return self._route("pluto", [], 0.95, f"intent '{intent}' → pluto")

        if intent.startswith("hephaestus_") and "hephaestus" in active_modules:
            return self._route("hephaestus", [], 0.95, f"intent '{intent}' → hephaestus")

        # _HERMES_INTENTS above only covers names the NLU has been
        # *observed* to emit (e.g. "hermes_read_email"). Any other
        # "hermes_*" name (e.g. "hermes_gmail" for "check my mail") fell
        # through every tier to Tier 4 → "core", which doesn't declare it,
        # and the orchestrator's can_handle() recovery only searches by the
        # *stripped* intent — so it never tried Hermes at all and silently
        # fell back to chat. Route by prefix here too, same as the modules
        # above, and let the orchestrator's own can_handle()/alias handling
        # sort out the exact intent name once inside the module.
        if intent.startswith("hermes_") and "hermes" in active_modules:
            return self._route("hermes", [], 0.9, f"intent '{intent}' → hermes (prefix)")
        

        if intent in {
            "add_goal", "get_goals", "list_goals", "update_goal",
            "remove_goal", "abandon_goal", "get_at_risk_goals",
            "add_habit", "complete_habit", "list_habits", "remove_habit",
            "productivity_summary", "get_motivation",
        } and "artemis" in active_modules:
            return self._route("artemis", [], 0.95, f"intent '{intent}' → artemis")
        
        # --- MNEMOSYNE ROUTING FIX ---
        # NOTE: "get_user_info" is deliberately NOT included here. Core also
        # declares "get_user_info" and its handler is a strict superset of
        # Mnemosyne's: it forwards to the same underlying fact store
        # (self._memory.db.get_fact) *and* special-cases the NLU's frequent
        # get_user_info misclassification of "what's today's date"/"what
        # time is it" (key="current_date"/"current_time") by answering
        # directly instead of doing a doomed fact lookup. Routing
        # get_user_info straight to Mnemosyne here bypassed that recovery
        # logic entirely — Mnemosyne has no such fallback, so those queries
        # got "I don't have that information yet." instead of the actual
        # date/time. Only "learn_fact"/"forget_fact" are Mnemosyne-only
        # (Core never declares them), so only those need a forced route.
        if intent in {"learn_fact", "forget_fact"} \
                and "mnemosyne" in active_modules:
            return self._route("mnemosyne", [], 0.95, f"intent '{intent}' → mnemosyne")
        

        # --- Tier 3.5: Cross-module queries ---
        cross_triggers = [
            "compare", "combine", "across", "and also",
            "along with", "together with", "as well as"
        ]
        if "athena" in active_modules and "mnemosyne" in active_modules:
            if self._match(q, cross_triggers) or (
                self._match(q, self._ATHENA_TRIGGERS) and
                self._match(q, self._MNEMOSYNE_TRIGGERS)
            ):
                return self._route(
                    "athena",
                    ["mnemosyne"],
                    0.85,
                    "cross-module: athena+mnemosyne",
                    synthesize=True,
                    intent="search",
                )

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
    def _route(primary: str, secondary: list, confidence: float,
            reason: str, synthesize: bool = False, intent: str = None) -> dict:
        """
        `intent`, when set, OVERRIDES the intent the orchestrator dispatches
        to `primary` with (instead of the NLU's raw intent, stripped of its
        module prefix).

        This matters for text-trigger tiers (Tier 2 / Tier 3.5): those match
        on the raw query string, not on `nlu_result["intent"]`, precisely
        *because* the NLU intent is unreliable or absent for these phrasings
        (e.g. it commonly falls back to "chat"). If Hecate names `primary`
        without also correcting the intent, the orchestrator calls
        `primary.can_handle(<unreliable NLU intent>)`, that returns False,
        and the query gets silently rerouted to whichever OTHER module
        happens to declare that intent (usually "core", via its chat
        fallback) — discarding this routing decision entirely even though
        Hecate matched it with full confidence. Tiers whose match already
        comes from a trustworthy intent (Tier 1 / 1.5 / X) don't need this;
        they leave `intent` unset and the NLU's own intent is used as-is.
        """
        return {
            "primary":    primary,
            "secondary":  secondary,
            "confidence": confidence,
            "reason":     reason,
            "synthesize": synthesize,
            "intent":     intent,
        }