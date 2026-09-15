# modules/hecate/engine.py

from modules.base import BaseModule
from modules.hecate.intent_registry import (
    INTENT_MODULE_MAP,
    PREFIX_TO_MODULE,
    strip_module_prefix,
)


class HecateEngine(BaseModule):
    """
    Decision engine. The only component allowed to perform routing.
    Hestia calls decide() once per query. No module calls decide() on another module.

    Routing tiers, in order:
      1. Registry lookup       — any intent registered in intent_registry.py
                                  dispatches directly, O(1). This covers
                                  every module and intent Hestia has; it
                                  used to be ~15 separate hand-written
                                  `if intent in {...}` / `.startswith()`
                                  blocks here (one per module, added
                                  piecemeal over time), each a chance for a
                                  new intent to be forgotten or for tier
                                  ordering to shadow another tier's match.
                                  See intent_registry.py's module docstring
                                  for the drift that caused in practice.
      2. Text-trigger fallback — only reached when the NLU didn't return a
                                  registered intent (typically "chat", or
                                  something unrecognised). These compensate
                                  for classification misses in phrasing the
                                  small local model struggles with — that's
                                  a model-accuracy problem, not a routing-
                                  table problem, so it still needs explicit
                                  handling here.
      3. Prefix fallback       — defense-in-depth for an intent that still
                                  carries a real module's prefix but isn't
                                  in the registry (e.g. a hallucinated
                                  provider response, or a new module intent
                                  added to a module before its registry
                                  entry). Best-effort; the target module's
                                  own can_handle() has final say.
      4. Keyword / cross-module — Artemis keyword matching and
                                  compare/combine cross-module synthesis.
      5. Confidence-based       — generic high/low-confidence fallback to
                                  core chat.
    """
    name = "hecate"

    # --- Tier 2 data: raw-text triggers ---------------------------------
    # Moved verbatim from the original tiered implementation — these exist
    # specifically because the NLU intent can't be trusted for these
    # phrasings (it commonly falls back to "chat"), so matching happens on
    # the raw query text instead.
    _ATHENA_TRIGGERS = [
        "from my notes", "in my documents", "from my files",
        "according to my notes", "what does my", "explain from",
        "in my notes", "from my docs", "search my documents",
    ]
    _ATHENA_INGEST_TRIGGERS = [
        "ingest my documents", "ingest my notes", "ingest documents",
        "index my documents", "index my notes", "scan my documents",
        "reindex my documents", "update my document index",
    ]
    _MNEMOSYNE_TRIGGERS = [
        "do you remember", "what do you know about me",
        # NOT the bare phrase "remind me" — that also matches genuine
        # reminder-creation requests ("remind me to call mom tomorrow"),
        # which belong to Chronos's "set_reminder" intent (now caught by
        # the Tier-1 registry lookup before this tier ever runs). Only
        # match the interrogative "remind me what/who/..." form, which
        # really is a recall question.
        "remind me what", "remind me who", "remind me where",
        "remind me when", "remind me why", "remind me how",
        "what have i told you", "forget that",
        # "what did we talk about" intentionally excluded — see
        # _RECENCY_TRIGGERS below. Bare "what did we talk about" (no time
        # anchor) still falls through to Tier 5, which can legitimately be
        # semantic recall for a genuinely topical phrasing like "what did
        # we talk about regarding the Hestia project?".
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
        # otherwise silently fail every registry/dict lookup below, even
        # though a matching module exists.
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

        # --- Tier 1: Registry-driven direct dispatch ------------------
        # Replaces the old Tier 1 (Chronos/Hermes exact-match), Tier 1.5
        # (Athena/Iris exact-match), Tier X (apollo_/ares_/orpheus_/metis_/
        # dionysus_/pluto_/hephaestus_/hermes_ prefix routing), the
        # standalone Artemis exact-intent block, and the Mnemosyne
        # learn_fact/forget_fact block — all of it was doing the same
        # thing (intent -> module) from data that now lives in one place.
        #
        # "chat" is deliberately excluded here even though it's registered
        # to "core": it's the NLU's catch-all/low-confidence intent, not a
        # genuinely classified one, and it must still fall through to the
        # Tier 2 text-trigger checks below (e.g. "do you remember..." /
        # "from my notes..." routinely arrive as intent="chat" but need to
        # reach Mnemosyne/Athena instead of being short-circuited to core
        # here).
        module = INTENT_MODULE_MAP.get(intent)
        if module and module in active_modules and intent != "chat":
            return self._route(
                module, [], max(confidence, 0.9),
                f"registry: intent '{intent}' -> {module}",
                intent=strip_module_prefix(intent),
            )

        # --- Tier 2: Text trigger matching -----------------------------
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
            return self._route("core", [], 1.0, "recency trigger -> get_history", intent="get_history")

        if "mnemosyne" in active_modules and self._match(q, self._MNEMOSYNE_TRIGGERS):
            return self._route("mnemosyne", [], 1.0, "mnemosyne trigger", intent="recall")

        if "iris" in active_modules and self._match(q, self._IRIS_TRIGGERS):
            ingest_triggers = {"ingest media", "ingest photos"}
            iris_intent = "ingest" if q in ingest_triggers or any(
                t in q for t in ingest_triggers
            ) else "search"
            return self._route("iris", [], 1.0, "iris trigger", intent=iris_intent)

        # --- Tier 3: Prefix fallback (defense-in-depth) -----------------
        # If Tier 1 didn't recognise the intent but it still carries a real
        # module's prefix (e.g. a provider hallucinated "pluto_rebalance"
        # instead of the registered "pluto_optimize_portfolio", or a module
        # gained a new intent before intent_registry.py was updated for
        # it), give that module a chance via its own can_handle() rather
        # than silently falling back to chat. One loop replaces what used
        # to be seven near-identical `if intent.startswith("x_")` blocks.
        for prefix, prefixed_module in PREFIX_TO_MODULE.items():
            if intent.startswith(prefix) and prefixed_module in active_modules:
                return self._route(
                    prefixed_module, [], 0.75,
                    f"unregistered intent '{intent}' -> {prefixed_module} (prefix fallback)",
                )

        # --- Tier 4: Keyword matching ------------------------------------
        # Tier 1 already routes every real, registered Artemis intent
        # (add_habit, list_habits, productivity_summary, ...) directly, so
        # by the time control reaches here `intent` is guaranteed NOT to be
        # one of those — no exclusion list needed (the old version had to
        # explicitly exclude Artemis's own intents to avoid this tier
        # stealing them; that's now structurally impossible).
        if "artemis" in active_modules and any(k in q for k in self._ARTEMIS_KEYWORDS):
            return self._route("artemis", [], 0.9, "artemis keyword match")

        # --- Tier 4.5: Cross-module queries ------------------------------
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

        # --- Tier 5: High-confidence NLU non-chat intent -----------------
        if confidence >= 0.85 and intent != "chat":
            return self._route("core", [], confidence, f"high-confidence intent '{intent}'")

        # --- Tier 6: Low-confidence -> force chat ------------------------
        if confidence < 0.5:
            return self._route("core", [], 0.4, "low confidence -> chat fallback")

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

        This matters for text-trigger tiers (Tier 2 / Tier 4.5): those match
        on the raw query string, not on `nlu_result["intent"]`, precisely
        *because* the NLU intent is unreliable or absent for these phrasings
        (e.g. it commonly falls back to "chat"). If Hecate names `primary`
        without also correcting the intent, the orchestrator calls
        `primary.can_handle(<unreliable NLU intent>)`, that returns False,
        and the query gets silently rerouted to whichever OTHER module
        happens to declare that intent (usually "core", via its chat
        fallback) — discarding this routing decision entirely even though
        Hecate matched it with full confidence. Tiers whose match already
        comes from a trustworthy intent (Tier 1 / 3) don't need this; they
        leave `intent` unset and the NLU's own intent is used as-is.
        """
        return {
            "primary":    primary,
            "secondary":  secondary,
            "confidence": confidence,
            "reason":     reason,
            "synthesize": synthesize,
            "intent":     intent,
        }