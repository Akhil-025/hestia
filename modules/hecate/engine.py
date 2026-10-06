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

    Two additions sit alongside the tiers:

      * Conference (backlog #158) — a ``conference`` intent is not routed to
        one module. Hecate picks 2–3 modules whose data bears on the topic
        and marks the decision with ``conference=[...]``; the orchestrator
        then gathers each module's perspective and merges them. Hecate
        still only *decides*; she never calls a module.
      * Audit trail (backlog #162) — every decision carries ``checked``, the
        ordered list of what was examined and what each check concluded,
        so "what did you check before answering that?" has a real answer
        rather than a reconstruction.
    """
    name = "hecate"

    # Below this, a registered-but-uncertain intent is not executed blind
    # (backlog #2). Tier 1 below would otherwise dispatch ANY registered
    # intent regardless of the NLU's own confidence — a single misheard
    # word turning into a real "send_email" or "pluto_log_expense" call
    # with no chance to double-check. Deliberately well below Hecate's own
    # 0.85 "high confidence" bar (Tier 5) and just above the "force chat"
    # floor (Tier 6, 0.5): this is specifically the band where the NLU
    # recognised *something* concrete but wasn't sure, which is exactly
    # the case a clarifying question serves — a genuinely low-confidence
    # "chat" guess is already routed to plain chat by Tier 6 and needs no
    # special handling here.
    _CLARIFY_CONFIDENCE_THRESHOLD = 0.45

    # Trained-classifier hook (backlog #4). In "primary" mode the classifier
    # gets a say before the hand-written text triggers whenever the NLU was
    # unsure or said "chat"; in "assist" mode only after every tier has failed.
    # Below this NLU confidence the NLU is treated as "unsure".
    _CLASSIFIER_UNSURE_BELOW = 0.6

    def __init__(self, classifier=None) -> None:
        self._classifier = classifier

    def attach_classifier(self, classifier) -> None:
        """Install (or clear, with None) a ``ClassifierService``."""
        self._classifier = classifier

    def _classifier_pick(self, query: str, active_modules: list, trace: list):
        """A (intent, probability) the classifier is confident about for a
        registered intent whose module is active, else None. Never raises."""
        clf = self._classifier
        if clf is None or not getattr(clf, "enabled", False):
            return None
        try:
            pred = clf.classify(query)
        except Exception:
            return None
        if pred is None:
            trace.append("classifier: not confident enough to answer")
            return None
        module = INTENT_MODULE_MAP.get(pred.intent)
        if not module or module not in active_modules or pred.intent == "chat":
            trace.append(f"classifier: suggested '{pred.intent}' but it can't be dispatched here")
            return None
        return pred.intent, pred.probability

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

    # --- Conference data (backlog #158) ----------------------------------
    # Which modules have something to say about a topic. Matching is on whole
    # words in the query; modules are ranked by how many topic words hit and
    # at most _CONFERENCE_MAX_MODULES take part. Only modules that are
    # actually active can be seated, and a conference needs at least two —
    # one module's view is just a normal answer.
    _CONFERENCE_TOPICS: tuple = (
        ("money", frozenset({
            "money", "spend", "spending", "budget", "invest", "investing",
            "investment", "salary", "loan", "debt", "savings", "save",
            "expense", "expenses", "finance", "finances", "financial",
            "afford", "buy", "purchase", "subscription", "income", "rent",
        }), ("pluto", "ares")),
        ("health", frozenset({
            "sleep", "burnout", "tired", "exhausted", "workout", "exercise",
            "health", "rest", "stress", "stressed", "energy", "mood",
            "overwhelmed",
        }), ("apollo", "artemis")),
        ("career and study", frozenset({
            "exam", "exams", "gate", "study", "studying", "career", "job",
            "interview", "deadline", "project", "course", "degree", "quit",
            "resign", "offer",
        }), ("ares", "artemis", "chronos")),
        ("habits and goals", frozenset({
            "habit", "habits", "streak", "routine", "goal", "goals",
            "productivity", "consistency",
        }), ("artemis", "apollo")),
    )
    _CONFERENCE_MAX_MODULES = 3

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
                "synthesize": bool,
                "intent":     str | None, # overrides the NLU intent when set
                "conference": list[str] | None,  # modules seated (#158)
                "checked":    list[str],  # audit trail, in order (#162)
            }
        """
        trace: list = []
        decision = self._decide(query, nlu_result, active_modules, trace)
        decision["checked"] = trace
        return decision

    def _decide(self, query: str, nlu_result: dict, active_modules: list,
                trace: list) -> dict:
        """The tiered decision itself. Appends one line to *trace* for each
        thing it examines, whether or not that check ended up deciding."""
        q          = query.lower().strip()
        intent     = nlu_result.get("intent", "chat")
        # A missing, None, non-numeric or NaN confidence all mean "unknown", the
        # same 0.5 the key's absence has always meant, rather than a crash.
        try:
            confidence = float(nlu_result.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        if confidence != confidence:
            confidence = 0.5

        # Normalise case defensively — the NLU has been observed to emit
        # intents in unexpected casing (e.g. "PLATFORM_ACTION") which would
        # otherwise silently fail every registry/dict lookup below, even
        # though a matching module exists.
        intent = intent.strip().lower() if isinstance(intent, str) else "chat"
        nlu_intent = intent

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
                trace.append(
                    f"unwrapped the generic '{nlu_intent}' envelope to the real intent '{intent}'"
                )

        trace.append(f"NLU reported intent '{intent}' at {confidence:.0%} confidence")

        # --- Tier 0.5: low-confidence clarification (backlog #2) -------
        # Runs before Tier 1 on purpose: Tier 1 dispatches any REGISTERED
        # intent unconditionally (it exists to guarantee reachability, not
        # to gate on confidence), so this has to intercept first or it
        # never fires for exactly the intents it's meant to protect —
        # recognised-but-uncertain ones. "chat" is excluded because a
        # low-confidence "chat" guess already has a correct home (plain
        # conversation, via Tier 6 below); there's nothing concrete to
        # confirm before answering it.
        if (
            intent != "chat"
            and intent in INTENT_MODULE_MAP
            and confidence < self._CLARIFY_CONFIDENCE_THRESHOLD
        ):
            module = INTENT_MODULE_MAP.get(intent)
            trace.append(
                f"confidence gate: {confidence:.2f} is below the "
                f"{self._CLARIFY_CONFIDENCE_THRESHOLD:.2f} floor for '{intent}' "
                "— asked for clarification instead of acting"
            )
            return self._route(
                "core", [], confidence,
                f"low confidence ({confidence:.2f}) for intent {intent!r} "
                f"(would route to {module!r}) -> asking for clarification",
                intent="clarify_intent",
            )
        if intent != "chat" and intent in INTENT_MODULE_MAP:
            trace.append(
                f"confidence gate: {confidence:.2f} clears the "
                f"{self._CLARIFY_CONFIDENCE_THRESHOLD:.2f} floor"
            )
        else:
            trace.append("confidence gate: not applicable (chat or an unregistered intent)")

        # --- Tier 0.7: conference (backlog #158) -----------------------
        # Placed after the confidence gate (an unsure "conference" should
        # still ask first) and before Tier 1 (which would otherwise send it
        # straight to a single module and so defeat the point).
        if intent == "conference" and "core" in active_modules:
            seats = self._conference_modules(
                q, nlu_result.get("entities") or {}, active_modules
            )
            if len(seats) >= 2:
                trace.append(
                    "conference: topic touches " + ", ".join(seats)
                    + " — gathering a view from each"
                )
                # secondary stays empty: the orchestrator asks each seat for
                # its read-only view itself (Conference.gather), so the
                # generic get_context pre-fetch would just duplicate work.
                return self._route(
                    "core", [], max(confidence, 0.9),
                    "conference: " + " + ".join(seats),
                    intent="conference", conference=list(seats),
                )
            trace.append(
                "conference: fewer than two active modules have a stake in "
                "this topic — nothing to convene"
            )
            return self._route(
                "core", [], max(confidence, 0.9),
                "conference requested but fewer than two relevant modules",
                intent="conference",
            )

        # --- Tier 0.8: trained classifier, "primary" mode (backlog #4) --
        # Only when the NLU didn't produce a dispatchable intent or was
        # unsure. Placed BEFORE Tier 1 because Tier 1 dispatches any
        # registered intent unconditionally, which would hide exactly the
        # registered-but-unsure case. The text triggers stay as the backstop.
        clf = self._classifier
        if (
            clf is not None
            and getattr(clf, "mode", "") == "primary"
            and (intent == "chat" or intent not in INTENT_MODULE_MAP
                 or confidence < self._CLASSIFIER_UNSURE_BELOW)
        ):
            picked = self._classifier_pick(q, active_modules, trace)
            if picked:
                c_intent, prob = picked
                c_module = INTENT_MODULE_MAP[c_intent]
                trace.append(
                    f"classifier (primary): '{c_intent}' at {prob:.0%} -> {c_module}"
                )
                return self._route(
                    c_module, [], max(prob, 0.9),
                    f"trained classifier: '{c_intent}' -> {c_module}",
                    intent=strip_module_prefix(c_intent),
                )

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
            trace.append(f"registry: '{intent}' is owned by {module}, which is active")
            return self._route(
                module, [], max(confidence, 0.9),
                f"registry: intent '{intent}' -> {module}",
                intent=strip_module_prefix(intent),
            )
        if module and intent != "chat":
            trace.append(
                f"registry: '{intent}' belongs to {module}, but that module "
                "isn't active — fell through"
            )
        elif intent == "chat":
            trace.append("registry: skipped because 'chat' is a catch-all, not a real match")
        else:
            trace.append(f"registry: no module registered for '{intent}'")

        # --- Tier 2: Text trigger matching -----------------------------
        # Ingest check comes first: "ingest my documents" would otherwise
        # never be reachable if a broader athena-search trigger happened to
        # overlap it (same ordering Iris uses for its ingest/search split).
        if "athena" in active_modules and self._match(q, self._ATHENA_INGEST_TRIGGERS):
            trace.append("text trigger matched: Athena ingest wording")
            return self._route("athena", [], 1.0, "athena ingest trigger", intent="ingest")

        if "athena" in active_modules and self._match(q, self._ATHENA_TRIGGERS):
            trace.append("text trigger matched: Athena document-search wording")
            return self._route(
                "athena", ["mnemosyne"] if "mnemosyne" in active_modules else [],
                1.0, "athena trigger", intent="search",
            )

        # Recency-anchored phrasing ("...yesterday/today/earlier") wants
        # chronological history, not semantic search — route to Core's
        # get_history before the Mnemosyne trigger check below gets a
        # chance to send it to vector recall instead.
        if "core" in active_modules and self._match(q, self._RECENCY_TRIGGERS):
            trace.append("text trigger matched: time-anchored 'what did we talk about'")
            return self._route("core", [], 1.0, "recency trigger -> get_history", intent="get_history")

        if "mnemosyne" in active_modules and self._match(q, self._MNEMOSYNE_TRIGGERS):
            trace.append("text trigger matched: Mnemosyne recall wording")
            return self._route("mnemosyne", [], 1.0, "mnemosyne trigger", intent="recall")

        if "iris" in active_modules and self._match(q, self._IRIS_TRIGGERS):
            trace.append("text trigger matched: Iris media wording")
            ingest_triggers = {"ingest media", "ingest photos"}
            iris_intent = "ingest" if q in ingest_triggers or any(
                t in q for t in ingest_triggers
            ) else "search"
            return self._route("iris", [], 1.0, "iris trigger", intent=iris_intent)

        trace.append(
            "text triggers: checked Athena ingest/search, history, Mnemosyne "
            "recall and Iris media wording — none matched"
        )

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
                trace.append(
                    f"prefix fallback: '{intent}' carries the {prefixed_module} prefix"
                )
                return self._route(
                    prefixed_module, [], 0.75,
                    f"unregistered intent '{intent}' -> {prefixed_module} (prefix fallback)",
                )

        trace.append("prefix fallback: the intent carries no module prefix")

        # --- Tier 4: Keyword matching ------------------------------------
        # Tier 1 already routes every real, registered Artemis intent
        # (add_habit, list_habits, productivity_summary, ...) directly, so
        # by the time control reaches here `intent` is guaranteed NOT to be
        # one of those — no exclusion list needed (the old version had to
        # explicitly exclude Artemis's own intents to avoid this tier
        # stealing them; that's now structurally impossible).
        if "artemis" in active_modules and any(k in q for k in self._ARTEMIS_KEYWORDS):
            trace.append("keyword match: habit/goal/streak wording -> Artemis")
            return self._route("artemis", [], 0.9, "artemis keyword match")
        trace.append("keyword match: no habit/goal wording")

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
                trace.append("cross-module wording: Athena documents plus Mnemosyne memory")
                return self._route(
                    "athena",
                    ["mnemosyne"],
                    0.85,
                    "cross-module: athena+mnemosyne",
                    synthesize=True,
                    intent="search",
                )

        # --- Tier 4.9: trained classifier, "assist" mode (backlog #4) ----
        # Last resort before the generic confidence fallbacks: every
        # hand-written tier has already declined.
        if clf is not None and getattr(clf, "mode", "") == "assist" and (
            intent == "chat" or intent not in INTENT_MODULE_MAP
        ):
            picked = self._classifier_pick(q, active_modules, trace)
            if picked:
                c_intent, prob = picked
                c_module = INTENT_MODULE_MAP[c_intent]
                trace.append(
                    f"classifier (assist): '{c_intent}' at {prob:.0%} -> {c_module}"
                )
                return self._route(
                    c_module, [], max(prob, 0.9),
                    f"trained classifier: '{c_intent}' -> {c_module}",
                    intent=strip_module_prefix(c_intent),
                )

        # --- Tier 5: High-confidence NLU non-chat intent -----------------
        if confidence >= 0.85 and intent != "chat":
            trace.append(f"confidence fallback: '{intent}' is high-confidence but unowned -> core")
            return self._route("core", [], confidence, f"high-confidence intent '{intent}'")

        # --- Tier 6: Low-confidence -> force chat ------------------------
        if confidence < 0.5:
            trace.append("confidence fallback: below 0.50 -> plain chat")
            return self._route("core", [], 0.4, "low confidence -> chat fallback")

        trace.append("nothing more specific matched -> default to core chat")
        return self._route("core", [], confidence, "default core")

    def _conference_modules(self, q: str, entities: dict, active_modules: list) -> list:
        """Pick the modules to seat at a conference (backlog #158).

        An explicit ``entities["modules"]`` list (from the NLU, or a caller)
        wins; otherwise modules are ranked by how many topic words the query
        contains. Unknown or inactive modules are dropped, order is stable
        (declaration order breaks ties), and the result is capped.
        """
        active = set(active_modules)
        named = entities.get("modules") if isinstance(entities, dict) else None
        if isinstance(named, str):
            named = [m.strip() for m in named.replace(" and ", ",").split(",")]
        if isinstance(named, (list, tuple)):
            picked: list = []
            for m in named:
                m = str(m).strip().lower()
                if m in active and m != "core" and m not in picked:
                    picked.append(m)
            if len(picked) >= 2:
                return picked[: self._CONFERENCE_MAX_MODULES]

        import re
        words = set(re.findall(r"[a-z']+", q))
        scores: dict = {}
        order: list = []
        for _label, vocab, mods in self._CONFERENCE_TOPICS:
            hits = len(words & vocab)
            if not hits:
                continue
            for m in mods:
                if m not in active:
                    continue
                if m not in scores:
                    order.append(m)
                    scores[m] = 0
                scores[m] += hits
        ranked = sorted(order, key=lambda m: (-scores[m], order.index(m)))
        return ranked[: self._CONFERENCE_MAX_MODULES]

    @staticmethod
    def _match(text: str, triggers: list) -> bool:
        import re
        return any(re.search(r"\b" + re.escape(t) + r"\b", text) for t in triggers)

    @staticmethod
    def _route(primary: str, secondary: list, confidence: float,
            reason: str, synthesize: bool = False, intent: str = None,
            conference: list = None) -> dict:
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
            "conference": conference,
        }