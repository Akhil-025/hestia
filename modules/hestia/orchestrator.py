"""
modules/hestia/orchestrator.py

HestiaOrchestrator: routes queries through Hecate to registered modules.

Responsibilities
----------------
- Accept module registrations during startup.
- Accept a Hecate routing engine.
- For each query: normalise intent → enrich context → dispatch → optionally
  synthesise → update rolling context → return a plain string response.

Hestia (main.py) owns initialisation and wiring.
Orchestrator owns dispatch logic only.
"""
from __future__ import annotations

import threading
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from modules.base import BaseModule
from modules.hecate.engine import HecateEngine
from modules.hecate.intent_registry import strip_module_prefix as _strip_module_prefix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Module-prefix stripping now lives in modules/hecate/intent_registry.py —
# a single source of truth shared with Hecate's router and the NLU's intent
# whitelist, instead of a fourth independent copy of the same prefix tuple
# living here. See that file's module docstring for the drift this used to
# cause (intents that existed in a module's own _INTENTS set but were
# missing from one or more of these copies, making them unreachable).

_FALLBACK_DECISION: dict[str, Any] = {
    "primary": "core",
    "secondary": [],
    "confidence": 0.5,
    "reason": "hecate unavailable",
    "synthesize": False,
}

_MAX_RECENT_INTENTS = 10
_GENERIC_ERROR = "I'm sorry, something went wrong. Please try again."

# ---------------------------------------------------------------------------
# Confirmation gating
# ---------------------------------------------------------------------------
# Some intents are irreversible or externally visible enough (sending an
# email, deleting every saved note, forgetting a remembered fact) that a
# single misheard word from STT — or a bad NLU guess — shouldn't be enough
# to actually do them. Those handlers don't perform the action on their
# first call; instead they return `needs_confirmation=True` with the
# `confirm_intent`/`confirm_entities` needed to actually execute it, and the
# orchestrator holds that as pending state until the very next query either
# confirms or cancels it (or it expires, below). See
# HestiaOrchestrator._execute_confirmed / _classify_yes_no.
_CONFIRMATION_TTL_SECONDS = 120

_AFFIRMATIVE_PHRASES: frozenset[str] = frozenset(
    {
        "yes", "yeah", "yea", "yep", "yup", "confirm", "confirmed",
        "do it", "send it", "go ahead", "sure", "please do",
        "affirmative", "ok", "okay", "correct", "proceed",
    }
)
_NEGATIVE_PHRASES: frozenset[str] = frozenset(
    {
        "no", "nope", "nah", "cancel", "don't", "do not", "stop",
        "never mind", "nevermind", "negative", "abort",
    }
)
# Single leading word that still counts even inside a longer reply, e.g.
# "yes please" or "no thanks" — the exact-phrase sets above only match the
# whole (stripped) utterance.
_AFFIRMATIVE_LEAD_WORDS: frozenset[str] = frozenset(
    {"yes", "yeah", "yea", "yep", "yup", "confirm", "sure", "ok", "okay"}
)
_NEGATIVE_LEAD_WORDS: frozenset[str] = frozenset(
    {"no", "nope", "nah", "cancel", "stop", "negative", "abort"}
)


def _classify_yes_no(text: str) -> Optional[bool]:
    """
    Return True for an affirmative reply, False for a negative one, or None
    if *text* doesn't clearly look like either (in which case the caller
    should NOT guess — see the pending-confirmation handling in dispatch()).
    """
    q = (text or "").strip().lower().rstrip(".!?")
    if not q:
        return None
    if q in _AFFIRMATIVE_PHRASES:
        return True
    if q in _NEGATIVE_PHRASES:
        return False
    words = q.split()
    if words and words[0] in _AFFIRMATIVE_LEAD_WORDS:
        return True
    if words and words[0] in _NEGATIVE_LEAD_WORDS:
        return False
    return None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OrchestratorError(Exception):
    """Base exception for orchestrator failures."""


class ModuleNotRegisteredError(OrchestratorError):
    """Raised when a required module is not found in the registry."""


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Normalised output from a module's handle() call."""

    response: str
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    context_update: dict[str, Any] = field(default_factory=dict)
    # Confirmation-gating fields — see "Confirmation gating" above. Unset
    # (needs_confirmation=False) for the overwhelming majority of handlers;
    # only irreversible/externally-visible actions set these.
    needs_confirmation: bool = False
    confirm_intent: Optional[str] = None
    confirm_entities: dict[str, Any] = field(default_factory=dict)
    confirm_label: str = "do that"


@dataclass
class PendingConfirmation:
    """
    A confirmation-gated action awaiting the user's next reply.

    Held as single, instance-level state on the orchestrator (not per-
    session) because Hestia is a single-user, single-conversation
    assistant — see HestiaOrchestrator's class docstring.
    """

    module: str
    intent: str
    entities: dict[str, Any]
    label: str
    created_at: float


@dataclass
class OrchestratorContext:
    """
    Rolling context passed to every module on each turn.

    Kept as a dataclass so the shape is explicit and type-checked.
    """

    recent_intents: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    active_modules: list[str] = field(default_factory=list)
    time_context: dict[str, Any] = field(default_factory=dict)
    memory_context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recent_intents": list(self.recent_intents),
            "entities": dict(self.entities),
            "active_modules": list(self.active_modules),
            "time_context": dict(self.time_context),
            "memory_context": dict(self.memory_context),
        }

    def push_intent(self, intent: str) -> None:
        self.recent_intents.append(intent)
        if len(self.recent_intents) > _MAX_RECENT_INTENTS:
            self.recent_intents = self.recent_intents[-_MAX_RECENT_INTENTS:]

    def apply_update(self, update: dict[str, Any]) -> None:
        for key, value in update.items():
            if hasattr(self, key):
                if key == "recent_intents" and not isinstance(value, list):
                    continue
                setattr(self, key, value)
            else:
                logger.warning(
                    "context_update contains unknown key %r; ignoring.", key
                )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class HestiaOrchestrator:
    """
    Routes all queries through Hecate to registered modules.

    Thread-safety
    -------------
    Thread-safe for context mutations via an internal lock.

    Concurrent dispatch calls are allowed, but may observe slightly stale
    context snapshots (eventual consistency model). If strict consistency
    is required, wrap dispatch() with an external lock.
    """

    def __init__(self, ollama_cfg: Optional[dict] = None) -> None:
        self._modules: dict[str, BaseModule] = {}
        self._hecate: Optional[HecateEngine] = None
        self._ctx = OrchestratorContext()
        self._lock = threading.Lock()
        # Set only while a confirmation-gated action (see "Confirmation
        # gating" above) is awaiting the user's yes/no on the *next* query.
        self._pending: Optional[PendingConfirmation] = None
        # Used by _synthesize() so synthesis honours the configured model/host/port
        # instead of silently falling back to core.ollama_client's hardcoded defaults.
        self._ollama_cfg: dict = ollama_cfg or {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_hecate(self, hecate: HecateEngine) -> None:
        """Attach the Hecate routing engine."""
        if not isinstance(hecate, HecateEngine):
            raise TypeError(f"Expected HecateEngine, got {type(hecate).__name__!r}.")
        self._hecate = hecate
        logger.info("Hecate engine registered.")

    def register(self, module: BaseModule) -> None:
        """
        Register a module.

        Call once per module during Hestia startup.

        Raises
        ------
        TypeError
            If *module* does not implement BaseModule.
        """
        if not isinstance(module, BaseModule):
            raise TypeError(
                f"{module!r} does not implement BaseModule."
            )
        name = module.name
        if name in self._modules:
            logger.warning(
                "Module %r is already registered; replacing.", name
            )
        self._modules[name] = module
        if name not in self._ctx.active_modules:
            self._ctx.active_modules.append(name)
        logger.debug("Registered module: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a module from the registry at runtime."""
        self._modules.pop(name, None)
        if name in self._ctx.active_modules:
            self._ctx.active_modules.remove(name)
        logger.info("Unregistered module: %s", name)

    @property
    def registered_modules(self) -> list[str]:
        """Names of all currently registered modules."""
        return list(self._modules)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, raw_query: str, nlu_result: dict[str, Any]) -> str:
        """
        Route a query to the appropriate module and return a response string.

        Steps
        -----
        1. Normalise the intent (strip module prefix).
        2. Ask Hecate for a routing decision.
        3. Enrich context from secondary modules.
        4. Dispatch to the primary module.
        5. Optionally synthesise secondary context into the response.
        6. Update rolling context.

        Never raises; all internal errors produce a graceful string response.
        """
        t_start = time.perf_counter()

        # A confirmation-gated action from the previous turn takes priority
        # over normal routing — this query is presumed to be the user's
        # answer to "send it?" / "delete them?" / etc., not a new request.
        pending_response = self._resolve_pending(raw_query)
        if pending_response is not None:
            return pending_response

        raw_intent: str = nlu_result.get("intent") or "chat"
        # Intent convention: NLU emits prefixed intents (e.g. "ares_analyse_risk").
        # The orchestrator strips the module prefix *here*, before dispatching, so
        # modules register unprefixed intents in their _INTENTS sets (e.g. "analyse_risk").
        # Do NOT put prefixed names in a module's _INTENTS — can_handle() receives stripped names.
        intent = _strip_module_prefix(raw_intent)
        entities: dict[str, Any] = dict(nlu_result.get("entities") or {})
        entities["raw_query"] = raw_query

        decision = self._route(raw_query, nlu_result)
        primary_name: str = decision.get("primary") or "core"

        # Hecate's text-trigger tiers (e.g. "mnemosyne trigger") match on the
        # raw query, not on nlu_result["intent"] — which is frequently just
        # "chat" for those phrasings. When Hecate supplies an explicit
        # `intent` override, use it instead of the NLU-derived one so the
        # primary module's can_handle()/handle() actually see an intent they
        # declare, rather than silently losing the routing decision to the
        # can_handle()-mismatch recovery path (which lands on "core" chat).
        override_intent = decision.get("intent")
        if isinstance(override_intent, str) and override_intent:
            intent = override_intent
        secondary_names: list[str] = [
            n for n in (decision.get("secondary") or []) if n != primary_name
        ]
        synthesize: bool = bool(decision.get("synthesize", False))

        logger.debug(
            "Hecate → primary=%s secondary=%s synthesize=%s reason=%r",
            primary_name,
            secondary_names,
            synthesize,
            decision.get("reason"),
        )

        # Build a per-request context snapshot
        with self._lock:
            context = self._ctx.as_dict()
        context, secondary_ctx_cache = self._enrich_context(context, secondary_names)

        # Primary dispatch
        response = self._dispatch_primary(
            primary_name=primary_name,
            intent=intent,
            raw_intent=raw_intent,
            entities=entities,
            context=context,
            raw_query=raw_query,
            nlu_result=nlu_result,
        )

        if isinstance(response, DispatchResult) and response.needs_confirmation:
            # The module validated the request but deliberately did NOT
            # perform it (see "Confirmation gating"). Hold it as pending and
            # hand the confirmation question back as this turn's response —
            # the actual action only runs if the *next* query reads as a
            # clear "yes" (see _resolve_pending).
            with self._lock:
                self._pending = PendingConfirmation(
                    module=primary_name,
                    intent=response.confirm_intent or intent,
                    entities=dict(response.confirm_entities or entities),
                    label=response.confirm_label,
                    created_at=time.time(),
                )
                self._ctx.push_intent(raw_intent)
            logger.info(
                "Pending confirmation set: module=%s intent=%s label=%r",
                primary_name, response.confirm_intent or intent, response.confirm_label,
            )
            return response.response

        if isinstance(response, DispatchResult):
            # Synthesis
            if synthesize and secondary_names:
                secondary_ctx = secondary_ctx_cache
                if secondary_ctx:
                    response.response = self._synthesize(
                        raw_query,
                        primary_name,
                        response.response,
                        secondary_ctx,
                    )

            # Roll context forward (thread-safe)
            with self._lock:
                self._ctx.apply_update(response.context_update)
                self._ctx.push_intent(raw_intent)

            elapsed = (time.perf_counter() - t_start) * 1000
            logger.debug(
                "dispatch() complete in %.1f ms → %r…",
                elapsed,
                response.response[:60],
            )
            return response.response

        # String fallback (should not normally reach here)
        with self._lock:
            self._ctx.push_intent(raw_intent)
        return response  # type: ignore[return-value]

    def try_stream_chat(self, raw_query: str, nlu_result: dict[str, Any]):
        """
        Return a generator of text chunks for a plain conversational turn
        that can be streamed straight into TTS as it's produced, or
        ``None`` when this turn needs the full dispatch() machinery
        instead.

        Streams "core module, chat intent, nothing pending confirmation"
        turns — including ones Hecate flagged for secondary-module
        synthesis. In the blocking dispatch() path, synthesis costs a
        second, fully-blocking LLM call after the first has already
        finished (see _synthesize()); here, secondary context is folded
        into the *same* streamed generation via core.stream_chat's
        context_block (see its docstring), so synthesis no longer
        disqualifies a turn from streaming. Any other case (any other
        primary module, a non-chat intent, an awaited yes/no) still
        returns None and the caller falls back to the ordinary blocking
        dispatch(). That keeps this additive: it never changes *what*
        gets answered, only whether — and how — the answer gets to start
        speaking before the whole thing exists.
        """
        with self._lock:
            if self._pending is not None:
                # A confirmation is awaiting this exact reply — that flow
                # needs dispatch()'s _resolve_pending handling, not a
                # fresh chat completion.
                return None

        raw_intent: str = nlu_result.get("intent") or "chat"
        intent = _strip_module_prefix(raw_intent)

        decision = self._route(raw_query, nlu_result)
        primary_name: str = decision.get("primary") or "core"

        override_intent = decision.get("intent")
        if isinstance(override_intent, str) and override_intent:
            intent = override_intent

        secondary_names = [
            n for n in (decision.get("secondary") or []) if n != primary_name
        ]
        synthesize = bool(decision.get("synthesize", False))

        if primary_name != "core" or intent != "chat":
            return None

        core = self._modules.get("core")
        if core is None or not hasattr(core, "stream_chat"):
            return None

        # Mirrors dispatch()'s own gate on when synthesis actually runs
        # ("if synthesize and secondary_names:") — if either is missing,
        # there's nothing to fold in and this is just plain chat.
        context_block = ""
        if synthesize and secondary_names:
            with self._lock:
                context = self._ctx.as_dict()
            _, secondary_ctx_cache = self._enrich_context(context, secondary_names)
            if secondary_ctx_cache:
                context_block = _build_context_block(secondary_ctx_cache)

        def _generator():
            try:
                yield from core.stream_chat(raw_query, context_block)
            finally:
                with self._lock:
                    self._ctx.push_intent(raw_intent)

        return _generator()

    # ------------------------------------------------------------------
    # Private – confirmation gating
    # ------------------------------------------------------------------

    def _resolve_pending(self, raw_query: str) -> Optional[str]:
        """
        If a confirmation-gated action is awaiting a reply, consume this
        query as that reply and return the resulting response string.
        Returns None if there is nothing pending (or it just expired), in
        which case *raw_query* should be routed normally instead.

        An ambiguous reply (neither a clear yes nor a clear no) silently
        drops the pending action rather than guessing — e.g. asking a
        follow-up question about something unrelated must not later have a
        stray "yes" accidentally send the email. The user simply has to
        re-ask if they actually wanted it.
        """
        with self._lock:
            pending = self._pending
            if pending is not None and (time.time() - pending.created_at) > _CONFIRMATION_TTL_SECONDS:
                logger.info(
                    "Pending confirmation for %s.%s expired unanswered.",
                    pending.module, pending.intent,
                )
                self._pending = None
                pending = None

        if pending is None:
            return None

        verdict = _classify_yes_no(raw_query)

        if verdict is True:
            with self._lock:
                self._pending = None
            return self._execute_confirmed(pending)

        if verdict is False:
            with self._lock:
                self._pending = None
            logger.info(
                "Pending confirmation for %s.%s declined by user.",
                pending.module, pending.intent,
            )
            return f"Okay, I won't {pending.label}."

        with self._lock:
            self._pending = None
        logger.info(
            "Pending confirmation for %s.%s abandoned (next query wasn't a yes/no).",
            pending.module, pending.intent,
        )
        return None

    def _execute_confirmed(self, pending: PendingConfirmation) -> str:
        """
        Actually perform a previously-confirmed action by re-dispatching
        straight to the module that asked for confirmation, bypassing
        Hecate — the routing decision was already made when the
        confirmation question was first asked.
        """
        entities = dict(pending.entities)
        entities["_confirmed"] = True
        raw_query = str(entities.get("raw_query") or "")

        with self._lock:
            context = self._ctx.as_dict()

        response = self._dispatch_primary(
            primary_name=pending.module,
            intent=pending.intent,
            raw_intent=pending.intent,
            entities=entities,
            context=context,
            raw_query=raw_query,
            nlu_result={"intent": pending.intent, "entities": entities, "confidence": 1.0},
        )

        if isinstance(response, DispatchResult):
            with self._lock:
                self._ctx.apply_update(response.context_update)
                self._ctx.push_intent(pending.intent)
            logger.info(
                "Confirmed action executed: module=%s intent=%s",
                pending.module, pending.intent,
            )
            return response.response

        with self._lock:
            self._ctx.push_intent(pending.intent)
        return response  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Private – routing
    # ------------------------------------------------------------------

    def _route(self, raw_query: str, nlu_result: dict[str, Any]) -> dict[str, Any]:
        """Return a Hecate routing decision, falling back gracefully."""
        if self._hecate is None:
            logger.debug("Hecate not registered; using fallback decision.")
            return _FALLBACK_DECISION.copy()

        try:
            with self._lock:
                active_modules = list(self._ctx.active_modules)

            return self._hecate.decide(raw_query, nlu_result, active_modules)
        except Exception:
            logger.exception("Hecate.decide() failed; using fallback decision.")
            return _FALLBACK_DECISION.copy()

    # ------------------------------------------------------------------
    # Private – context enrichment
    # ------------------------------------------------------------------

    def _enrich_context(
        self, context: dict[str, Any], module_names: list[str]
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """
        Update *context* in-place with each secondary module's get_context().

        Failures are logged and skipped; a partial enrichment is always
        better than aborting the request.
        """
        cache: dict[str, dict[str, Any]] = {}
        for name in module_names:
            mod = self._modules.get(name)
            if mod is None:
                logger.debug("Secondary module %r not registered; skipping.", name)
                continue
            try:
                ctx = mod.get_context()
                if isinstance(ctx, dict):
                    context.update(ctx)
                    if ctx:
                        cache[name] = ctx
                else:
                    logger.warning(
                        "Module %r get_context() returned %s, expected dict; skipping.",
                        name,
                        type(ctx).__name__,
                    )
            except Exception:
                logger.exception(
                    "get_context() raised for module %r; skipping.", name
                )
        return context, cache

 
    # ------------------------------------------------------------------
    # Private – primary dispatch
    # ------------------------------------------------------------------

    def _dispatch_primary(
        self,
        *,
        primary_name: str,
        intent: str,
        raw_intent: str,
        entities: dict[str, Any],
        context: dict[str, Any],
        raw_query: str,
        nlu_result: dict[str, Any],
    ) -> DispatchResult | str:
        """
        Invoke the primary module's handle() and return a DispatchResult.

        Falls back to chat on routing or capability mismatches.
        """
        mod = self._modules.get(primary_name)
        if mod is None:
            logger.warning(
                "Primary module %r not registered; falling back to chat.",
                primary_name,
            )
            return self._chat_fallback(nlu_result)

        if not mod.can_handle(intent):
            # Hecate's routing guess doesn't match what the named module
            # actually declares — most commonly because the NLU model
            # emitted a plausible-looking but wrong module prefix (e.g.
            # "pluto_get_calendar_event" for a calendar query, which
            # Hecate's Tier-X prefix match routes straight to pluto without
            # ever checking whether pluto declares "get_calendar_event").
            # Before giving up and falling back to chat — which silently
            # discards the user's actual request — check whether any OTHER
            # registered module declares this (stripped) intent and
            # dispatch there instead.
            alternate = self._find_alternate_module(intent, exclude=primary_name)
            if alternate is not None:
                alt_name, alt_mod = alternate
                logger.warning(
                    "Hecate routed intent %r to module %r, which doesn't "
                    "declare it; module %r does — dispatching there instead.",
                    raw_intent, primary_name, alt_name,
                )
                primary_name, mod = alt_name, alt_mod
            else:
                logger.warning(
                    "Module %r cannot handle intent %r; falling back to chat.",
                    primary_name,
                    raw_intent,
                )
                return self._chat_fallback(nlu_result)

        try:
            raw_result = mod.handle(intent, entities, context)
        except NotImplementedError:
            logger.error(
                "Module %r raised NotImplementedError for intent %r.",
                primary_name,
                intent,
            )
            return _GENERIC_ERROR
        except Exception:
            logger.exception(
                "Module %r handle() raised for intent %r.",
                primary_name,
                intent,
            )
            return _GENERIC_ERROR

        return _to_dispatch_result(raw_result)

    def _find_alternate_module(
        self, intent: str, exclude: str
    ) -> Optional[tuple[str, BaseModule]]:
        """
        Look for a registered module (other than *exclude*) whose
        ``can_handle(intent)`` returns True.

        Used only as a recovery path when Hecate's routing decision names
        a module that doesn't actually support the (prefix-stripped)
        intent — see ``_dispatch_primary``. Every module's ``can_handle``
        is exact-set membership (never a wildcard), so this can't
        accidentally hijack an unrelated query; it only catches genuinely
        misrouted-but-validly-named intents.
        """
        for name, candidate in self._modules.items():
            if name == exclude:
                continue
            try:
                if candidate.can_handle(intent):
                    return name, candidate
            except Exception:
                logger.exception(
                    "can_handle() raised for module %r while searching "
                    "for an alternate handler of intent %r.", name, intent,
                )
        return None

    # ------------------------------------------------------------------
    # Private – synthesis
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        query: str,
        primary_name: str,
        primary_response: str,
        secondary_context: dict[str, dict[str, Any]],
    ) -> str:
        """
        Ask the LLM to merge the primary response with secondary context.

        Returns *primary_response* unchanged if synthesis fails.
        """
        from core.ollama_client import generate  # late import – optional dep

        context_block = _build_context_block(secondary_context)
        prompt = _build_synthesis_prompt(query, primary_name, primary_response, context_block)

        try:
            t0 = time.perf_counter()
            result = generate(
                prompt,
                model=self._ollama_cfg.get("model", "mistral"),
                host=self._ollama_cfg.get("host", "127.0.0.1"),
                port=self._ollama_cfg.get("port", 11434),
                timeout=5,
            )
            logger.debug("Synthesis took %.2f ms", (time.perf_counter() - t0)*1000)
            if result and result.strip():
                return result.strip()
            logger.warning("Synthesis returned empty result; using primary response.")
        except Exception:
            logger.exception("Synthesis LLM call failed; using primary response.")

        return primary_response

    # ------------------------------------------------------------------
    # Private – chat fallback
    # ------------------------------------------------------------------

    def _chat_fallback(self, nlu_result: dict[str, Any]) -> str:
        core = self._modules.get("core")
        if core and core.can_handle("chat"):
            try:
                return core.handle("chat", {}, {})["response"]
            except Exception:
                logger.exception("Core chat fallback failed.")
        return "I'm not sure how to help with that."


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _to_dispatch_result(raw: Any) -> DispatchResult:
    """
    Coerce a module's handle() return value to a DispatchResult.

    Modules are expected to return a dict with at least a ``response`` key,
    but we handle strings and unexpected types defensively.
    """
    if isinstance(raw, DispatchResult):
        return raw

    if isinstance(raw, dict):
        return DispatchResult(
            response=str(raw.get("response") or ""),
            data=raw.get("data") or {},
            confidence=float(raw.get("confidence") or 0.0),
            context_update=raw.get("context_update") or {},
            needs_confirmation=bool(raw.get("needs_confirmation", False)),
            confirm_intent=raw.get("confirm_intent"),
            confirm_entities=raw.get("confirm_entities") or {},
            confirm_label=str(raw.get("confirm_label") or "do that"),
        )

    if isinstance(raw, str):
        logger.debug("Module returned a bare string; wrapping in DispatchResult.")
        return DispatchResult(response=raw)

    logger.warning(
        "Module returned unexpected type %s; converting to string.",
        type(raw).__name__,
    )
    return DispatchResult(response=str(raw))


def _build_context_block(secondary_context: dict[str, dict[str, Any]]) -> str:
    """Render secondary context dicts as a readable block for the prompt."""
    parts: list[str] = []
    for mod_name, ctx in secondary_context.items():
        if not ctx:
            continue
        lines = "\n".join(f"  {k}: {v}" for k, v in ctx.items())
        parts.append(f"[{mod_name.upper()} CONTEXT]\n{lines}")
    return "\n\n".join(parts)


def _build_synthesis_prompt(
    query: str,
    primary_name: str,
    primary_response: str,
    context_block: str,
) -> str:
    """Build the LLM synthesis prompt."""
    extra = (
        f"Additional context from other sources:\n{context_block}\n\n"
        if context_block
        else ""
    )
    return (
        f'The user asked: "{query}"\n\n'
        f"Primary answer from {primary_name}:\n{primary_response}\n\n"
        f"{extra}"
        "Synthesize this into a single, coherent, natural response in 2–3 sentences. "
        "Use the additional context to enrich the answer where relevant. "
        "Do not mention source names. Just answer the question directly."
    )