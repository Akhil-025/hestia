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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODULE_PREFIXES: tuple[str, ...] = (
    "apollo_",
    "ares_",
    "orpheus_",
    "dionysus_",
    "pluto_",
)

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

    def __init__(self) -> None:
        self._modules: dict[str, BaseModule] = {}
        self._hecate: Optional[HecateEngine] = None
        self._ctx = OrchestratorContext()
        self._lock = threading.Lock()

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

        raw_intent: str = nlu_result.get("intent") or "chat"
        intent = _strip_module_prefix(raw_intent)
        entities: dict[str, Any] = dict(nlu_result.get("entities") or {})
        entities["raw_query"] = raw_query

        decision = self._route(raw_query, nlu_result)
        primary_name: str = decision.get("primary") or "core"
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
        return context

 
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
            result = generate(prompt, timeout=5)
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

def _strip_module_prefix(intent: str) -> str:
    """Remove a known module prefix from *intent*, if present."""
    for prefix in _MODULE_PREFIXES:
        if intent.startswith(prefix):
            return intent[len(prefix):]
    return intent


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