# modules/hestia/orchestrator.py

import logging
from typing import Optional
from modules.base import BaseModule
from modules.hecate.engine import HecateEngine

log = logging.getLogger("hestia.orchestrator")


class HestiaOrchestrator:
    """
    Routes all queries through Hecate to registered modules.
    Hestia (main.py) owns init/wiring. Orchestrator owns dispatch.
    """

    def __init__(self):
        self._modules: dict[str, BaseModule] = {}
        self._hecate: Optional[HecateEngine] = None
        self._context: dict = {
            "recent_intents": [],
            "entities":       {},
            "active_modules": [],
            "time_context":   {},
            "memory_context": {},
        }

    def register_hecate(self, hecate: HecateEngine) -> None:
        self._hecate = hecate

    def register(self, module: BaseModule) -> None:
        """Register a module. Call once per module during Hestia.__init__."""
        if not isinstance(module, BaseModule):
            raise TypeError(f"{module} does not implement BaseModule")
        self._modules[module.name] = module
        if module.name not in self._context["active_modules"]:
            self._context["active_modules"].append(module.name)
        log.debug("Registered module: %s", module.name)

    def dispatch(self, raw_query: str, nlu_result: dict) -> str:
        intent   = nlu_result.get("intent", "chat")
        entities = nlu_result.get("entities", {})
        entities["raw_query"] = raw_query

        context = dict(self._context)

        if self._hecate:
            decision = self._hecate.decide(
                raw_query, nlu_result, self._context["active_modules"]
            )
        else:
            decision = {
                "primary": "core", "secondary": [],
                "confidence": 0.5, "reason": "no hecate",
                "synthesize": False,
            }

        primary    = decision.get("primary", "core")
        secondary  = decision.get("secondary", [])
        synthesize = decision.get("synthesize", False)
        log.debug(
            "Hecate routed → %s (secondary: %s, synthesize: %s) — %s",
            primary, secondary, synthesize, decision.get("reason")
        )

        # Secondary context enrichment (always)
        for mod_name in secondary:
            mod = self._modules.get(mod_name)
            if mod:
                try:
                    context.update(mod.get_context())
                except Exception as e:
                    log.debug("get_context() failed for %s: %s", mod_name, e)

        safe_context = dict(context)

        # Primary dispatch
        primary_mod = self._modules.get(primary)
        if not primary_mod:
            log.warning("Module '%s' not found. Falling back to chat.", primary)
            return self._chat_fallback(raw_query, nlu_result)

        if not primary_mod.can_handle(intent):
            log.warning(
                "Module '%s' cannot handle intent '%s'. Falling back to chat.",
                primary, intent
            )
            return self._chat_fallback(raw_query, nlu_result)

        try:
            primary_result = primary_mod.handle(intent, entities, safe_context)
        except NotImplementedError:
            log.error("Dispatched to Hecate — routing error.")
            return "I had a routing error. Please try again."
        except Exception as e:
            log.exception("Module '%s' handle() raised: %s", primary, e)
            return f"I had trouble with that — {primary} module encountered an error."

        # ── SYNTHESIS PATH ───────────────────────────────────
        if synthesize and secondary:
            secondary_results = []
            for mod_name in secondary:
                mod = self._modules.get(mod_name)
                if mod and mod.can_handle(intent):
                    try:
                        result = mod.handle(intent, entities, safe_context)
                        if result.get("response"):
                            secondary_results.append({
                                "module":   mod_name,
                                "response": result["response"],
                            })
                    except Exception as e:
                        log.debug("Secondary module %s failed: %s", mod_name, e)

            if secondary_results:
                primary_result["response"] = self._synthesize(
                    raw_query,
                    primary,
                    primary_result.get("response", ""),
                    secondary_results,
                )

        # Context updates
        if "context_update" in primary_result:
            context.update(primary_result["context_update"])
        self._context = context
        self._context["recent_intents"].append(intent)
        self._context["recent_intents"] = self._context["recent_intents"][-10:]

        return primary_result.get("response", "...")

    def _synthesize(self, query: str, primary_name: str,
                    primary_response: str, secondary_results: list) -> str:
        """Combine multiple module responses into a single coherent answer."""
        from core.ollama_client import generate

        parts = [f"[{primary_name.upper()}]: {primary_response}"]
        for r in secondary_results:
            parts.append(f"[{r['module'].upper()}]: {r['response']}")

        combined = "\n".join(parts)

        prompt = (
            f"The user asked: \"{query}\"\n\n"
            f"Multiple sources returned the following information:\n{combined}\n\n"
            "Synthesize this into a single, coherent, natural response in 2-3 sentences. "
            "Do not mention the source names. Just answer the question directly."
        )

        try:
            result = generate(prompt)
            return result if result else primary_response
        except Exception:
            return primary_response  # graceful fallback to primary alone

    def _chat_fallback(self, raw_query: str, nlu_result: dict) -> str:
        """Used when no module matches — returns NLU's pre-generated response."""
        return nlu_result.get("response", "I'm not sure how to help with that.")