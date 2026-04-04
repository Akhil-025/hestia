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
        """
        Full routing cycle:
          1. Ask Hecate for a routing decision
          2. Collect context from secondary modules
          3. Call primary module's handle()
          4. Apply context updates
          5. Return response string
        """
        intent   = nlu_result.get("intent", "chat")
        entities = nlu_result.get("entities", {})
        entities["raw_query"] = raw_query

        # 1. Hecate decides
        if self._hecate:
            decision = self._hecate.decide(raw_query, nlu_result, self._context["active_modules"])
        else:
            decision = {"primary": "core", "secondary": [], "confidence": 0.5, "reason": "no hecate"}

        primary   = decision.get("primary", "core")
        secondary = decision.get("secondary", [])
        log.debug("Hecate routed → %s (secondary: %s) — %s", primary, secondary, decision.get("reason"))

        # 2. Secondary context enrichment
        for mod_name in secondary:
            mod = self._modules.get(mod_name)
            if mod:
                try:
                    ctx_update = mod.get_context()
                    self._context.update(ctx_update)
                except Exception as e:
                    log.debug("get_context() failed for %s: %s", mod_name, e)

        # 3. Dispatch to primary module
        primary_mod = self._modules.get(primary)
        if not primary_mod:
            log.warning("Module '%s' not found in registry. Falling back to chat.", primary)
            return self._chat_fallback(raw_query, nlu_result)

        try:
            result = primary_mod.handle(intent, entities, self._context)
        except NotImplementedError:
            # Hecate itself was accidentally dispatched — shouldn't happen
            log.error("Dispatched to Hecate — routing error.")
            return "I had a routing error. Please try again."
        except Exception as e:
            log.exception("Module '%s' handle() raised: %s", primary, e)
            return f"I had trouble with that — {primary} module encountered an error."

        # 4. Context updates from module
        if "context_update" in result:
            self._context.update(result["context_update"])

        # 5. Rolling intent history
        self._context["recent_intents"].append(intent)
        self._context["recent_intents"] = self._context["recent_intents"][-10:]

        return result.get("response", "...")

    def _chat_fallback(self, raw_query: str, nlu_result: dict) -> str:
        """Used when no module matches — returns NLU's pre-generated response."""
        return nlu_result.get("response", "I'm not sure how to help with that.")