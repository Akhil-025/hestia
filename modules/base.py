# modules/base.py

import re
from abc import ABC, abstractmethod

# Intent naming convention: modules register UNPREFIXED, snake_case intents
# in their `_INTENTS` set/frozenset (e.g. "analyse_risk", not "ares_analyse_risk").
# The NLU layer emits module-prefixed intents (e.g. "ares_analyse_risk"); the
# orchestrator strips the module prefix via `_strip_module_prefix()` before
# calling `can_handle()` / `handle()`. Modules therefore never see the prefix.
_INTENT_RE = re.compile(r'^[a-z][a-z0-9_]*$')


class BaseModule(ABC):
    """
    Mandatory contract for every Hestia module.
    Hestia calls only: can_handle(), handle(), get_context().
    Internal methods (query, search, remember, etc.) are private implementation details.

    Intent naming convention: `_INTENTS` (if defined by a subclass) must contain
    unprefixed, lowercase snake_case intent names (e.g. "analyse_risk"). Do NOT
    include the module prefix (e.g. "ares_") — the orchestrator strips it before
    dispatching, so a prefixed entry in `_INTENTS` will never match.
    """
    name: str = "base"

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        intents = getattr(cls, "_INTENTS", None)
        if intents:
            for intent in intents:
                if not _INTENT_RE.match(intent):
                    raise ValueError(
                        f"{cls.__name__}._INTENTS contains invalid intent {intent!r}. "
                        "Use unprefixed snake_case (e.g. 'analyse_risk', not 'ares_analyse_risk')."
                    )

    @abstractmethod
    def can_handle(self, intent: str) -> bool:
        """Return True if this module can process the given intent."""
        ...

    @abstractmethod
    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Process the intent and return a response dict.
        Must always return:
            {
                "response": str,        # spoken/displayed output
                "data": dict,           # structured payload (may be empty)
                "confidence": float,    # 0.0–1.0
            }
        Optionally include:
            "context_update": dict      # merged into Hestia's shared context
        """
        ...

    def get_context(self) -> dict:
        """
        Return ambient context this module wants to share with others.
        Called by Hestia before dispatching to secondary modules.
        Default: empty — override where useful.
        """
        return {}