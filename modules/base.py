# modules/base.py

from abc import ABC, abstractmethod


class BaseModule(ABC):
    """
    Mandatory contract for every Hestia module.
    Hestia calls only: can_handle(), handle(), get_context().
    Internal methods (query, search, remember, etc.) are private implementation details.
    """
    name: str = "base"

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