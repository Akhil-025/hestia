"""
modules/hephaestus/engine.py

HephaestusEngine: browser automation and web interaction module.

Design notes
------------
- The browser agent is injected at construction time; every handler checks
  readiness before delegating, returning a clear "not available" response
  rather than raising AttributeError.
- Intent dispatch is explicit and exhaustive; the fallback path is
  unreachable for registered intents but safe if the registry and dispatch
  table drift.
- All browser-agent calls are wrapped in try/except so a Playwright /
  Selenium crash never propagates to the orchestrator.
- Entity extraction and validation are delegated to pure module-level
  helpers so they can be unit-tested without an engine instance.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from modules.base import BaseModule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NAVIGATE_ACTIONS: frozenset[str] = frozenset(
    {"open", "browse", "navigate", "go", "go to", "visit", "load", ""}
)

_NOT_AVAILABLE = (
    "Browser automation is not available. "
    "Ask me to enable it or check that the browser agent is configured."
)
_UNHANDLED = "I'm not sure what browser action to take."


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HephaestusError(Exception):
    """Base exception for HephaestusEngine failures."""


class BrowserAgentError(HephaestusError):
    """Raised when the browser agent returns an error or raises unexpectedly."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HephaestusEngine(BaseModule):
    """
    Browser automation and web interaction module.

    Supported intents
    -----------------
    ``browser_action``
        Open a URL or execute a named browser action.
    ``search_web``
        Perform a web search and return a summarised result.
    ``check_flight``
        Look up real-time flight status by flight number.

    Parameters
    ----------
    browser_agent:
        A ``HestiaBrowserAgent`` instance (or compatible duck-typed object).
        Injected by the orchestrator at startup.
    """

    name = "hephaestus"

    _INTENTS: frozenset[str] = frozenset(
        {
            "browser_action",
            "search_web",
            "check_flight",
        }
    )

    def __init__(self, browser_agent: Any = None) -> None:
        self._browser = browser_agent
        logger.info(
            "HephaestusEngine ready (browser_agent=%s).",
            type(browser_agent).__name__ if browser_agent else "None",
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Dispatch an intent to the appropriate handler.

        Returns a "not available" response when the browser agent is absent.
        Never raises.
        """
        if not self._is_ready():
            logger.warning("handle(%r): browser agent not ready.", intent)
            return _err(_NOT_AVAILABLE)

        try:
            return self._dispatch(intent, entities)
        except Exception:
            logger.exception(
                "HephaestusEngine.handle() raised for intent=%s.", intent
            )
            return _err("Something went wrong in the browser module.")

    def get_context(self) -> dict:
        return {"hephaestus_available": self._is_ready()}

    # ------------------------------------------------------------------
    # Private – readiness
    # ------------------------------------------------------------------

    def _is_ready(self) -> bool:
        """Return True when the browser agent is present and operational."""
        return self._browser is not None

    # ------------------------------------------------------------------
    # Private – dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, entities: dict) -> dict:
        """Route a validated intent to its handler."""
        if intent == "check_flight":
            return self._check_flight(entities)
        if intent == "search_web":
            return self._search_web(entities)
        if intent == "browser_action":
            return self._browser_action(entities)
        return _err(_UNHANDLED)

    # ------------------------------------------------------------------
    # Private – intent handlers
    # ------------------------------------------------------------------

    def _check_flight(self, entities: dict) -> dict:
        """Look up real-time flight status by flight number."""
        flight = _extract(entities, "flight_number", "flight", "query", "raw_query")
        if not flight:
            return _clarify(
                "Which flight number should I look up? "
                "(e.g. AI202, EK507)"
            )

        try:
            result: str = self._browser.check_flight_status(flight)
        except Exception:
            logger.exception("check_flight_status() raised for flight=%r.", flight)
            return _err(f"I couldn't retrieve the status for flight {flight!r}.")

        if not result or not result.strip():
            logger.warning("check_flight_status() returned empty result for %r.", flight)
            return _err(f"No status information found for flight {flight!r}.")

        logger.info("check_flight: status retrieved for %r.", flight)
        return _ok(result.strip(), confidence=0.9)

    def _search_web(self, entities: dict) -> dict:
        """Run a web search and return a summarised result."""
        query = _extract(entities, "query", "topic", "raw_query")
        if not query:
            return _clarify("What would you like me to search for?")

        try:
            result: str = self._browser.search_web(query)
        except Exception:
            logger.exception("search_web() raised for query=%r.", query[:80])
            return _err("I couldn't complete that web search.")

        if not result or not result.strip():
            logger.warning("search_web() returned empty result for query=%r.", query[:80])
            return _err(f"I didn't find anything useful for {query!r}.")

        logger.info("search_web: results returned for query=%r.", query[:60])
        return _ok(result.strip(), confidence=0.85)

    def _browser_action(self, entities: dict) -> dict:
        """
        Execute a browser action: navigate to a URL or search by query.

        Resolution order
        ----------------
        1. If a ``url`` entity is present and the action is a navigation
           verb (or absent), open the URL directly.
        2. If a ``query`` / ``topic`` / ``raw_query`` is present, perform
           a web search.
        3. Otherwise ask for clarification.
        """
        action = (entities.get("action") or "").strip().lower()
        url = (entities.get("url") or "").strip()
        query = _extract(entities, "query", "topic", "raw_query")

        if url:
            if action not in _NAVIGATE_ACTIONS:
                logger.debug(
                    "_browser_action: action=%r is not a navigation verb; "
                    "treating as navigate anyway (url present).",
                    action,
                )
            return self._open_url(url)

        if query:
            return self._search_web(entities)

        return _clarify(
            "What would you like me to do in the browser? "
            "You can give me a URL to open or something to search for."
        )

    def _open_url(self, url: str) -> dict:
        """Navigate the browser to *url* and return the agent's response."""
        if not _looks_like_url(url):
            logger.warning("_open_url: %r does not look like a URL.", url)
            return _clarify(
                f"{url!r} doesn't look like a valid URL. "
                "Please include https:// or www."
            )

        try:
            result: str = self._browser.open_url(url)
        except Exception:
            logger.exception("open_url() raised for url=%r.", url)
            return _err(f"I couldn't open {url!r}.")

        if not result or not result.strip():
            logger.warning("open_url() returned empty result for %r.", url)
            return _err(f"I opened {url!r} but received no response.")

        logger.info("open_url: navigated to %r.", url)
        return _ok(result.strip(), confidence=0.9)


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _extract(entities: dict, *keys: str) -> str:
    """Return the first non-empty string value found under any of *keys*."""
    for key in keys:
        value = entities.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _looks_like_url(value: str) -> bool:
    """
    Return True if *value* plausibly represents a URL.

    Accepts:
    - Strings starting with ``http://`` or ``https://``
    - Strings starting with ``www.``
    - Strings containing a dot followed by a known TLD token
      (lightweight heuristic, not RFC-compliant)
    """
    lower = value.lower()
    if lower.startswith(("http://", "https://", "www.")):
        return True
    # Minimal heuristic: at least one dot with something on both sides
    parts = lower.split(".")
    return len(parts) >= 2 and all(p for p in parts)


def _ok(
    response: str,
    data: Optional[dict[str, Any]] = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict[str, Any]:
    return {"response": response, "data": {}, "confidence": 0.0}


def _clarify(question: str) -> dict[str, Any]:
    return {
        "response": question,
        "data": {"needs_clarification": True},
        "confidence": 0.5,
    }