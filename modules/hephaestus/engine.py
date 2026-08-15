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
import os
import platform
import re
import subprocess
from typing import Any, Optional

from modules.base import BaseModule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NAVIGATE_ACTIONS: frozenset[str] = frozenset(
    {"open", "browse", "navigate", "go", "go to", "visit", "load"}
)

_NOT_AVAILABLE = (
    "Browser automation is not available. "
    "Ask me to enable it or check that the browser agent is configured."
)
_UNHANDLED = "I'm not sure what browser action to take."
_APP_UNAVAILABLE = "Desktop application launching is not available on this platform."

# Default desktop-app name → executable/path map. Overridable per-deployment
# via HephaestusEngine(app_map=...) (wired from config in main.py), since
# paths like the Spotify example are per-user and can't be hardcoded safely.
_DEFAULT_APP_MAP: dict[str, str] = {
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "firefox": "firefox.exe",
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "spotify": "spotify.exe",
}


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
    ``scrape_page``
        Fetch and extract readable text from a URL, or search for a topic
        and scrape the top result.
    ``open_app``
        Launch a native desktop application (not browser automation).

    Parameters
    ----------
    browser_agent:
        A ``HestiaBrowserAgent`` instance (or compatible duck-typed object).
        Injected by the orchestrator at startup.
    app_map:
        Optional override/extension of the built-in desktop-app name →
        executable map used by ``open_app``. Merged over the defaults so a
        deployment only needs to specify the apps it wants to add or change
        (e.g. a user-specific Spotify install path).
    """

    name = "hephaestus"

    _INTENTS: frozenset[str] = frozenset(
        {
            "browser_action",
            "search_web",
            "check_flight",
            "scrape_page",
            "open_app",
        }
    )

    def __init__(self, browser_agent: Any = None, app_map: Optional[dict[str, str]] = None) -> None:
        self._browser = browser_agent
        # Normalise all keys to lowercase at merge time. Lookups in
        # _open_app() always match on app_name.strip().lower(), so a
        # deployment-supplied override keyed with natural capitalisation
        # (e.g. {"Spotify": "D:/Spotify/Spotify.exe"} in config) would
        # otherwise silently fail to override the default "spotify" entry
        # — it'd just sit in the map as an unreachable extra key while
        # "open spotify" kept resolving to the built-in "spotify.exe".
        self._app_map: dict[str, str] = {
            **_DEFAULT_APP_MAP,
            **{k.strip().lower(): v for k, v in (app_map or {}).items()},
        }
        logger.info(
            "HephaestusEngine ready (browser_agent=%s, %d app(s) mapped).",
            type(browser_agent).__name__ if browser_agent else "None",
            len(self._app_map),
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Dispatch an intent to the appropriate handler.

        Returns a "not available" response when the browser agent is absent
        — except for ``open_app``, which launches a native process and has
        no dependency on browser automation being configured. Never raises.
        """
        if intent != "open_app" and not self._is_ready():
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
        if intent == "scrape_page":
            return self._scrape_page(entities)
        if intent == "open_app":
            return self._open_app(entities)
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
            if not action or action in _NAVIGATE_ACTIONS:
                return self._open_url(url)
            logger.debug(
                "_browser_action: action=%r is not a recognized navigation verb "
                "but a url is present; opening it anyway (no other url-consuming "
                "action exists in this module).",
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

    def _scrape_page(self, entities: dict) -> dict:
        """
        Fetch and extract readable text.

        Resolution order
        -----------------
        1. If a ``url`` entity is present, scrape that URL directly.
        2. Otherwise, if a ``query``/``topic`` is present, search for it and
           scrape the top result (e.g. "scrape wikipedia for Alan Turing").
        3. Otherwise ask for clarification.
        """
        url = (entities.get("url") or "").strip()
        query = _extract(entities, "query", "topic", "raw_query")

        if url:
            try:
                text = self.scrape_url(url)
            except Exception:
                logger.exception("scrape_url() raised for url=%r.", url)
                return _err(f"I couldn't read {url!r}.")
            if not text:
                return _err(f"I couldn't extract any readable content from {url!r}.")
            return _ok(text, data={"url": url}, confidence=0.85)

        if query:
            try:
                results = self.search_and_summarize(query)
            except Exception:
                logger.exception("search_and_summarize() raised for query=%r.", query[:80])
                return _err("I couldn't complete that search.")
            if not results:
                return _clarify(f"I couldn't find anything to scrape for {query!r}.")
            top = results[0]
            return _ok(
                top.get("summary") or top.get("title") or "",
                data={"results": results, "query": query},
                confidence=0.8,
            )

        return _clarify(
            "What should I scrape? Give me a URL, or a topic to search and read."
        )

    def _open_app(self, entities: dict) -> dict:
        """Launch a native desktop application by name (not browser automation)."""
        app_name = _extract(entities, "app", "app_name", "application", "name")
        if not app_name:
            # Last-resort fallback: NLU sometimes omits the app entity
            # entirely. Strip a leading "open"/"launch"/"start" so at least
            # "open chrome" (with no entities at all) resolves to "chrome"
            # instead of failing an app-map lookup on the whole sentence.
            raw = (entities.get("raw_query") or "").strip().lower()
            app_name = re.sub(r'^(open|launch|start)\s+', '', raw).strip()
        if not app_name:
            return _clarify("Which application should I open?")

        # The NLU sometimes classifies "open <url>" (e.g. "open google.com")
        # as open_app instead of browser_action, handing us a domain string
        # that will never be in _app_map — it's not a desktop app at all.
        # Rather than reporting "Unknown application: google.com", detect
        # the URL shape here and redirect to the browser instead, same as
        # if browser_action had been picked correctly in the first place.
        if app_name.strip().lower() not in self._app_map and _looks_like_url(app_name):
            logger.info(
                "_open_app: %r looks like a URL, not an app; redirecting to browser_action.",
                app_name,
            )
            if not self._is_ready():
                return _err(_NOT_AVAILABLE)
            url = app_name.strip()
            if not url.lower().startswith(("http://", "https://")):
                url = f"https://{url}"
            return self._open_url(url)

        target = self._app_map.get(app_name.strip().lower())
        if not target:
            logger.warning("_open_app: unknown application %r.", app_name)
            return _err(
                f"Unknown application: {app_name}. "
                f"I can open: {', '.join(sorted(set(self._app_map)))}."
            )

        try:
            _launch_app(target)
        except FileNotFoundError:
            logger.warning("_open_app: executable not found for %r (%r).", app_name, target)
            return _err(
                f"I couldn't find {app_name} on this system "
                f"(looked for {target!r}). Is it installed?"
            )
        except Exception:
            logger.exception("_open_app: launch failed for %r (%r).", app_name, target)
            return _err(f"I couldn't open {app_name}.")

        logger.info("_open_app: launched %r (%r).", app_name, target)
        return _ok(f"Opening {app_name}.", data={"app": app_name}, confidence=0.9)

    # ------------------------------------------------------------------
    # Reusable helpers for OTHER modules to call directly
    # ------------------------------------------------------------------
    #
    # These are plain methods a god can call on an injected HephaestusEngine
    # instance to get web content without going through Hecate/NLU/intent
    # routing — e.g. Dionysus wanting "cheap restaurants nearby" doesn't
    # need a full NLU round-trip just to fetch a web page.

    def scrape_url(self, url: str) -> str:
        """Fetch *url* and return its readable text content."""
        if not self._is_ready():
            raise HephaestusError("Browser automation is not available.")
        try:
            text: str = self._browser.get_page_text(url)
        except Exception as exc:
            # This method (and search_and_summarize below) is documented as
            # a direct-call helper for OTHER modules that bypass handle()'s
            # own try/except — so a raw browser-agent crash must not escape
            # here uncaught, same guarantee every intent handler above gets.
            raise BrowserAgentError(f"get_page_text failed for {url!r}: {exc}") from exc
        return (text or "").strip()

    def search_and_summarize(self, query: str, max_results: int = 3) -> list[dict[str, str]]:
        """
        Search the web for *query* and return up to *max_results* results,
        each scraped for a short text summary.

        Returns a list of ``{"title": ..., "url": ..., "summary": ...}``
        dicts (``url``/``summary`` may be empty if unavailable — the browser
        agent's search only guarantees titles; scraping each result page is
        best-effort and failures are skipped rather than raised).
        """
        if not self._is_ready():
            raise HephaestusError("Browser automation is not available.")

        try:
            raw_results = self._browser.search_web_results(query, max_results=max_results) \
                if hasattr(self._browser, "search_web_results") \
                else [{"title": t, "url": ""} for t in _split_titles(self._browser.search_web(query))]
        except Exception as exc:
            raise BrowserAgentError(f"web search failed for {query!r}: {exc}") from exc

        out: list[dict[str, str]] = []
        for r in raw_results[:max_results]:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            summary = ""
            if url:
                try:
                    summary = self.scrape_url(url)
                except Exception:
                    logger.debug("search_and_summarize: scrape failed for %r; skipping summary.", url)
            out.append({"title": title, "url": url, "summary": summary})
        return out


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _launch_app(target: str) -> None:
    """
    Launch a native application by executable name or path.

    Windows: relies on PATH resolution (``os.startfile`` / bare exe name
    via Popen) so a name like ``"chrome.exe"`` works without a hardcoded
    absolute path as long as it's on PATH; a full path works too.
    macOS/Linux: falls back to ``open``/``xdg-open`` respectively, since
    the default app map is Windows-oriented (this repo runs on Windows —
    see main.py — but the fallback keeps this module importable/testable
    elsewhere).

    Raises
    ------
    FileNotFoundError
        If the executable cannot be found.
    """
    system = platform.system()
    if system == "Windows":
        os.startfile(target)  # noqa: S606 - intentional, user-directed launch
        return
    if system == "Darwin":
        subprocess.Popen(["open", target])
        return
    subprocess.Popen(["xdg-open", target])


def _split_titles(joined: str) -> list[str]:
    """Fallback for browser agents without search_web_results(): split the
    legacy ``" | "``-joined title string search_web() returns."""
    if not joined or joined.startswith("No results") or joined.startswith("Search failed"):
        return []
    return [t.strip() for t in joined.split(" | ") if t.strip()]


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