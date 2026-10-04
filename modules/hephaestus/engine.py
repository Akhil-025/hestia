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
- Everything added for backlog #101-#110 is optional and off unless wired in
  (monitor store, scrapers, form profiles, repo roots, politeness delay), so
  an engine built the old way, ``HephaestusEngine(browser, app_map)``, behaves
  exactly as before.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from modules.base import BaseModule
from modules.hephaestus.monitor import (
    DEFAULT_INTERVAL_MINUTES,
    MIN_INTERVAL_MINUTES,
    FAILURE_ALERT_AFTER,
    MonitorStore,
    analyse_change,
    extract_price,
    format_price,
    host_of,
    normalize,
    validate_monitor_url,
)
from modules.hephaestus.repo_summary import format_summary, summarize_repo
from modules.hephaestus.scrapers import ScraperRegistry

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
_NO_MONITORING = (
    "Page watching isn't set up. It needs a monitor database; check the "
    "hephaestus section of your config."
)

# Intents that don't drive the browser, so they work even when it's absent.
_BROWSERLESS_INTENTS: frozenset[str] = frozenset(
    {"open_app", "list_watches", "stop_watching", "summarize_repo"}
)

# Text the browser agent returns *instead of* page content when it fails. Only
# relevant to agents without fetch_text(): a monitor must never record one of
# these as the page's new content.
_AGENT_ERROR_TEXTS: frozenset[str] = frozenset({
    "Browser is not available.",
    "I couldn't read that page.",
    "Page loaded but no readable content found.",
})

_MAX_CHECKS_PER_RUN = 10        # monitors fetched per heartbeat tick / "check now"
_MONITOR_FETCH_TIMEOUT = 120.0  # seconds before one monitor fetch counts as failed
_SCRAPER_TEXT_CAP = 2000        # chars of site-scraper output handed back to chat
_MAX_POLITE_WAIT = 10.0         # never stall a request longer than this
_PRICE_WORDS = re.compile(r"price|cheaper|discount|sale|deal|drop", re.I)

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
    ``watch_page`` / ``list_watches`` / ``stop_watching`` / ``check_watches``
        Scheduled page monitoring (#101, #109): record a baseline, re-check on
        the heartbeat, and speak up only about meaningful changes.
    ``fill_form``
        Fill (and optionally submit) a form saved in ``hephaestus.forms``
        (#102), behind the usual say-yes confirmation.
    ``summarize_repo``
        Structure/health summary of a local code folder (#110).

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
    monitor_store:
        ``MonitorStore`` backing the watch intents. Without one they reply
        that watching isn't set up.
    monitor_browser:
        A *separate* browser agent for monitors. Playwright's sync API is tied
        to the thread that started it, so monitor fetches always run on one
        dedicated worker thread; sharing the chat browser would break whichever
        thread didn't launch it. Falls back to ``browser_agent``.
    scrapers:
        ``ScraperRegistry`` of site-specific scrapers (#105).
    forms:
        ``{name: {"url", "fields": {selector: value}, "submit_selector"}}``
        profiles for ``fill_form`` (#102). Invalid entries are dropped.
    repo_roots:
        Optional allow-list of folders ``summarize_repo`` may scan. Empty
        means any folder.
    min_host_interval:
        Seconds to leave between two requests to the same site (#106).
        0 disables the delay.
    quiet_hours:
        ``(start_hour, end_hour)`` during which monitor alerts are held back
        and delivered afterwards; may wrap midnight, e.g. ``(23, 7)``.
    """

    name = "hephaestus"

    _INTENTS: frozenset[str] = frozenset(
        {
            "browser_action",
            "search_web",
            "check_flight",
            "scrape_page",
            "open_app",
            "watch_page",
            "list_watches",
            "stop_watching",
            "check_watches",
            "fill_form",
            "summarize_repo",
        }
    )

    def __init__(
        self,
        browser_agent: Any = None,
        app_map: Optional[dict[str, str]] = None,
        *,
        monitor_store: Optional[MonitorStore] = None,
        monitor_browser: Any = None,
        scrapers: Optional[ScraperRegistry] = None,
        forms: Optional[dict[str, dict]] = None,
        repo_roots: Optional[list[str]] = None,
        min_host_interval: float = 0.0,
        quiet_hours: Optional[tuple[int, int]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now_fn: Optional[Callable[[], datetime]] = None,
        local_now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._browser = browser_agent
        self._monitor_store = monitor_store
        self._monitor_browser = monitor_browser or browser_agent
        self._scrapers = scrapers if scrapers is not None else ScraperRegistry()
        self._forms = _clean_forms(forms)
        self._repo_roots = [os.path.realpath(os.path.expanduser(r)) for r in (repo_roots or [])]
        self._min_host_interval = max(0.0, float(min_host_interval or 0.0))
        self._quiet_hours = _clean_quiet_hours(quiet_hours)
        self._clock = clock
        self._sleep = sleep
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._local_now = local_now_fn or datetime.now
        self._host_last: dict[str, float] = {}
        self._host_lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()
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
        — except for the intents in ``_BROWSERLESS_INTENTS`` (``open_app``
        launches a native process; listing/stopping watches and summarising a
        local folder never touch the browser). Never raises.
        """
        if intent not in _BROWSERLESS_INTENTS and not self._is_ready():
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
        ctx: dict[str, Any] = {"hephaestus_available": self._is_ready()}
        if self._monitor_store is not None:
            try:
                ctx["hephaestus_watches"] = len(self._monitor_store.list())
            except Exception:
                logger.debug("get_context: could not count watches.", exc_info=True)
        return ctx

    def close(self) -> None:
        """Release the monitor worker thread and the dedicated monitor browser."""
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if self._monitor_browser is not None and self._monitor_browser is not self._browser:
            try:
                if pool is not None:
                    pool.submit(self._monitor_browser.close).result(timeout=15)
                else:
                    self._monitor_browser.close()
            except Exception:
                logger.debug("monitor browser close raised; ignoring.", exc_info=True)
        if pool is not None:
            pool.shutdown(wait=False)

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
        if intent == "watch_page":
            return self._watch_page(entities)
        if intent == "list_watches":
            return self._list_watches()
        if intent == "stop_watching":
            return self._stop_watching(entities)
        if intent == "check_watches":
            return self._check_watches(entities)
        if intent == "fill_form":
            return self._fill_form(entities)
        if intent == "summarize_repo":
            return self._summarize_repo(entities)
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
    # Page monitoring (#101 scheduled checks, #109 change detection)
    # ------------------------------------------------------------------

    def _watch_page(self, entities: dict) -> dict:
        """Start watching a page: read it once now, store the baseline, and let
        the heartbeat re-check it. Refuses unreadable or unsafe pages up front
        rather than leaving a watch that can never work."""
        store = self._monitor_store
        if store is None:
            return _err(_NO_MONITORING)
        raw_url = _extract(entities, "url", "link", "page", "site")
        if not raw_url:
            return _clarify("Which page should I watch? Give me its address.")
        url, problem = validate_monitor_url(raw_url)
        if problem:
            return _clarify(f"I can't watch that: {problem}.")

        keywords = _as_list(entities.get("keywords") or entities.get("keyword"))
        watch_for = _extract(entities, "watch_for", "condition")
        track_price = _truthy(entities.get("track_price")) or bool(
            watch_for and _PRICE_WORDS.search(watch_for)
        ) or bool(_extract(entities, "target_price", "price_below"))
        target_price = None
        raw_target = _extract(entities, "target_price", "price_below")
        if raw_target:
            target_price = _parse_price_value(raw_target)
            if target_price is None:
                return _clarify(f"I couldn't read {raw_target!r} as a price. What price should I alert you at?")
        selector = _extract(entities, "selector", "css_selector") or None
        interval = _parse_interval_minutes(entities)
        if interval == 0:
            return _clarify("How often should I check? For example: every hour, daily, or weekly.")
        interval = interval or DEFAULT_INTERVAL_MINUTES
        clamped = interval < MIN_INTERVAL_MINUTES
        interval = max(interval, MIN_INTERVAL_MINUTES)

        name = _extract(entities, "name", "label", "title") or _default_watch_name(url)
        if any(m["name"].lower() == name.lower() for m in store.list()):
            return _err(f"I'm already watching something called {name!r}. Stop that one first, or give this a different name.")

        scraper = self._scrapers.find(url)
        draft = {
            "name": name, "url": url, "selector": selector,
            "scraper": scraper.name if scraper else None,
        }
        now = self._now()
        try:
            text = self._run_in_monitor_thread(lambda: self._fetch_monitor_snapshot(draft))
        except Exception:
            logger.exception("watch_page: baseline fetch raised for %r.", url)
            text = None
        finally:
            self._release_monitor_browser()
        if not text or not text.strip():
            return _err(
                f"I couldn't read {url}, so I haven't set up the watch. "
                "Check the address, or try again in a bit."
            )
        snapshot = normalize(text)
        first = analyse_change(
            None, snapshot, keywords=keywords, track_price=track_price,
            target_price=target_price,
        )
        if track_price and first["price"] is None:
            return _clarify(
                "I couldn't find a price on that page. Give me a CSS selector for the "
                "price, or I can watch for keywords instead."
            )
        try:
            monitor_id = store.add(
                name=name, url=url, now=now, keywords=keywords, track_price=track_price,
                target_price=target_price, selector=selector,
                scraper=draft["scraper"], interval_minutes=interval,
            )
        except ValueError as exc:
            if str(exc) == "limit":
                return _err(
                    f"I'm already watching the maximum of {store.max_monitors} pages. "
                    "Stop one before adding another."
                )
            return _err(f"I'm already watching something called {name!r}.")
        store.record_success(
            monitor_id, snapshot=snapshot, now=now, changed=False,
            price=first["price"], currency=first["symbol"],
        )
        for alert in first["alerts"]:   # e.g. already at or below the target price
            store.add_alert(monitor_id, f"{name}: {alert}", now)

        bits = [f"Watching {name}, checking {_describe_interval(interval)}."]
        if clamped:
            bits.append(f"That's as often as I check ({MIN_INTERVAL_MINUTES} minutes).")
        if track_price:
            bits.append(f"The price right now is {format_price(first['price'], first['symbol'])}.")
        if keywords:
            bits.append("I'll tell you when " + _join_words([f"“{k}”" for k in keywords]) + " appears.")
        if not keywords and not track_price:
            bits.append("I'll tell you when the content meaningfully changes.")
        if scraper:
            bits.append(f"Using the {scraper.name} scraper.")
        return _ok(" ".join(bits), data={"monitor_id": monitor_id, "name": name, "url": url,
                                         "interval_minutes": interval}, confidence=0.9)

    def _list_watches(self) -> dict:
        store = self._monitor_store
        if store is None:
            return _err(_NO_MONITORING)
        monitors = store.list()
        if not monitors:
            return _ok("You're not watching any pages.", data={"watches": []}, confidence=0.9)
        lines = []
        for m in monitors:
            what = []
            if m["keywords"]:
                what.append("for " + _join_words([f"“{k}”" for k in m["keywords"]]))
            if m["track_price"]:
                what.append("price" + (
                    f" (now {format_price(m['last_price'], m['currency'])})" if m["last_price"] is not None else ""))
            if not what:
                what.append("any meaningful change")
            state = ""
            if m["failures"]:
                state = f", failing ({m['failures']} in a row)"
            lines.append(f"{m['name']} ({', '.join(what)}; {_describe_interval(m['interval_minutes'])}{state})")
        n = len(monitors)
        return _ok(
            f"You're watching {n} page{'s' if n != 1 else ''}: " + "; ".join(lines) + ".",
            data={"watches": [{k: m[k] for k in ("id", "name", "url", "interval_minutes",
                                                 "keywords", "track_price", "last_checked")}
                              for m in monitors]},
            confidence=0.9,
        )

    def _stop_watching(self, entities: dict) -> dict:
        store = self._monitor_store
        if store is None:
            return _err(_NO_MONITORING)
        target = _extract(entities, "name", "label", "url", "page", "query", "raw_query")
        monitors = store.list()
        if not monitors:
            return _ok("You're not watching any pages.", confidence=0.9)
        names = ", ".join(m["name"] for m in monitors)
        if not target:
            return _clarify(f"Which one should I stop watching? You have: {names}.")
        matches = store.find(target)
        if not matches:
            # NLU often hands over the whole sentence; accept a watch name inside it.
            low = target.lower()
            matches = [m for m in monitors if m["name"].lower() in low]
        if not matches:
            return _clarify(f"I don't have a watch matching {target!r}. You have: {names}.")
        if len(matches) > 1:
            return _clarify("That matches more than one watch: " + ", ".join(m["name"] for m in matches) + ". Which one?")
        m = matches[0]
        store.remove(m["id"])
        return _ok(f"Okay, I've stopped watching {m['name']}.", data={"name": m["name"]}, confidence=0.9)

    def _check_watches(self, entities: dict) -> dict:
        """Check watched pages right now instead of waiting for the heartbeat."""
        store = self._monitor_store
        if store is None:
            return _err(_NO_MONITORING)
        monitors = store.list()
        if not monitors:
            return _ok("You're not watching any pages.", confidence=0.9)
        target = _extract(entities, "name", "label", "url", "page")
        if target:
            monitors = store.find(target)
            if not monitors:
                return _clarify(f"I don't have a watch matching {target!r}.")
        checked = self._run_checks(monitors[:_MAX_CHECKS_PER_RUN])
        text = self._deliver_alerts(respect_quiet=False)
        tail = ""
        if len(monitors) > _MAX_CHECKS_PER_RUN:
            tail = f" I checked the first {_MAX_CHECKS_PER_RUN}; ask again for the rest."
        if text:
            return _ok(text + tail, data={"checked": checked}, confidence=0.9)
        n = len(monitors[:_MAX_CHECKS_PER_RUN])
        return _ok(f"Nothing new on {n} watched page{'s' if n != 1 else ''}." + tail,
                   data={"checked": checked}, confidence=0.9)

    def check_web_monitors(self) -> Optional[str]:
        """Heartbeat hook (#101): re-check whatever is due, return text to speak.

        Cadence lives here (each monitor has its own interval), so calling this
        on every tick is safe. Alerts raised during quiet hours are kept and
        delivered on the first tick after they end. Returns ``None`` when there
        is nothing to say. Never raises."""
        store = self._monitor_store
        if store is None or self._monitor_browser is None:
            return None
        try:
            due = store.due(self._now())[:_MAX_CHECKS_PER_RUN]
            if due:
                self._run_checks(due)
            return self._deliver_alerts(respect_quiet=True)
        except Exception:
            logger.exception("check_web_monitors failed.")
            return None

    # -- monitor internals ------------------------------------------------

    def _run_checks(self, monitors: list[dict]) -> int:
        done = 0
        try:
            for m in monitors:
                try:
                    self._check_monitor(m)
                    done += 1
                except Exception:
                    logger.exception("Checking monitor %r failed.", m.get("name"))
        finally:
            self._release_monitor_browser()
        return done

    def _check_monitor(self, m: dict) -> None:
        store = self._monitor_store
        now = self._now()
        try:
            text = self._run_in_monitor_thread(lambda: self._fetch_monitor_snapshot(m))
        except Exception:
            logger.warning("Monitor %r: fetch failed.", m["name"], exc_info=True)
            text = None
        if not text or not text.strip():
            count, should_alert = store.record_failure(m["id"], now)
            if should_alert:
                store.add_alert(
                    m["id"],
                    f"I couldn't read {m['name']} for {count} checks in a row. "
                    "The page may be down or its layout may have changed.",
                    now,
                )
            return
        snapshot = normalize(text)
        result = analyse_change(
            m["snapshot"], snapshot, keywords=m["keywords"], track_price=m["track_price"],
            old_price=m["last_price"], target_price=m["target_price"],
        )
        store.record_success(
            m["id"], snapshot=snapshot, now=now,
            changed=bool(m["snapshot"]) and snapshot != m["snapshot"],
            price=result["price"], currency=result["symbol"],
        )
        for alert in result["alerts"]:
            store.add_alert(m["id"], f"{m['name']}: {alert}", now)

    def _fetch_monitor_snapshot(self, m: dict) -> Optional[str]:
        """Page text for one monitor, or ``None`` on any failure."""
        browser = self._monitor_browser
        url = m["url"]
        self._polite_wait(url)
        scraper = self._scrapers.get(m["scraper"]) if m.get("scraper") else None
        if scraper is not None:
            return scraper.scrape(url, browser)
        if m.get("selector"):
            fetch_elements = getattr(browser, "fetch_elements", None)
            if not callable(fetch_elements):
                return None
            items = fetch_elements(url, m["selector"])
            return "\n".join(items) if items else None
        fetch_text = getattr(browser, "fetch_text", None)
        if callable(fetch_text):
            return fetch_text(url)
        text = browser.get_page_text(url)
        return None if (not text or text in _AGENT_ERROR_TEXTS) else text

    def _run_in_monitor_thread(self, fn: Callable[[], Any]) -> Any:
        """Run *fn* on the single monitor worker thread (see class docs)."""
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hephaestus-monitor")
            pool = self._pool
        return pool.submit(fn).result(timeout=_MONITOR_FETCH_TIMEOUT)

    def _release_monitor_browser(self) -> None:
        """Close a dedicated monitor browser after a batch so a daily check
        doesn't leave Chromium running all day. Never closes the chat browser."""
        browser = self._monitor_browser
        if browser is None or browser is self._browser:
            return
        close = getattr(browser, "close", None)
        if not callable(close):
            return
        try:
            self._run_in_monitor_thread(close)
        except Exception:
            logger.debug("monitor browser close raised; ignoring.", exc_info=True)

    def _deliver_alerts(self, *, respect_quiet: bool) -> Optional[str]:
        store = self._monitor_store
        if respect_quiet and self._in_quiet_hours():
            return None
        alerts = store.pending_alerts()
        if not alerts:
            return None
        store.mark_delivered([a["id"] for a in alerts])
        texts = [a["text"] for a in alerts]
        if len(texts) > 3:
            return " ".join(texts[:3]) + f" Plus {len(texts) - 3} more watch alerts."
        return " ".join(texts)

    def _in_quiet_hours(self) -> bool:
        if not self._quiet_hours:
            return False
        start, end = self._quiet_hours
        hour = self._local_now().hour
        return start <= hour < end if start < end else (hour >= start or hour < end)

    def _polite_wait(self, url: str) -> None:
        """Space out requests to the same site (#106). Slots are reserved under
        a lock, so two threads hitting one host queue instead of racing."""
        if self._min_host_interval <= 0:
            return
        host = host_of(url)
        if not host:
            return
        with self._host_lock:
            now = self._clock()
            last = self._host_last.get(host)
            wait = 0.0 if last is None else max(0.0, self._min_host_interval - (now - last))
            wait = min(wait, _MAX_POLITE_WAIT)
            self._host_last[host] = now + wait
        if wait > 0:
            self._sleep(wait)

    # ------------------------------------------------------------------
    # Form filling (#102)
    # ------------------------------------------------------------------

    def _fill_form(self, entities: dict) -> dict:
        """Fill a form saved under ``hephaestus.forms``.

        Selectors and values live in config, never in what's spoken, so a
        misheard command can't invent fields. Always two-phase: the first call
        only describes what would happen (host and field count, never the
        values) and the orchestrator asks for a yes before the second call
        touches the browser. Nothing is submitted unless the saved profile
        has a ``submit_selector``."""
        if not self._forms:
            return _err(
                "No forms are saved yet. Add them under hephaestus.forms in your config "
                "with a URL, the fields to fill and optionally a submit button."
            )
        listing = ", ".join(f["label"] for f in self._forms.values())
        query = _extract(entities, "form", "form_name", "name", "query", "raw_query")
        if not query:
            return _clarify(f"Which form should I fill? I have: {listing}.")
        low = query.lower()
        hits = [k for k in self._forms if k in low]
        if not hits:
            return _clarify(f"I don't have a form matching {query!r}. I have: {listing}.")
        key = max(hits, key=len)
        if sum(1 for k in hits if len(k) == len(key)) > 1:
            return _clarify(f"That matches more than one form: {listing}. Which one?")
        profile = self._forms[key]

        values = entities.get("values")
        values = {str(k): str(v) for k, v in values.items()} if isinstance(values, dict) else {}
        fields: dict[str, str] = {}
        missing: list[str] = []
        for selector, template in profile["fields"].items():
            def _fill(m, _missing=missing):
                name = m.group(1)
                if name not in values:
                    if name not in _missing:
                        _missing.append(name)
                    return m.group(0)
                return values[name]
            fields[selector] = _PLACEHOLDER.sub(_fill, template)
        if missing:
            return _clarify("To fill that form I also need: " + ", ".join(missing) + ".")

        submit = profile["submit_selector"]
        host = host_of(profile["url"])
        if not entities.get("_confirmed"):
            action = "and submit it" if submit else "without submitting it"
            keep = {k: v for k, v in entities.items() if k != "_confirmed"}
            keep["form"] = profile["label"]
            return {
                "response": (
                    f"Fill in the {profile['label']} form at {host} ({len(fields)} "
                    f"field{'s' if len(fields) != 1 else ''}) {action}? Say yes to go ahead."
                ),
                "data": {"form": profile["label"], "url": profile["url"], "fields": len(fields),
                         "submit": bool(submit)},
                "confidence": 0.9,
                "needs_confirmation": True,
                "confirm_intent": "fill_form",
                "confirm_entities": keep,
                "confirm_label": f"fill in the {profile['label']} form",
            }

        self._polite_wait(profile["url"])
        try:
            result = self._browser.fill_form(profile["url"], fields, submit)
        except Exception:
            logger.exception("fill_form() raised for form %r.", profile["label"])
            return _err(f"Something went wrong filling the {profile['label']} form.")
        if not result or not str(result).strip():
            return _err(f"I tried the {profile['label']} form but got no response from the browser.")
        result = str(result).strip()
        return _ok(result, data={"form": profile["label"], "submitted": result.startswith("Form submitted")},
                   confidence=0.85)

    # ------------------------------------------------------------------
    # Repository summary (#110)
    # ------------------------------------------------------------------

    def _summarize_repo(self, entities: dict) -> dict:
        raw = _extract(entities, "path", "repo", "folder", "directory", "project")
        if not raw:
            return _clarify("Which folder should I look at? Give me its path.")
        path = os.path.realpath(os.path.expanduser(raw.strip().strip("\"'")))
        if not os.path.isdir(path):
            return _err(f"I couldn't find a folder at {raw!r}.")
        if self._repo_roots and not any(
            path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in self._repo_roots
        ):
            return _err("That folder is outside the places I'm allowed to scan.")
        try:
            summary = summarize_repo(path)
        except Exception:
            logger.exception("summarize_repo() raised for %r.", path)
            return _err(f"I couldn't scan {raw!r}.")
        return _ok(format_summary(summary), data={"summary": summary}, confidence=0.85)

    # ------------------------------------------------------------------
    # Reusable helpers for OTHER modules to call directly
    # ------------------------------------------------------------------
    #
    # These are plain methods a god can call on an injected HephaestusEngine
    # instance to get web content without going through Hecate/NLU/intent
    # routing — e.g. Dionysus wanting "cheap restaurants nearby" doesn't
    # need a full NLU round-trip just to fetch a web page.

    def scrape_url(self, url: str) -> str:
        """Fetch *url* and return its readable text content.

        A registered site scraper (#105) is tried first and its output is
        returned in full (up to ``_SCRAPER_TEXT_CAP`` chars); if none matches,
        or it can't read the page, the generic scraper runs. Requests to the
        same site are spaced by ``min_host_interval`` (#106).
        """
        if not self._is_ready():
            raise HephaestusError("Browser automation is not available.")
        self._polite_wait(url)
        scraper = self._scrapers.find(url)
        if scraper is not None:
            try:
                scraped = scraper.scrape(url, self._browser)
            except Exception:
                logger.exception("Scraper %r raised for %r; using generic scrape.", scraper.name, url)
                scraped = None
            if scraped and scraped.strip():
                return scraped.strip()[:_SCRAPER_TEXT_CAP]
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


# ---------------------------------------------------------------------------
# Helpers for the monitor / form / config features
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_INTERVAL_UNITS = {"minute": 1, "min": 1, "hour": 60, "hr": 60, "day": 24 * 60, "week": 7 * 24 * 60}


def _parse_price_value(raw: Any) -> Optional[float]:
    """'₹50,000', 'Rs. 499.50', '$20' or a bare number -> float, else None."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(raw))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_list(value: Any) -> list[str]:
    """Entity that may be a list or a comma/\"and\"-separated string."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = re.split(r",|;|\band\b|\bor\b", str(value))
    return [i.strip().strip("\"'“”") for i in items if i and i.strip().strip("\"'“”")]


def _join_words(words: list[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " or " + words[-1]


def _default_watch_name(url: str) -> str:
    host = host_of(url).removeprefix("www.")
    path = url.split("://", 1)[-1].split("/", 1)[1:] or [""]
    tail = path[0].split("?")[0].strip("/").split("/")[-1]
    return f"{host} {tail}".strip() if tail else host


def _parse_interval_minutes(entities: dict) -> Optional[int]:
    """Minutes between checks. ``None`` = not specified; ``0`` = unparseable."""
    for key, factor in (("interval_minutes", 1), ("interval_hours", 60), ("interval_days", 1440)):
        raw = entities.get(key)
        if raw not in (None, ""):
            try:
                return max(0, int(float(raw) * factor))
            except (TypeError, ValueError):
                return 0
    text = _extract(entities, "interval", "frequency", "schedule").lower()
    if not text:
        return None
    if re.search(r"twice a day|twice daily", text):
        return 12 * 60
    if re.fullmatch(r"(every\s*)?(hourly|hour)|every hour|each hour", text):
        return 60
    if re.fullmatch(r"(daily|every day|each day|day|once a day)", text):
        return 24 * 60
    if re.fullmatch(r"(weekly|every week|each week|week|once a week)", text):
        return 7 * 24 * 60
    m = re.search(r"(\d+(?:\.\d+)?)\s*(minute|min|hour|hr|day|week)s?", text)
    if m:
        return max(0, int(float(m.group(1)) * _INTERVAL_UNITS[m.group(2)]))
    return 0


def _describe_interval(minutes: int) -> str:
    if minutes % (7 * 24 * 60) == 0:
        n = minutes // (7 * 24 * 60)
        return "weekly" if n == 1 else f"every {n} weeks"
    if minutes % (24 * 60) == 0:
        n = minutes // (24 * 60)
        return "daily" if n == 1 else f"every {n} days"
    if minutes % 60 == 0:
        n = minutes // 60
        return "hourly" if n == 1 else f"every {n} hours"
    return f"every {minutes} minutes"


def _clean_quiet_hours(value: Any) -> Optional[tuple[int, int]]:
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (0 <= start <= 23 and 0 <= end <= 24) or start == end:
        return None
    return start, end


def _clean_forms(forms: Optional[dict]) -> dict[str, dict]:
    """Validate ``hephaestus.forms`` config. A bad profile is logged and
    skipped; it must not stop the module (or Hestia) from starting."""
    out: dict[str, dict] = {}
    for name, cfg in (forms or {}).items():
        label = str(name).strip()
        try:
            url, problem = validate_monitor_url(str((cfg or {}).get("url", "")))
            fields = (cfg or {}).get("fields") or {}
            if problem or not label or not isinstance(fields, dict) or not fields:
                raise ValueError(problem or "needs a name, a url and at least one field")
            out[label.lower()] = {
                "label": label,
                "url": url,
                "fields": {str(k): str(v) for k, v in fields.items()},
                "submit_selector": (str((cfg or {}).get("submit_selector") or "").strip() or None),
            }
        except Exception as exc:
            logger.warning("Ignoring hephaestus form %r: %s", name, exc)
    return out
