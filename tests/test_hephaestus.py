# tests/test_hephaestus.py
"""
Extensive regression tests for modules/hephaestus/engine.py — Hephaestus,
Hestia's browser-automation / web-interaction module (navigate, search,
scrape, flight status, native desktop-app launching).

This module previously had zero test coverage. It is one of the two
highest-risk modules in the codebase (the other being Pluto/finance)
because a silent bug here means Hestia can click around the web, launch
processes, or hand back stale/garbage content without the user knowing.

Coverage includes: the BaseModule contract, readiness gating, every
registered intent's happy path / clarification path / error path, the
URL-vs-app disambiguation heuristic in `_open_app`, the reusable
`scrape_url` / `search_and_summarize` helpers other modules call directly,
and every module-level pure helper (`_looks_like_url`, `_extract`,
`_split_titles`, `_launch_app`, `_ok`/`_err`/`_clarify`).

Run with:  pytest tests/test_hephaestus.py -v
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.hephaestus.engine import (
    HephaestusEngine,
    HephaestusError,
    BrowserAgentError,
    _looks_like_url,
    _extract,
    _split_titles,
    _launch_app,
    _ok,
    _err,
    _clarify,
    _NOT_AVAILABLE,
    _DEFAULT_APP_MAP,
)


# ===========================================================================
# Fakes / fixtures
# ===========================================================================

class FakeBrowserAgent:
    """Stand-in for HestiaBrowserAgent with knobs for every method used."""

    def __init__(self):
        self.open_url_result = "Opened successfully."
        self.search_web_result = "Result A | Result B | Result C"
        self.scrape_result = "Some readable page text."
        self.flight_result = "AI202: On time, departing gate 14."
        self.search_results = None  # used if search_web_results is set
        self.raise_on = set()  # method names that should raise
        self.calls = []

    def _maybe_raise(self, name):
        if name in self.raise_on:
            raise RuntimeError(f"{name} boom")

    def open_url(self, url):
        self.calls.append(("open_url", url))
        self._maybe_raise("open_url")
        return self.open_url_result

    def search_web(self, query):
        self.calls.append(("search_web", query))
        self._maybe_raise("search_web")
        return self.search_web_result

    def get_page_text(self, url):
        self.calls.append(("get_page_text", url))
        self._maybe_raise("get_page_text")
        return self.scrape_result

    def check_flight_status(self, flight):
        self.calls.append(("check_flight_status", flight))
        self._maybe_raise("check_flight_status")
        return self.flight_result


class FakeBrowserAgentWithSearchResults(FakeBrowserAgent):
    """Variant exposing the richer search_web_results() API."""

    def search_web_results(self, query, max_results=3):
        self.calls.append(("search_web_results", query, max_results))
        self._maybe_raise("search_web_results")
        if self.search_results is not None:
            return self.search_results
        return [
            {"title": "Title 1", "url": "https://a.example.com"},
            {"title": "Title 2", "url": "https://b.example.com"},
            {"title": "Title 3", "url": ""},
        ]


def make_engine(browser=None, app_map=None):
    return HephaestusEngine(browser_agent=browser, app_map=app_map)


# ===========================================================================
# BaseModule contract
# ===========================================================================

class TestHephaestusContract:
    def test_can_handle_all_registered_intents(self):
        engine = make_engine(FakeBrowserAgent())
        for intent in (
            "browser_action", "search_web", "check_flight",
            "scrape_page", "open_app",
        ):
            assert engine.can_handle(intent) is True

    def test_can_handle_rejects_unknown_intent(self):
        engine = make_engine(FakeBrowserAgent())
        assert engine.can_handle("totally_unknown") is False

    def test_can_handle_is_case_sensitive(self):
        engine = make_engine(FakeBrowserAgent())
        assert engine.can_handle("Search_Web") is False

    def test_name_is_hephaestus(self):
        assert HephaestusEngine.name == "hephaestus"

    def test_get_context_reports_ready_when_browser_present(self):
        engine = make_engine(FakeBrowserAgent())
        assert engine.get_context() == {"hephaestus_available": True}

    def test_get_context_reports_not_ready_when_browser_absent(self):
        engine = make_engine(None)
        assert engine.get_context() == {"hephaestus_available": False}

    def test_handle_never_raises_on_unexpected_exception(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        with patch.object(engine, "_dispatch", side_effect=RuntimeError("boom")):
            result = engine.handle("search_web", {"query": "x"}, {})
        assert result["confidence"] == 0.0
        assert "went wrong" in result["response"].lower()

    def test_unknown_intent_via_dispatch_returns_unhandled(self):
        engine = make_engine(FakeBrowserAgent())
        result = engine._dispatch("not_a_real_intent", {})
        assert result["confidence"] == 0.0


# ===========================================================================
# Readiness gating
# ===========================================================================

class TestReadinessGating:
    def test_handle_returns_not_available_when_browser_absent(self):
        engine = make_engine(None)
        result = engine.handle("search_web", {"query": "cats"}, {})
        assert result["confidence"] == 0.0
        assert result["response"] == _NOT_AVAILABLE

    def test_handle_returns_not_available_for_scrape_when_browser_absent(self):
        engine = make_engine(None)
        result = engine.handle("scrape_page", {"url": "https://x.com"}, {})
        assert result["response"] == _NOT_AVAILABLE

    def test_handle_returns_not_available_for_check_flight_when_browser_absent(self):
        engine = make_engine(None)
        result = engine.handle("check_flight", {"flight": "AI202"}, {})
        assert result["response"] == _NOT_AVAILABLE

    def test_open_app_bypasses_readiness_gate(self):
        """open_app must work even with no browser agent configured — it
        launches a native process, unrelated to browser automation."""
        engine = make_engine(None)
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            result = engine.handle("open_app", {"app": "notepad"}, {})
        assert result["confidence"] == 0.9
        mock_launch.assert_called_once()

    def test_is_ready_true_with_browser(self):
        engine = make_engine(FakeBrowserAgent())
        assert engine._is_ready() is True

    def test_is_ready_false_without_browser(self):
        engine = make_engine(None)
        assert engine._is_ready() is False


# ===========================================================================
# check_flight
# ===========================================================================

class TestCheckFlight:
    def test_happy_path(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        result = engine.handle("check_flight", {"flight_number": "AI202"}, {})
        assert result["confidence"] == 0.9
        assert result["response"] == browser.flight_result
        assert browser.calls == [("check_flight_status", "AI202")]

    def test_falls_back_to_flight_entity(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("check_flight", {"flight": "EK507"}, {})
        assert browser.calls[0] == ("check_flight_status", "EK507")

    def test_falls_back_to_raw_query(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("check_flight", {"raw_query": "BA123"}, {})
        assert browser.calls[0] == ("check_flight_status", "BA123")

    def test_missing_flight_number_asks_for_clarification(self):
        engine = make_engine(FakeBrowserAgent())
        result = engine.handle("check_flight", {}, {})
        assert result["confidence"] == 0.5
        assert result["data"]["needs_clarification"] is True

    def test_browser_agent_exception_returns_graceful_error(self):
        browser = FakeBrowserAgent()
        browser.raise_on.add("check_flight_status")
        engine = make_engine(browser)
        result = engine.handle("check_flight", {"flight": "AI202"}, {})
        assert result["confidence"] == 0.0
        assert "AI202" in result["response"]

    def test_empty_result_returns_no_status_found(self):
        browser = FakeBrowserAgent()
        browser.flight_result = ""
        engine = make_engine(browser)
        result = engine.handle("check_flight", {"flight": "ZZ1"}, {})
        assert result["confidence"] == 0.0
        assert "No status information" in result["response"]

    def test_whitespace_only_result_treated_as_empty(self):
        browser = FakeBrowserAgent()
        browser.flight_result = "   \n  "
        engine = make_engine(browser)
        result = engine.handle("check_flight", {"flight": "ZZ1"}, {})
        assert result["confidence"] == 0.0

    def test_result_is_stripped(self):
        browser = FakeBrowserAgent()
        browser.flight_result = "  On time  \n"
        engine = make_engine(browser)
        result = engine.handle("check_flight", {"flight": "AI202"}, {})
        assert result["response"] == "On time"


# ===========================================================================
# search_web
# ===========================================================================

class TestSearchWeb:
    def test_happy_path(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        result = engine.handle("search_web", {"query": "best pizza"}, {})
        assert result["confidence"] == 0.85
        assert result["response"] == browser.search_web_result
        assert browser.calls == [("search_web", "best pizza")]

    def test_falls_back_to_topic_entity(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("search_web", {"topic": "quantum computing"}, {})
        assert browser.calls[0] == ("search_web", "quantum computing")

    def test_falls_back_to_raw_query(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("search_web", {"raw_query": "weather today"}, {})
        assert browser.calls[0] == ("search_web", "weather today")

    def test_missing_query_asks_for_clarification(self):
        engine = make_engine(FakeBrowserAgent())
        result = engine.handle("search_web", {}, {})
        assert result["confidence"] == 0.5

    def test_browser_agent_exception_returns_graceful_error(self):
        browser = FakeBrowserAgent()
        browser.raise_on.add("search_web")
        engine = make_engine(browser)
        result = engine.handle("search_web", {"query": "x"}, {})
        assert result["confidence"] == 0.0
        assert "couldn't complete" in result["response"].lower()

    def test_empty_result_reports_nothing_found(self):
        browser = FakeBrowserAgent()
        browser.search_web_result = ""
        engine = make_engine(browser)
        result = engine.handle("search_web", {"query": "asdkjfhaskjdfh"}, {})
        assert result["confidence"] == 0.0
        assert "didn't find anything" in result["response"].lower()


# ===========================================================================
# browser_action (navigate / search dispatch)
# ===========================================================================

class TestBrowserAction:
    def test_url_with_no_action_opens_directly(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        result = engine.handle("browser_action", {"url": "https://example.com"}, {})
        assert result["confidence"] == 0.9
        assert browser.calls == [("open_url", "https://example.com")]

    @pytest.mark.parametrize("verb", ["open", "browse", "navigate", "go", "go to", "visit", "load"])
    def test_url_with_navigation_verb_opens(self, verb):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("browser_action", {"url": "https://example.com", "action": verb}, {})
        assert browser.calls == [("open_url", "https://example.com")]

    def test_url_with_unrecognized_action_still_opens(self):
        """No other url-consuming action exists, so an unrecognised verb
        alongside a URL still opens it (documented fallback behaviour)."""
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("browser_action", {"url": "https://example.com", "action": "bookmark"}, {})
        assert browser.calls == [("open_url", "https://example.com")]

    def test_no_url_but_query_falls_back_to_search(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        result = engine.handle("browser_action", {"query": "cats"}, {})
        assert browser.calls == [("search_web", "cats")]
        assert result["confidence"] == 0.85

    def test_no_url_no_query_asks_for_clarification(self):
        engine = make_engine(FakeBrowserAgent())
        result = engine.handle("browser_action", {}, {})
        assert result["confidence"] == 0.5
        assert result["data"]["needs_clarification"] is True

    def test_action_and_query_with_no_url_prefers_search(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("browser_action", {"action": "search", "topic": "news"}, {})
        assert browser.calls == [("search_web", "news")]


# ===========================================================================
# _open_url (also exercised through browser_action above)
# ===========================================================================

class TestOpenUrl:
    def test_invalid_url_asks_for_clarification(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        result = engine.handle("browser_action", {"url": "not a url at all"}, {})
        assert result["confidence"] == 0.5
        assert browser.calls == []  # never reached the browser

    def test_www_prefixed_url_is_accepted(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        result = engine.handle("browser_action", {"url": "www.example.com"}, {})
        assert result["confidence"] == 0.9
        assert browser.calls == [("open_url", "www.example.com")]

    def test_open_url_exception_returns_graceful_error(self):
        browser = FakeBrowserAgent()
        browser.raise_on.add("open_url")
        engine = make_engine(browser)
        result = engine.handle("browser_action", {"url": "https://example.com"}, {})
        assert result["confidence"] == 0.0
        assert "example.com" in result["response"]

    def test_empty_open_url_result_still_reports_something(self):
        browser = FakeBrowserAgent()
        browser.open_url_result = ""
        engine = make_engine(browser)
        result = engine.handle("browser_action", {"url": "https://example.com"}, {})
        assert result["confidence"] == 0.0
        assert "no response" in result["response"].lower()

    def test_result_is_stripped(self):
        browser = FakeBrowserAgent()
        browser.open_url_result = "  Navigated OK  "
        engine = make_engine(browser)
        result = engine.handle("browser_action", {"url": "https://example.com"}, {})
        assert result["response"] == "Navigated OK"


# ===========================================================================
# scrape_page
# ===========================================================================

class TestScrapePage:
    def test_scrape_by_url_happy_path(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        result = engine.handle("scrape_page", {"url": "https://example.com"}, {})
        assert result["confidence"] == 0.85
        assert result["response"] == browser.scrape_result
        assert result["data"]["url"] == "https://example.com"

    def test_scrape_by_url_empty_text_returns_error(self):
        browser = FakeBrowserAgent()
        browser.scrape_result = ""
        engine = make_engine(browser)
        result = engine.handle("scrape_page", {"url": "https://example.com"}, {})
        assert result["confidence"] == 0.0
        assert "couldn't extract" in result["response"].lower()

    def test_scrape_by_url_exception_returns_graceful_error(self):
        browser = FakeBrowserAgent()
        browser.raise_on.add("get_page_text")
        engine = make_engine(browser)
        result = engine.handle("scrape_page", {"url": "https://example.com"}, {})
        assert result["confidence"] == 0.0

    def test_scrape_by_query_uses_search_and_summarize(self):
        browser = FakeBrowserAgentWithSearchResults()
        engine = make_engine(browser)
        result = engine.handle("scrape_page", {"query": "Alan Turing"}, {})
        assert result["confidence"] == 0.8
        assert result["data"]["query"] == "Alan Turing"
        assert len(result["data"]["results"]) == 3

    def test_scrape_by_query_no_results_asks_for_clarification(self):
        browser = FakeBrowserAgentWithSearchResults()
        browser.search_results = []
        engine = make_engine(browser)
        result = engine.handle("scrape_page", {"query": "nonexistent topic"}, {})
        assert result["confidence"] == 0.5

    def test_scrape_by_query_search_and_summarize_exception_returns_graceful_error(self):
        browser = FakeBrowserAgentWithSearchResults()
        browser.raise_on.add("search_web_results")
        engine = make_engine(browser)
        result = engine.handle("scrape_page", {"query": "x"}, {})
        assert result["confidence"] == 0.0
        assert "couldn't complete" in result["response"].lower()

    def test_no_url_no_query_asks_for_clarification(self):
        engine = make_engine(FakeBrowserAgent())
        result = engine.handle("scrape_page", {}, {})
        assert result["confidence"] == 0.5

    def test_url_takes_priority_over_query(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("scrape_page", {"url": "https://example.com", "query": "ignored"}, {})
        assert browser.calls == [("get_page_text", "https://example.com")]

    def test_top_result_falls_back_to_title_when_summary_empty(self):
        browser = FakeBrowserAgentWithSearchResults()
        browser.search_results = [{"title": "Only Title", "url": ""}]
        engine = make_engine(browser)
        result = engine.handle("scrape_page", {"query": "x"}, {})
        assert result["response"] == "Only Title"


# ===========================================================================
# open_app
# ===========================================================================

class TestOpenApp:
    def test_happy_path_known_app(self):
        engine = make_engine(None)
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            result = engine.handle("open_app", {"app": "chrome"}, {})
        assert result["confidence"] == 0.9
        assert "Opening chrome" in result["response"]
        mock_launch.assert_called_once_with("chrome.exe")

    def test_app_name_case_and_whitespace_insensitive(self):
        engine = make_engine(None)
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            engine.handle("open_app", {"app": "  Spotify  "}, {})
        mock_launch.assert_called_once_with("spotify.exe")

    def test_custom_app_map_overrides_default(self):
        engine = make_engine(None, app_map={"Spotify": "D:/Spotify/Spotify.exe"})
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            engine.handle("open_app", {"app": "spotify"}, {})
        mock_launch.assert_called_once_with("D:/Spotify/Spotify.exe")

    def test_custom_app_map_adds_new_app(self):
        engine = make_engine(None, app_map={"vscode": "code.exe"})
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            result = engine.handle("open_app", {"app": "vscode"}, {})
        assert result["confidence"] == 0.9
        mock_launch.assert_called_once_with("code.exe")

    def test_unknown_app_lists_available_apps(self):
        engine = make_engine(None)
        result = engine.handle("open_app", {"app": "some_random_app_xyz"}, {})
        assert result["confidence"] == 0.0
        assert "Unknown application" in result["response"]
        assert "chrome" in result["response"]

    def test_missing_app_name_asks_for_clarification(self):
        engine = make_engine(None)
        result = engine.handle("open_app", {}, {})
        assert result["confidence"] == 0.5

    def test_missing_app_falls_back_to_raw_query_stripping_open(self):
        engine = make_engine(None)
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            engine.handle("open_app", {"raw_query": "open chrome"}, {})
        mock_launch.assert_called_once_with("chrome.exe")

    def test_missing_app_falls_back_to_raw_query_stripping_launch(self):
        engine = make_engine(None)
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            engine.handle("open_app", {"raw_query": "launch notepad"}, {})
        mock_launch.assert_called_once_with("notepad.exe")

    def test_missing_app_falls_back_to_raw_query_stripping_start(self):
        engine = make_engine(None)
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            engine.handle("open_app", {"raw_query": "start calculator"}, {})
        mock_launch.assert_called_once_with("calc.exe")

    def test_raw_query_with_no_verb_prefix_still_resolves(self):
        engine = make_engine(None)
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            engine.handle("open_app", {"raw_query": "firefox"}, {})
        mock_launch.assert_called_once_with("firefox.exe")

    def test_file_not_found_reports_friendly_error(self):
        engine = make_engine(None)
        with patch("modules.hephaestus.engine._launch_app", side_effect=FileNotFoundError):
            result = engine.handle("open_app", {"app": "chrome"}, {})
        assert result["confidence"] == 0.0
        assert "couldn't find" in result["response"].lower()
        assert "chrome" in result["response"]

    def test_generic_launch_exception_reports_friendly_error(self):
        engine = make_engine(None)
        with patch("modules.hephaestus.engine._launch_app", side_effect=RuntimeError("boom")):
            result = engine.handle("open_app", {"app": "chrome"}, {})
        assert result["confidence"] == 0.0
        assert "couldn't open chrome" in result["response"].lower()

    def test_url_looking_app_name_redirects_to_browser_when_available(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        result = engine.handle("open_app", {"app": "google.com"}, {})
        assert result["confidence"] == 0.9
        assert browser.calls == [("open_url", "https://google.com")]

    def test_url_looking_app_name_adds_https_scheme(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("open_app", {"app": "www.example.com"}, {})
        assert browser.calls[0][1] in ("https://www.example.com", "www.example.com")

    def test_url_looking_app_name_preserves_existing_scheme(self):
        browser = FakeBrowserAgent()
        engine = make_engine(browser)
        engine.handle("open_app", {"app": "https://example.com"}, {})
        assert browser.calls == [("open_url", "https://example.com")]

    def test_url_looking_app_name_without_browser_reports_not_available(self):
        engine = make_engine(None)
        result = engine.handle("open_app", {"app": "google.com"}, {})
        assert result["response"] == _NOT_AVAILABLE

    def test_known_app_name_that_also_contains_a_dot_is_not_redirected(self):
        """A literal known app-map key always wins over the URL heuristic,
        even though app names rarely contain dots in practice — this
        guards the precedence order in _open_app."""
        engine = make_engine(None, app_map={"notepad.exe": "notepad.exe"})
        with patch("modules.hephaestus.engine._launch_app") as mock_launch:
            engine.handle("open_app", {"app": "notepad.exe"}, {})
        mock_launch.assert_called_once_with("notepad.exe")


# ===========================================================================
# scrape_url() — reusable helper for other modules
# ===========================================================================

class TestScrapeUrlHelper:
    def test_returns_stripped_text(self):
        browser = FakeBrowserAgent()
        browser.scrape_result = "  padded text  "
        engine = make_engine(browser)
        assert engine.scrape_url("https://example.com") == "padded text"

    def test_raises_when_browser_not_ready(self):
        engine = make_engine(None)
        with pytest.raises(HephaestusError):
            engine.scrape_url("https://example.com")

    def test_wraps_browser_exception_as_browser_agent_error(self):
        browser = FakeBrowserAgent()
        browser.raise_on.add("get_page_text")
        engine = make_engine(browser)
        with pytest.raises(BrowserAgentError):
            engine.scrape_url("https://example.com")

    def test_empty_page_text_returns_empty_string_not_none(self):
        browser = FakeBrowserAgent()
        browser.scrape_result = None
        engine = make_engine(browser)
        assert engine.scrape_url("https://example.com") == ""


# ===========================================================================
# search_and_summarize() — reusable helper for other modules
# ===========================================================================

class TestSearchAndSummarizeHelper:
    def test_raises_when_browser_not_ready(self):
        engine = make_engine(None)
        with pytest.raises(HephaestusError):
            engine.search_and_summarize("query")

    def test_uses_search_web_results_when_available(self):
        browser = FakeBrowserAgentWithSearchResults()
        engine = make_engine(browser)
        results = engine.search_and_summarize("query", max_results=2)
        assert len(results) == 2
        assert results[0]["title"] == "Title 1"
        assert results[0]["url"] == "https://a.example.com"
        assert results[0]["summary"] == browser.scrape_result

    def test_falls_back_to_split_titles_when_search_web_results_absent(self):
        browser = FakeBrowserAgent()  # no search_web_results method
        browser.search_web_result = "Foo | Bar | Baz"
        engine = make_engine(browser)
        results = engine.search_and_summarize("query")
        titles = [r["title"] for r in results]
        assert titles == ["Foo", "Bar", "Baz"]
        assert all(r["url"] == "" for r in results)
        assert all(r["summary"] == "" for r in results)  # no url -> no scrape attempted

    def test_max_results_truncates(self):
        browser = FakeBrowserAgentWithSearchResults()
        results = make_engine(browser).search_and_summarize("query", max_results=1)
        assert len(results) == 1

    def test_search_failure_raises_browser_agent_error(self):
        browser = FakeBrowserAgentWithSearchResults()
        browser.raise_on.add("search_web_results")
        engine = make_engine(browser)
        with pytest.raises(BrowserAgentError):
            engine.search_and_summarize("query")

    def test_per_result_scrape_failure_is_skipped_not_raised(self):
        browser = FakeBrowserAgentWithSearchResults()
        browser.raise_on.add("get_page_text")
        engine = make_engine(browser)
        results = engine.search_and_summarize("query")
        # search itself succeeded; individual scrape failures degrade to ""
        assert len(results) == 3
        assert all(r["summary"] == "" for r in results)

    def test_result_missing_title_and_url_handled_gracefully(self):
        browser = FakeBrowserAgentWithSearchResults()
        browser.search_results = [{}]
        engine = make_engine(browser)
        results = engine.search_and_summarize("query")
        assert results == [{"title": "", "url": "", "summary": ""}]


# ===========================================================================
# Module-level pure helpers
# ===========================================================================

class TestLooksLikeUrl:
    @pytest.mark.parametrize("value", [
        "https://example.com",
        "http://example.com",
        "www.example.com",
        "example.com",
        "sub.example.co.uk",
    ])
    def test_positive_cases(self, value):
        assert _looks_like_url(value) is True

    @pytest.mark.parametrize("value", [
        "chrome",
        "notepad",
        "",
        "just some words",
        "no-dot-here",
    ])
    def test_negative_cases(self, value):
        assert _looks_like_url(value) is False

    def test_trailing_dot_is_not_a_url(self):
        assert _looks_like_url("example.") is False

    def test_leading_dot_is_not_a_url(self):
        assert _looks_like_url(".com") is False

    def test_is_case_insensitive_for_scheme(self):
        assert _looks_like_url("HTTPS://Example.com") is True


class TestExtractHelper:
    def test_returns_first_nonempty_key(self):
        assert _extract({"a": "", "b": "value", "c": "other"}, "a", "b", "c") == "value"

    def test_returns_empty_string_when_all_missing(self):
        assert _extract({}, "a", "b") == ""

    def test_strips_whitespace(self):
        assert _extract({"a": "  hi  "}, "a") == "hi"

    def test_whitespace_only_value_is_skipped(self):
        assert _extract({"a": "   ", "b": "real"}, "a", "b") == "real"

    def test_coerces_non_string_values(self):
        assert _extract({"a": 123}, "a") == "123"


class TestSplitTitles:
    def test_splits_pipe_joined_titles(self):
        assert _split_titles("A | B | C") == ["A", "B", "C"]

    def test_returns_empty_list_for_no_results_message(self):
        assert _split_titles("No results found for query") == []

    def test_returns_empty_list_for_search_failed_message(self):
        assert _split_titles("Search failed: timeout") == []

    def test_returns_empty_list_for_empty_string(self):
        assert _split_titles("") == []

    def test_strips_each_title(self):
        assert _split_titles(" A  |  B ") == ["A", "B"]

    def test_drops_empty_segments(self):
        # Splitting is on the literal " | " separator, so a doubled
        # separator ("A |  | B") leaves an empty middle segment that gets
        # dropped, while "A | | B" (single space between pipes) doesn't
        # contain two full " | " separators and splits differently.
        assert _split_titles("A |  | B") == ["A", "B"]

    def test_single_title_no_separator(self):
        assert _split_titles("Only One") == ["Only One"]


class TestLaunchApp:
    def test_windows_uses_startfile(self):
        with patch("modules.hephaestus.engine.platform.system", return_value="Windows"), \
             patch("modules.hephaestus.engine.os.startfile", create=True) as mock_start:
            _launch_app("chrome.exe")
        mock_start.assert_called_once_with("chrome.exe")

    def test_macos_uses_open_subprocess(self):
        with patch("modules.hephaestus.engine.platform.system", return_value="Darwin"), \
             patch("modules.hephaestus.engine.subprocess.Popen") as mock_popen:
            _launch_app("Spotify")
        mock_popen.assert_called_once_with(["open", "Spotify"])

    def test_linux_uses_xdg_open(self):
        with patch("modules.hephaestus.engine.platform.system", return_value="Linux"), \
             patch("modules.hephaestus.engine.subprocess.Popen") as mock_popen:
            _launch_app("firefox")
        mock_popen.assert_called_once_with(["xdg-open", "firefox"])


class TestResponseShapeHelpers:
    def test_ok_default_confidence(self):
        result = _ok("hi")
        assert result == {"response": "hi", "data": {}, "confidence": 0.9}

    def test_ok_custom_confidence_and_data(self):
        result = _ok("hi", data={"x": 1}, confidence=0.5)
        assert result == {"response": "hi", "data": {"x": 1}, "confidence": 0.5}

    def test_err_shape(self):
        result = _err("bad")
        assert result == {"response": "bad", "data": {}, "confidence": 0.0}

    def test_clarify_shape(self):
        result = _clarify("which one?")
        assert result == {
            "response": "which one?",
            "data": {"needs_clarification": True},
            "confidence": 0.5,
        }


# ===========================================================================
# App map construction
# ===========================================================================

class TestAppMapConstruction:
    def test_default_app_map_used_when_no_override(self):
        engine = make_engine(None)
        assert engine._app_map == {k.lower(): v for k, v in _DEFAULT_APP_MAP.items()}

    def test_override_keys_are_lowercased(self):
        engine = make_engine(None, app_map={"MyApp": "myapp.exe"})
        assert engine._app_map["myapp"] == "myapp.exe"
        assert "MyApp" not in engine._app_map

    def test_override_merges_rather_than_replaces(self):
        engine = make_engine(None, app_map={"vscode": "code.exe"})
        assert "chrome" in engine._app_map
        assert "vscode" in engine._app_map

    def test_empty_app_map_override_keeps_defaults(self):
        engine = make_engine(None, app_map={})
        assert engine._app_map == {k.lower(): v for k, v in _DEFAULT_APP_MAP.items()}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
