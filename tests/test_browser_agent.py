"""
tests/test_browser_agent.py

Covers core/browser_agent.py's HestiaBrowserAgent.

`from playwright.sync_api import sync_playwright` happens lazily inside
_get_browser(), so each test that needs it installs a fake
`playwright.sync_api` module into sys.modules via monkeypatch — no real
browser is ever launched.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

import core.browser_agent as browser_module


def _install_fake_sync_playwright(monkeypatch, browser):
    """Install a fake playwright.sync_api module whose sync_playwright()
    context-manager-like object's .chromium.launch() returns `browser`."""
    pw_instance = MagicMock()
    pw_instance.chromium.launch.return_value = browser
    sync_playwright_factory = MagicMock()
    sync_playwright_factory.return_value.__enter__ = MagicMock(return_value=pw_instance)

    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = sync_playwright_factory
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)
    return pw_instance


def _agent(confirm_fn=None, headless=True):
    return browser_module.HestiaBrowserAgent(confirm_fn=confirm_fn, headless=headless)


# ---------------------------------------------------------------------------
# _get_browser
# ---------------------------------------------------------------------------

class TestGetBrowser:
    def test_lazily_launches_chromium(self, monkeypatch):
        browser = MagicMock()
        browser.is_connected.return_value = True
        pw_instance = _install_fake_sync_playwright(monkeypatch, browser)

        agent = _agent()
        result = agent._get_browser()

        assert result is browser
        pw_instance.chromium.launch.assert_called_once_with(headless=True)

    def test_reuses_connected_browser_without_relaunching(self, monkeypatch):
        browser = MagicMock()
        browser.is_connected.return_value = True
        _install_fake_sync_playwright(monkeypatch, browser)

        agent = _agent()
        first = agent._get_browser()
        second = agent._get_browser()

        assert first is second is browser

    def test_relaunches_when_existing_browser_connection_check_raises(self, monkeypatch):
        dead_browser = MagicMock()
        dead_browser.is_connected.side_effect = Exception("connection lost")
        fresh_browser = MagicMock()
        fresh_browser.is_connected.return_value = True

        pw_instance = MagicMock()
        pw_instance.chromium.launch.side_effect = [dead_browser, fresh_browser]
        sync_playwright_factory = MagicMock()
        sync_playwright_factory.return_value.__enter__ = MagicMock(return_value=pw_instance)
        fake_module = types.ModuleType("playwright.sync_api")
        fake_module.sync_playwright = sync_playwright_factory
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)

        agent = _agent()
        first = agent._get_browser()   # launches dead_browser, caches it
        second = agent._get_browser()  # is_connected() raises -> relaunch

        assert first is dead_browser
        assert second is fresh_browser
        assert pw_instance.chromium.launch.call_count == 2

    def test_returns_none_when_playwright_not_installed(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _raise_for_playwright(name, *args, **kwargs):
            if name == "playwright.sync_api" or name.startswith("playwright"):
                raise ImportError("no module named playwright")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "playwright.sync_api", raising=False)
        monkeypatch.setattr(builtins, "__import__", _raise_for_playwright)

        agent = _agent()
        result = agent._get_browser()
        assert result is None

    def test_passes_headless_flag_through(self, monkeypatch):
        browser = MagicMock()
        browser.is_connected.return_value = True
        pw_instance = _install_fake_sync_playwright(monkeypatch, browser)

        agent = _agent(headless=False)
        agent._get_browser()

        pw_instance.chromium.launch.assert_called_once_with(headless=False)


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

class TestClose:
    def test_closes_browser_and_playwright(self, monkeypatch):
        browser = MagicMock()
        browser.is_connected.return_value = True
        _install_fake_sync_playwright(monkeypatch, browser)

        agent = _agent()
        agent._get_browser()
        agent.close()

        browser.close.assert_called_once()
        assert agent._browser is None
        assert agent._playwright is None

    def test_close_is_safe_when_never_launched(self):
        agent = _agent()
        agent.close()  # must not raise
        assert agent._browser is None

    def test_close_swallows_exceptions(self, monkeypatch):
        browser = MagicMock()
        browser.is_connected.return_value = True
        browser.close.side_effect = Exception("already dead")
        _install_fake_sync_playwright(monkeypatch, browser)

        agent = _agent()
        agent._get_browser()
        agent.close()  # must not raise despite browser.close() throwing
        assert agent._browser is None


# ---------------------------------------------------------------------------
# _confirm
# ---------------------------------------------------------------------------

class TestConfirm:
    def test_auto_confirms_when_no_confirm_fn(self):
        agent = _agent(confirm_fn=None)
        assert agent._confirm("proceed?") is True

    def test_delegates_to_confirm_fn(self):
        confirm_fn = MagicMock(return_value=False)
        agent = _agent(confirm_fn=confirm_fn)
        result = agent._confirm("should I do this?")
        assert result is False
        confirm_fn.assert_called_once_with("should I do this?")


# ---------------------------------------------------------------------------
# _new_page
# ---------------------------------------------------------------------------

class TestNewPage:
    def test_returns_none_when_browser_unavailable(self, monkeypatch):
        agent = _agent()
        monkeypatch.setattr(agent, "_get_browser", lambda: None)
        assert agent._new_page() is None

    def test_creates_context_with_custom_user_agent_and_returns_page(self, monkeypatch):
        browser = MagicMock()
        context = MagicMock()
        page = MagicMock()
        context.new_page.return_value = page
        browser.new_context.return_value = context

        agent = _agent()
        monkeypatch.setattr(agent, "_get_browser", lambda: browser)

        result = agent._new_page()

        assert result is page
        _, kwargs = browser.new_context.call_args
        assert "Chrome" in kwargs["user_agent"]

    def test_returns_none_on_exception(self, monkeypatch):
        browser = MagicMock()
        browser.new_context.side_effect = Exception("boom")
        agent = _agent()
        monkeypatch.setattr(agent, "_get_browser", lambda: browser)
        assert agent._new_page() is None


# ---------------------------------------------------------------------------
# search_web / search_web_results
# ---------------------------------------------------------------------------

class TestSearchWeb:
    def _link(self, title, href):
        link = MagicMock()
        link.inner_text.return_value = title
        link.get_attribute.return_value = href
        return link

    def test_search_web_results_returns_title_url_dicts(self, monkeypatch):
        page = MagicMock()
        page.query_selector_all.return_value = [
            self._link("Result A", "https://a.example"),
            self._link("Result B", "https://b.example"),
        ]
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: page)

        results = agent.search_web_results("cats", max_results=2)

        assert results == [
            {"title": "Result A", "url": "https://a.example"},
            {"title": "Result B", "url": "https://b.example"},
        ]
        page.close.assert_called_once()

    def test_search_web_results_returns_empty_list_when_no_browser(self, monkeypatch):
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: None)
        assert agent.search_web_results("cats") == []

    def test_search_web_results_closes_page_and_returns_empty_on_exception(self, monkeypatch):
        page = MagicMock()
        page.goto.side_effect = Exception("timeout")
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: page)

        results = agent.search_web_results("cats")

        assert results == []
        page.close.assert_called_once()

    def test_search_web_joins_titles_with_pipe(self, monkeypatch):
        agent = _agent()
        monkeypatch.setattr(
            agent,
            "search_web_results",
            lambda query, max_results=3: [{"title": "A", "url": "x"}, {"title": "B", "url": "y"}],
        )
        assert agent.search_web("cats") == "A | B"

    def test_search_web_reports_no_results(self, monkeypatch):
        agent = _agent()
        monkeypatch.setattr(agent, "search_web_results", lambda query, max_results=3: [])
        assert agent.search_web("cats") == "No results found for 'cats'."


# ---------------------------------------------------------------------------
# open_url
# ---------------------------------------------------------------------------

class TestOpenUrl:
    def test_declines_without_confirmation(self):
        confirm_fn = MagicMock(return_value=False)
        agent = _agent(confirm_fn=confirm_fn)
        result = agent.open_url("example.com")
        assert result == "Okay, I won't open that."
        confirm_fn.assert_called_once()

    def test_opens_and_returns_title_on_confirm(self, monkeypatch):
        page = MagicMock()
        page.title.return_value = "Example Domain"
        agent = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(agent, "_new_page", lambda: page)

        result = agent.open_url("example.com")

        assert result == "Opened Example Domain."
        page.goto.assert_called_once()
        called_url = page.goto.call_args[0][0]
        assert called_url == "https://example.com"

    def test_skip_confirmation_when_confirm_false(self, monkeypatch):
        page = MagicMock()
        page.title.return_value = "No Confirm Needed"
        confirm_fn = MagicMock()
        agent = _agent(confirm_fn=confirm_fn)
        monkeypatch.setattr(agent, "_new_page", lambda: page)

        result = agent.open_url("example.com", confirm=False)

        assert result == "Opened No Confirm Needed."
        confirm_fn.assert_not_called()

    def test_browser_unavailable_message(self, monkeypatch):
        agent = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(agent, "_new_page", lambda: None)
        assert agent.open_url("example.com") == "Browser is not available."

    def test_exception_returns_friendly_message(self, monkeypatch):
        page = MagicMock()
        page.goto.side_effect = Exception("dns fail")
        agent = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(agent, "_new_page", lambda: page)
        assert agent.open_url("example.com") == "I couldn't open that page."


# ---------------------------------------------------------------------------
# fill_form
# ---------------------------------------------------------------------------

class TestFillForm:
    def test_declines_without_confirmation(self):
        confirm_fn = MagicMock(return_value=False)
        agent = _agent(confirm_fn=confirm_fn)
        result = agent.fill_form("example.com/form", {"#name": "Bob"})
        assert result == "Okay, cancelled."

    def test_browser_unavailable(self, monkeypatch):
        agent = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(agent, "_new_page", lambda: None)
        assert agent.fill_form("example.com", {"#name": "Bob"}) == "Browser is not available."

    def test_fills_fields_without_submit_selector(self, monkeypatch):
        page = MagicMock()
        agent = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(agent, "_new_page", lambda: page)
        monkeypatch.setattr(browser_module.time, "sleep", lambda s: None)

        result = agent.fill_form("example.com", {"#name": "Bob", "#email": "b@x.com"})

        assert "not submitted" in result
        assert page.fill.call_count == 2

    def test_continues_filling_after_a_field_error(self, monkeypatch):
        page = MagicMock()
        page.fill.side_effect = [Exception("bad selector"), None]
        agent = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(agent, "_new_page", lambda: page)
        monkeypatch.setattr(browser_module.time, "sleep", lambda s: None)

        result = agent.fill_form("example.com", {"#bad": "x", "#good": "y"})

        assert page.fill.call_count == 2
        assert "not submitted" in result

    def test_submit_confirmation_declined_leaves_form_filled(self, monkeypatch):
        page = MagicMock()
        confirm_calls = []

        def confirm_fn(question):
            confirm_calls.append(question)
            return "submit" not in question.lower()

        agent = _agent(confirm_fn=confirm_fn)
        monkeypatch.setattr(agent, "_new_page", lambda: page)
        monkeypatch.setattr(browser_module.time, "sleep", lambda s: None)

        result = agent.fill_form("example.com", {"#name": "Bob"}, submit_selector="#submit")

        assert result == "Form filled but not submitted."
        page.click.assert_not_called()

    def test_submits_and_returns_new_title_on_confirm(self, monkeypatch):
        page = MagicMock()
        page.title.return_value = "Thank You"
        agent = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(agent, "_new_page", lambda: page)
        monkeypatch.setattr(browser_module.time, "sleep", lambda s: None)

        result = agent.fill_form("example.com", {"#name": "Bob"}, submit_selector="#submit")

        assert result == "Form submitted. Page title is now: Thank You."
        page.click.assert_called_once_with("#submit")

    def test_exception_returns_friendly_message(self, monkeypatch):
        page = MagicMock()
        page.goto.side_effect = Exception("boom")
        agent = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(agent, "_new_page", lambda: page)

        result = agent.fill_form("example.com", {"#name": "Bob"})
        assert result == "Something went wrong filling that form."


# ---------------------------------------------------------------------------
# get_page_text
# ---------------------------------------------------------------------------

class TestGetPageText:
    def test_truncates_to_500_chars(self, monkeypatch):
        page = MagicMock()
        page.inner_text.return_value = "word " * 300
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: page)

        result = agent.get_page_text("example.com")
        assert len(result) <= 500

    def test_browser_unavailable(self, monkeypatch):
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: None)
        assert agent.get_page_text("example.com") == "Browser is not available."

    def test_no_readable_content_message(self, monkeypatch):
        page = MagicMock()
        page.inner_text.return_value = "   "
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: page)
        assert agent.get_page_text("example.com") == "Page loaded but no readable content found."

    def test_exception_returns_friendly_message(self, monkeypatch):
        page = MagicMock()
        page.goto.side_effect = Exception("boom")
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: page)
        assert agent.get_page_text("example.com") == "I couldn't read that page."


# ---------------------------------------------------------------------------
# check_flight_status
# ---------------------------------------------------------------------------

class TestCheckFlightStatus:
    def test_returns_text_from_first_matching_selector(self, monkeypatch):
        page = MagicMock()
        found_el = MagicMock()
        found_el.inner_text.return_value = "On time, arriving 4:15 PM"
        page.query_selector.side_effect = [None, found_el]
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: page)

        result = agent.check_flight_status("BA123")
        assert "On time" in result

    def test_no_selector_matches(self, monkeypatch):
        page = MagicMock()
        page.query_selector.return_value = None
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: page)

        result = agent.check_flight_status("BA123")
        assert result == "I couldn't find status for flight BA123."

    def test_browser_unavailable(self, monkeypatch):
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: None)
        assert agent.check_flight_status("BA123") == "Browser is not available."

    def test_exception_returns_friendly_message(self, monkeypatch):
        page = MagicMock()
        page.goto.side_effect = Exception("boom")
        agent = _agent()
        monkeypatch.setattr(agent, "_new_page", lambda: page)
        assert agent.check_flight_status("BA123") == "I couldn't check that flight status."
