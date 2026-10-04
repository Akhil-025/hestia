# tests/test_hephaestus_backlog.py
"""
Tests for the Hephaestus work in backlog #101-#103 and #105-#110:

  #101 scheduled page checks   #102 form filling      #103 failure screenshots
  #105 site scrapers           #106 politeness delay  #107 browser session reuse
  #108 headed / slow-mo        #109 change detection  #110 repo summary

Everything runs against fakes (no Playwright, no network). Run with:
    pytest tests/test_hephaestus_backlog.py -v
"""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.browser_agent as browser_module
from core.browser_agent import HestiaBrowserAgent
from core.heartbeat import HestiaHeartbeat
from modules.hecate.intent_registry import INTENT_MODULE_MAP
from modules.hephaestus.engine import HephaestusEngine
from modules.hephaestus.monitor import (
    MonitorStore,
    analyse_change,
    extract_price,
    format_price,
    validate_monitor_url,
)
from modules.hephaestus.repo_summary import format_summary, summarize_repo
from modules.hephaestus.scrapers import ScraperRegistry, SelectorScraper, SiteScraper

T0 = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
NEW_INTENTS = (
    "watch_page", "list_watches", "stop_watching", "check_watches",
    "fill_form", "summarize_repo",
)


# ===========================================================================
# Fakes
# ===========================================================================

class Clock:
    """Controllable time for the engine's monitor scheduling."""

    def __init__(self, start=T0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


class MonBrowser:
    """Duck-typed browser agent with the monitor-facing methods."""

    def __init__(self, pages=None):
        self.pages = dict(pages or {})
        self.elements = {}
        self.threads = []
        self.closed = 0
        self.fetches = []
        self.fill_calls = []
        self.fill_result = "Form submitted. Page title is now: Thanks."

    def fetch_text(self, url, max_chars=20000, wait_ms=1500):
        self.threads.append(threading.get_ident())
        self.fetches.append(url)
        return self.pages.get(url)

    def fetch_elements(self, url, selector, max_items=50, wait_ms=1500):
        self.threads.append(threading.get_ident())
        self.fetches.append((url, selector))
        return self.elements.get((url, selector))

    def get_page_text(self, url):
        return self.pages.get(url) or "I couldn't read that page."

    def fill_form(self, url, fields, submit_selector=None):
        self.fill_calls.append((url, dict(fields), submit_selector))
        return self.fill_result

    def close(self):
        self.closed += 1


def make_engine(chat=None, monitor=None, store=True, clock=None, **kw):
    chat = chat if chat is not None else MonBrowser()
    return HephaestusEngine(
        chat,
        monitor_store=MonitorStore(":memory:") if store is True else (store or None),
        monitor_browser=monitor,
        now_fn=clock or Clock(),
        local_now_fn=kw.pop("local_now_fn", lambda: datetime(2026, 10, 3, 12, 0)),
        **kw,
    )


# ===========================================================================
# Registry / contract
# ===========================================================================

class TestContract:
    def test_new_intents_are_handled_and_registered(self):
        engine = HephaestusEngine(MonBrowser())
        for i in NEW_INTENTS:
            assert engine.can_handle(i)
            assert INTENT_MODULE_MAP[f"hephaestus_{i}"] == "hephaestus"

    def test_old_construction_still_works_and_watching_reports_unset(self):
        engine = HephaestusEngine(MonBrowser())
        r = engine.handle("watch_page", {"url": "https://example.com"}, {})
        assert "isn't set up" in r["response"]
        assert r["confidence"] == 0.0

    def test_get_context_unchanged_without_store_and_counts_with_one(self):
        assert HephaestusEngine(MonBrowser()).get_context() == {"hephaestus_available": True}
        engine = make_engine()
        assert engine.get_context()["hephaestus_watches"] == 0

    def test_browserless_intents_work_without_a_browser(self, tmp_path):
        engine = HephaestusEngine(None, monitor_store=MonitorStore(":memory:"))
        assert "not watching" in engine.handle("list_watches", {}, {})["response"]
        (tmp_path / "a.py").write_text("x = 1\n")
        assert engine.handle("summarize_repo", {"path": str(tmp_path)}, {})["confidence"] > 0
        # ...but anything that needs the page does not.
        assert engine.handle("watch_page", {"url": "https://example.com"}, {})["confidence"] == 0.0
        assert engine.handle("check_watches", {}, {})["confidence"] == 0.0


# ===========================================================================
# Browser agent: #103 screenshots, #107 session reuse, #108 headed
# ===========================================================================

def _agent(**kw):
    return HestiaBrowserAgent(**kw)


class TestFailureScreenshots:
    def test_saves_screenshot_on_failure_when_dir_set(self, tmp_path):
        page = MagicMock()
        page.screenshot.side_effect = lambda path, full_page: open(path, "wb").write(b"png")
        a = _agent(screenshot_dir=str(tmp_path))
        out = a._snap_failure(page, "open-https://x.com/a b")
        assert out and os.path.exists(out)
        assert a.last_failure_screenshot == out
        assert out.endswith(".png") and "/" not in os.path.basename(out)

    def test_off_by_default(self):
        page = MagicMock()
        assert _agent()._snap_failure(page, "x") is None
        page.screenshot.assert_not_called()

    def test_never_raises_and_keeps_only_the_newest(self, tmp_path):
        page = MagicMock()
        page.screenshot.side_effect = lambda path, full_page: open(path, "wb").write(b"p")
        a = _agent(screenshot_dir=str(tmp_path))
        for i in range(25):
            a._snap_failure(page, f"n{i}")
        assert len(list(tmp_path.glob("*.png"))) == 20
        page.screenshot.side_effect = RuntimeError("closed")
        assert a._snap_failure(page, "boom") is None

    def test_failed_open_url_takes_a_screenshot(self, tmp_path, monkeypatch):
        page = MagicMock()
        page.goto.side_effect = Exception("dns")
        a = _agent(screenshot_dir=str(tmp_path), confirm_fn=None)
        monkeypatch.setattr(a, "_new_page", lambda: page)
        assert a.open_url("example.com") == "I couldn't open that page."
        page.screenshot.assert_called_once()


class TestSessionReuse:
    def _wired(self, monkeypatch, clock=None, **kw):
        browser = MagicMock()
        browser.is_connected.return_value = True
        pw = MagicMock()
        pw.chromium.launch.return_value = browser
        fake = MagicMock()
        fake.sync_playwright.return_value.__enter__.return_value = pw
        monkeypatch.setitem(sys.modules, "playwright", MagicMock())
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake)
        a = _agent(clock=clock or (lambda: 0.0), **kw)
        return a, browser, pw

    def test_pages_share_one_context(self, monkeypatch):
        a, browser, _ = self._wired(monkeypatch)
        a._new_page()
        a._new_page()
        assert browser.new_context.call_count == 1

    def test_context_rebuilt_after_it_fails(self, monkeypatch):
        a, browser, _ = self._wired(monkeypatch)
        ctx = browser.new_context.return_value
        ctx.new_page.side_effect = [Exception("closed"), MagicMock()]
        assert a._new_page() is None
        assert a._new_page() is not None
        assert browser.new_context.call_count == 2

    def test_idle_browser_is_closed_and_relaunched(self, monkeypatch):
        t = {"now": 0.0}
        a, browser, pw = self._wired(monkeypatch, clock=lambda: t["now"], idle_timeout_seconds=60)
        a._get_browser()
        t["now"] = 30
        a._get_browser()
        assert pw.chromium.launch.call_count == 1
        t["now"] = 200
        a._get_browser()
        assert pw.chromium.launch.call_count == 2
        browser.close.assert_called()

    def test_close_if_idle(self, monkeypatch):
        t = {"now": 0.0}
        a, browser, _ = self._wired(monkeypatch, clock=lambda: t["now"], idle_timeout_seconds=60)
        a._get_browser()
        assert a.close_if_idle() is False
        t["now"] = 100
        assert a.close_if_idle() is True
        assert a._browser is None and a.close_if_idle() is False

    def test_headed_and_slow_mo_reach_launch(self, monkeypatch):
        a, _, pw = self._wired(monkeypatch, headless=False, slow_mo_ms=250)
        a._get_browser()
        pw.chromium.launch.assert_called_once_with(headless=False, slow_mo=250)

    def test_defaults_do_not_add_slow_mo(self, monkeypatch):
        a, _, pw = self._wired(monkeypatch)
        a._get_browser()
        pw.chromium.launch.assert_called_once_with(headless=True)


class TestFetchHelpers:
    def test_fetch_text_returns_none_on_every_failure(self, monkeypatch):
        a = _agent()
        monkeypatch.setattr(a, "_new_page", lambda: None)
        assert a.fetch_text("example.com") is None
        page = MagicMock()
        page.goto.side_effect = Exception("boom")
        monkeypatch.setattr(a, "_new_page", lambda: page)
        assert a.fetch_text("example.com") is None
        page2 = MagicMock()
        page2.inner_text.return_value = "   "
        monkeypatch.setattr(a, "_new_page", lambda: page2)
        assert a.fetch_text("example.com") is None

    def test_fetch_text_returns_text_up_to_the_limit(self, monkeypatch):
        page = MagicMock()
        page.inner_text.return_value = "word " * 5000
        a = _agent()
        monkeypatch.setattr(a, "_new_page", lambda: page)
        text = a.fetch_text("example.com", max_chars=1000)
        assert text and 500 < len(text) <= 1000

    def test_fetch_elements(self, monkeypatch):
        els = [MagicMock(), MagicMock(), MagicMock()]
        for el, t in zip(els, ["Row  one", "Row two", "   "]):
            el.inner_text.return_value = t
        page = MagicMock()
        page.query_selector_all.return_value = els
        a = _agent()
        monkeypatch.setattr(a, "_new_page", lambda: page)
        assert a.fetch_elements("example.com", "tr") == ["Row one", "Row two"]

    def test_fetch_elements_none_on_failure_and_empty_list_on_no_match(self, monkeypatch):
        a = _agent()
        monkeypatch.setattr(a, "_new_page", lambda: None)
        assert a.fetch_elements("example.com", "tr") is None
        page = MagicMock()
        page.query_selector_all.return_value = []
        monkeypatch.setattr(a, "_new_page", lambda: page)
        assert a.fetch_elements("example.com", "tr") == []

    def test_get_page_text_messages_unchanged(self, monkeypatch):
        a = _agent()
        monkeypatch.setattr(a, "_new_page", lambda: None)
        assert a.get_page_text("x.com") == "Browser is not available."


class TestFillFormSafety:
    def test_does_not_submit_when_a_field_failed(self, monkeypatch):
        page = MagicMock()
        page.fill.side_effect = [Exception("no such field"), None]
        a = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(a, "_new_page", lambda: page)
        monkeypatch.setattr(browser_module.time, "sleep", lambda s: None)
        out = a.fill_form("example.com", {"#bad": "x", "#ok": "y"}, submit_selector="#go")
        assert "didn't submit" in out and "#bad" in out
        page.click.assert_not_called()

    def test_says_so_when_submit_click_did_not_settle(self, monkeypatch):
        page = MagicMock()
        page.wait_for_load_state.side_effect = Exception("timeout")
        a = _agent(confirm_fn=lambda q: True)
        monkeypatch.setattr(a, "_new_page", lambda: page)
        monkeypatch.setattr(browser_module.time, "sleep", lambda s: None)
        out = a.fill_form("example.com", {"#a": "b"}, submit_selector="#go")
        assert "can't confirm it went through" in out


# ===========================================================================
# monitor.py: URL guard, price, change analysis, store
# ===========================================================================

class TestUrlGuard:
    @pytest.mark.parametrize("url", [
        "http://localhost:5000", "127.0.0.1/admin", "http://192.168.1.1/", "10.0.0.5",
        "http://169.254.169.254/latest", "http://[::1]/", "printer.local", "http://nas.lan/x",
        "ftp://example.com/file", "file:///etc/passwd", "http://user:pw@example.com/",
        "intranet", "",
    ])
    def test_rejects(self, url):
        clean, reason = validate_monitor_url(url)
        assert clean is None and reason

    @pytest.mark.parametrize("url,expected", [
        ("example.com/page", "https://example.com/page"),
        ("http://shop.example.co.in/p?id=3", "http://shop.example.co.in/p?id=3"),
        ("  https://example.org  ", "https://example.org"),
    ])
    def test_accepts(self, url, expected):
        assert validate_monitor_url(url) == (expected, None)


class TestPrice:
    @pytest.mark.parametrize("text,value,symbol", [
        ("Now only ₹74,900 today", 74900.0, "₹"),
        ("Price: $1,299.99", 1299.99, "$"),
        ("Rs. 499", 499.0, "Rs."),
        ("€ 12.50 incl. VAT", 12.5, "€"),
        ("INR 1,00,000", 100000.0, "INR"),
    ])
    def test_extract(self, text, value, symbol):
        assert extract_price(text) == (value, symbol)

    def test_none_when_no_price(self):
        assert extract_price("Out of stock, 3 reviews") == (None, "")

    def test_format(self):
        assert format_price(74900.0, "₹") == "₹74,900"
        assert format_price(12.5, "$") == "$12.5"


class TestAnalyseChange:
    def test_first_check_is_a_silent_baseline(self):
        r = analyse_change(None, "Hello world page", keywords=["hello"])
        assert r["baseline"] and r["alerts"] == []

    def test_whitespace_only_change_is_ignored(self):
        assert analyse_change("a long page of text", "a  long   page of\ntext")["alerts"] == []

    def test_digit_only_changes_are_ignored(self):
        old = "Welcome back. 1,204 people viewing. Updated 12:01:55"
        new = "Welcome back. 1,311 people viewing. Updated 12:07:09"
        assert analyse_change(old, new)["alerts"] == []

    def test_reordering_is_not_a_change(self):
        assert analyse_change("alpha beta gamma delta", "delta gamma beta alpha")["alerts"] == []

    def test_small_text_change_below_threshold_is_ignored(self):
        assert analyse_change("sale ends soon", "sale ends today")["alerts"] == []

    def test_meaningful_change_alerts_with_an_excerpt(self):
        old = "Admissions open for the new batch apply online"
        new = "Admissions open for the new batch apply online. Results declared for entrance exam"
        r = analyse_change(old, new)
        assert len(r["alerts"]) == 1
        assert "words added" in r["alerts"][0] and "Results declared" in r["alerts"][0]

    def test_threshold_is_adjustable(self):
        assert analyse_change("sale ends soon", "sale ends today", min_changed_words=1)["alerts"]

    def test_keyword_alerts_only_when_it_newly_appears(self):
        old = "Results coming soon"
        new = "Results coming soon. Admit Card released"
        assert analyse_change(old, new, keywords=["admit card"])["alerts"] == ["“admit card” now appears"]
        # Already present before -> no repeat alert.
        assert analyse_change(new, new + " more words here", keywords=["admit card"])["alerts"] == []

    def test_keyword_watch_ignores_other_changes(self):
        assert analyse_change("one two three", "totally different words appear now",
                              keywords=["admit card"])["alerts"] == []

    def test_price_drop_alerts_and_rise_is_silent(self):
        drop = analyse_change("Phone ₹80,000", "Phone ₹74,900", track_price=True, old_price=80000.0)
        assert "dropped from ₹80,000 to ₹74,900" in drop["alerts"][0]
        rise = analyse_change("Phone ₹80,000", "Phone ₹85,000", track_price=True, old_price=80000.0)
        assert rise["alerts"] == [] and rise["price"] == 85000.0

    def test_price_watch_ignores_text_changes(self):
        r = analyse_change("Phone ₹80,000 reviews", "Phone ₹80,000 brand new banner text here",
                           track_price=True, old_price=80000.0)
        assert r["alerts"] == []

    def test_target_price_on_baseline_and_on_crossing_only_once(self):
        first = analyse_change(None, "₹49,000", track_price=True, target_price=50000.0)
        assert "at or below your target" in first["alerts"][0]
        cross = analyse_change("₹55,000", "₹49,000", track_price=True, old_price=55000.0, target_price=50000.0)
        assert len(cross["alerts"]) == 1 and "target" in cross["alerts"][0]
        again = analyse_change("₹49,000", "₹49,000", track_price=True, old_price=49000.0, target_price=50000.0)
        assert again["alerts"] == []

    def test_missing_price_does_not_alert(self):
        r = analyse_change("₹80,000", "sold out", track_price=True, old_price=80000.0)
        assert r["alerts"] == [] and r["price"] is None


class TestStore:
    def test_add_list_find_remove(self):
        s = MonitorStore(":memory:")
        mid = s.add(name="Phone", url="https://a.com/p", now=T0, keywords=["x"], interval_minutes=60)
        assert [m["name"] for m in s.list()] == ["Phone"]
        assert s.list()[0]["keywords"] == ["x"]
        assert s.find("pho")[0]["id"] == mid and s.find("a.com")[0]["id"] == mid
        assert s.remove(mid) and s.list() == []

    def test_duplicate_name_and_limit(self):
        s = MonitorStore(":memory:", max_monitors=2)
        s.add(name="A", url="https://a.com", now=T0)
        with pytest.raises(ValueError, match="duplicate"):
            s.add(name="a", url="https://b.com", now=T0)
        s.add(name="B", url="https://b.com", now=T0)
        with pytest.raises(ValueError, match="limit"):
            s.add(name="C", url="https://c.com", now=T0)

    def test_interval_is_clamped(self):
        s = MonitorStore(":memory:")
        s.add(name="A", url="https://a.com", now=T0, interval_minutes=1)
        assert s.list()[0]["interval_minutes"] == 30

    def test_due_logic_with_slack(self):
        s = MonitorStore(":memory:")
        mid = s.add(name="A", url="https://a.com", now=T0, interval_minutes=60)
        assert len(s.due(T0)) == 1                       # never checked
        s.record_success(mid, snapshot="x", now=T0, changed=False, price=None, currency="")
        assert s.due(T0 + timedelta(minutes=30)) == []
        assert len(s.due(T0 + timedelta(minutes=56))) == 1   # within the 5-minute slack

    def test_failure_streak_alerts_once_and_resets_on_success(self):
        s = MonitorStore(":memory:")
        mid = s.add(name="A", url="https://a.com", now=T0)
        assert s.record_failure(mid, T0) == (1, False)
        assert s.record_failure(mid, T0) == (2, False)
        assert s.record_failure(mid, T0) == (3, True)
        assert s.record_failure(mid, T0) == (4, False)
        s.record_success(mid, snapshot="ok", now=T0, changed=False, price=None, currency="")
        assert s.list()[0]["failures"] == 0
        assert [s.record_failure(mid, T0)[1] for _ in range(3)] == [False, False, True]

    def test_alert_queue_survives_and_marks_delivered(self, tmp_path):
        path = str(tmp_path / "h.db")
        s = MonitorStore(path)
        mid = s.add(name="A", url="https://a.com", now=T0)
        s.add_alert(mid, "A: changed", T0)
        s2 = MonitorStore(path)                          # "restart"
        pending = s2.pending_alerts()
        assert [a["text"] for a in pending] == ["A: changed"]
        s2.mark_delivered([a["id"] for a in pending])
        assert s2.pending_alerts() == []

    def test_success_keeps_last_price_when_page_has_none(self):
        s = MonitorStore(":memory:")
        mid = s.add(name="A", url="https://a.com", now=T0, track_price=True)
        s.record_success(mid, snapshot="₹5", now=T0, changed=False, price=5.0, currency="₹")
        s.record_success(mid, snapshot="sold out", now=T0, changed=True, price=None, currency="")
        m = s.list()[0]
        assert m["last_price"] == 5.0 and m["currency"] == "₹"


# ===========================================================================
# Scrapers (#105)
# ===========================================================================

class TestScrapers:
    def test_selector_scraper_matching(self):
        s = SelectorScraper("r", "tr", url_contains="Example.org/Results")
        assert s.matches("https://example.org/results/2026") and not s.matches("https://other.org")
        rx = SelectorScraper("r2", "tr", url_regex=r"gate\d+\.example")
        assert rx.matches("https://gate26.example.org") and not rx.matches("https://gate.example.org")

    def test_selector_scraper_needs_name_selector_and_matcher(self):
        for args, kw in [(("", "tr"), {"url_contains": "a"}), (("n", ""), {"url_contains": "a"}), (("n", "tr"), {})]:
            with pytest.raises(ValueError):
                SelectorScraper(*args, **kw)

    def test_scrape_joins_items_and_returns_none_on_miss(self):
        b = MonBrowser()
        s = SelectorScraper("r", "tr", url_contains="a.com")
        b.elements[("https://a.com", "tr")] = ["Row 1", "Row 2"]
        assert s.scrape("https://a.com", b) == "Row 1\nRow 2"
        b.elements[("https://a.com", "tr")] = []
        assert s.scrape("https://a.com", b) is None       # layout changed
        assert s.scrape("https://b.com", b) is None       # load failure
        assert s.scrape("https://a.com", object()) is None

    def test_registry_first_match_and_duplicate_names(self):
        reg = ScraperRegistry()
        reg.register(SelectorScraper("one", "p", url_contains="a.com"))
        reg.register(SelectorScraper("two", "p", url_contains="a.com"))
        assert reg.find("https://a.com").name == "one"
        with pytest.raises(ValueError):
            reg.register(SelectorScraper("one", "p", url_contains="z"))

    def test_config_loading_skips_bad_entries(self):
        reg = ScraperRegistry()
        reg.load_config([
            {"name": "ok", "selector": "td", "url_contains": "x.com"},
            {"name": "no-selector", "url_contains": "y.com"},
            "not a dict",
            {"name": "bad-regex", "selector": "td", "url_regex": "("},
        ])
        assert reg.names == ["ok"]

    def test_a_scraper_that_raises_in_matches_is_skipped(self):
        class Bad(SiteScraper):
            name = "bad"
            def matches(self, url): raise RuntimeError("x")
        reg = ScraperRegistry()
        reg.register(Bad())
        assert reg.find("https://a.com") is None

    def test_plugin_loading(self, tmp_path):
        (tmp_path / "jobs.py").write_text(
            "from modules.hephaestus.scrapers import SiteScraper\n"
            "class Jobs(SiteScraper):\n"
            "    name='jobs'\n"
            "    def matches(self, url): return 'jobs.example' in url\n"
            "    def scrape(self, url, browser): return 'JOBS'\n"
            "SCRAPERS=[Jobs()]\n")
        (tmp_path / "broken.py").write_text("raise RuntimeError('nope')\n")
        (tmp_path / "_private.py").write_text("SCRAPERS = [1/0]\n")
        reg = ScraperRegistry()
        assert reg.load_plugins(str(tmp_path)) == 1
        assert reg.find("https://jobs.example.com").scrape("u", None) == "JOBS"
        assert ScraperRegistry().load_plugins(str(tmp_path / "missing")) == 0
        assert ScraperRegistry().load_plugins(None) == 0

    def test_engine_scrape_uses_scraper_then_falls_back(self):
        b = MonBrowser({"https://a.com/r": "generic text"})
        reg = ScraperRegistry()
        reg.register(SelectorScraper("r", "tr", url_contains="a.com"))
        engine = HephaestusEngine(b, scrapers=reg)
        b.elements[("https://a.com/r", "tr")] = ["Row 1", "Row 2"]
        assert engine.scrape_url("https://a.com/r") == "Row 1\nRow 2"
        b.elements.clear()
        assert engine.scrape_url("https://a.com/r") == "generic text"

    def test_engine_caps_long_scraper_output(self):
        b = MonBrowser()
        reg = ScraperRegistry()
        reg.register(SelectorScraper("r", "tr", url_contains="a.com"))
        b.elements[("https://a.com", "tr")] = ["x" * 500] * 10
        assert len(HephaestusEngine(b, scrapers=reg).scrape_url("https://a.com")) == 2000


# ===========================================================================
# Politeness (#106)
# ===========================================================================

class TestPoliteness:
    def _engine(self, interval=2.0):
        t = {"now": 100.0}
        waits = []
        engine = HephaestusEngine(
            MonBrowser({"https://a.com/1": "t", "https://a.com/2": "t", "https://b.com": "t"}),
            min_host_interval=interval, clock=lambda: t["now"],
            sleep=lambda s: (waits.append(round(s, 2)), t.__setitem__("now", t["now"] + s)),
        )
        return engine, waits, t

    def test_same_host_is_spaced_and_other_hosts_are_not(self):
        engine, waits, t = self._engine()
        engine.scrape_url("https://a.com/1")
        engine.scrape_url("https://b.com")
        engine.scrape_url("https://a.com/2")
        assert waits == [2.0]

    def test_no_wait_once_enough_time_has_passed(self):
        engine, waits, t = self._engine()
        engine.scrape_url("https://a.com/1")
        t["now"] += 5
        engine.scrape_url("https://a.com/2")
        assert waits == []

    def test_disabled_by_default(self):
        engine = HephaestusEngine(MonBrowser({"https://a.com": "t"}), sleep=lambda s: 1 / 0)
        engine.scrape_url("https://a.com")
        engine.scrape_url("https://a.com")

    def test_wait_is_capped(self):
        engine, waits, _ = self._engine(interval=500)
        engine.scrape_url("https://a.com/1")
        engine.scrape_url("https://a.com/2")
        assert waits == [10.0]


# ===========================================================================
# Watching pages (#101, #109)
# ===========================================================================

URL = "https://shop.example.com/phone"


class TestWatchPage:
    def test_requires_a_url(self):
        r = make_engine().handle("watch_page", {}, {})
        assert r["data"]["needs_clarification"]

    @pytest.mark.parametrize("url", ["http://localhost:8000", "192.168.0.1", "file:///etc/passwd"])
    def test_refuses_unsafe_urls_without_touching_the_browser(self, url):
        b = MonBrowser()
        r = make_engine(b).handle("watch_page", {"url": url}, {})
        assert r["data"]["needs_clarification"] and "can't watch" in r["response"]
        assert b.fetches == []

    def test_unreadable_page_creates_no_watch(self):
        e = make_engine(MonBrowser({}))
        r = e.handle("watch_page", {"url": URL}, {})
        assert r["confidence"] == 0.0 and "haven't set up" in r["response"]
        assert "not watching" in e.handle("list_watches", {}, {})["response"]

    def test_creates_baseline_without_alerting(self):
        b = MonBrowser({URL: "Phone page with specs and reviews"})
        e = make_engine(b)
        r = e.handle("watch_page", {"url": URL}, {})
        assert r["confidence"] > 0 and "Watching" in r["response"] and "daily" in r["response"]
        assert e._monitor_store.pending_alerts() == []
        assert len(e._monitor_store.list()) == 1

    def test_price_watch_reports_current_price_and_requires_one(self):
        e = make_engine(MonBrowser({URL: "Phone ₹80,000 in stock"}))
        r = e.handle("watch_page", {"url": URL, "track_price": True}, {})
        assert "₹80,000" in r["response"]
        e2 = make_engine(MonBrowser({URL: "Phone, no price shown"}))
        r2 = e2.handle("watch_page", {"url": URL, "track_price": True}, {})
        assert r2["data"]["needs_clarification"] and "selector" in r2["response"]
        assert e2._monitor_store.list() == []

    def test_watch_for_text_and_target_imply_price_tracking(self):
        e = make_engine(MonBrowser({URL: "Phone ₹80,000"}))
        r = e.handle("watch_page", {"url": URL, "watch_for": "a price drop", "target_price": "₹75,000"}, {})
        m = e._monitor_store.list()[0]
        assert m["track_price"] and m["target_price"] == 75000.0 and r["confidence"] > 0

    def test_bad_target_price_is_clarified(self):
        e = make_engine(MonBrowser({URL: "Phone ₹80,000"}))
        r = e.handle("watch_page", {"url": URL, "target_price": "cheap"}, {})
        assert r["data"]["needs_clarification"]

    def test_keywords_as_string_or_list(self):
        e = make_engine(MonBrowser({URL: "page"}))
        e.handle("watch_page", {"url": URL, "keywords": "admit card and results"}, {})
        assert e._monitor_store.list()[0]["keywords"] == ["admit card", "results"]

    @pytest.mark.parametrize("entities,minutes", [
        ({"interval": "hourly"}, 60), ({"interval": "daily"}, 1440), ({"interval": "weekly"}, 10080),
        ({"interval": "every 6 hours"}, 360), ({"interval": "every 2 days"}, 2880),
        ({"interval_hours": 3}, 180), ({"interval_minutes": 45}, 45), ({}, 1440),
    ])
    def test_interval_parsing(self, entities, minutes):
        e = make_engine(MonBrowser({URL: "page"}))
        e.handle("watch_page", {"url": URL, **entities}, {})
        assert e._monitor_store.list()[0]["interval_minutes"] == minutes

    def test_unparseable_interval_is_asked_about(self):
        r = make_engine(MonBrowser({URL: "page"})).handle("watch_page", {"url": URL, "interval": "whenever"}, {})
        assert r["data"]["needs_clarification"] and "How often" in r["response"]

    def test_too_frequent_is_raised_to_the_minimum_and_said_so(self):
        e = make_engine(MonBrowser({URL: "page"}))
        r = e.handle("watch_page", {"url": URL, "interval_minutes": 5}, {})
        assert e._monitor_store.list()[0]["interval_minutes"] == 30 and "30 minutes" in r["response"]

    def test_duplicate_name_and_limit(self):
        b = MonBrowser({URL: "page", "https://b.com": "page"})
        e = make_engine(b, store=MonitorStore(":memory:", max_monitors=1))
        assert e.handle("watch_page", {"url": URL, "name": "phone"}, {})["confidence"] > 0
        dup = e.handle("watch_page", {"url": "https://b.com", "name": "Phone"}, {})
        assert "already watching" in dup["response"]
        limit = e.handle("watch_page", {"url": "https://b.com", "name": "other"}, {})
        assert "maximum of 1" in limit["response"]

    def test_registered_scraper_is_attached_and_used(self):
        reg = ScraperRegistry()
        reg.register(SelectorScraper("shop", ".price", url_contains="shop.example.com"))
        b = MonBrowser()
        b.elements[(URL, ".price")] = ["₹80,000"]
        e = make_engine(b, scrapers=reg)
        r = e.handle("watch_page", {"url": URL, "track_price": True}, {})
        assert "shop scraper" in r["response"] and e._monitor_store.list()[0]["scraper"] == "shop"

    def test_selector_entity_is_used(self):
        b = MonBrowser()
        b.elements[(URL, "#price")] = ["₹80,000"]
        e = make_engine(b)
        e.handle("watch_page", {"url": URL, "track_price": True, "selector": "#price"}, {})
        assert (URL, "#price") in b.fetches


class TestListAndStop:
    def _with_two(self):
        b = MonBrowser({URL: "Phone ₹80,000", "https://jobs.example.com/list": "jobs"})
        e = make_engine(b)
        e.handle("watch_page", {"url": URL, "name": "Phone price", "track_price": True}, {})
        e.handle("watch_page", {"url": "https://jobs.example.com/list", "name": "Job board",
                                "keywords": ["intern"], "interval": "hourly"}, {})
        return e

    def test_list_describes_each_watch(self):
        r = self._with_two().handle("list_watches", {}, {})
        assert "2 pages" in r["response"] and "Phone price" in r["response"]
        assert "₹80,000" in r["response"] and "“intern”" in r["response"] and "hourly" in r["response"]
        assert len(r["data"]["watches"]) == 2

    def test_stop_by_name_url_and_inside_a_sentence(self):
        e = self._with_two()
        assert "stopped watching Phone price" in e.handle("stop_watching", {"name": "phone"}, {})["response"]
        assert "stopped watching Job board" in e.handle(
            "stop_watching", {"raw_query": "please stop watching the job board for me"}, {})["response"]
        assert e._monitor_store.list() == []

    def test_stop_unknown_ambiguous_or_unspecified_is_clarified(self):
        e = self._with_two()
        for ents in ({"name": "nothing"}, {"name": "e"}, {}):
            r = e.handle("stop_watching", ents, {})
            assert r["data"]["needs_clarification"], ents
        assert len(e._monitor_store.list()) == 2


class TestScheduledChecks:
    def _setup(self, **kw):
        clock = Clock()
        b = MonBrowser({URL: "Phone page. Price ₹80,000. In stock."})
        e = make_engine(b, monitor=kw.pop("monitor", None), clock=clock, **kw)
        e.handle("watch_page", {"url": URL, "name": "Phone", "track_price": True, "interval": "hourly"}, {})
        return e, b, clock

    def test_only_due_watches_are_fetched(self):
        e, b, clock = self._setup()
        b.fetches.clear()
        assert e.check_web_monitors() is None and b.fetches == []
        clock.advance(minutes=30)
        e.check_web_monitors()
        assert b.fetches == []
        clock.advance(minutes=30)
        e.check_web_monitors()
        assert b.fetches == [URL]

    def test_price_drop_is_returned_once(self):
        e, b, clock = self._setup()
        b.pages[URL] = "Phone page. Price ₹74,900. In stock."
        clock.advance(hours=1)
        text = e.check_web_monitors()
        assert "Phone" in text and "dropped from ₹80,000 to ₹74,900" in text
        clock.advance(hours=1)
        assert e.check_web_monitors() is None           # same price, nothing new

    def test_no_change_stays_silent(self):
        e, b, clock = self._setup()
        clock.advance(hours=1)
        assert e.check_web_monitors() is None

    def test_failures_are_silent_until_the_third_then_alert_once(self):
        e, b, clock = self._setup()
        b.pages[URL] = None
        out = []
        for _ in range(5):
            clock.advance(hours=1)
            out.append(e.check_web_monitors())
        assert out[:2] == [None, None]
        assert "couldn't read Phone for 3 checks" in out[2]
        assert out[3:] == [None, None]

    def test_error_text_from_an_old_style_agent_is_not_treated_as_content(self):
        class OldAgent:
            def get_page_text(self, url):
                return self.text
        agent = OldAgent()
        agent.text = "Phone ₹80,000 great phone with a display"
        clock = Clock()
        e = HephaestusEngine(agent, monitor_store=MonitorStore(":memory:"), now_fn=clock,
                             local_now_fn=lambda: datetime(2026, 10, 3, 12))
        e.handle("watch_page", {"url": URL, "name": "p", "keywords": ["sale"], "interval": "hourly"}, {})
        agent.text = "I couldn't read that page."
        clock.advance(hours=1)
        e.check_web_monitors()
        m = e._monitor_store.list()[0]
        assert m["failures"] == 1 and "great phone" in m["snapshot"]

    def test_quiet_hours_hold_alerts_until_they_end(self):
        hour = {"h": 23}
        e, b, clock = self._setup(quiet_hours=(22, 7), local_now_fn=lambda: datetime(2026, 10, 3, hour["h"]))
        b.pages[URL] = "Phone page. Price ₹70,000. In stock."
        clock.advance(hours=1)
        assert e.check_web_monitors() is None
        assert len(e._monitor_store.pending_alerts()) == 1
        hour["h"] = 8
        assert "dropped" in e.check_web_monitors()
        assert e._monitor_store.pending_alerts() == []

    def test_quiet_hours_window_that_does_not_wrap(self):
        e = make_engine(quiet_hours=(1, 5), local_now_fn=lambda: datetime(2026, 10, 3, 3))
        assert e._in_quiet_hours()
        e2 = make_engine(quiet_hours=(1, 5), local_now_fn=lambda: datetime(2026, 10, 3, 6))
        assert not e2._in_quiet_hours()
        assert not make_engine(quiet_hours=None)._in_quiet_hours()

    def test_alerts_survive_a_restart_before_delivery(self, tmp_path):
        path = str(tmp_path / "h.db")
        clock = Clock()
        b = MonBrowser({URL: "Phone ₹80,000"})
        e = make_engine(b, store=MonitorStore(path), clock=clock, quiet_hours=(22, 7),
                        local_now_fn=lambda: datetime(2026, 10, 3, 2))
        e.handle("watch_page", {"url": URL, "name": "Phone", "track_price": True, "interval": "hourly"}, {})
        b.pages[URL] = "Phone ₹70,000"
        clock.advance(hours=1)
        assert e.check_web_monitors() is None
        e2 = make_engine(b, store=MonitorStore(path), clock=clock, quiet_hours=(22, 7),
                         local_now_fn=lambda: datetime(2026, 10, 3, 9))
        assert "dropped" in e2.check_web_monitors()

    def test_many_alerts_are_summarised(self):
        e = make_engine()
        for i in range(5):
            e._monitor_store.add_alert(None, f"Alert {i}.", T0)
        text = e._deliver_alerts(respect_quiet=False)
        assert "Alert 0." in text and "Alert 2." in text and "Plus 2 more" in text and "Alert 4" not in text

    def test_no_store_or_no_browser_is_a_noop_and_never_raises(self):
        assert HephaestusEngine(MonBrowser()).check_web_monitors() is None
        e = make_engine(MonBrowser({URL: "page"}))
        e.handle("watch_page", {"url": URL}, {})
        e._monitor_store.list = MagicMock(side_effect=RuntimeError("db gone"))
        e._monitor_store.due = MagicMock(side_effect=RuntimeError("db gone"))
        assert e.check_web_monitors() is None

    def test_one_run_checks_at_most_ten(self):
        clock = Clock()
        b = MonBrowser({f"https://s{i}.example.com": "page" for i in range(14)})
        e = make_engine(b, clock=clock, store=MonitorStore(":memory:", max_monitors=20))
        for i in range(14):
            e.handle("watch_page", {"url": f"https://s{i}.example.com", "name": f"s{i}"}, {})
        b.fetches.clear()
        clock.advance(days=1)
        e.check_web_monitors()
        assert len(b.fetches) == 10
        e.check_web_monitors()
        assert len(b.fetches) == 14

    def test_check_watches_now_reports_changes_or_nothing(self):
        e, b, clock = self._setup()
        assert "Nothing new on 1 watched page" in e.handle("check_watches", {}, {})["response"]
        b.pages[URL] = "Phone page. Price ₹60,000."
        assert "dropped" in e.handle("check_watches", {"name": "phone"}, {})["response"]
        assert e.handle("check_watches", {"name": "zzz"}, {})["data"]["needs_clarification"]

    def test_check_watches_ignores_quiet_hours_because_the_user_asked(self):
        e, b, clock = self._setup(quiet_hours=(0, 23), local_now_fn=lambda: datetime(2026, 10, 3, 3))
        b.pages[URL] = "Phone page. Price ₹60,000."
        assert "dropped" in e.handle("check_watches", {}, {})["response"]

    def test_keyword_watch_end_to_end(self):
        clock = Clock()
        b = MonBrowser({URL: "Notices. Results soon."})
        e = make_engine(b, clock=clock)
        e.handle("watch_page", {"url": URL, "name": "Exam", "keywords": ["admit card"], "interval": "hourly"}, {})
        b.pages[URL] = "Notices. Results soon. Admit Card released today."
        clock.advance(hours=1)
        assert "Exam: “admit card” now appears" in e.check_web_monitors()


class TestMonitorThreading:
    """Playwright's sync API is bound to the thread that started it, so every
    monitor fetch must run on the same dedicated worker thread."""

    def test_all_fetches_run_on_one_non_main_thread(self):
        clock = Clock()
        mon = MonBrowser({URL: "Phone ₹80,000 and more words", "https://b.example.com": "b"})
        e = make_engine(MonBrowser(), monitor=mon, clock=clock)
        e.handle("watch_page", {"url": URL, "name": "a", "interval": "hourly"}, {})      # caller thread 1
        t = threading.Thread(target=lambda: e.handle("watch_page", {"url": "https://b.example.com", "name": "b"}, {}))
        t.start(); t.join()                                                                # caller thread 2
        clock.advance(days=1)
        e.check_web_monitors()
        e.handle("check_watches", {}, {})
        assert len(set(mon.threads)) == 1
        assert mon.threads[0] != threading.get_ident()
        e.close()

    def test_dedicated_monitor_browser_is_closed_after_a_batch_but_chat_browser_is_not(self):
        chat, mon = MonBrowser(), MonBrowser({URL: "page of words"})
        e = make_engine(chat, monitor=mon)
        e.handle("watch_page", {"url": URL}, {})
        assert mon.closed >= 1 and chat.closed == 0
        e.close()

    def test_shared_browser_is_never_closed_by_monitoring(self):
        shared = MonBrowser({URL: "page of words"})
        e = make_engine(shared)
        e.handle("watch_page", {"url": URL}, {})
        e.handle("check_watches", {}, {})
        assert shared.closed == 0
        e.close()
        assert shared.closed == 0

    def test_a_hung_fetch_counts_as_a_failure_not_a_hang(self, monkeypatch):
        import modules.hephaestus.engine as eng
        monkeypatch.setattr(eng, "_MONITOR_FETCH_TIMEOUT", 0.05)
        release = threading.Event()

        class Slow(MonBrowser):
            def fetch_text(self, url, **kw):
                if self.pages.get("slow"):
                    release.wait(2)
                return super().fetch_text(url, **kw)

        mon = Slow({URL: "page of words"})
        clock = Clock()
        e = make_engine(MonBrowser(), monitor=mon, clock=clock)
        e.handle("watch_page", {"url": URL, "interval": "hourly"}, {})
        mon.pages["slow"] = True
        clock.advance(hours=1)
        assert e.check_web_monitors() is None
        assert e._monitor_store.list()[0]["failures"] == 1
        release.set()
        e.close()


# ===========================================================================
# Form filling (#102)
# ===========================================================================

FORMS = {
    "Scholarship": {
        "url": "https://apply.example.org/form",
        "fields": {"#name": "Asha Rao", "#email": "asha@example.com", "#role": "{position}"},
        "submit_selector": "button[type=submit]",
    },
    "Newsletter": {"url": "https://example.org/news", "fields": {"#e": "asha@example.com"}},
}


class TestFillForm:
    def _engine(self, **kw):
        return HephaestusEngine(MonBrowser(), forms=FORMS, **kw)

    def test_no_forms_configured(self):
        r = HephaestusEngine(MonBrowser()).handle("fill_form", {"form": "x"}, {})
        assert r["confidence"] == 0.0 and "hephaestus.forms" in r["response"]

    def test_unknown_or_missing_name_lists_the_choices(self):
        e = self._engine()
        for ents in ({}, {"form": "passport"}):
            r = e.handle("fill_form", ents, {})
            assert r["data"]["needs_clarification"] and "Scholarship" in r["response"]

    def test_missing_placeholder_is_asked_for(self):
        r = self._engine().handle("fill_form", {"form": "scholarship"}, {})
        assert r["data"]["needs_clarification"] and "position" in r["response"]

    def test_first_call_only_asks_and_never_touches_the_browser(self):
        e = self._engine()
        r = e.handle("fill_form", {"form": "scholarship", "values": {"position": "Analyst"}}, {})
        assert r["needs_confirmation"] and r["confirm_intent"] == "fill_form"
        assert "and submit it" in r["response"] and "apply.example.org" in r["response"]
        assert "3 fields" in r["response"]
        assert e._browser.fill_calls == []

    def test_confirmation_never_reveals_field_values(self):
        r = self._engine().handle("fill_form", {"form": "scholarship", "values": {"position": "Analyst"}}, {})
        assert "Asha" not in r["response"] and "asha@example.com" not in r["response"]
        assert "Analyst" not in r["response"]

    def test_confirmed_call_fills_with_substituted_values_and_submits(self):
        e = self._engine()
        first = e.handle("fill_form", {"form": "scholarship", "values": {"position": "Analyst"}}, {})
        r = e.handle("fill_form", {**first["confirm_entities"], "_confirmed": True}, {})
        url, fields, submit = e._browser.fill_calls[0]
        assert url == "https://apply.example.org/form" and submit == "button[type=submit]"
        assert fields == {"#name": "Asha Rao", "#email": "asha@example.com", "#role": "Analyst"}
        assert r["data"]["submitted"] is True and r["confidence"] > 0

    def test_form_without_submit_selector_says_it_will_not_submit(self):
        e = self._engine()
        r = e.handle("fill_form", {"form": "newsletter"}, {})
        assert "without submitting" in r["response"]
        e.handle("fill_form", {"form": "newsletter", "_confirmed": True}, {})
        assert e._browser.fill_calls[0][2] is None

    def test_matches_the_form_name_inside_a_spoken_sentence_longest_first(self):
        forms = {"Gate": {"url": "https://a.org/g", "fields": {"#a": "1"}},
                 "Gate application": {"url": "https://a.org/ga", "fields": {"#a": "1"}}}
        e = HephaestusEngine(MonBrowser(), forms=forms)
        r = e.handle("fill_form", {"raw_query": "fill in my gate application form please"}, {})
        assert r["data"]["url"] == "https://a.org/ga"

    def test_browser_error_is_reported(self):
        e = self._engine()
        e._browser.fill_form = MagicMock(side_effect=RuntimeError("boom"))
        r = e.handle("fill_form", {"form": "newsletter", "_confirmed": True}, {})
        assert r["confidence"] == 0.0 and "went wrong" in r["response"]

    def test_invalid_profiles_are_dropped_not_fatal(self):
        e = HephaestusEngine(MonBrowser(), forms={
            "local": {"url": "http://localhost/f", "fields": {"#a": "b"}},
            "empty": {"url": "https://a.org", "fields": {}},
            "nourl": {"fields": {"#a": "b"}},
            "good": {"url": "https://a.org/f", "fields": {"#a": "b"}},
        })
        assert list(e._forms) == ["good"]

    def test_placeholders_only_come_from_the_values_dict(self):
        e = self._engine()
        r = e.handle("fill_form", {"form": "scholarship", "position": "Analyst"}, {})
        assert r["data"]["needs_clarification"]


# ===========================================================================
# Repo summary (#110)
# ===========================================================================

def _build_repo(root):
    (root / "pkg").mkdir()
    (root / "tests").mkdir()
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.js").write_text("TODO\n" * 50)
    (root / "main.py").write_text("def run():\n    return 1\n")
    long_body = "\n".join(f"    x{i} = {i}" for i in range(90))
    (root / "pkg" / "core.py").write_text(
        f"def big():\n{long_body}\n\ndef f():\n    try:\n        pass\n    except:\n        pass\n"
        "# TODO: fix this\n# FIXME later\n")
    (root / "pkg" / "broken.py").write_text("def oops(:\n")
    (root / "tests" / "test_core.py").write_text("def test_a():\n    assert True\n")
    (root / "requirements.txt").write_text("requests\n")
    (root / "README.md").write_text("# Title\n")


class TestRepoSummary:
    def test_counts_and_findings(self, tmp_path):
        _build_repo(tmp_path)
        s = summarize_repo(str(tmp_path))
        assert s["code_files"] == 4 and s["test_files"] == 1
        assert [l for l, *_ in s["languages"]][0] == "Python"
        assert "main.py" in s["entry_points"] and "requirements.txt" in s["manifests"]
        assert s["markers"] == {"TODO": 1, "FIXME": 1}          # node_modules is skipped
        assert s["bare_excepts"] == 1
        assert s["unparsable"] == ["pkg/broken.py"]
        assert s["long_functions"][0][1].startswith("pkg/core.py:1 big()")
        assert not s["truncated"]

    def test_output_contains_no_file_contents(self, tmp_path):
        _build_repo(tmp_path)
        (tmp_path / "secret.py").write_text("API_KEY = 'hunter2-super-secret'\n")
        text = format_summary(summarize_repo(str(tmp_path)))
        assert "hunter2" not in text and "API_KEY" not in text

    def test_spoken_summary_mentions_the_key_facts(self, tmp_path):
        _build_repo(tmp_path)
        text = format_summary(summarize_repo(str(tmp_path)), name="demo")
        for needle in ("demo:", "Python", "tests", "very long function", "bare except", "don't parse", "TODO"):
            assert needle in text, needle

    def test_symlinks_are_not_followed(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("x = 1\n" * 100)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("x = 1\n")
        try:
            os.symlink(str(outside), str(repo / "link"))
            os.symlink(str(outside / "secret.py"), str(repo / "s.py"))
        except (OSError, NotImplementedError):
            return
        s = summarize_repo(str(repo))
        assert s["code_files"] == 1 and s["code_lines"] == 1

    def test_empty_and_missing_folders(self, tmp_path):
        assert "didn't find any source code" in format_summary(summarize_repo(str(tmp_path)))
        with pytest.raises(NotADirectoryError):
            summarize_repo(str(tmp_path / "nope"))

    def test_file_cap_marks_truncated(self, tmp_path, monkeypatch):
        import modules.hephaestus.repo_summary as rs
        monkeypatch.setattr(rs, "_MAX_FILES", 3)
        for i in range(6):
            (tmp_path / f"f{i}.py").write_text("x = 1\n")
        s = summarize_repo(str(tmp_path))
        assert s["truncated"] and "stopped after" in format_summary(s)

    def test_intent_happy_path_and_errors(self, tmp_path):
        _build_repo(tmp_path)
        e = HephaestusEngine(MonBrowser())
        ok = e.handle("summarize_repo", {"path": str(tmp_path)}, {})
        assert ok["confidence"] > 0 and "code files" in ok["response"]
        assert e.handle("summarize_repo", {}, {})["data"]["needs_clarification"]
        assert "couldn't find a folder" in e.handle("summarize_repo", {"path": str(tmp_path / "x")}, {})["response"]
        assert "couldn't find a folder" in e.handle("summarize_repo", {"path": str(tmp_path / "main.py")}, {})["response"]

    def test_repo_roots_allow_list(self, tmp_path):
        inside = tmp_path / "work" / "proj"
        inside.mkdir(parents=True)
        (inside / "a.py").write_text("x = 1\n")
        outside = tmp_path / "other"
        outside.mkdir()
        (outside / "b.py").write_text("x = 1\n")
        sneaky = tmp_path / "work-evil"
        sneaky.mkdir()
        e = HephaestusEngine(MonBrowser(), repo_roots=[str(tmp_path / "work")])
        assert e.handle("summarize_repo", {"path": str(inside)}, {})["confidence"] > 0
        assert "outside" in e.handle("summarize_repo", {"path": str(outside)}, {})["response"]
        assert "outside" in e.handle("summarize_repo", {"path": str(sneaky)}, {})["response"]


# ===========================================================================
# Heartbeat hook (#101) and CLI (#108)
# ===========================================================================

class TestHeartbeatHook:
    def _hb(self, hephaestus):
        return HestiaHeartbeat(interval=1800, hephaestus=hephaestus)

    def test_speaks_the_alert(self):
        heph = MagicMock()
        heph.check_web_monitors.return_value = "Phone: price dropped."
        with patch("core.heartbeat.bus") as bus:
            self._hb(heph)._maybe_run_hephaestus_checks()
        bus.emit.assert_called_once_with("speak", {"text": "Phone: price dropped."})

    @pytest.mark.parametrize("value", [None, "", "   ", 5])
    def test_silent_when_nothing_to_say(self, value):
        heph = MagicMock()
        heph.check_web_monitors.return_value = value
        with patch("core.heartbeat.bus") as bus:
            self._hb(heph)._maybe_run_hephaestus_checks()
        bus.emit.assert_not_called()

    def test_exception_is_swallowed(self):
        heph = MagicMock()
        heph.check_web_monitors.side_effect = RuntimeError("boom")
        with patch("core.heartbeat.bus") as bus:
            self._hb(heph)._maybe_run_hephaestus_checks()
        bus.emit.assert_not_called()

    def test_absent_or_incompatible_collaborator_is_skipped(self):
        with patch("core.heartbeat.bus") as bus:
            self._hb(None)._maybe_run_hephaestus_checks()
            self._hb(object())._maybe_run_hephaestus_checks()
        bus.emit.assert_not_called()

    def test_runs_on_every_tick(self):
        heph = MagicMock()
        heph.check_web_monitors.return_value = None
        with patch("core.heartbeat.bus"):
            self._hb(heph)._run_heartbeat()
        heph.check_web_monitors.assert_called_once()


class TestHeadedFlag:
    def test_flag_parses(self):
        import main as main_module
        assert main_module._parse_args(["--headed"]).headed is True
        assert main_module._parse_args([]).headed is False
