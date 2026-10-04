import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# How many failure screenshots to keep on disk (oldest are deleted first).
_MAX_FAILURE_SHOTS = 20


class HestiaBrowserAgent:
    """Playwright-powered browser automation with voice confirmation before acting."""

    def __init__(
        self,
        confirm_fn: Callable[[str], bool] = None,
        headless: bool = True,
        *,
        screenshot_dir: Optional[str] = None,
        slow_mo_ms: int = 0,
        idle_timeout_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            confirm_fn: Callable that speaks a question and returns True if user
                        confirms (says yes), False otherwise.
                        If None, all actions auto-confirm (use carefully).
            headless: Run browser headlessly (default True).
                      Set False for debugging or actions needing visible browser
                      (backlog #108; ``python main.py --headed`` does this).
            screenshot_dir: When set, a failed page action saves a full-page
                        screenshot here (backlog #103) and logs the path. Off by
                        default because a screenshot can capture personal pages.
                        Only the newest few are kept.
            slow_mo_ms: Milliseconds Playwright waits between actions. Useful
                        with headless=False so you can follow what it's doing
                        (backlog #108). 0 = full speed.
            idle_timeout_seconds: Close the browser after this long without use
                        and relaunch on the next request (backlog #107). 0 = keep
                        it open until close(). Checked on the next call and by
                        close_if_idle(); there is no background thread.
        """
        self.confirm_fn = confirm_fn
        self.headless = headless
        self.screenshot_dir = screenshot_dir
        self.slow_mo_ms = max(0, int(slow_mo_ms or 0))
        self.idle_timeout_seconds = max(0.0, float(idle_timeout_seconds or 0.0))
        self.last_failure_screenshot: Optional[str] = None
        self._clock = clock
        self._browser = None
        self._playwright = None
        # One long-lived browser context shared by every page (backlog #107):
        # the browser is launched once, and pages no longer each create (and
        # leak) a fresh context. Cookies and consent choices therefore persist
        # between tasks until close() or the idle timeout.
        self._context = None
        self._context_browser = None
        self._last_used = self._clock()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Browser / session lifecycle
    # ------------------------------------------------------------------

    def _get_browser(self):
        """Lazy-init Playwright and Chromium browser. Reuse if already running."""
        with self._lock:
            if self._browser and self._idle_expired():
                logger.info("Browser idle for over %.0fs; closing it.", self.idle_timeout_seconds)
                self._close_locked()
            if self._browser:
                try:
                    if self._browser.is_connected():
                        self._last_used = self._clock()
                        return self._browser
                except Exception:
                    # Playwright crashed or the connection is dead — reset and
                    # fall through to re-initialize below.
                    self._browser = None
                    self._context = None
                    self._context_browser = None
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                logger.error("playwright not installed. Run: pip install playwright && playwright install chromium")
                return None
            self._playwright = sync_playwright().__enter__()
            launch_kwargs = {"headless": self.headless}
            if self.slow_mo_ms:
                launch_kwargs["slow_mo"] = self.slow_mo_ms
            self._browser = self._playwright.chromium.launch(**launch_kwargs)
            self._last_used = self._clock()
            return self._browser

    def _idle_expired(self) -> bool:
        return (
            self.idle_timeout_seconds > 0
            and (self._clock() - self._last_used) > self.idle_timeout_seconds
        )

    def _close_locked(self) -> None:
        """Close context, browser and Playwright. Caller holds self._lock."""
        for closer in (
            lambda: self._context.close() if self._context else None,
            lambda: self._browser.close() if self._browser else None,
            lambda: self._playwright.__exit__(None, None, None) if self._playwright else None,
        ):
            try:
                closer()
            except Exception:
                pass
        self._context = None
        self._context_browser = None
        self._browser = None
        self._playwright = None

    def close(self) -> None:
        """Close browser and Playwright instance cleanly."""
        with self._lock:
            self._close_locked()

    def close_if_idle(self) -> bool:
        """Close the browser if it has sat unused past ``idle_timeout_seconds``.
        Returns True if it was closed. Safe to call from a periodic job."""
        with self._lock:
            if self._browser and self._idle_expired():
                self._close_locked()
                return True
        return False

    def _confirm(self, question: str) -> bool:
        """Ask user to confirm an action. Returns True if confirmed."""
        if self.confirm_fn is None:
            return True
        return self.confirm_fn(question)

    def _get_context(self, browser):
        """Return the shared context, creating it on first use (or if the
        browser was replaced since)."""
        if self._context is not None and self._context_browser is browser:
            return self._context
        self._context = browser.new_context(user_agent=_USER_AGENT)
        self._context_browser = browser
        return self._context

    def _new_page(self):
        """Open a new browser page. Returns page or None on failure."""
        browser = self._get_browser()
        if not browser:
            return None
        try:
            return self._get_context(browser).new_page()
        except Exception as e:
            logger.warning("Failed to open page: %s", e)
            # The shared context may have been closed or crashed; forget it so
            # the next call builds a fresh one rather than failing forever.
            self._context = None
            self._context_browser = None
            return None

    # ------------------------------------------------------------------
    # Failure diagnostics (backlog #103)
    # ------------------------------------------------------------------

    def _snap_failure(self, page, label: str) -> Optional[str]:
        """Save a screenshot of *page* for debugging, if screenshot_dir is set.
        Never raises; returns the file path or None."""
        if not self.screenshot_dir or page is None:
            return None
        try:
            directory = Path(self.screenshot_dir)
            directory.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", label).strip("-")[:40] or "page"
            path = directory / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{slug}.png"
            page.screenshot(path=str(path), full_page=True)
            self.last_failure_screenshot = str(path)
            logger.warning("Saved failure screenshot: %s", path)
            shots = sorted(directory.glob("*.png"))
            for old in shots[:-_MAX_FAILURE_SHOTS]:
                try:
                    old.unlink()
                except OSError:
                    pass
            return str(path)
        except Exception as e:
            logger.debug("Could not save failure screenshot: %s", e)
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_web(self, query: str) -> str:
        """
        Open DuckDuckGo, search for query, return top 3 result titles and snippets.
        Does NOT require confirmation — read-only action.
        """
        results = self.search_web_results(query, max_results=3)
        if results:
            return " | ".join(r["title"] for r in results if r.get("title"))
        return f"No results found for '{query}'."

    def search_web_results(self, query: str, max_results: int = 3) -> list[dict[str, str]]:
        """
        Open DuckDuckGo, search for query, return up to *max_results*
        ``{"title": ..., "url": ...}`` dicts. Unlike search_web(), this keeps
        the result URLs so callers (e.g. Hephaestus.search_and_summarize)
        can scrape each result page instead of just showing its title.
        Does NOT require confirmation — read-only action.
        """
        page = self._new_page()
        if not page:
            logger.warning("search_web_results: browser not available.")
            return []

        try:
            from urllib.parse import quote_plus
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            if not url.startswith("http"):
                url = "https://" + url
            page.goto(url, timeout=20000)

            # wait for basic content instead of fragile selectors
            page.wait_for_load_state("domcontentloaded", timeout=10000)

            links = page.query_selector_all("a.result__a")[:max_results]

            out: list[dict[str, str]] = []
            for link in links:
                title = (link.inner_text() or "").strip()
                href = (link.get_attribute("href") or "").strip()
                if title:
                    out.append({"title": title, "url": href})

            page.close()
            return out

        except Exception as e:
            logger.warning("search_web_results error: %s", e)
            self._snap_failure(page, "search")
            try:
                page.close()
            except Exception:
                pass
            return []

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def open_url(self, url: str, confirm: bool = True) -> str:
        """
        Navigate to a URL and return the page title.
        Requires confirmation if confirm=True.
        """
        if confirm:
            if not self._confirm(f"Should I open {url} in the browser?"):
                return "Okay, I won't open that."
        page = self._new_page()
        if not page:
            return "Browser is not available."
        try:
            if not url.startswith("http"):
                url = "https://" + url
            page.goto(url, timeout=20000)
            title = page.title()
            page.close()
            return f"Opened {title}."
        except Exception as e:
            logger.warning("open_url error: %s", e)
            self._snap_failure(page, f"open-{url}")
            try:
                page.close()
            except Exception:
                pass
            return f"I couldn't open that page."

    def fill_form(self, url: str, fields: dict[str, str], submit_selector: str = None) -> str:
        """
        Navigate to URL, fill form fields, optionally submit.
        fields: dict mapping CSS selector -> value to type.
        submit_selector: CSS selector for submit button (optional).
        Always requires confirmation before submitting.

        If any field can't be filled the form is NOT submitted: a half-filled
        application is worse than no application, so the reply names the
        selectors that failed instead.
        """
        if not self._confirm(f"I'll fill out a form at {url}. Should I proceed?"):
            return "Okay, cancelled."
        page = self._new_page()
        if not page:
            return "Browser is not available."
        try:
            if not url.startswith("http"):
                url = "https://" + url
            page.goto(url, timeout=20000)
            failed: list[str] = []
            for selector, value in fields.items():
                try:
                    page.fill(selector, value)
                    time.sleep(0.3)
                except Exception as e:
                    logger.warning("fill_form field error (%s): %s", selector, e)
                    failed.append(selector)

            if failed:
                self._snap_failure(page, "form-fields")

            if submit_selector:
                if failed:
                    page.close()
                    return (
                        f"I couldn't fill {len(failed)} field(s) ({', '.join(failed)}), "
                        "so I didn't submit the form."
                    )
                if not self._confirm("Form filled. Should I submit it now?"):
                    page.close()
                    return "Form filled but not submitted."
                page.click(submit_selector)
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                except Exception as e:
                    # The click already happened; don't claim failure or success.
                    logger.warning("fill_form: page did not settle after submit: %s", e)
                    self._snap_failure(page, "form-after-submit")
                    page.close()
                    return (
                        "I clicked submit, but the page didn't finish loading, "
                        "so I can't confirm it went through."
                    )
                title = page.title()
                page.close()
                return f"Form submitted. Page title is now: {title}."
            page.close()
            if failed:
                return (
                    f"Form partly filled ({len(failed)} field(s) failed) "
                    "but not submitted — no submit button specified."
                )
            return "Form filled but not submitted — no submit button specified."
        except Exception as e:
            logger.warning("fill_form error: %s", e)
            self._snap_failure(page, "form")
            try:
                page.close()
            except Exception:
                pass
            return "Something went wrong filling that form."

    # ------------------------------------------------------------------
    # Reading pages
    # ------------------------------------------------------------------

    def _load_text(self, url: str, max_chars: int, wait_ms: int = 0):
        """Load *url* and return ``(text, error)``. ``error`` is None on
        success, ``"unavailable"`` when no browser could be opened, or
        ``"failed"`` when the page couldn't be read."""
        page = self._new_page()
        if not page:
            return None, "unavailable"
        try:
            if not url.startswith("http"):
                url = "https://" + url
            page.goto(url, timeout=20000)
            page.wait_for_load_state("domcontentloaded", timeout=8000)
            if wait_ms:
                # Let client-side rendering finish on script-built pages.
                page.wait_for_timeout(wait_ms)
            text = page.inner_text("body")
            text = " ".join(text.split())[:max_chars]
            page.close()
            return text, None
        except Exception as e:
            logger.warning("get_page_text error: %s", e)
            self._snap_failure(page, f"read-{url}")
            try:
                page.close()
            except Exception:
                pass
            return None, "failed"

    def get_page_text(self, url: str) -> str:
        """
        Fetch a page and return visible text content (first 500 chars).
        Read-only — no confirmation needed.
        """
        text, error = self._load_text(url, 500)
        if error == "unavailable":
            return "Browser is not available."
        if error:
            return "I couldn't read that page."
        return text or "Page loaded but no readable content found."

    def fetch_text(self, url: str, max_chars: int = 20000, wait_ms: int = 1500) -> Optional[str]:
        """
        Fetch a page's visible text, up to *max_chars* (default 20,000).

        Unlike get_page_text() this returns ``None`` on any failure — browser
        unavailable, load error, or an empty page — instead of a sentence
        that looks like content. Page monitors rely on that: an error message
        must never be mistaken for the page having changed.
        """
        max_chars = max(100, min(int(max_chars), 100_000))
        text, error = self._load_text(url, max_chars, wait_ms=wait_ms)
        if error or not text:
            return None
        return text

    def fetch_elements(
        self, url: str, selector: str, max_items: int = 50, wait_ms: int = 1500
    ) -> Optional[list[str]]:
        """
        Return the visible text of every element matching CSS *selector*
        (up to *max_items*), for site-specific scrapers (backlog #105).

        ``None`` means the page couldn't be loaded; ``[]`` means it loaded but
        nothing matched, which usually means the site's layout changed.
        """
        page = self._new_page()
        if not page:
            return None
        try:
            if not url.startswith("http"):
                url = "https://" + url
            page.goto(url, timeout=20000)
            page.wait_for_load_state("domcontentloaded", timeout=8000)
            if wait_ms:
                page.wait_for_timeout(wait_ms)
            items: list[str] = []
            for el in page.query_selector_all(selector)[: max(1, int(max_items))]:
                text = " ".join((el.inner_text() or "").split())
                if text:
                    items.append(text)
            page.close()
            return items
        except Exception as e:
            logger.warning("fetch_elements error: %s", e)
            self._snap_failure(page, f"select-{url}")
            try:
                page.close()
            except Exception:
                pass
            return None

    def check_flight_status(self, flight_number: str) -> str:
        """
        Search Google Flights for a flight status.
        Read-only — no confirmation needed.
        """
        query = f"{flight_number} flight status today"
        page = self._new_page()
        if not page:
            return "Browser is not available."
        try:
            page.goto(
                f"https://www.google.com/search?q={query.replace(' ', '+')}",
                timeout=15000
            )
            page.wait_for_load_state("domcontentloaded", timeout=8000)
            # Try to get the flight info card
            selectors = [
                "[data-attrid='kc:/travel/flight:status']",
                ".kp-wholepage",
                "#rso .g"
            ]
            for sel in selectors:
                try:
                    el = page.query_selector(sel)
                    if el:
                        text = el.inner_text()
                        text = " ".join(text.split())[:300]
                        page.close()
                        return text
                except Exception:
                    continue
            page.close()
            return f"I couldn't find status for flight {flight_number}."
        except Exception as e:
            logger.warning("check_flight_status error: %s", e)
            self._snap_failure(page, f"flight-{flight_number}")
            try:
                page.close()
            except Exception:
                pass
            return "I couldn't check that flight status."
