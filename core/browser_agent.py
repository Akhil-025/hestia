import logging
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class HestiaBrowserAgent:
    """Playwright-powered browser automation with voice confirmation before acting."""

    def __init__(self, confirm_fn: Callable[[str], bool] = None, headless: bool = True):
        """
        Args:
            confirm_fn: Callable that speaks a question and returns True if user
                        confirms (says yes), False otherwise.
                        If None, all actions auto-confirm (use carefully).
            headless: Run browser headlessly (default True).
                      Set False for debugging or actions needing visible browser.
        """
        self.confirm_fn = confirm_fn
        self.headless = headless
        self._browser = None
        self._playwright = None
        self._lock = threading.Lock()

    def _get_browser(self):
        """Lazy-init Playwright and Chromium browser. Reuse if already running."""
        with self._lock:
            if self._browser:
                try:
                    if self._browser.is_connected():
                        return self._browser
                except Exception:
                    # Playwright crashed or the connection is dead — reset and
                    # fall through to re-initialize below.
                    self._browser = None
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                logger.error("playwright not installed. Run: pip install playwright && playwright install chromium")
                return None
            self._playwright = sync_playwright().__enter__()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            return self._browser

    def close(self) -> None:
        """Close browser and Playwright instance cleanly."""
        with self._lock:
            try:
                if self._browser:
                    self._browser.close()
                if self._playwright:
                    self._playwright.__exit__(None, None, None)
            except Exception:
                pass
            self._browser = None
            self._playwright = None

    def _confirm(self, question: str) -> bool:
        """Ask user to confirm an action. Returns True if confirmed."""
        if self.confirm_fn is None:
            return True
        return self.confirm_fn(question)

    def _new_page(self):
        """Open a new browser page. Returns page or None on failure."""
        browser = self._get_browser()
        if not browser:
            return None
        try:
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            return context.new_page()
        except Exception as e:
            logger.warning("Failed to open page: %s", e)
            return None

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
            try:
                page.close()
            except Exception:
                pass
            return []

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
            for selector, value in fields.items():
                try:
                    page.fill(selector, value)
                    time.sleep(0.3)
                except Exception as e:
                    logger.warning("fill_form field error (%s): %s", selector, e)

            if submit_selector:
                if not self._confirm("Form filled. Should I submit it now?"):
                    page.close()
                    return "Form filled but not submitted."
                page.click(submit_selector)
                page.wait_for_load_state("networkidle", timeout=10000)
                title = page.title()
                page.close()
                return f"Form submitted. Page title is now: {title}."
            page.close()
            return "Form filled but not submitted — no submit button specified."
        except Exception as e:
            logger.warning("fill_form error: %s", e)
            try:
                page.close()
            except Exception:
                pass
            return "Something went wrong filling that form."

    def get_page_text(self, url: str) -> str:
        """
        Fetch a page and return visible text content (first 500 chars).
        Read-only — no confirmation needed.
        """
        page = self._new_page()
        if not page:
            return "Browser is not available."
        try:
            if not url.startswith("http"):
                url = "https://" + url
            page.goto(url, timeout=20000)
            page.wait_for_load_state("domcontentloaded", timeout=8000)
            text = page.inner_text("body")
            text = " ".join(text.split())[:500]
            page.close()
            return text or "Page loaded but no readable content found."
        except Exception as e:
            logger.warning("get_page_text error: %s", e)
            try:
                page.close()
            except Exception:
                pass
            return "I couldn't read that page."

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
            try:
                page.close()
            except Exception:
                pass
            return "I couldn't check that flight status."