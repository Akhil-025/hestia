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

# Upper bound for ``pool_size``: every pooled browser is a full Chromium
# process, so a typo like 50 must not be able to exhaust the machine.
_MAX_POOL_SIZE = 8

# Form-field discovery (backlog #102). Only plain typed-text controls can be
# filled by ``page.fill`` (selects, checkboxes and files need other calls), so
# only those are offered. The sensitive-name filter is a guard, not a promise:
# card, ID and one-time-code fields are left out so a spoken command can never
# end up typing them into a page.
_FILLABLE_TYPES = frozenset({
    "input", "text", "email", "tel", "number", "url", "search", "date", "time",
    "datetime-local", "month", "week", "textarea",
})
_SENSITIVE_FIELD = re.compile(
    r"pass(word|code)?|pwd|card|cvv|cvc|ccnum|iban|routing|account.?n(o|um)|"
    r"\bpin\b|otp|one.?time|ssn|social.?security|aadhaa?r|\bpan\b|passport",
    re.I,
)
_MAX_DISCOVERED_FIELDS = 30
_SAFE_ID = re.compile(r"^[A-Za-z_][\w-]*$")
_SAFE_NAME = re.compile(r"^[^\"'\\\s\]\[]+$")

# Runs in the page. Returns plain data only; every decision about what is safe
# to offer is made in Python (``form_fields_from_raw``) where it can be tested.
_DISCOVER_JS = """
() => {
  const text = (v) => (v || '').replace(/\\s+/g, ' ').trim().slice(0, 80);
  const labelFor = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return text(aria);
    if (el.id) {
      for (const l of document.querySelectorAll('label[for]')) {
        if (l.getAttribute('for') === el.id) return text(l.innerText);
      }
    }
    const wrap = el.closest('label');
    if (wrap) return text(wrap.innerText);
    return text(el.getAttribute('placeholder') || el.getAttribute('name') || '');
  };
  const fields = [];
  for (const el of document.querySelectorAll('input, textarea, select')) {
    const rect = el.getBoundingClientRect();
    fields.push({
      tag: el.tagName.toLowerCase(),
      type: (el.getAttribute('type') || el.tagName).toLowerCase(),
      id: el.id || '',
      name: el.getAttribute('name') || '',
      label: labelFor(el),
      disabled: !!(el.disabled || el.readOnly),
      hidden: rect.width === 0 && rect.height === 0,
    });
  }
  const submits = [];
  for (const el of document.querySelectorAll(
      'button[type=submit], input[type=submit], form button:not([type])')) {
    submits.push({
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      type: (el.getAttribute('type') || '').toLowerCase(),
    });
  }
  return {fields, submits};
}
"""


class _Slot:
    """One pooled browser: the browser itself, its shared context, and the
    pages opened on it (so the pool can tell which browser is least busy)."""

    __slots__ = ("browser", "context", "pages")

    def __init__(self, browser):
        self.browser = browser
        self.context = None
        self.pages: list = []

    def live_pages(self) -> int:
        """Number of pages still open. Forgets pages that have been closed.
        Anything that can't confirm it is open (``is_closed()`` raising or not
        returning False) counts as closed, so a broken page never pins a
        browser as busy."""
        alive = []
        for page in self.pages:
            try:
                if page.is_closed() is False:
                    alive.append(page)
            except Exception:
                continue
        self.pages = alive
        return len(alive)


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
        pool_size: int = 1,
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
            idle_timeout_seconds: Close the browser(s) after this long without
                        use and relaunch on the next request (backlog #107).
                        0 = keep them open until close(). Checked on the next
                        call and by close_if_idle(); there is no background
                        thread.
            pool_size: Most browsers to keep open at once (backlog #107),
                        clamped to 1-8. Default 1 = one reused browser, exactly
                        as before. With more than one, a request goes to the
                        least-busy open browser and another is launched only
                        when every open one still has a page in use, so the
                        pool grows under concurrent work and stays at one
                        browser for sequential work. Each browser has its own
                        context, so cookies are shared per browser, not across
                        the pool.
        """
        self.confirm_fn = confirm_fn
        self.headless = headless
        self.screenshot_dir = screenshot_dir
        self.slow_mo_ms = max(0, int(slow_mo_ms or 0))
        self.idle_timeout_seconds = max(0.0, float(idle_timeout_seconds or 0.0))
        try:
            requested = int(pool_size or 1)
        except (TypeError, ValueError):
            requested = 1
        self.pool_size = max(1, min(requested, _MAX_POOL_SIZE))
        self.last_failure_screenshot: Optional[str] = None
        self._clock = clock
        self._playwright = None
        # Pooled browsers, oldest first (backlog #107). Each keeps one
        # long-lived context shared by every page opened on it, so pages no
        # longer each create (and leak) a fresh context, and cookies and
        # consent choices persist until close() or the idle timeout.
        self._slots: list[_Slot] = []
        self._last_used = self._clock()
        self._lock = threading.Lock()

    # Single-browser views of the pool, kept so code and tests written for one
    # browser still read naturally. They describe the first (oldest) browser.
    @property
    def _browser(self):
        return self._slots[0].browser if self._slots else None

    @property
    def _context(self):
        return self._slots[0].context if self._slots else None

    @property
    def _context_browser(self):
        return self._browser if self._context is not None else None

    # ------------------------------------------------------------------
    # Browser / session lifecycle
    # ------------------------------------------------------------------

    def _get_browser(self):
        """Lazy-init Playwright and return a browser to open a page on: the
        least-busy open one, or a newly launched one while the pool has room
        and every open browser is in use. Returns None if Playwright isn't
        installed."""
        with self._lock:
            if self._slots and self._idle_expired():
                logger.info("Browser idle for over %.0fs; closing it.", self.idle_timeout_seconds)
                self._close_locked()
            self._drop_dead_slots()
            slot = self._pick_slot()
            if slot is None:
                slot = self._launch_slot()
            if slot is None:
                return None
            self._last_used = self._clock()
            return slot.browser

    def _drop_dead_slots(self) -> None:
        """Forget browsers whose connection is gone so they get relaunched."""
        alive = []
        for slot in self._slots:
            try:
                connected = bool(slot.browser.is_connected())
            except Exception:
                # Playwright crashed or the connection is dead.
                connected = False
            if connected:
                alive.append(slot)
                continue
            logger.warning("A pooled browser lost its connection; dropping it.")
            for closer in (
                lambda s=slot: s.context.close() if s.context else None,
                lambda s=slot: s.browser.close(),
            ):
                try:
                    closer()
                except Exception:
                    pass
        self._slots = alive

    def _pick_slot(self) -> Optional[_Slot]:
        """Least-busy open browser, or None when a new one should be launched
        (nothing open yet, or all busy and the pool still has room)."""
        if not self._slots:
            return None
        least = min(self._slots, key=lambda s: s.live_pages())
        if len(self._slots) < self.pool_size and least.live_pages() > 0:
            return None
        return least

    def _launch_slot(self) -> Optional[_Slot]:
        """Launch one more browser. Caller holds self._lock. The first launch
        failing raises, as it always has; failing to grow an already-working
        pool just reuses an open browser."""
        reused_playwright = self._playwright is not None
        if self._playwright is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                logger.error("playwright not installed. Run: pip install playwright && playwright install chromium")
                return None
            self._playwright = sync_playwright().__enter__()
        launch_kwargs = {"headless": self.headless}
        if self.slow_mo_ms:
            launch_kwargs["slow_mo"] = self.slow_mo_ms
        try:
            browser = self._playwright.chromium.launch(**launch_kwargs)
        except Exception:
            if self._slots:
                logger.warning("Could not launch another browser; reusing an open one.", exc_info=True)
                return min(self._slots, key=lambda s: s.live_pages())
            if reused_playwright:
                # The driver itself may have died with the last browser; one
                # retry on a fresh Playwright before giving up.
                self._exit_playwright()
                return self._launch_slot()
            raise
        slot = _Slot(browser)
        self._slots.append(slot)
        return slot

    def _idle_expired(self) -> bool:
        return (
            self.idle_timeout_seconds > 0
            and (self._clock() - self._last_used) > self.idle_timeout_seconds
        )

    def _exit_playwright(self) -> None:
        try:
            if self._playwright:
                self._playwright.__exit__(None, None, None)
        except Exception:
            pass
        self._playwright = None

    def _close_locked(self) -> None:
        """Close every context and browser, then Playwright. Caller holds self._lock."""
        for slot in self._slots:
            for closer in (
                lambda s=slot: s.context.close() if s.context else None,
                lambda s=slot: s.browser.close(),
            ):
                try:
                    closer()
                except Exception:
                    pass
        self._slots = []
        self._exit_playwright()

    def close(self) -> None:
        """Close every browser and the Playwright instance cleanly."""
        with self._lock:
            self._close_locked()

    def close_if_idle(self) -> bool:
        """Close the browser(s) if they have sat unused past ``idle_timeout_seconds``.
        Returns True if they were closed. Safe to call from a periodic job."""
        with self._lock:
            if self._slots and self._idle_expired():
                self._close_locked()
                return True
        return False

    def pool_stats(self) -> dict:
        """Snapshot for diagnostics: configured size, browsers open, pages in use."""
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "browsers_open": len(self._slots),
                "pages_open": sum(s.live_pages() for s in self._slots),
            }

    def _confirm(self, question: str) -> bool:
        """Ask user to confirm an action. Returns True if confirmed."""
        if self.confirm_fn is None:
            return True
        return self.confirm_fn(question)

    def _slot_for(self, browser) -> _Slot:
        """The pool slot holding *browser* (adopting it if it isn't pooled yet)."""
        for slot in self._slots:
            if slot.browser is browser:
                return slot
        slot = _Slot(browser)
        self._slots.append(slot)
        return slot

    def _get_context(self, slot: _Slot):
        """Return the slot's shared context, creating it on first use (or
        after it was discarded because it failed)."""
        if slot.context is None:
            slot.context = slot.browser.new_context(user_agent=_USER_AGENT)
        return slot.context

    def _new_page(self):
        """Open a new browser page. Returns page or None on failure."""
        browser = self._get_browser()
        if not browser:
            return None
        slot = None
        try:
            slot = self._slot_for(browser)
            page = self._get_context(slot).new_page()
            slot.pages.append(page)
            return page
        except Exception as e:
            logger.warning("Failed to open page: %s", e)
            # The shared context may have been closed or crashed; forget it so
            # the next call builds a fresh one rather than failing forever.
            if slot is not None:
                slot.context = None
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

    def discover_form_fields(self, url: str) -> Optional[dict]:
        """Open *url* and list the fields a person could fill (backlog #102),
        so a form can be saved by voice without anyone speaking a CSS selector.

        Returns ``{"fields": [{"selector", "label", "key"}...], "submit_selector",
        "skipped_sensitive", "skipped_other"}``, or ``None`` when the page
        couldn't be loaded. Read-only: nothing is typed or clicked, so it needs
        no confirmation.
        """
        page = self._new_page()
        if not page:
            return None
        try:
            if not url.startswith("http"):
                url = "https://" + url
            page.goto(url, timeout=20000)
            page.wait_for_load_state("domcontentloaded", timeout=8000)
            raw = page.evaluate(_DISCOVER_JS)
            page.close()
            return form_fields_from_raw(raw)
        except Exception as e:
            logger.warning("discover_form_fields error: %s", e)
            self._snap_failure(page, f"discover-{url}")
            try:
                page.close()
            except Exception:
                pass
            return None

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


# ----------------------------------------------------------------------
# Form-field discovery helpers (backlog #102). Pure functions so the safety
# rules (what is offered, what is left out) are testable without a browser.
# ----------------------------------------------------------------------

def _field_key(label: str, name: str, ident: str) -> str:
    """A short spoken-friendly key (``full_name``) for a field."""
    for source in (label, name, ident):
        slug = re.sub(r"[^a-z0-9]+", "_", str(source or "").lower()).strip("_")
        if slug:
            return slug[:30]
    return "field"


def _field_selector(ident: str, name: str) -> Optional[str]:
    if ident and _SAFE_ID.match(ident):
        return f"#{ident}"
    if name and _SAFE_NAME.match(name):
        return f'[name="{name}"]'
    return None


def form_fields_from_raw(raw) -> dict:
    """Turn the page's raw control list into fields safe to offer for saving.

    Left out, and counted: password, card, ID and one-time-code fields
    (``skipped_sensitive``); and anything else ``page.fill`` can't type into or
    that has no usable selector (``skipped_other``: hidden, disabled, selects,
    checkboxes, files, buttons)."""
    raw = raw if isinstance(raw, dict) else {}
    fields: list[dict] = []
    seen_selectors: set[str] = set()
    used_keys: set[str] = set()
    sensitive = other = 0
    for item in raw.get("fields") or []:
        if not isinstance(item, dict):
            continue
        ident, name = str(item.get("id") or ""), str(item.get("name") or "")
        label = str(item.get("label") or "")
        ftype = str(item.get("type") or "").lower()
        if ftype == "password" or _SENSITIVE_FIELD.search(" ".join((label, name, ident))):
            sensitive += 1
            continue
        selector = _field_selector(ident, name)
        if (
            ftype not in _FILLABLE_TYPES
            or item.get("hidden") or item.get("disabled")
            or selector is None or selector in seen_selectors
        ):
            other += 1
            continue
        if len(fields) >= _MAX_DISCOVERED_FIELDS:
            other += 1
            continue
        key = base = _field_key(label, name, ident)
        n = 2
        while key in used_keys:
            key = f"{base}_{n}"
            n += 1
        used_keys.add(key)
        seen_selectors.add(selector)
        fields.append({"selector": selector, "label": label or name or ident, "key": key})

    submit = None
    for item in raw.get("submits") or []:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "")
        if ident and _SAFE_ID.match(ident):
            submit = f"#{ident}"
        elif item.get("tag") == "input":
            submit = "input[type=submit]"
        else:
            submit = "button[type=submit]" if item.get("type") == "submit" else "form button"
        break
    return {
        "fields": fields,
        "submit_selector": submit,
        "skipped_sensitive": sensitive,
        "skipped_other": other,
    }
