# modules/hephaestus/scrapers.py
"""
Pluggable site-specific scrapers for Hephaestus (backlog #105).

Generic page scraping returns the first 500 characters of a page's visible
text, which is useless for a results table or a job board. A scraper knows one
site (or kind of site) and returns just the part that matters.

Two ways to add one, neither needing a change to engine.py:

1. Config only — a CSS selector per site, in ``hephaestus.scrapers``::

       hephaestus:
         scrapers:
           - name: gate-results
             url_contains: "gate.example.org/results"   # or url_regex: "..."
             selector: "table.results tr"
             max_items: 20

2. A Python file in ``hephaestus.scraper_dir`` that defines ``SCRAPERS``
   (a list of ``SiteScraper`` instances) or a ``get_scrapers()`` function.
   That directory is executed as code, exactly like ``skills/`` — only point
   it at a folder you control.

A scraper returns ``None`` when the page couldn't be read, and the engine then
falls back to generic scraping rather than failing outright.
"""
from __future__ import annotations

import importlib.util
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SiteScraper:
    """Base class. Subclass and override ``matches`` and ``scrape``."""

    name: str = "scraper"

    def matches(self, url: str) -> bool:  # pragma: no cover - interface
        return False

    def scrape(self, url: str, browser: Any) -> Optional[str]:  # pragma: no cover - interface
        return None


class SelectorScraper(SiteScraper):
    """Config-driven scraper: every element matching *selector*, one per line."""

    def __init__(
        self, name: str, selector: str, *, url_contains: str = "",
        url_regex: str = "", max_items: int = 20,
    ) -> None:
        if not name or not selector:
            raise ValueError("a scraper needs a name and a selector")
        if not url_contains and not url_regex:
            raise ValueError(f"scraper {name!r} needs url_contains or url_regex")
        self.name = name
        self.selector = selector
        self.url_contains = url_contains.lower()
        self._regex = re.compile(url_regex, re.I) if url_regex else None
        self.max_items = max(1, min(int(max_items), 200))

    def matches(self, url: str) -> bool:
        low = (url or "").lower()
        if self.url_contains and self.url_contains in low:
            return True
        return bool(self._regex and self._regex.search(url or ""))

    def scrape(self, url: str, browser: Any) -> Optional[str]:
        fetch = getattr(browser, "fetch_elements", None)
        if not callable(fetch):
            return None
        items = fetch(url, self.selector, max_items=self.max_items)
        if items is None:
            return None
        if not items:
            logger.warning("Scraper %r matched no elements at %s; the layout may have changed.",
                           self.name, url)
            return None
        return "\n".join(items)


class ScraperRegistry:
    """Ordered collection; the first scraper whose ``matches`` is true wins."""

    def __init__(self) -> None:
        self._scrapers: list[SiteScraper] = []

    def register(self, scraper: SiteScraper) -> None:
        if any(s.name == scraper.name for s in self._scrapers):
            raise ValueError(f"duplicate scraper name {scraper.name!r}")
        self._scrapers.append(scraper)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._scrapers]

    def get(self, name: str) -> Optional[SiteScraper]:
        return next((s for s in self._scrapers if s.name == name), None)

    def find(self, url: str) -> Optional[SiteScraper]:
        for s in self._scrapers:
            try:
                if s.matches(url):
                    return s
            except Exception:
                logger.exception("Scraper %r.matches() raised; skipping it.", s.name)
        return None

    def load_config(self, entries: Any) -> None:
        """Register selector scrapers from the ``hephaestus.scrapers`` list.
        A bad entry is logged and skipped so it can't stop Hestia booting."""
        for entry in entries or []:
            try:
                self.register(SelectorScraper(
                    str(entry.get("name", "")).strip(), str(entry.get("selector", "")).strip(),
                    url_contains=str(entry.get("url_contains", "") or ""),
                    url_regex=str(entry.get("url_regex", "") or ""),
                    max_items=entry.get("max_items", 20),
                ))
            except Exception as exc:
                logger.warning("Ignoring scraper config %r: %s", entry, exc)

    def load_plugins(self, directory: Any) -> int:
        """Import every ``*.py`` in *directory* and register what it exports.
        Returns how many scrapers were added. Failures are logged and skipped."""
        if not directory:
            return 0
        path = Path(str(directory))
        if not path.is_dir():
            logger.warning("hephaestus.scraper_dir %s is not a directory.", path)
            return 0
        added = 0
        for file in sorted(path.glob("*.py")):
            if file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"hephaestus_scraper_{file.stem}", file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                found = list(getattr(module, "SCRAPERS", []) or [])
                factory = getattr(module, "get_scrapers", None)
                if callable(factory):
                    found.extend(factory() or [])
                for s in found:
                    if isinstance(s, SiteScraper):
                        self.register(s)
                        added += 1
            except Exception as exc:
                logger.warning("Could not load scraper plugin %s: %s", file.name, exc)
        return added
