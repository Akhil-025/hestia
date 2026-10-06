"""
modules/mnemosyne/ieee_source.py

IEEE Xplore as a paper source for ``PaperMonitor`` (backlog #47, the IEEE half).

STATUS: written from IEEE's documented "Metadata Search" response shape and
tested against hand-written fixtures. It has NOT been run against the live API
(the build environment has no network and no key). Run ``python
scripts/check_ieee.py`` once with your key; it reports which of the fields this
parser relies on are really present and can save the raw response as a fixture.

The key
-------
Read from the ``IEEE_API_KEY`` environment variable (``.env`` works), never from
the YAML config. The API takes it as a URL parameter, so it ends up in request
URLs and in the text of ``requests`` exceptions; ``IeeeSource.redact`` strips
it, and the monitor logs only redacted text.

Limits
------
The key you registered allows 10 calls/second and 200 calls/day. ``min_gap``
keeps well under the first. ``daily_limit`` (default 100, hard-capped at 200) is
a self-imposed ceiling that the monitor counts in its database and stops at, so
a bug or a long interest list cannot spend the allowance. HTTP 403/429 is treated
as "quota used" for the rest of the day. Each check costs one call per saved
topic.

Dates
-----
Xplore usually gives a month ("March 2024", "Mar.-Apr. 2024") or only a year,
not a day. ``Paper.date_precision`` records that and the recency filter
compares months/years accordingly, so a paper from this month is not dropped for
being "older than 14 days". "Seen before" is what actually prevents repeats.

Interests
---------
Free text goes to ``querytext``. ``ti:`` / ``abs:`` / ``au:`` map to
``article_title`` / ``abstract`` / ``author``; ``all:`` is stripped. ``cat:``,
``co:``, ``jr:``, ``rn:`` are arXiv-only and are skipped for IEEE.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from .paper_monitor import Paper, PaperSource, _clean

logger = logging.getLogger(__name__)

IEEE_API = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
MAX_DAILY_LIMIT = 200          # the registered key's allowance; our ceiling can never exceed it
DEFAULT_DAILY_LIMIT = 100
MIN_GAP_SECONDS = 0.25         # well under 10 calls/second
MAX_RECORDS = 25

_UNSUPPORTED_PREFIXES = ("cat:", "co:", "jr:", "rn:")
_FIELD_PARAMS = {"ti:": "article_title", "abs:": "abstract", "au:": "author"}

_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december"]


def _month_of(word: str) -> int:
    """1-12 if *word* is a month name or abbreviation ("Mar", "Sept", "March"), else 0.
    A prefix of at least three letters of the full name; "Marketing" is not March."""
    w = word.lower()
    if len(w) < 3:
        return 0
    for i, name in enumerate(_MONTH_NAMES, 1):
        if name.startswith(w) or (w == "sept" and i == 9):
            return i
    return 0


def parse_ieee_date(text: str, year: Any = None) -> tuple[str, str]:
    """``(iso_date, precision)`` from an Xplore date string.

    "12 March 2024" -> ("2024-03-12", "day"); "March 2024" and the range
    "Mar.-Apr. 2024" (the LAST month wins) -> ("2024-04-01", "month"); only a
    year -> ("<year>-12-31", "year"); nothing usable -> ("", "day").
    """
    text = _clean(str(text or ""))
    ym = re.search(r"\b(19|20)\d{2}\b", text)
    yr = int(ym.group(0)) if ym else None
    if yr is None:
        try:
            yr = int(year)
        except (TypeError, ValueError):
            return "", "day"
    months = [m for m in (_month_of(w) for w in re.findall(r"[A-Za-z]+", text)) if m]
    if not months:
        return f"{yr:04d}-12-31", "year"
    month = months[-1]
    dm = re.search(r"\b(\d{1,2})\b(?=\s+[A-Za-z])", text)
    if dm and len(months) == 1 and 1 <= int(dm.group(1)) <= 31:
        try:
            return datetime(yr, month, int(dm.group(1))).date().isoformat(), "day"
        except ValueError:
            pass
    return f"{yr:04d}-{month:02d}-01", "month"


def _authors(raw: Any) -> list[str]:
    """Authors arrive as {"authors": [{"full_name": ...}]}; accept a bare list too."""
    items = raw.get("authors") if isinstance(raw, dict) else raw
    out: list[str] = []
    for a in items or []:
        name = a.get("full_name") if isinstance(a, dict) else a
        name = _clean(str(name or ""))
        if name:
            out.append(name)
    return out


def _terms(article: dict) -> list[str]:
    out: list[str] = []
    idx = article.get("index_terms")
    if isinstance(idx, dict):
        for group in ("ieee_terms", "author_terms"):
            terms = (idx.get(group) or {}).get("terms") if isinstance(idx.get(group), dict) else None
            for t in terms or []:
                t = _clean(str(t))
                if t and t not in out:
                    out.append(t)
    return out[:8]


def parse_ieee_response(text: str) -> list[Paper]:
    """
    Parse a Metadata Search JSON response. Malformed JSON, an error payload or
    an unexpected shape yields ``[]``; an article without a number or title is
    skipped. One bad article never discards the rest.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        logger.warning("parse_ieee_response: response was not valid JSON.")
        return []
    if not isinstance(data, dict):
        return []
    articles = data.get("articles")
    if not isinstance(articles, list):
        if data.get("error") or data.get("message"):
            logger.warning("IEEE Xplore returned an error payload: %s",
                           _clean(str(data.get("error") or data.get("message")))[:200])
        return []

    papers: list[Paper] = []
    for art in articles:
        if not isinstance(art, dict):
            continue
        number = _clean(str(art.get("article_number") or ""))
        title = _clean(str(art.get("title") or ""))
        if not number or not title:
            continue
        published, precision = parse_ieee_date(art.get("publication_date"), art.get("publication_year"))
        link = _clean(str(art.get("html_url") or art.get("abstract_url") or ""))
        papers.append(Paper(
            arxiv_id="",
            title=title,
            summary=_clean(str(art.get("abstract") or "")),
            authors=_authors(art.get("authors")),
            published=published,
            updated="",
            link=link or f"https://ieeexplore.ieee.org/document/{number}",
            pdf_link=_clean(str(art.get("pdf_url") or "")),
            categories=_terms(art),
            source="ieee",
            doi=_clean(str(art.get("doi") or "")).lower(),
            external_id=number,
            venue=_clean(str(art.get("publication_title") or "")),
            date_precision=precision,
        ))
    return papers


class IeeeSource(PaperSource):
    name = "ieee"
    min_gap = MIN_GAP_SECONDS

    def __init__(self, api_key: str, daily_limit: int = DEFAULT_DAILY_LIMIT) -> None:
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("IeeeSource needs an API key (set IEEE_API_KEY).")
        self._key = api_key
        self.daily_limit = max(1, min(int(daily_limit), MAX_DAILY_LIMIT))

    def supports(self, query: str) -> bool:
        return not (query or "").strip().lower().startswith(_UNSUPPORTED_PREFIXES)

    def build_url(self, query: str, max_results: int) -> str:
        q = " ".join((query or "").split())
        param = "querytext"
        low = q.lower()
        if low.startswith("all:"):
            q = q[4:].strip()
        else:
            for prefix, name in _FIELD_PARAMS.items():
                if low.startswith(prefix):
                    param, q = name, q[len(prefix):].strip()
                    break
        params = {
            "apikey": self._key,
            "format": "json",
            param: q,
            "max_records": max(1, min(int(max_results), MAX_RECORDS)),
            "start_record": 1,
            "sort_field": "publication_year",
            "sort_order": "desc",
        }
        return f"{IEEE_API}?{urlencode(params)}"

    def parse(self, text: str) -> list[Paper]:
        return parse_ieee_response(text)

    def recent(self, paper: Paper, cutoff: datetime) -> bool:
        from .paper_monitor import PaperMonitor
        return PaperMonitor._recent_enough(paper, cutoff)

    def redact(self, text: str) -> str:
        return str(text).replace(self._key, "***")

    def __repr__(self) -> str:                    # never show the key
        return f"IeeeSource(daily_limit={self.daily_limit})"
