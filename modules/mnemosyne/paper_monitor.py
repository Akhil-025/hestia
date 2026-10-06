"""
modules/mnemosyne/paper_monitor.py

Research-paper monitoring (backlog #47): on a schedule, fetch new papers
matching saved interest queries, summarise them with the local model, and
queue them in Athena so they become searchable with the rest of your
documents.

Sources
-------
Each source is a small adapter (``PaperSource``: ``name``, ``supports``,
``build_url``, ``parse``, ``recent``, ``min_gap``, optional ``daily_limit``).

**arXiv** (``ArxivSource``): public Atom API, no key.
**IEEE Xplore** (``ieee_source.IeeeSource``): needs your own API key, written
from the documented response shape and NOT yet verified against the live
API; ``scripts/check_ieee.py`` is the one-time check. See that module.

The interest / seen / queue machinery is shared. A paper found by two sources
(an arXiv preprint and its IEEE version) is queued once: matched by DOI, or by
an exact normalised title of 20+ characters. A source with a ``daily_limit``
has its calls counted in the database and stops for the day at the limit.

How a paper is "queued in Athena"
---------------------------------
Athena ingests a documents folder and already skips unchanged files
(backlog #61), so the monitor writes one small markdown file per paper
into ``<athena documents>/arxiv/``. The next Athena ingest embeds and
indexes it like any other document — no private Athena API is touched.

Politeness: arXiv asks clients to make at most one request every three
seconds; ``check`` sleeps between interests (injectable, so tests don't).
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
ARXIV_API = "https://export.arxiv.org/api/query"
_MIN_REQUEST_GAP_SECONDS = 3.0
_FIELD_PREFIXES = ("all:", "ti:", "abs:", "au:", "cat:", "co:", "jr:", "rn:")


@dataclass
class Paper:
    arxiv_id: str
    title: str
    summary: str                       # the abstract
    authors: list[str] = field(default_factory=list)
    published: str = ""                # ISO date of first version
    updated: str = ""
    link: str = ""
    pdf_link: str = ""
    categories: list[str] = field(default_factory=list)
    matched_query: str = ""
    digest: str = ""                   # model summary (filled by the monitor)
    source: str = "arxiv"              # which PaperSource produced it
    doi: str = ""
    external_id: str = ""              # non-arXiv sources: their own id (IEEE article number)
    venue: str = ""
    date_precision: str = "day"        # "day" | "month" | "year" (IEEE often gives only a month)

    @property
    def paper_id(self) -> str:
        """Key used for de-duplication: the bare arXiv id (unchanged, so existing
        rows stay valid), or ``<source>:<id>`` for other sources."""
        if self.source == "arxiv":
            return self.arxiv_id
        return f"{self.source}:{self.external_id}"


# ---------------------------------------------------------------------------
# arXiv (pure)
# ---------------------------------------------------------------------------

def build_arxiv_url(query: str, max_results: int = 10) -> str:
    """
    API URL for the newest papers matching *query*. Plain text is searched
    across all fields as a phrase; a query already using arXiv field syntax
    (``cat:cs.LG``, ``au:Hinton``) is passed through untouched.
    """
    q = (query or "").strip()
    if not q.lower().startswith(_FIELD_PREFIXES):
        q = f'all:"{q}"' if " " in q else f"all:{q}"
    return (
        f"{ARXIV_API}?search_query={quote_plus(q)}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&start=0&max_results={max(1, min(int(max_results), 50))}"
    )


def _clean(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def normalise_arxiv_id(raw: str) -> str:
    """``http://arxiv.org/abs/2401.01234v2`` -> ``2401.01234`` (old-style ids keep their category)."""
    tail = raw.rsplit("/abs/", 1)[-1] if "/abs/" in raw else raw.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", tail.strip())


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    """
    Parse an arXiv Atom response. A malformed document yields ``[]`` and an
    entry missing an id or title is skipped — one bad entry never discards
    the rest of the feed.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("parse_arxiv_feed: response was not valid XML.")
        return []

    papers: list[Paper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = _clean(entry.findtext(f"{_ATOM}id"))
        title = _clean(entry.findtext(f"{_ATOM}title"))
        # An API error comes back as a single entry titled "Error".
        if not raw_id or not title or title.lower() == "error":
            continue
        link = pdf = ""
        for ln in entry.findall(f"{_ATOM}link"):
            if ln.get("title") == "pdf" or ln.get("type") == "application/pdf":
                pdf = ln.get("href", "")
            elif ln.get("rel") == "alternate":
                link = ln.get("href", "")
        papers.append(Paper(
            arxiv_id=normalise_arxiv_id(raw_id),
            title=title,
            summary=_clean(entry.findtext(f"{_ATOM}summary")),
            authors=[_clean(a.findtext(f"{_ATOM}name")) for a in entry.findall(f"{_ATOM}author")],
            published=_clean(entry.findtext(f"{_ATOM}published")),
            updated=_clean(entry.findtext(f"{_ATOM}updated")),
            link=link or raw_id,
            pdf_link=pdf,
            categories=[c.get("term", "") for c in entry.findall(f"{_ATOM}category") if c.get("term")],
            doi=_clean(entry.findtext(f"{_ARXIV_NS}doi")).lower(),
        ))
    return papers


def paper_markdown(paper: Paper) -> str:
    """The document Athena will ingest for one paper."""
    authors = ", ".join(paper.authors[:8]) + (" et al." if len(paper.authors) > 8 else "")
    id_line = (f"- arXiv: {paper.arxiv_id}" if paper.source == "arxiv"
               else f"- {paper.source.upper()}: {paper.external_id}")
    lines = [
        f"# {paper.title}", "",
        id_line,
        f"- Authors: {authors or 'unknown'}",
        f"- Published: {paper.published[:10] or 'unknown'}",
        f"- Link: {paper.link}",
    ]
    if paper.doi:
        lines.append(f"- DOI: {paper.doi}")
    if paper.venue:
        lines.append(f"- Venue: {paper.venue}")
    if paper.categories:
        lines.append(f"- Categories: {', '.join(paper.categories)}")
    if paper.matched_query:
        lines.append(f"- Matched interest: {paper.matched_query}")
    if paper.digest:
        lines += ["", "## Summary", paper.digest]
    lines += ["", "## Abstract", paper.summary, ""]
    return "\n".join(lines)


def fallback_digest(abstract: str, sentences: int = 2) -> str:
    """No model available: the first sentences of the abstract are a fair digest."""
    parts = re.split(r"(?<=[.!?])\s+", _clean(abstract))
    return " ".join(parts[:sentences]).strip()


# ---------------------------------------------------------------------------
# Store + monitor
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_interests (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at TEXT,
    last_checked TEXT
);
CREATE TABLE IF NOT EXISTS papers_seen (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT,
    query TEXT,
    published TEXT,
    first_seen TEXT,
    digest TEXT
);
"""

# Columns added after the first release; added to an existing database on open.
_MIGRATIONS = (
    ("papers_seen", "source", "TEXT DEFAULT 'arxiv'"),
    ("papers_seen", "doi", "TEXT DEFAULT ''"),
    ("papers_seen", "norm_title", "TEXT DEFAULT ''"),
)

_USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_usage (
    source TEXT NOT NULL,
    day TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source, day)
);
"""

_MIN_TITLE_CHARS = 20


def normalise_title(title: str) -> str:
    """Lower-case letters and digits only, for cross-source title matching."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


class PaperSource:
    """Interface every source implements (duck-typed; subclassing is optional)."""

    name: str = ""
    min_gap: float = 0.0              # seconds between two calls in one check
    daily_limit: Optional[int] = None  # None = not counted

    def supports(self, query: str) -> bool:
        return True

    def build_url(self, query: str, max_results: int) -> str:
        raise NotImplementedError

    def parse(self, text: str) -> list["Paper"]:
        raise NotImplementedError

    def recent(self, paper: "Paper", cutoff: datetime) -> bool:
        return True

    def redact(self, text: str) -> str:
        """Remove secrets (an API key) from text before it is logged."""
        return text


class ArxivSource(PaperSource):
    name = "arxiv"
    min_gap = _MIN_REQUEST_GAP_SECONDS

    def build_url(self, query: str, max_results: int) -> str:
        return build_arxiv_url(query, max_results=max_results)

    def parse(self, text: str) -> list["Paper"]:
        return parse_arxiv_feed(text)

    def recent(self, paper: "Paper", cutoff: datetime) -> bool:
        return PaperMonitor._recent_enough(paper, cutoff)


_PROMPT = (
    "Summarise this research paper abstract in exactly 2 plain sentences for "
    "a busy reader: what was done and what was found. No preamble.\n\n"
    "Title: {title}\n\nAbstract: {abstract}"
)


def _default_fetch(url: str) -> str:
    import requests

    resp = requests.get(
        url, timeout=20, headers={"User-Agent": "Hestia-paper-monitor/1.0 (personal assistant)"}
    )
    resp.raise_for_status()
    return resp.text


class PaperMonitor:
    def __init__(
        self,
        db_path: str,
        docs_dir: str,
        llm: Any = None,
        fetch_fn: Optional[Callable[[str], str]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        max_new_per_query: int = 5,
        lookback_days: int = 14,
        sources: Optional[list] = None,
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.sources: list = list(sources) if sources else [ArxivSource()]
        self.docs_dir = Path(docs_dir)
        self.llm = llm
        self.fetch_fn = fetch_fn or _default_fetch
        self.sleep_fn = sleep_fn
        self.max_new_per_query = max(1, max_new_per_query)
        self.lookback_days = lookback_days
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.executescript(_USAGE_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the first release, and backfill the
        title key for rows written before it existed. Idempotent."""
        with self._lock, self._conn:
            for table, column, decl in _MIGRATIONS:
                cols = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                if column not in cols:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            rows = self._conn.execute(
                "SELECT arxiv_id, title FROM papers_seen WHERE norm_title IS NULL OR norm_title = ''"
            ).fetchall()
            for r in rows:
                self._conn.execute(
                    "UPDATE papers_seen SET norm_title = ? WHERE arxiv_id = ?",
                    (normalise_title(r["title"]), r["arxiv_id"]))

    # -- per-source daily call counter ---------------------------------

    @staticmethod
    def _day(now: datetime) -> str:
        return now.astimezone(timezone.utc).strftime("%Y-%m-%d")

    def usage_today(self, source: str, now: Optional[datetime] = None) -> int:
        row = self._conn.execute(
            "SELECT calls FROM api_usage WHERE source = ? AND day = ?",
            (source, self._day(now or datetime.now(timezone.utc)))).fetchone()
        return int(row["calls"]) if row else 0

    def _bump_usage(self, source: str, now: datetime, to: Optional[int] = None) -> None:
        day = self._day(now)
        with self._lock, self._conn:
            if to is None:
                self._conn.execute(
                    "INSERT INTO api_usage (source, day, calls) VALUES (?, ?, 1) "
                    "ON CONFLICT(source, day) DO UPDATE SET calls = calls + 1", (source, day))
            else:
                self._conn.execute(
                    "INSERT INTO api_usage (source, day, calls) VALUES (?, ?, ?) "
                    "ON CONFLICT(source, day) DO UPDATE SET calls = MAX(calls, excluded.calls)",
                    (source, day, to))

    # -- interests -------------------------------------------------------

    def add_interest(self, query: str) -> bool:
        """Returns False if the interest already exists."""
        query = " ".join((query or "").split())
        if not query:
            raise ValueError("add_interest() requires a non-empty query.")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO paper_interests (query, created_at) VALUES (?, ?)",
                (query, datetime.now(timezone.utc).isoformat()),
            )
            return cur.rowcount > 0

    def remove_interest(self, query: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM paper_interests WHERE query = ?", (" ".join((query or "").split()),)
            )
            return cur.rowcount > 0

    def list_interests(self) -> list[dict]:
        return [
            dict(r) for r in self._conn.execute(
                "SELECT query, created_at, last_checked FROM paper_interests ORDER BY id"
            ).fetchall()
        ]

    def seen_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM papers_seen").fetchone()["n"]

    def recent_papers(self, limit: int = 10) -> list[dict]:
        return [
            dict(r) for r in self._conn.execute(
                "SELECT arxiv_id, title, query, published, digest, source FROM papers_seen "
                "ORDER BY first_seen DESC LIMIT ?", (limit,)
            ).fetchall()
        ]

    def _is_seen(self, arxiv_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM papers_seen WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone() is not None

    def _is_cross_source_duplicate(self, paper: Paper, pending: list) -> bool:
        """True if the same work is already stored or queued from another source:
        same DOI, or the same normalised title (only when it is long enough that
        a collision between different papers is implausible)."""
        key = normalise_title(paper.title)
        long_title = len(key) >= _MIN_TITLE_CHARS
        for q in pending:
            if q.paper_id == paper.paper_id:
                continue
            if paper.doi and q.doi and paper.doi == q.doi:
                return True
            if long_title and normalise_title(q.title) == key:
                return True
        row = self._conn.execute(
            "SELECT 1 FROM papers_seen WHERE arxiv_id != ? AND "
            "((? != '' AND doi = ?) OR (? AND norm_title = ?)) LIMIT 1",
            (paper.paper_id, paper.doi, paper.doi, 1 if long_title else 0, key)).fetchone()
        return row is not None

    # -- checking --------------------------------------------------------

    def check(self, now: Optional[datetime] = None) -> dict:
        """
        Fetch every interest from every source, process unseen recent papers,
        return ``{"new": [Paper...], "errors": n, "interests": n,
        "by_source": {name: n_new}, "quota_skipped": [names]}``.

        Failure isolation: a failure for one (interest, source) is logged and
        counted, and the rest still run. A paper is marked seen only AFTER its
        file is written, so a crash mid-run retries it next time instead of
        silently dropping it. A source with a daily limit stops for the day
        when the limit is reached, or when the service reports it was exceeded.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.lookback_days)
        interests = self.list_interests()
        result: dict[str, Any] = {
            "new": [], "errors": 0, "interests": len(interests),
            "by_source": {s.name: 0 for s in self.sources}, "quota_skipped": [],
        }
        calls = {s.name: 0 for s in self.sources}
        halted: set[str] = set()

        for interest in interests:
            query = interest["query"]
            for src in self.sources:
                if src.name in halted or not src.supports(query):
                    continue
                if src.daily_limit is not None and self.usage_today(src.name, now) >= src.daily_limit:
                    logger.info("%s: daily call limit (%d) reached; skipping until tomorrow.",
                                src.name, src.daily_limit)
                    if src.name not in result["quota_skipped"]:
                        result["quota_skipped"].append(src.name)
                    halted.add(src.name)
                    continue
                if calls[src.name] > 0:
                    self.sleep_fn(src.min_gap)
                calls[src.name] += 1
                if src.daily_limit is not None:
                    self._bump_usage(src.name, now)      # an attempt counts, successful or not
                try:
                    papers = src.parse(
                        self.fetch_fn(src.build_url(query, self.max_new_per_query * 4)))
                except Exception as exc:
                    result["errors"] += 1
                    self._on_fetch_error(src, query, exc, halted, now, result)
                    continue

                taken = 0
                for paper in papers:
                    if taken >= self.max_new_per_query:
                        break
                    pid = paper.paper_id
                    if self._is_seen(pid) or any(p.paper_id == pid for p in result["new"]):
                        continue
                    if self._is_cross_source_duplicate(paper, result["new"]):
                        continue
                    if not src.recent(paper, cutoff):
                        continue
                    paper.matched_query = query
                    paper.digest = self._digest(paper)
                    try:
                        self._write_paper(paper)
                    except OSError:
                        logger.exception("Could not write paper file for %s; will retry.", pid)
                        result["errors"] += 1
                        continue
                    self._mark_seen(paper, now)
                    result["new"].append(paper)
                    result["by_source"][src.name] = result["by_source"].get(src.name, 0) + 1
                    taken += 1

            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE paper_interests SET last_checked = ? WHERE query = ?",
                    (now.isoformat(), query),
                )
        return result

    def _on_fetch_error(self, src, query: str, exc: Exception, halted: set,
                        now: datetime, result: dict) -> None:
        """Log a failed fetch (secrets removed) and react to quota/auth errors."""
        status = getattr(getattr(exc, "response", None), "status_code", None)
        msg = src.redact(f"{type(exc).__name__}: {exc}")
        if status in (403, 429) and src.daily_limit is not None:
            # The service says we're over its limit: stop for the day and remember it.
            logger.warning("%s refused the request (HTTP %s): treating the daily quota as used. %s",
                           src.name, status, msg)
            self._bump_usage(src.name, now, to=src.daily_limit)
            halted.add(src.name)
            if src.name not in result["quota_skipped"]:
                result["quota_skipped"].append(src.name)
        elif status == 401:
            logger.warning("%s rejected the API key (HTTP 401); skipping it for this check. %s",
                           src.name, msg)
            halted.add(src.name)
        else:
            logger.warning("Paper check failed for interest %r on %s; continuing. %s",
                           query, src.name, msg)

    @staticmethod
    def _recent_enough(paper: Paper, cutoff: datetime) -> bool:
        if not paper.published:
            return True          # unknown date: don't discard on a guess
        try:
            dt = datetime.fromisoformat(paper.published.replace("Z", "+00:00"))
        except ValueError:
            return True
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # A source that only knows the month (or year) must not lose a paper
        # published this month just because the 1st is older than the cutoff.
        if paper.date_precision == "month":
            return (dt.year, dt.month) >= (cutoff.year, cutoff.month)
        if paper.date_precision == "year":
            return dt.year >= cutoff.year
        return dt >= cutoff

    def _digest(self, paper: Paper) -> str:
        if self.llm is not None:
            try:
                text = self.llm.generate(
                    _PROMPT.format(title=paper.title, abstract=paper.summary[:3000])
                ).strip()
                if text:
                    return text
            except Exception:
                logger.exception("Paper summarisation failed for %s; using abstract.", paper.paper_id)
        return fallback_digest(paper.summary)

    def _write_paper(self, paper: Paper) -> Path:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        # arXiv ids are [\w./-]; old-style ids contain a slash (hep-th/9901001).
        # Other sources: "<source>_<their id>.md" (ieee_1234567.md).
        ident = paper.arxiv_id if paper.source == "arxiv" else paper.external_id
        safe = re.sub(r"[^\w.\-]", "_", ident)
        path = self.docs_dir / f"{paper.source}_{safe}.md"
        path.write_text(paper_markdown(paper), encoding="utf-8")
        return path

    def _mark_seen(self, paper: Paper, now: datetime) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO papers_seen "
                "(arxiv_id, title, query, published, first_seen, digest, source, doi, norm_title) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (paper.paper_id, paper.title, paper.matched_query, paper.published,
                 now.isoformat(), paper.digest, paper.source, paper.doi,
                 normalise_title(paper.title)),
            )


def digest_markdown(papers: list[Paper], when: Optional[datetime] = None) -> str:
    """Body of the vault note summarising one check's new papers."""
    when = when or datetime.now(timezone.utc)
    lines = [f"{len(papers)} new paper(s) found on {when:%Y-%m-%d}.", ""]
    for p in papers:
        label = f"arXiv:{p.arxiv_id}" if p.source == "arxiv" else f"{p.source.upper()}:{p.external_id}"
        lines += [f"## {p.title}", f"- [{label}]({p.link}) — interest: {p.matched_query}"]
        if p.digest:
            lines.append(f"- {p.digest}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
