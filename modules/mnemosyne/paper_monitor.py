"""
modules/mnemosyne/paper_monitor.py

Research-paper monitoring (backlog #47): on a schedule, fetch new papers
matching saved interest queries, summarise them with the local model, and
queue them in Athena so they become searchable with the rest of your
documents.

Sources
-------
**arXiv** is implemented (public Atom API, no key). **IEEE Xplore is NOT**:
its API needs a registered key and its response format could not be
verified from here, so rather than ship an untested adapter the fetch step
is a plain ``fetch_fn(url) -> str`` and the parser is per-source — an IEEE
adapter would be a second ``parse_*`` function plus a URL builder, with
the interest/seen/queue machinery below unchanged.

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
        ))
    return papers


def paper_markdown(paper: Paper) -> str:
    """The document Athena will ingest for one paper."""
    authors = ", ".join(paper.authors[:8]) + (" et al." if len(paper.authors) > 8 else "")
    lines = [
        f"# {paper.title}", "",
        f"- arXiv: {paper.arxiv_id}",
        f"- Authors: {authors or 'unknown'}",
        f"- Published: {paper.published[:10] or 'unknown'}",
        f"- Link: {paper.link}",
    ]
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
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
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
                "SELECT arxiv_id, title, query, published, digest FROM papers_seen "
                "ORDER BY first_seen DESC LIMIT ?", (limit,)
            ).fetchall()
        ]

    def _is_seen(self, arxiv_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM papers_seen WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone() is not None

    # -- checking --------------------------------------------------------

    def check(self, now: Optional[datetime] = None) -> dict:
        """
        Fetch every interest, process unseen recent papers, return
        ``{"new": [Paper...], "errors": n, "interests": n}``.

        Failure isolation: a network error on one interest is logged and
        counted, and the others still run. A paper is marked seen only
        AFTER its file is written, so a crash mid-run retries it next time
        instead of silently dropping it.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.lookback_days)
        interests = self.list_interests()
        result: dict[str, Any] = {"new": [], "errors": 0, "interests": len(interests)}

        for i, interest in enumerate(interests):
            if i > 0:
                self.sleep_fn(_MIN_REQUEST_GAP_SECONDS)
            query = interest["query"]
            try:
                papers = parse_arxiv_feed(
                    self.fetch_fn(build_arxiv_url(query, max_results=self.max_new_per_query * 4))
                )
            except Exception:
                logger.exception("Paper check failed for interest %r; continuing.", query)
                result["errors"] += 1
                continue

            taken = 0
            for paper in papers:
                if taken >= self.max_new_per_query:
                    break
                if self._is_seen(paper.arxiv_id) or any(p.arxiv_id == paper.arxiv_id for p in result["new"]):
                    continue
                if not self._recent_enough(paper, cutoff):
                    continue
                paper.matched_query = query
                paper.digest = self._digest(paper)
                try:
                    self._write_paper(paper)
                except OSError:
                    logger.exception("Could not write paper file for %s; will retry.", paper.arxiv_id)
                    result["errors"] += 1
                    continue
                self._mark_seen(paper, now)
                result["new"].append(paper)
                taken += 1

            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE paper_interests SET last_checked = ? WHERE query = ?",
                    (now.isoformat(), query),
                )
        return result

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
                logger.exception("Paper summarisation failed for %s; using abstract.", paper.arxiv_id)
        return fallback_digest(paper.summary)

    def _write_paper(self, paper: Paper) -> Path:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        # arXiv ids are [\w./-]; old-style ids contain a slash (hep-th/9901001).
        safe = re.sub(r"[^\w.\-]", "_", paper.arxiv_id)
        path = self.docs_dir / f"arxiv_{safe}.md"
        path.write_text(paper_markdown(paper), encoding="utf-8")
        return path

    def _mark_seen(self, paper: Paper, now: datetime) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO papers_seen (arxiv_id, title, query, published, first_seen, digest) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (paper.arxiv_id, paper.title, paper.matched_query, paper.published,
                 now.isoformat(), paper.digest),
            )


def digest_markdown(papers: list[Paper], when: Optional[datetime] = None) -> str:
    """Body of the vault note summarising one check's new papers."""
    when = when or datetime.now(timezone.utc)
    lines = [f"{len(papers)} new paper(s) found on {when:%Y-%m-%d}.", ""]
    for p in papers:
        lines += [f"## {p.title}", f"- [arXiv:{p.arxiv_id}]({p.link}) — interest: {p.matched_query}"]
        if p.digest:
            lines.append(f"- {p.digest}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
