"""
modules/athena/pdf_structures.py

Tables and figure captions from PDFs, indexed as their own chunks
(backlog #58, #59).

Why separate chunks
-------------------
Plain-text extraction flattens a table into a stream of cell values
("ResNet 92.1 25M VGG 90.4 138M") that no question can be matched against,
and buries a figure caption mid-paragraph. Here:

* each **table** becomes chunks of *labelled rows* ("Model: ResNet;
  Accuracy: 92.1; Params: 25M"), so "what accuracy did ResNet get" matches
  a row that actually carries its column names. Large tables split by rows
  with the header repeated, so every chunk is self-describing.
* each **figure caption** ("Figure 3: Training loss over epochs") becomes
  its own small chunk with its page number, so "find the graph that shows
  training loss" can return the right figure and page.

Chunks carry ``content_type`` ("table" / "figure") in their metadata and use
non-overlapping chunk numbers (tables from 1000, figures from 2000), so
their ids never collide with the ordinary text chunks of the same page.

Limits, stated plainly
----------------------
* Detection uses ``pdfplumber``'s ruled-line table finder and a caption
  regex. Tables drawn without ruling lines, and captions not starting with
  "Figure/Fig./Table N" plus a separator, are missed rather than guessed at.
* Scanned (image-only) pages have no text layer, so nothing is found there.
* The figure *image* is not extracted — only its caption and page.
* Only the first ``max_pages`` pages are scanned, to bound ingest time.

Everything here fails soft: an unreadable PDF, a pdfplumber error or a
missing pdfplumber yields no extra chunks and never affects text ingestion.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

CONTENT_TEXT = "text"
CONTENT_TABLE = "table"
CONTENT_FIGURE = "figure"

TABLE_CHUNK_BASE = 1000      # chunk_number offsets: keep ids distinct from text chunks
FIGURE_CHUNK_BASE = 2000

_MAX_TABLE_CHUNK_CHARS = 1200
_MAX_CAPTION_CONTINUATION_LINES = 2
_MAX_CAPTION_CHARS = 500
_MIN_TABLE_ROWS = 2
_MIN_TABLE_COLS = 2

# "Figure 3: ...", "Fig. 2. ...", "Table 1 - ...", "Figure A1| ..." — the label
# must START the line and be followed by a separator, which is what stops a
# body sentence such as "Figure 3 shows ..." from being read as a caption.
_CAPTION_RE = re.compile(
    r"^\s*(?P<kind>Fig(?:ure)?s?|Table)\.?\s*(?P<num>[A-Z]?\d+(?:\.\d+)?[a-z]?)"
    r"\s*[.:|\-\u2013\u2014]\s*(?P<text>\S.*)$",
    re.IGNORECASE,
)
_SENTENCE_END = (".", "?", "!")


# ---------------------------------------------------------------------------
# Tables (pure)
# ---------------------------------------------------------------------------

def clean_cell(cell: Any) -> str:
    """One table cell as single-spaced text; None becomes empty."""
    return " ".join(str(cell).split()) if cell is not None else ""


def normalise_table(rows: Any) -> Optional[list[list[str]]]:
    """
    Clean a raw pdfplumber table. Drops empty rows and empty columns, pads
    ragged rows, and returns None for anything too small to be a real table
    (a one-column "table" is almost always a layout artefact).
    """
    if not rows:
        return None
    cleaned = [[clean_cell(c) for c in (row or [])] for row in rows]
    cleaned = [r for r in cleaned if any(r)]
    if len(cleaned) < _MIN_TABLE_ROWS:
        return None
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    keep = [i for i in range(width) if any(r[i] for r in cleaned)]
    cleaned = [[r[i] for i in keep] for r in cleaned]
    if len(keep) < _MIN_TABLE_COLS:
        return None
    return cleaned


def table_to_texts(
    rows: list[list[str]], page: int, label: Optional[str] = None,
    caption: Optional[str] = None, max_chars: int = _MAX_TABLE_CHUNK_CHARS,
) -> list[str]:
    """
    Render a normalised table as one or more chunk texts. The first row is
    the header; each following row becomes ``Header: value; Header: value``.
    Every chunk repeats the title and column line so it stands alone.
    """
    header = [h or f"Column {i + 1}" for i, h in enumerate(rows[0])]
    title = f"{label or 'Table'} (page {page})"
    if caption:
        title += f": {caption}"
    prefix = f"{title}\nColumns: {' | '.join(header)}\n"

    lines = []
    for r in rows[1:]:
        pairs = [f"{h}: {v}" for h, v in zip(header, r) if v]
        if pairs:
            lines.append("; ".join(pairs))
    if not lines:
        return []

    texts: list[str] = []
    current: list[str] = []
    size = len(prefix)
    for line in lines:
        if current and size + len(line) + 1 > max_chars:
            texts.append(prefix + "\n".join(current))
            current, size = [], len(prefix)
        current.append(line)
        size += len(line) + 1
    if current:
        texts.append(prefix + "\n".join(current))
    return texts


# ---------------------------------------------------------------------------
# Captions (pure)
# ---------------------------------------------------------------------------

def extract_page_captions(text: str) -> list[dict[str, str]]:
    """
    Captions found in one page's text, in reading order. Each is
    ``{"kind": "figure"|"table", "label": "Figure 3", "number": "3", "text": ...}``.

    A caption may wrap onto following lines. Heuristic: if the first line
    already ends a sentence it is taken alone; otherwise up to two further
    lines are joined, stopping early at a sentence end or at another caption.
    """
    lines = (text or "").splitlines()
    out: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        m = _CAPTION_RE.match(lines[i])
        if not m:
            i += 1
            continue
        body = m.group("text").strip()
        used = 0
        while (
            not body.endswith(_SENTENCE_END)
            and used < _MAX_CAPTION_CONTINUATION_LINES
            and i + 1 + used < len(lines)
        ):
            nxt = lines[i + 1 + used].strip()
            if not nxt or _CAPTION_RE.match(nxt):
                break
            body = f"{body} {nxt}"
            used += 1
        kind = "table" if m.group("kind").lower() == "table" else "figure"
        number = m.group("num")
        out.append({
            "kind": kind,
            "label": f"{'Table' if kind == 'table' else 'Figure'} {number}",
            "number": number,
            "text": body[:_MAX_CAPTION_CHARS],
        })
        i += 1 + used
    return out


def figure_text(label: str, page: int, caption: str) -> str:
    return f"{label} (page {page}): {caption}"


# ---------------------------------------------------------------------------
# PDF glue
# ---------------------------------------------------------------------------

def extract_structures(
    file_path: str,
    extract_tables: bool = True,
    index_figures: bool = True,
    max_pages: int = 300,
) -> list[dict[str, Any]]:
    """
    Table and figure chunk dicts for one PDF (same shape as the text chunks
    produced by PDFProcessor.process_pdf, plus ``content_type``). Returns
    [] on any problem.
    """
    if not (extract_tables or index_figures):
        return []
    try:
        import pdfplumber
    except ImportError:
        logger.debug("pdfplumber not installed; skipping table/figure extraction.")
        return []

    file_name = os.path.basename(file_path)
    chunks: list[dict[str, Any]] = []
    table_idx = figure_idx = 0

    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page_no, page in enumerate(pdf.pages[:max_pages], start=1):
                try:
                    captions = []
                    if index_figures or extract_tables:
                        captions = extract_page_captions(page.extract_text() or "")
                    table_captions = [c for c in captions if c["kind"] == "table"]

                    if extract_tables:
                        found = [t for t in (normalise_table(r) for r in (page.extract_tables() or [])) if t]
                        for n, rows in enumerate(found):
                            cap = table_captions[n] if n < len(table_captions) else None
                            texts = table_to_texts(
                                rows, page_no,
                                label=cap["label"] if cap else None,
                                caption=cap["text"] if cap else None,
                            )
                            for text in texts:
                                chunks.append(_chunk(text, file_path, file_name, page_no, total_pages,
                                                     TABLE_CHUNK_BASE + table_idx, CONTENT_TABLE))
                                table_idx += 1

                    if index_figures:
                        for cap in (c for c in captions if c["kind"] == "figure"):
                            chunks.append(_chunk(
                                figure_text(cap["label"], page_no, cap["text"]),
                                file_path, file_name, page_no, total_pages,
                                FIGURE_CHUNK_BASE + figure_idx, CONTENT_FIGURE,
                            ))
                            figure_idx += 1
                except Exception:
                    logger.debug("Structure extraction failed on page %d of %s.", page_no, file_name,
                                 exc_info=True)
    except Exception:
        logger.warning("Could not scan %s for tables/figures; text ingestion is unaffected.", file_name,
                       exc_info=True)
        return []

    for c in chunks:
        c["total_chunks"] = len(chunks)
    return chunks


def _chunk(text: str, file_path: str, file_name: str, page: int, total_pages: int,
           chunk_number: int, content_type: str) -> dict[str, Any]:
    return {
        "text": text,
        "file_name": file_name,
        "file_path": file_path,
        "page_number": page,
        "chunk_number": chunk_number,
        "total_pages": total_pages,
        "content_type": content_type,
    }


# ---------------------------------------------------------------------------
# Query-side helpers
# ---------------------------------------------------------------------------

_STRUCT_QUERY_RE = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+)?"
    r"(?:find|show|locate|get|which|where(?:'s|\s+is|\s+are)?|what)\b.*?"
    r"\b(?P<noun>graph|chart|figure|plot|diagram|table)s?\b",
    re.IGNORECASE,
)


def infer_content_type(query: str) -> Optional[str]:
    """
    "find the graph that shows X" -> "figure"; "which table lists Y" -> "table";
    anything else -> None. Deliberately narrow: it only fires when the query
    OPENS as a find/show/which request naming a figure or table, so an
    ordinary question that merely mentions a graph ("explain graph theory")
    is never hijacked into a caption search.
    """
    m = _STRUCT_QUERY_RE.match(query or "")
    if not m:
        return None
    return CONTENT_TABLE if m.group("noun").lower() == "table" else CONTENT_FIGURE


_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_QUERY_FILLER = frozenset(
    "the that this which what where find show locate get graph chart figure plot diagram "
    "table tables figures graphs charts plots with and for from about shows showing "
    "please can you could".split()
)


def lexical_overlap(query: str, text: str) -> float:
    """Fraction of the query's content words that appear in *text* (0..1)."""
    terms = {w for w in _WORD_RE.findall((query or "").lower()) if w not in _QUERY_FILLER}
    if not terms:
        return 0.0
    words = set(_WORD_RE.findall((text or "").lower()))
    return len(terms & words) / len(terms)