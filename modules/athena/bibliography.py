"""
modules/athena/bibliography.py

Reading a paper's reference list and working out which of YOUR other
documents each reference points at (backlog #60, cross-document citation
graphs).

Why this exists
---------------
Athena's chunk index cannot answer "which paper cites which". Ingestion runs
``PDFProcessor.clean_text`` over every page, which deletes URLs (so every
``https://doi.org/...`` is gone), flattens line breaks (so reference entries
run together) and the sentence-based chunker then drops short fragments. The
reference list is the one part of a paper where all of that matters. So this
module works on RAW text read from the source file, and the index is only used
to learn which files exist.

What is here (all pure Python, no I/O, no third-party imports)
--------------------------------------------------------------
* ``find_reference_section`` / ``split_references`` / ``parse_reference``:
  raw text -> a list of :class:`Reference`.
* ``identity_from_*``: a document's own title / arXiv id / DOI / year.
* ``match_references``: references x library documents -> citation links,
  each carrying the METHOD that matched and a confidence.

Honesty rules (same spirit as ``citation_service``)
---------------------------------------------------
* A link is only made on evidence: an exact DOI, an exact arXiv id, or the
  document's own title appearing in the reference entry. Author + year alone
  is NOT evidence (one author publishes several papers a year).
* A reference that two different library documents match equally well is
  reported as ambiguous and produces NO link, rather than a coin-flip.
* A document with a short or generic title (fewer than three words) can only
  be matched by DOI or arXiv id, because "Attention" or "Deep Learning"
  appears inside hundreds of unrelated entries.
* Nothing is invented: a title, year or id that cannot be read stays empty.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Optional

# Bump when parsing/matching behaviour changes: cached parses carry this, so a
# better parser re-reads every document once instead of trusting stale output.
PARSER_VERSION = 1

# ---------------------------------------------------------------------------
# Small normalisers
# ---------------------------------------------------------------------------

_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-", "\u00ad": "",
})


def fold(text: str) -> str:
    """Lower-case, strip accents and ligatures. Keeps punctuation."""
    text = (text or "").translate(_LIGATURES)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def tokens(text: str) -> list[str]:
    """Alphanumeric word tokens of ``fold(text)`` - the unit title matching works on."""
    return re.findall(r"[a-z0-9]+", fold(text))


def norm_title(text: str) -> str:
    return " ".join(tokens(text))


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>\]\[{}|\\^`]+)", re.I)
_ARXIV_NEW_RE = re.compile(r"(?:arxiv\s*[:/]?\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_OLD_RE = re.compile(
    r"(?:arxiv\s*[:/]?\s*|arxiv\.org/(?:abs|pdf)/)([a-z]+(?:-[a-z]+)*(?:\.[a-z]{2})?/\d{7})(?:v\d+)?", re.I)
_ARXIV_FILENAME_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
_YEAR_RE = re.compile(r"(?<![\d.])((?:19|20)\d{2})(?![\d])")


def clean_doi(raw: str) -> str:
    """A DOI as it should be compared: lower case, trailing punctuation removed."""
    doi = (raw or "").strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    # Prose punctuation glued to the end of a DOI is not part of it.
    while doi and doi[-1] in ".,;:)'\"":
        doi = doi[:-1]
    return doi


def find_doi(text: str) -> str:
    m = _DOI_RE.search(text or "")
    return clean_doi(m.group(1)) if m else ""


def find_arxiv_id(text: str) -> str:
    """The first arXiv id in *text*, version suffix removed; '' if none."""
    m = _ARXIV_NEW_RE.search(text or "")
    if m:
        return m.group(1)
    m = _ARXIV_OLD_RE.search(text or "")
    return m.group(1).lower() if m else ""


def arxiv_id_from_filename(file_name: str) -> str:
    """
    'arxiv_2301.01234.md' (what paper_monitor writes) or '1706.03762v5.pdf'
    (what arXiv itself names a download). Old-style ids contain a slash, which
    paper_monitor writes as '_' ('arxiv_hep-th_9901001.md').
    """
    stem = (file_name or "").rsplit(".", 1)[0] if "." in (file_name or "") else (file_name or "")
    m = re.match(r"(?i)^arxiv[_\-\s]*([a-z\-]+)[_/](\d{7})(?:v\d+)?$", stem)
    if m:
        return f"{m.group(1).lower()}/{m.group(2)}"
    m = _ARXIV_FILENAME_RE.search(stem)
    if m and (stem.lower().startswith("arxiv") or re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", stem)):
        return m.group(1)
    return ""


def year_from_arxiv_id(arxiv_id: str) -> Optional[int]:
    """New-style ids start YYMM: '1706.03762' -> 2017. Old style: 'hep-th/9901001' -> 1999."""
    m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", arxiv_id or "")
    if m and 1 <= int(m.group(2)) <= 12:
        yy = int(m.group(1))
        return 2000 + yy if yy < 90 else 1900 + yy
    m = re.match(r"^[a-z\-]+(?:\.[a-z]{2})?/(\d{2})(\d{2})\d{3}$", arxiv_id or "")
    if m and 1 <= int(m.group(2)) <= 12:
        yy = int(m.group(1))
        return 2000 + yy if yy < 90 else 1900 + yy
    return None


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """One entry of a paper's reference list."""
    raw: str
    doi: str = ""
    arxiv_id: str = ""
    years: list = field(default_factory=list)
    title_guess: str = ""

    def to_dict(self) -> dict:
        return {"raw": self.raw, "doi": self.doi, "arxiv_id": self.arxiv_id,
                "years": list(self.years), "title_guess": self.title_guess}

    @classmethod
    def from_dict(cls, d: dict) -> "Reference":
        return cls(raw=d.get("raw", ""), doi=d.get("doi", ""), arxiv_id=d.get("arxiv_id", ""),
                   years=list(d.get("years") or []), title_guess=d.get("title_guess", ""))


@dataclass
class Identity:
    """Who a document says it is. Every field may be empty; ``title_source`` says how the title was found."""
    title: str = ""
    title_source: str = "filename"      # front-matter | heading | font-size | pdf-metadata | first-line | filename
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    year_source: str = ""               # arxiv | front-matter | text ('' when the year is unknown)
    arxiv_id: str = ""
    doi: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "title_source": self.title_source, "authors": list(self.authors),
                "year": self.year, "year_source": self.year_source, "arxiv_id": self.arxiv_id, "doi": self.doi}

    @classmethod
    def from_dict(cls, d: dict) -> "Identity":
        return cls(title=d.get("title", ""), title_source=d.get("title_source", "filename"),
                   authors=list(d.get("authors") or []), year=d.get("year"),
                   year_source=d.get("year_source", ""),
                   arxiv_id=d.get("arxiv_id", ""), doi=d.get("doi", ""))


# ---------------------------------------------------------------------------
# Finding the reference section
# ---------------------------------------------------------------------------

# A heading must be the whole line: "References", "7. References", "## References",
# "REFERENCES". A table-of-contents line ("References ........ 12") is not one.
_REF_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\d{1,2}(?:\.\d+)*\.?\s+|[IVX]{1,4}\.\s+)?"
    r"(references?(?:\s+list)?|bibliography|works\s+cited|literature\s+cited|references\s+and\s+notes)"
    r"\s*:?\s*$", re.I)

# Headings that normally come AFTER the reference list and so end it.
_AFTER_REFS_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[A-Z]\.?\s+|\d{1,2}\.?\s+)?"
    r"(appendix|appendices|supplementary(?:\s+\w+)?|supplemental(?:\s+\w+)?|acknowledge?ments?|"
    r"author\s+(?:biograph\w+|contributions?)|about\s+the\s+authors?|index|checklist)"
    r"(?:\s+[A-Z0-9][\w ,:\-]{0,60})?\s*$", re.I)


def text_before_references(text: str) -> str:
    """*text* up to its first reference-list heading: the part that says who the document IS."""
    pos = 0
    for ln in (text or "").splitlines(keepends=True):
        if len(ln.strip()) <= 60 and _REF_HEADING_RE.match(ln.strip()):
            return text[:pos]
        pos += len(ln)
    return text or ""


def has_reference_heading(text: str) -> bool:
    """True if any line of *text* is a reference-list heading (used to find the right tail page)."""
    return any(len(ln) <= 60 and _REF_HEADING_RE.match(ln) for ln in (text or "").splitlines())


def find_reference_section(text: str, early_cutoff: bool = True) -> Optional[str]:
    """
    The text after the LAST reference-list heading, cut at a following
    appendix-type heading. None when the document has no such heading.

    The last heading is used because "References" can also open a short
    section earlier (a related-work note, a table of contents). A heading in
    the first third of a long document is ignored for the same reason, unless
    ``early_cutoff`` is False - which is what a caller passes when *text* is
    already only the tail of the document, so "early in this text" means
    nothing.
    """
    if not text:
        return None
    lines = text.splitlines()
    total = len(text)
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    start_idx = None
    for i, ln in enumerate(lines):
        if len(ln) <= 60 and _REF_HEADING_RE.match(ln):
            if early_cutoff and total > 6000 and offsets[i] < total * 0.33:
                continue
            start_idx = i
    if start_idx is None:
        return None

    body = lines[start_idx + 1:]
    # Skip an appendix cut when it is the first thing (an empty list followed by
    # an appendix is just an empty list, not a reason to read the appendix).
    for j, ln in enumerate(body):
        if j > 2 and len(ln) <= 90 and _AFTER_REFS_RE.match(ln):
            body = body[:j]
            break
    section = "\n".join(body).strip()
    return section or None


# ---------------------------------------------------------------------------
# Splitting into entries
# ---------------------------------------------------------------------------

_BRACKET_START_RE = re.compile(r"^\s*\[(\d{1,3})\]\s*", re.M)
_NUMDOT_START_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+(?=[A-Z\[“\"'])", re.M)
# "Smith, J." / "Smith J." / "J. Smith" / "van der Berg, A." - the start of an author-year entry.
_AUTHOR_START_RE = re.compile(
    r"^(?:[A-Z][\w'’\-]+(?:\s(?:[a-z]{2,4}\s)*[A-Z][\w'’\-]+)?,\s+(?:[A-Z]\.|[A-Z][a-z]+\b)"
    r"|[A-Z]\.(?:\s?[A-Z]\.)*\s+[A-Z][\w'’\-]{2,}"
    r"|[A-Z][\w'’\-]+\s+[A-Z]{1,3}(?:,|\s+and\b|\s+&|\.))")
_PAGE_NOISE_RE = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$|^\s*-\s*\d{1,4}\s*-\s*$", re.I)


def _is_increasing(nums: list[int]) -> bool:
    """Mostly 1,2,3,... - tolerant of a few missed numbers, not of random digits."""
    if len(nums) < 3:
        return False
    ups = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)
    return ups >= 0.6 * (len(nums) - 1)


def _join_lines(lines: Iterable[str]) -> str:
    """Re-join wrapped lines of one entry, repairing hyphenation at a line break."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln          # "learn-" + "ing"  -> "learning"
        elif re.search(r"(?:/|\.org/\S*|10\.\d{4,9}/\S*)$", out) and not ln[:1].isspace():
            out = out + ln               # a DOI / URL broken across lines
        else:
            out = out + " " + ln
    return re.sub(r"\s+", " ", out).strip()


def split_references(section: str) -> list[str]:
    """
    Split a reference section into one string per entry.

    Handles ``[1] ...``, ``1. ...`` and author-year lists. Anything it cannot
    confidently split is returned as one entry per blank-line paragraph, or as
    nothing - never as a guess that merges two papers into one.
    """
    if not section:
        return []
    text = "\n".join(ln for ln in section.splitlines() if not _PAGE_NOISE_RE.match(ln))
    # fitz often puts the marker on its own line: "[12]\nSmith, ..." -> one line.
    text = re.sub(r"^(\s*\[\d{1,3}\])[ \t]*\n(?=\S)", r"\1 ", text, flags=re.M)

    marks = list(_BRACKET_START_RE.finditer(text))
    nums = [int(m.group(1)) for m in marks]
    # A one- or two-entry list is accepted only if it is exactly [1] or [1] [2]; longer
    # lists may skip a number or two (a lost line) but must mostly count upwards.
    if marks and (nums == list(range(1, len(nums) + 1)) if len(marks) < 3 else _is_increasing(nums)):
        return _cut(text, marks)

    marks = list(_NUMDOT_START_RE.finditer(text))
    if len(marks) >= 3 and _is_increasing([int(m.group(1)) for m in marks]):
        return _cut(text, marks)

    # Author-year (or unnumbered). Paragraph breaks are the strongest signal.
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 3 and sum(1 for p in paragraphs if len(p.splitlines()) <= 6) >= 0.7 * len(paragraphs):
        return [_join_lines(p.splitlines()) for p in paragraphs if len(p.strip()) > 15]

    entries, cur = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        starts_new = bool(_AUTHOR_START_RE.match(s)) and cur and re.search(r"[.)\d\]]\s*$", cur[-1])
        if starts_new:
            entries.append(_join_lines(cur))
            cur = [s]
        else:
            cur.append(s)
    if cur:
        entries.append(_join_lines(cur))
    return [e for e in entries if len(e) > 15]


def _cut(text: str, marks: list) -> list[str]:
    entries = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = _join_lines(text[m.end():end].splitlines())
        if len(body) > 10:
            entries.append(body)
    return entries


# ---------------------------------------------------------------------------
# One reference entry
# ---------------------------------------------------------------------------

_QUOTED_TITLE_RE = re.compile(r"[\"“]([^\"”]{12,300})[\"”]")


def _entry_years(text: str) -> list[int]:
    """Plausible publication years in an entry, ignoring digits inside DOIs / arXiv ids / page ranges."""
    cleaned = _DOI_RE.sub(" ", text)
    cleaned = _ARXIV_NEW_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\b\d{4}\.\d{4,5}\b", " ", cleaned)
    years = []
    for m in _YEAR_RE.finditer(cleaned):
        y = int(m.group(1))
        if 1900 <= y <= 2100 and y not in years:
            years.append(y)
    return years


_VENUE_START_RE = re.compile(
    r"(?i)^(?:in\s|proc\b|proceedings|journal\b|conference|transactions|arxiv|ieee\b|acm\b|vol\b|pp\b)")


def _author_like(seg: str) -> bool:
    """A short segment that reads as an author list: 'Smith, J', 'et al', 'A. B. Jones'."""
    words = seg.split()
    if len(words) > 4:
        return False
    return ("," in seg or "et al" in seg.lower()
            or any(re.fullmatch(r"[A-Z]\.?", w) for w in words))


def _usable_title(t: str) -> str:
    t = t.strip(" .,;:")
    if len(t.split()) < 3 or _VENUE_START_RE.match(t) or re.search(r"\(\d{4}\)", t):
        return ""
    return t


def guess_title(entry: str) -> str:
    """
    Best-effort title of a reference entry, used for DISPLAY and for grouping
    unresolved references. Link-making never relies on this: it matches the
    library document's own title against the whole entry instead. When no
    guess looks trustworthy the answer is '' rather than a venue or an author.
    """
    m = _QUOTED_TITLE_RE.search(entry)
    if m:
        return m.group(1).strip(" .,")
    # APA-ish: "Authors (2017). Title. Venue." / "Authors. 2017. Title. Venue."
    m = re.search(r"\(?\b(?:19|20)\d{2}[a-z]?\)?[.,]?\s+(.{12,250}?)(?:\.\s|\?\s|$)", entry)
    if m and m.start() < 0.6 * len(entry):
        t = _usable_title(m.group(1))
        if t:
            return t
    # Authors. Title. Venue (year).  -> the first segment that is not an author list.
    for seg in re.split(r"\.\s+(?=[A-Z\"\u201c])", entry):
        if _author_like(seg):
            continue
        t = _usable_title(seg)
        if t:
            return t
    return ""


def parse_reference(entry: str) -> Reference:
    entry = re.sub(r"\s+", " ", entry or "").strip()
    return Reference(raw=entry, doi=find_doi(entry), arxiv_id=find_arxiv_id(entry),
                     years=_entry_years(entry), title_guess=guess_title(entry))


def parse_reference_section(section: str) -> list[Reference]:
    """References of a section that has ALREADY been cut at its heading."""
    return [parse_reference(e) for e in split_references(section)]


def parse_references(text: str, early_cutoff: bool = True) -> tuple[list[Reference], bool]:
    """(references, found_a_reference_section) for a document's raw text."""
    section = find_reference_section(text, early_cutoff=early_cutoff)
    if section is None:
        return [], False
    return parse_reference_section(section), True


# ---------------------------------------------------------------------------
# A document's own identity
# ---------------------------------------------------------------------------

_BAD_TITLE_RE = re.compile(
    r"(?i)^(?:untitled|microsoft word|document\d*|slide \d+|powerpoint|title|\w*\.(?:docx?|tex|pdf|indd))\b|\.(?:docx?|tex|dvi)\b|^\s*$")
_ARXIV_STAMP_ID_RE = re.compile(r"arxiv:\s*(?:\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(?:v\d+)?\s*\[[^\]]+\]", re.I)
_ARXIV_STAMP_YEAR_RE = re.compile(r"arxiv:\S+\s*\[[^\]]+\]\s*\d{1,2}\s+[A-Za-z]{3,9}\s+((?:19|20)\d{2})", re.I)

# A year is only taken from page-1 text when it sits in a date-like context. A bare
# year ("... since 2012 ...") is just as likely to be something the paper CITES.
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_TEXT_YEAR_PATTERNS = [
    re.compile(r"(?:\u00a9|\(c\)|copyright)\s*(?:\u00a9\s*)?((?:19|20)\d{2})", re.I),
    re.compile(r"\b(?:published|received|accepted|submitted|revised|preprint|available online|presented)\b[^\n]{0,60}?\b((?:19|20)\d{2})\b", re.I),
    re.compile(rf"\b{_MONTH}\.?\s+(?:\d{{1,2}},?\s+)?((?:19|20)\d{{2}})\b", re.I),
    re.compile(rf"\b\d{{1,2}}\s+{_MONTH}\.?,?\s+((?:19|20)\d{{2}})\b", re.I),
]


def year_from_text(text: str) -> Optional[int]:
    """The publication year a first page states in a date-like context, else None."""
    head = _DOI_RE.sub(" ", (text or "")[:3500])
    for pat in _TEXT_YEAR_PATTERNS:
        m = pat.search(head)
        if m and 1900 <= int(m.group(1)) <= 2100:
            return int(m.group(1))
    return None


def identity_from_markdown(text: str, file_name: str = "") -> Optional[Identity]:
    """
    The Markdown that mnemosyne's paper_monitor writes for each arXiv paper
    ('# Title' then '- arXiv: ..', '- Authors: ..', '- Published: ..').
    None when the text does not look like one of those files.
    """
    head = (text or "")[:4000]
    m = re.match(r"\s*#\s+(.+)", head)
    meta = dict(re.findall(r"^-\s*(arXiv|Authors|Published)\s*:\s*(.+)$", head, flags=re.M))
    if not m or "arXiv" not in meta:
        return None
    ident = Identity(title=m.group(1).strip(), title_source="front-matter")
    ident.arxiv_id = find_arxiv_id("arXiv:" + meta["arXiv"]) or meta["arXiv"].strip().lower()
    authors = meta.get("Authors", "").strip()
    if authors and authors.lower() != "unknown":
        ident.authors = [a.strip() for a in re.sub(r"\s+et al\.?$", "", authors).split(",") if a.strip()]
    py = _YEAR_RE.search(meta.get("Published", ""))
    ident.year = int(py.group(1)) if py else year_from_arxiv_id(ident.arxiv_id)
    ident.year_source = "front-matter" if py else ("arxiv" if ident.year else "")
    return ident


def identity_from_first_page(
    lines: list[tuple[str, float]],
    page_text: str,
    file_name: str = "",
    pdf_title: str = "",
    pdf_author: str = "",
) -> Identity:
    """
    Identity of a document from its first page.

    *lines* are ``(text, font_size)`` in reading order (``font_size`` 0 when
    the source has no sizes, e.g. a Word file). The title is, in order of
    trust: the largest-type block on the page; the file's own metadata title
    when it looks like a real title; the first reasonable line; the file name.
    """
    ident = Identity()
    # The arXiv stamp ("arXiv:1706.03762v5 [cs.CL] 6 Dec 2017") names THIS paper. A bare
    # "arXiv:..." elsewhere on the page is as likely to be something it cites.
    stamp = _ARXIV_STAMP_ID_RE.search(page_text or "")
    ident.arxiv_id = (find_arxiv_id(stamp.group(0)) if stamp else "") or arxiv_id_from_filename(file_name) \
        or find_arxiv_id((page_text or "")[:600])
    ident.doi = find_doi(page_text[:3000]) if page_text else ""

    # -- year: the arXiv stamp date, else the arXiv id, else a plain year near the top.
    m = _ARXIV_STAMP_YEAR_RE.search(page_text or "")
    if m:
        ident.year, ident.year_source = int(m.group(1)), "arxiv"
    elif ident.arxiv_id and year_from_arxiv_id(ident.arxiv_id):
        ident.year, ident.year_source = year_from_arxiv_id(ident.arxiv_id), "arxiv"
    else:
        ident.year = year_from_text(page_text)
        ident.year_source = "text" if ident.year else ""

    # -- title
    sized = [(t.strip(), s) for t, s in lines if t.strip() and s > 0]
    title = ""
    if sized:
        cand = [(t, s) for t, s in sized if 2 <= len(t.split()) <= 40 and not _BAD_TITLE_RE.match(t)
                and not t.lower().startswith(("arxiv:", "http", "doi", "preprint", "under review", "published as"))]
        if cand:
            biggest = max(s for _, s in cand)
            block, started = [], False
            for t, s in sized:
                if s >= biggest * 0.93 and (t, s) in cand:
                    block.append(t)
                    started = True
                elif started:
                    break
            title = " ".join(block).strip()
    if title and len(title) >= 8:
        ident.title, ident.title_source = _tidy_title(title), "font-size"
    elif pdf_title and not _BAD_TITLE_RE.match(pdf_title.strip()) and len(pdf_title.split()) >= 2 \
            and norm_title(pdf_title) != norm_title(file_name.rsplit(".", 1)[0]):
        ident.title, ident.title_source = _tidy_title(pdf_title), "pdf-metadata"
    else:
        first = next((t.strip() for t, _ in lines if len(t.split()) >= 3 and not _BAD_TITLE_RE.match(t.strip())), "")
        if first and len(first) <= 200:
            ident.title, ident.title_source = _tidy_title(first), "first-line"
        else:
            stem = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
            ident.title, ident.title_source = re.sub(r"[_\-]+", " ", stem).strip(), "filename"

    if pdf_author and len(pdf_author) < 200 and not re.search(r"(?i)\.(docx?|pdf)|administrator|user", pdf_author):
        ident.authors = [a.strip() for a in re.split(r"[;,]|\band\b", pdf_author) if a.strip()][:12]
    return ident


def _tidy_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    return re.sub(r"(?<=\w)- (?=[a-z])", "", title)      # "inter- national" from a wrapped heading


def identity_from_filename(file_name: str) -> Identity:
    stem = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
    aid = arxiv_id_from_filename(file_name)
    year = year_from_arxiv_id(aid) if aid else None
    return Identity(title=re.sub(r"[_\-]+", " ", stem).strip(), title_source="filename",
                    arxiv_id=aid, year=year, year_source="arxiv" if year else "")


# ---------------------------------------------------------------------------
# Matching references to library documents
# ---------------------------------------------------------------------------

MIN_TITLE_WORDS = 3          # fewer than this: only DOI / arXiv id can match
LONG_TITLE_WORDS = 5


@dataclass
class LibraryDoc:
    """What the matcher needs to know about one library document."""
    key: str
    identity: Identity
    n_title_tokens: int = 0
    title_norm: str = ""
    title_set: frozenset = frozenset()

    def __post_init__(self) -> None:
        self.title_norm = norm_title(self.identity.title)
        words = self.title_norm.split()
        self.n_title_tokens = len(words)
        self.title_set = frozenset(words)


@dataclass
class Link:
    """``source`` cites ``target``."""
    source: str
    target: str
    method: str            # doi | arxiv | title | title-fuzzy
    confidence: float
    ref_index: int
    year_mismatch: bool = False

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "method": self.method,
                "confidence": round(self.confidence, 2), "ref_index": self.ref_index,
                "year_mismatch": self.year_mismatch}


@dataclass
class MatchOutcome:
    links: list = field(default_factory=list)
    unresolved: dict = field(default_factory=dict)     # source key -> [Reference not matched to any library doc]
    ambiguous: list = field(default_factory=list)      # (source key, ref_index, [candidate keys])


def _year_gap(doc_year: Optional[int], ref_years: list) -> Optional[int]:
    """Smallest distance between the document's year and any year in the entry; None if either is unknown."""
    if not doc_year or not ref_years:
        return None
    return min(abs(doc_year - y) for y in ref_years)


def _fuzzy_title_hit(title_toks: list[str], entry_toks: list[str]) -> float:
    """Best similarity of the title against any same-length window of the entry (0 when hopeless)."""
    n = len(title_toks)
    if n < MIN_TITLE_WORDS or len(entry_toks) < n - 1:
        return 0.0
    want = " ".join(title_toks)
    best = 0.0
    for size in {max(n - 1, 1), n, n + 1}:
        for i in range(0, max(len(entry_toks) - size, 0) + 1):
            window = " ".join(entry_toks[i:i + size])
            if abs(len(window) - len(want)) > 0.2 * len(want):
                continue
            best = max(best, SequenceMatcher(None, want, window).ratio())
    return best


def _score_title(doc: LibraryDoc, ref: Reference, entry_toks: list[str], entry_set: frozenset,
                 entry_norm: str):
    """(method, confidence, year_mismatch) for a title-based match, or None."""
    if doc.n_title_tokens < MIN_TITLE_WORDS:
        return None
    # Cheapest test first: most of the title's words must be in the entry at all.
    # (One C-speed set intersection rejects almost every document for almost every entry.)
    overlap = len(doc.title_set & entry_set)
    if overlap < 0.8 * len(doc.title_set):
        return None
    ident = doc.identity
    # A title taken from the file name is a guess about the file, not the paper.
    if ident.title_source == "filename" and doc.n_title_tokens < LONG_TITLE_WORDS:
        return None

    if overlap == len(doc.title_set) and f" {doc.title_norm} " in f" {entry_norm} ":
        method, conf = "title", 0.95 if doc.n_title_tokens >= LONG_TITLE_WORDS else 0.85
    else:
        sim = _fuzzy_title_hit(doc.title_norm.split(), entry_toks)
        if sim < 0.9:
            return None
        method, conf = "title-fuzzy", 0.8 if doc.n_title_tokens >= LONG_TITLE_WORDS else 0.72

    gap = _year_gap(ident.year, ref.years)
    mismatch = gap is not None and gap > 2
    if mismatch:
        if doc.n_title_tokens < LONG_TITLE_WORDS:
            return None                  # a short title AND a different year: a different paper
        conf -= 0.15
    if ident.title_source in ("filename", "first-line"):
        conf -= 0.1                      # the title itself is less certain
    return method, round(max(conf, 0.0), 2), mismatch


def match_references(
    refs_by_doc: dict[str, list[Reference]],
    docs: Iterable[LibraryDoc],
) -> MatchOutcome:
    """
    Resolve every reference of every document against the library.

    ``refs_by_doc`` maps a document key to its parsed references; ``docs`` is
    the whole library (a document may be cited without having a reference list
    of its own, e.g. an arXiv abstract file).
    """
    docs = list(docs)
    by_doi = {d.identity.doi: d for d in docs if d.identity.doi}
    by_arxiv = {d.identity.arxiv_id: d for d in docs if d.identity.arxiv_id}
    out = MatchOutcome()

    for src_key, refs in refs_by_doc.items():
        seen_targets: set[str] = set()
        for idx, ref in enumerate(refs):
            hit = None            # (doc, method, conf, mismatch)
            if ref.doi and ref.doi in by_doi:
                hit = (by_doi[ref.doi], "doi", 1.0, False)
            elif ref.arxiv_id and ref.arxiv_id in by_arxiv:
                hit = (by_arxiv[ref.arxiv_id], "arxiv", 1.0, False)
            else:
                entry_norm = norm_title(ref.raw)
                entry_toks = entry_norm.split()
                entry_set = frozenset(entry_toks)
                scored = []
                for d in docs:
                    if d.key == src_key:
                        continue
                    s = _score_title(d, ref, entry_toks, entry_set, entry_norm)
                    if s:
                        scored.append((d, *s))
                if scored:
                    # When one matching title is a sub-phrase of another's ("A Survey of X" inside
                    # "A Survey of X and Y"), the longer title is the more specific claim.
                    top_len = max(s[0].n_title_tokens for s in scored if s[1] == "title") \
                        if any(s[1] == "title" for s in scored) else None
                    if top_len is not None:
                        scored = [s for s in scored if not (s[1] == "title" and s[0].n_title_tokens < top_len)]
                    scored.sort(key=lambda s: s[2], reverse=True)
                    best = scored[0]
                    rivals = [s for s in scored[1:] if s[2] >= best[2] - 0.05 and s[0].key != best[0].key]
                    if rivals:
                        out.ambiguous.append((src_key, idx, [best[0].key] + [r[0].key for r in rivals]))
                        out.unresolved.setdefault(src_key, []).append(ref)
                        continue
                    hit = best

            if hit is None or hit[0].key == src_key:
                out.unresolved.setdefault(src_key, []).append(ref)
                continue
            doc, method, conf, mismatch = hit
            if doc.key in seen_targets:
                continue                 # the same paper listed twice in one reference list
            seen_targets.add(doc.key)
            out.links.append(Link(src_key, doc.key, method, conf, idx, mismatch))
    return out


# ---------------------------------------------------------------------------
# Grouping references nobody in the library matched
# ---------------------------------------------------------------------------

def reference_group_key(ref: Reference) -> str:
    """
    A key two entries share only when they are the same work: DOI, arXiv id,
    or an identical normalised title of at least four words. '' otherwise,
    which keeps weak guesses from merging different papers.
    """
    if ref.doi:
        return "doi:" + ref.doi
    if ref.arxiv_id:
        return "arxiv:" + ref.arxiv_id
    t = norm_title(ref.title_guess)
    return "title:" + t if len(t.split()) >= 4 else ""
