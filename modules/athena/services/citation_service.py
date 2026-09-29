"""
modules/athena/services/citation_service.py

Citation management (backlog #55): auto-generate a bibliography from the
sources a search or synthesis actually drew on.

Scope, honestly stated
-------------------------
Athena doesn't extract academic metadata (author, publication year,
journal) from ingested PDFs — only file_name/subject/module/page are
tracked per chunk. So these citations are FILE-based, not full academic
citations: "notes.pdf" stands in for a title, there is no author or year
unless the file name happens to contain one. This is stated plainly in
every formatted citation's docstring below rather than pretending to a
precision the underlying data doesn't have. Extracting real bibliographic
metadata from PDF content (title pages, headers, reference lists) would
be a substantially larger feature — worth doing, but a different one from
"track which files a claim's sources came from and format a reference
list", which is what's implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Citation:
    file_name: str
    subject: str = ""
    pages: set = field(default_factory=set)

    @property
    def title_guess(self) -> str:
        """The file name without its extension, used as a title stand-in."""
        return Path(self.file_name).stem.replace("_", " ").replace("-", " ")

    def to_bibtex(self, key: str) -> str:
        """
        A minimal BibTeX @misc entry. Real author/year/journal fields are
        omitted rather than fabricated — an empty field a person can fill
        in is honest; a plausible-looking guess is not.
        """
        pages = f", pages = {{{min(self.pages)}--{max(self.pages)}}}" if self.pages else ""
        return (
            f"@misc{{{key},\n"
            f"  title = {{{self.title_guess}}},\n"
            f"  note = {{Source file: {self.file_name}}}{pages}\n"
            f"}}"
        )

    def to_apa_style(self) -> str:
        """
        An APA-SHAPED reference, not a verified APA citation — no author
        or year is available from what Athena tracks, so those fields are
        simply absent rather than invented.
        """
        pages = f" (pp. {min(self.pages)}\u2013{max(self.pages)})" if self.pages else ""
        return f"{self.title_guess}{pages}. [{self.file_name}]"


class CitationRegistry:
    """
    Accumulates citations from one or more search/synthesis results, then
    formats them as a bibliography. Built fresh per "which sources did
    this answer use" request rather than persisted — the citation set is
    a property of one specific answer, not standing state.
    """

    def __init__(self) -> None:
        self._citations: dict[str, Citation] = {}

    def add_source(self, source: dict[str, Any]) -> None:
        """
        Register one source — accepts the same dict shape
        `SourceDocument.to_dict()` already produces (file_name, subject,
        page), so a search response's `data["sources"]` can be fed here
        directly with no reshaping.
        """
        file_name = source.get("file_name")
        if not file_name:
            return
        citation = self._citations.setdefault(
            file_name, Citation(file_name=file_name, subject=source.get("subject") or "")
        )
        page = source.get("page") or source.get("page_number")
        if page:
            citation.pages.add(int(page))

    def add_sources(self, sources: list[dict[str, Any]]) -> None:
        for s in sources:
            self.add_source(s)

    def citations(self) -> list[Citation]:
        return sorted(self._citations.values(), key=lambda c: c.file_name)

    def to_bibtex(self) -> str:
        entries = []
        for i, citation in enumerate(self.citations(), start=1):
            key = f"src{i}_" + "".join(c for c in Path(citation.file_name).stem if c.isalnum())[:20]
            entries.append(citation.to_bibtex(key))
        return "\n\n".join(entries)

    def to_apa_style(self) -> str:
        return "\n".join(f"- {c.to_apa_style()}" for c in self.citations())

    def __len__(self) -> int:
        return len(self._citations)
