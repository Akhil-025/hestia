# tests/test_athena_citations.py
"""
Tests for modules/athena/services/citation_service.py (backlog #55).

Pure Python — no LLM, no ChromaDB — so no stub setup needed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_athena import _ensure_stubs  # noqa: E402
_ensure_stubs()

from modules.athena.services.citation_service import Citation, CitationRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

def test_title_guess_strips_extension_and_underscores():
    c = Citation(file_name="turbulence_modeling_notes.pdf")
    assert c.title_guess == "turbulence modeling notes"


def test_title_guess_handles_hyphens_too():
    c = Citation(file_name="heat-transfer-2024.pdf")
    assert c.title_guess == "heat transfer 2024"


def test_bibtex_includes_the_source_file_name():
    c = Citation(file_name="notes.pdf")
    entry = c.to_bibtex("key1")
    assert "notes.pdf" in entry
    assert "@misc{key1" in entry


def test_bibtex_never_fabricates_author_or_year():
    c = Citation(file_name="notes.pdf")
    entry = c.to_bibtex("key1")
    assert "author" not in entry
    assert "year" not in entry


def test_bibtex_includes_page_range_when_present():
    c = Citation(file_name="notes.pdf", pages={3, 1, 5})
    entry = c.to_bibtex("key1")
    assert "pages = {1--5}" in entry


def test_bibtex_omits_pages_when_none_recorded():
    c = Citation(file_name="notes.pdf")
    entry = c.to_bibtex("key1")
    assert "pages" not in entry


def test_apa_style_includes_file_name_as_bracketed_source():
    c = Citation(file_name="notes.pdf")
    apa = c.to_apa_style()
    assert "[notes.pdf]" in apa


def test_apa_style_never_fabricates_author_or_year():
    c = Citation(file_name="notes.pdf")
    apa = c.to_apa_style()
    assert "20" not in apa  # no plausible-looking year digits


# ---------------------------------------------------------------------------
# CitationRegistry
# ---------------------------------------------------------------------------

def test_add_source_accepts_source_document_to_dict_shape():
    registry = CitationRegistry()
    registry.add_source({"file_name": "notes.pdf", "subject": "Bio", "page": 3})
    assert len(registry) == 1


def test_repeated_sources_from_the_same_file_merge_into_one_citation():
    registry = CitationRegistry()
    registry.add_source({"file_name": "notes.pdf", "page": 1})
    registry.add_source({"file_name": "notes.pdf", "page": 5})
    assert len(registry) == 1
    assert registry.citations()[0].pages == {1, 5}


def test_different_files_produce_separate_citations():
    registry = CitationRegistry()
    registry.add_source({"file_name": "a.pdf"})
    registry.add_source({"file_name": "b.pdf"})
    assert len(registry) == 2


def test_source_without_a_file_name_is_ignored():
    registry = CitationRegistry()
    registry.add_source({"subject": "Bio"})  # no file_name
    assert len(registry) == 0


def test_add_sources_handles_a_list():
    registry = CitationRegistry()
    registry.add_sources([
        {"file_name": "a.pdf", "page": 1},
        {"file_name": "b.pdf", "page": 2},
    ])
    assert len(registry) == 2


def test_citations_are_sorted_by_file_name():
    registry = CitationRegistry()
    registry.add_source({"file_name": "zebra.pdf"})
    registry.add_source({"file_name": "apple.pdf"})
    names = [c.file_name for c in registry.citations()]
    assert names == ["apple.pdf", "zebra.pdf"]


def test_to_bibtex_with_no_citations_is_empty():
    assert CitationRegistry().to_bibtex() == ""


def test_to_apa_style_with_no_citations_is_empty():
    assert CitationRegistry().to_apa_style() == ""


def test_to_bibtex_produces_one_entry_per_citation():
    registry = CitationRegistry()
    registry.add_source({"file_name": "a.pdf"})
    registry.add_source({"file_name": "b.pdf"})
    bibtex = registry.to_bibtex()
    assert bibtex.count("@misc") == 2


def test_bibtex_keys_are_unique_for_similarly_named_files():
    registry = CitationRegistry()
    registry.add_source({"file_name": "notes.pdf"})
    registry.add_source({"file_name": "notes (1).pdf"})
    bibtex = registry.to_bibtex()
    keys = [line.split("{")[1].split(",")[0] for line in bibtex.splitlines() if line.startswith("@misc")]
    assert len(set(keys)) == 2


def test_page_uses_page_number_key_as_a_fallback():
    registry = CitationRegistry()
    registry.add_source({"file_name": "a.pdf", "page_number": 4})  # not "page"
    assert registry.citations()[0].pages == {4}
