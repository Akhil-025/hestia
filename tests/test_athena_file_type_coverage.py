# tests/test_athena_file_type_coverage.py
"""
File-type coverage checks in CI (backlog #67): a real, minimal fixture is
generated at test time for every format `document_processor.py` claims to
support, using that format's OWN writer library — python-docx for .docx,
python-pptx for .pptx, EbookLib for .epub — rather than checked-in binary
fixture files (which rot silently and can't be diffed meaningfully).
.pdf is included too, via PyMuPDF, and skipped gracefully with
`pytest.importorskip` if it isn't installed in this environment — it's
still a real, unconditional dependency in `pdf_processor.py`, just not
present in every sandbox this suite might run in.

The point of these tests: if a document-processing library's API changes
in a way that silently breaks extraction (returns empty text instead of
raising, say), a hand-maintained mock would never catch it — only running
the REAL library against a REAL file does.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_athena import _ensure_stubs  # noqa: E402
_ensure_stubs()

from modules.athena.document_processor import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    extract_text_from_file,
)
from modules.athena.pdf_processor import get_supported_files  # noqa: E402

_SAMPLE_TEXT = "The mitochondria is the powerhouse of the cell."


# ---------------------------------------------------------------------------
# .txt / .md — no library needed, just a real file on disk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".txt", ".md"])
def test_plain_text_formats_extract_real_content(tmp_path, ext):
    path = tmp_path / f"notes{ext}"
    path.write_text(_SAMPLE_TEXT, encoding="utf-8")
    pages = extract_text_from_file(str(path))
    assert pages
    assert _SAMPLE_TEXT in pages[0]["text"]


# ---------------------------------------------------------------------------
# .docx — via python-docx, the same library document_processor.py uses
# ---------------------------------------------------------------------------

def test_docx_extraction_with_a_real_generated_file(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "notes.docx"
    doc = docx.Document()
    doc.add_paragraph(_SAMPLE_TEXT)
    doc.save(str(path))

    pages = extract_text_from_file(str(path))
    assert pages
    assert _SAMPLE_TEXT in pages[0]["text"]


# ---------------------------------------------------------------------------
# .pptx — via python-pptx
# ---------------------------------------------------------------------------

def test_pptx_extraction_with_a_real_generated_file(tmp_path):
    pptx_lib = pytest.importorskip("pptx")
    path = tmp_path / "slides.pptx"
    prs = pptx_lib.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Cell Biology"
    slide.placeholders[1].text = _SAMPLE_TEXT
    prs.save(str(path))

    pages = extract_text_from_file(str(path))
    assert pages
    combined = " ".join(p["text"] for p in pages)
    assert _SAMPLE_TEXT in combined


def test_pptx_extraction_yields_one_page_per_slide(tmp_path):
    pptx_lib = pytest.importorskip("pptx")
    path = tmp_path / "slides.pptx"
    prs = pptx_lib.Presentation()
    for i in range(3):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"Slide {i}"
        slide.placeholders[1].text = f"Content for slide {i}, with enough text to matter."
    prs.save(str(path))

    pages = extract_text_from_file(str(path))
    assert len(pages) == 3


# ---------------------------------------------------------------------------
# .epub — via EbookLib
# ---------------------------------------------------------------------------

def test_epub_extraction_with_a_real_generated_file(tmp_path):
    pytest.importorskip("ebooklib")
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title("Test Book")
    book.set_language("en")

    chapter = epub.EpubHtml(title="Chapter 1", file_name="chap1.xhtml", lang="en")
    chapter.content = f"<html><body><p>{_SAMPLE_TEXT}</p></body></html>"
    book.add_item(chapter)
    book.toc = (chapter,)
    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)

    pages = extract_text_from_file(str(path))
    assert pages
    combined = " ".join(p["text"] for p in pages)
    assert "mitochondria" in combined.lower()


# ---------------------------------------------------------------------------
# .pdf — via PyMuPDF (fitz), skipped if not installed in this environment
# ---------------------------------------------------------------------------

def test_pdf_extraction_with_a_real_generated_file(tmp_path):
    fitz = pytest.importorskip("fitz")
    if not hasattr(fitz, "Document") and not hasattr(fitz, "Page"):
        # This session's chromadb/torch/etc. stub setup (see
        # test_athena._ensure_stubs) also writes a fake `fitz` module —
        # a real-looking import that deliberately raises on use, so other
        # Athena tests in this same session can run without a real
        # PyMuPDF install. `importorskip` alone can't tell that apart
        # from the genuine library, since the stub file really does
        # exist and really does import cleanly.
        pytest.skip("fitz is stubbed in this test session, not really installed")

    path = tmp_path / "notes.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), _SAMPLE_TEXT)
    doc.save(str(path))
    doc.close()

    from modules.athena.pdf_processor import PDFProcessor
    processor = PDFProcessor()
    pages = processor.extract_text_from_pdf(str(path))
    assert pages
    combined = " ".join(p["text"] for p in pages)
    assert "mitochondria" in combined.lower()


# ---------------------------------------------------------------------------
# Coverage invariant: every extension document_processor.py CLAIMS to
# support actually has a real test above exercising it — this is the
# test that fails first if someone adds a new supported extension without
# adding fixture coverage for it.
# ---------------------------------------------------------------------------

def test_every_declared_supported_extension_has_coverage_above():
    # .pdf is handled by pdf_processor.py, not document_processor.py, so
    # it's outside SUPPORTED_EXTENSIONS but still covered by the test
    # above — accounted for explicitly here.
    covered_elsewhere = {".pdf"}
    tested_extensions = {".txt", ".md", ".docx", ".pptx", ".epub"} | covered_elsewhere
    assert SUPPORTED_EXTENSIONS <= tested_extensions, (
        f"document_processor.py now claims to support "
        f"{SUPPORTED_EXTENSIONS - tested_extensions} with no fixture "
        f"coverage in this file — add a test above."
    )


# ---------------------------------------------------------------------------
# get_supported_files respects the same extension set
# ---------------------------------------------------------------------------

def test_get_supported_files_only_returns_known_extensions(tmp_path):
    (tmp_path / "notes.txt").write_text(_SAMPLE_TEXT, encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")  # unsupported
    (tmp_path / "notes.md").write_text(_SAMPLE_TEXT, encoding="utf-8")

    files = get_supported_files(str(tmp_path))
    names = {f["file_name"] for f in files}
    assert "notes.txt" in names
    assert "notes.md" in names
    assert "image.png" not in names


def test_get_supported_files_on_an_empty_directory_returns_empty(tmp_path):
    assert get_supported_files(str(tmp_path)) == []
