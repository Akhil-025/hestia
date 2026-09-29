# tests/test_athena_chunking.py
"""
Tests for configurable chunk size/overlap per document type (backlog
#62): `PDFProcessor.semantic_chunking` accepting per-call overrides, the
overlap value actually being used (it was previously stored on the
instance and never read), and `AthenaConfig.chunk_config_by_type` being
resolved per file extension.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_athena import _ensure_stubs  # noqa: E402
_ensure_stubs()

from modules.athena.config import AthenaConfig  # noqa: E402
from modules.athena.pdf_processor import PDFProcessor  # noqa: E402


def make_processor(chunk_size=1000, chunk_overlap=200):
    return PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


_LONG_TEXT = " ".join(
    f"This is sentence number {i} discussing a distinct topic in detail."
    for i in range(60)
)


# ---------------------------------------------------------------------------
# Per-call overrides
# ---------------------------------------------------------------------------

def test_smaller_chunk_size_produces_more_chunks():
    proc = make_processor(chunk_size=1000, chunk_overlap=100)
    big_chunks = proc.semantic_chunking(_LONG_TEXT, chunk_size=2000, chunk_overlap=100)
    small_chunks = proc.semantic_chunking(_LONG_TEXT, chunk_size=300, chunk_overlap=50)
    assert len(small_chunks) > len(big_chunks)


def test_override_does_not_mutate_instance_defaults():
    proc = make_processor(chunk_size=1000, chunk_overlap=200)
    proc.semantic_chunking(_LONG_TEXT, chunk_size=100, chunk_overlap=10)
    assert proc.chunk_size == 1000
    assert proc.chunk_overlap == 200


def test_no_override_falls_back_to_instance_defaults():
    proc = make_processor(chunk_size=300, chunk_overlap=50)
    with_default = proc.semantic_chunking(_LONG_TEXT)
    explicit = proc.semantic_chunking(_LONG_TEXT, chunk_size=300, chunk_overlap=50)
    assert with_default == explicit


# ---------------------------------------------------------------------------
# Overlap is actually used (previously dead config)
# ---------------------------------------------------------------------------

def test_larger_overlap_carries_more_text_into_the_next_chunk():
    proc = make_processor(chunk_size=200)
    small_overlap = proc.semantic_chunking(_LONG_TEXT, chunk_size=200, chunk_overlap=10)
    large_overlap = proc.semantic_chunking(_LONG_TEXT, chunk_size=200, chunk_overlap=150)
    # More overlap means later chunks are systematically longer, since
    # more trailing text from the previous chunk is carried forward.
    if len(small_overlap) > 1 and len(large_overlap) > 1:
        avg_small = sum(len(c) for c in small_overlap[1:]) / (len(small_overlap) - 1)
        avg_large = sum(len(c) for c in large_overlap[1:]) / (len(large_overlap) - 1)
        assert avg_large > avg_small


def test_zero_overlap_means_no_carry_forward():
    proc = make_processor(chunk_size=200)
    chunks = proc.semantic_chunking(_LONG_TEXT, chunk_size=200, chunk_overlap=0)
    # With zero overlap, no chunk after the first should start with text
    # that's a suffix of the previous chunk.
    for i in range(1, len(chunks)):
        assert not chunks[i].startswith(chunks[i - 1][-20:]) or len(chunks[i - 1]) < 20


def test_overlap_larger_than_chunk_size_does_not_crash():
    proc = make_processor(chunk_size=100, chunk_overlap=100000)
    chunks = proc.semantic_chunking(_LONG_TEXT, chunk_size=100, chunk_overlap=100000)
    assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# AthenaConfig.chunk_config_by_type
# ---------------------------------------------------------------------------

def test_default_chunk_config_has_entries_for_common_types():
    cfg = AthenaConfig()
    assert ".pdf" in cfg.chunk_config_by_type
    assert ".txt" in cfg.chunk_config_by_type
    assert ".pptx" in cfg.chunk_config_by_type


def test_pdf_chunk_size_is_larger_than_slide_chunk_size_by_default():
    # Dense academic PDFs warrant more context per chunk than terse
    # slide-deck bullet points.
    cfg = AthenaConfig()
    pdf_size, _ = cfg.chunk_config_by_type[".pdf"]
    pptx_size, _ = cfg.chunk_config_by_type[".pptx"]
    assert pdf_size > pptx_size


def test_two_config_instances_have_independent_dicts():
    # dataclass field(default_factory=...) must produce a fresh dict per
    # instance — a shared mutable default would leak edits across configs.
    a = AthenaConfig()
    b = AthenaConfig()
    a.chunk_config_by_type[".pdf"] = (99, 1)
    assert b.chunk_config_by_type[".pdf"] != (99, 1)


def test_custom_chunk_config_overrides_defaults():
    cfg = AthenaConfig(chunk_config_by_type={".pdf": (5000, 500)})
    assert cfg.chunk_config_by_type[".pdf"] == (5000, 500)
    assert ".txt" not in cfg.chunk_config_by_type  # fully replaced, not merged
