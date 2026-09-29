# tests/test_athena_debug_scores.py
"""
Tests for surfacing semantic/BM25 retrieval scores in debug mode
(backlog #64).
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_athena import _ensure_stubs, make_engine  # noqa: E402
_ensure_stubs()

from modules.athena.models import SearchResults, SourceDocument  # noqa: E402


# ---------------------------------------------------------------------------
# SourceDocument.to_dict
# ---------------------------------------------------------------------------

def test_to_dict_without_breakdown_omits_score_fields():
    doc = SourceDocument(
        text="x", file_name="a.pdf", file_path="/a.pdf", page_number=1,
        score=0.8, semantic_score=0.9, bm25_score=0.5,
    )
    d = doc.to_dict()
    assert "semantic_score" not in d
    assert "bm25_score" not in d
    assert d["score"] == 0.8


def test_to_dict_with_breakdown_includes_score_fields():
    doc = SourceDocument(
        text="x", file_name="a.pdf", file_path="/a.pdf", page_number=1,
        score=0.8, semantic_score=0.9, bm25_score=0.5,
    )
    d = doc.to_dict(include_score_breakdown=True)
    assert d["semantic_score"] == 0.9
    assert d["bm25_score"] == 0.5


def test_to_dict_breakdown_with_no_scores_computed_is_none_not_missing():
    doc = SourceDocument(text="x", file_name="a.pdf", file_path="/a.pdf", page_number=1)
    d = doc.to_dict(include_score_breakdown=True)
    assert d["semantic_score"] is None
    assert d["bm25_score"] is None


# ---------------------------------------------------------------------------
# SearchResults.to_source_documents populates the breakdown
# ---------------------------------------------------------------------------

def test_to_source_documents_carries_semantic_and_bm25_scores():
    results = SearchResults(
        documents=["chunk one", "chunk two"],
        metadatas=[{"file_name": "a.pdf"}, {"file_name": "b.pdf"}],
        scores=[0.9, 0.7],
        semantic_scores=[0.95, 0.6],
        bm25_scores=[0.3, 0.8],
    )
    sources = results.to_source_documents()
    assert sources[0].semantic_score == 0.95
    assert sources[0].bm25_score == 0.3
    assert sources[1].semantic_score == 0.6
    assert sources[1].bm25_score == 0.8


def test_to_source_documents_without_breakdown_data_leaves_scores_none():
    # An old-style dict response, or a pure-semantic search with no BM25
    # component — the breakdown just isn't available.
    results = SearchResults(
        documents=["chunk one"],
        metadatas=[{"file_name": "a.pdf"}],
        scores=[0.9],
    )
    sources = results.to_source_documents()
    assert sources[0].semantic_score is None
    assert sources[0].bm25_score is None


def test_to_source_documents_merged_score_is_always_populated_regardless():
    # The breakdown being unavailable must never affect the primary score
    # every existing caller already depends on.
    results = SearchResults(
        documents=["chunk one"], metadatas=[{"file_name": "a.pdf"}], scores=[0.75],
    )
    sources = results.to_source_documents()
    assert sources[0].score == 0.75


def test_to_source_documents_handles_a_shorter_breakdown_list_gracefully():
    # Defensive: if semantic_scores/bm25_scores ever come back shorter
    # than documents (a partial computation), later documents just get
    # None rather than an IndexError.
    results = SearchResults(
        documents=["a", "b", "c"],
        metadatas=[{}, {}, {}],
        scores=[0.9, 0.8, 0.7],
        semantic_scores=[0.9],  # only one entry for three documents
        bm25_scores=[0.9, 0.8],  # only two entries
    )
    sources = results.to_source_documents()
    assert sources[0].semantic_score == 0.9
    assert sources[1].semantic_score is None
    assert sources[1].bm25_score == 0.8
    assert sources[2].semantic_score is None
    assert sources[2].bm25_score is None


# ---------------------------------------------------------------------------
# End-to-end through AthenaEngine._handle_search (real chunks, fake LLM)
# ---------------------------------------------------------------------------

def _write_and_ingest(engine, tmp, text):
    path = os.path.join(tmp, "documents", "Bio", "notes.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    engine.handle("ingest", {}, {})


def test_search_without_debug_flag_omits_score_breakdown():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        _write_and_ingest(engine, tmp, "The mitochondria is the powerhouse of the cell and produces ATP.")
        result = engine.handle("search", {"query": "mitochondria"}, {})
        assert "metrics" not in result["data"]
        for source in result["data"]["sources"]:
            assert "semantic_score" not in source
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_search_with_debug_flag_includes_score_breakdown():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        _write_and_ingest(engine, tmp, "The mitochondria is the powerhouse of the cell and produces ATP.")
        result = engine.handle("search", {"query": "mitochondria", "debug": True}, {})
        assert "metrics" in result["data"]
        assert result["data"]["sources"]
        for source in result["data"]["sources"]:
            assert "semantic_score" in source
            assert "bm25_score" in source
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_search_with_show_scores_flag_also_works():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        _write_and_ingest(engine, tmp, "The mitochondria is the powerhouse of the cell and produces ATP.")
        result = engine.handle("search", {"query": "mitochondria", "show_scores": True}, {})
        assert result["data"]["sources"]
        assert "semantic_score" in result["data"]["sources"][0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
