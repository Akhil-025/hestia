# tests/test_athena_feedback.py
"""
Tests for per-chunk relevance feedback and down-weighting (backlog #63):
core/chunk_feedback_store.py and MergedLocalRAG.mark_feedback /
_apply_feedback_weighting.
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from core.chunk_feedback_store import ChunkFeedbackStore
from test_athena import _ensure_stubs, make_engine  # noqa: E402
_ensure_stubs()

from modules.athena.local_rag import SearchResponse, SearchResult, _chunk_id_from_metadata  # noqa: E402


# ---------------------------------------------------------------------------
# ChunkFeedbackStore
# ---------------------------------------------------------------------------

def test_new_chunk_has_zero_counts(tmp_path):
    store = ChunkFeedbackStore(str(tmp_path / "fb.db"))
    assert store.get("unknown-chunk") == {"relevant_count": 0, "irrelevant_count": 0}


def test_record_relevant_increments_relevant_count(tmp_path):
    store = ChunkFeedbackStore(str(tmp_path / "fb.db"))
    store.record("chunk-1", relevant=True)
    store.record("chunk-1", relevant=True)
    assert store.get("chunk-1") == {"relevant_count": 2, "irrelevant_count": 0}


def test_record_irrelevant_increments_irrelevant_count(tmp_path):
    store = ChunkFeedbackStore(str(tmp_path / "fb.db"))
    store.record("chunk-1", relevant=False)
    assert store.get("chunk-1") == {"relevant_count": 0, "irrelevant_count": 1}


def test_counts_are_independent_per_chunk(tmp_path):
    store = ChunkFeedbackStore(str(tmp_path / "fb.db"))
    store.record("chunk-a", relevant=False)
    store.record("chunk-b", relevant=True)
    assert store.get("chunk-a")["irrelevant_count"] == 1
    assert store.get("chunk-b")["relevant_count"] == 1


def test_get_all_returns_every_recorded_chunk(tmp_path):
    store = ChunkFeedbackStore(str(tmp_path / "fb.db"))
    store.record("chunk-a", relevant=False)
    store.record("chunk-b", relevant=True)
    all_fb = store.get_all()
    assert set(all_fb) == {"chunk-a", "chunk-b"}


def test_get_all_empty_when_nothing_recorded(tmp_path):
    store = ChunkFeedbackStore(str(tmp_path / "fb.db"))
    assert store.get_all() == {}


# ---------------------------------------------------------------------------
# _chunk_id_from_metadata matches the real ingestion-time id format
# ---------------------------------------------------------------------------

def test_chunk_id_from_metadata_matches_ingestion_format():
    from modules.athena.local_rag import _chunk_id
    file_info = {"full_path": "/docs/Bio/notes.pdf", "subject": "Bio", "module": "cells"}
    chunk = {"page_number": 3, "chunk_number": 2}
    ingestion_id = _chunk_id(file_info, chunk)

    metadata = {"subject": "Bio", "module": "cells", "file_name": "notes.pdf", "page_number": 3, "chunk_number": 2}
    search_id = _chunk_id_from_metadata(metadata)

    assert ingestion_id == search_id


# ---------------------------------------------------------------------------
# Down-weighting adjusts scores and re-sorts
# ---------------------------------------------------------------------------

def _result(subject, module, file_name, page, chunk, score):
    return SearchResult(
        document="text",
        metadata={"subject": subject, "module": module, "file_name": file_name, "page_number": page, "chunk_number": chunk},
        score=score,
        semantic_score=score,
    )


def make_rag(tmp):
    engine = make_engine(tmp)
    return engine.rag


def test_no_feedback_leaves_scores_and_order_unchanged():
    tmp = tempfile.mkdtemp()
    try:
        rag = make_rag(tmp)
        r1 = _result("Bio", "cells", "a.txt", 1, 1, 0.9)
        r2 = _result("Bio", "cells", "b.txt", 1, 1, 0.7)
        response = SearchResponse(results=[r1, r2], query="q")
        adjusted = rag._apply_feedback_weighting(response)
        assert [r.score for r in adjusted.results] == [0.9, 0.7]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_marked_irrelevant_chunk_is_down_weighted():
    tmp = tempfile.mkdtemp()
    try:
        rag = make_rag(tmp)
        metadata = {"subject": "Bio", "module": "cells", "file_name": "a.txt", "page_number": 1, "chunk_number": 1}
        rag.mark_feedback(metadata, relevant=False)

        r1 = _result("Bio", "cells", "a.txt", 1, 1, 0.9)  # the marked chunk
        r2 = _result("Bio", "cells", "b.txt", 1, 1, 0.5)
        response = SearchResponse(results=[r1, r2], query="q")
        adjusted = rag._apply_feedback_weighting(response)

        marked = next(r for r in adjusted.results if r.metadata["file_name"] == "a.txt")
        assert marked.score < 0.9
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_down_weighting_never_zeroes_the_score_out():
    tmp = tempfile.mkdtemp()
    try:
        rag = make_rag(tmp)
        metadata = {"subject": "Bio", "module": "cells", "file_name": "a.txt", "page_number": 1, "chunk_number": 1}
        for _ in range(20):  # a lot of negative feedback
            rag.mark_feedback(metadata, relevant=False)

        r1 = _result("Bio", "cells", "a.txt", 1, 1, 1.0)
        response = SearchResponse(results=[r1], query="q")
        adjusted = rag._apply_feedback_weighting(response)
        assert adjusted.results[0].score >= 0.1 * 1.0  # floors at 10%, never zero
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_down_weighting_reorders_results_when_it_changes_ranking():
    tmp = tempfile.mkdtemp()
    try:
        rag = make_rag(tmp)
        metadata = {"subject": "Bio", "module": "cells", "file_name": "a.txt", "page_number": 1, "chunk_number": 1}
        for _ in range(5):
            rag.mark_feedback(metadata, relevant=False)  # floors a.txt at 10%

        r1 = _result("Bio", "cells", "a.txt", 1, 1, 0.9)  # was top, now demoted
        r2 = _result("Bio", "cells", "b.txt", 1, 1, 0.5)
        response = SearchResponse(results=[r1, r2], query="q")
        adjusted = rag._apply_feedback_weighting(response)

        assert adjusted.results[0].metadata["file_name"] == "b.txt"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_relevant_feedback_does_not_affect_score():
    tmp = tempfile.mkdtemp()
    try:
        rag = make_rag(tmp)
        metadata = {"subject": "Bio", "module": "cells", "file_name": "a.txt", "page_number": 1, "chunk_number": 1}
        rag.mark_feedback(metadata, relevant=True)

        r1 = _result("Bio", "cells", "a.txt", 1, 1, 0.9)
        response = SearchResponse(results=[r1], query="q")
        adjusted = rag._apply_feedback_weighting(response)
        assert adjusted.results[0].score == 0.9
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_apply_feedback_weighting_on_empty_results_does_not_raise():
    tmp = tempfile.mkdtemp()
    try:
        rag = make_rag(tmp)
        response = SearchResponse(results=[], query="q")
        adjusted = rag._apply_feedback_weighting(response)
        assert adjusted.results == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# End-to-end via AthenaEngine
# ---------------------------------------------------------------------------

def test_mark_feedback_intent_records_and_confirms():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle("mark_feedback", {
            "file_name": "a.txt", "subject": "Bio", "module": "cells",
            "page_number": 1, "chunk_number": 1, "relevant": False,
        }, {})
        assert "not relevant" in result["response"]
        counts = engine.rag._feedback.get(
            "Bio::cells::a.txt::p1::c1"
        )
        assert counts["irrelevant_count"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mark_feedback_intent_missing_fields_asks_for_them():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle("mark_feedback", {"file_name": "a.txt"}, {})
        assert result["confidence"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mark_feedback_dispatchable_prefixed_and_stripped():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        for intent in ("mark_feedback", "athena_mark_feedback"):
            assert engine.can_handle(intent)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
