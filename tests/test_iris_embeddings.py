# tests/test_iris_embeddings.py
"""
Extensive regression tests for Iris's CLIP-based semantic image search:

  - modules/iris/embeddings.py    (ClipEmbedder, ImageVectorIndex)
  - modules/iris/analyser.py      (IrisAnalyser._embed_and_index integration)
  - modules/iris/iris_engine.py   (IrisEngine._semantic_matches / search() merge)

Design of these tests
----------------------
`ImageVectorIndex` is tested against a REAL, locally-persisted ChromaDB
collection (no mocking of chromadb itself) so the vector round-trip is
genuinely verified. `ClipEmbedder` is tested with a fake
`sentence_transformers.SentenceTransformer` injected into `sys.modules`
rather than the real model — loading the real CLIP model requires
downloading weights from huggingface.co, which is a real network
dependency this test suite has no business requiring. The fake still
exercises every real code path in ClipEmbedder (lazy loading, caching,
failure handling, image mode conversion, vector shape).

Run with:  pytest tests/test_iris_embeddings.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.iris.embeddings import ClipEmbedder, ImageVectorIndex
from modules.iris.analyser import IrisAnalyser
from modules.iris.db import IrisDB
from modules.iris.iris_engine import IrisEngine


# ===========================================================================
# Fakes / fixtures
# ===========================================================================

class FakeSentenceTransformerModel:
    """
    Fake CLIP model: deterministic 4-dim embeddings derived from a hash of
    the input, so equal inputs give equal vectors and different inputs
    give (almost certainly) different vectors — good enough to exercise
    nearest-neighbour behaviour in ImageVectorIndex without needing a real
    CLIP model.
    """
    def __init__(self, name):
        self.name = name
        self.encode_calls = []

    def encode(self, item, convert_to_numpy=True):
        import numpy as np
        self.encode_calls.append(item)
        if isinstance(item, str):
            seed = sum(ord(c) for c in item)
        else:
            # PIL Image or similar — use its repr/size as a stable seed.
            seed = hash(getattr(item, "size", str(item))) % 100000
        rng = np.random.RandomState(seed % (2**31))
        return rng.rand(4).astype(float)


@pytest.fixture()
def fake_sentence_transformers(monkeypatch):
    """Install a fake `sentence_transformers` module so ClipEmbedder's
    lazy `from sentence_transformers import SentenceTransformer` succeeds
    without needing the real package or model weights."""
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformerModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    yield fake_module


@pytest.fixture()
def broken_sentence_transformers(monkeypatch):
    """Install a `sentence_transformers` module whose SentenceTransformer
    raises on construction, simulating a missing/corrupt model download."""
    fake_module = types.ModuleType("sentence_transformers")

    class _Boom:
        def __init__(self, *a, **k):
            raise OSError("no internet, model weights not cached")

    fake_module.SentenceTransformer = _Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    yield fake_module


@pytest.fixture()
def chroma_dir():
    d = tempfile.mkdtemp(prefix="iris_chroma_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def sample_image(tmp_path):
    """A real small RGBA PNG on disk, to exercise embed_image()'s RGBA->RGB
    conversion path and actual file I/O."""
    from PIL import Image
    path = tmp_path / "sample.png"
    img = Image.new("RGBA", (16, 16), color=(255, 0, 0, 128))
    img.save(path, format="PNG")
    return path


@pytest.fixture()
def iris_db(tmp_path):
    db = IrisDB(str(tmp_path / "iris.db"))
    yield db


# ===========================================================================
# ClipEmbedder
# ===========================================================================

class TestClipEmbedderLazyLoading:
    def test_not_available_before_first_use(self):
        embedder = ClipEmbedder()
        assert embedder.available is False

    def test_available_becomes_true_after_successful_embed(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        embedder.embed_text("a dog on a beach")
        assert embedder.available is True

    def test_model_loaded_lazily_not_at_construction(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        assert embedder._model is None  # not loaded yet
        embedder.embed_text("hello")
        assert embedder._model is not None

    def test_model_loaded_only_once_across_multiple_calls(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        embedder.embed_text("first query")
        first_model = embedder._model
        embedder.embed_text("second query")
        assert embedder._model is first_model  # same instance, not reloaded


class TestClipEmbedderTextEmbedding:
    def test_embed_text_returns_vector_of_floats(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        vector = embedder.embed_text("a cat sitting on a windowsill")
        assert isinstance(vector, list)
        assert len(vector) == 4
        assert all(isinstance(v, float) for v in vector)

    def test_embed_text_none_for_empty_string(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        assert embedder.embed_text("") is None

    def test_embed_text_none_for_whitespace_only(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        assert embedder.embed_text("   \n\t  ") is None

    def test_embed_text_does_not_trigger_model_load_for_empty_input(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        embedder.embed_text("")
        assert embedder._model is None  # short-circuited before _ensure_loaded

    def test_same_text_yields_identical_vector(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        v1 = embedder.embed_text("mountains at sunset")
        v2 = embedder.embed_text("mountains at sunset")
        assert v1 == v2

    def test_different_text_yields_different_vector(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        v1 = embedder.embed_text("a red bicycle")
        v2 = embedder.embed_text("a blue whale")
        assert v1 != v2


class TestClipEmbedderImageEmbedding:
    def test_embed_image_returns_vector(self, fake_sentence_transformers, sample_image):
        embedder = ClipEmbedder()
        vector = embedder.embed_image(sample_image)
        assert isinstance(vector, list)
        assert len(vector) == 4

    def test_embed_image_converts_rgba_to_rgb_before_encoding(self, fake_sentence_transformers, sample_image):
        embedder = ClipEmbedder()
        embedder.embed_image(sample_image)
        model = embedder._model
        assert len(model.encode_calls) == 1
        encoded_img = model.encode_calls[0]
        assert encoded_img.mode == "RGB"

    def test_embed_image_returns_none_for_nonexistent_path(self, fake_sentence_transformers, tmp_path):
        embedder = ClipEmbedder()
        vector = embedder.embed_image(tmp_path / "does_not_exist.png")
        assert vector is None

    def test_embed_image_accepts_string_path(self, fake_sentence_transformers, sample_image):
        embedder = ClipEmbedder()
        vector = embedder.embed_image(str(sample_image))
        assert vector is not None


class TestClipEmbedderFailureHandling:
    def test_model_load_failure_disables_embedder_permanently(self, broken_sentence_transformers):
        embedder = ClipEmbedder()
        assert embedder.embed_text("hello") is None
        assert embedder.available is False
        assert embedder._load_failed is True

    def test_after_load_failure_no_repeated_import_attempts(self, broken_sentence_transformers, monkeypatch):
        embedder = ClipEmbedder()
        embedder.embed_text("first")
        # Swap in a working module — if the embedder retried the import on
        # every call it would now succeed; it must not, because
        # _load_failed latches the failure to avoid hammering a broken
        # model load on every single query/embed call.
        working_module = types.ModuleType("sentence_transformers")
        working_module.SentenceTransformer = FakeSentenceTransformerModel
        monkeypatch.setitem(sys.modules, "sentence_transformers", working_module)
        result = embedder.embed_text("second")
        assert result is None
        assert embedder._model is None

    def test_embed_image_exception_during_encode_returns_none(self, fake_sentence_transformers, sample_image):
        embedder = ClipEmbedder()
        embedder._ensure_loaded()
        embedder._model.encode = MagicMock(side_effect=RuntimeError("GPU OOM"))
        assert embedder.embed_image(sample_image) is None

    def test_embed_text_exception_during_encode_returns_none(self, fake_sentence_transformers):
        embedder = ClipEmbedder()
        embedder._ensure_loaded()
        embedder._model.encode = MagicMock(side_effect=RuntimeError("boom"))
        assert embedder.embed_text("hello") is None

    def test_custom_model_name_is_respected(self, fake_sentence_transformers):
        embedder = ClipEmbedder(model_name="clip-ViT-B-16")
        embedder.embed_text("x")
        assert embedder._model.name == "clip-ViT-B-16"


# ===========================================================================
# ImageVectorIndex (real ChromaDB)
# ===========================================================================

class TestImageVectorIndexLifecycle:
    def test_not_available_until_first_access(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        assert index._collection is None

    def test_available_true_with_real_chroma_backend(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        assert index.available is True

    def test_creates_chroma_dir_if_missing(self, tmp_path):
        target = tmp_path / "nested" / "chroma"
        assert not target.exists()
        index = ImageVectorIndex(target)
        assert index.available is True
        assert target.exists()

    def test_client_and_collection_created_only_once(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        assert index.available is True
        first_client = index._client
        assert index.available is True
        assert index._client is first_client


class TestImageVectorIndexUpsertQueryDelete:
    def test_upsert_then_query_round_trip(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        assert index.upsert(1, [0.1, 0.2, 0.3, 0.4]) is True
        assert index.upsert(2, [0.9, 0.8, 0.7, 0.6]) is True

        hits = index.query([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert len(hits) == 2
        # The nearest neighbour to the exact vector for file_id=1 must be
        # file_id=1 itself, at (near-)zero distance.
        nearest_id, nearest_dist = hits[0]
        assert nearest_id == 1
        assert nearest_dist == pytest.approx(0.0, abs=1e-6)

    def test_upsert_overwrites_existing_embedding_for_same_file_id(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        index.upsert(42, [0.0, 0.0, 0.0, 0.0])
        index.upsert(42, [1.0, 1.0, 1.0, 1.0])

        hits = index.query([1.0, 1.0, 1.0, 1.0], top_k=5)
        assert len(hits) == 1  # not duplicated
        assert hits[0][0] == 42
        assert hits[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_query_on_empty_collection_returns_empty_list(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        assert index.query([0.1, 0.2, 0.3, 0.4]) == []

    def test_query_top_k_is_clamped_to_collection_size(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        index.upsert(1, [0.1, 0.1, 0.1, 0.1])
        index.upsert(2, [0.2, 0.2, 0.2, 0.2])
        hits = index.query([0.1, 0.1, 0.1, 0.1], top_k=1000)
        assert len(hits) == 2  # never more than what's actually stored

    def test_delete_removes_entry(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        index.upsert(7, [0.5, 0.5, 0.5, 0.5])
        index.delete(7)
        assert index.query([0.5, 0.5, 0.5, 0.5]) == []

    def test_delete_on_missing_id_does_not_raise(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        index.delete(999)  # never inserted — must be a silent no-op

    def test_persists_across_new_index_instance_same_dir(self, chroma_dir):
        index1 = ImageVectorIndex(chroma_dir)
        index1.upsert(5, [0.3, 0.3, 0.3, 0.3])

        index2 = ImageVectorIndex(chroma_dir)  # fresh instance, same directory
        hits = index2.query([0.3, 0.3, 0.3, 0.3])
        assert len(hits) == 1
        assert hits[0][0] == 5


class TestImageVectorIndexFailureHandling:
    def test_init_failure_disables_index_permanently(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        with patch("modules.iris.embeddings.chromadb.PersistentClient",
                   side_effect=RuntimeError("disk full")):
            assert index.available is False
        assert index._init_failed is True
        # Subsequent calls must not raise, just no-op.
        assert index.upsert(1, [0.1, 0.2, 0.3, 0.4]) is False
        assert index.query([0.1, 0.2, 0.3, 0.4]) == []

    def test_upsert_exception_returns_false_not_raise(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        index.available  # force init
        index._collection.upsert = MagicMock(side_effect=RuntimeError("boom"))
        assert index.upsert(1, [0.1, 0.2, 0.3, 0.4]) is False

    def test_query_exception_returns_empty_list_not_raise(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        index.upsert(1, [0.1, 0.2, 0.3, 0.4])
        index._collection.query = MagicMock(side_effect=RuntimeError("boom"))
        assert index.query([0.1, 0.2, 0.3, 0.4]) == []

    def test_delete_exception_does_not_raise(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        index.upsert(1, [0.1, 0.2, 0.3, 0.4])
        index._collection.delete = MagicMock(side_effect=RuntimeError("boom"))
        index.delete(1)  # must swallow the exception silently

    def test_query_skips_malformed_ids_defensively(self, chroma_dir):
        index = ImageVectorIndex(chroma_dir)
        index.upsert(1, [0.1, 0.2, 0.3, 0.4])
        index._ensure_client()
        index._collection.query = MagicMock(return_value={
            "ids": [["not_an_int", "2"]],
            "distances": [[0.1, 0.2]],
        })
        hits = index.query([0.1, 0.2, 0.3, 0.4])
        assert hits == [(2, 0.2)]


# ===========================================================================
# IrisAnalyser integration (_embed_and_index)
# ===========================================================================

class TestAnalyserEmbedIntegration:
    def test_embed_and_index_noop_when_embedder_and_index_absent(self, iris_db, sample_image):
        analyser = IrisAnalyser(iris_db, "127.0.0.1", 11434, embedder=None, vector_index=None)
        # Must not raise even though both are None.
        analyser._embed_and_index(1, sample_image)

    def test_embed_and_index_stores_real_embedding(
        self, fake_sentence_transformers, chroma_dir, iris_db, sample_image
    ):
        embedder = ClipEmbedder()
        index = ImageVectorIndex(chroma_dir)
        analyser = IrisAnalyser(
            iris_db, "127.0.0.1", 11434, embedder=embedder, vector_index=index
        )
        analyser._embed_and_index(1, sample_image)

        hits = index.query(embedder.embed_text("irrelevant probe query"), top_k=5)
        # We can't easily predict the exact vector; assert *something* got
        # stored for file_id=1 by checking count directly instead.
        assert index._collection.count() == 1

    def test_embed_and_index_never_raises_on_embedder_exception(self, chroma_dir, iris_db, sample_image):
        broken_embedder = MagicMock()
        broken_embedder.embed_image = MagicMock(side_effect=RuntimeError("boom"))
        index = ImageVectorIndex(chroma_dir)
        analyser = IrisAnalyser(iris_db, "127.0.0.1", 11434, embedder=broken_embedder, vector_index=index)
        analyser._embed_and_index(1, sample_image)  # must not propagate

    def test_embed_and_index_skips_upsert_when_embedding_is_none(self, chroma_dir, iris_db, sample_image):
        empty_embedder = MagicMock()
        empty_embedder.embed_image = MagicMock(return_value=None)
        index = ImageVectorIndex(chroma_dir)
        analyser = IrisAnalyser(iris_db, "127.0.0.1", 11434, embedder=empty_embedder, vector_index=index)
        analyser._embed_and_index(1, sample_image)
        assert index._collection.count() == 0

    def test_analyse_file_runs_embedding_step_independent_of_ollama_result(
        self, fake_sentence_transformers, chroma_dir, iris_db, sample_image
    ):
        """Embedding must run even if the Ollama vision-caption call fails —
        the two are documented as independent best-effort steps."""
        file_id = iris_db.insert_file(
            str(sample_image), "hash1", "phash1", 123, "image", "image/png"
        )
        embedder = ClipEmbedder()
        index = ImageVectorIndex(chroma_dir)
        analyser = IrisAnalyser(
            iris_db, "127.0.0.1", 11434, embedder=embedder, vector_index=index
        )
        with patch.object(analyser, "_send_to_ollama", return_value=""):
            result = analyser.analyse_file(file_id)

        assert result is False  # captioning failed (empty Ollama response)
        assert index._collection.count() == 1  # but embedding still happened


# ===========================================================================
# IrisEngine semantic search integration
# ===========================================================================

def _make_iris_engine(tmp_path, embedder=None, vector_index=None):
    import modules.iris.iris_engine as iris_engine_mod
    from modules.iris.config import IrisConfig

    cfg = IrisConfig(
        db_path=str(tmp_path / "iris.db"),
        source_dir=str(tmp_path / "source"),
        output_dir=str(tmp_path / "output"),
        cache_dir=str(tmp_path / "cache"),
        chroma_dir=str(tmp_path / "chroma"),
    )
    with patch.object(iris_engine_mod, "get_config", return_value=cfg):
        engine = IrisEngine(embedder=embedder, vector_index=vector_index)
    return engine


class TestIrisEngineSemanticSearch:
    def test_semantic_matches_returns_empty_when_embedder_unavailable(self, tmp_path, chroma_dir):
        embedder = ClipEmbedder()  # sentence_transformers not faked -> unavailable
        index = ImageVectorIndex(chroma_dir)
        engine = _make_iris_engine(tmp_path, embedder=embedder, vector_index=index)
        assert engine._semantic_matches("a dog on a beach", 10) == []

    def test_semantic_matches_returns_db_records_for_vector_hits(
        self, fake_sentence_transformers, tmp_path, chroma_dir
    ):
        embedder = ClipEmbedder()
        index = ImageVectorIndex(chroma_dir)
        engine = _make_iris_engine(tmp_path, embedder=embedder, vector_index=index)

        file_id = engine.db.insert_file("/photos/dog.jpg", "h1", "p1", 100, "image", "image/jpeg")
        engine.db.update_file_analysis(file_id, "A dog running", '["dog"]', None, "happy", False, None)
        vector = embedder.embed_image(Path(__file__))  # any deterministic vector
        index.upsert(file_id, vector)

        matches = engine._semantic_matches("some query text", 10)
        # embed_text("some query text") won't equal embed_image's vector
        # exactly, but with only one point in the index it must still be
        # returned as the (only) nearest neighbour.
        assert len(matches) == 1
        assert matches[0]["id"] == file_id

    def test_semantic_matches_exception_degrades_to_empty_list(self, fake_sentence_transformers, tmp_path, chroma_dir):
        embedder = ClipEmbedder()
        index = ImageVectorIndex(chroma_dir)
        engine = _make_iris_engine(tmp_path, embedder=embedder, vector_index=index)
        with patch.object(index, "query", side_effect=RuntimeError("boom")):
            assert engine._semantic_matches("query", 10) == []

    def test_search_merges_semantic_caption_and_tag_hits_deduped(
        self, fake_sentence_transformers, tmp_path, chroma_dir
    ):
        embedder = ClipEmbedder()
        index = ImageVectorIndex(chroma_dir)
        engine = _make_iris_engine(tmp_path, embedder=embedder, vector_index=index)

        # File A: only matches via semantic index (caption/tags unrelated).
        id_a = engine.db.insert_file("/photos/a.jpg", "ha", "pa", 10, "image", "image/jpeg")
        engine.db.update_file_analysis(id_a, "unrelated caption", '["unrelated"]', None, "neutral", False, None)
        engine.db.mark_file_processed(id_a)
        index.upsert(id_a, embedder.embed_text("beach vacation photo"))

        # File B: matches via caption text search only (not embedded).
        id_b = engine.db.insert_file("/photos/b.jpg", "hb", "pb", 10, "image", "image/jpeg")
        engine.db.update_file_analysis(id_b, "a beach at sunset", '["sunset"]', None, "calm", False, None)
        engine.db.mark_file_processed(id_b)

        # File C: matches via tag search only.
        id_c = engine.db.insert_file("/photos/c.jpg", "hc", "pc", 10, "image", "image/jpeg")
        engine.db.update_file_analysis(id_c, "totally unrelated", '["beach"]', None, "neutral", False, None)
        engine.db.mark_file_processed(id_c)

        result_text = engine.search("beach", limit=10)
        assert result_text is not None
        assert "Found 3 photos" in result_text
        for path in ("/photos/a.jpg", "/photos/b.jpg", "/photos/c.jpg"):
            assert path in result_text

    def test_search_deduplicates_file_matched_by_multiple_signals(
        self, fake_sentence_transformers, tmp_path, chroma_dir
    ):
        embedder = ClipEmbedder()
        index = ImageVectorIndex(chroma_dir)
        engine = _make_iris_engine(tmp_path, embedder=embedder, vector_index=index)

        file_id = engine.db.insert_file("/photos/beach.jpg", "h1", "p1", 10, "image", "image/jpeg")
        engine.db.update_file_analysis(file_id, "a beach scene", '["beach"]', None, "calm", False, None)
        engine.db.mark_file_processed(file_id)
        index.upsert(file_id, embedder.embed_text("beach"))

        result_text = engine.search("beach", limit=10)
        assert result_text.count("/photos/beach.jpg") == 1  # not repeated 3x
        assert "Found 1 photos" in result_text

    def test_search_returns_none_when_nothing_matches(self, fake_sentence_transformers, tmp_path, chroma_dir):
        embedder = ClipEmbedder()
        index = ImageVectorIndex(chroma_dir)
        engine = _make_iris_engine(tmp_path, embedder=embedder, vector_index=index)
        assert engine.search("nonexistent query xyz") is None

    def test_search_gracefully_falls_back_when_embedder_totally_unavailable(self, tmp_path, chroma_dir):
        """No sentence_transformers faked in at all — embedder stays
        unavailable, but caption/tag search must still work."""
        embedder = ClipEmbedder()
        index = ImageVectorIndex(chroma_dir)
        engine = _make_iris_engine(tmp_path, embedder=embedder, vector_index=index)

        file_id = engine.db.insert_file("/photos/x.jpg", "h1", "p1", 10, "image", "image/jpeg")
        engine.db.update_file_analysis(file_id, "a mountain view", '["mountain"]', None, "calm", False, None)
        engine.db.mark_file_processed(file_id)

        result_text = engine.search("mountain")
        assert result_text is not None
        assert "/photos/x.jpg" in result_text

    def test_iris_engine_constructs_with_default_real_embedder_and_index(self, tmp_path):
        """Without explicit injection, IrisEngine must wire up real
        ClipEmbedder/ImageVectorIndex instances (not None)."""
        engine = _make_iris_engine(tmp_path)
        assert isinstance(engine.embedder, ClipEmbedder)
        assert isinstance(engine.vector_index, ImageVectorIndex)
        assert engine.analyser.embedder is engine.embedder
        assert engine.analyser.vector_index is engine.vector_index


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))