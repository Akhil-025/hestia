# tests/test_iris_albums_and_compare.py
"""
Tests for album/collection auto-organization via embedding clustering
(backlog #79) and "describe what changed" photo comparison (backlog #76).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.iris.db import IrisDB
from modules.iris.iris_engine import IrisEngine


def make_db(tmp_path):
    return IrisDB(str(tmp_path / "iris.db"))


def make_bare_engine(tmp_path):
    engine = IrisEngine.__new__(IrisEngine)
    engine.db = make_db(tmp_path)
    return engine


# ---------------------------------------------------------------------------
# _cluster_by_distance (pure, static — no DB/vector store needed)
# ---------------------------------------------------------------------------

def test_identical_vectors_cluster_together():
    embeddings = [(1, [1.0, 0.0]), (2, [1.0, 0.0]), (3, [1.0, 0.0])]
    clusters = IrisEngine._cluster_by_distance(embeddings, threshold=0.1)
    assert len(clusters) == 1
    assert set(clusters[0]) == {1, 2, 3}


def test_orthogonal_vectors_do_not_cluster():
    embeddings = [(1, [1.0, 0.0]), (2, [0.0, 1.0])]
    clusters = IrisEngine._cluster_by_distance(embeddings, threshold=0.1)
    assert len(clusters) == 2


def test_two_separate_tight_clusters():
    embeddings = [
        (1, [1.0, 0.0]), (2, [0.99, 0.01]),
        (3, [0.0, 1.0]), (4, [0.01, 0.99]),
    ]
    clusters = IrisEngine._cluster_by_distance(embeddings, threshold=0.05)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 2]


def test_single_embedding_is_its_own_cluster():
    embeddings = [(1, [1.0, 0.0])]
    clusters = IrisEngine._cluster_by_distance(embeddings, threshold=0.1)
    assert clusters == [[1]]


def test_empty_embeddings_returns_no_clusters():
    assert IrisEngine._cluster_by_distance([], threshold=0.1) == []


def test_zero_vector_does_not_crash_distance_calculation():
    embeddings = [(1, [0.0, 0.0]), (2, [1.0, 0.0])]
    clusters = IrisEngine._cluster_by_distance(embeddings, threshold=0.1)
    assert len(clusters) == 2  # a zero vector has undefined direction — no false match


def test_tighter_threshold_produces_more_clusters():
    embeddings = [(1, [1.0, 0.0]), (2, [0.9, 0.1]), (3, [0.8, 0.2])]
    loose = IrisEngine._cluster_by_distance(embeddings, threshold=0.5)
    tight = IrisEngine._cluster_by_distance(embeddings, threshold=0.001)
    assert len(loose) <= len(tight)


# ---------------------------------------------------------------------------
# organize_into_albums (fake vector index, real DB)
# ---------------------------------------------------------------------------

class _FakeVectorIndex:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def get_all_embeddings(self):
        return self._embeddings


def insert(db, path, file_hash):
    return db.insert_file(path, file_hash, perceptual_hash=f"p{file_hash}", file_size=100, file_type="image", mime_type="image/jpeg")


def test_organize_into_albums_creates_events_for_large_enough_clusters(tmp_path):
    engine = make_bare_engine(tmp_path)
    ids = [insert(engine.db, f"{i}.jpg", f"h{i}") for i in range(4)]
    embeddings = [(fid, [1.0, 0.0]) for fid in ids]  # all identical — one cluster
    engine.vector_index = _FakeVectorIndex(embeddings)

    albums = engine.organize_into_albums()
    assert len(albums) == 1
    assert set(albums[0]["files"]) == set(ids)


def test_organize_into_albums_skips_clusters_below_minimum_size(tmp_path):
    engine = make_bare_engine(tmp_path)
    ids = [insert(engine.db, f"{i}.jpg", f"h{i}") for i in range(2)]  # below _ALBUM_MIN_SIZE (3)
    embeddings = [(fid, [1.0, 0.0]) for fid in ids]
    engine.vector_index = _FakeVectorIndex(embeddings)

    albums = engine.organize_into_albums()
    assert albums == []


def test_organize_into_albums_with_too_few_total_embeddings_returns_empty(tmp_path):
    engine = make_bare_engine(tmp_path)
    engine.vector_index = _FakeVectorIndex([(1, [1.0, 0.0])])
    assert engine.organize_into_albums() == []


def test_organize_into_albums_persists_to_the_events_table(tmp_path):
    engine = make_bare_engine(tmp_path)
    ids = [insert(engine.db, f"{i}.jpg", f"h{i}") for i in range(4)]
    embeddings = [(fid, [1.0, 0.0]) for fid in ids]
    engine.vector_index = _FakeVectorIndex(embeddings)

    engine.organize_into_albums()
    events = engine.db.get_events()
    assert len(events) == 1
    files_in_event = engine.db.get_files_in_event(events[0]["id"])
    assert len(files_in_event) == 4


def test_organize_into_albums_reclusters_from_scratch_each_call(tmp_path):
    engine = make_bare_engine(tmp_path)
    ids = [insert(engine.db, f"{i}.jpg", f"h{i}") for i in range(4)]
    embeddings = [(fid, [1.0, 0.0]) for fid in ids]
    engine.vector_index = _FakeVectorIndex(embeddings)

    engine.organize_into_albums()
    first_count = len(engine.db.get_events())
    engine.organize_into_albums()
    second_count = len(engine.db.get_events())
    # Old albums are cleared, not accumulated — a second identical run
    # produces one album again, not two.
    assert first_count == second_count == 1


def test_organize_albums_intent_end_to_end(tmp_path):
    engine = make_bare_engine(tmp_path)
    ids = [insert(engine.db, f"{i}.jpg", f"h{i}") for i in range(4)]
    embeddings = [(fid, [1.0, 0.0]) for fid in ids]
    engine.vector_index = _FakeVectorIndex(embeddings)

    result = engine.handle("organize_albums", {}, {})
    assert "1 album" in result["response"]
    assert len(result["data"]["albums"]) == 1


def test_organize_albums_intent_with_nothing_to_cluster(tmp_path):
    engine = make_bare_engine(tmp_path)
    engine.vector_index = _FakeVectorIndex([])
    result = engine.handle("organize_albums", {}, {})
    assert "not enough" in result["response"].lower()


# ---------------------------------------------------------------------------
# describe_change (#76) — fake analyser, real DB
# ---------------------------------------------------------------------------

class _FakeAnalyser:
    def __init__(self, response="The lighting changed from day to night."):
        self.response = response
        self.calls = []

    def _send_to_ollama(self, images, prompt):
        self.calls.append((images, prompt))
        return self.response


def test_describe_change_calls_the_vision_model_with_both_images(tmp_path, tmp_path_factory):
    engine = make_bare_engine(tmp_path)
    engine.analyser = _FakeAnalyser()

    dir_a = tmp_path_factory.mktemp("imgs")
    from PIL import Image
    path_a = dir_a / "a.jpg"
    path_b = dir_a / "b.jpg"
    Image.new("RGB", (10, 10), "red").save(path_a)
    Image.new("RGB", (10, 10), "blue").save(path_b)

    id_a = insert(engine.db, str(path_a), "ha")
    id_b = insert(engine.db, str(path_b), "hb")

    result = engine.describe_change(id_a, id_b)
    assert result["description"] == "The lighting changed from day to night."
    assert len(engine.analyser.calls[0][0]) == 2  # both images passed together


def test_describe_change_with_missing_file_ids_reports_not_found(tmp_path):
    engine = make_bare_engine(tmp_path)
    engine.analyser = _FakeAnalyser()
    result = engine.describe_change(9999, 8888)
    assert "error" in result
    assert result["description"] == ""


def test_describe_change_without_a_configured_analyser(tmp_path):
    engine = make_bare_engine(tmp_path)
    engine.analyser = None
    result = engine.describe_change(1, 2)
    assert "not configured" in result["error"]


def test_compare_photos_intent_missing_entities(tmp_path):
    engine = make_bare_engine(tmp_path)
    engine.analyser = _FakeAnalyser()
    result = engine.handle("compare_photos", {}, {})
    assert result["confidence"] == 0.0


def test_compare_photos_intent_end_to_end(tmp_path, tmp_path_factory):
    engine = make_bare_engine(tmp_path)
    engine.analyser = _FakeAnalyser("Something changed.")

    dir_a = tmp_path_factory.mktemp("imgs2")
    from PIL import Image
    path_a = dir_a / "a.jpg"
    path_b = dir_a / "b.jpg"
    Image.new("RGB", (10, 10), "red").save(path_a)
    Image.new("RGB", (10, 10), "blue").save(path_b)
    id_a = insert(engine.db, str(path_a), "ha")
    id_b = insert(engine.db, str(path_b), "hb")

    result = engine.handle("compare_photos", {"file_id_a": id_a, "file_id_b": id_b}, {})
    assert result["response"] == "Something changed."
