# tests/test_iris_duplicates_and_captions.py
"""
Tests for whole-library duplicate scanning (backlog #73) and manual
caption/tag correction (backlog #78).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from modules.iris.db import IrisDB
from modules.iris.ingestion import DuplicateDetector


def make_db(tmp_path):
    return IrisDB(str(tmp_path / "iris.db"))


def insert(db, path, file_hash, phash="samehash0000"):
    return db.insert_file(path, file_hash, perceptual_hash=phash, file_size=100, file_type="image", mime_type="image/jpeg")


# ---------------------------------------------------------------------------
# Exact duplicates (same file_hash)
# ---------------------------------------------------------------------------

def test_exact_duplicates_are_grouped(tmp_path):
    db = make_db(tmp_path)
    insert(db, "a.jpg", "samehash", "p1")
    insert(db, "b.jpg", "samehash", "p2")
    detector = DuplicateDetector(db)
    groups = detector.find_all_duplicate_groups()
    exact = [g for g in groups if g["kind"] == "exact"]
    assert len(exact) == 1
    assert len(exact[0]["files"]) == 2
    assert exact[0]["similarity"] == 1.0


def test_files_with_unique_hashes_are_not_grouped(tmp_path):
    db = make_db(tmp_path)
    insert(db, "a.jpg", "hash1", "p1")
    insert(db, "b.jpg", "hash2", "p2")
    detector = DuplicateDetector(db)
    groups = detector.find_all_duplicate_groups()
    assert groups == []


def test_three_way_exact_duplicate_is_one_group_of_three(tmp_path):
    db = make_db(tmp_path)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        insert(db, name, "samehash", f"p_{name}")
    detector = DuplicateDetector(db)
    groups = detector.find_all_duplicate_groups()
    assert len(groups) == 1
    assert len(groups[0]["files"]) == 3


# ---------------------------------------------------------------------------
# Near-duplicates (perceptual hash within threshold)
# ---------------------------------------------------------------------------

def test_near_duplicates_within_threshold_are_grouped(tmp_path):
    # Needs the `imagehash` library (a real dependency — requirements.txt)
    # to compute Hamming distance; not installed in every sandbox.
    pytest.importorskip("imagehash")
    db = make_db(tmp_path)
    # Identical hashes but different file_hash — these should be caught
    # by the near-duplicate (perceptual) path, distance 0.
    insert(db, "a.jpg", "hash1", "0000000000000000:0000000000000000")
    insert(db, "b.jpg", "hash2", "0000000000000000:0000000000000000")
    detector = DuplicateDetector(db, perceptual_hash_threshold=12)
    groups = detector.find_all_duplicate_groups()
    near = [g for g in groups if g["kind"] == "near"]
    assert len(near) == 1
    assert len(near[0]["files"]) == 2


def test_perceptually_distant_files_are_not_grouped(tmp_path):
    pytest.importorskip("imagehash")
    db = make_db(tmp_path)
    insert(db, "a.jpg", "hash1", "0000000000000000:0000000000000000")
    insert(db, "b.jpg", "hash2", "ffffffffffffffff:ffffffffffffffff")  # maximally different
    detector = DuplicateDetector(db, perceptual_hash_threshold=12)
    groups = detector.find_all_duplicate_groups()
    assert groups == []


def test_a_file_already_in_an_exact_group_is_not_also_in_a_near_group(tmp_path):
    pytest.importorskip("imagehash")
    db = make_db(tmp_path)
    insert(db, "a.jpg", "samehash", "0000000000000000:0000000000000000")
    insert(db, "b.jpg", "samehash", "0000000000000000:0000000000000000")  # exact dup of a
    insert(db, "c.jpg", "hash3", "0000000000000000:0000000000000000")     # near-dup of a/b only
    detector = DuplicateDetector(db, perceptual_hash_threshold=12)
    groups = detector.find_all_duplicate_groups()
    all_grouped_ids = [f["id"] for g in groups for f in g["files"]]
    # No id should appear in two different groups.
    assert len(all_grouped_ids) == len(set(all_grouped_ids))


def test_groups_are_sorted_largest_first(tmp_path):
    db = make_db(tmp_path)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        insert(db, name, "hash_big", f"p_{name}")
    insert(db, "d.jpg", "hash_small1")
    insert(db, "e.jpg", "hash_small1")
    detector = DuplicateDetector(db)
    groups = detector.find_all_duplicate_groups()
    assert len(groups[0]["files"]) >= len(groups[-1]["files"])


def test_empty_library_returns_no_groups(tmp_path):
    db = make_db(tmp_path)
    detector = DuplicateDetector(db)
    assert detector.find_all_duplicate_groups() == []


def test_single_file_library_returns_no_groups(tmp_path):
    db = make_db(tmp_path)
    insert(db, "a.jpg", "hash1", "p1")
    detector = DuplicateDetector(db)
    assert detector.find_all_duplicate_groups() == []


# ---------------------------------------------------------------------------
# Manual caption/tag correction (#78)
# ---------------------------------------------------------------------------

def test_correct_caption_updates_the_caption(tmp_path):
    db = make_db(tmp_path)
    file_id = insert(db, "a.jpg", "hash1", "p1")
    ok = db.correct_caption(file_id, caption="A sunset over the ocean")
    assert ok is True
    row = db.get_file(file_id)
    assert row["caption"] == "A sunset over the ocean"


def test_correct_caption_marks_source_as_user(tmp_path):
    db = make_db(tmp_path)
    file_id = insert(db, "a.jpg", "hash1", "p1")
    db.correct_caption(file_id, caption="Fixed caption")
    row = db.get_file(file_id)
    assert row["caption_source"] == "user"


def test_correct_caption_can_update_tags_only(tmp_path):
    db = make_db(tmp_path)
    file_id = insert(db, "a.jpg", "hash1", "p1")
    db.correct_caption(file_id, tags="beach,sunset,ocean")
    row = db.get_file(file_id)
    assert row["tags"] == "beach,sunset,ocean"


def test_correct_caption_can_update_both_at_once(tmp_path):
    db = make_db(tmp_path)
    file_id = insert(db, "a.jpg", "hash1", "p1")
    db.correct_caption(file_id, caption="New caption", tags="new,tags")
    row = db.get_file(file_id)
    assert row["caption"] == "New caption"
    assert row["tags"] == "new,tags"


def test_correct_caption_with_neither_field_returns_false(tmp_path):
    db = make_db(tmp_path)
    file_id = insert(db, "a.jpg", "hash1", "p1")
    assert db.correct_caption(file_id) is False


def test_correct_caption_for_unknown_file_returns_false(tmp_path):
    db = make_db(tmp_path)
    assert db.correct_caption(99999, caption="x") is False


def test_default_caption_source_is_ai(tmp_path):
    db = make_db(tmp_path)
    file_id = insert(db, "a.jpg", "hash1", "p1")
    row = db.get_file(file_id)
    assert row["caption_source"] == "ai"


# ---------------------------------------------------------------------------
# End-to-end via IrisEngine
# ---------------------------------------------------------------------------

def test_correct_caption_intent_success(tmp_path):
    from modules.iris.iris_engine import IrisEngine
    engine = IrisEngine.__new__(IrisEngine)
    engine.db = make_db(tmp_path)
    file_id = insert(engine.db, "a.jpg", "hash1", "p1")

    result = engine.handle("correct_caption", {"file_id": file_id, "caption": "Fixed"}, {})
    assert result["confidence"] > 0.5
    assert engine.db.get_file(file_id)["caption"] == "Fixed"


def test_correct_caption_intent_missing_fields(tmp_path):
    from modules.iris.iris_engine import IrisEngine
    engine = IrisEngine.__new__(IrisEngine)
    engine.db = make_db(tmp_path)

    result = engine.handle("correct_caption", {}, {})
    assert result["confidence"] == 0.0


def test_correct_caption_intent_unknown_file(tmp_path):
    from modules.iris.iris_engine import IrisEngine
    engine = IrisEngine.__new__(IrisEngine)
    engine.db = make_db(tmp_path)

    result = engine.handle("correct_caption", {"file_id": 9999, "caption": "x"}, {})
    assert "couldn't find" in result["response"].lower()


def test_find_duplicates_intent_end_to_end(tmp_path):
    from modules.iris.iris_engine import IrisEngine
    from modules.iris.ingestion import FileIngestor
    from modules.iris.config import IrisConfig

    engine = IrisEngine.__new__(IrisEngine)
    engine.db = make_db(tmp_path)
    config = IrisConfig(
        db_path=str(tmp_path / "iris.db"), source_dir=str(tmp_path), output_dir=str(tmp_path),
        cache_dir=str(tmp_path), chroma_dir=str(tmp_path),
    )
    engine.ingestor = FileIngestor(config, engine.db)

    insert(engine.db, "a.jpg", "samehash", "p1")
    insert(engine.db, "b.jpg", "samehash", "p2")

    result = engine.handle("find_duplicates", {}, {})
    assert "1 duplicate group" in result["response"]
    assert len(result["data"]["groups"]) == 1
