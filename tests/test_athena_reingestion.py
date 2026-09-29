# tests/test_athena_reingestion.py
"""
Tests for change-aware re-ingestion (backlog #61) and the "what's new
since I last checked" digest (backlog #57).

Reuses tests/test_athena.py's `make_engine`/`_ensure_stubs()` fixtures via
import, the same pattern as tests/test_mnemosyne_extended.py uses for
test_mnemosyne.py.
"""
import os
import sys
import shutil
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_athena import make_engine  # noqa: E402


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Re-ingestion is change-aware, not just presence-aware (#61)
# ---------------------------------------------------------------------------

def test_unchanged_file_is_skipped_on_second_ingest():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        path = os.path.join(tmp, "documents", "Bio", "notes.txt")
        write_file(path, "The mitochondria is the powerhouse of the cell.")

        first = engine.handle("ingest", {}, {})
        assert first["data"]["new_files"] == 1
        assert first["data"]["total_chunks"] > 0

        second = engine.handle("ingest", {}, {})
        assert second["data"]["new_files"] == 0
        assert second["data"]["updated_files"] == 0
        assert second["data"]["unchanged_files"] == 1
        assert "already" in second["response"].lower() or "up to date" in second["response"].lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_modified_file_is_reingested_not_skipped():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        path = os.path.join(tmp, "documents", "Bio", "notes.txt")
        write_file(path, "The mitochondria is the powerhouse of the cell.")
        engine.handle("ingest", {}, {})

        # Force a distinct mtime+size signature (the file might otherwise
        # get the exact same mtime as before on a fast filesystem).
        time.sleep(1.1)
        write_file(path, "Photosynthesis converts light energy into chemical energy in plants.")

        second = engine.handle("ingest", {}, {})
        assert second["data"]["updated_files"] == 1
        assert second["data"]["new_files"] == 0
        assert "updated" in second["response"].lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_modified_file_search_reflects_new_content_not_stale_content():
    # The real point of change detection: searching for the OLD content
    # should no longer find it once the file has genuinely changed, and
    # the NEW content should be findable.
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        path = os.path.join(tmp, "documents", "Bio", "notes.txt")
        write_file(path, "The mitochondria is the powerhouse of the cell.")
        engine.handle("ingest", {}, {})

        time.sleep(1.1)
        write_file(path, "Photosynthesis converts light energy into chemical energy.")
        engine.handle("ingest", {}, {})

        result = engine.handle("search", {"query": "photosynthesis light energy"}, {})
        texts = " ".join(s["text"] for s in result["data"]["sources"])
        assert "photosynthesis" in texts.lower()
        assert "mitochondria" not in texts.lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stale_chunks_are_removed_not_left_alongside_new_ones():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        path = os.path.join(tmp, "documents", "Bio", "notes.txt")
        write_file(path, "The mitochondria is the powerhouse of the cell.")
        engine.handle("ingest", {}, {})
        before = engine.stats()["total_chunks"]

        time.sleep(1.1)
        write_file(path, "The mitochondria is the powerhouse of the cell.")  # same length/content, new mtime
        engine.handle("ingest", {}, {})
        after = engine.stats()["total_chunks"]

        # Re-ingesting the same-shaped content must not accumulate
        # duplicate chunks — old ones are deleted before new ones are added.
        assert after == before
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_new_and_existing_files_are_both_handled_in_one_pass():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        path_a = os.path.join(tmp, "documents", "Bio", "a.txt")
        write_file(path_a, "Content about cells and mitochondria and organelles in biology.")
        engine.handle("ingest", {}, {})

        path_b = os.path.join(tmp, "documents", "Bio", "b.txt")
        write_file(path_b, "Content about photosynthesis and chlorophyll pigments in plants.")

        result = engine.handle("ingest", {}, {})
        assert result["data"]["new_files"] == 1
        assert result["data"]["unchanged_files"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# "What's new since I last checked" digest (#57) — a dry preview
# ---------------------------------------------------------------------------

def test_check_updates_reports_new_files_without_ingesting():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        write_file(os.path.join(tmp, "documents", "Bio", "a.txt"), "Cell biology notes with plenty of extra detail here.")

        result = engine.handle("check_updates", {}, {})
        assert len(result["data"]["new_files"]) == 1
        assert result["data"]["unchanged_count"] == 0

        # Nothing was actually ingested by the check.
        assert engine.stats()["total_chunks"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_check_updates_reports_updated_files():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        path = os.path.join(tmp, "documents", "Bio", "a.txt")
        write_file(path, "Cell biology notes with plenty of extra detail here.")
        engine.handle("ingest", {}, {})

        time.sleep(1.1)
        write_file(path, "Updated cell biology notes with a lot more added detail now.")

        result = engine.handle("check_updates", {}, {})
        assert len(result["data"]["updated_files"]) == 1
        assert result["data"]["new_files"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_check_updates_with_no_changes_reports_clean():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        write_file(os.path.join(tmp, "documents", "Bio", "a.txt"), "Cell biology notes with plenty of extra detail here.")
        engine.handle("ingest", {}, {})

        result = engine.handle("check_updates", {}, {})
        assert result["data"]["new_files"] == []
        assert result["data"]["updated_files"] == []
        assert result["data"]["unchanged_count"] == 1
        assert "up to date" in result["response"].lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_check_updates_intent_is_dispatchable_prefixed_and_stripped():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        for intent in ("check_updates", "athena_check_updates"):
            assert engine.can_handle(intent), f"can_handle() rejected {intent!r}"
            r = engine.handle(intent, {}, {})
            assert r["response"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_check_updates_survives_an_empty_directory():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle("check_updates", {}, {})
        assert result["data"]["new_files"] == []
        assert result["data"]["total_files_scanned"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
