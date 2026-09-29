# tests/test_hot_reload.py
"""
Tests for core/hot_reload.py (backlog #15).

Drives the watcher via check_once() rather than starting the real
background thread and sleeping — deterministic and fast, and it's exactly
what the background loop calls on every tick anyway.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.hot_reload import FileWatcher


def touch(path, content="x"):
    path.write_text(content, encoding="utf-8")
    # Some filesystems have 1s mtime resolution; force it forward so a
    # rapid write-then-check in a test can't land on the same mtime as
    # the watcher's initial snapshot.
    new_time = time.time() + 1
    os.utime(path, (new_time, new_time))


# ---------------------------------------------------------------------------
# check_once
# ---------------------------------------------------------------------------

def test_no_change_does_not_fire(tmp_path):
    path = tmp_path / "f.txt"
    touch(path)
    calls = []
    watcher = FileWatcher(path, lambda: calls.append(1))
    assert watcher.check_once() is False
    assert calls == []


def test_a_real_change_fires_once(tmp_path):
    path = tmp_path / "f.txt"
    touch(path, "v1")
    calls = []
    watcher = FileWatcher(path, lambda: calls.append(1))
    touch(path, "v2")
    assert watcher.check_once() is True
    assert calls == [1]
    assert watcher.check_once() is False  # settles until the next change
    assert calls == [1]


def test_multiple_changes_each_fire_once(tmp_path):
    path = tmp_path / "f.txt"
    touch(path, "v1")
    calls = []
    watcher = FileWatcher(path, lambda: calls.append(1))
    touch(path, "v2")
    watcher.check_once()
    touch(path, "v3")
    watcher.check_once()
    assert calls == [1, 1]


def test_missing_file_does_not_raise_or_fire(tmp_path):
    path = tmp_path / "does_not_exist.txt"
    calls = []
    watcher = FileWatcher(path, lambda: calls.append(1))
    assert watcher.check_once() is False
    assert calls == []


def test_file_created_after_watcher_construction(tmp_path):
    path = tmp_path / "later.txt"
    calls = []
    watcher = FileWatcher(path, lambda: calls.append(1))
    touch(path, "now it exists")
    assert watcher.check_once() is True
    assert calls == [1]


def test_handler_exception_is_swallowed(tmp_path):
    path = tmp_path / "f.txt"
    touch(path, "v1")

    def bad_handler():
        raise RuntimeError("reload blew up")

    watcher = FileWatcher(path, bad_handler)
    touch(path, "v2")
    assert watcher.check_once() is True  # must not raise despite the handler


# ---------------------------------------------------------------------------
# start/stop lifecycle
# ---------------------------------------------------------------------------

def test_start_and_stop_do_not_raise(tmp_path):
    path = tmp_path / "f.txt"
    touch(path)
    watcher = FileWatcher(path, lambda: None, poll_seconds=0.05)
    watcher.start()
    watcher.stop(timeout=1.0)


def test_background_thread_detects_a_real_change(tmp_path):
    path = tmp_path / "f.txt"
    touch(path, "v1")
    calls = []
    watcher = FileWatcher(path, lambda: calls.append(1), poll_seconds=0.05)
    watcher.start()
    try:
        touch(path, "v2")
        # Give the poll loop a few cycles to notice.
        for _ in range(40):
            if calls:
                break
            time.sleep(0.05)
    finally:
        watcher.stop(timeout=1.0)
    assert calls == [1]


def test_start_is_idempotent(tmp_path):
    path = tmp_path / "f.txt"
    touch(path)
    watcher = FileWatcher(path, lambda: None, poll_seconds=0.05)
    watcher.start()
    first_thread = watcher._thread
    watcher.start()  # must not spawn a second thread
    try:
        assert watcher._thread is first_thread
    finally:
        watcher.stop(timeout=1.0)


def test_stop_before_start_does_not_raise(tmp_path):
    path = tmp_path / "f.txt"
    touch(path)
    FileWatcher(path, lambda: None).stop()
