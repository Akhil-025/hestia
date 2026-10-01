"""
modules/athena/progress.py

Live progress for a document ingestion run (backlog #70).

``IngestionProgress`` is a small, thread-safe record the ingest loop updates
after every file, so a web request can ask "how far along is it?" while the
work runs on another thread. It holds numbers and short strings only, never
document text.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

_MAX_ERRORS_KEPT = 20


class IngestionProgress:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.running = False
        self.total = 0
        self.done = 0
        self.current: Optional[str] = None
        self.failed: list[str] = []
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.result: Optional[dict] = None
        self.error: Optional[str] = None

    # -- writer side (the ingest loop) -----------------------------------

    def start(self, total: int) -> None:
        with self._lock:
            self._reset_locked()
            self.running = True
            self.total = max(0, int(total))
            self.started_at = time.time()

    def begin_file(self, name: str) -> None:
        with self._lock:
            self.current = name

    def file_done(self, name: str, status: str) -> None:
        with self._lock:
            self.done += 1
            if status == "failed" and len(self.failed) < _MAX_ERRORS_KEPT:
                self.failed.append(name)

    def finish(self, result: Optional[dict] = None, error: Optional[str] = None) -> None:
        with self._lock:
            self.running = False
            self.current = None
            self.finished_at = time.time()
            self.result = result
            self.error = error

    # -- reader side -----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """A consistent copy, safe to JSON-encode."""
        with self._lock:
            now = time.time()
            elapsed = None
            if self.started_at is not None:
                elapsed = round((self.finished_at or now) - self.started_at, 1)
            percent = (
                100 if (not self.running and self.started_at and self.total == 0)
                else round(100 * self.done / self.total) if self.total else 0
            )
            eta = None
            if self.running and self.done and elapsed:
                eta = round(elapsed / self.done * (self.total - self.done))
            return {
                "running": self.running,
                "total": self.total,
                "done": self.done,
                "percent": min(100, percent),
                "current": self.current,
                "failed": list(self.failed),
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "error": self.error,
                "result": self.result,
            }
