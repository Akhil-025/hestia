"""
core/hot_reload.py

File-change watching for backlog #15: tune `config/nlu_prompt.txt` (and
detect changes to `config/laptop_config.yaml`) without a full restart.

Design
------
mtime polling on a daemon thread, not inotify/watchdog. Hestia is a
single-user personal assistant checking two small text files; a filesystem
event library is a dependency and a platform-support surface (inotify is
Linux-only) for a problem a 3-second poll solves with zero new
dependencies and identical practical latency for a human hand-editing a
file in an editor.

Scope, honestly stated
-----------------------
- `config/nlu_prompt.txt`: fully hot-reloaded. `HestiaNLU.reload_prompt()`
  re-reads the file, rebuilds the JSON schema sent to Ollama, re-runs the
  registry drift check, and invalidates the classification cache (a cached
  answer classified under the OLD prompt must not survive a prompt
  change). This is the safe half of #15 — the prompt is pure input to a
  stateless call, so reloading it has no ordering dependencies on
  anything else in the process.

- `config/laptop_config.yaml`: NOT fully hot-applied, and this file does
  not pretend otherwise. Most keys there (`database.path`, `ollama.host`,
  every module's own config block) are read exactly once, at construction
  time, by code scattered across `HestiaBuilder`'s dozen `build_*`
  methods — reapplying a change to `ollama.port` after the fact would mean
  tearing down and rebuilding live Ollama connections, in-flight module
  state, and the orchestrator's registration, which is a much bigger
  change than "tune a prompt" and is legitimately out of scope for a
  hot-reload feature. What IS implemented: the change is detected,
  re-validated with `core.config_validation` exactly as it would be at
  startup, and logged with a diff of which top-level sections changed —
  so at minimum you find out immediately that a restart is needed, and
  whether the edit you just made would even pass validation, instead of
  only discovering either on the next restart.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_POLL_SECONDS = 3.0


class FileWatcher:
    """
    Polls one file's mtime on a daemon thread and calls *on_change* when
    it changes.

    `on_change` receives no arguments and is called on the watcher thread,
    not the caller's — it must be safe to call from a background thread
    (both `HestiaNLU.reload_prompt` and the config-diff logger used here
    are). Any exception it raises is caught and logged: a broken reload
    handler must silence itself, not silence the entire watcher.
    """

    def __init__(
        self,
        path: str | Path,
        on_change: Callable[[], None],
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
    ) -> None:
        self.path = Path(path)
        self._on_change = on_change
        self._poll_seconds = float(poll_seconds)
        self._last_mtime: Optional[float] = self._current_mtime()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _current_mtime(self) -> Optional[float]:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

    def start(self) -> None:
        """Idempotent: calling start() on an already-running watcher is a no-op."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"hot-reload:{self.path.name}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._poll_seconds)
            if self._stop_event.is_set():
                break
            self.check_once()

    def check_once(self) -> bool:
        """
        Check the file once, synchronously, firing `on_change` if it
        changed since the last check. Returns whether it fired.

        Exposed publicly (not just via the polling loop) so tests can
        drive the watcher deterministically instead of racing a real
        background thread against a sleep.
        """
        mtime = self._current_mtime()
        if mtime is None:
            return False  # file (temporarily?) missing — nothing to react to
        if self._last_mtime is not None and mtime == self._last_mtime:
            return False

        self._last_mtime = mtime
        try:
            self._on_change()
        except Exception:
            logger.exception(
                "Hot-reload handler for %s raised; watcher continues.", self.path
            )
        return True
