"""
core/db_maintenance.py  (backlog #233: DB maintenance)

Off-peak housekeeping for Hestia's SQLite files. SQLite doesn't hand freed
pages back to the OS: a database that had lots of rows deleted stays big until
it is VACUUMed. This checks how much of each file is free space and only
vacuums when that is worth doing.

Safety properties (each is tested):
- Never deletes or changes data: VACUUM only rebuilds the file.
- Skips a database that is busy/locked and reports it, so the caller retries
  on its next tick instead of blocking a live module.
- Checks free disk first (VACUUM can need up to ~2x the file size).
- At most once per ``min_interval_days`` per database.
- Only touches files that really are SQLite databases.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SQLITE_MAGIC = b"SQLite format 3\x00"


class DBMaintenance:
    def __init__(
        self,
        paths: Optional[Iterable[str | Path]] = None,
        root: Optional[Path] = None,
        *,
        discover: bool = True,
        free_ratio_threshold: float = 0.2,
        min_free_pages: int = 64,
        min_interval_days: int = 7,
        state_path: Optional[Path] = None,
        busy_timeout_ms: int = 2000,
        disk_headroom: float = 2.0,
    ) -> None:
        self.root = Path(root) if root else _PROJECT_ROOT
        self._extra = [Path(p) for p in (paths or [])]
        self._discover = discover
        self.free_ratio_threshold = free_ratio_threshold
        self.min_free_pages = min_free_pages
        self.min_interval = timedelta(days=min_interval_days)
        self.state_path = Path(state_path) if state_path else self.root / "data" / "db_maintenance.json"
        self.busy_timeout_ms = busy_timeout_ms
        self.disk_headroom = disk_headroom

    # ------------------------------------------------------------------
    # Discovery and state
    # ------------------------------------------------------------------

    def databases(self) -> list[Path]:
        found: dict[str, Path] = {}
        if self._discover:
            data_dir = self.root / "data"
            if data_dir.is_dir():
                for p in data_dir.rglob("*.db"):
                    if p.is_file():
                        found[str(p.resolve())] = p.resolve()
        for p in self._extra:
            if p.is_file():
                found[str(p.resolve())] = p.resolve()
        return sorted(found.values())

    def _load_state(self) -> dict[str, str]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self, state: dict[str, str]) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            logger.warning("db_maintenance: couldn't save state file.", exc_info=True)

    @staticmethod
    def _is_sqlite(path: Path) -> bool:
        try:
            with open(path, "rb") as fh:
                return fh.read(16) == _SQLITE_MAGIC
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Inspect / maintain
    # ------------------------------------------------------------------

    def inspect(self, path: Path) -> dict[str, Any]:
        """Free-page statistics, read-only."""
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                               timeout=self.busy_timeout_ms / 1000)
        try:
            pages = conn.execute("PRAGMA page_count").fetchone()[0]
            free = conn.execute("PRAGMA freelist_count").fetchone()[0]
            size = conn.execute("PRAGMA page_size").fetchone()[0]
        finally:
            conn.close()
        return {
            "pages": pages, "free_pages": free, "page_size": size,
            "free_ratio": (free / pages) if pages else 0.0,
            "size_bytes": pages * size,
        }

    def maintain(self, path: Path, now: Optional[datetime] = None,
                 state: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """One database. Returns ``{"path", "status", ...}``.

        status: missing | not_sqlite | recent | healthy | vacuumed | locked |
        no_space | error. Only ``healthy`` and ``vacuumed`` count as done.
        """
        now = now or datetime.now(timezone.utc)
        state = state if state is not None else self._load_state()
        key = str(path)
        result: dict[str, Any] = {"path": key, "status": "error"}

        if not path.is_file():
            result["status"] = "missing"
            return result
        if not self._is_sqlite(path):
            result["status"] = "not_sqlite"
            return result
        last = state.get(key)
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                if now - last_dt < self.min_interval:
                    result["status"] = "recent"
                    return result
            except ValueError:
                pass

        try:
            info = self.inspect(path)
            result.update(info)
            worth_it = (
                info["free_ratio"] >= self.free_ratio_threshold
                and info["free_pages"] >= self.min_free_pages
            )
            conn = sqlite3.connect(str(path), timeout=self.busy_timeout_ms / 1000,
                                   isolation_level=None)
            try:
                conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)}")
                if not worth_it:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    result["status"] = "healthy"
                    state[key] = now.isoformat()
                    return result

                free_disk = shutil.disk_usage(path.parent).free
                if free_disk < info["size_bytes"] * self.disk_headroom:
                    result["status"] = "no_space"
                    return result

                busy, _log, _ckpt = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if busy:
                    result["status"] = "locked"
                    return result
                before = path.stat().st_size
                conn.execute("VACUUM")
                result["reclaimed_bytes"] = max(0, before - path.stat().st_size)
                result["status"] = "vacuumed"
                state[key] = now.isoformat()
                return result
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            result["status"] = "locked" if ("locked" in msg or "busy" in msg) else "error"
            result["error"] = str(exc)
            return result
        except Exception as exc:  # noqa: BLE001 - maintenance must never crash the heartbeat
            logger.exception("db_maintenance: unexpected failure on %s", path)
            result["error"] = str(exc)
            return result

    def run(self, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        """Maintain every known database; returns one result per database."""
        state = self._load_state()
        results = [self.maintain(p, now, state) for p in self.databases()]
        self._save_state(state)
        for r in results:
            if r["status"] in ("vacuumed", "locked", "no_space", "error"):
                logger.info("db_maintenance: %s -> %s", r["path"], r["status"])
        return results

    @staticmethod
    def needs_retry(results: list[dict[str, Any]]) -> bool:
        """True if any database was skipped for a transient reason."""
        return any(r["status"] in ("locked", "no_space") for r in results)
