# modules/dionysus/db.py

import sqlite3
import threading
from pathlib import Path
from typing import Optional


class DionysusDB:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self):
        with self._conn:
            self._conn.executescript("""
CREATE TABLE IF NOT EXISTS recommendations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT    NOT NULL,
    title        TEXT    NOT NULL,
    detail       TEXT,
    rating       REAL,
    dismissed    BOOLEAN DEFAULT 0,
    seen         BOOLEAN DEFAULT 0,
    logged_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dismissed_at TIMESTAMP,
    feedback     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS recharge_routines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT NOT NULL,
    schedule    TEXT NOT NULL,
    reminder_id TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_recs_type      ON recommendations(type);
CREATE INDEX IF NOT EXISTS idx_recs_dismissed ON recommendations(dismissed);
CREATE INDEX IF NOT EXISTS idx_recs_seen      ON recommendations(seen);
""")
            # Migration path for pre-existing databases created before the
            # `seen` column existed.
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(recommendations)")}
            if "seen" not in cols:
                self._conn.execute(
                    "ALTER TABLE recommendations ADD COLUMN seen BOOLEAN DEFAULT 0"
                )
            # Backlog #147: dismissals can expire, so we need to know when they
            # happened. Dismissals made before this column existed are dated
            # "now" so an upgrade doesn't make them all expire at once.
            if "dismissed_at" not in cols:
                self._conn.execute(
                    "ALTER TABLE recommendations ADD COLUMN dismissed_at TIMESTAMP"
                )
                self._conn.execute(
                    "UPDATE recommendations SET dismissed_at = CURRENT_TIMESTAMP "
                    "WHERE dismissed = 1"
                )
            # Backlog #269: +1 "more like this", -1 "less like this", 0 none.
            if "feedback" not in cols:
                self._conn.execute(
                    "ALTER TABLE recommendations ADD COLUMN feedback INTEGER DEFAULT 0"
                )

    def log(self, type_: str, title: str, detail: str = "", rating: float = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO recommendations (type, title, detail, rating) VALUES (?, ?, ?, ?)",
                (type_, title, detail, rating)
            )
            return cur.lastrowid

    def dismiss(self, title: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE recommendations SET dismissed=1, dismissed_at=CURRENT_TIMESTAMP "
                "WHERE title=?",
                (title,),
            )

    def dismissed_titles(self, type_: str, expire_days: Optional[float] = None) -> list[str]:
        """Titles the user dismissed. With ``expire_days`` set, dismissals older
        than that many days no longer count (backlog #147), so a title can come
        back after a long enough gap. ``None`` or 0 means never expire."""
        sql = "SELECT title FROM recommendations WHERE type=? AND dismissed=1"
        params: list = [type_]
        if expire_days:
            sql += (
                " AND (dismissed_at IS NULL OR "
                "julianday('now') - julianday(dismissed_at) < ?)"
            )
            params.append(float(expire_days))
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [r["title"] for r in cur.fetchall()]

    def mark_seen(self, title: str, type_: str = "movie") -> bool:
        """
        Mark the most recent matching recommendation as watched/seen.

        Returns True if a row was updated, False if no matching (type,
        title) recommendation exists. Matching is case-insensitive since
        the title usually round-trips through an LLM.
        """
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE recommendations SET seen=1
                WHERE id = (
                    SELECT id FROM recommendations
                    WHERE type = ? AND title = ? COLLATE NOCASE
                    ORDER BY logged_at DESC LIMIT 1
                )
                """,
                (type_, title),
            )
            return cur.rowcount > 0

    def seen_titles(self, type_: str) -> list[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT title FROM recommendations WHERE type=? AND seen=1",
                (type_,)
            )
            return [r["title"] for r in cur.fetchall()]

    # -- #269: more / less like this ----------------------------------------

    def set_feedback(self, title: str, value: int, type_: Optional[str] = None) -> Optional[dict]:
        """Record +1 (more like this) or -1 (less like this) on the most recent
        matching recommendation. Matching is case-insensitive. ``type_=None``
        matches any type. Returns ``{"title", "type"}`` of the row updated, or
        None if nothing matched."""
        value = 1 if value > 0 else -1
        sql = "SELECT id, type, title FROM recommendations WHERE title = ? COLLATE NOCASE"
        params: list = [title]
        if type_:
            sql += " AND type = ?"
            params.append(type_)
        sql += " ORDER BY logged_at DESC, id DESC LIMIT 1"
        with self._lock, self._conn:
            row = self._conn.execute(sql, params).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE recommendations SET feedback=? WHERE id=?", (value, row["id"])
            )
            return {"title": row["title"], "type": row["type"]}

    def latest(self, type_: Optional[str] = None) -> Optional[dict]:
        """Most recently logged recommendation (optionally of one type)."""
        sql = "SELECT * FROM recommendations"
        params: list = []
        if type_:
            sql += " WHERE type = ?"
            params.append(type_)
        sql += " ORDER BY logged_at DESC, id DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def feedback_titles(self, type_: str, value: int, limit: int = 8) -> list[str]:
        """Distinct titles with the given feedback (+1 / -1), newest first."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT title FROM recommendations WHERE type=? AND feedback=? "
                "ORDER BY logged_at DESC, id DESC",
                (type_, 1 if value > 0 else -1),
            )
            out: list[str] = []
            for r in cur.fetchall():
                if r["title"] not in out:
                    out.append(r["title"])
                if len(out) >= limit:
                    break
            return out

    def recent_titles(self, type_: str, limit: int = 10) -> list[str]:
        """Distinct recently logged titles, including dismissed ones, so
        "surprise me" can tell what the user's current taste looks like."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT title FROM recommendations WHERE type=? "
                "ORDER BY logged_at DESC, id DESC LIMIT ?",
                (type_, limit * 3),
            )
            out: list[str] = []
            for r in cur.fetchall():
                if r["title"] not in out:
                    out.append(r["title"])
                if len(out) >= limit:
                    break
            return out

    # -- #152: recharge routines --------------------------------------------

    def add_routine(self, label: str, schedule: str, reminder_id: Optional[str] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO recharge_routines (label, schedule, reminder_id) VALUES (?, ?, ?)",
                (label, schedule, None if reminder_id is None else str(reminder_id)),
            )
            return cur.lastrowid

    def list_routines(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM recharge_routines ORDER BY id DESC"
            )
            return [dict(r) for r in cur.fetchall()]

    def get_history(self, type_: str, limit: int = 20) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM recommendations
                WHERE type=? AND dismissed=0
                ORDER BY logged_at DESC LIMIT ?
                """,
                (type_, limit)
            )
            return [dict(r) for r in cur.fetchall()]
