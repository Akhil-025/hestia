# modules/dionysus/db.py

import sqlite3
import threading
from pathlib import Path


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
    logged_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                "UPDATE recommendations SET dismissed=1 WHERE title=?", (title,)
            )

    def dismissed_titles(self, type_: str) -> list[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT title FROM recommendations WHERE type=? AND dismissed=1",
                (type_,)
            )
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