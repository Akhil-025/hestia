# modules/orpheus/db.py

import sqlite3
import threading
from pathlib import Path


class OrpheusDB:
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
CREATE TABLE IF NOT EXISTS creations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT    NOT NULL,
    title       TEXT,
    content     TEXT    NOT NULL,
    metadata    TEXT,
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_creations_type      ON creations(type);
CREATE INDEX IF NOT EXISTS idx_creations_logged_at ON creations(logged_at);
""")

    def save(self, type_: str, content: str,
             title: str = "", metadata: str = "") -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO creations (type, title, content, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (type_, title, content, metadata)
            )
            return cur.lastrowid

    def get_recent(self, type_: str, limit: int = 10) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM creations
            WHERE type=?
            ORDER BY logged_at DESC LIMIT ?
            """,
            (type_, limit)
        )
        return [dict(r) for r in cur.fetchall()]

    def get_all(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM creations
            ORDER BY logged_at DESC LIMIT ?
            """,
            (limit,)
        )
        return [dict(r) for r in cur.fetchall()]