# modules/orpheus/db.py

import sqlite3
import threading


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

    def search(self, type_: str | None = None, keyword: str | None = None,
               limit: int = 10) -> list[dict]:
        """
        Return past creations, optionally filtered by exact `type_` and/or a
        `keyword` matched against title/content (case-insensitive substring).

        Both filters are optional and combine with AND. `limit` bounds the
        result set; callers are expected to have already clamped it.
        """
        query = "SELECT * FROM creations WHERE 1=1"
        params: list = []
        if type_:
            query += " AND type=?"
            params.append(type_)
        if keyword:
            query += " AND (title LIKE ? OR content LIKE ?)"
            like = f"%{keyword}%"
            params += [like, like]
        query += " ORDER BY logged_at DESC LIMIT ?"
        params.append(limit)
        cur = self._conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]