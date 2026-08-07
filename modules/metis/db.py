# modules/metis/db.py

import sqlite3
import threading


class MetisDB:
    """
    Persistence for Metis writing-assistant output.

    Schema mirrors modules/orpheus/db.py's `creations` table (same
    type/title/content/metadata/logged_at shape) so the two modules stay
    consistent, but adds `input_chars`/`output_chars` columns so
    `writing_stats` can report real throughput numbers instead of
    re-deriving them from `content` (which only ever stores the *output*,
    never the original input text — we deliberately never persist raw
    user input verbatim beyond a short preview, see `save()`).
    """

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
CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    type          TEXT    NOT NULL,
    title         TEXT,
    content       TEXT    NOT NULL,
    input_preview TEXT,
    input_chars   INTEGER DEFAULT 0,
    output_chars  INTEGER DEFAULT 0,
    metadata      TEXT,
    logged_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_items_type      ON items(type);
CREATE INDEX IF NOT EXISTS idx_items_logged_at ON items(logged_at);
""")

    def save(self, type_: str, content: str, title: str = "",
              input_preview: str = "", input_chars: int = 0,
              output_chars: int = 0, metadata: str = "") -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO items
                    (type, title, content, input_preview,
                     input_chars, output_chars, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (type_, title, content, input_preview,
                 input_chars, output_chars, metadata),
            )
            return cur.lastrowid

    def get_recent(self, type_: str, limit: int = 10) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM items
            WHERE type=?
            ORDER BY logged_at DESC LIMIT ?
            """,
            (type_, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_all(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM items
            ORDER BY logged_at DESC LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_stats(self) -> dict:
        """
        Aggregate counts and character throughput, grouped by type, plus
        an overall total. Used by the `writing_stats` intent.
        """
        cur = self._conn.execute(
            """
            SELECT type,
                   COUNT(*)          AS n,
                   SUM(input_chars)  AS in_chars,
                   SUM(output_chars) AS out_chars
            FROM items
            GROUP BY type
            """
        )
        by_type = {
            row["type"]: {
                "count": row["n"],
                "input_chars": row["in_chars"] or 0,
                "output_chars": row["out_chars"] or 0,
            }
            for row in cur.fetchall()
        }
        total = self._conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"]
        return {"by_type": by_type, "total": total}