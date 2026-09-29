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

-- Samples of the user's own writing, used to learn a voice profile (#165).
-- Unlike `items` (which stores only a short preview of input), these are
-- kept in full because they ARE the data the profile is built from; the
-- user supplies them deliberately and can wipe them with clear_style().
CREATE TABLE IF NOT EXISTS style_samples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    label      TEXT,
    text       TEXT    NOT NULL,
    word_count INTEGER DEFAULT 0,
    added_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Exactly one row (id = 1): the current derived profile.
CREATE TABLE IF NOT EXISTS style_profile (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    profile_json TEXT    NOT NULL,
    sample_count INTEGER DEFAULT 0,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
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
            ORDER BY logged_at DESC, id DESC LIMIT ?
            """,
            (type_, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_all(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM items
            ORDER BY logged_at DESC, id DESC LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_item(self, item_id: int) -> dict | None:
        cur = self._conn.execute("SELECT * FROM items WHERE id=?", (item_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_latest(self, type_: str) -> dict | None:
        """Newest item of *type_* (id breaks same-second ties)."""
        cur = self._conn.execute(
            "SELECT * FROM items WHERE type=? ORDER BY logged_at DESC, id DESC LIMIT 1",
            (type_,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Style profile (#165)
    # ------------------------------------------------------------------

    def add_style_sample(self, text: str, label: str = "",
                         word_count: int = 0) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO style_samples (label, text, word_count) VALUES (?, ?, ?)",
                (label, text, word_count),
            )
            return cur.lastrowid

    def get_style_samples(self, limit: int = 200) -> list[dict]:
        """Samples oldest-first, so a profile rebuild is order-stable."""
        cur = self._conn.execute(
            "SELECT * FROM style_samples ORDER BY id ASC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]

    def save_style_profile(self, profile_json: str, sample_count: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO style_profile (id, profile_json, sample_count, updated_at)
                VALUES (1, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    sample_count = excluded.sample_count,
                    updated_at   = CURRENT_TIMESTAMP
                """,
                (profile_json, sample_count),
            )

    def get_style_profile(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM style_profile WHERE id=1"
        ).fetchone()
        return dict(row) if row else None

    def clear_style(self) -> int:
        """Delete every sample and the profile. Returns samples removed."""
        with self._lock, self._conn:
            n = self._conn.execute("SELECT COUNT(*) AS n FROM style_samples").fetchone()["n"]
            self._conn.execute("DELETE FROM style_samples")
            self._conn.execute("DELETE FROM style_profile")
            return n

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