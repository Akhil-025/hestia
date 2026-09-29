# modules/orpheus/db.py

import sqlite3
import threading


class OrpheusDB:
    """
    Persistence for Orpheus creations, with per-creation version history
    (backlog #167).

    Model
    -----
    ``creations`` holds one row per piece and always reflects the *latest*
    text (so every existing reader — recall, search, export — keeps working
    unchanged). ``creation_versions`` holds every version of that text,
    including version 1: the original draft exactly as first generated.

    Versions are append-only. Revising, polishing or restoring a piece adds
    a new version; nothing ever updates or deletes a version row, so an
    edit can never overwrite the original.

    Existing databases are upgraded in place on open: the
    ``current_version`` column is added if missing and every creation that
    has no version rows gets its current text recorded as version 1 (before
    this feature no edit path existed, so that text *is* the original).
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

CREATE TABLE IF NOT EXISTS creation_versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    creation_id  INTEGER NOT NULL REFERENCES creations(id),
    version      INTEGER NOT NULL,
    content      TEXT    NOT NULL,
    note         TEXT,
    metadata     TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (creation_id, version)
);

CREATE INDEX IF NOT EXISTS idx_versions_creation
    ON creation_versions(creation_id, version);
""")
            cols = {r["name"] for r in
                    self._conn.execute("PRAGMA table_info(creations)")}
            if "current_version" not in cols:
                self._conn.execute(
                    "ALTER TABLE creations "
                    "ADD COLUMN current_version INTEGER NOT NULL DEFAULT 1"
                )
            # Backfill v1 for pre-versioning rows (idempotent).
            self._conn.execute(
                """
                INSERT INTO creation_versions
                    (creation_id, version, content, note, metadata)
                SELECT c.id, 1, c.content, 'original', c.metadata
                FROM creations c
                WHERE NOT EXISTS (
                    SELECT 1 FROM creation_versions v WHERE v.creation_id = c.id
                )
                """
            )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def save(self, type_: str, content: str,
             title: str = "", metadata: str = "") -> int:
        """Insert a creation and record its text as version 1."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO creations (type, title, content, metadata,
                                       current_version)
                VALUES (?, ?, ?, ?, 1)
                """,
                (type_, title, content, metadata)
            )
            creation_id = cur.lastrowid
            self._conn.execute(
                """
                INSERT INTO creation_versions
                    (creation_id, version, content, note, metadata)
                VALUES (?, 1, ?, 'original', ?)
                """,
                (creation_id, content, metadata),
            )
            return creation_id

    def add_version(self, creation_id: int, content: str,
                    note: str = "", metadata: str = "") -> int:
        """
        Append a new version and make it the creation's current text.

        Returns the new version number. Earlier versions are untouched.

        Raises
        ------
        KeyError
            If *creation_id* does not exist.
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT current_version FROM creations WHERE id=?",
                (creation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"No creation with id {creation_id}.")
            latest = self._conn.execute(
                "SELECT MAX(version) AS v FROM creation_versions "
                "WHERE creation_id=?",
                (creation_id,),
            ).fetchone()["v"] or row["current_version"] or 0
            version = latest + 1
            self._conn.execute(
                """
                INSERT INTO creation_versions
                    (creation_id, version, content, note, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (creation_id, version, content, note, metadata),
            )
            self._conn.execute(
                "UPDATE creations SET content=?, current_version=? WHERE id=?",
                (content, version, creation_id),
            )
            return version

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, creation_id: int) -> dict | None:
        cur = self._conn.execute(
            "SELECT * FROM creations WHERE id=?", (creation_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_versions(self, creation_id: int) -> list[dict]:
        """All versions of a creation, oldest first."""
        cur = self._conn.execute(
            """
            SELECT * FROM creation_versions
            WHERE creation_id=? ORDER BY version ASC
            """,
            (creation_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_version(self, creation_id: int, version: int) -> dict | None:
        cur = self._conn.execute(
            "SELECT * FROM creation_versions WHERE creation_id=? AND version=?",
            (creation_id, version),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_recent(self, type_: str, limit: int = 10) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM creations
            WHERE type=?
            ORDER BY logged_at DESC, id DESC LIMIT ?
            """,
            (type_, limit)
        )
        return [dict(r) for r in cur.fetchall()]

    def get_all(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM creations
            ORDER BY logged_at DESC, id DESC LIMIT ?
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
        query += " ORDER BY logged_at DESC, id DESC LIMIT ?"
        params.append(limit)
        cur = self._conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]
