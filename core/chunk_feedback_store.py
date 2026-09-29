"""
core/chunk_feedback_store.py

Per-chunk relevance feedback storage (backlog #63): "mark this source as
not helpful" should make that chunk less likely to surface again, without
ever fully excluding it (a chunk irrelevant to one question can still be
exactly right for a different one).

Kept as its own small SQLite store — mirrors core/quiz_store.py's
conventions (stdlib sqlite3, one purpose, no shared schema with anything
else) rather than bolting feedback columns onto Athena's ChromaDB
metadata, which isn't a great fit for frequently-updated counters.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk_feedback (
    chunk_id TEXT PRIMARY KEY,
    relevant_count INTEGER DEFAULT 0,
    irrelevant_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class ChunkFeedbackStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def record(self, chunk_id: str, relevant: bool) -> None:
        column = "relevant_count" if relevant else "irrelevant_count"
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                INSERT INTO chunk_feedback (chunk_id, {column})
                VALUES (?, 1)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    {column} = {column} + 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chunk_id,),
            )

    def get(self, chunk_id: str) -> dict:
        cur = self._conn.execute(
            "SELECT relevant_count, irrelevant_count FROM chunk_feedback WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        if row is None:
            return {"relevant_count": 0, "irrelevant_count": 0}
        return {"relevant_count": row["relevant_count"], "irrelevant_count": row["irrelevant_count"]}

    def get_all(self) -> dict[str, dict]:
        """Every chunk with any recorded feedback, keyed by chunk_id — used to
        batch-load counts once per search rather than one query per result."""
        cur = self._conn.execute(
            "SELECT chunk_id, relevant_count, irrelevant_count FROM chunk_feedback"
        )
        return {
            row["chunk_id"]: {
                "relevant_count": row["relevant_count"],
                "irrelevant_count": row["irrelevant_count"],
            }
            for row in cur.fetchall()
        }
