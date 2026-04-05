"""
modules/mnemosyne/db.py

SQLite3 wrapper for Mnemosyne database. Thread-safe, no ORM, uses only standard library.
"""
import sqlite3
import threading
from pathlib import Path
from typing import Optional
from . import schema

class MnemosyneDB:
    def __init__(self, db_path: str):
        schema.init_db(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")

    # Interaction log
    def push_interaction(self, user_text, hestia_response, intent, source_device="hestia") -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO interaction_log (user_text, hestia_response, intent, source_device)
                VALUES (?, ?, ?, ?)
                """,
                (user_text, hestia_response, intent, source_device)
            )
            return cur.lastrowid

    def get_unsummarised(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM interaction_log WHERE summarised = 0 ORDER BY id ASC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cur.fetchall()]

    def mark_summarised(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE interaction_log SET summarised = 1 WHERE id = ?",
                [(i,) for i in ids]
            )

    # Facts
    def set_fact(self, key, value, source="user", confidence=1.0) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO facts (key, value, source, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    source=excluded.source,
                    confidence=excluded.confidence,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (key, value, source, confidence)
            )

    def get_fact(self, key) -> Optional[str]:
        cur = self._conn.execute(
            "SELECT value FROM facts WHERE key = ?",
            (key,)
        )
        row = cur.fetchone()
        return row["value"] if row else None

    def get_all_facts(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM facts")
        return [dict(row) for row in cur.fetchall()]

    def delete_fact(self, key) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM facts WHERE key = ?", (key,))

    # Summaries
    def add_summary(self, period_start, period_end, content, topic, interaction_count) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO summaries (period_start, period_end, content, topic, interaction_count, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (period_start, period_end, content, topic, interaction_count)
            )
            return cur.lastrowid

    def get_recent_summaries(self, n: int = 10) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM summaries ORDER BY period_start DESC LIMIT ?",
            (n,)
        )
        return [dict(row) for row in cur.fetchall()]

    # Goals
    def add_goal(self, text, due_date=None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO goals (text, status, due_date, created_at)
                VALUES (?, 'active', ?, CURRENT_TIMESTAMP)
                """,
                (text, due_date)
            )
            return cur.lastrowid

    def get_goals(self, status="active") -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM goals WHERE status = ? ORDER BY due_date ASC, created_at ASC",
            (status,)
        )
        return [dict(row) for row in cur.fetchall()]

    def complete_goal(self, goal_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE goals SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (goal_id,)
            )

    def cancel_goal(self, goal_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE goals SET status = 'cancelled' WHERE id = ?",
                (goal_id,)
            )

    # Semantic refs
    def add_semantic_ref(self, table_name, row_id, chroma_id, embedding_type) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO semantic_refs (table_name, row_id, chroma_id, embedding_type, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (table_name, row_id, chroma_id, embedding_type)
            )

    def get_chroma_id(self, table_name, row_id, embedding_type) -> Optional[str]:
        cur = self._conn.execute(
            """
            SELECT chroma_id FROM semantic_refs
            WHERE table_name = ? AND row_id = ? AND embedding_type = ?
            ORDER BY id DESC LIMIT 1
            """,
            (table_name, row_id, embedding_type)
        )
        row = cur.fetchone()
        return row["chroma_id"] if row else None
    
    def get_recent_interactions(self, limit: int = 5):
        cur = self._conn.execute(
            """
            SELECT user_text, hestia_response, intent, pushed_at
            FROM interaction_log
            ORDER BY id DESC LIMIT ?
            """,
            (limit,)
        )
        rows = cur.fetchall()
        result = [
            {"query": r["user_text"], "response": r["hestia_response"], "intent": r["intent"],"pushed_at": r["pushed_at"]}
            for r in rows
        ]
        result.reverse()
        return result


    def get_by_intent(self, intent: str, limit: int = 10):
        cur = self._conn.execute(
            """
            SELECT user_text, hestia_response, intent, pushed_at
            FROM interaction_log
            WHERE intent = ?
            ORDER BY id DESC LIMIT ?
            """,
            (intent, limit)
        )
        rows = cur.fetchall()
        result = [
            {
                "query": r["user_text"],
                "response": r["hestia_response"],
                "intent": r["intent"],
                "pushed_at": r["pushed_at"],  
            }
            for r in rows
        ]
        return result
            
    # ── Reminders ─────────────────────────────────────

    def add_reminder(self, text, due_time):
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO reminders (text, due_time, status) VALUES (?, ?, 'pending')",
                (text, due_time)
            )

    def get_due_reminders(self, now_iso):
        cur = self._conn.execute(
            "SELECT id, text FROM reminders WHERE due_time <= ? AND status = 'pending'",
            (now_iso,)
        )
        return [(row["id"], row["text"]) for row in cur.fetchall()]

    def mark_reminder_done(self, reminder_id):
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE reminders SET status = 'done' WHERE id = ?",
                (reminder_id,)
            )
