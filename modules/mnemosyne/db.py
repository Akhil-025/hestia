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
    _MAX_USER_TEXT = 10_000
    _MAX_RESPONSE = 50_000
    _MAX_INTENT = 256

    def push_interaction(self, user_text, hestia_response, intent, source_device="hestia") -> int:
        user_text = str(user_text)[: self._MAX_USER_TEXT]
        hestia_response = str(hestia_response)[: self._MAX_RESPONSE]
        intent = str(intent)[: self._MAX_INTENT]
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
                INSERT INTO facts (key, value, source, confidence, created_at, updated_at, last_accessed)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    source=excluded.source,
                    confidence=excluded.confidence,
                    updated_at=CURRENT_TIMESTAMP,
                    last_accessed=CURRENT_TIMESTAMP,
                    stale=0
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

    def get_fact_row(self, key) -> Optional[dict]:
        """Full metadata for one fact (value/source/confidence/created_at/...), not just its value."""
        cur = self._conn.execute(
            """
            SELECT key, value, source, confidence, created_at, updated_at,
                   last_accessed, access_count, importance, stale
            FROM facts WHERE key = ?
            """,
            (key,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_facts(self, limit: int = 100, offset: int = 0) -> list[dict]:
        limit = max(1, min(limit, 1000))  # hard cap

        cur = self._conn.execute(
            """
            SELECT key, value, source, confidence, created_at, updated_at
            FROM facts
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        return [dict(row) for row in cur.fetchall()]

    def delete_fact(self, key) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM facts WHERE key = ?", (key,))

    # -- fact lifecycle: access tracking, decay, importance (#36, #45) --

    def touch_fact(self, key: str) -> None:
        """
        Record that *key* was just referenced/recalled: bump its access
        count, refresh last_accessed, and clear any stale flag — being
        recalled is direct evidence the fact is still relevant, which is
        exactly the signal the decay job (get_stale_facts) needs to not
        re-flag it next run.
        """
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE facts
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP,
                    stale = 0
                WHERE key = ?
                """,
                (key,),
            )

    def get_stale_facts(self, cutoff_iso: str) -> list[dict]:
        """Facts last accessed before *cutoff_iso* and not already flagged stale."""
        cur = self._conn.execute(
            """
            SELECT key, value, last_accessed, access_count
            FROM facts
            WHERE COALESCE(last_accessed, created_at) < ? AND stale = 0
            ORDER BY last_accessed ASC
            """,
            (cutoff_iso,),
        )
        return [dict(row) for row in cur.fetchall()]

    def flag_stale(self, keys: list[str]) -> int:
        """Mark the given fact keys stale=1. Returns the number updated."""
        if not keys:
            return 0
        with self._lock, self._conn:
            cur = self._conn.executemany(
                "UPDATE facts SET stale = 1 WHERE key = ? AND stale = 0",
                [(k,) for k in keys],
            )
            return cur.rowcount if cur.rowcount is not None else len(keys)

    def get_flagged_stale_facts(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT key, value, last_accessed
            FROM facts WHERE stale = 1
            ORDER BY last_accessed ASC LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def set_fact_importance(self, key: str, importance: float) -> bool:
        """Set an explicit importance weight (0..1). Returns False if the key doesn't exist."""
        importance = max(0.0, min(1.0, float(importance)))
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE facts SET importance = ? WHERE key = ?", (importance, key)
            )
            return cur.rowcount > 0

    def get_facts_for_scoring(self, limit: int = 200) -> list[dict]:
        """
        A candidate pool for importance-based ranking (#45): every field
        `MnemosyneEngine._score_fact` needs, capped at *limit* rows so the
        Python-side scoring pass (see engine.py — deliberately NOT done
        as a giant SQL expression, for testability and tunability) stays
        cheap even with a large fact table.
        """
        cur = self._conn.execute(
            """
            SELECT key, value, updated_at, access_count, importance, confidence
            FROM facts
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    # -- bulk operations (#41) -------------------------------------------

    def search_fact_keys(self, pattern: str) -> list[str]:
        """
        Keys matching *pattern* as a SQL LIKE substring (case-insensitive
        by SQLite's default LIKE collation for ASCII). Used by "forget
        everything about X" to find candidate keys before deleting.
        """
        cur = self._conn.execute(
            "SELECT key FROM facts WHERE key LIKE ? ORDER BY key", (f"%{pattern}%",)
        )
        return [row["key"] for row in cur.fetchall()]

    def delete_facts(self, keys: list[str]) -> int:
        """Delete multiple fact rows by key. Returns the number actually deleted."""
        if not keys:
            return 0
        with self._lock, self._conn:
            cur = self._conn.executemany(
                "DELETE FROM facts WHERE key = ?", [(k,) for k in keys]
            )
            return cur.rowcount if cur.rowcount is not None else 0

    # -- dated queries (#48) ----------------------------------------------

    def get_interactions_on_date(self, date_str: str) -> list[dict]:
        """
        Interactions whose pushed_at falls on the calendar date *date_str*
        (YYYY-MM-DD), independent of what timezone pushed_at was recorded
        in relative to the caller's — the comparison is a plain string
        prefix match against the ISO timestamp, matching how pushed_at is
        actually stored (CURRENT_TIMESTAMP, UTC, ISO-ish).
        """
        cur = self._conn.execute(
            """
            SELECT user_text, hestia_response, intent, pushed_at
            FROM interaction_log
            WHERE pushed_at LIKE ?
            ORDER BY id ASC
            """,
            (f"{date_str}%",),
        )
        return [
            {
                "query": r["user_text"], "response": r["hestia_response"],
                "intent": r["intent"], "pushed_at": r["pushed_at"],
            }
            for r in cur.fetchall()
        ]

    def get_facts_created_on_date(self, date_str: str) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT key, value, source, confidence, created_at
            FROM facts WHERE created_at LIKE ? ORDER BY created_at ASC
            """,
            (f"{date_str}%",),
        )
        return [dict(row) for row in cur.fetchall()]

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

    def get_interactions_since(self, since: str, limit: int = 1000) -> list[dict]:
        """
        Return up to `limit` interactions with pushed_at > since, oldest
        first. Unlike filtering a fixed-size `get_recent_interactions()`
        fetch client-side, this queries the backlog directly so a caller
        can page through it (by repeatedly calling with the last-seen
        pushed_at) without silently skipping rows when the backlog is
        larger than any single fetch.
        """
        cur = self._conn.execute(
            """
            SELECT user_text, hestia_response, intent, pushed_at
            FROM interaction_log
            WHERE pushed_at > ?
            ORDER BY id ASC LIMIT ?
            """,
            (since, limit)
        )
        rows = cur.fetchall()
        return [
            {"query": r["user_text"], "response": r["hestia_response"], "intent": r["intent"], "pushed_at": r["pushed_at"]}
            for r in rows
        ]


    def get_recent_interactions_excluding(self, limit: int, exclude_intents: list[str]) -> list[dict]:
        if not exclude_intents:
            return self.get_recent_interactions(limit)
        placeholders = ",".join("?" * len(exclude_intents))
        cur = self._conn.execute(
            f"""
            SELECT user_text, hestia_response, intent, pushed_at
            FROM interaction_log
            WHERE intent NOT IN ({placeholders})
            ORDER BY id DESC LIMIT ?
            """,
            (*exclude_intents, limit)
        )
        rows = cur.fetchall()
        result = [
            {"query": r["user_text"], "response": r["hestia_response"], "intent": r["intent"], "pushed_at": r["pushed_at"]}
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
    
    def delete_by_intent(self, intent: str) -> int:
        """Delete all interaction_log rows with the given intent. Returns rows deleted."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM interaction_log WHERE intent = ?",
                (intent,)
            )
            return cur.rowcount

    def get_top_facts(self, limit: int = 5):
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT key, value
                FROM facts
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,)
            )
            return [{"key": r[0], "value": r[1]} for r in cursor.fetchall()]

    def get_interaction_stats(self) -> dict:
        with self._lock:
            cur = self._conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN intent = 'take_note' THEN 1 ELSE 0 END) as notes,
                    COUNT(DISTINCT intent) as unique_intents
                FROM interaction_log
            """)
            row = cur.fetchone()
            return {
                "total": row[0] or 0,
                "notes": row[1] or 0,
                "unique_intents": row[2] or 0,
            }

    def get_memory_stats(self) -> dict:
        """Facts / active goals / summaries counts, queried under the DB lock."""
        with self._lock:
            facts = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            stale_facts = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE stale = 1"
            ).fetchone()[0]
            goals = self._conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status = 'active'"
            ).fetchone()[0]
            summaries = self._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            interactions = self._conn.execute(
                "SELECT COUNT(*) FROM interaction_log"
            ).fetchone()[0]
        # File size read outside the lock — os.stat doesn't touch the
        # connection, and WAL mode means the on-disk .db file is not the
        # full picture anyway (see get_db_size_bytes below for the honest
        # version that also counts -wal/-shm).
        return {
            "facts": facts,
            "stale_facts": stale_facts,
            "goals": goals,
            "summaries": summaries,
            "interactions": interactions,
        }

    def get_db_size_bytes(self) -> int:
        """
        Total on-disk size of the SQLite database, including the WAL and
        shared-memory sidecar files (`-wal`/`-shm`) — this connection runs
        in WAL mode (see __init__), so recently-written data can live in
        those files rather than the main `.db` file until the next
        checkpoint. Reporting only the main file's size would understate
        actual disk usage right after a burst of writes.
        """
        import os

        try:
            main_path = self._conn.execute("PRAGMA database_list").fetchone()[2]
        except Exception:
            return 0
        if not main_path:
            return 0
        total = 0
        for suffix in ("", "-wal", "-shm"):
            path = f"{main_path}{suffix}"
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
        return total

    # ── Reminders ─────────────────────────────────────

    def add_reminder(self, text, due_time):
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO reminders (text, due_time, status) VALUES (?, ?, 'pending')",
                (text, due_time)
            )

    def get_due_reminders(self, now_iso=None):
        if now_iso is None:
            from datetime import datetime
            now_iso = datetime.utcnow().isoformat()
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