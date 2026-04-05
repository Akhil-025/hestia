# modules/apollo/db.py

import sqlite3
import threading
from pathlib import Path
from typing import Optional


class ApolloDB:
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
CREATE TABLE IF NOT EXISTS workouts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT,
    duration    INTEGER,
    notes       TEXT,
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sleep_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    hours       REAL    NOT NULL,
    quality     TEXT,
    notes       TEXT,
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mood_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mood        TEXT    NOT NULL,
    notes       TEXT,
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_workouts_logged_at  ON workouts(logged_at);
CREATE INDEX IF NOT EXISTS idx_sleep_logged_at     ON sleep_logs(logged_at);
CREATE INDEX IF NOT EXISTS idx_mood_logged_at      ON mood_logs(logged_at);
""")

    # ── workouts ─────────────────────────────────────────

    def log_workout(self, type_: str, duration: int, notes: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO workouts (type, duration, notes) VALUES (?, ?, ?)",
                (type_, duration, notes)
            )
            return cur.lastrowid

    def get_workouts(self, days: int = 7) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM workouts
            WHERE logged_at >= datetime('now', ? || ' days')
            ORDER BY logged_at DESC
            """,
            (f"-{days}",)
        )
        return [dict(r) for r in cur.fetchall()]

    def workout_count(self, days: int = 7) -> int:
        cur = self._conn.execute(
            """
            SELECT COUNT(*) FROM workouts
            WHERE logged_at >= datetime('now', ? || ' days')
            """,
            (f"-{days}",)
        )
        return cur.fetchone()[0]

    # ── sleep ─────────────────────────────────────────────

    def log_sleep(self, hours: float, quality: str, notes: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO sleep_logs (hours, quality, notes) VALUES (?, ?, ?)",
                (hours, quality, notes)
            )
            return cur.lastrowid

    def get_sleep(self, days: int = 7) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM sleep_logs
            WHERE logged_at >= datetime('now', ? || ' days')
            ORDER BY logged_at DESC
            """,
            (f"-{days}",)
        )
        return [dict(r) for r in cur.fetchall()]

    def avg_sleep(self, days: int = 7) -> Optional[float]:
        cur = self._conn.execute(
            """
            SELECT AVG(hours) FROM sleep_logs
            WHERE logged_at >= datetime('now', ? || ' days')
            """,
            (f"-{days}",)
        )
        val = cur.fetchone()[0]
        return round(val, 1) if val else None

    # ── mood ──────────────────────────────────────────────

    def log_mood(self, mood: str, notes: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO mood_logs (mood, notes) VALUES (?, ?)",
                (mood, notes)
            )
            return cur.lastrowid

    def get_mood(self, days: int = 7) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM mood_logs
            WHERE logged_at >= datetime('now', ? || ' days')
            ORDER BY logged_at DESC
            """,
            (f"-{days}",)
        )
        return [dict(r) for r in cur.fetchall()]

    def recent_moods(self, limit: int = 5) -> list[str]:
        cur = self._conn.execute(
            "SELECT mood FROM mood_logs ORDER BY logged_at DESC LIMIT ?",
            (limit,)
        )
        return [r["mood"] for r in cur.fetchall()]