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

CREATE TABLE IF NOT EXISTS weight_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    weight_kg   REAL    NOT NULL,
    notes       TEXT,
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS water_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    amount_ml   INTEGER NOT NULL,
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS health_goals (
    goal_type    TEXT PRIMARY KEY,
    target_value REAL NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_workouts_logged_at  ON workouts(logged_at);
CREATE INDEX IF NOT EXISTS idx_sleep_logged_at     ON sleep_logs(logged_at);
CREATE INDEX IF NOT EXISTS idx_mood_logged_at      ON mood_logs(logged_at);
CREATE INDEX IF NOT EXISTS idx_weight_logged_at    ON weight_logs(logged_at);
CREATE INDEX IF NOT EXISTS idx_water_logged_at     ON water_logs(logged_at);
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

    # ── weight ────────────────────────────────────────────

    def log_weight(self, weight_kg: float, notes: str) -> int:
        """Weight is always persisted in kg; unit conversion happens in the engine."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO weight_logs (weight_kg, notes) VALUES (?, ?)",
                (weight_kg, notes)
            )
            return cur.lastrowid

    def get_weight(self, days: int = 30) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM weight_logs
            WHERE logged_at >= datetime('now', ? || ' days')
            ORDER BY logged_at DESC
            """,
            (f"-{days}",)
        )
        return [dict(r) for r in cur.fetchall()]

    def latest_weight(self) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM weight_logs ORDER BY logged_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def previous_weight(self) -> Optional[dict]:
        """The second-most-recent entry, used to compute a log-to-log delta."""
        cur = self._conn.execute(
            "SELECT * FROM weight_logs ORDER BY logged_at DESC LIMIT 1 OFFSET 1"
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # ── water ─────────────────────────────────────────────

    def log_water(self, amount_ml: int) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO water_logs (amount_ml) VALUES (?)",
                (amount_ml,)
            )
            return cur.lastrowid

    def water_today(self) -> int:
        cur = self._conn.execute(
            """
            SELECT COALESCE(SUM(amount_ml), 0) FROM water_logs
            WHERE date(logged_at) = date('now')
            """
        )
        return int(cur.fetchone()[0])

    def get_water(self, days: int = 7) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM water_logs
            WHERE logged_at >= datetime('now', ? || ' days')
            ORDER BY logged_at DESC
            """,
            (f"-{days}",)
        )
        return [dict(r) for r in cur.fetchall()]

    # ── health goals ──────────────────────────────────────

    def set_goal(self, goal_type: str, target_value: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO health_goals (goal_type, target_value)
                VALUES (?, ?)
                ON CONFLICT(goal_type) DO UPDATE SET
                    target_value = excluded.target_value,
                    updated_at   = CURRENT_TIMESTAMP
                """,
                (goal_type, target_value)
            )

    def get_goal(self, goal_type: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM health_goals WHERE goal_type = ?", (goal_type,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_goals(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM health_goals ORDER BY goal_type")
        return [dict(r) for r in cur.fetchall()]

    # ── streaks ───────────────────────────────────────────

    def workout_dates(self, days: int = 60) -> list[str]:
        """Distinct calendar dates (YYYY-MM-DD, newest first) with >=1 workout."""
        cur = self._conn.execute(
            """
            SELECT DISTINCT date(logged_at) AS d FROM workouts
            WHERE logged_at >= datetime('now', ? || ' days')
            ORDER BY d DESC
            """,
            (f"-{days}",)
        )
        return [r["d"] for r in cur.fetchall()]