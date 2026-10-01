# modules/apollo/db.py

import sqlite3
import threading
from pathlib import Path
from typing import Optional


# ── schema migrations ─────────────────────────────────────
#
# The base tables above use CREATE IF NOT EXISTS, which can't add columns to
# an existing apollo.db. Migrations are keyed off PRAGMA user_version: each
# runs once, in order, inside a transaction, and is written to be idempotent
# (columns are only added when missing) so a half-applied or re-run migration
# is harmless. Never edit a shipped migration; append a new one.

def _add_column_if_missing(conn, table: str, column: str, decl: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _migrate_1(conn) -> None:
    """Backlog #111-#120: sleep times/rating, goal pace, profile, meals,
    pain, imported steps, reminder state."""
    _add_column_if_missing(conn, "sleep_logs", "bed_time", "TEXT")
    _add_column_if_missing(conn, "sleep_logs", "wake_time", "TEXT")
    _add_column_if_missing(conn, "sleep_logs", "rating", "INTEGER")
    _add_column_if_missing(conn, "health_goals", "start_value", "REAL")
    _add_column_if_missing(conn, "health_goals", "deadline", "TEXT")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS profile (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS meal_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            kcal      REAL,
            protein_g REAL,
            grams     REAL,
            estimated INTEGER DEFAULT 0,
            source    TEXT,
            notes     TEXT,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pain_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            area      TEXT NOT NULL,
            severity  INTEGER NOT NULL,
            notes     TEXT,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS step_logs (
            day         TEXT PRIMARY KEY,
            steps       INTEGER NOT NULL,
            source      TEXT,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS reminder_state (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meal_logged_at ON meal_logs(logged_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pain_logged_at ON pain_logs(logged_at)")


MIGRATIONS = [_migrate_1]


class ApolloDB:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()
        self._migrate()

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        for target, step in enumerate(MIGRATIONS, start=1):
            if target <= version:
                continue
            with self._conn:
                step(self._conn)
                self._conn.execute(f"PRAGMA user_version = {target}")

    @property
    def schema_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

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

    def log_workout(self, type_: str, duration: int, notes: str,
                    logged_at: Optional[str] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO workouts (type, duration, notes, logged_at) "
                "VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
                (type_, duration, notes, logged_at)
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

    def log_sleep(self, hours: float, quality: str, notes: str,
                  bed_time: Optional[str] = None,
                  wake_time: Optional[str] = None,
                  rating: Optional[int] = None,
                  logged_at: Optional[str] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO sleep_logs "
                "(hours, quality, notes, bed_time, wake_time, rating, logged_at) "
                "VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
                (hours, quality, notes, bed_time, wake_time, rating, logged_at)
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

    def log_mood(self, mood: str, notes: str,
                 logged_at: Optional[str] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO mood_logs (mood, notes, logged_at) "
                "VALUES (?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
                (mood, notes, logged_at)
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

    def log_weight(self, weight_kg: float, notes: str,
                   logged_at: Optional[str] = None) -> int:
        """Weight is always persisted in kg; unit conversion happens in the engine."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO weight_logs (weight_kg, notes, logged_at) "
                "VALUES (?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
                (weight_kg, notes, logged_at)
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

    def log_water(self, amount_ml: int, logged_at: Optional[str] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO water_logs (amount_ml, logged_at) "
                "VALUES (?, COALESCE(?, CURRENT_TIMESTAMP))",
                (amount_ml, logged_at)
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

    def set_goal(self, goal_type: str, target_value: float,
                 start_value: Optional[float] = None,
                 deadline: Optional[str] = None) -> None:
        """Upsert a goal. ``start_value``/``deadline`` are kept from the
        previous row when not supplied, so re-setting a target doesn't wipe
        the pace baseline."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO health_goals (goal_type, target_value, start_value, deadline)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(goal_type) DO UPDATE SET
                    target_value = excluded.target_value,
                    start_value  = COALESCE(excluded.start_value, health_goals.start_value),
                    deadline     = COALESCE(excluded.deadline, health_goals.deadline),
                    updated_at   = CURRENT_TIMESTAMP
                """,
                (goal_type, target_value, start_value, deadline)
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

    # ── profile (units and settings) ──────────────────────

    def get_profile(self) -> dict:
        cur = self._conn.execute("SELECT key, value FROM profile")
        return {r["key"]: r["value"] for r in cur.fetchall()}

    def set_profile(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO profile (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (key, str(value))
            )

    # ── meals ─────────────────────────────────────────────

    def log_meal(self, name: str, kcal: Optional[float],
                 protein_g: Optional[float] = None,
                 grams: Optional[float] = None, estimated: bool = False,
                 source: Optional[str] = None, notes: Optional[str] = None,
                 logged_at: Optional[str] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO meal_logs "
                "(name, kcal, protein_g, grams, estimated, source, notes, logged_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
                (name, kcal, protein_g, grams, 1 if estimated else 0,
                 source, notes, logged_at)
            )
            return cur.lastrowid

    def get_meals(self, days: int = 7) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM meal_logs
            WHERE logged_at >= datetime('now', ? || ' days')
            ORDER BY logged_at DESC
            """,
            (f"-{days}",)
        )
        return [dict(r) for r in cur.fetchall()]

    # ── pain / injury ─────────────────────────────────────

    def log_pain(self, area: str, severity: int, notes: Optional[str] = None,
                 logged_at: Optional[str] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO pain_logs (area, severity, notes, logged_at) "
                "VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
                (area, severity, notes, logged_at)
            )
            return cur.lastrowid

    def get_pain(self, days: int = 30, area: Optional[str] = None) -> list[dict]:
        sql = ("SELECT * FROM pain_logs "
               "WHERE logged_at >= datetime('now', ? || ' days')")
        params: list = [f"-{days}"]
        if area:
            sql += " AND lower(area) = lower(?)"
            params.append(area)
        cur = self._conn.execute(sql + " ORDER BY logged_at DESC", params)
        return [dict(r) for r in cur.fetchall()]

    # ── imported steps ────────────────────────────────────

    def upsert_steps(self, day: str, steps: int,
                     source: Optional[str] = None) -> str:
        """Insert or replace one day's steps. Returns inserted/updated/unchanged
        so re-importing the same file is idempotent and reports honestly."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT steps FROM step_logs WHERE day = ?", (day,)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO step_logs (day, steps, source) VALUES (?, ?, ?)",
                    (day, steps, source)
                )
                return "inserted"
            if int(row["steps"]) == int(steps):
                return "unchanged"
            self._conn.execute(
                "UPDATE step_logs SET steps = ?, source = ?, "
                "imported_at = CURRENT_TIMESTAMP WHERE day = ?",
                (steps, source, day)
            )
            return "updated"

    def get_steps(self, days: int = 30) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM step_logs
            WHERE day >= date('now', ? || ' days')
            ORDER BY day DESC
            """,
            (f"-{days}",)
        )
        return [dict(r) for r in cur.fetchall()]

    # ── reminder state (heartbeat markers) ────────────────

    def get_state(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM reminder_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO reminder_state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (key, value)
            )
