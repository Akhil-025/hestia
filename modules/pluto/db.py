# modules/pluto/db.py

import sqlite3
import threading
from datetime import datetime
from typing import Optional

class PlutoDB:
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
CREATE TABLE IF NOT EXISTS expenses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    amount      REAL    NOT NULL,
    description TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS investments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    quantity    REAL,
    buy_price   REAL,
    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_expenses_category  ON expenses(category);
CREATE INDEX IF NOT EXISTS idx_expenses_logged_at ON expenses(logged_at);
CREATE INDEX IF NOT EXISTS idx_investments_name   ON investments(name);
""")

    # ── expenses ─────────────────────────────────────────

    def log_expense(self, amount: float, description: str, category: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO expenses (amount, description, category) VALUES (?, ?, ?)",
                (amount, description, category)
            )
            return cur.lastrowid

    def get_expenses(self, limit: int = 100) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM expenses ORDER BY logged_at DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]

    def get_totals_by_category(self) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT category,
                   SUM(amount)  AS total,
                   COUNT(*)     AS count
            FROM expenses
            GROUP BY category
            ORDER BY total DESC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def get_grand_total(self) -> float:
        cur = self._conn.execute("SELECT COALESCE(SUM(amount), 0) FROM expenses")
        return cur.fetchone()[0]

    # ── investments ──────────────────────────────────────

    def log_investment(self, name: str, type_: str,
                       quantity: float, buy_price: float) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO investments (name, type, quantity, buy_price)
                VALUES (?, ?, ?, ?)
                """,
                (name, type_, quantity, buy_price)
            )
            return cur.lastrowid

    def get_investments(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM investments ORDER BY logged_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]