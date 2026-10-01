"""
modules/mnemosyne/spaced_repetition.py

Spaced-repetition scheduling (SM-2) for facts tagged as study material
(backlog #33).

Design
------
* ``sm2_update`` is a pure function: (ease, interval, reps, quality) in,
  new scheduling state out. Every constant is visible and unit-tested; no
  clock, no database.
* ``StudyStore`` persists one "card" per tagged fact. A card is just a
  pointer (``fact_key``) plus scheduling state — the fact's text stays in
  Mnemosyne's ``facts`` table, so editing or forgetting a fact never leaves
  a stale copy here. The engine removes a card when its fact is forgotten.
* Like ``core/quiz_store.py`` this owns its own table (created with
  ``CREATE TABLE IF NOT EXISTS`` on open), so an existing Mnemosyne
  database needs no migration.

Dates are plain ``YYYY-MM-DD`` strings and every method that needs "today"
takes it as a parameter (default: the machine's local date), which is what
makes the due-today logic testable without patching the clock.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# SM-2 constants (Wozniak, 1990).
_INITIAL_EASE = 2.5
_MIN_EASE = 1.3
_FIRST_INTERVAL_DAYS = 1
_SECOND_INTERVAL_DAYS = 6
_PASS_QUALITY = 3          # quality >= 3 is a successful recall

# Replies that mean "I don't know" — graded as a lapse without pretending the
# text was an attempt at the answer.
SKIP_WORDS = frozenset({
    "skip", "pass", "idk", "i don't know", "i dont know", "don't know",
    "dont know", "no idea", "not sure", "forgot", "i forgot",
})

# Similarity thresholds for grade_recall (difflib ratio on normalised text).
_GOOD_RATIO = 0.85
_HARD_RATIO = 0.55


def sm2_update(ease: float, interval_days: int, reps: int, quality: int) -> dict:
    """
    Apply one SM-2 review.

    Returns ``{"ease", "interval_days", "reps", "lapsed"}``. A quality
    below 3 is a lapse: repetitions reset and the card comes back tomorrow
    (ease is still adjusted, so a card you keep failing gets reviewed more
    often than an easy one once it recovers).
    """
    quality = max(0, min(5, int(quality)))
    ease = float(ease) if ease else _INITIAL_EASE

    # The ease update applies to every review, pass or fail.
    ease = max(
        _MIN_EASE,
        ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
    )

    if quality < _PASS_QUALITY:
        return {
            "ease": round(ease, 4),
            "interval_days": _FIRST_INTERVAL_DAYS,
            "reps": 0,
            "lapsed": True,
        }

    if reps == 0:
        interval = _FIRST_INTERVAL_DAYS
    elif reps == 1:
        interval = _SECOND_INTERVAL_DAYS
    else:
        interval = max(1, round(interval_days * ease))

    return {
        "ease": round(ease, 4),
        "interval_days": interval,
        "reps": reps + 1,
        "lapsed": False,
    }


def _norm(text: str) -> str:
    return " ".join(
        "".join(c if c.isalnum() or c.isspace() else " " for c in (text or "").lower()).split()
    )


def grade_recall(answer: str, expected: str) -> int:
    """
    Auto-grade a free-text recall attempt into an SM-2 quality (0-5).

    Deliberately simple and deterministic so it is predictable by voice:
    an (almost) exact match is "good" (or "easy" when it is exact), a
    partial match is "hard", anything else — or an explicit "I don't
    know" — is a lapse. The caller always reveals the expected answer, so
    a harsh grade is corrected in front of the user rather than hidden.
    """
    import difflib

    a, e = _norm(answer), _norm(expected)
    if not a or a in SKIP_WORDS or not e:
        return 1
    if a == e:
        return 5
    if e in a.split() or e in a:
        # The answer contains the expected text ("it's Priya" vs "priya").
        return 4
    ratio = difflib.SequenceMatcher(None, a, e).ratio()
    if ratio >= _GOOD_RATIO:
        return 4
    if ratio >= _HARD_RATIO:
        return 3
    return 1


def _today(today: Optional[date]) -> date:
    return today or date.today()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS study_cards (
    fact_key TEXT PRIMARY KEY,
    subject TEXT,
    ease REAL DEFAULT 2.5,
    interval_days INTEGER DEFAULT 0,
    reps INTEGER DEFAULT 0,
    lapses INTEGER DEFAULT 0,
    due_date TEXT,              -- YYYY-MM-DD; a new card is due immediately
    last_reviewed TEXT,         -- YYYY-MM-DD
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_study_cards_due ON study_cards(due_date);
"""


class StudyStore:
    """SQLite-backed scheduling state for study cards."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # -- tagging ---------------------------------------------------------

    def add_card(
        self, fact_key: str, subject: Optional[str] = None,
        today: Optional[date] = None,
    ) -> bool:
        """Tag *fact_key* as study material. Returns False if already tagged."""
        if not fact_key:
            raise ValueError("add_card() requires a non-empty fact_key.")
        day = _today(today).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO study_cards
                    (fact_key, subject, ease, interval_days, reps, lapses,
                     due_date, created_at)
                VALUES (?, ?, ?, 0, 0, 0, ?, ?)
                """,
                (fact_key, subject, _INITIAL_EASE, day, day),
            )
            return cur.rowcount > 0

    def remove_card(self, fact_key: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM study_cards WHERE fact_key = ?", (fact_key,)
            )
            return cur.rowcount > 0

    def get_card(self, fact_key: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM study_cards WHERE fact_key = ?", (fact_key,)
        ).fetchone()
        return dict(row) if row else None

    def list_cards(self, limit: int = 200) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM study_cards ORDER BY due_date ASC, fact_key ASC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    # -- scheduling ------------------------------------------------------

    def due_cards(self, today: Optional[date] = None, limit: int = 50) -> list[dict]:
        """Cards due on or before *today*, most overdue first."""
        day = _today(today).isoformat()
        cur = self._conn.execute(
            """
            SELECT * FROM study_cards
            WHERE due_date <= ?
            ORDER BY due_date ASC, reps ASC, fact_key ASC
            LIMIT ?
            """,
            (day, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def review(
        self, fact_key: str, quality: int, today: Optional[date] = None
    ) -> Optional[dict]:
        """
        Record a review and reschedule. Returns the updated card, or None
        if *fact_key* isn't a study card (never raises for an unknown key —
        a misheard reference must not crash a review session).
        """
        card = self.get_card(fact_key)
        if card is None:
            return None
        day = _today(today)
        result = sm2_update(
            card["ease"], card["interval_days"], card["reps"], quality
        )
        due = day + timedelta(days=result["interval_days"])
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE study_cards
                SET ease = ?, interval_days = ?, reps = ?, lapses = lapses + ?,
                    due_date = ?, last_reviewed = ?
                WHERE fact_key = ?
                """,
                (
                    result["ease"], result["interval_days"], result["reps"],
                    1 if result["lapsed"] else 0,
                    due.isoformat(), day.isoformat(), fact_key,
                ),
            )
        return self.get_card(fact_key)

    def stats(self, today: Optional[date] = None) -> dict:
        day = _today(today).isoformat()
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN due_date <= ? THEN 1 ELSE 0 END) AS due,
                   SUM(CASE WHEN reps = 0 THEN 1 ELSE 0 END) AS new_cards
            FROM study_cards
            """,
            (day,),
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "due": row["due"] or 0,
            "new": row["new_cards"] or 0,
        }
