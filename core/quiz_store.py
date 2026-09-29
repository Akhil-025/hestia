"""
core/quiz_store.py

SQLite storage for quiz questions and attempt history (backlog #34, #35).

Kept independent of modules/mnemosyne/ deliberately: quiz questions and
attempts are a genuinely different data shape (multiple-choice options,
correct-index, per-subject scoring) from Mnemosyne's key/value facts, and
Mnemosyne's schema/db.py file is already large. A separate, small,
single-purpose store — mirroring MnemosyneDB's own conventions (stdlib
sqlite3, threading.Lock, row_factory=Row) so it reads as consistent with
the rest of the codebase rather than inventing a new pattern.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quiz_questions (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    question TEXT NOT NULL,
    choices TEXT NOT NULL,      -- JSON list of strings
    correct_index INTEGER NOT NULL,
    source_snippet TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES quiz_questions(id),
    subject TEXT NOT NULL,
    correct INTEGER NOT NULL,   -- 0/1
    chosen_index INTEGER NOT NULL,
    answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_quiz_questions_subject ON quiz_questions(subject);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_subject ON quiz_attempts(subject);
"""


class QuizStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # -- questions --------------------------------------------------------

    def add_question(
        self, subject: str, question: str, choices: list[str],
        correct_index: int, source_snippet: str = "",
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO quiz_questions (subject, question, choices, correct_index, source_snippet)
                VALUES (?, ?, ?, ?, ?)
                """,
                (subject, question, json.dumps(choices), correct_index, source_snippet),
            )
            return cur.lastrowid

    def get_question(self, question_id: int) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM quiz_questions WHERE id = ?", (question_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["choices"] = json.loads(d["choices"])
        return d

    def get_questions_by_subject(self, subject: str, limit: int = 20) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM quiz_questions WHERE subject = ? ORDER BY created_at DESC LIMIT ?",
            (subject, limit),
        )
        out = []
        for row in cur.fetchall():
            d = dict(row)
            d["choices"] = json.loads(d["choices"])
            out.append(d)
        return out

    # -- attempts -----------------------------------------------------------

    def record_attempt(
        self, question_id: int, subject: str, correct: bool, chosen_index: int
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO quiz_attempts (question_id, subject, correct, chosen_index)
                VALUES (?, ?, ?, ?)
                """,
                (question_id, subject, int(correct), chosen_index),
            )
            return cur.lastrowid

    # -- scoring / strength-weakness map (#35) -------------------------------

    def get_subject_stats(self) -> list[dict]:
        """Per-subject attempt counts and accuracy, every subject ever attempted."""
        cur = self._conn.execute(
            """
            SELECT subject,
                   COUNT(*) AS attempts,
                   SUM(correct) AS correct
            FROM quiz_attempts
            GROUP BY subject
            ORDER BY subject
            """
        )
        out = []
        for row in cur.fetchall():
            attempts = row["attempts"]
            correct = row["correct"] or 0
            out.append({
                "subject": row["subject"],
                "attempts": attempts,
                "correct": correct,
                "accuracy": round(correct / attempts, 3) if attempts else 0.0,
            })
        return out

    def get_subject_stats_over_time(self, subject: str) -> list[dict]:
        """
        Attempts for one subject in chronological order — the raw series
        a "how am I trending" view or a future chart would consume.
        """
        cur = self._conn.execute(
            """
            SELECT correct, answered_at FROM quiz_attempts
            WHERE subject = ? ORDER BY answered_at ASC
            """,
            (subject,),
        )
        return [dict(row) for row in cur.fetchall()]
