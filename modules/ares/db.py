# modules/ares/db.py
"""
Small SQLite store for the parts of Ares that need to remember things
between sessions, beyond what Mnemosyne facts can hold:

  decisions   every plan / decision / career ranking Ares produced, so it
              can be revisited later (#154) and given an outcome (#155).
  playbooks   saved, named analysis templates (#157).

Same conventions as modules/dionysus/db.py: one connection, one lock,
WAL when on disk, schema created idempotently on open.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone

# Words that carry no signal when matching "that job decision" to a saved row.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "about", "decision", "decisions", "plan",
    "plans", "my", "this", "that", "our", "into", "from", "what", "how",
    "outcome", "result", "revisit", "review", "went", "turned", "out",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


class AresDB:
    def __init__(self, db_path: str = ":memory:"):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            with self._conn:
                self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript("""
CREATE TABLE IF NOT EXISTS decisions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    kind                 TEXT NOT NULL,          -- plan | decision | career | manual
    topic                TEXT NOT NULL,
    summary              TEXT,
    options              TEXT,                   -- JSON list, optional
    predicted_confidence REAL,                   -- 0..1, optional
    created_at           TEXT NOT NULL,
    review_at            TEXT,
    outcome              TEXT,                   -- success | mixed | failure
    outcome_note         TEXT,
    outcome_at           TEXT
);
CREATE INDEX IF NOT EXISTS idx_ares_decisions_outcome ON decisions(outcome);

CREATE TABLE IF NOT EXISTS playbooks (
    name_key   TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    analysis   TEXT NOT NULL,
    criteria   TEXT,
    topic      TEXT,
    options    TEXT,
    created_at TEXT NOT NULL,
    last_used  TEXT,
    uses       INTEGER NOT NULL DEFAULT 0
);
""")

    # ── decisions ────────────────────────────────────────────────────────────

    def add_decision(
        self,
        kind: str,
        topic: str,
        summary: str = "",
        options: list | None = None,
        predicted_confidence: float | None = None,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO decisions (kind, topic, summary, options, "
                "predicted_confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    kind, topic, summary,
                    json.dumps(options) if options else None,
                    predicted_confidence, _now(),
                ),
            )
            return cur.lastrowid

    def get_decision(self, decision_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
        return dict(row) if row else None

    def find_decision(self, topic: str, open_only: bool = False) -> dict | None:
        """
        Best match for *topic* among the 200 most recent rows: most shared
        meaningful words wins, newest wins ties. With no topic, returns the
        newest row. None when nothing overlaps.
        """
        where = "WHERE outcome IS NULL" if open_only else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM decisions {where} ORDER BY id DESC LIMIT 200"
            ).fetchall()
        if not rows:
            return None
        want = _tokens(topic)
        if not want:
            return dict(rows[0])
        best, best_score = None, 0
        for row in rows:  # newest first, so strict > keeps the newest on ties
            score = len(want & _tokens(row["topic"]))
            if score > best_score:
                best, best_score = row, score
        return dict(best) if best is not None else None

    def set_review(self, decision_id: int, review_at: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE decisions SET review_at = ? WHERE id = ?",
                (review_at, decision_id),
            )

    def record_outcome(self, decision_id: int, outcome: str, note: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE decisions SET outcome = ?, outcome_note = ?, outcome_at = ? "
                "WHERE id = ?",
                (outcome, note, _now(), decision_id),
            )

    def resolved_with_confidence(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT predicted_confidence, outcome FROM decisions "
                "WHERE outcome IS NOT NULL AND predicted_confidence IS NOT NULL"
            ).fetchall()
        return [dict(r) for r in rows]

    def recent(self, limit: int = 5) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def outcome_counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT COALESCE(outcome, 'pending') AS o, COUNT(*) AS n "
                "FROM decisions GROUP BY o"
            ).fetchall()
        return {r["o"]: r["n"] for r in rows}

    def due_reviews(self, now_iso: str) -> list[dict]:
        """Unresolved decisions whose review date has passed."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM decisions WHERE outcome IS NULL "
                "AND review_at IS NOT NULL AND julianday(review_at) <= julianday(?) "
                "ORDER BY review_at",
                (now_iso,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── playbooks ────────────────────────────────────────────────────────────

    @staticmethod
    def _key(name: str) -> str:
        return re.sub(r"\s+", " ", (name or "").strip().lower())

    def save_playbook(
        self,
        name: str,
        analysis: str,
        criteria: str = "",
        topic: str = "",
        options: str = "",
    ) -> bool:
        """Insert or replace. Returns True if a playbook of that name already existed."""
        key = self._key(name)
        with self._lock, self._conn:
            existed = self._conn.execute(
                "SELECT 1 FROM playbooks WHERE name_key = ?", (key,)
            ).fetchone() is not None
            self._conn.execute(
                "INSERT INTO playbooks (name_key, name, analysis, criteria, topic, "
                "options, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name_key) DO UPDATE SET name=excluded.name, "
                "analysis=excluded.analysis, criteria=excluded.criteria, "
                "topic=excluded.topic, options=excluded.options",
                (key, name.strip(), analysis, criteria, topic, options, _now()),
            )
        return existed

    def get_playbook(self, name: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM playbooks WHERE name_key = ?", (self._key(name),)
            ).fetchone()
        return dict(row) if row else None

    def list_playbooks(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM playbooks ORDER BY name_key"
            ).fetchall()
        return [dict(r) for r in rows]

    def find_playbook_in_text(self, text: str) -> dict | None:
        """Longest saved playbook name that appears in *text*, if any."""
        hay = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
        best = None
        for pb in self.list_playbooks():
            if f" {pb['name_key']} " in hay or pb["name_key"] in hay:
                if best is None or len(pb["name_key"]) > len(best["name_key"]):
                    best = pb
        return best

    def touch_playbook(self, name: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE playbooks SET uses = uses + 1, last_used = ? WHERE name_key = ?",
                (_now(), self._key(name)),
            )

    def delete_playbook(self, name: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM playbooks WHERE name_key = ?", (self._key(name),)
            )
            return cur.rowcount > 0
