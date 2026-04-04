# core/memory.py

import sqlite3
import os
import datetime
import threading
from typing import Optional

class HestiaMemory:
    """SQLite-backed persistent memory for conversation history and user preferences."""

    def __init__(self, db_path="data/hestia.db"):
        """Initialize database connection, enable WAL mode, create tables and indexes."""
        folder = os.path.dirname(db_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._init_db()

    def _init_db(self):
        """Create tables and indexes if they do not exist."""
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            response TEXT,
            intent TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS preferences (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )''')
        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_intent ON interactions(intent)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)")
        c.execute('''CREATE TABLE IF NOT EXISTS user_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source TEXT DEFAULT 'inferred',
            created_at TEXT,
            updated_at TEXT
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS mood_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            valence TEXT NOT NULL,
            note TEXT,
            created_at TEXT
        )''')

        c.execute("CREATE INDEX IF NOT EXISTS idx_fact_key ON user_facts(key)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_mood_date ON mood_log(date)")
        self.conn.commit()

    def upsert_fact(self, key: str, value: str, confidence: float = 1.0, source: str = "inferred") -> None:
        """Insert or update a user fact by key."""
        ts = datetime.datetime.now().isoformat()
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT id FROM user_facts WHERE key = ?", (key,))
            row = c.fetchone()
            if row:
                c.execute(
                    "UPDATE user_facts SET value=?, confidence=?, source=?, updated_at=? WHERE key=?",
                    (value, confidence, source, ts, key)
                )
            else:
                c.execute(
                    "INSERT INTO user_facts (key, value, confidence, source, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                    (key, value, confidence, source, ts, ts)
                )
            self.conn.commit()

    def get_fact(self, key: str) -> Optional[str]:
        """Retrieve a user fact value by key. Returns None if not found."""
        c = self.conn.cursor()
        c.execute("SELECT value FROM user_facts WHERE key = ?", (key,))
        row = c.fetchone()
        return row[0] if row else None

    def get_all_facts(self, min_confidence: float = 0.5) -> list[dict]:
        """Return all user facts above confidence threshold, newest first."""
        c = self.conn.cursor()
        c.execute(
            "SELECT key, value, confidence, source, updated_at FROM user_facts WHERE confidence >= ? ORDER BY updated_at DESC",
            (min_confidence,)
        )
        rows = c.fetchall()
        return [{"key": r[0], "value": r[1], "confidence": r[2], "source": r[3], "updated_at": r[4]} for r in rows]

    def log_mood(self, valence: str, note: str = "") -> None:
        """
        Log today's mood valence.
        valence should be one of: "positive", "neutral", "negative", "stressed", "happy"
        If a mood for today already exists, update it.
        """
        ts = datetime.datetime.now().isoformat()
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT id FROM mood_log WHERE date = ?", (today,))
            row = c.fetchone()
            if row:
                c.execute(
                    "UPDATE mood_log SET valence=?, note=?, created_at=? WHERE date=?",
                    (valence, note, ts, today)
                )
            else:
                c.execute(
                    "INSERT INTO mood_log (date, valence, note, created_at) VALUES (?,?,?,?)",
                    (today, valence, note, ts)
                )
            self.conn.commit()

    def get_recent_moods(self, days: int = 7) -> list[dict]:
        """Return mood log entries for the last N days, newest first."""
        c = self.conn.cursor()
        c.execute(
            "SELECT date, valence, note FROM mood_log ORDER BY date DESC LIMIT ?",
            (days,)
        )
        rows = c.fetchall()
        return [{"date": r[0], "valence": r[1], "note": r[2]} for r in rows]

    def get_top_facts_for_context(self, limit: int = 10) -> str:
        """
        Return a formatted string of top user facts suitable for injecting
        into an LLU prompt context window.
        Format: "Known about user: key=value, key=value, ..."
        Returns empty string if no facts exist.
        """
        facts = self.get_all_facts()[:limit]
        if not facts:
            return ""
        parts = [f"{f['key']}={f['value']}" for f in facts]
        return "Known about user: " + ", ".join(parts)

    def add_interaction(self, query: str, response: str, intent: str) -> None:
        """Store a new interaction with timestamp."""
        ts = datetime.datetime.now().isoformat()
        with self._lock:
            c = self.conn.cursor()
            c.execute('''INSERT INTO interactions (timestamp, query, response, intent)
                         VALUES (?, ?, ?, ?)''', (ts, query, response, intent))
            self.conn.commit()

    def get_recent(self, limit: int = 5) -> list[dict]:
        """Return most recent interactions, oldest first."""
        c = self.conn.cursor()
        c.execute('''SELECT query, response, intent, timestamp FROM interactions
                     ORDER BY id DESC LIMIT ?''', (limit,))
        rows = c.fetchall()
        result = [{"query": r[0], "response": r[1], "intent": r[2], "timestamp": r[3]} for r in rows]
        result.reverse()
        return result

    def get_recent_filtered(self, limit: int = 5, exclude_intents: list = None) -> list:
        """Return most recent interactions, oldest first, optionally excluding certain intents."""
        c = self.conn.cursor()
        if exclude_intents:
            placeholders = ",".join("?" * len(exclude_intents))
            c.execute(
                f'''SELECT query, response, intent, timestamp FROM interactions
                    WHERE intent NOT IN ({placeholders})
                    ORDER BY id DESC LIMIT ?''',
                (*exclude_intents, limit)
            )
        else:
            c.execute(
                '''SELECT query, response, intent, timestamp FROM interactions
                   ORDER BY id DESC LIMIT ?''', (limit,)
            )
        rows = c.fetchall()
        result = [{"query": r[0], "response": r[1], "intent": r[2], "timestamp": r[3]} for r in rows]
        result.reverse()
        return result


    def get_preference(self, key: str, default=None) -> Optional[str]:
        """Retrieve a preference value by key."""
        c = self.conn.cursor()
        c.execute("SELECT value FROM preferences WHERE key=?", (key,))
        row = c.fetchone()
        return row[0] if row else default

    def set_preference(self, key: str, value: str) -> None:
        """Set or update a preference value."""
        ts = datetime.datetime.now().isoformat()
        with self._lock:
            c = self.conn.cursor()
            c.execute("INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
                      (key, value, ts))
            self.conn.commit()

    def get_all_preferences(self) -> dict[str, str]:
        """Return all preferences as a dictionary."""
        c = self.conn.cursor()
        c.execute("SELECT key, value FROM preferences")
        return {key: value for key, value in c.fetchall()}

    def search_interactions(self, keyword: str, limit: int = 5) -> list[dict]:
        """Search interactions by keyword in query or response, newest first."""
        like = f"%{keyword}%"
        c = self.conn.cursor()
        c.execute('''SELECT query, response, intent, timestamp FROM interactions
                     WHERE query LIKE ? OR response LIKE ?
                     ORDER BY id DESC LIMIT ?''', (like, like, limit))
        rows = c.fetchall()
        return [{"query": r[0], "response": r[1], "intent": r[2], "timestamp": r[3]} for r in rows]

    def get_by_intent(self, intent: str, limit: int = 10) -> list[dict]:
        """Fetch recent interactions filtered by intent, newest first."""
        c = self.conn.cursor()
        c.execute('''SELECT query, response, intent, timestamp FROM interactions
                     WHERE intent = ?
                     ORDER BY id DESC LIMIT ?''', (intent, limit))
        rows = c.fetchall()
        return [{"query": r[0], "response": r[1], "intent": r[2], "timestamp": r[3]} for r in rows]

    def clear_history(self) -> None:
        """Delete all interaction records (preferences remain)."""
        with self._lock:
            self.conn.execute("DELETE FROM interactions")
            self.conn.commit()


if __name__ == "__main__":
    # Smoke test
    mem = HestiaMemory(":memory:")
    mem.add_interaction("Hello", "Hi there!", "greeting")
    mem.add_interaction("What's the time?", "It's 10:00 AM", "get_time")
    print("Recent interactions:", mem.get_recent(5))
    mem.set_preference("theme", "dark")
    print("Preference theme:", mem.get_preference("theme"))
    print("Search 'time':", mem.search_interactions("time"))
    print("By intent 'greeting':", mem.get_by_intent("greeting"))
    mem.clear_history()
    print("After clear_history:", mem.get_recent(5))