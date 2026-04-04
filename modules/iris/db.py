"""
modules/iris/db.py

Plain sqlite3 DB for Iris image/video/audio management.
"""
import sqlite3
import threading
from typing import Optional, List, Dict, Any

class IrisDB:
        def search_files_by_tags(self, query: str, limit: int = 10) -> list:
            cur = self._conn.execute(
                "SELECT * FROM files WHERE tags LIKE ? AND processed = 1 ORDER BY ingested_at DESC LIMIT ?",
                (f"%{query}%", limit)
            )
            return [dict(row) for row in cur.fetchall()]
        def __init__(self, db_path: str):
            self._lock = threading.Lock()
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            with self._conn:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA foreign_keys=ON;")
            self._init_schema()

        def _init_schema(self):
            with self._conn:
                self._conn.executescript('''
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE NOT NULL,
        file_hash TEXT NOT NULL,
        perceptual_hash TEXT,
        file_size INTEGER,
        file_type TEXT,
        mime_type TEXT,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        analyzed_at TIMESTAMP,
        processed BOOLEAN DEFAULT 0,
        caption TEXT,
        tags TEXT,
        objects TEXT,
        mood TEXT,
        is_sensitive BOOLEAN DEFAULT 0,
        blur_score REAL,
        error TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_files_file_hash ON files(file_hash);
    CREATE INDEX IF NOT EXISTS idx_files_processed ON files(processed);

    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        location_name TEXT,
        file_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS event_files (
        event_id INTEGER REFERENCES events(id),
        file_id INTEGER REFERENCES files(id),
        PRIMARY KEY (event_id, file_id)
    );

    CREATE TABLE IF NOT EXISTS processing_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER REFERENCES files(id),
        task_type TEXT DEFAULT 'analyze',
        status TEXT DEFAULT 'pending',
        priority INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        error TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_queue_status ON processing_queue(status);

    CREATE TABLE IF NOT EXISTS ingestion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        ingested INTEGER DEFAULT 0,
        duplicates INTEGER DEFAULT 0,
        errors INTEGER DEFAULT 0,
        total_size INTEGER DEFAULT 0
    );
    ''')

        # --- Files ---
        def file_exists(self, file_path: str) -> bool:
            cur = self._conn.execute("SELECT 1 FROM files WHERE file_path = ?", (file_path,))
            return cur.fetchone() is not None

        def file_exists_by_hash(self, file_hash: str) -> bool:
            cur = self._conn.execute("SELECT 1 FROM files WHERE file_hash = ?", (file_hash,))
            return cur.fetchone() is not None

        def insert_file(self, file_path, file_hash, perceptual_hash, file_size, file_type, mime_type) -> int:
            with self._lock, self._conn:
                cur = self._conn.execute(
                    """
                    INSERT INTO files (file_path, file_hash, perceptual_hash, file_size, file_type, mime_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (file_path, file_hash, perceptual_hash, file_size, file_type, mime_type)
                )
                return cur.lastrowid

        def update_file_analysis(self, file_id, caption, tags, objects, mood, is_sensitive, blur_score) -> None:
            with self._lock, self._conn:
                self._conn.execute(
                    """
                    UPDATE files SET caption=?, tags=?, objects=?, mood=?, is_sensitive=?, blur_score=?, analyzed_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (caption, tags, objects, mood, is_sensitive, blur_score, file_id)
                )

        def mark_file_processed(self, file_id: int) -> None:
            with self._lock, self._conn:
                self._conn.execute("UPDATE files SET processed=1 WHERE id=?", (file_id,))

        def get_file(self, file_id: int) -> Optional[dict]:
            cur = self._conn.execute("SELECT * FROM files WHERE id=?", (file_id,))
            row = cur.fetchone()
            return dict(row) if row else None

        def get_all_files(self, limit: int = 100) -> List[dict]:
            cur = self._conn.execute("SELECT * FROM files ORDER BY ingested_at DESC LIMIT ?", (limit,))
            return [dict(row) for row in cur.fetchall()]

        def search_files_by_caption(self, query: str, limit: int = 10) -> List[dict]:
            cur = self._conn.execute("SELECT * FROM files WHERE caption LIKE ? ORDER BY ingested_at DESC LIMIT ?", (f"%{query}%", limit))
            return [dict(row) for row in cur.fetchall()]

        # --- Queue ---
        def enqueue(self, file_id: int, task_type: str = 'analyze', priority: int = 0) -> None:
            with self._lock, self._conn:
                self._conn.execute(
                    """
                    INSERT INTO processing_queue (file_id, task_type, priority)
                    VALUES (?, ?, ?)
                    """,
                    (file_id, task_type, priority)
                )

        def get_next_queued(self) -> Optional[dict]:
            cur = self._conn.execute(
                "SELECT * FROM processing_queue WHERE status='pending' ORDER BY priority DESC, created_at ASC LIMIT 1"
            )
            row = cur.fetchone()
            return dict(row) if row else None

        def mark_queue_processing(self, queue_id: int) -> None:
            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE processing_queue SET status='processing', started_at=CURRENT_TIMESTAMP WHERE id=?",
                    (queue_id,)
                )

        def mark_queue_done(self, queue_id: int) -> None:
            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE processing_queue SET status='done', completed_at=CURRENT_TIMESTAMP WHERE id=?",
                    (queue_id,)
                )

        def mark_queue_failed(self, queue_id: int, error: str) -> None:
            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE processing_queue SET status='failed', error=? WHERE id=?",
                    (error, queue_id)
                )

        def pending_count(self) -> int:
            cur = self._conn.execute("SELECT COUNT(*) FROM processing_queue WHERE status='pending'")
            return cur.fetchone()[0]

        # --- Stats ---
        def file_count(self) -> int:
            cur = self._conn.execute("SELECT COUNT(*) FROM files")
            return cur.fetchone()[0]

        def processed_count(self) -> int:
            cur = self._conn.execute("SELECT COUNT(*) FROM files WHERE processed=1")
            return cur.fetchone()[0]
