"""
modules/iris/db.py

Plain sqlite3 DB for Iris image/video/audio management.
"""
import sqlite3
import threading
from typing import Optional, List, Dict, Any, Tuple

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
        analysis_status TEXT DEFAULT 'pending',
        caption TEXT,
        tags TEXT,
        objects TEXT,
        mood TEXT,
        is_sensitive BOOLEAN DEFAULT 0,
        blur_score REAL,
        error TEXT,
        date_taken TIMESTAMP,
        gps_lat REAL,
        gps_lon REAL,
        camera_make TEXT,
        camera_model TEXT,
        caption_source TEXT DEFAULT 'ai'
    );
    CREATE INDEX IF NOT EXISTS idx_files_file_hash ON files(file_hash);
    CREATE INDEX IF NOT EXISTS idx_files_processed ON files(processed);
    CREATE INDEX IF NOT EXISTS idx_files_perceptual_hash ON files(perceptual_hash)
        WHERE perceptual_hash IS NOT NULL;

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

    -- Face grouping (#72). Embeddings are biometric data: they live only in
    -- this file, and delete_all_face_data() removes them. Opt-in feature; the
    -- tables stay empty unless iris.faces.enabled is set.
    CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL REFERENCES files(id),
        person_id INTEGER REFERENCES people(id),
        box TEXT,
        score REAL,
        embedding TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
    CREATE INDEX IF NOT EXISTS idx_faces_file ON faces(file_id);
    CREATE TABLE IF NOT EXISTS face_scans (
        file_id INTEGER PRIMARY KEY REFERENCES files(id),
        scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        face_count INTEGER DEFAULT 0,
        error TEXT
    );
    ''')
                # Migration: add analysis_status to pre-existing DBs (CREATE TABLE
                # IF NOT EXISTS above only applies to brand-new databases).
                try:
                    self._conn.execute(
                        "ALTER TABLE files ADD COLUMN analysis_status TEXT DEFAULT 'pending'"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists

                # Migration: EXIF columns (backlog #74) and a caption_source
                # marker (backlog #78) — same idempotent pattern, needed for
                # any database that predates these columns being added to
                # the CREATE TABLE above.
                for column, coltype in (
                    ("date_taken", "TIMESTAMP"),
                    ("gps_lat", "REAL"),
                    ("gps_lon", "REAL"),
                    ("camera_make", "TEXT"),
                    ("camera_model", "TEXT"),
                    # 'ai' (default, set by IrisAnalyser) or 'user' (set by
                    # correct_caption) — lets a future re-analysis pass
                    # skip files a person has already manually corrected,
                    # instead of silently overwriting their correction.
                    ("caption_source", "TEXT DEFAULT 'ai'"),
                    # backlog #75: length of a video, in seconds (NULL for photos)
                    ("duration_seconds", "REAL"),
                ):
                    try:
                        self._conn.execute(f"ALTER TABLE files ADD COLUMN {column} {coltype}")
                    except sqlite3.OperationalError:
                        pass  # column already exists

                # Only safe to create now — on a pre-existing database, the
                # migration loop just above is what actually added
                # date_taken; creating this index inside the earlier
                # CREATE TABLE IF NOT EXISTS executescript (a no-op against
                # an existing table) would fail with "no such column" on
                # any database that predates this column.
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_files_date_taken "
                    "ON files(date_taken) WHERE date_taken IS NOT NULL"
                )

        # --- Files ---
        def file_exists(self, file_path: str) -> bool:
            cur = self._conn.execute("SELECT 1 FROM files WHERE file_path = ?", (file_path,))
            return cur.fetchone() is not None

        def file_exists_by_hash(self, file_hash: str) -> bool:
            cur = self._conn.execute("SELECT 1 FROM files WHERE file_hash = ?", (file_hash,))
            return cur.fetchone() is not None

        def get_perceptual_hashes(self) -> List[Tuple[int, str, str]]:
            """Return (id, file_path, perceptual_hash) for every ingested file
            that has a stored perceptual hash. Used for near-duplicate (not
            just exact-hash) detection — sqlite has no native Hamming-distance
            operator, so the comparison itself happens in Python (see
            DuplicateDetector.find_duplicates in ingestion.py); this just
            gives it the candidate set to compare against."""
            cur = self._conn.execute(
                "SELECT id, file_path, perceptual_hash FROM files WHERE perceptual_hash IS NOT NULL"
            )
            return [(row["id"], row["file_path"], row["perceptual_hash"]) for row in cur.fetchall()]

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
                self._conn.execute(
                    "UPDATE files SET processed=1, analysis_status='processed' WHERE id=?",
                    (file_id,)
                )

        def mark_file_error(self, file_id: int, error_msg: str) -> None:
            """Mark a file as errored (not processed) so it can be retried later."""
            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE files SET analysis_status='error', error=? WHERE id=?",
                    (error_msg, file_id)
                )

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

        def update_file_exif(
            self, file_id: int, date_taken=None, gps_lat=None, gps_lon=None,
            camera_make=None, camera_model=None,
        ) -> None:
            """backlog #74 — called once per file during ingestion/analysis."""
            with self._lock, self._conn:
                self._conn.execute(
                    """
                    UPDATE files SET date_taken=?, gps_lat=?, gps_lon=?,
                                      camera_make=?, camera_model=?
                    WHERE id=?
                    """,
                    (date_taken, gps_lat, gps_lon, camera_make, camera_model, file_id),
                )

        def search_files_by_exif(
            self, date_from=None, date_to=None, camera=None, has_location=None,
            limit: int = 50,
        ) -> List[dict]:
            """
            backlog #74. Any combination of filters may be given; all
            supplied filters are ANDed together. `camera` matches against
            either camera_make or camera_model (substring, case-insensitive
            via LIKE's default collation).
            """
            clauses: List[str] = []
            params: List[Any] = []
            if date_from:
                clauses.append("date_taken >= ?")
                params.append(date_from)
            if date_to:
                clauses.append("date_taken <= ?")
                params.append(date_to)
            if camera:
                clauses.append("(camera_make LIKE ? OR camera_model LIKE ?)")
                params.extend([f"%{camera}%", f"%{camera}%"])
            if has_location is True:
                clauses.append("gps_lat IS NOT NULL AND gps_lon IS NOT NULL")
            elif has_location is False:
                clauses.append("(gps_lat IS NULL OR gps_lon IS NULL)")

            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur = self._conn.execute(
                f"SELECT * FROM files {where} ORDER BY date_taken DESC LIMIT ?",
                (*params, limit),
            )
            return [dict(row) for row in cur.fetchall()]

        # --- Manual corrections (#78) ---
        def correct_caption(
            self, file_id: int, caption: Optional[str] = None, tags: Optional[str] = None,
        ) -> bool:
            """
            Overwrite an AI-generated caption/tags with a user correction,
            marking caption_source='user' so a future re-analysis pass
            (should one ever be added) knows not to silently clobber it.
            At least one of caption/tags must be given. Returns False if
            the file doesn't exist.
            """
            if caption is None and tags is None:
                return False
            if not self.get_file(file_id):
                return False
            sets = ["caption_source = 'user'"]
            params: List[Any] = []
            if caption is not None:
                sets.append("caption = ?")
                params.append(caption)
            if tags is not None:
                sets.append("tags = ?")
                params.append(tags)
            params.append(file_id)
            with self._lock, self._conn:
                self._conn.execute(
                    f"UPDATE files SET {', '.join(sets)} WHERE id = ?", params
                )
            return True

        # --- Whole-library duplicate scan (#73) ---
        def get_all_file_hashes(self) -> List[Tuple[int, str, str]]:
            """(id, file_path, file_hash) for every ingested file — the
            candidate set for an EXACT-hash whole-library duplicate scan,
            as distinct from get_perceptual_hashes' near-duplicate one."""
            cur = self._conn.execute("SELECT id, file_path, file_hash FROM files")
            return [(row["id"], row["file_path"], row["file_hash"]) for row in cur.fetchall()]

        # --- Storage budget (#80) ---
        def get_total_ingested_bytes(self) -> int:
            cur = self._conn.execute("SELECT COALESCE(SUM(file_size), 0) FROM files")
            return cur.fetchone()[0]

        # --- Albums / events (#79) ---
        def create_event(self, name: str) -> int:
            with self._lock, self._conn:
                cur = self._conn.execute(
                    "INSERT INTO events (name, file_count) VALUES (?, 0)", (name,)
                )
                return cur.lastrowid

        def add_files_to_event(self, event_id: int, file_ids: List[int]) -> None:
            if not file_ids:
                return
            with self._lock, self._conn:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO event_files (event_id, file_id) VALUES (?, ?)",
                    [(event_id, fid) for fid in file_ids],
                )
                self._conn.execute(
                    "UPDATE events SET file_count = "
                    "(SELECT COUNT(*) FROM event_files WHERE event_id = ?) WHERE id = ?",
                    (event_id, event_id),
                )

        def get_events(self, limit: int = 50) -> List[dict]:
            cur = self._conn.execute(
                "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cur.fetchall()]

        def get_files_in_event(self, event_id: int) -> List[dict]:
            cur = self._conn.execute(
                """
                SELECT f.* FROM files f
                JOIN event_files ef ON ef.file_id = f.id
                WHERE ef.event_id = ?
                """,
                (event_id,),
            )
            return [dict(row) for row in cur.fetchall()]

        def clear_all_events(self) -> None:
            """Wipe existing album groupings before a fresh re-cluster (organize_into_albums re-runs from scratch each time, not incrementally)."""
            with self._lock, self._conn:
                self._conn.execute("DELETE FROM event_files")
                self._conn.execute("DELETE FROM events")


        # --- Video and object metadata (#75, #77, #71 re-index) ---
        def update_video_info(self, file_id: int, duration_seconds: Optional[float]) -> None:
            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE files SET duration_seconds=? WHERE id=?", (duration_seconds, file_id)
                )

        def get_files_by_type(self, file_types, limit: int = 1000) -> List[dict]:
            """Files of the given type(s), oldest first (stable order for re-indexing)."""
            types = list(file_types)
            if not types:
                return []
            marks = ",".join("?" for _ in types)
            cur = self._conn.execute(
                f"SELECT * FROM files WHERE file_type IN ({marks}) ORDER BY id LIMIT ?",
                (*types, limit),
            )
            return [dict(row) for row in cur.fetchall()]

        def update_file_objects(self, file_id: int, objects_json: str) -> None:
            with self._lock, self._conn:
                self._conn.execute("UPDATE files SET objects=? WHERE id=?", (objects_json, file_id))

        def search_files_by_objects(self, query: str, limit: int = 10) -> List[dict]:
            """Files whose detected-objects field mentions *query* (a quoted JSON key,
            so 'cup' doesn't match 'cupboard')."""
            q = (query or "").strip().lower()
            if not q:
                return []
            cur = self._conn.execute(
                "SELECT * FROM files WHERE objects LIKE ? ORDER BY ingested_at DESC LIMIT ?",
                (f'%"{q}"%', limit),
            )
            return [dict(row) for row in cur.fetchall()]

        # --- Faces and people (#72) ---
        def get_unscanned_images(self, limit: int = 100) -> List[dict]:
            cur = self._conn.execute(
                "SELECT id, file_path FROM files WHERE file_type='image' "
                "AND id NOT IN (SELECT file_id FROM face_scans) ORDER BY id LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

        def count_unscanned_images(self) -> int:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM files WHERE file_type='image' "
                "AND id NOT IN (SELECT file_id FROM face_scans)"
            )
            return cur.fetchone()[0]

        def record_face_scan(self, file_id: int, face_count: int, error: Optional[str] = None) -> None:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO face_scans (file_id, face_count, error) VALUES (?, ?, ?)",
                    (file_id, face_count, error),
                )

        def add_face(self, file_id: int, box_json: str, score: float, embedding_json: str) -> int:
            with self._lock, self._conn:
                cur = self._conn.execute(
                    "INSERT INTO faces (file_id, box, score, embedding) VALUES (?, ?, ?, ?)",
                    (file_id, box_json, score, embedding_json),
                )
                return cur.lastrowid

        def get_faces(self, unassigned_only: bool = False) -> List[dict]:
            where = "WHERE person_id IS NULL" if unassigned_only else ""
            cur = self._conn.execute(
                f"SELECT id, file_id, person_id, box, score, embedding FROM faces {where} ORDER BY id"
            )
            return [dict(row) for row in cur.fetchall()]

        def set_face_person(self, face_ids: List[int], person_id: Optional[int]) -> None:
            if not face_ids:
                return
            with self._lock, self._conn:
                self._conn.executemany(
                    "UPDATE faces SET person_id=? WHERE id=?", [(person_id, i) for i in face_ids]
                )

        def create_person(self, name: Optional[str] = None) -> int:
            with self._lock, self._conn:
                cur = self._conn.execute("INSERT INTO people (name) VALUES (?)", (name,))
                return cur.lastrowid

        def rename_person(self, person_id: int, name: Optional[str]) -> None:
            with self._lock, self._conn:
                self._conn.execute("UPDATE people SET name=? WHERE id=?", (name, person_id))

        def get_people(self) -> List[dict]:
            cur = self._conn.execute(
                """
                SELECT p.id AS id, p.name AS name,
                       COUNT(f.id) AS face_count,
                       COUNT(DISTINCT f.file_id) AS photo_count
                FROM people p LEFT JOIN faces f ON f.person_id = p.id
                GROUP BY p.id ORDER BY photo_count DESC, p.id
                """
            )
            return [dict(row) for row in cur.fetchall()]

        def merge_people(self, from_id: int, into_id: int) -> None:
            if from_id == into_id:
                return
            with self._lock, self._conn:
                self._conn.execute("UPDATE faces SET person_id=? WHERE person_id=?", (into_id, from_id))
                self._conn.execute("DELETE FROM people WHERE id=?", (from_id,))

        def get_files_for_person(self, person_id: int) -> List[dict]:
            cur = self._conn.execute(
                """
                SELECT DISTINCT fl.* FROM files fl
                JOIN faces fa ON fa.file_id = fl.id
                WHERE fa.person_id = ?
                ORDER BY COALESCE(fl.date_taken, fl.ingested_at) DESC
                """,
                (person_id,),
            )
            return [dict(row) for row in cur.fetchall()]

        def delete_all_face_data(self) -> dict:
            """Remove every face, person and scan record. Returns what was removed."""
            with self._lock, self._conn:
                counts = {
                    "faces": self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
                    "people": self._conn.execute("SELECT COUNT(*) FROM people").fetchone()[0],
                    "scans": self._conn.execute("SELECT COUNT(*) FROM face_scans").fetchone()[0],
                }
                self._conn.execute("DELETE FROM faces")
                self._conn.execute("DELETE FROM people")
                self._conn.execute("DELETE FROM face_scans")
            return counts

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