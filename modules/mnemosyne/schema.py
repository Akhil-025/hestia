"""
modules/mnemosyne/schema.py

Defines the SQLite schema for Mnemosyne using only the sqlite3 standard library.
"""
import sqlite3

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE,
    value TEXT,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY,
    period_start TIMESTAMP,
    period_end TIMESTAMP,
    content TEXT,
    topic TEXT,
    interaction_count INTEGER,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY,
    text TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('active','completed','cancelled')),
    due_date TIMESTAMP,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS semantic_refs (
    id INTEGER PRIMARY KEY,
    table_name TEXT,
    row_id INTEGER,
    chroma_id TEXT,
    embedding_type TEXT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interaction_log (
    id INTEGER PRIMARY KEY,
    user_text TEXT,
    hestia_response TEXT,
    intent TEXT,
    source_device TEXT DEFAULT 'hestia',
    pushed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summarised BOOLEAN DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_interaction_log_summarised ON interaction_log(summarised);
CREATE INDEX IF NOT EXISTS idx_summaries_period_start ON summaries(period_start);
"""

def init_db(db_path: str):
    """Initializes the database at db_path with the Mnemosyne schema."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
