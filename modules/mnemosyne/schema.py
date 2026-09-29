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
    updated_at TIMESTAMP,
    -- Added for backlog #36 (expiry/decay) and #45 (importance scoring).
    -- last_accessed starts equal to created_at (set at INSERT time by
    -- MnemosyneDB.set_fact) and advances whenever the fact is recalled —
    -- see MnemosyneDB.touch_fact. access_count is the raw reference
    -- count; importance is an explicit 0..1 user-settable multiplier,
    -- separate from confidence (confidence is "how sure am I this is
    -- true"; importance is "how much should this matter for ranking even
    -- if true"). stale is set by the decay job (#36) to flag — never
    -- silently delete — a fact that hasn't been touched in a long time.
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    importance REAL DEFAULT 0.5,
    stale INTEGER DEFAULT 0
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

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY,
    text TEXT,
    due_time TIMESTAMP,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reminders_due_time ON reminders(due_time);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_interaction_log_summarised ON interaction_log(summarised);
CREATE INDEX IF NOT EXISTS idx_summaries_period_start ON summaries(period_start);
"""

def init_db(db_path: str):
    """Initializes the database at db_path with the Mnemosyne schema."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate_facts_columns(conn)
        conn.commit()


# Columns added to `facts` after the table already existed in the wild
# (backlog #36, #45). `CREATE TABLE IF NOT EXISTS` above is a no-op on an
# existing table, so an existing database needs these added explicitly.
# ALTER TABLE ... ADD COLUMN has no "IF NOT EXISTS" guard in SQLite, so
# idempotency comes from catching "duplicate column" instead — this makes
# init_db safe to call every startup, on both a fresh database (where the
# CREATE TABLE above already includes these columns, so every ALTER here
# is a no-op duplicate-column error) and an old one (where they're
# genuinely new).
_FACTS_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("last_accessed", "TIMESTAMP"),
    ("access_count", "INTEGER DEFAULT 0"),
    ("importance", "REAL DEFAULT 0.5"),
    ("stale", "INTEGER DEFAULT 0"),
)


def _migrate_facts_columns(conn: sqlite3.Connection) -> None:
    for column, coltype in _FACTS_MIGRATION_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE facts ADD COLUMN {column} {coltype}")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    # Backfill last_accessed for any pre-existing row that predates the
    # column (it's NULL by default from ADD COLUMN, not created_at) —
    # without this, every fact that existed before this migration would
    # look infinitely stale to the #36 decay job on its very first run.
    conn.execute(
        "UPDATE facts SET last_accessed = created_at WHERE last_accessed IS NULL"
    )
