"""

modules/pluto/db.py
Database connections with pooling and context managers.

"""


import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any

import psycopg2
from psycopg2 import pool
import redis
from qdrant_client import QdrantClient

from .config import PlutoConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connections with pooling and lazy initialization."""
    _instance: Optional["DatabaseManager"] = None
    _pg_pool: Optional[pool.SimpleConnectionPool] = None
    _redis_client: Optional[redis.Redis] = None
    _qdrant_client: Optional[QdrantClient] = None

    def __init__(self, config: PlutoConfig):
        self.config = config
        self._lock = threading.RLock()
        self._init_redis()
        self._init_qdrant()

    @classmethod
    def get_instance(cls, config: PlutoConfig) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def _init_redis(self) -> None:
        """Initialize Redis client with connection pooling."""
        if self._redis_client is None:
            self._redis_client = redis.Redis.from_url(
                str(self.config.redis_dsn),
                decode_responses=True,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            logger.info("Redis client initialized")

    def _init_qdrant(self) -> None:
        """Initialize Qdrant client."""
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                timeout=self.config.qdrant_timeout,
            )
            logger.info("Qdrant client initialized")

    def _init_pg_pool(self) -> None:
        """Initialize PostgreSQL connection pool."""
        with self._lock:
            if self._pg_pool is None:
                self._pg_pool = pool.SimpleConnectionPool(
                    minconn=self.config.pg_pool_min,
                    maxconn=self.config.pg_pool_max,
                    dsn=str(self.config.pg_dsn),
                )
                logger.info(
                    f"PostgreSQL pool initialized (min={self.config.pg_pool_min}, "
                    f"max={self.config.pg_pool_max})"
                )

    @contextmanager
    def get_pg_connection(self) -> Generator:
        """Context manager for PostgreSQL connections."""
        self._init_pg_pool()
        conn = self._pg_pool.getconn()
        try:
            yield conn
        finally:
            self._pg_pool.putconn(conn)

    def get_redis_client(self) -> redis.Redis:
        """Return the Redis client."""
        return self._redis_client

    def get_qdrant_client(self) -> QdrantClient:
        """Return the Qdrant client."""
        return self._qdrant_client

    def close_all(self) -> None:
        """Close all connections."""
        if self._pg_pool:
            self._pg_pool.closeall()
            logger.info("PostgreSQL pool closed")
        if self._redis_client:
            self._redis_client.close()
            logger.info("Redis client closed")
        # Qdrant client doesn't need explicit close


class PlutoDB:
    """
    Lightweight local SQLite database with improved thread safety.
    Stores expenses and investments.
    """

    def __init__(self, db_path: str):
        self._lock = threading.RLock()
        self._db_path = db_path
        # Use a single connection with WAL and auto-commit mode
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # auto-commit
        )
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS expenses (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount      REAL NOT NULL,
                    description TEXT NOT NULL,
                    category    TEXT NOT NULL,
                    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS investments (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    type        TEXT NOT NULL,
                    quantity    REAL,
                    buy_price   REAL,
                    logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_expenses_category
                    ON expenses(category);

                CREATE INDEX IF NOT EXISTS idx_expenses_logged_at
                    ON expenses(logged_at);

                CREATE INDEX IF NOT EXISTS idx_investments_name
                    ON investments(name);
                """
            )

    @contextmanager
    def transaction(self) -> Generator:
        """Context manager for transactions (explicit BEGIN/COMMIT)."""
        with self._lock:
            cursor = self._conn.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                yield cursor
                cursor.execute("COMMIT")
            except Exception:
                cursor.execute("ROLLBACK")
                raise
            finally:
                cursor.close()

    # ---- Expenses ----

    def log_expense(self, amount: float, description: str, category: str) -> int:
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO expenses (amount, description, category)
                VALUES (?, ?, ?)
                """,
                (amount, description, category),
            )
            return cur.lastrowid

    def get_expenses(self, limit: int = 100) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM expenses
            ORDER BY logged_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_totals_by_category(self) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT
                category,
                SUM(amount) AS total,
                COUNT(*) AS count
            FROM expenses
            GROUP BY category
            ORDER BY total DESC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def get_grand_total(self) -> float:
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM expenses"
        )
        return cur.fetchone()[0]

    # ---- Investments ----

    def log_investment(
        self,
        name: str,
        type_: str,
        quantity: float,
        buy_price: float,
    ) -> int:
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO investments (name, type, quantity, buy_price)
                VALUES (?, ?, ?, ?)
                """,
                (name, type_, quantity, buy_price),
            )
            return cur.lastrowid

    def get_investments(self) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT * FROM investments
            ORDER BY logged_at DESC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


# Backward compatibility functions (use DatabaseManager instead)
def get_pg_connection():
    """Deprecated: use DatabaseManager.get_pg_connection."""
    raise NotImplementedError("Use DatabaseManager.get_pg_connection context manager.")


def get_redis_client():
    """Deprecated: use DatabaseManager.get_redis_client."""
    raise NotImplementedError("Use DatabaseManager.get_redis_client.")


def get_vector_db():
    """Deprecated: use DatabaseManager.get_qdrant_client."""
    raise NotImplementedError("Use DatabaseManager.get_qdrant_client.")