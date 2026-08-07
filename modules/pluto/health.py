"""

modules/pluto/health.py
Health checks for all dependencies.

"""


import time
from enum import Enum
from typing import Dict, Any, Optional

from .config import PlutoConfig
from .db import DatabaseManager
from .logging_config import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthChecker:
    def __init__(self, config: PlutoConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.start_time = time.time()

    def get_uptime(self) -> float:
        return time.time() - self.start_time

    def check_all(self) -> Dict[str, Any]:
        """Check all dependencies and return overall health."""
        checks = {
            "postgres": self._check_postgres(),
            "redis": self._check_redis(),
            "qdrant": self._check_qdrant(),
            "ollama": self._check_ollama(),
            "kimi": self._check_kimi() if self.config.kimi_api_key else None,
        }
        # Filter out None checks
        checks = {k: v for k, v in checks.items() if v is not None}

        overall = HealthStatus.HEALTHY
        for c in checks.values():
            if c.get("status") == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
                break
            if c.get("status") == HealthStatus.DEGRADED and overall == HealthStatus.HEALTHY:
                overall = HealthStatus.DEGRADED

        return {
            "status": overall,
            "timestamp": time.time(),
            "uptime_seconds": self.get_uptime(),
            "checks": checks,
        }

    def _check_postgres(self) -> Dict[str, Any]:
        try:
            with self.db_manager.get_pg_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return {"status": HealthStatus.HEALTHY}
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return {"status": HealthStatus.UNHEALTHY, "error": str(e)}

    def _check_redis(self) -> Dict[str, Any]:
        try:
            client = self.db_manager.get_redis_client()
            client.ping()
            return {"status": HealthStatus.HEALTHY}
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": HealthStatus.UNHEALTHY, "error": str(e)}

    def _check_qdrant(self) -> Dict[str, Any]:
        try:
            client = self.db_manager.get_qdrant_client()
            # Simple status call
            client.get_collections()
            return {"status": HealthStatus.HEALTHY}
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return {"status": HealthStatus.UNHEALTHY, "error": str(e)}

    def _check_ollama(self) -> Dict[str, Any]:
        import requests
        try:
            url = f"http://{self.config.ollama_host}:{self.config.ollama_port}/api/tags"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return {"status": HealthStatus.HEALTHY}
            else:
                return {"status": HealthStatus.DEGRADED, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": HealthStatus.UNHEALTHY, "error": str(e)}

    def _check_kimi(self) -> Dict[str, Any]:
        # Kimi check could be a simple ping if API supports it
        # For now, assume if API key exists, it's healthy
        return {"status": HealthStatus.HEALTHY}