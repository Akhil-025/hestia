"""
modules/pluto/logging_config.py
Structured JSON logging with correlation IDs.

"""

import logging
import json
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # logger.exception()/exc_info=True attach a traceback to the record;
        # the previous version silently dropped it since nothing here ever
        # called self.formatException(), making every logged exception
        # (e.g. PlutoEngine.handle()'s catch-all) impossible to debug from
        # the logs alone.
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        return json.dumps(log_data)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with structured JSON output."""
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    logger = logging.getLogger(name)
    # Ensure at least one handler (if configure_logging not called)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # main.py already calls logging.basicConfig() on the root logger for
        # the whole Hestia app. Without this, every Pluto log record would
        # print twice: once via this module's own JSON handler, and again
        # via the root logger's handler after propagating up.
        logger.propagate = False
    return logger