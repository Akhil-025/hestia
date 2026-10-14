# modules/pluto/__init__.py

from .engine import PlutoEngine
from .config import PlutoConfig
from .db import DatabaseManager, PlutoDB
from .personal_finance import PersonalFinanceManager
from .market_intelligence import MarketIntelligenceManager
from .llm_client import LLMClient
from .health import HealthChecker
from .metrics import MetricsCollector
from .logging_config import configure_logging


def __getattr__(name):
    # LangGraphOrchestrator pulls in langgraph; load it only when asked for.
    if name == "LangGraphOrchestrator":
        from .agents import LangGraphOrchestrator
        return LangGraphOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PlutoEngine",
    "PlutoConfig",
    "DatabaseManager",
    "PlutoDB",
    "PersonalFinanceManager",
    "MarketIntelligenceManager",
    "LangGraphOrchestrator",
    "LLMClient",
    "HealthChecker",
    "MetricsCollector",
    "configure_logging",
]