"""

modules/pluto/engine.py
PlutoEngine coordinator with dependency injection and health checks.

"""


from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from modules.base import BaseModule

from .config import PlutoConfig
from .db import DatabaseManager
from .personal_finance import PersonalFinanceManager, _err
from .market_intelligence import MarketIntelligenceManager
from .llm_client import LLMClient
from .health import HealthChecker
from .metrics import MetricsCollector, track_metrics
from .logging_config import get_logger, correlation_id_var

logger = get_logger(__name__)


class PlutoEngine(BaseModule):
    """
    Coordinator module routing to personal finance and market intelligence subsystems.
    """

    name = "pluto"

    _PF_INTENTS: frozenset[str] = frozenset({
        "log_expense",
        "get_budget_summary",
        "track_investment",
        "spending_report",
    })

    _PF_INTENT_ALIASES: dict[str, str] = {
        "spend": "log_expense",
        "spent": "log_expense",
        "add_expense": "log_expense",
        "log_spend": "log_expense",
        "log_spending": "log_expense",
        "track_expense": "log_expense",
        "record_expense": "log_expense",
        "budget": "get_budget_summary",
        "get_budget": "get_budget_summary",
        "budget_check": "get_budget_summary",
        "check_budget": "get_budget_summary",
        "add_investment": "track_investment",
        "log_investment": "track_investment",
        "investment": "track_investment",
        "report": "spending_report",
        "get_report": "spending_report",
        "expense_report": "spending_report",
    }

    _MI_INTENTS: frozenset[str] = frozenset({
        "analyze_asset"
    })

    def __init__(
        self,
        config: Optional[PlutoConfig] = None,
        ollama_cfg: Optional[dict] = None,
        pf_manager: Optional[PersonalFinanceManager] = None,
        mi_manager: Optional[MarketIntelligenceManager] = None,
        llm_client: Optional[LLMClient] = None,
        kimi_client: Optional[Any] = None,
        db_manager: Optional[DatabaseManager] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        """
        `ollama_cfg` mirrors the `{host, port, model}` dict every other
        Hestia engine (Apollo, Ares, Orpheus, Dionysus, ...) accepts from
        main.py's shared `laptop_config.yaml` -> `ollama:` section. It's
        merged into `PlutoConfig` here so Pluto stays wired to the same
        Ollama instance as the rest of the app instead of silently falling
        back to its own hardcoded defaults.
        """
        self.config = config or self._build_config(ollama_cfg)
        self.db_manager = db_manager or DatabaseManager(self.config)
        self.metrics = metrics or MetricsCollector()
        self.llm_client = llm_client or LLMClient(self.config, fallback_llm=kimi_client)

        self.pf_manager = pf_manager or PersonalFinanceManager(
            config=self.config,
            db_manager=self.db_manager,
            llm_client=self.llm_client,
            metrics=self.metrics,
        )
        self.mi_manager = mi_manager or MarketIntelligenceManager(
            config=self.config,
            db_manager=self.db_manager,
            kimi_client=kimi_client,
            llm_client=self.llm_client,
            metrics=self.metrics,
        )

        self.health_checker = HealthChecker(self.config, self.db_manager)

        logger.info("PlutoEngine coordinator initialized.")

    @staticmethod
    def _build_config(ollama_cfg: Optional[dict]) -> PlutoConfig:
        """Build a PlutoConfig, layering shared ollama_cfg over .env/defaults."""
        if not ollama_cfg:
            return PlutoConfig()
        overrides: dict[str, Any] = {}
        if ollama_cfg.get("host") is not None:
            overrides["ollama_host"] = ollama_cfg["host"]
        if ollama_cfg.get("port") is not None:
            overrides["ollama_port"] = ollama_cfg["port"]
        if ollama_cfg.get("model") is not None:
            overrides["ollama_model"] = ollama_cfg["model"]
        return PlutoConfig(**overrides)

    def can_handle(self, intent: str) -> bool:
        return (
            intent in self._PF_INTENTS or
            intent in self._PF_INTENT_ALIASES or
            intent in self._MI_INTENTS
        )

    @track_metrics("handle")
    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Dispatch an intent to the appropriate sub-manager handler.
        Never raises; all errors produce a graceful response dict.
        """
        try:
            resolved_intent = self._PF_INTENT_ALIASES.get(intent, intent)

            # Personal Finance Routes
            if resolved_intent == "log_expense":
                return self.pf_manager.log_expense(entities)
            if resolved_intent == "get_budget_summary":
                return self.pf_manager.budget_summary()
            if resolved_intent == "track_investment":
                return self.pf_manager.track_investment(entities)
            if resolved_intent == "spending_report":
                return self.pf_manager.spending_report()

            # Market Intelligence Routes
            if resolved_intent == "analyze_asset":
                ticker = entities.get("ticker", "").strip()
                if not ticker:
                    return _err("No ticker provided for asset analysis.")

                result = self.mi_manager.analyze_asset(ticker)
                return {
                    "response": result["explanation"],
                    "data": result,
                    "confidence": min(result["quant_score"], 0.95),
                }

            return _err(f"Unknown intent: {intent!r}")

        except Exception as e:
            logger.exception("PlutoEngine.handle() raised an exception for intent=%s.", intent)
            return _err("Something went wrong in the Pluto module.")

    def get_context(self) -> dict:
        """Return a lightweight context snapshot for NLU enrichment."""
        return self.pf_manager.get_context()

    def health_check(self) -> dict:
        """Return health status of all dependencies."""
        return self.health_checker.check_all()