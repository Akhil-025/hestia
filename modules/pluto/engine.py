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
from .portfolio import PortfolioOptimizer
from .backtest import StrategyBacktester
from .forecasting import SpendForecaster
from .advisor_agent import FinancialAdvisorAgent
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
        "convert_currency",
        "company_lookup",
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
        "convert": "convert_currency",
        "currency_convert": "convert_currency",
        "fx": "convert_currency",
        "company_info": "company_lookup",
        "lookup_company": "company_lookup",
        "sec_lookup": "company_lookup",
    }

    _MI_INTENTS: frozenset[str] = frozenset({
        "analyze_asset"
    })

    # Quant/ML features (portfolio.py, backtest.py, forecasting.py,
    # advisor_agent.py) — separate from _PF_INTENTS/_MI_INTENTS because
    # they operate on different backing managers (self.portfolio_optimizer,
    # self.backtester, self.forecaster, self.advisor) rather than
    # pf_manager/mi_manager.
    _QUANT_INTENTS: frozenset[str] = frozenset({
        "optimize_portfolio",
        "backtest_strategy",
        "forecast_spending",
        "financial_advisor_chat",
    })

    _QUANT_INTENT_ALIASES: dict[str, str] = {
        "optimise_portfolio": "optimize_portfolio",
        "portfolio_optimization": "optimize_portfolio",
        "rebalance_portfolio": "optimize_portfolio",
        "backtest": "backtest_strategy",
        "run_backtest": "backtest_strategy",
        "forecast_expenses": "forecast_spending",
        "predict_spending": "forecast_spending",
        "spending_forecast": "forecast_spending",
        "ask_pluto": "financial_advisor_chat",
        "finance_advisor": "financial_advisor_chat",
        "ask_financial_advisor": "financial_advisor_chat",
    }

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
        portfolio_optimizer: Optional[Any] = None,
        backtester: Optional[Any] = None,
        forecaster: Optional[Any] = None,
        advisor: Optional[Any] = None,
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

        # Quant/ML features. These read from pf_manager.db (the local
        # SQLite investments/expenses tables) rather than db_manager's
        # Postgres/Redis/Qdrant stack, matching the local-first scope
        # described in each module's docstring.
        self.portfolio_optimizer = portfolio_optimizer or PortfolioOptimizer(
            db=self.pf_manager.db, currency=self.config.currency
        )
        self.backtester = backtester or StrategyBacktester()
        self.forecaster = forecaster or SpendForecaster(
            db=self.pf_manager.db, currency=self.config.currency
        )
        self.advisor = advisor or FinancialAdvisorAgent(
            pf_manager=self.pf_manager, llm_client=self.llm_client
        )

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
            intent in self._MI_INTENTS or
            intent in self._QUANT_INTENTS or
            intent in self._QUANT_INTENT_ALIASES
        )

    @track_metrics("handle")
    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Dispatch an intent to the appropriate sub-manager handler.
        Never raises; all errors produce a graceful response dict.
        """
        try:
            resolved_intent = self._PF_INTENT_ALIASES.get(intent, intent)
            resolved_intent = self._QUANT_INTENT_ALIASES.get(resolved_intent, resolved_intent)

            # Quant/ML Routes
            if resolved_intent == "optimize_portfolio":
                return self.portfolio_optimizer.optimize()

            if resolved_intent == "backtest_strategy":
                ticker = (entities.get("ticker") or entities.get("name") or "").strip()
                if not ticker:
                    return _err("Which ticker should I backtest? e.g. 'backtest RELIANCE'.")
                fast = int(entities.get("fast_window", 10) or 10)
                slow = int(entities.get("slow_window", 50) or 50)
                return self.backtester.backtest_sma_crossover(
                    ticker, fast_window=fast, slow_window=slow
                )

            if resolved_intent == "forecast_spending":
                horizon = int(entities.get("horizon_days", 7) or 7)
                return self.forecaster.forecast(horizon_days=horizon)

            if resolved_intent == "financial_advisor_chat":
                question = (
                    entities.get("question")
                    or entities.get("raw_query")
                    or entities.get("query")
                    or ""
                ).strip()
                return self.advisor.ask(question)

            # Personal Finance Routes
            if resolved_intent == "log_expense":
                return self.pf_manager.log_expense(entities)
            if resolved_intent == "get_budget_summary":
                return self.pf_manager.budget_summary()
            if resolved_intent == "track_investment":
                return self.pf_manager.track_investment(entities)
            if resolved_intent == "spending_report":
                return self.pf_manager.spending_report()
            if resolved_intent == "convert_currency":
                return self.pf_manager.convert_currency(entities)
            if resolved_intent == "company_lookup":
                return self.pf_manager.company_lookup(entities)

            # Market Intelligence Routes
            if resolved_intent == "analyze_asset":
                ticker = entities.get("ticker", "").strip()
                if not ticker:
                    return _err("No ticker provided for asset analysis.")

                result = self.mi_manager.analyze_asset(ticker)
                # Confidence reflects how much clean data the score is
                # standing on (result["reliability"]), not the score's
                # own bullish/bearish reading — a bearish-but-well-
                # supported call shouldn't report low confidence just
                # because quant_score itself is low.
                return {
                    "response": result["explanation"],
                    "data": result,
                    "confidence": min(result.get("reliability", 0.5), 0.95),
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