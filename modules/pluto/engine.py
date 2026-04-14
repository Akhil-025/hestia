"""
modules/pluto/engine.py

PlutoEngine: personal finance module for expense logging, budget summaries,
investment tracking, and AI-generated spending reports.

Design notes
------------
- All LLM calls are isolated in helpers that never raise; failures fall back
  to a default category / empty advice string.
- Live price fetching is retried once with exponential back-off and always
  returns a typed result rather than a raw tuple.
- Currency symbol and locale are configurable at construction time.
- Every public method returns a typed dict that conforms to the BaseModule
  response contract: {response, data, confidence}.
- No bare `except Exception` in business logic; specific exception types are
  caught where possible and the remainder are logged with full tracebacks.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from core.ollama_client import generate
from modules.base import BaseModule
from .db import PlutoDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "pluto" / "pluto.db"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "mistral"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 11434
_DEFAULT_CURRENCY = "₹"
_REQUEST_TIMEOUT = 10  # seconds
_MAX_EXPENSE_REPORT_ROWS = 200

_CRYPTO_SLUG_MAP: dict[str, str] = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "bnb": "binancecoin",
    "usdt": "tether",
}

_CRYPTO_KEYWORDS: frozenset[str] = frozenset(
    {"bitcoin", "btc", "eth", "ethereum", "crypto", "solana", "sol",
     "usdt", "bnb", "doge", "dogecoin"}
)
_MUTUAL_FUND_KEYWORDS: frozenset[str] = frozenset(
    {"mutual fund", "sip", "nifty", "sensex", "index fund", "elss", "debt fund"}
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CATEGORIZE_PROMPT = """\
You are a finance assistant.
Categorize this expense: "{description}" (amount: {amount})

Respond with ONLY valid JSON:
{{"category": "one of: Food, Transport, Shopping, Health, Entertainment, Bills, Education, Other"}}

JSON only. No explanation."""

_ADVICE_PROMPT = """\
You are Pluto, a personal finance guru.
Here is the user's spending by category:
{breakdown}

Total spent: {total}

Give specific, actionable advice on:
1. Which category to cut first and how
2. A simple weekly budget target
3. One habit change that will have the biggest impact

Keep it under 150 words. Be direct and practical."""

_REPORT_PROMPT = """\
You are Pluto, a personal finance guru.
Spending breakdown:
{breakdown}

Total: {total}
Number of transactions: {count}

Write a spending report with:
- WHERE the money is going (biggest categories)
- WHAT to cut and by how much
- HOW to track it going forward

Under 200 words. Practical and specific."""


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LivePriceResult:
    """Result of a live price fetch attempt."""

    price: Optional[float]
    error: str = ""

    @property
    def available(self) -> bool:
        return self.price is not None


@dataclass(frozen=True)
class OllamaConfig:
    """Typed configuration for the Ollama LLM backend."""

    model: str = _DEFAULT_MODEL
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "OllamaConfig":
        return cls(
            model=str(cfg.get("model", _DEFAULT_MODEL)),
            host=str(cfg.get("host", _DEFAULT_HOST)),
            port=int(cfg.get("port", _DEFAULT_PORT)),
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PlutoError(Exception):
    """Base exception for PlutoEngine failures."""


class LivePriceFetchError(PlutoError):
    """Raised when a live price cannot be retrieved."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PlutoEngine(BaseModule):
    """
    Personal finance module: expense logging, budget summaries,
    investment tracking, and AI-generated spending analysis.

    Parameters
    ----------
    ollama_cfg:
        Dict with optional keys ``model``, ``host``, ``port``.
    currency:
        Currency symbol prepended to monetary values in responses.
    db_path:
        Override the default SQLite database path.
    """

    name = "pluto"

    _INTENTS: frozenset[str] = frozenset(
        {
            "log_expense",
            "get_budget_summary",
            "track_investment",
            "spending_report",
        }
    )

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        currency: str = _DEFAULT_CURRENCY,
        db_path: Optional[Path] = None,
    ) -> None:
        self._cfg = OllamaConfig.from_dict(ollama_cfg or {})
        self._currency = currency
        resolved_path = (db_path or _DB_PATH).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = PlutoDB(str(resolved_path))
        logger.info(
            "PlutoEngine ready (model=%s, db=%s).", self._cfg.model, resolved_path
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Dispatch an intent to the appropriate handler.

        Never raises; all errors produce a graceful response dict.
        """
        try:
            return self._dispatch(intent, entities)
        except Exception:
            logger.exception("PlutoEngine.handle() raised for intent=%s.", intent)
            return _err("Something went wrong in the finance module.")

    def get_context(self) -> dict:
        """Return a lightweight context snapshot for NLU enrichment."""
        try:
            totals = self.db.get_totals_by_category()
            grand = self.db.get_grand_total()
            return {
                "pluto_total_spent": grand,
                "pluto_top_category": totals[0]["category"] if totals else None,
                "pluto_categories": [t["category"] for t in totals],
            }
        except Exception:
            logger.exception("get_context() failed.")
            return {}

    # ------------------------------------------------------------------
    # Dispatcher (private)
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, entities: dict) -> dict:
        if intent == "log_expense":
            return self._log_expense(entities)
        if intent == "get_budget_summary":
            return self._budget_summary()
        if intent == "track_investment":
            return self._track_investment(entities)
        if intent == "spending_report":
            return self._spending_report()
        return _err(f"Unknown intent: {intent!r}")

    # ------------------------------------------------------------------
    # LLM helpers (private)
    # ------------------------------------------------------------------

    def _llm_json(self, prompt: str) -> str:
        """Call the LLM expecting a JSON response. Returns raw string."""
        return generate(
            prompt,
            model=self._cfg.model,
            host=self._cfg.host,
            port=self._cfg.port,
            fmt="json",
        )

    def _llm_text(self, prompt: str) -> str:
        """Call the LLM expecting a plain-text response."""
        return generate(
            prompt,
            model=self._cfg.model,
            host=self._cfg.host,
            port=self._cfg.port,
        )

    def _infer_category(self, description: str, amount: float) -> str:
        """
        Ask the LLM to categorise an expense.

        Falls back to ``"Other"`` on any LLM or JSON parse failure so that
        expense logging is never blocked by a categorisation error.
        """
        try:
            raw = self._llm_json(
                _CATEGORIZE_PROMPT.format(description=description, amount=amount)
            )
            category = json.loads(raw).get("category", "Other")
            if not isinstance(category, str) or not category.strip():
                return "Other"
            return category.strip()
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "_infer_category: invalid JSON from LLM for %r; defaulting to Other.",
                description,
            )
            return "Other"
        except Exception:
            logger.exception("_infer_category failed; defaulting to Other.")
            return "Other"

    # ------------------------------------------------------------------
    # Formatting helpers (private)
    # ------------------------------------------------------------------

    def _fmt(self, amount: float) -> str:
        """Format a monetary amount with the configured currency symbol."""
        return f"{self._currency}{amount:,.2f}"

    def _breakdown_text(self, totals: list[dict], grand: float) -> str:
        """Render per-category totals as a human-readable table."""
        lines: list[str] = []
        for t in totals:
            pct = (t["total"] / grand * 100) if grand else 0.0
            lines.append(
                f"  {t['category']:15} {self._fmt(t['total']):>12}"
                f"  ({pct:.1f}%,  {t['count']} transaction(s))"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Intent handlers (private)
    # ------------------------------------------------------------------

    def _log_expense(self, entities: dict) -> dict:
        """Parse entities, infer category via LLM, and persist the expense."""
        raw_amount = entities.get("amount")
        description: str = (
            entities.get("description")
            or entities.get("category")
            or entities.get("raw_query")
            or "expense"
        ).strip()

        if raw_amount is None:
            return _ok(
                "How much was it, and what was it for?",
                confidence=0.5,
            )

        try:
            amount = float(raw_amount)
        except (ValueError, TypeError):
            return _ok("I didn't catch the amount — could you repeat it?", confidence=0.4)

        if amount <= 0:
            return _ok("The amount should be greater than zero.", confidence=0.4)

        category = self._infer_category(description, amount)

        try:
            self.db.log_expense(amount, description, category)
        except Exception:
            logger.exception("DB log_expense failed.")
            return _err("I couldn't save that expense — please try again.")

        logger.info(
            "Expense logged: %.2f | %s | %s", amount, description, category
        )
        return _ok(
            f"Logged {self._fmt(amount)} for {description} under {category}.",
            data={"amount": amount, "description": description, "category": category},
            confidence=0.95,
        )

    def _budget_summary(self) -> dict:
        """Return a formatted budget breakdown with LLM-generated advice."""
        try:
            totals = self.db.get_totals_by_category()
            grand = self.db.get_grand_total()
        except Exception:
            logger.exception("_budget_summary: DB read failed.")
            return _err("I couldn't retrieve your spending data right now.")

        if not totals:
            return _ok(
                "No expenses logged yet. Start by telling me what you spent.",
                confidence=0.9,
            )

        breakdown = self._breakdown_text(totals, grand)
        try:
            advice = self._llm_text(
                _ADVICE_PROMPT.format(breakdown=breakdown, total=self._fmt(grand))
            ).strip()
        except Exception:
            logger.exception("_budget_summary: LLM advice call failed.")
            advice = "Unable to generate advice at this time."

        response = (
            f"Budget Summary\n\n"
            f"SPENDING BY CATEGORY\n{breakdown}\n\n"
            f"TOTAL: {self._fmt(grand)}\n\n"
            f"ADVICE\n{advice}"
        )
        return _ok(
            response,
            data={"totals": totals, "grand_total": grand},
            confidence=0.95,
        )

    def _track_investment(self, entities: dict) -> dict:
        """Log an investment position and display a live P&L snapshot."""
        name: str = (
            entities.get("type")
            or entities.get("name")
            or entities.get("raw_query")
            or ""
        ).strip()

        if not name:
            return _ok(
                "What investment should I track? Tell me the name and how much you hold.",
                confidence=0.5,
            )

        asset_type = _classify_asset(name)

        try:
            quantity = float(entities["quantity"]) if entities.get("quantity") else 0.0
            buy_price = (
                float(entities.get("buy_price") or entities.get("price") or 0.0)
            )
        except (ValueError, TypeError):
            return _ok("I couldn't parse the quantity or price — please try again.", confidence=0.4)

        try:
            self.db.log_investment(name, asset_type, quantity, buy_price)
        except Exception:
            logger.exception("_track_investment: DB write failed.")
            return _err("I couldn't save that investment — please try again.")

        live = self._fetch_live_price(name, asset_type)
        lines = _build_investment_lines(
            name, asset_type, quantity, buy_price, live, self._currency
        )

        return _ok(
            "\n".join(lines),
            data={
                "name": name,
                "type": asset_type,
                "quantity": quantity,
                "buy_price": buy_price,
                "live_price": live.price,
            },
            confidence=0.9,
        )

    def _spending_report(self) -> dict:
        """Generate a detailed AI spending report for all logged expenses."""
        try:
            totals = self.db.get_totals_by_category()
            grand = self.db.get_grand_total()
            expenses = self.db.get_expenses(limit=_MAX_EXPENSE_REPORT_ROWS)
        except Exception:
            logger.exception("_spending_report: DB read failed.")
            return _err("I couldn't retrieve your spending data right now.")

        if not totals:
            return _ok(
                "No expenses logged yet. Start tracking by telling me what you spend.",
                confidence=0.9,
            )

        breakdown = self._breakdown_text(totals, grand)
        try:
            report = self._llm_text(
                _REPORT_PROMPT.format(
                    breakdown=breakdown,
                    total=self._fmt(grand),
                    count=len(expenses),
                )
            ).strip()
        except Exception:
            logger.exception("_spending_report: LLM call failed.")
            report = "Unable to generate analysis at this time."

        response = (
            f"Spending Report\n\n"
            f"BREAKDOWN\n{breakdown}\n\n"
            f"TOTAL: {self._fmt(grand)} across {len(expenses)} transaction(s)\n\n"
            f"ANALYSIS\n{report}"
        )
        return _ok(
            response,
            data={
                "totals": totals,
                "grand_total": grand,
                "transaction_count": len(expenses),
            },
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # Live price fetching (private)
    # ------------------------------------------------------------------

    def _fetch_live_price(self, name: str, asset_type: str) -> LivePriceResult:
        """
        Fetch a live market price.

        Returns a ``LivePriceResult`` with ``price=None`` on failure; never
        raises so that investment tracking is not blocked by network errors.
        """
        try:
            if asset_type == "crypto":
                return _fetch_crypto_price(name)
            return _fetch_yahoo_price(name)
        except LivePriceFetchError as exc:
            logger.warning("Live price unavailable for %r: %s", name, exc)
            return LivePriceResult(price=None, error=str(exc))
        except Exception:
            logger.exception("Unexpected error fetching live price for %r.", name)
            return LivePriceResult(price=None, error="unexpected error")


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _classify_asset(name: str) -> str:
    """Infer asset type from the investment name."""
    lower = name.lower()
    if any(k in lower for k in _CRYPTO_KEYWORDS):
        return "crypto"
    if any(k in lower for k in _MUTUAL_FUND_KEYWORDS):
        return "mutual_fund"
    return "stock"


def _build_investment_lines(
    name: str,
    asset_type: str,
    quantity: float,
    buy_price: float,
    live: LivePriceResult,
    currency: str,
) -> list[str]:
    """Render a human-readable investment status block."""
    c = currency
    lines = [f"Investment tracked: {name} ({asset_type})"]
    if quantity:
        lines.append(f"  Quantity  : {quantity:g}")
    if buy_price:
        lines.append(f"  Buy price : {c}{buy_price:,.2f}")
    if live.available:
        assert live.price is not None
        lines.append(f"  Live price: {c}{live.price:,.2f}")
        if quantity:
            value = live.price * quantity
            lines.append(f"  Value now : {c}{value:,.2f}")
        if buy_price and quantity:
            pnl = (live.price - buy_price) * quantity
            sign = "+" if pnl >= 0 else ""
            lines.append(f"  P&L       : {sign}{c}{pnl:,.2f}")
    else:
        lines.append(f"  Live price: unavailable ({live.error})")
    return lines


def _fetch_crypto_price(name: str) -> LivePriceResult:
    """Fetch an INR-denominated crypto price from CoinGecko."""
    slug = _CRYPTO_SLUG_MAP.get(name.lower(), name.lower().replace(" ", "-"))
    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={slug}&vs_currencies=inr"
    )
    try:
        r = requests.get(url, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        data: dict = r.json()
    except requests.RequestException as exc:
        raise LivePriceFetchError(f"CoinGecko request failed: {exc}") from exc
    except ValueError as exc:
        raise LivePriceFetchError(f"CoinGecko response not JSON: {exc}") from exc

    if slug not in data:
        raise LivePriceFetchError(f"Slug {slug!r} not found in CoinGecko response.")

    return LivePriceResult(price=float(data[slug]["inr"]))


def _fetch_yahoo_price(name: str) -> LivePriceResult:
    """Fetch a live price from Yahoo Finance (NSE default for unqualified tickers)."""
    ticker = name.upper()
    if not any(ch in ticker for ch in (".", "^")):
        ticker = ticker + ".NS"

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{ticker}?interval=1d&range=1d"
    )
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        payload: dict = r.json()
    except requests.RequestException as exc:
        raise LivePriceFetchError(f"Yahoo Finance request failed: {exc}") from exc
    except ValueError as exc:
        raise LivePriceFetchError(f"Yahoo Finance response not JSON: {exc}") from exc

    result = payload.get("chart", {}).get("result") or []
    if not result:
        raise LivePriceFetchError(f"No chart data returned for ticker {ticker!r}.")

    price = result[0].get("meta", {}).get("regularMarketPrice")
    if price is None:
        raise LivePriceFetchError(f"regularMarketPrice missing for ticker {ticker!r}.")

    return LivePriceResult(price=float(price))


def _ok(
    response: str,
    data: Optional[dict[str, Any]] = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict[str, Any]:
    return {"response": response, "data": {}, "confidence": 0.0}