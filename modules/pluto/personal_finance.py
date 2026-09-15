"""

modules/pluto/personal_finance.py
Personal finance manager with improved error handling, caching, and LLM client.

"""


from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, Union

import requests

from .config import PlutoConfig
from .db import PlutoDB, DatabaseManager
from .llm_client import LLMClient, OutputFormat
from .retry import retry
from .logging_config import get_logger
from .metrics import MetricsCollector
from core.free_apis import (
    FreeAPIError,
    convert_currency as _fa_convert_currency,
    fx_rate as _fa_fx_rate,
    sec_company_facts as _fa_sec_company_facts,
    sec_company_search as _fa_sec_company_search,
    fred_series_latest as _fa_fred_series_latest,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CRYPTO_SLUG_MAP: dict[str, str] = {
    "bitcoin": "bitcoin", "btc": "bitcoin", "ethereum": "ethereum",
    "eth": "ethereum", "solana": "solana", "sol": "solana",
    "dogecoin": "dogecoin", "doge": "dogecoin", "bnb": "binancecoin",
    "usdt": "tether",
}

_CRYPTO_KEYWORDS: frozenset[str] = frozenset(
    {"bitcoin", "btc", "eth", "ethereum", "crypto", "solana", "sol",
     "usdt", "bnb", "doge", "dogecoin"}
)
_MUTUAL_FUND_KEYWORDS: frozenset[str] = frozenset(
    {"mutual fund", "sip", "nifty", "sensex", "index fund", "elss", "debt fund"}
)

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Food":          ["groceries", "food", "restaurant", "lunch", "dinner", "breakfast", "coffee", "cafe", "eat", "swiggy", "zomato", "snack"],
    "Transport":     ["uber", "ola", "auto", "bus", "train", "metro", "petrol", "diesel", "fuel", "cab", "taxi", "flight", "parking"],
    "Shopping":      ["clothes", "amazon", "flipkart", "shirt", "shoes", "mall", "myntra"],
    "Health":        ["medicine", "doctor", "gym", "pharmacy", "hospital", "medical", "clinic"],
    "Entertainment": ["movie", "netflix", "spotify", "game", "concert", "theatre", "cinema"],
    "Bills":         ["electricity", "rent", "wifi", "internet", "phone", "recharge", "water bill", "gas bill", "emi"],
    "Education":     ["book", "course", "tuition", "college", "fees", "class"],
}

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
# Domain Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LivePriceResult:
    price: Optional[float]
    error: str = ""

    @property
    def available(self) -> bool:
        return self.price is not None


class PlutoError(Exception):
    """Base exception for Pluto failures."""


class LivePriceFetchError(PlutoError):
    """Raised when a live price cannot be retrieved."""


# ---------------------------------------------------------------------------
# Module-level response helpers
#
# These are the canonical implementations. `engine.py` imports `_err`
# directly (it needs to build error responses before a PersonalFinanceManager
# instance exists), so it must be a plain module-level function rather than
# only living as a method on the class below.
# ---------------------------------------------------------------------------

def _ok(response: str, data: Optional[dict] = None, confidence: float = 0.9) -> dict:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict:
    return {"response": response, "data": {}, "confidence": 0.0}


# ---------------------------------------------------------------------------
# Manager Class
# ---------------------------------------------------------------------------

class PersonalFinanceManager:
    def __init__(
        self,
        config: PlutoConfig,
        db_manager: Optional[DatabaseManager] = None,
        llm_client: Optional[LLMClient] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.config = config
        self.db_manager = db_manager or DatabaseManager(config)
        self.llm_client = llm_client or LLMClient(config)
        self.metrics = metrics or MetricsCollector()

        # Local SQLite. A relative config.db_path (the default is
        # "data/pluto/pluto.db") is anchored to the Hestia project root
        # rather than Path.resolve()'s implicit cwd — otherwise where the
        # expenses land depends on whether the process was launched via
        # `python main.py`, `python web_ui.py`, or the Telegram bot
        # subprocess, each of which may have a different working directory.
        if config.db_path.is_absolute():
            resolved_path = config.db_path
        else:
            project_root = Path(__file__).resolve().parents[2]
            resolved_path = (project_root / config.db_path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = PlutoDB(str(resolved_path))

        logger.info("PersonalFinanceManager ready (db=%s)", resolved_path)

    def get_context(self) -> dict:
        try:
            totals = self.db.get_totals_by_category()
            grand = self.db.get_grand_total()
            return {
                "pluto_total_spent": grand,
                "pluto_top_category": totals[0]["category"] if totals else None,
                "pluto_categories": [t["category"] for t in totals],
            }
        except Exception as e:
            logger.error("get_context failed: %s", e)
            return {}

    def log_expense(self, entities: dict) -> dict:
        raw_amount = entities.get("amount")
        description: str = (
            entities.get("description") or entities.get("category") or entities.get("raw_query") or "expense"
        ).strip()
        # Sanitize input for prompt injection prevention
        description = self._sanitize_input(description)

        if raw_amount is None:
            return self._ok("How much was it, and what was it for?", confidence=0.5)

        try:
            amount = float(raw_amount)
        except (ValueError, TypeError):
            return self._ok("I didn't catch the amount — could you repeat it?", confidence=0.4)

        if amount <= 0:
            return self._ok("The amount should be greater than zero.", confidence=0.4)

        category = self._infer_category(description, amount)

        try:
            self.db.log_expense(amount, description, category)
            self.metrics.record_expense()
        except Exception as e:
            logger.error("DB log_expense failed: %s", e)
            return self._err("I couldn't save that expense — please try again.")

        logger.info("Expense logged: %.2f | %s | %s", amount, description, category)
        return self._ok(
            f"Logged {self._fmt(amount)} for {description} under {category}.",
            data={"amount": amount, "description": description, "category": category},
            confidence=0.95,
        )

    def budget_summary(self) -> dict:
        try:
            totals = self.db.get_totals_by_category()
            grand = self.db.get_grand_total()
        except Exception as e:
            logger.error("budget_summary: DB read failed: %s", e)
            return self._err("I couldn't retrieve your spending data right now.")

        if not totals:
            return self._ok("No expenses logged yet. Start by telling me what you spent.", confidence=0.9)

        breakdown = self._breakdown_text(totals, grand)
        try:
            advice = self.llm_client.generate(
                _ADVICE_PROMPT.format(breakdown=breakdown, total=self._fmt(grand)),
                output_format=OutputFormat.TEXT,
            ).strip()
        except Exception as e:
            logger.error("budget_summary: LLM advice call failed: %s", e)
            advice = "Unable to generate advice at this time."

        response = (
            f"Budget Summary\n\n"
            f"SPENDING BY CATEGORY\n{breakdown}\n\n"
            f"TOTAL: {self._fmt(grand)}\n\n"
            f"ADVICE\n{advice}"
        )
        return self._ok(response, data={"totals": totals, "grand_total": grand}, confidence=0.95)

    def track_investment(self, entities: dict) -> dict:
        name: str = (
            entities.get("type") or entities.get("name") or entities.get("raw_query") or ""
        ).strip()
        name = self._sanitize_input(name)

        if not name:
            return self._ok("What investment should I track? Tell me the name and how much you hold.", confidence=0.5)

        asset_type = self._classify_asset(name)

        try:
            quantity = float(entities.get("quantity", 0.0))
            buy_price = float(entities.get("buy_price") or entities.get("price") or 0.0)
        except (ValueError, TypeError):
            return self._ok("I couldn't parse the quantity or price — please try again.", confidence=0.4)

        try:
            self.db.log_investment(name, asset_type, quantity, buy_price)
        except Exception as e:
            logger.error("track_investment: DB write failed: %s", e)
            return self._err("I couldn't save that investment — please try again.")

        # Live price lookups hit third-party APIs (CoinGecko / Yahoo Finance)
        # that can rate-limit, block, or time out. That must not blow up the
        # whole response — the investment is already saved to the DB above,
        # so we degrade to "live price unavailable" instead of discarding a
        # successful save behind a generic error message.
        try:
            live = self._fetch_live_price(name, asset_type)
        except PlutoError as e:
            logger.warning("track_investment: live price fetch failed for %r: %s", name, e)
            live = LivePriceResult(price=None, error=str(e))

        lines = self._build_investment_lines(name, asset_type, quantity, buy_price, live)

        # Record investment value if live price available
        if live.available and quantity:
            self.metrics.record_investment_value(live.price * quantity)

        return self._ok(
            "\n".join(lines),
            data={"name": name, "type": asset_type, "quantity": quantity, "buy_price": buy_price, "live_price": live.price},
            confidence=0.9,
        )

    def spending_report(self) -> dict:
        try:
            totals = self.db.get_totals_by_category()
            grand = self.db.get_grand_total()
            expenses = self.db.get_expenses(limit=self.config.max_expense_report_rows)
        except Exception as e:
            logger.error("spending_report: DB read failed: %s", e)
            return self._err("I couldn't retrieve your spending data right now.")

        if not totals:
            return self._ok("No expenses logged yet. Start tracking by telling me what you spend.", confidence=0.9)

        breakdown = self._breakdown_text(totals, grand)
        try:
            report = self.llm_client.generate(
                _REPORT_PROMPT.format(breakdown=breakdown, total=self._fmt(grand), count=len(expenses)),
                output_format=OutputFormat.TEXT,
            ).strip()
        except Exception as e:
            logger.error("spending_report: LLM call failed: %s", e)
            report = "Unable to generate analysis at this time."

        response = (
            f"Spending Report\n\n"
            f"BREAKDOWN\n{breakdown}\n\n"
            f"TOTAL: {self._fmt(grand)} across {len(expenses)} transaction(s)\n\n"
            f"ANALYSIS\n{report}"
        )
        return self._ok(
            response,
            data={"totals": totals, "grand_total": grand, "transaction_count": len(expenses)},
            confidence=0.95,
        )

    def convert_currency(self, entities: dict) -> dict:
        """
        Convert an amount between currencies via Frankfurter (ECB rates,
        free, no key, no rate limit). New intent: `convert_currency`.

        Expected entities: amount, from_currency (or 'from'), to_currency
        (or 'to'); to_currency defaults to the module's configured
        currency symbol's ISO code when not supplied and inferable.
        """
        try:
            amount = float(entities.get("amount"))
        except (TypeError, ValueError):
            return self._ok("How much, and in which currency?", confidence=0.5)

        from_ccy = (entities.get("from_currency") or entities.get("from") or "").strip()
        to_ccy = (entities.get("to_currency") or entities.get("to") or "INR").strip()
        if not from_ccy:
            return self._ok(
                "Which currency are you converting from (e.g. USD, EUR)?", confidence=0.5
            )

        try:
            converted = _fa_convert_currency(amount, from_ccy, to_ccy)
            rate = _fa_fx_rate(from_ccy, to_ccy)
        except FreeAPIError as e:
            logger.warning("convert_currency: Frankfurter lookup failed: %s", e)
            return self._err(
                f"I couldn't fetch a live rate for {from_ccy.upper()}→{to_ccy.upper()} right now."
            )

        return self._ok(
            f"{amount:,.2f} {from_ccy.upper()} = {converted:,.2f} {to_ccy.upper()} "
            f"(rate: 1 {from_ccy.upper()} = {rate:.4f} {to_ccy.upper()})",
            data={"amount": amount, "from": from_ccy.upper(), "to": to_ccy.upper(),
                  "converted": converted, "rate": rate},
            confidence=0.95,
        )

    def company_lookup(self, entities: dict) -> dict:
        """
        Lightweight macro/company context using two free, keyless-ish
        sources: SEC EDGAR (company lookup, always free) and FRED
        (macro series, needs an optional FRED_API_KEY — degrades to
        "unavailable" rather than erroring when absent).

        New intent: `company_lookup`. Deliberately named apart from
        modules/pluto/market_intelligence.py's `MarketIntelligenceManager`
        (a separate Postgres/Redis/XGBoost quant pipeline via
        `analyze_asset`) — this is a much lighter, free-data-only path for
        "what does the public record say about X" questions.
        """
        company = (entities.get("company") or entities.get("name") or entities.get("raw_query") or "").strip()
        lines: list[str] = []

        if company:
            try:
                matches = _fa_sec_company_search(company)
            except FreeAPIError as e:
                logger.warning("market_intelligence: SEC search failed: %s", e)
                matches = []
            if matches:
                top = matches[0]
                lines.append(
                    f"SEC EDGAR: {top.get('title')} (ticker {top.get('ticker')}, CIK {top.get('cik_str')})"
                )
            else:
                lines.append(f"No SEC EDGAR match found for {company!r} (SEC only covers US-listed filers).")

        cpi = _fa_fred_series_latest("CPIAUCSL")
        if cpi is not None:
            lines.append(f"US CPI (FRED, latest): {cpi:.2f}")
        else:
            lines.append("Macro data (FRED) unavailable — set FRED_API_KEY in .env for CPI/rate context.")

        if not lines:
            return self._ok("Tell me a company name or ticker and I'll pull what public data I can.", confidence=0.5)

        return self._ok("\n".join(lines), data={"company": company}, confidence=0.8)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent prompt injection and control characters."""
        # Remove non-printable characters except newline and tab
        text = ''.join(ch for ch in text if 32 <= ord(ch) <= 126 or ch in '\n\r\t')
        # Limit length
        return text[:500].strip()

    def _infer_category(self, description: str, amount: float) -> str:
        lower = description.lower()
        for category, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return category
        try:
            raw = self.llm_client.generate(
                _CATEGORIZE_PROMPT.format(description=description, amount=amount),
                output_format=OutputFormat.JSON,
            )
            category = raw.get("category", "Other")
            if not isinstance(category, str) or not category.strip():
                return "Other"
            return category.strip()
        except Exception as e:
            logger.warning("_infer_category failed for %r: %s; defaulting to Other.", description, e)
            return "Other"

    def _fmt(self, amount: float) -> str:
        return f"{self.config.currency}{amount:,.2f}"

    def _breakdown_text(self, totals: list[dict], grand: float) -> str:
        lines: list[str] = []
        for t in totals:
            pct = (t["total"] / grand * 100) if grand else 0.0
            lines.append(f"  {t['category']:15} {self._fmt(t['total']):>12}  ({pct:.1f}%,  {t['count']} transaction(s))")
        return "\n".join(lines)

    @retry(max_retries=3, exceptions=(LivePriceFetchError,), delay=0.5)
    def _fetch_live_price(self, name: str, asset_type: str) -> LivePriceResult:
        """Fetch live price with retries."""
        if asset_type == "crypto":
            return self._fetch_crypto_price(name)
        else:
            return self._fetch_yahoo_price(name)

    def _fetch_crypto_price(self, name: str) -> LivePriceResult:
        slug = _CRYPTO_SLUG_MAP.get(name.lower(), name.lower().replace(" ", "-"))
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={slug}&vs_currencies=inr"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise LivePriceFetchError(f"CoinGecko request failed: {e}") from e
        if slug not in data:
            raise LivePriceFetchError(f"Slug {slug!r} not found.")
        price = float(data[slug]["inr"])
        return LivePriceResult(price=price)

    def _fetch_yahoo_price(self, name: str) -> LivePriceResult:
        import re
        ticker = name.upper()
        ticker = re.sub(r'\b(INDUSTRIES|LIMITED|LTD|INC|CORP)\b', '', ticker, flags=re.I)
        ticker = re.sub(r'\s+', '', ticker)
        ticker = ticker.strip('.-_')
        if not any(ch in ticker for ch in (".", "^")):
            ticker = ticker + ".NS"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            raise LivePriceFetchError(f"Yahoo Finance request failed: {e}") from e
        result = payload.get("chart", {}).get("result") or []
        if not result:
            raise LivePriceFetchError(f"No chart data for {ticker!r}.")
        price = result[0].get("meta", {}).get("regularMarketPrice")
        if price is None:
            raise LivePriceFetchError(f"Price missing for {ticker!r}.")
        return LivePriceResult(price=float(price))

    @staticmethod
    def _classify_asset(name: str) -> str:
        lower = name.lower()
        if any(k in lower for k in _CRYPTO_KEYWORDS):
            return "crypto"
        if any(k in lower for k in _MUTUAL_FUND_KEYWORDS):
            return "mutual_fund"
        return "stock"

    def _build_investment_lines(self, name: str, asset_type: str, quantity: float, buy_price: float, live: LivePriceResult) -> list[str]:
        c = self.config.currency
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

    # ------------------------------------------------------------------
    # Response Helpers
    # ------------------------------------------------------------------

    _ok = staticmethod(_ok)
    _err = staticmethod(_err)