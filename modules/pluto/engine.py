# modules/pluto/engine.py

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from core.ollama_client import generate
from modules.base import BaseModule
from .db import PlutoDB

log = logging.getLogger(__name__)

_DB_PATH = str(Path(__file__).parent.parent.parent / "data" / "pluto" / "pluto.db")

# ── Prompts ───────────────────────────────────────────────────────────────────

_CATEGORIZE_PROMPT = """You are a finance assistant.
Categorize this expense: "{description}" (amount: {amount})

Respond with ONLY valid JSON:
{{"category": "one of: Food, Transport, Shopping, Health, Entertainment, Bills, Education, Other"}}

JSON only. No explanation."""

_ADVICE_PROMPT = """You are Pluto, a personal finance guru.
Here is the user's spending by category:
{breakdown}

Total spent: {total}

Give specific, actionable advice on:
1. Which category to cut first and how
2. A simple weekly budget target
3. One habit change that will have the biggest impact

Keep it under 150 words. Be direct and practical."""

_REPORT_PROMPT = """You are Pluto, a personal finance guru.
Spending breakdown:
{breakdown}

Total: {total}
Number of transactions: {count}

Write a spending report with:
- WHERE the money is going (biggest categories)
- WHAT to cut and by how much
- HOW to track it going forward

Under 200 words. Practical and specific."""


class PlutoEngine(BaseModule):
    name = "pluto"
    _INTENTS = {
        "log_expense",
        "get_budget_summary",
        "track_investment",
        "spending_report",
    }

    def __init__(self, ollama_cfg: dict = None):
        self._ollama = ollama_cfg or {}
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        self.db = PlutoDB(_DB_PATH)

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        if intent == "log_expense":
            return self._log_expense(entities)
        if intent == "get_budget_summary":
            return self._budget_summary()
        if intent == "track_investment":
            return self._track_investment(entities)
        if intent == "spending_report":
            return self._spending_report()
        return {"response": "Unknown Pluto intent.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        try:
            totals = self.db.get_totals_by_category()
            grand  = self.db.get_grand_total()
            return {
                "pluto_total_spent":   grand,
                "pluto_top_category":  totals[0]["category"] if totals else None,
                "pluto_categories":    [t["category"] for t in totals],
            }
        except Exception:
            return {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ollama_call(self, prompt: str) -> str:
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
            fmt="json",
        )

    def _ollama_text(self, prompt: str) -> str:
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
        )

    def _infer_category(self, description: str, amount: float) -> str:
        try:
            raw = self._ollama_call(
                _CATEGORIZE_PROMPT.format(description=description, amount=amount)
            )
            return json.loads(raw).get("category", "Other")
        except Exception:
            return "Other"

    def _breakdown_text(self, totals: list[dict], grand: float) -> str:
        lines = []
        for t in totals:
            pct = (t["total"] / grand * 100) if grand else 0
            lines.append(
                f"  {t['category']:15} ₹{t['total']:>10.2f}  ({pct:.1f}%,  {t['count']} transactions)"
            )
        return "\n".join(lines)

    # ── log_expense ───────────────────────────────────────────────────────────

    def _log_expense(self, entities: dict) -> dict:
        amount = entities.get("amount")
        desc   = (
            entities.get("description")
            or entities.get("category")
            or entities.get("raw_query", "expense")
        )

        if not amount:
            return {
                "response": "How much was it, and what was it for?",
                "data": {},
                "confidence": 0.5,
            }

        try:
            amount = float(amount)
        except (ValueError, TypeError):
            return {"response": "I didn't catch the amount.", "data": {}, "confidence": 0.4}

        category = self._infer_category(desc, amount)
        self.db.log_expense(amount, desc, category)

        return {
            "response": f"Logged ₹{amount:.2f} for {desc} under {category}.",
            "data": {"amount": amount, "description": desc, "category": category},
            "confidence": 0.95,
        }

    # ── get_budget_summary ────────────────────────────────────────────────────

    def _budget_summary(self) -> dict:
        totals = self.db.get_totals_by_category()
        grand  = self.db.get_grand_total()

        if not totals:
            return {
                "response": "No expenses logged yet. Start by telling me what you spent.",
                "data": {},
                "confidence": 0.9,
            }

        breakdown = self._breakdown_text(totals, grand)
        advice    = self._ollama_text(
            _ADVICE_PROMPT.format(breakdown=breakdown, total=f"₹{grand:.2f}")
        )

        response = (
            f"Budget Summary\n\n"
            f"SPENDING BY CATEGORY\n{breakdown}\n\n"
            f"TOTAL: ₹{grand:.2f}\n\n"
            f"ADVICE\n{advice.strip()}"
        )

        return {
            "response": response,
            "data": {"totals": totals, "grand_total": grand},
            "confidence": 0.95,
        }

    # ── track_investment ──────────────────────────────────────────────────────

    def _track_investment(self, entities: dict) -> dict:
        name     = entities.get("type") or entities.get("name") or entities.get("raw_query", "")
        quantity = entities.get("quantity")
        buy_price= entities.get("buy_price") or entities.get("price")

        if not name:
            return {
                "response": "What investment should I track? Tell me the name and how much you hold.",
                "data": {},
                "confidence": 0.5,
            }

        # infer asset type
        crypto_keywords = {"bitcoin", "btc", "eth", "ethereum", "crypto",
                           "solana", "sol", "usdt", "bnb", "doge"}
        mf_keywords     = {"mutual fund", "sip", "nifty", "sensex",
                           "index fund", "elss", "debt fund"}

        name_lower = name.lower()
        if any(k in name_lower for k in crypto_keywords):
            asset_type = "crypto"
        elif any(k in name_lower for k in mf_keywords):
            asset_type = "mutual_fund"
        else:
            asset_type = "stock"

        # log to DB
        qty       = float(quantity)   if quantity  else 0.0
        buy_price = float(buy_price)  if buy_price else 0.0
        self.db.log_investment(name, asset_type, qty, buy_price)

        # fetch live price
        live_price, live_msg = self._fetch_live_price(name, asset_type)

        lines = [f"Investment tracked: {name} ({asset_type})"]
        if qty:
            lines.append(f"  Quantity  : {qty}")
        if buy_price:
            lines.append(f"  Buy price : ₹{buy_price:.2f}")
        if live_price:
            current_value = live_price * qty if qty else live_price
            pnl = (live_price - buy_price) * qty if buy_price and qty else None
            lines.append(f"  Live price: ₹{live_price:.2f}")
            if qty:
                lines.append(f"  Value now : ₹{current_value:.2f}")
            if pnl is not None:
                sign = "+" if pnl >= 0 else ""
                lines.append(f"  P&L       : {sign}₹{pnl:.2f}")
        else:
            lines.append(f"  Live price: {live_msg}")

        return {
            "response": "\n".join(lines),
            "data": {
                "name": name, "type": asset_type,
                "quantity": qty, "buy_price": buy_price,
                "live_price": live_price,
            },
            "confidence": 0.9,
        }

    def _fetch_live_price(self, name: str, asset_type: str) -> tuple[Optional[float], str]:
        """Returns (price_float, error_message). Price is None on failure."""
        try:
            if asset_type == "crypto":
                return self._fetch_crypto(name)
            else:
                return self._fetch_yahoo(name)
        except Exception as e:
            log.warning("Pluto: live price fetch failed for %s: %s", name, e)
            return None, "unavailable"

    def _fetch_crypto(self, name: str) -> tuple[Optional[float], str]:
        slug_map = {
            "bitcoin": "bitcoin", "btc": "bitcoin",
            "ethereum": "ethereum", "eth": "ethereum",
            "solana": "solana", "sol": "solana",
            "dogecoin": "dogecoin", "doge": "dogecoin",
            "bnb": "binancecoin",
        }
        slug = slug_map.get(name.lower(), name.lower().replace(" ", "-"))
        url  = f"https://api.coingecko.com/api/v3/simple/price?ids={slug}&vs_currencies=inr"
        r    = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        if slug in data:
            return float(data[slug]["inr"]), ""
        return None, "symbol not found"

    def _fetch_yahoo(self, name: str) -> tuple[Optional[float], str]:
        # append .NS for Indian stocks, .BO for BSE
        ticker = name.upper()
        if not any(x in ticker for x in (".", "^")):
            ticker = ticker + ".NS"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        result = r.json().get("chart", {}).get("result", [])
        if result:
            price = result[0].get("meta", {}).get("regularMarketPrice")
            if price:
                return float(price), ""
        return None, "price unavailable"

    # ── spending_report ───────────────────────────────────────────────────────

    def _spending_report(self) -> dict:
        totals    = self.db.get_totals_by_category()
        grand     = self.db.get_grand_total()
        expenses  = self.db.get_expenses(limit=200)

        if not totals:
            return {
                "response": "No expenses logged yet. Start tracking by telling me what you spend.",
                "data": {},
                "confidence": 0.9,
            }

        breakdown = self._breakdown_text(totals, grand)
        report    = self._ollama_text(
            _REPORT_PROMPT.format(
                breakdown=breakdown,
                total=f"₹{grand:.2f}",
                count=len(expenses),
            )
        )

        response = (
            f"Spending Report\n\n"
            f"BREAKDOWN\n{breakdown}\n\n"
            f"TOTAL: ₹{grand:.2f} across {len(expenses)} transactions\n\n"
            f"ANALYSIS\n{report.strip()}"
        )

        return {
            "response": response,
            "data": {"totals": totals, "grand_total": grand, "transaction_count": len(expenses)},
            "confidence": 0.95,
        }