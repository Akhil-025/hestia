"""
modules/pluto/networth.py

Multi-currency net worth (backlog #136).

Pluto's ``investments`` table has no currency column and every other feature
assumes one currency. Rather than change that schema, the currency of a holding
is kept as a small ``{name: ISO code}`` map in the settings table. A holding
without an entry is taken to be in the home currency, except that ``.NS`` / ``.BO`` names
are INR. The reply says how many were assumed, because a wrong assumption
is the main way this total can mislead.

Valuation: live price x quantity where a live price exists, otherwise cost.
A stock's live price is taken to be in the holding's own currency (that is how
Yahoo quotes a listing). Crypto live prices come from CoinGecko in INR, so those
are converted from INR whatever the holding's currency is set to.

Conversion uses ``fx_fn(from, to)``. A holding whose rate cannot be fetched is
left out of the total and named, so the total is then a floor, not a sum.
Only investments are counted: no cash, property or loans.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from .logging_config import get_logger

logger = get_logger(__name__)

SETTING_KEY = "holding_currency"
MAX_HOLDINGS_SHOWN = 12

_SYMBOLS = {"$": "USD", "\u20ac": "EUR", "\u00a3": "GBP", "\u20b9": "INR", "\u00a5": "JPY"}
_WORDS = {
    "dollar": "USD", "dollars": "USD", "usd": "USD", "us": "USD", "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "rupee": "INR", "rupees": "INR",
    "yen": "JPY", "dirham": "AED", "dirhams": "AED", "yuan": "CNY", "franc": "CHF",
}
KNOWN = {"USD", "EUR", "GBP", "INR", "JPY", "AED", "SGD", "CAD", "AUD", "CHF", "CNY", "HKD",
         "NZD", "SEK", "NOK", "DKK", "ZAR", "KRW", "THB", "MYR"}


def parse_currency(raw: Any) -> Optional[str]:
    """ISO code for '$', 'dollars', 'usd', 'EUR' ...; None when not recognised."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text[:1] in _SYMBOLS:
        return _SYMBOLS[text[:1]]
    for word in re.findall(r"[A-Za-z]+", text):
        if word.lower() in _WORDS:
            return _WORDS[word.lower()]
        if word.upper() in KNOWN:
            return word.upper()
    return None


def base_code(symbol: str) -> str:
    return _SYMBOLS.get(symbol, "INR")


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def aggregate_lots(investments: list[dict]) -> dict:
    """name(lower) -> {name, type, quantity, cost}, summing every lot of that holding."""
    out: dict = {}
    for row in investments:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        q = float(row.get("quantity") or 0)
        p = float(row.get("buy_price") or 0)
        h = out.setdefault(_key(name), {"name": name, "type": row.get("type") or "stock",
                                        "quantity": 0.0, "cost": 0.0})
        h["quantity"] += q
        h["cost"] += q * p
    return out


class NetWorth:
    def __init__(self, db: Any, base_currency: str = "INR", symbol: str = "\u20b9",
                 price_fn: Optional[Callable[[str, str], Optional[float]]] = None,
                 fx_fn: Optional[Callable[[str, str], float]] = None):
        self.db = db
        self.base = base_currency
        self.symbol = symbol
        self._price_fn = price_fn
        self._fx_fn = fx_fn

    # -- stored currency map ---------------------------------------------------

    def _load(self) -> dict:
        try:
            data = json.loads(self.db.get_setting(SETTING_KEY) or "{}")
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}

    def _save(self, mapping: dict) -> None:
        self.db.set_setting(SETTING_KEY, json.dumps(mapping, sort_keys=True))

    def _currency_of(self, name: str, mapping: dict) -> tuple[str, bool]:
        """(ISO code, was_assumed)."""
        k = _key(name)
        if k in mapping:
            return mapping[k], False
        if k.upper().endswith((".NS", ".BO")):
            return "INR", False
        return self.base, True

    def set_currency(self, entities: dict) -> dict:
        name = str(entities.get("name") or entities.get("ticker") or "").strip()
        if not name:
            return _reply("Which holding, and which currency? e.g. 'my AAPL holding is in USD'.", 0.5)
        held = aggregate_lots(self.db.get_investments())
        match = held.get(_key(name))
        if match is None:
            return _reply(f"I don't have a holding called {name!r}. Track it first.", 0.5)
        raw = entities.get("currency")
        mapping = self._load()
        if str(raw or "").strip().lower() in ("clear", "default", "reset", "none"):
            mapping.pop(_key(name), None)
            self._save(mapping)
            return _reply(f"{match['name']} goes back to the default currency ({self.base}).", 0.9)
        code = parse_currency(raw)
        if code is None:
            return _reply("I didn't recognise that currency. Use a code such as USD, EUR, GBP or INR.", 0.4)
        mapping[_key(name)] = code
        self._save(mapping)
        return _reply(f"Noted: {match['name']} is held in {code}.", 0.9,
                      {"name": match["name"], "currency": code})

    # -- the aggregate ---------------------------------------------------------

    def net_worth(self, entities: Optional[dict] = None) -> dict:
        held = aggregate_lots(self.db.get_investments())
        held = {k: h for k, h in held.items() if h["quantity"] > 0}
        if not held:
            return _reply("You haven't tracked any investments yet, so there is no net worth to add up.", 0.6)
        mapping = self._load()
        rates: dict = {}

        def rate(src: str) -> Optional[float]:
            if src == self.base:
                return 1.0
            if src not in rates:
                try:
                    rates[src] = float(self._fx_fn(src, self.base)) if self._fx_fn else None
                except Exception as e:
                    logger.info("fx rate %s->%s unavailable: %s", src, self.base, e)
                    rates[src] = None
                if rates[src] is not None and not rates[src] > 0:
                    rates[src] = None
            return rates[src]

        rows, per_ccy, skipped, assumed, at_cost = [], {}, [], 0, 0
        total = 0.0
        for h in sorted(held.values(), key=lambda x: x["name"].lower()):
            ccy, was_assumed = self._currency_of(h["name"], mapping)
            assumed += was_assumed
            price = self._price_fn(h["name"], h["type"]) if self._price_fn else None
            if price is not None and price > 0:
                value = h["quantity"] * price
                value_ccy = "INR" if h["type"] == "crypto" else ccy
                basis = "live"
            else:
                value, value_ccy, basis = h["cost"], ccy, "cost"
                at_cost += 1
            r = rate(value_ccy)
            if r is None:
                skipped.append(f"{h['name']} ({value_ccy})")
                continue
            base_value = value * r
            total += base_value
            per_ccy[value_ccy] = per_ccy.get(value_ccy, 0.0) + value
            rows.append({"name": h["name"], "currency": value_ccy, "value": value,
                         "value_base": base_value, "basis": basis})
        rows.sort(key=lambda r: -r["value_base"])
        lines = [f"Net worth (investments only): {self.symbol}{total:,.0f}"
                 + (" at least" if skipped else "") + f" in {self.base}."]
        if len(per_ccy) > 1 or (per_ccy and self.base not in per_ccy):
            lines.append("By currency: " + ", ".join(
                f"{c} {v:,.0f}" for c, v in sorted(per_ccy.items())) + ".")
        for r in rows[:MAX_HOLDINGS_SHOWN]:
            native = "" if r["currency"] == self.base else f" ({r['currency']} {r['value']:,.0f})"
            lines.append(f"  {r['name']}: {self.symbol}{r['value_base']:,.0f}{native}"
                         + ("  [at cost]" if r["basis"] == "cost" else ""))
        if len(rows) > MAX_HOLDINGS_SHOWN:
            lines.append(f"  ...and {len(rows) - MAX_HOLDINGS_SHOWN} more.")
        if skipped:
            lines.append("Left out because no exchange rate was available: " + ", ".join(skipped)
                         + ". The total above is therefore a floor.")
        if assumed:
            lines.append(f"{assumed} holding(s) have no currency set and are taken to be in {self.base}; "
                         "say 'my X holding is in USD' to fix that.")
        if at_cost:
            lines.append(f"{at_cost} holding(s) valued at what you paid because no live price was available.")
        lines.append("Rates are today's. No cash, property or loans are included.")
        return _reply("\n".join(lines), 0.8 if not skipped else 0.5, {
            "total": total, "base": self.base, "by_currency": per_ccy, "holdings": rows,
            "skipped": skipped, "assumed_currency": assumed, "at_cost": at_cost})


def _reply(text: str, confidence: float = 0.9, data: Optional[dict] = None) -> dict:
    return {"response": text, "data": data or {}, "confidence": confidence}
