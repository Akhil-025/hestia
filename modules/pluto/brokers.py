"""
modules/pluto/brokers.py

Broker holdings sync (backlog #131).

Two sources, both read-only (Pluto never places orders):

* Zerodha Kite Connect: ``GET https://api.kite.trade/portfolio/holdings``. Needs
  ``KITE_API_KEY`` and ``KITE_ACCESS_TOKEN`` in the environment. Kite access
  tokens expire every day (the login is a browser step Pluto cannot do), so an
  expired token gets a clear message, not a retry loop. Credentials are never
  accepted through chat, so they cannot end up in logs or the conversation history.
* A holdings CSV (Groww, Zerodha Console, or any export with a symbol/name,
  quantity and average price column). Groww has no API Pluto uses.

Syncing is a preview unless ``confirm`` is set. It only ever ADDS a lot for
the shortfall when the broker shows more of a holding than Pluto tracks. When
Pluto tracks more than the broker shows, it reports that and removes
nothing, because the investments table is a log you may have entered by hand.
Holdings are matched ignoring case, spaces and a .NS/.BO suffix; new ones are
saved as SYMBOL.NS (or .BO) so live prices work.

Not tested against the live Kite endpoint here (no network access to it):
parsing and sync logic are exercised with fixtures shaped like Kite's documented
response.
"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from .logging_config import get_logger
from .networth import aggregate_lots
from .throttle import MONITOR

logger = get_logger(__name__)

KITE_URL = "https://api.kite.trade/portfolio/holdings"
SOURCE = "kite"
MAX_ROWS = 500
MAX_CSV_BYTES = 2_000_000
MAX_QTY = 1e9
_TRUE = {"1", "y", "yes", "true", "confirm", "confirmed", "ok", "go"}


class BrokerError(Exception):
    """A failure phrased for the user."""


@dataclass(frozen=True)
class BrokerHolding:
    symbol: str
    exchange: str
    quantity: float
    avg_price: float


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        v = float(str(value).replace(",", "").replace("\u20b9", "").strip())
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_kite_holdings(payload: Any) -> list[BrokerHolding]:
    """Holdings from Kite's JSON. Skips rows that are unusable instead of failing the lot."""
    if not isinstance(payload, dict) or payload.get("status") not in (None, "success"):
        raise BrokerError("Kite returned an error response.")
    out = []
    for row in (payload.get("data") or [])[:MAX_ROWS]:
        if not isinstance(row, dict):
            continue
        symbol = re.sub(r"[^A-Z0-9&\-_]", "", str(row.get("tradingsymbol") or "").upper())
        qty = (_num(row.get("quantity")) or 0) + (_num(row.get("t1_quantity")) or 0)
        avg = _num(row.get("average_price"))
        if not symbol or avg is None or avg < 0 or not 0 < qty <= MAX_QTY:
            continue
        exchange = "BO" if str(row.get("exchange") or "").upper() == "BSE" else "NS"
        out.append(BrokerHolding(symbol, exchange, qty, avg))
    return out


def fetch_kite_holdings(api_key: str, access_token: str, get: Callable = requests.get) -> list[BrokerHolding]:
    """Call Kite. Raises BrokerError with a user-facing message on any failure."""
    import time as _t
    if MONITOR.cooldown_remaining(SOURCE) > 0:
        raise BrokerError(f"Kite asked us to slow down; try again in about "
                          f"{math.ceil(MONITOR.cooldown_remaining(SOURCE))}s.")
    try:
        resp = get(KITE_URL, headers={"X-Kite-Version": "3",
                                      "Authorization": f"token {api_key}:{access_token}"}, timeout=15)
    except Exception as e:
        MONITOR.record_error(SOURCE, type(e).__name__)
        raise BrokerError("I couldn't reach Kite right now.") from e
    status = getattr(resp, "status_code", None)
    if MONITOR.note_http(SOURCE, status, getattr(resp, "headers", None)):
        raise BrokerError("Kite is rate-limiting requests; I'll back off for a bit.")
    if status in (401, 403):
        MONITOR.record_error(SOURCE, f"HTTP {status}")
        raise BrokerError("Kite rejected the login. Its access token expires every day; "
                          "log in again and update KITE_ACCESS_TOKEN.")
    try:
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        MONITOR.record_error(SOURCE, type(e).__name__)
        raise BrokerError("Kite sent a reply I couldn't read.") from e
    MONITOR.record_success(SOURCE)
    return parse_kite_holdings(payload)


# ---- CSV ---------------------------------------------------------------------

_SYMBOL_HEADERS = ("symbol", "tradingsymbol", "trading symbol", "instrument", "stock name", "company name", "name", "scrip")
_QTY_HEADERS = ("quantity available", "quantity", "qty", "net qty", "shares", "units")
_PRICE_HEADERS = ("average buy price", "avg. cost", "avg cost", "average price", "avg price",
                  "average cost", "buy avg", "avg. price")


def _pick(headers: dict, wanted: tuple) -> Optional[str]:
    for w in wanted:
        if w in headers:
            return headers[w]
    return None


def parse_holdings_csv(path_text: str) -> list[BrokerHolding]:
    path = Path(str(path_text).strip().strip("'\"")).expanduser()
    if path.suffix.lower() != ".csv":
        raise BrokerError("I can read .csv holdings files only. Export the holdings as CSV first.")
    try:
        if not path.is_file() or path.stat().st_size > MAX_CSV_BYTES:
            raise BrokerError("I can't find that file, or it is too large.")
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except BrokerError:
        raise
    except OSError:
        raise BrokerError("I couldn't open that file.")
    lines = text.splitlines()
    # Exports often have a few title lines first; find the header row.
    start = next((i for i, ln in enumerate(lines[:30])
                  if any(h in ln.lower() for h in _QTY_HEADERS) and any(h in ln.lower() for h in _SYMBOL_HEADERS)), None)
    if start is None:
        raise BrokerError("I couldn't find symbol, quantity and average price columns in that file.")
    reader = csv.DictReader(lines[start:])
    headers = {(h or "").strip().lower().rstrip("."): h for h in (reader.fieldnames or [])}
    sym_c, qty_c, px_c = _pick(headers, _SYMBOL_HEADERS), _pick(headers, _QTY_HEADERS), _pick(headers, _PRICE_HEADERS)
    if not (sym_c and qty_c and px_c):
        raise BrokerError("I couldn't find symbol, quantity and average price columns in that file.")
    out = []
    for row in reader:
        if len(out) >= MAX_ROWS:
            break
        symbol = re.sub(r"[^A-Z0-9&\-_]", "", str(row.get(sym_c) or "").upper())
        qty, avg = _num(row.get(qty_c)), _num(row.get(px_c))
        if symbol and qty and 0 < qty <= MAX_QTY and avg is not None and avg >= 0:
            out.append(BrokerHolding(symbol, "NS", qty, avg))
    if not out:
        raise BrokerError("That file had no usable holdings rows.")
    return out


# ---- the sync ------------------------------------------------------------------

def _match_key(name: str) -> str:
    return re.sub(r"(\.NS|\.BO)$", "", re.sub(r"\s+", "", str(name).upper()))


class BrokerSync:
    def __init__(self, db: Any, symbol: str = "\u20b9",
                 env: Optional[dict] = None, kite_fn: Callable = fetch_kite_holdings):
        self.db = db
        self.symbol = symbol
        self._env = env if env is not None else os.environ
        self._kite = kite_fn

    def sync(self, entities: dict) -> dict:
        confirm = str(entities.get("confirm") or "").strip().lower() in _TRUE
        csv_path = entities.get("csv_path") or entities.get("file_path") or entities.get("path")
        broker = str(entities.get("broker") or "zerodha").strip().lower()
        try:
            if csv_path:
                holdings, source = parse_holdings_csv(str(csv_path)), "your CSV"
            elif broker in ("zerodha", "kite"):
                key, token = self._env.get("KITE_API_KEY"), self._env.get("KITE_ACCESS_TOKEN")
                if not key or not token:
                    return _reply("Zerodha sync needs KITE_API_KEY and KITE_ACCESS_TOKEN set in the environment "
                                  "(the token is renewed daily). Or give me a holdings CSV path instead.", 0.5)
                holdings, source = self._kite(key, token), "Zerodha"
            else:
                return _reply(f"I can sync Zerodha, or a holdings CSV from any broker (including Groww). "
                              f"I don't have a connection for {broker!r}.", 0.5)
        except BrokerError as e:
            return _reply(str(e), 0.3)
        if not holdings:
            return _reply(f"{source} shows no holdings.", 0.7)

        tracked = {k: h for k, h in
                   ((_match_key(h["name"]), h) for h in aggregate_lots(self.db.get_investments()).values())}
        add, same, extra = [], 0, []
        for h in holdings:
            t = tracked.get(_match_key(h.symbol))
            have, cost = (t["quantity"], t["cost"]) if t else (0.0, 0.0)
            delta = h.quantity - have
            if abs(delta) < 1e-9 or abs(delta) / max(h.quantity, 1.0) < 1e-9:
                same += 1
            elif delta > 0:
                price = (h.quantity * h.avg_price - cost) / delta if have else h.avg_price
                if not price > 0:
                    price = h.avg_price
                name = t["name"] if t else f"{h.symbol}.{h.exchange}"
                add.append((name, delta, price))
            else:
                extra.append(f"{t['name']} (you track {have:g}, broker shows {h.quantity:g})")
        lines = [f"Compared {len(holdings)} holding(s) from {source} with what Pluto tracks."]
        if add:
            lines.append(("Added " if confirm else "Would add ") + f"{len(add)} lot(s):")
            for name, q, p in add[:15]:
                lines.append(f"  {name}: {q:g} @ {self.symbol}{p:,.2f}")
            if len(add) > 15:
                lines.append(f"  ...and {len(add) - 15} more.")
        if same:
            lines.append(f"{same} already match.")
        if extra:
            lines.append("Tracked here but fewer at the broker (nothing removed): " + "; ".join(extra[:8]) + ".")
        if confirm:
            try:
                with self.db.transaction() as cur:
                    for name, q, p in add:
                        cur.execute("INSERT INTO investments (name, type, quantity, buy_price) VALUES (?, ?, ?, ?)",
                                    (name, "stock", q, p))
            except Exception:
                logger.exception("broker sync write failed")
                return _reply("I couldn't save the synced holdings; nothing was changed.", 0.3)
        elif add:
            lines.append("Say 'sync my zerodha, confirm' to save these. Prices for new lots are the broker's averages.")
        if not add and not extra:
            lines.append("Nothing to change.")
        return _reply("\n".join(lines), 0.85, {"source": source, "applied": bool(confirm and add),
                                               "added": [{"name": n, "quantity": q, "buy_price": p} for n, q, p in add],
                                               "matched": same, "broker_has_fewer": extra})


def _reply(text: str, confidence: float, data: Optional[dict] = None) -> dict:
    return {"response": text, "data": data or {}, "confidence": confidence}
