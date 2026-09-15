"""

modules/pluto/market_data.py
Shared historical price-series fetching for Pluto's quant features
(portfolio.py, backtest.py). Kept separate from personal_finance.py's
_fetch_yahoo_price (which only needs the latest tick) because
optimize_portfolio and backtest_strategy both need a full OHLC time
series and would otherwise duplicate this exact HTTP/parsing logic.

Uses the same unauthenticated Yahoo Finance "chart" endpoint the rest
of Pluto already relies on — no API key, but best-effort and can be
rate-limited or blocked without notice. Every caller must treat a
MarketDataError as a normal, expected failure mode (missing ticker,
delisted symbol, Yahoo hiccup) and degrade gracefully rather than
propagate a stack trace to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from .retry import retry
from .logging_config import get_logger

logger = get_logger(__name__)

_RANGE_TO_YAHOO = {
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
}


class MarketDataError(Exception):
    """Raised when a historical price series can't be retrieved or is unusable."""


@dataclass(frozen=True)
class PriceSeries:
    ticker: str
    dates: list[str]     # ISO date strings, ascending
    closes: list[float]  # aligned 1:1 with `dates`

    @property
    def available(self) -> bool:
        return len(self.closes) >= 2

    def daily_returns(self) -> list[float]:
        """Simple day-over-day percentage returns, one shorter than closes."""
        return [
            (self.closes[i] / self.closes[i - 1]) - 1.0
            for i in range(1, len(self.closes))
            if self.closes[i - 1]
        ]


def normalise_ticker(name: str) -> str:
    """
    Mirror PersonalFinanceManager._fetch_yahoo_price's ticker normalisation
    so a name typed the same way ("HDFC Bank") resolves to the same symbol
    across log_expense-style tracking and the new quant features.
    """
    ticker = name.upper()
    ticker = re.sub(r"\b(INDUSTRIES|LIMITED|LTD|INC|CORP)\b", "", ticker, flags=re.I)
    ticker = re.sub(r"\s+", "", ticker)
    ticker = ticker.strip(".-_")
    if not any(ch in ticker for ch in (".", "^")):
        ticker = ticker + ".NS"
    return ticker


@retry(max_retries=2, exceptions=(MarketDataError,), delay=0.5)
def fetch_price_history(ticker: str, range_: str = "1y", interval: str = "1d") -> PriceSeries:
    """
    Fetch a daily close-price history for `ticker` from Yahoo Finance.
    Raises MarketDataError (never a bare requests exception) on any failure
    so callers can catch a single, documented exception type.
    """
    yahoo_range = _RANGE_TO_YAHOO.get(range_, "1y")
    symbol = normalise_ticker(ticker)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval={interval}&range={yahoo_range}"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        raise MarketDataError(f"Yahoo Finance request failed for {symbol!r}: {e}") from e

    result = (payload.get("chart", {}) or {}).get("result") or []
    if not result:
        raise MarketDataError(f"No chart data returned for {symbol!r}.")

    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quote_blocks = (chart.get("indicators", {}) or {}).get("quote") or []
    closes_raw = (quote_blocks[0].get("close") if quote_blocks else None) or []

    if not timestamps or not closes_raw:
        raise MarketDataError(f"Empty price series for {symbol!r}.")

    dates: list[str] = []
    closes: list[float] = []
    for ts, close in zip(timestamps, closes_raw):
        # Yahoo pads non-trading days / gaps with null closes — drop them
        # rather than feeding NaN-shaped holes into downstream optimizers.
        if close is None:
            continue
        dates.append(datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"))
        closes.append(float(close))

    if len(closes) < 2:
        raise MarketDataError(
            f"Not enough usable price points for {symbol!r} ({len(closes)} found)."
        )

    return PriceSeries(ticker=symbol, dates=dates, closes=closes)
