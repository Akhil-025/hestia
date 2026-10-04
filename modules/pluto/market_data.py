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

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from .retry import retry
from .logging_config import get_logger
from .throttle import MONITOR

logger = get_logger(__name__)

# Name this fetcher reports under in the throttle monitor (backlog #144).
SOURCE = "yahoo_chart"

# Retry tuning, visible and changeable in one place. Read at call time, so
# configure_retry() takes effect on the next request. Note max_retries counts
# attempts in total, as in retry.py (2 means one retry).
RETRY_SETTINGS: dict = {"max_retries": 2, "delay": 0.5, "backoff": 2.0, "jitter": 0.5}


def retry_settings() -> dict:
    """A copy of the retry tuning currently in force."""
    return dict(RETRY_SETTINGS)


def configure_retry(**changes) -> dict:
    """Change retry tuning (max_retries, delay, backoff, jitter); returns the new settings.

    Unknown names and out-of-range values raise ValueError instead of being
    ignored, so a typo can't leave the old setting in place unnoticed.
    """
    for key, value in changes.items():
        if key not in RETRY_SETTINGS:
            raise ValueError(f"unknown retry setting {key!r}; known: {sorted(RETRY_SETTINGS)}")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"retry setting {key!r} must be a non-negative number, got {value!r}")
        if key == "max_retries" and (int(value) != value or value < 1):
            raise ValueError("max_retries must be a whole number of at least 1")
    RETRY_SETTINGS.update({k: (int(v) if k == "max_retries" else float(v)) for k, v in changes.items()})
    return retry_settings()

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


class MarketDataThrottled(MarketDataError):
    """The data source answered 429 (or is still inside a Retry-After window).

    A subclass of MarketDataError, so existing ``except MarketDataError``
    handlers keep working; callers that want to say "rate-limited, try again
    in N seconds" can catch this one first. ``retry_after`` is in seconds.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


def throttle_hint(exc: Exception) -> str:
    """A short ' (rate-limited, try again in about Ns)' suffix, or '' for other errors."""
    if isinstance(exc, MarketDataThrottled):
        if exc.retry_after:
            return f" (Yahoo Finance is rate-limiting requests; try again in about {math.ceil(exc.retry_after)}s)"
        return " (Yahoo Finance is rate-limiting requests; try again shortly)"
    return ""


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


def fetch_price_history(ticker: str, range_: str = "1y", interval: str = "1d") -> PriceSeries:
    """
    Fetch a daily close-price history for `ticker` from Yahoo Finance.
    Raises MarketDataError (never a bare requests exception) on any failure
    so callers can catch a single, documented exception type. A 429 raises
    its subclass MarketDataThrottled, is not retried, and starts a cooldown
    (the server's Retry-After, else a default) during which further calls
    fail fast instead of hitting the API again. Every outcome is recorded in
    modules/pluto/throttle.py's MONITOR (backlog #144).
    """
    remaining = MONITOR.cooldown_remaining(SOURCE)
    if remaining > 0:
        raise MarketDataThrottled(
            f"Yahoo Finance asked us to slow down; {math.ceil(remaining)}s of cooldown left.",
            retry_after=remaining,
        )
    attempt = retry(
        exceptions=(MarketDataError,),
        give_up_on=(MarketDataThrottled,),
        on_retry=lambda n, exc, wait: MONITOR.record_retry(SOURCE),
        **RETRY_SETTINGS,
    )(_fetch_price_history_once)
    return attempt(ticker, range_, interval)


def _fetch_price_history_once(ticker: str, range_: str, interval: str) -> PriceSeries:
    yahoo_range = _RANGE_TO_YAHOO.get(range_, "1y")
    symbol = normalise_ticker(ticker)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval={interval}&range={yahoo_range}"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
    except Exception as e:
        MONITOR.record_error(SOURCE, str(e))
        raise MarketDataError(f"Yahoo Finance request failed for {symbol!r}: {e}") from e

    if MONITOR.note_http(SOURCE, getattr(response, "status_code", None), getattr(response, "headers", None)):
        wait = MONITOR.cooldown_remaining(SOURCE)
        raise MarketDataThrottled(f"Yahoo Finance rate-limited the request for {symbol!r} (HTTP 429).", wait)

    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        # A 429 can also surface as an HTTPError carrying its response.
        err_resp = getattr(e, "response", None)
        if MONITOR.note_http(SOURCE, getattr(err_resp, "status_code", None), getattr(err_resp, "headers", None)):
            wait = MONITOR.cooldown_remaining(SOURCE)
            raise MarketDataThrottled(
                f"Yahoo Finance rate-limited the request for {symbol!r} (HTTP 429).", wait) from e
        MONITOR.record_error(SOURCE, str(e))
        raise MarketDataError(f"Yahoo Finance request failed for {symbol!r}: {e}") from e
    MONITOR.record_success(SOURCE)

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
