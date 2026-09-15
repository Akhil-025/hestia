"""

modules/pluto/portfolio.py
Real mean-variance portfolio optimisation over the user's own tracked
investments (modules/pluto/db.py's `investments` table), using
PyPortfolioOpt. Replaces the previous state where PyPortfolioOpt sat in
requirements.txt entirely unused.

Design notes:
- Only operates on the user's *own* holdings (PlutoDB.get_investments()),
  grouped by ticker — this is "how should I rebalance what I already
  hold", not a general stock screener.
- Needs >= 2 distinct tickers with fetchable history to produce a
  meaningful efficient frontier; degrades to a clear, non-crashing
  message otherwise (a single holding has nothing to diversify against).
- Never raises out of `optimize()` — every failure path returns a
  response dict, matching the rest of Pluto's manager classes.
"""

from __future__ import annotations

from typing import Optional

from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.exceptions import OptimizationError

from .db import PlutoDB
from .market_data import MarketDataError, fetch_price_history
from .logging_config import get_logger

logger = get_logger(__name__)

MIN_TICKERS_FOR_OPTIMIZATION = 2
MIN_PRICE_POINTS = 30  # ~6 weeks of trading days; anything less is too noisy


def _ok(response: str, data: Optional[dict] = None, confidence: float = 0.9) -> dict:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict:
    return {"response": response, "data": {}, "confidence": 0.0}


class PortfolioOptimizer:
    """Wraps PyPortfolioOpt's EfficientFrontier over the user's held tickers."""

    def __init__(self, db: PlutoDB, currency: str = "\u20b9"):
        self.db = db
        self.currency = currency

    def _distinct_holdings(self) -> dict[str, float]:
        """
        Return {name: total_quantity} for non-crypto, non-zero-quantity
        holdings. Crypto is excluded: Yahoo's chart endpoint used by
        fetch_price_history is equities/index-focused, and mixing a
        24/7 crypto return series with equity trading-day series would
        silently misalign the covariance matrix.
        """
        totals: dict[str, float] = {}
        for inv in self.db.get_investments():
            if inv.get("type") == "crypto":
                continue
            qty = inv.get("quantity") or 0.0
            if qty <= 0:
                continue
            name = inv["name"]
            totals[name] = totals.get(name, 0.0) + qty
        return totals

    def optimize(self, risk_free_rate: float = 0.03) -> dict:
        holdings = self._distinct_holdings()

        if len(holdings) < MIN_TICKERS_FOR_OPTIMIZATION:
            return _err(
                "I need at least two different stock/mutual-fund holdings with a "
                "quantity greater than zero to run a portfolio optimization — "
                f"you currently have {len(holdings)}. Track another holding with "
                "'track_investment' first."
            )

        price_by_ticker: dict[str, list[float]] = {}
        skipped: list[str] = []
        for name in holdings:
            try:
                series = fetch_price_history(name, range_="1y")
            except MarketDataError as e:
                logger.warning("optimize_portfolio: skipping %r: %s", name, e)
                skipped.append(name)
                continue
            if len(series.closes) < MIN_PRICE_POINTS:
                skipped.append(name)
                continue
            price_by_ticker[series.ticker] = series.closes

        if len(price_by_ticker) < MIN_TICKERS_FOR_OPTIMIZATION:
            return _err(
                "I couldn't fetch enough live price history to optimize your "
                f"portfolio (usable: {len(price_by_ticker)}, needed: "
                f"{MIN_TICKERS_FOR_OPTIMIZATION}). Skipped: "
                f"{', '.join(skipped) if skipped else 'none'}."
            )

        try:
            prices_df = self._align_prices(price_by_ticker)
            mu = expected_returns.mean_historical_return(prices_df, frequency=252)
            cov = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()

            ef = EfficientFrontier(mu, cov)
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            cleaned = ef.clean_weights()
            perf_return, perf_vol, perf_sharpe = ef.portfolio_performance(
                risk_free_rate=risk_free_rate
            )
        except OptimizationError as e:
            logger.warning("optimize_portfolio: solver failed: %s", e)
            return _err(
                "The optimizer couldn't find a solution for your current holdings "
                "— this usually means the price histories are too short or too "
                "highly correlated to distinguish. Try again once you have more "
                "price history."
            )
        except Exception as e:
            logger.exception("optimize_portfolio: unexpected failure")
            return _err("Something went wrong while optimizing your portfolio.")

        lines = ["Suggested allocation (max Sharpe ratio):"]
        for ticker, weight in sorted(cleaned.items(), key=lambda kv: -kv[1]):
            if weight <= 0.0001:
                continue
            lines.append(f"  {ticker:12} {weight * 100:5.1f}%")
        lines.append("")
        lines.append(
            f"Expected annual return: {perf_return * 100:.1f}%  |  "
            f"Annual volatility: {perf_vol * 100:.1f}%  |  Sharpe: {perf_sharpe:.2f}"
        )
        if skipped:
            lines.append(f"(Skipped, insufficient data: {', '.join(skipped)})")
        lines.append(
            "This is a mean-variance estimate from trailing 1-year prices, not "
            "financial advice — treat it as one input among several."
        )

        return _ok(
            "\n".join(lines),
            data={
                "weights": cleaned,
                "expected_annual_return": perf_return,
                "annual_volatility": perf_vol,
                "sharpe_ratio": perf_sharpe,
                "skipped": skipped,
            },
            confidence=0.85,
        )

    @staticmethod
    def _align_prices(price_by_ticker: dict[str, list[float]]):
        """
        Build a pandas DataFrame of aligned closing prices, truncated to the
        shortest series (PyPortfolioOpt needs a rectangular price matrix;
        tickers with different listing/history lengths won't naturally align).
        """
        import pandas as pd

        min_len = min(len(v) for v in price_by_ticker.values())
        trimmed = {t: v[-min_len:] for t, v in price_by_ticker.items()}
        return pd.DataFrame(trimmed)
