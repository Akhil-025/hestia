"""

modules/pluto/backtest.py
Real strategy backtesting via vectorbt, replacing the previously-unused
vectorbt entry in requirements.txt.

Deliberately scoped to a single, well-understood strategy — an SMA
(simple moving average) crossover — rather than a general strategy
DSL. Hestia has no code-execution surface exposed to a voice/chat
assistant, so "backtest this arbitrary strategy" isn't a safe or
sane feature to expose; a fixed, parameterised strategy with two
window-length inputs is.

Known packaging caveat (documented so it isn't rediscovered blind):
vectorbt 1.1.0 imports `plotly.graph_objects.Scattermapbox`, which
newer Plotly (6.x+) renamed to `Scattermap` — importing vectorbt
against Plotly 6+ raises at import time. requirements.txt pins
`plotly<6` for this reason. If that pin is ever removed, vectorbt
must be re-verified or dropped.
"""

from __future__ import annotations

from typing import Optional

from .market_data import MarketDataError, fetch_price_history
from .logging_config import get_logger

logger = get_logger(__name__)

MIN_PRICE_POINTS = 60  # need enough bars for the slower of the two windows to warm up


def _ok(response: str, data: Optional[dict] = None, confidence: float = 0.9) -> dict:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict:
    return {"response": response, "data": {}, "confidence": 0.0}


class StrategyBacktester:
    """SMA-crossover backtest over a single ticker's trailing price history."""

    def backtest_sma_crossover(
        self,
        ticker: str,
        fast_window: int = 10,
        slow_window: int = 50,
        initial_cash: float = 100_000.0,
        range_: str = "1y",
    ) -> dict:
        if fast_window <= 0 or slow_window <= 0:
            return _err("Window lengths must be positive numbers of trading days.")
        if fast_window >= slow_window:
            return _err(
                f"The fast window ({fast_window}) must be shorter than the slow "
                f"window ({slow_window}) for a crossover strategy to mean anything."
            )

        try:
            series = fetch_price_history(ticker, range_=range_)
        except MarketDataError as e:
            logger.warning("backtest_strategy: price fetch failed for %r: %s", ticker, e)
            return _err(f"I couldn't fetch price history for {ticker!r}: {e}")

        if len(series.closes) < max(MIN_PRICE_POINTS, slow_window + 5):
            return _err(
                f"Not enough price history for {series.ticker} to backtest a "
                f"{slow_window}-day slow window (got {len(series.closes)} bars, "
                f"need at least {max(MIN_PRICE_POINTS, slow_window + 5)}). "
                "Try a longer `range_` or shorter windows."
            )

        try:
            stats = self._run_vectorbt(series, fast_window, slow_window, initial_cash)
        except Exception as e:
            logger.exception("backtest_strategy: vectorbt run failed for %r", ticker)
            return _err(
                "The backtest engine failed to run — this can happen if vectorbt's "
                "Plotly dependency isn't pinned correctly (see backtest.py's module "
                "docstring). Try again or report this."
            )

        lines = [
            f"SMA({fast_window}/{slow_window}) crossover backtest — {series.ticker}, "
            f"{series.dates[0]} to {series.dates[-1]}:",
            f"  Total return   : {stats['total_return_pct']:+.1f}%",
            f"  Buy & hold     : {stats['buy_and_hold_pct']:+.1f}%",
            f"  Max drawdown   : {stats['max_drawdown_pct']:.1f}%",
            f"  Sharpe ratio   : {stats['sharpe_ratio']:.2f}"
            if stats["sharpe_ratio"] is not None
            else "  Sharpe ratio   : n/a (not enough variance in returns)",
            f"  Number of trades: {stats['num_trades']}",
        ]
        lines.append(
            "Backtests on past prices don't predict future performance — this is "
            "informational, not a trading recommendation."
        )

        return _ok("\n".join(lines), data=stats, confidence=0.8)

    @staticmethod
    def _run_vectorbt(series, fast_window: int, slow_window: int, initial_cash: float) -> dict:
        import numpy as np
        import pandas as pd
        import vectorbt as vbt

        index = pd.to_datetime(series.dates)
        close = pd.Series(series.closes, index=index)

        fast_ma = vbt.MA.run(close, fast_window)
        slow_ma = vbt.MA.run(close, slow_window)

        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)

        portfolio = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            init_cash=initial_cash,
            fees=0.001,  # 10 bps per trade, a reasonable retail-brokerage stand-in
        )

        total_return_pct = float(portfolio.total_return()) * 100
        buy_and_hold_pct = float((close.iloc[-1] / close.iloc[0] - 1.0)) * 100
        max_dd_pct = float(portfolio.max_drawdown()) * 100

        try:
            sharpe = float(portfolio.sharpe_ratio())
            if not np.isfinite(sharpe):
                sharpe = None
        except Exception:
            sharpe = None

        num_trades = int(portfolio.trades.count()) if hasattr(portfolio, "trades") else 0

        return {
            "ticker": series.ticker,
            "fast_window": fast_window,
            "slow_window": slow_window,
            "total_return_pct": total_return_pct,
            "buy_and_hold_pct": buy_and_hold_pct,
            "max_drawdown_pct": max_dd_pct,
            "sharpe_ratio": sharpe,
            "num_trades": num_trades,
        }
