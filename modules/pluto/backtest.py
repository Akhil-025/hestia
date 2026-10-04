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

Parameter sweeps and equity curves (backlog #143)
-------------------------------------------------
``sweep_sma_crossover`` runs a grid of (fast, slow) windows over one price
download and returns every combination's stats plus equity curves for the web
UI. It uses ``simulate_sma_crossover``, a small pure-Python simulator that
follows the same rules as the vectorbt run in ``backtest_sma_crossover``
(enter on the bar where the fast average crosses above the slow one, exit on
the cross below, trade at that bar's close, all-in, 0.1% fee per side, long
only). It exists because a grid of dozens of runs, and an equity curve per
run, is simpler and faster without building a vectorbt Portfolio each time,
and because it runs wherever vectorbt does not (vectorbt does not install on
Python 3.12). Its numbers have not been cross-checked against a live vectorbt
run; expect small differences (annualisation, how open trades are counted).

Picking the best cell of a grid on the same prices it was tested on is
in-sample fitting. The sweep reports how many cells beat buy-and-hold and the
median across the grid, so one lucky cell is visible as one lucky cell.

Known packaging caveat (documented so it isn't rediscovered blind):
vectorbt 1.1.0 imports `plotly.graph_objects.Scattermapbox`, which
newer Plotly (6.x+) renamed to `Scattermap` — importing vectorbt
against Plotly 6+ raises at import time. requirements.txt pins
`plotly<6` for this reason. If that pin is ever removed, vectorbt
must be re-verified or dropped.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional, Sequence

from .market_data import MarketDataError, MarketDataThrottled, fetch_price_history, throttle_hint
from .logging_config import get_logger

logger = get_logger(__name__)

MIN_PRICE_POINTS = 60  # need enough bars for the slower of the two windows to warm up

FEE_RATE = 0.001            # 10 bps per side, same as the vectorbt run
TRADING_DAYS = 252
MAX_SWEEP_VALUES = 8        # per axis
MAX_SWEEP_CELLS = 64
MAX_WINDOW = 400
CURVE_POINTS = 250          # equity curves are thinned to about this many points


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
            return _err(f"I couldn't fetch price history for {ticker!r}: {e}{throttle_hint(e)}")

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

    def sweep_sma_crossover(self, ticker: str, fast_windows, slow_windows,
                            initial_cash: float = 100_000.0, range_: str = "1y") -> dict:
        """Grid of (fast, slow) SMA windows over one download, with equity curves.
        See the module docstring and ``_sweep_sma_crossover``. Never raises."""
        try:
            return _sweep_sma_crossover(ticker, fast_windows, slow_windows, initial_cash, range_)
        except Exception:
            logger.exception("sweep_sma_crossover failed for %r", ticker)
            return _err("Something went wrong while running the sweep.")

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


# ---------------------------------------------------------------------------
# Simulator and parameter sweep (#143)
# ---------------------------------------------------------------------------

def _sma(closes: Sequence[float], window: int) -> list[Optional[float]]:
    """Simple moving average; None until ``window`` closes exist."""
    out: list[Optional[float]] = [None] * len(closes)
    running = 0.0
    for i, c in enumerate(closes):
        running += c
        if i >= window:
            running -= closes[i - window]
        if i >= window - 1:
            out[i] = running / window
    return out


def simulate_sma_crossover(closes: Sequence[float], fast: int, slow: int,
                           initial_cash: float = 100_000.0, fee: float = FEE_RATE) -> dict:
    """Long-only SMA crossover over ``closes``. Returns stats and the equity curve.

    Rules (match backtest_sma_crossover): buy everything on the bar where the
    fast SMA crosses above the slow SMA, sell everything on the cross below,
    both at that bar's close, paying ``fee`` per side. ``equity[i]`` is cash
    plus holdings marked at ``closes[i]``; before both averages exist it is
    just the starting cash.
    """
    n = len(closes)
    fast_ma, slow_ma = _sma(closes, fast), _sma(closes, slow)
    cash, shares = float(initial_cash), 0.0
    equity: list[float] = []
    trades = 0
    for i in range(n):
        if i > 0 and fast_ma[i] is not None and slow_ma[i] is not None \
                and fast_ma[i - 1] is not None and slow_ma[i - 1] is not None:
            crossed_up = fast_ma[i - 1] <= slow_ma[i - 1] and fast_ma[i] > slow_ma[i]
            crossed_down = fast_ma[i - 1] >= slow_ma[i - 1] and fast_ma[i] < slow_ma[i]
            if crossed_up and shares == 0.0 and cash > 0.0:
                shares = cash / (closes[i] * (1.0 + fee))
                cash = 0.0
                trades += 1
            elif crossed_down and shares > 0.0:
                cash = shares * closes[i] * (1.0 - fee)
                shares = 0.0
        equity.append(cash + shares * closes[i])

    returns = [equity[i] / equity[i - 1] - 1.0 for i in range(1, n) if equity[i - 1] > 0]
    sharpe: Optional[float] = None
    if len(returns) > 1:
        sd = statistics.pstdev(returns)
        if sd > 1e-12:
            sharpe = statistics.fmean(returns) / sd * math.sqrt(TRADING_DAYS)
    peak, max_dd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)
    return {
        "fast_window": fast,
        "slow_window": slow,
        "total_return_pct": (equity[-1] / initial_cash - 1.0) * 100,
        "max_drawdown_pct": max_dd * 100,
        "sharpe_ratio": sharpe,
        "num_trades": trades,
        "in_market_at_end": shares > 0.0,
        "equity": equity,
    }


def _thin(values: Sequence, max_points: int = CURVE_POINTS) -> list:
    """Evenly thinned copy, always keeping the first and last element."""
    n = len(values)
    if n <= max_points:
        return list(values)
    step = (n - 1) / (max_points - 1)
    idx = sorted({int(round(i * step)) for i in range(max_points)} | {0, n - 1})
    return [values[i] for i in idx]


def _thin_index(n: int, max_points: int = CURVE_POINTS) -> list[int]:
    if n <= max_points:
        return list(range(n))
    step = (n - 1) / (max_points - 1)
    return sorted({int(round(i * step)) for i in range(max_points)} | {0, n - 1})


def parse_window_list(raw, name: str) -> list[int]:
    """'5, 10 20' / [5, 10] -> [5, 10, 20]; raises ValueError with a plain message."""
    if isinstance(raw, str):
        parts = [p for p in raw.replace(",", " ").split() if p]
    else:
        parts = list(raw or [])
    out: list[int] = []
    for p in parts:
        try:
            f = float(p)
        except (TypeError, ValueError):
            raise ValueError(f"{name} windows must be whole numbers; {p!r} is not.")
        if not math.isfinite(f) or f != int(f) or not 1 <= int(f) <= MAX_WINDOW:
            raise ValueError(f"{name} windows must be whole numbers from 1 to {MAX_WINDOW}; got {p!r}.")
        if int(f) not in out:
            out.append(int(f))
    if not out:
        raise ValueError(f"Give at least one {name} window.")
    if len(out) > MAX_SWEEP_VALUES:
        raise ValueError(f"At most {MAX_SWEEP_VALUES} {name} windows per sweep; got {len(out)}.")
    return sorted(out)


def _sweep_sma_crossover(ticker: str, fast_windows, slow_windows,
                         initial_cash: float = 100_000.0, range_: str = "1y") -> dict:
    """Run every (fast, slow) pair with fast < slow over one price download.

    ``data`` holds ``grid`` (one dict per cell, without curves), ``best_by_return``
    and ``best_by_sharpe`` (the cell dicts), ``curves`` (dates, buy-and-hold, and
    the equity curve of the best-by-return and best-by-Sharpe cells) and summary
    counts. Never raises.
    """
    try:
        fasts = parse_window_list(fast_windows, "fast")
        slows = parse_window_list(slow_windows, "slow")
    except ValueError as e:
        return _err(str(e))
    pairs = [(f, s) for f in fasts for s in slows if f < s]
    if not pairs:
        return _err("None of those pairs has the fast window shorter than the slow one, so there is nothing to test.")
    if len(pairs) > MAX_SWEEP_CELLS:
        return _err(f"That is {len(pairs)} combinations; the limit is {MAX_SWEEP_CELLS}. Use fewer windows.")

    try:
        series = fetch_price_history(ticker, range_=range_)
    except MarketDataError as e:
        logger.warning("sweep_sma_crossover: price fetch failed for %r: %s", ticker, e)
        return _err(f"I couldn't fetch price history for {ticker!r}: {e}{throttle_hint(e)}")

    closes = series.closes
    need = max(MIN_PRICE_POINTS, max(s for _, s in pairs) + 5)
    usable = [(f, s) for f, s in pairs if len(closes) >= max(MIN_PRICE_POINTS, s + 5)]
    dropped = [(f, s) for f, s in pairs if (f, s) not in usable]
    if not usable:
        return _err(
            f"Not enough price history for {series.ticker} (got {len(closes)} bars, the slowest "
            f"window needs {need}). Try a longer range or shorter windows."
        )

    runs = {pair: simulate_sma_crossover(closes, pair[0], pair[1], initial_cash) for pair in usable}
    buy_hold_pct = (closes[-1] / closes[0] - 1.0) * 100
    buy_hold_curve = [initial_cash * c / closes[0] for c in closes]

    def cell(run: dict) -> dict:
        return {k: v for k, v in run.items() if k != "equity"}

    grid = [cell(runs[p]) for p in usable]
    by_return = max(usable, key=lambda p: runs[p]["total_return_pct"])
    sharpe_ok = [p for p in usable if runs[p]["sharpe_ratio"] is not None]
    by_sharpe = max(sharpe_ok, key=lambda p: runs[p]["sharpe_ratio"]) if sharpe_ok else None
    returns = [runs[p]["total_return_pct"] for p in usable]
    beat = sum(1 for r in returns if r > buy_hold_pct)

    idx = _thin_index(len(closes))
    curves = {
        "dates": [series.dates[i] for i in idx],
        "buy_and_hold": [round(buy_hold_curve[i], 2) for i in idx],
        "best_by_return": {"fast": by_return[0], "slow": by_return[1],
                           "equity": [round(runs[by_return]["equity"][i], 2) for i in idx]},
        "best_by_sharpe": ({"fast": by_sharpe[0], "slow": by_sharpe[1],
                            "equity": [round(runs[by_sharpe]["equity"][i], 2) for i in idx]}
                           if by_sharpe else None),
    }

    lines = [
        f"SMA crossover sweep, {series.ticker}, {series.dates[0]} to {series.dates[-1]} "
        f"({len(usable)} combinations):",
        f"  Buy & hold            : {buy_hold_pct:+.1f}%",
        f"  Best total return     : SMA({by_return[0]}/{by_return[1]}) {runs[by_return]['total_return_pct']:+.1f}% "
        f"(max drawdown {runs[by_return]['max_drawdown_pct']:.1f}%, {runs[by_return]['num_trades']} trade(s))",
    ]
    if by_sharpe:
        lines.append(
            f"  Best Sharpe ratio     : SMA({by_sharpe[0]}/{by_sharpe[1]}) {runs[by_sharpe]['sharpe_ratio']:.2f} "
            f"(return {runs[by_sharpe]['total_return_pct']:+.1f}%)"
        )
    lines.append(f"  Median across the grid: {statistics.median(returns):+.1f}%; "
                 f"{beat} of {len(usable)} combinations beat buy & hold")
    if dropped:
        lines.append("  Skipped (not enough history): " + ", ".join(f"{f}/{s}" for f, s in dropped))
    lines.append(
        "The best cell is the best on these same prices, so it is flattering by construction. "
        "Look at the median and the neighbours of a good cell before trusting it. "
        "Past performance doesn't predict future results; this is not a trading recommendation."
    )
    data = {
        "ticker": series.ticker,
        "range": range_,
        "bars": len(closes),
        "buy_and_hold_pct": buy_hold_pct,
        "grid": grid,
        "best_by_return": cell(runs[by_return]),
        "best_by_sharpe": cell(runs[by_sharpe]) if by_sharpe else None,
        "median_return_pct": statistics.median(returns),
        "beat_buy_and_hold": beat,
        "cells": len(usable),
        "skipped": [{"fast": f, "slow": s} for f, s in dropped],
        "curves": curves,
    }
    return _ok("\n".join(lines), data=data, confidence=0.75)

