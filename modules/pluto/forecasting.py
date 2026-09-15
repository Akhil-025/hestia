"""

modules/pluto/forecasting.py
Spending forecasting via LightGBM, trained on the user's own logged
expenses (modules/pluto/db.py's `expenses` table). Replaces the previously
-unused lightgbm entry in requirements.txt.

Deliberately local-data-only: unlike market_intelligence.py's stock
"quant_score" (which depends on a Postgres/TimescaleDB market-data
pipeline that isn't wired up in the laptop_config.yaml single-user
deployment), this trains a small model on the user's own SQLite expense
history every time it runs. That fits Hestia's "local-first" design
far better than a pretrained/registry model would, and it's honest
about what it is: a short-horizon trend extrapolation, not a market
predictor.

The model is deliberately tiny (LightGBM with a handful of trees, no
external labels) — this is meant to answer "at this rate, what will I
spend next week", not to be a rigorous forecasting product.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional

from .db import PlutoDB
from .logging_config import get_logger

logger = get_logger(__name__)

MIN_DAYS_OF_HISTORY = 14       # need at least two weeks of daily totals to fit anything
MIN_DISTINCT_DAYS = 10         # ...spread across at least this many distinct days
LOOKBACK_WINDOW = 7            # rolling-window features (days)
FORECAST_HORIZON_DAYS = 7


def _ok(response: str, data: Optional[dict] = None, confidence: float = 0.85) -> dict:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict:
    return {"response": response, "data": {}, "confidence": 0.0}


class SpendForecaster:
    def __init__(self, db: PlutoDB, currency: str = "\u20b9"):
        self.db = db
        self.currency = currency

    def _daily_totals(self) -> dict[date, float]:
        """
        Bucket every logged expense into its calendar day and sum. Days
        with zero spending are NOT present here yet — _dense_series()
        below fills those gaps with 0.0 so the model sees genuine "no
        spend that day" signal rather than a hole in the index.
        """
        totals: dict[date, float] = defaultdict(float)
        for row in self.db.get_expenses(limit=100_000):
            logged_at = row.get("logged_at")
            if not logged_at:
                continue
            try:
                # SQLite CURRENT_TIMESTAMP format: "YYYY-MM-DD HH:MM:SS"
                d = datetime.strptime(logged_at[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            totals[d] += float(row.get("amount") or 0.0)
        return dict(totals)

    @staticmethod
    def _dense_series(totals: dict[date, float]) -> tuple[list[date], list[float]]:
        if not totals:
            return [], []
        start, end = min(totals), max(totals)
        days: list[date] = []
        values: list[float] = []
        d = start
        while d <= end:
            days.append(d)
            values.append(totals.get(d, 0.0))
            d += timedelta(days=1)
        return days, values

    @staticmethod
    def _build_features(days: list[date], values: list[float]) -> tuple[list[list[float]], list[float]]:
        """
        For each day i >= LOOKBACK_WINDOW, build a feature row from the
        preceding LOOKBACK_WINDOW days plus day-of-week, with `values[i]`
        as the label. Pure lag/rolling features — no leakage from the
        future.
        """
        X: list[list[float]] = []
        y: list[float] = []
        for i in range(LOOKBACK_WINDOW, len(values)):
            window = values[i - LOOKBACK_WINDOW:i]
            row = [
                sum(window) / LOOKBACK_WINDOW,         # rolling mean
                max(window),
                min(window),
                window[-1],                             # yesterday's spend
                float(days[i].weekday()),                # 0=Mon .. 6=Sun
                float(sum(1 for v in window if v > 0)),  # active-spend days in window
            ]
            X.append(row)
            y.append(values[i])
        return X, y

    def forecast(self, horizon_days: int = FORECAST_HORIZON_DAYS) -> dict:
        totals = self._daily_totals()
        if len(totals) < MIN_DISTINCT_DAYS:
            return _err(
                f"I need at least {MIN_DISTINCT_DAYS} distinct days of logged "
                f"expenses to forecast spending — you currently have {len(totals)}. "
                "Keep logging expenses and check back."
            )

        days, values = self._dense_series(totals)
        if len(days) < MIN_DAYS_OF_HISTORY:
            return _err(
                f"I need at least {MIN_DAYS_OF_HISTORY} days of history (including "
                f"gaps) to forecast — you currently span {len(days)} day(s)."
            )

        X, y = self._build_features(days, values)
        if len(X) < 5:
            return _err(
                "Not enough data points after feature engineering to train a "
                "forecast model yet. Keep logging expenses and check back."
            )

        try:
            model = self._fit_model(X, y)
        except Exception:
            logger.exception("forecast_spending: LightGBM training failed")
            return _err("Something went wrong while training the forecast model.")

        # Iteratively roll the model forward, day by day, feeding each
        # prediction back in as the newest "actual" so the rolling-window
        # features stay consistent across the horizon.
        rolling_values = list(values)
        rolling_days = list(days)
        predictions: list[float] = []
        for step in range(horizon_days):
            next_day = rolling_days[-1] + timedelta(days=1)
            window = rolling_values[-LOOKBACK_WINDOW:]
            feature_row = [
                sum(window) / LOOKBACK_WINDOW,
                max(window),
                min(window),
                window[-1],
                float(next_day.weekday()),
                float(sum(1 for v in window if v > 0)),
            ]
            pred = max(0.0, float(model.predict([feature_row])[0]))
            predictions.append(pred)
            rolling_values.append(pred)
            rolling_days.append(next_day)

        total_forecast = sum(predictions)
        recent_avg_daily = sum(values[-LOOKBACK_WINDOW:]) / min(LOOKBACK_WINDOW, len(values))

        lines = [
            f"Spending forecast for the next {horizon_days} day(s):",
            f"  Predicted total : {self._fmt(total_forecast)}",
            f"  Predicted daily avg: {self._fmt(total_forecast / horizon_days)}",
            f"  Recent {LOOKBACK_WINDOW}-day daily avg: {self._fmt(recent_avg_daily)}",
        ]
        lines.append(
            "Based on a small model trained on your own logged history — "
            "expect it to be noisy with less than a month of data."
        )

        return _ok(
            "\n".join(lines),
            data={
                "horizon_days": horizon_days,
                "daily_predictions": predictions,
                "total_forecast": total_forecast,
                "training_days": len(days),
                "training_rows": len(X),
            },
        )

    @staticmethod
    def _fit_model(X: list[list[float]], y: list[float]):
        import lightgbm as lgb

        # A handful of tiny, shallow trees — this is meant to smooth a
        # short local history, not fit a large-scale time series. Overly
        # deep/numerous trees on <100 rows would just memorise noise.
        n_estimators = max(5, min(30, len(X)))
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            num_leaves=7,
            min_child_samples=max(1, len(X) // 10),
            learning_rate=0.1,
            verbosity=-1,
        )
        model.fit(X, y)
        return model

    def _fmt(self, amount: float) -> str:
        return f"{self.currency}{amount:,.2f}"
