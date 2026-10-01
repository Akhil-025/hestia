"""
modules/apollo/schedule.py

Pure timing logic for Apollo's proactive features (backlog #115, #118, #120).
Nothing here reads the clock or the database: callers pass ``now`` and the
persisted state in, and get a decision (and the new state) back.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

DEFAULT_WATER_GOAL_ML = 2000.0


def parse_hhmm(text: Any, default_min: int) -> int:
    """'07:30' -> 450. Falls back to *default_min* on anything malformed."""
    try:
        h, m = str(text).strip().split(":")
        h, m = int(h), int(m)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60 + m
    except (ValueError, AttributeError):
        pass
    return default_min


def expected_fraction(now_min: int, wake_min: int, sleep_min: int) -> float:
    """Fraction of the daily target expected by *now_min* (linear over the
    waking window; 0 before waking, 1 after bedtime)."""
    if sleep_min <= wake_min:
        return 1.0
    if now_min <= wake_min:
        return 0.0
    if now_min >= sleep_min:
        return 1.0
    return (now_min - wake_min) / (sleep_min - wake_min)


def hydration_status(
    total_ml: float, goal_ml: float, now_min: int, wake_min: int, sleep_min: int
) -> dict[str, Any]:
    """Actual intake vs the pace needed to reach *goal_ml* by bedtime."""
    if now_min < wake_min or now_min >= sleep_min:
        phase = "quiet"
    else:
        phase = "active"
    frac = expected_fraction(now_min, wake_min, sleep_min)
    expected = goal_ml * frac
    return {
        "total_ml": total_ml,
        "goal_ml": goal_ml,
        "expected_ml": expected,
        "behind_ml": max(0.0, expected - total_ml),
        "ahead_ml": max(0.0, total_ml - expected),
        "fraction_of_day": frac,
        "phase": phase,
    }


def nudge_decision(
    status: dict[str, Any],
    state: dict[str, Any],
    now: datetime,
    *,
    threshold_ml: float = 400.0,
    cooldown_min: int = 90,
    daily_cap: int = 4,
) -> tuple[bool, str, dict[str, Any]]:
    """Should we nudge now? Returns (send, reason, new_state).

    Only nudges when the user is *behind* pace by at least ``threshold_ml``,
    inside waking hours (quiet hours never nudge), no more than ``daily_cap``
    per day, and not within ``cooldown_min`` of the previous nudge.
    """
    today = now.date().isoformat()
    count = int(state.get("count", 0)) if state.get("date") == today else 0
    last_raw = state.get("last") if state.get("date") == today else None
    new_state = {"date": today, "count": count, "last": last_raw}

    if status["phase"] != "active":
        return False, "quiet_hours", new_state
    if status["behind_ml"] < threshold_ml:
        return False, "on_pace", new_state
    if count >= daily_cap:
        return False, "daily_cap", new_state
    if last_raw:
        try:
            last = datetime.fromisoformat(last_raw)
            if last.tzinfo is None and now.tzinfo is not None:
                last = last.replace(tzinfo=now.tzinfo)
            if now - last < timedelta(minutes=cooldown_min):
                return False, "cooldown", new_state
        except ValueError:
            pass
    return True, "behind", {"date": today, "count": count + 1, "last": now.isoformat()}


def iso_week_key(d: date) -> str:
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def weekly_due(
    now: datetime, last_key: Optional[str], weekday: int = 6, hour: int = 18
) -> bool:
    """True once per ISO week, on/after *weekday* at/after *hour*.

    Using "on or after" (not "on") means a machine that was off on Sunday
    still sends the summary when it comes back, rather than skipping a week.
    """
    if last_key == iso_week_key(now.date()):
        return False
    if now.weekday() > weekday:
        return True
    return now.weekday() == weekday and now.hour >= hour


def every_n_days_due(today: date, last_iso: Optional[str], n: int) -> bool:
    """Cadence check for reminders. ``n <= 0`` means off."""
    if n <= 0:
        return False
    if not last_iso:
        return True
    try:
        last = date.fromisoformat(last_iso)
    except ValueError:
        return True
    return (today - last).days >= n
