"""
artemis/extras.py

Pure logic for the Artemis features that sit beside habits and goals:

- focus sessions / Pomodoro (#122)
- badges (#127)
- typical completion time for smart nudges (#129)

Everything works on the tracker's plain-dict state, so it is testable without
a file, a clock or an LLM; ArtemisTracker adds the lock and persistence.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median
from typing import Any, Optional

# --- focus sessions (#122) -------------------------------------------------

DEFAULT_FOCUS_MINUTES = 25
DEFAULT_BREAK_MINUTES = 5
MAX_FOCUS_MINUTES = 180
SESSIONS_CAP = 1000          # oldest sessions are dropped first


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _parse(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def focus_start(focus: dict, now: datetime, minutes: int, task: str = "") -> dict:
    """Start a session in *focus* (mutated). Returns the active session."""
    active = {"start": _iso(now), "planned": int(minutes), "task": task.strip()[:120]}
    focus["active"] = active
    return active


def focus_finish(focus: dict, now: datetime) -> Optional[dict]:
    """
    Close the active session and log it. Minutes are what actually elapsed, capped at the
    planned length (walking away does not earn extra minutes). None when nothing was running.
    """
    active = focus.get("active")
    if not active:
        return None
    start = _parse(active.get("start", ""))
    planned = int(active.get("planned", DEFAULT_FOCUS_MINUTES))
    elapsed = max(0.0, (now - start).total_seconds() / 60) if start else 0.0
    minutes = int(min(elapsed, planned))
    record = {
        "start": active.get("start", ""), "minutes": minutes, "planned": planned,
        "task": active.get("task", ""), "completed": elapsed >= planned,
    }
    sessions = focus.setdefault("sessions", [])
    sessions.append(record)
    del sessions[:-SESSIONS_CAP]
    focus["active"] = None
    return record


def focus_remaining(active: dict, now: datetime) -> Optional[float]:
    """Minutes left in *active* (negative once over); None if its start can't be read."""
    start = _parse(active.get("start", ""))
    if start is None:
        return None
    return int(active.get("planned", DEFAULT_FOCUS_MINUTES)) - (now - start).total_seconds() / 60


def focus_stats(focus: dict, now: datetime, days: int = 7) -> dict[str, Any]:
    """Minutes and sessions over the last *days* calendar days (today included), plus all-time."""
    cutoff = (now - timedelta(days=days - 1)).date().isoformat()
    window = [s for s in focus.get("sessions", []) if str(s.get("start", ""))[:10] >= cutoff]
    all_min = sum(int(s.get("minutes", 0)) for s in focus.get("sessions", []))
    return {
        "days": days,
        "minutes": sum(int(s.get("minutes", 0)) for s in window),
        "sessions": len(window),
        "completed": sum(1 for s in window if s.get("completed")),
        "total_minutes": all_min,
        "total_sessions": len(focus.get("sessions", [])),
        "tasks": sorted({s["task"] for s in window if s.get("task")})[:5],
    }


# --- badges (#127) ---------------------------------------------------------

# id -> (label, description). Thresholds live in evaluate_badges().
BADGES: dict[str, tuple[str, str]] = {
    "streak_7": ("Week Warrior", "a 7-day habit streak"),
    "streak_30": ("Monthly Master", "a 30-day habit streak"),
    "streak_100": ("Centurion", "a 100-day habit streak"),
    "streak_365": ("Year of Fire", "a 365-day habit streak"),
    "done_50": ("Fifty Strong", "50 habit completions"),
    "done_250": ("Quarter Thousand", "250 habit completions"),
    "habits_3": ("Habit Builder", "tracking 3 habits at once"),
    "goal_first": ("Goal Getter", "completing your first goal"),
    "goal_5": ("Finisher", "completing 5 goals"),
    "focus_10h": ("In the Zone", "10 hours of focus time"),
    "focus_50h": ("Deep Work", "50 hours of focus time"),
}


def evaluate_badges(habits: dict, goals: dict, focus: dict) -> set[str]:
    """Badge ids currently earned. *habits*/*goals* are Habit/Goal objects keyed by name."""
    best = max((h.best_streak for h in habits.values()), default=0)
    done = sum(h.total_completions for h in habits.values())
    goals_done = sum(1 for g in goals.values() if g.status == "completed")
    focus_min = sum(int(s.get("minutes", 0)) for s in focus.get("sessions", []))
    earned = set()
    for bid, ok in (
        ("streak_7", best >= 7), ("streak_30", best >= 30), ("streak_100", best >= 100),
        ("streak_365", best >= 365), ("done_50", done >= 50), ("done_250", done >= 250),
        ("habits_3", len(habits) >= 3), ("goal_first", goals_done >= 1), ("goal_5", goals_done >= 5),
        ("focus_10h", focus_min >= 600), ("focus_50h", focus_min >= 3000),
    ):
        if ok:
            earned.add(bid)
    return earned


# --- smart nudges (#129) ---------------------------------------------------

MIN_NUDGE_SAMPLES = 5
TIMES_CAP = 30               # completion times remembered per habit


def typical_minute(times: list[int]) -> Optional[int]:
    """Median minute-of-day a habit is usually logged, or None with too few samples."""
    vals = [int(t) for t in times if isinstance(t, (int, float)) and 0 <= t < 1440]
    return int(median(vals)) if len(vals) >= MIN_NUDGE_SAMPLES else None


def fmt_minute(minute: int) -> str:
    h, m = divmod(int(minute), 60)
    return f"{h % 12 or 12}:{m:02d} {'am' if h < 12 else 'pm'}"
