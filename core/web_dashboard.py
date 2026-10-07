"""
core/web_dashboard.py - the single-screen "Today" view (backlog #180).

``build_today(ui)`` gathers, in one dict, what the Project_Hestia "launcher"
home screen shows at a glance: today's agenda, the one thing to do first,
deadlines coming up, habits left today, the health numbers so far today, this
month's spending and study cards due.

Every section is fetched independently. A module that is switched off gives
``None`` for its section; one that raises gives ``None`` plus a line in
``errors``. One broken module never blanks the page.

Top priority
------------
There is no "priority" field across Hestia, so the pick is a score over
things that really do have one (all arithmetic below, nothing hidden):

  goal          priority weight (high 30 / medium 20 / none 15 / low 10)
                + urgency from its due date (overdue 50, due today 40,
                  within 3 days 25, within 7 days 10)
                - up to 10 for progress already made
  reminder      35 when due in the next hours today, 45 when already overdue
  habit streak  25 + streak/3 (max 10) when a streak of 3+ would break if
                today is skipped

The highest score wins and carries a plain-English ``reason`` so the page can
say *why* it is the top item.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_PRIORITY_WEIGHT = {"high": 30, "medium": 20, "low": 10, None: 15}
DEADLINE_HORIZON_DAYS = 14


# --------------------------------------------------------------------------
# scoring (pure functions; unit-tested)
# --------------------------------------------------------------------------

def _urgency(days_left: Optional[int]) -> tuple[int, str]:
    if days_left is None:
        return 0, ""
    if days_left < 0:
        n = -days_left
        return 50, f"overdue by {n} day{'s' if n != 1 else ''}"
    if days_left == 0:
        return 40, "due today"
    if days_left == 1:
        return 25, "due tomorrow"
    if days_left <= 3:
        return 25, f"due in {days_left} days"
    if days_left <= 7:
        return 10, f"due in {days_left} days"
    return 0, f"due in {days_left} days"


def score_goal(goal: dict, today: date) -> Optional[dict]:
    """Score one active goal. *goal*: name, priority, due_date, progress."""
    prio = goal.get("priority")
    days_left = None
    due = goal.get("due_date")
    if due:
        try:
            days_left = (date.fromisoformat(str(due)[:10]) - today).days
        except ValueError:
            days_left = None
    urgency, due_text = _urgency(days_left)
    progress = max(0.0, min(1.0, float(goal.get("progress") or 0.0)))
    score = _PRIORITY_WEIGHT.get(prio, 15) + urgency - round(10 * progress)
    parts = []
    if prio:
        parts.append(f"{prio}-priority goal")
    else:
        parts.append("goal")
    if due_text:
        parts.append(due_text)
    if progress:
        parts.append(f"{round(progress * 100)}% done")
    return {
        "kind": "goal", "title": goal["name"], "score": score,
        "reason": ", ".join(parts), "due_date": due, "priority": prio,
        "progress": progress,
    }


def score_reminder(item: dict, now: datetime) -> Optional[dict]:
    """Score a reminder-sourced agenda item (dict from AgendaItem.to_dict)."""
    if item.get("source") != "reminder":
        return None
    if item.get("overdue"):
        return {"kind": "reminder", "title": item["text"], "score": 45,
                "reason": "reminder is overdue", "when": item.get("when")}
    when = item.get("when")
    if when:
        try:
            at = datetime.fromisoformat(when)
            if now.tzinfo and at.tzinfo is None:
                at = at.replace(tzinfo=now.tzinfo)
            delta = at - now
            if timedelta(0) <= delta <= timedelta(hours=6):
                return {"kind": "reminder", "title": item["text"], "score": 35,
                        "reason": "reminder coming up today", "when": when}
        except ValueError:
            pass
    return None


def score_habit(name: str, streak: int, done_today: bool) -> Optional[dict]:
    if done_today or streak < 3:
        return None
    return {
        "kind": "habit", "title": name, "score": 25 + min(10, streak // 3),
        "reason": f"keeps your {streak}-day streak alive",
    }


def pick_top(candidates: list[Optional[dict]]) -> Optional[dict]:
    """Highest score; ties go to the earlier candidate (goals list first)."""
    best: Optional[dict] = None
    for c in candidates:
        if c is not None and (best is None or c["score"] > best["score"]):
            best = c
    return best


# --------------------------------------------------------------------------
# section builders
# --------------------------------------------------------------------------

def _goals(ui) -> list[dict]:
    out = []
    for name, g in ui.artemis.tracker.get_goals().items():
        if g.status != "active":
            continue
        out.append({"name": name, "priority": g.priority, "due_date": g.due_date,
                    "progress": g.progress})
    return out


def _habits(ui, today: date) -> dict:
    habits = ui.artemis.tracker.get_habits()
    iso = today.isoformat()
    rows = []
    for name, h in habits.items():
        if h.paused_on(today):
            continue
        rows.append({"name": name, "streak": h.streak, "done_today": h.last_done == iso})
    rows.sort(key=lambda r: (r["done_today"], -r["streak"], r["name"]))
    done = sum(1 for r in rows if r["done_today"])
    return {"total": len(rows), "done": done, "left": [r for r in rows if not r["done_today"]][:8],
            "all": rows}


def _health(ui, today: date) -> dict:
    data = ui.apollo.dashboard_data(7)
    iso = today.isoformat()

    def last(series, key):
        for row in reversed(series):
            if row.get("date") == iso or row.get("day") == iso:
                return row.get(key)
        return None

    sleep_last = data["sleep"][-1] if data.get("sleep") else None
    water_today = next((r for r in data.get("water", []) if r["date"] == iso), None)
    return {
        "water_ml": water_today["ml"] if water_today else 0,
        "water_goal_ml": water_today["goal_ml"] if water_today else None,
        "steps": last(data.get("steps", []), "steps"),
        "mood": last(data.get("mood", []), "score"),
        "sleep_hours": sleep_last["hours"] if sleep_last else None,
        "sleep_date": sleep_last["date"] if sleep_last else None,
        "units": data.get("units", {}),
    }


def _finance(ui, today: date) -> dict:
    db = ui.pluto.pf_manager.db
    start = today.replace(day=1)
    nxt = (start + timedelta(days=32)).replace(day=1)
    by_cat = db.get_totals_between(start.isoformat(), nxt.isoformat())
    total = sum(float(r["total"] or 0) for r in by_cat)
    return {
        "month": start.strftime("%B %Y"),
        "spent": round(total, 2),
        "top_categories": [{"category": r["category"], "total": round(float(r["total"]), 2)}
                           for r in by_cat[:3]],
    }


def _study(ui) -> Optional[dict]:
    store = getattr(ui.memory, "study_store", None)
    return store.stats() if store is not None else None


def _deadlines(goals: list[dict], upcoming: list[dict], today: date) -> list[dict]:
    """Goals with a due date and reminders over the next fortnight, soonest first."""
    rows: list[dict] = []
    for g in goals:
        if not g.get("due_date"):
            continue
        try:
            days = (date.fromisoformat(str(g["due_date"])[:10]) - today).days
        except ValueError:
            continue
        if days <= DEADLINE_HORIZON_DAYS:
            rows.append({"kind": "goal", "title": g["name"], "date": g["due_date"],
                         "days_left": days, "priority": g.get("priority"),
                         "overdue": days < 0})
    for item in upcoming:
        day = item.get("day")
        if not day:
            continue
        days = (date.fromisoformat(day) - today).days
        rows.append({"kind": "reminder", "title": item["text"], "date": day,
                     "days_left": days, "when": item.get("when"), "overdue": False})
    rows.sort(key=lambda r: (r["days_left"], r["kind"], r["title"].lower()))
    return rows[:12]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_today(ui, now: Optional[datetime] = None) -> dict:
    errors: dict[str, str] = {}

    def section(name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:
            logger.exception("[WebUI] dashboard section %r failed", name)
            errors[name] = type(exc).__name__
            return None

    chronos = section("chronos", lambda: ui.chronos.dashboard_data(7)
                      if getattr(ui, "chronos", None) is not None else None)
    now = now or datetime.now().astimezone()
    today = date.fromisoformat(chronos["date"]) if chronos else now.date()

    goals = section("goals", lambda: _goals(ui) if ui.artemis is not None else None)
    habits = section("habits", lambda: _habits(ui, today) if ui.artemis is not None else None)
    health = section("health", lambda: _health(ui, today) if ui.apollo is not None else None)
    finance = section("finance", lambda: _finance(ui, today) if ui.pluto is not None else None)
    study = section("study", lambda: _study(ui))

    candidates: list[Optional[dict]] = []
    for g in goals or []:
        candidates.append(score_goal(g, today))
    for item in (chronos or {}).get("today", []):
        candidates.append(score_reminder(item, now))
    for h in (habits or {}).get("all", []):
        candidates.append(score_habit(h["name"], h["streak"], h["done_today"]))

    return {
        "date": today.isoformat(),
        "generated_at": now.isoformat(timespec="seconds"),
        "holiday": (chronos or {}).get("holiday"),
        "agenda": (chronos or {}).get("today"),
        "agenda_notes": (chronos or {}).get("notes", []),
        "top_priority": pick_top(candidates),
        "deadlines": _deadlines(goals or [], (chronos or {}).get("upcoming", []), today),
        "habits": habits,
        "health": health,
        "finance": finance,
        "study": study,
        "modules": {
            "chronos": getattr(ui, "chronos", None) is not None,
            "artemis": ui.artemis is not None,
            "apollo": ui.apollo is not None,
            "pluto": ui.pluto is not None,
            "study": study is not None,
        },
        "errors": errors,
    }
