"""
modules/chronos/agenda.py

Pure helpers behind two Chronos features that combine several sources:

  #86  "What's on my plate today": one timeline merging Chronos reminders,
       Hermes calendar events and Artemis goals that are due.
  #88  Weather-triggered suggestions: spot outdoor plans in that timeline
       and check them against an hourly forecast.

Nothing here performs I/O of its own. Sources are passed in (a
``ReminderService``, a Hermes-shaped module, an Artemis-shaped module) and a
forecast is passed in as a plain dict, so every rule can be tested with
fakes. A source that is absent or fails contributes a *note* rather than
breaking the whole answer.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Optional

from .recurrence import Recurrence, next_after
from .reminders import ReminderService, describe_when, parse_iso

logger = logging.getLogger(__name__)

_MAX_OCCURRENCES_PER_DAY = 24
_HEADS_UP_DAYS = 7


# ---------------------------------------------------------------------------
# Agenda (#86)
# ---------------------------------------------------------------------------

@dataclass
class AgendaItem:
    when: Optional[datetime]        # aware; None = no particular time
    source: str                     # "reminder" | "calendar" | "goal"
    text: str
    detail: str = ""
    all_day: bool = False
    overdue: bool = False
    heads_up: bool = False
    end: Optional[datetime] = None

    def sort_key(self) -> tuple:
        # Untimed items first (they're "today, any time"), then by clock.
        return (
            0 if self.when is None else 1,
            self.when.timestamp() if self.when is not None else 0.0,
            self.source,
            self.text.lower(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "when": self.when.isoformat() if self.when else None,
            "end": self.end.isoformat() if self.end else None,
            "source": self.source,
            "text": self.text,
            "detail": self.detail,
            "all_day": self.all_day,
            "overdue": self.overdue,
            "heads_up": self.heads_up,
        }


@dataclass
class Agenda:
    day: date
    items: list[AgendaItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    holiday: Optional[str] = None


def _day_bounds(day: date, tz: Any) -> tuple[datetime, datetime]:
    start = datetime.combine(day, dtime.min).replace(tzinfo=tz)
    return start, start + timedelta(days=1)


def reminder_items(
    service: ReminderService, day: date, tz: Any, now: datetime,
) -> tuple[list[AgendaItem], int]:
    """Pending Chronos reminders that fall on *day* (every occurrence of a
    recurring one). Returns ``(items, skipped_for_holiday)``."""
    start, end = _day_bounds(day, tz)
    # "Overdue" only means something for *today*: a reminder still pending from
    # an earlier moment. On any other day an earlier occurrence is simply a
    # different day's, and must not be dragged onto this one.
    is_today = start <= now < end
    items: list[AgendaItem] = []
    skipped = 0
    holiday = service.holiday_label(day)
    for row in service.list_pending():
        text = row.get("text") or "reminder"
        if row.get("place_lat") is not None and not row.get("due_time"):
            items.append(AgendaItem(
                None, "reminder", text,
                detail=f"when you get to {row.get('place_label') or 'the place'}",
            ))
            continue
        rtz = service.zone_for(row)
        due = parse_iso(row.get("due_time"), rtz)
        if due is None or due >= end:
            continue
        if row.get("skip_holidays") and holiday:
            skipped += 1
            continue
        rule = Recurrence.from_json(row["recurrence"]) if row.get("recurrence") else None
        occurrences = [due]
        if rule is not None:
            cursor = due
            while len(occurrences) < _MAX_OCCURRENCES_PER_DAY:
                nxt = next_after(rule, cursor, rtz)
                if nxt is None or nxt >= end:
                    break
                occurrences.append(nxt)
                cursor = nxt
        for occ in occurrences:
            if occ >= end:
                continue
            if occ < start and not is_today:
                continue
            items.append(AgendaItem(
                occ if occ >= start else None, "reminder", text,
                detail=rule.describe(rtz) if rule else "",
                overdue=occ < start,
            ))
    return items, skipped


def _parse_event_time(value: Any, tz: Any) -> tuple[Optional[datetime], bool]:
    """-> (aware datetime or None, is_all_day)."""
    if not value:
        return None, False
    text = str(value).strip()
    try:
        if len(text) <= 10:
            d = date.fromisoformat(text)
            return datetime.combine(d, dtime.min).replace(tzinfo=tz), True
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None, False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz), False


def calendar_items(
    hermes: Any, day: date, tz: Any, today: date,
) -> tuple[list[AgendaItem], Optional[str]]:
    """Calendar events on *day* via a Hermes-shaped module. Returns
    ``(items, note)``; *note* explains why the calendar contributed nothing."""
    if hermes is None:
        return [], None
    days = max(1, min(14, (day - today).days + 1))
    try:
        result = hermes.handle("list_events", {"days": days}, {})
    except Exception:
        logger.exception("agenda: calendar lookup raised.")
        return [], "I couldn't read your calendar."
    data = (result or {}).get("data") or {}
    if "events" not in data:
        return [], "Your calendar isn't connected, so it's not included."
    items: list[AgendaItem] = []
    for ev in data["events"]:
        start, all_day = _parse_event_time(ev.get("start"), tz)
        if start is None or start.astimezone(tz).date() != day:
            continue
        end, _ = _parse_event_time(ev.get("end"), tz)
        loc = (ev.get("location") or "").strip()
        items.append(AgendaItem(
            None if all_day else start, "calendar", ev.get("title") or "event",
            detail=f"at {loc}" if loc else "", all_day=all_day,
            end=None if all_day else end,
        ))
    return items, None


def goal_items(artemis: Any, day: date) -> list[AgendaItem]:
    """Active Artemis goals due on/before *day* (overdue ones flagged) plus
    a heads-up for goals due within a week that are under half done."""
    tracker = getattr(artemis, "tracker", None)
    if tracker is None:
        return []
    try:
        goals = tracker.get_goals()
    except Exception:
        logger.exception("agenda: goal lookup failed.")
        return []
    items: list[AgendaItem] = []
    for name, goal in goals.items():
        if getattr(goal, "status", "active") != "active":
            continue
        left = goal.days_until_due(day)
        if left is None:
            continue
        pct = f"{round(goal.progress * 100)}% done"
        if left < 0:
            items.append(AgendaItem(None, "goal", name, detail=f"overdue by {-left} day{'s' if left != -1 else ''}, {pct}", overdue=True))
        elif left == 0:
            items.append(AgendaItem(None, "goal", name, detail=f"due today, {pct}"))
        elif left <= _HEADS_UP_DAYS and goal.progress < 0.5:
            items.append(AgendaItem(
                None, "goal", name, detail=f"due in {left} day{'s' if left != 1 else ''}, {pct}",
                heads_up=True,
            ))
    return items


def build_agenda(
    service: Optional[ReminderService],
    day: date,
    tz: Any,
    now: datetime,
    *,
    hermes: Any = None,
    artemis: Any = None,
) -> Agenda:
    """Merge every available source for *day*. *service* may be ``None``
    (no memory module): calendar events and goals still work."""
    agenda = Agenda(day=day, holiday=service.holiday_label(day) if service else None)
    if service is not None:
        items, skipped = reminder_items(service, day, tz, now)
        agenda.items.extend(items)
        if skipped:
            agenda.notes.append(
                f"{skipped} reminder{'s' if skipped != 1 else ''} skipped because it's a holiday."
            )
    cal, note = calendar_items(hermes, day, tz, now.astimezone(tz).date())
    agenda.items.extend(cal)
    if note:
        agenda.notes.append(note)
    agenda.items.extend(goal_items(artemis, day))
    agenda.items.sort(key=AgendaItem.sort_key)
    return agenda


def day_label(day: date, today: date) -> str:
    delta = (day - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta == -1:
        return "yesterday"
    return f"{day.strftime('%A, %B')} {day.day}"


def format_agenda(agenda: Agenda, today: date, tz: Any, *, spoken_limit: int = 8) -> str:
    label = day_label(agenda.day, today)
    parts: list[str] = []
    if agenda.holiday:
        parts.append(f"{label.capitalize()} is marked as a holiday ({agenda.holiday}).")
    real = [i for i in agenda.items if not i.heads_up]
    heads = [i for i in agenda.items if i.heads_up]
    if not agenda.items:
        parts.append(f"You have nothing on your plate for {label}.")
    else:
        if real:
            n = len(real)
            parts.append(f"You have {n} thing{'s' if n != 1 else ''} on your plate {label}.")
        for item in real[:spoken_limit]:
            parts.append(_format_item(item, tz))
        if len(real) > spoken_limit:
            parts.append(f"And {len(real) - spoken_limit} more.")
        for item in heads[:3]:
            parts.append(f"Heads up: {item.text} ({item.detail}).")
    parts.extend(agenda.notes)
    return " ".join(parts)


def _format_item(item: AgendaItem, tz: Any) -> str:
    if item.when is not None and not item.all_day:
        clock = item.when.astimezone(tz).strftime("%I:%M %p").lstrip("0")
        head = f"At {clock}, {item.text}"
    elif item.all_day:
        head = f"All day, {item.text}"
    elif item.source == "goal":
        head = f"Goal: {item.text}"
    else:
        head = f"Any time, {item.text}" if not item.overdue else f"Overdue, {item.text}"
    detail = f" ({item.detail})" if item.detail else ""
    return f"{head}{detail}."


# ---------------------------------------------------------------------------
# Weather-triggered suggestions (#88)
# ---------------------------------------------------------------------------

_OUTDOOR_RE = re.compile(
    r"\b(run|running|jog|jogging|walk|walking|hike|hiking|trek|trekking|picnic|"
    r"cycle|cycling|bike|biking|cricket|football|soccer|tennis|badminton|beach|"
    r"park|outdoor|outdoors|bbq|barbecue|garden|gardening|camping|marathon|"
    r"drive|road trip|sightseeing)\b",
    re.IGNORECASE,
)

# WMO weather codes that mean precipitation / storms.
_WET_CODES = frozenset(
    list(range(51, 68)) + list(range(71, 78)) + list(range(80, 87)) + [95, 96, 99]
)
_CODE_LABEL = {
    51: "drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "showers", 81: "showers", 82: "heavy showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}
RAIN_PROBABILITY_THRESHOLD = 50
_DEFAULT_ACTIVITY_HOURS = 2


def is_outdoor(text: str) -> bool:
    return bool(_OUTDOOR_RE.search(text or ""))


@dataclass
class WeatherConcern:
    item: AgendaItem
    probability: int
    condition: str
    window: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(), "probability": self.probability,
            "condition": self.condition, "window": self.window,
        }


def _hourly_slots(forecast: dict[str, Any], tz: Any) -> list[tuple[datetime, int, int]]:
    """Forecast dict (Open-Meteo hourly, requested in the user's timezone)
    -> [(aware hour start, precipitation probability %, weather code)]."""
    times = forecast.get("time") or []
    probs = forecast.get("precipitation_probability") or []
    # Open-Meteo is moving from "weathercode" to "weather_code"; accept both so
    # a rename on their side can't silently turn every forecast into "no rain".
    codes = forecast.get("weathercode") or forecast.get("weather_code") or []
    slots = []
    for i, t in enumerate(times):
        try:
            start = datetime.fromisoformat(t)
        except (TypeError, ValueError):
            continue
        if start.tzinfo is None:
            start = start.replace(tzinfo=tz)
        p = probs[i] if i < len(probs) and probs[i] is not None else 0
        c = codes[i] if i < len(codes) and codes[i] is not None else 0
        slots.append((start, int(p), int(c)))
    return slots


def rain_outlook(forecast: dict[str, Any], day: date, tz: Any) -> Optional[tuple[int, str]]:
    """Peak rain probability between 08:00 and 20:00 on *day* and the hour it
    happens, or ``None`` when the forecast doesn't cover that day."""
    best: Optional[tuple[int, datetime]] = None
    for start, prob, code in _hourly_slots(forecast, tz):
        local = start.astimezone(tz)
        if local.date() != day or not 8 <= local.hour < 20:
            continue
        score = max(prob, 100 if code in _WET_CODES else 0)
        if best is None or score > best[0]:
            best = (score, local)
    if best is None:
        return None
    return best[0], best[1].strftime("%I:%M %p").lstrip("0")


def assess_item(item: AgendaItem, forecast: dict[str, Any], tz: Any) -> Optional[WeatherConcern]:
    """Return a concern if rain/snow/storm is likely while *item* takes place."""
    slots = _hourly_slots(forecast, tz)
    if not slots:
        return None
    if item.when is not None and not item.all_day:
        start = item.when
        end = item.end if item.end and item.end > start else start + timedelta(hours=_DEFAULT_ACTIVITY_HOURS)
    else:
        base = item.when.astimezone(tz).date() if item.when else slots[0][0].astimezone(tz).date()
        start = datetime.combine(base, dtime(8, 0)).replace(tzinfo=tz)
        end = datetime.combine(base, dtime(20, 0)).replace(tzinfo=tz)
    hour = timedelta(hours=1)
    covered = [s for s in slots if s[0] < end and s[0] + hour > start]
    if not covered:
        return None
    worst = max(covered, key=lambda s: (s[2] in _WET_CODES, s[1]))
    wet = worst[2] in _WET_CODES
    if worst[1] < RAIN_PROBABILITY_THRESHOLD and not wet:
        return None
    condition = _CODE_LABEL.get(worst[2], "rain")
    first = min(s[0] for s in covered if s[1] >= RAIN_PROBABILITY_THRESHOLD or s[2] in _WET_CODES)
    window = first.astimezone(tz).strftime("%I:%M %p").lstrip("0")
    return WeatherConcern(item, worst[1], condition, window)


def assess_agenda(items: list[AgendaItem], forecast: dict[str, Any], tz: Any) -> list[WeatherConcern]:
    concerns = []
    for item in items:
        if item.source == "goal" or not is_outdoor(item.text):
            continue
        concern = assess_item(item, forecast, tz)
        if concern is not None:
            concerns.append(concern)
    return concerns


def format_concerns(concerns: list[WeatherConcern], tz: Any) -> str:
    lines = []
    for c in concerns[:3]:
        when = ""
        if c.item.when is not None and not c.item.all_day:
            when = " at " + c.item.when.astimezone(tz).strftime("%I:%M %p").lstrip("0")
        lines.append(
            f"{c.condition.capitalize()} looks likely from around {c.window} "
            f"({c.probability}% chance), which could affect '{c.item.text}'{when}."
        )
    return " ".join(lines)

# ---------------------------------------------------------------------------
# What actually needs attention this week (#163)
# ---------------------------------------------------------------------------
#
# "What's on my plate" (above) lists everything for one day. This answers a
# different question: of everything coming up in the next week, which few
# things actually need *attention*, and why those? It reads the same three
# sources (Chronos reminders, Hermes calendar, Artemis goals/habits) and ranks
# them with plain, inspectable rules rather than a model.
#
# The "critical path" part is the calendar. A goal due Friday that is 30% done
# is a different problem when Wednesday and Thursday are full of meetings than
# when they're empty, so each goal is also scored by how many *open* days
# remain before it is due.
#
# Rules (higher score = more pressing):
#   goal overdue ................ 100 + days overdue (capped at 30 extra)
#   goal due in the window ...... 50 + 30 x (1 - progress) + 2 x days closer, +10 if only 0-2 open
#                                 days remain; -25 once it's 80%+ done
#   reminder overdue ............ 60
#   calendar event that is a
#     deadline (exam, submit...) . 55 - 3 per day away
#   habit streak ending today ... 35 + streak (capped at 21)
#   reminder due in the window .. 25 - 2 per day away (one-off reminders only)

_FOCUS_WINDOW_DAYS = 7
_FOCUS_LIMIT = 5
_BUSY_DAY_HOURS = 4.0
_DEADLINE_RE = re.compile(
    r"\b(deadline|due|exam|test|viva|submit|submission|assignment|interview|"
    r"presentation|defen[cs]e|final|tax|renewal|expires?)\b",
    re.IGNORECASE,
)


@dataclass
class FocusItem:
    score: float
    source: str                     # "goal" | "reminder" | "calendar" | "habit"
    text: str
    why: str
    due: Optional[date] = None
    open_days: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1), "source": self.source, "text": self.text,
            "why": self.why, "due": self.due.isoformat() if self.due else None,
            "open_days": self.open_days,
        }


@dataclass
class WeekFocus:
    start: date
    days: int
    items: list[FocusItem] = field(default_factory=list)
    considered: int = 0              # how many candidates were scored before the cut
    notes: list[str] = field(default_factory=list)


def _events_in_window(
    hermes: Any, tz: Any, today: date, days: int,
) -> tuple[list[tuple[datetime, Optional[datetime], bool, str]], Optional[str]]:
    """Calendar events across the window as ``(start, end, all_day, title)``."""
    if hermes is None:
        return [], None
    try:
        result = hermes.handle("list_events", {"days": days}, {})
    except Exception:
        logger.exception("week focus: calendar lookup raised.")
        return [], "I couldn't read your calendar."
    data = (result or {}).get("data") or {}
    if "events" not in data:
        return [], "Your calendar isn't connected, so it's not included."
    out = []
    for ev in data["events"]:
        start, all_day = _parse_event_time(ev.get("start"), tz)
        if start is None:
            continue
        end, _ = _parse_event_time(ev.get("end"), tz)
        out.append((start, end, all_day, ev.get("title") or "event"))
    return out, None


def _busy_days(events: list, today: date, window_end: date) -> set[date]:
    """Days in [today, window_end] too full to count as open: an all-day
    event, or timed events adding up to _BUSY_DAY_HOURS or more."""
    hours: dict[date, float] = {}
    busy: set[date] = set()
    for start, end, all_day, _title in events:
        d = start.date()
        if not (today <= d <= window_end):
            continue
        if all_day:
            busy.add(d)
            continue
        if end is not None and end > start:
            hours[d] = hours.get(d, 0.0) + (end - start).total_seconds() / 3600.0
    busy.update(d for d, h in hours.items() if h >= _BUSY_DAY_HOURS)
    return busy


def build_week_focus(
    service: Optional[ReminderService],
    tz: Any,
    now: datetime,
    *,
    hermes: Any = None,
    artemis: Any = None,
    days: int = _FOCUS_WINDOW_DAYS,
    limit: int = _FOCUS_LIMIT,
) -> WeekFocus:
    """Rank what needs attention over the next *days* days. Any source that is
    missing or fails just contributes a note; the rest still answer."""
    days = max(1, min(14, int(days)))
    today = now.astimezone(tz).date()
    window_end = today + timedelta(days=days - 1)
    focus = WeekFocus(start=today, days=days)
    candidates: list[FocusItem] = []

    events, note = _events_in_window(hermes, tz, today, days)
    if note:
        focus.notes.append(note)
    busy = _busy_days(events, today, window_end)

    # -- goals --------------------------------------------------------
    tracker = getattr(artemis, "tracker", None)
    if tracker is not None:
        try:
            goals = tracker.get_goals()
        except Exception:
            logger.exception("week focus: goal lookup failed.")
            goals = {}
            focus.notes.append("I couldn't read your goals.")
        for name, goal in goals.items():
            if getattr(goal, "status", "active") != "active":
                continue
            left = goal.days_until_due(today)
            if left is None or left > days:
                continue
            pct = round(goal.progress * 100)
            due = today + timedelta(days=left)
            if left < 0:
                candidates.append(FocusItem(
                    100 + min(-left, 30), "goal", name,
                    f"overdue by {-left} day{'s' if left != -1 else ''}, {pct}% done", due,
                ))
                continue
            # Open days between now and the due date, counting the due day itself.
            span = [today + timedelta(days=i) for i in range(left + 1)]
            open_days = sum(1 for d in span if d not in busy)
            score = 50 + 30 * (1 - goal.progress) + 2 * (days - left)
            if open_days <= 2:
                score += 10
            if goal.progress >= 0.8:
                score -= 25
            when = "today" if left == 0 else f"in {left} day{'s' if left != 1 else ''}"
            why = f"due {when}, {pct}% done"
            if events:
                why += (f", and only {open_days} open day{'s' if open_days != 1 else ''} "
                        "before then" if open_days <= 2 else f", {open_days} open days before then")
            candidates.append(FocusItem(score, "goal", name, why, due, open_days))

        # -- habit streaks that end today ------------------------------
        try:
            default_grace = getattr(tracker, "default_grace_days", 0)
            for name, habit in tracker.get_habits().items():
                check = getattr(habit, "streak_ends_if_skipped_today", None)
                if habit.streak >= 3 and callable(check) and check(today, default_grace):
                    candidates.append(FocusItem(
                        35 + min(habit.streak, 21), "habit", name,
                        f"your {habit.streak}-day streak ends if it's skipped today", today,
                    ))
        except Exception:
            logger.exception("week focus: habit lookup failed.")

    # -- calendar deadlines -------------------------------------------
    for start, _end, all_day, title in events:
        d = start.date()
        if today <= d <= window_end and _DEADLINE_RE.search(title):
            away = (d - today).days
            when = "today" if away == 0 else "tomorrow" if away == 1 else d.strftime("%A")
            clock = "" if all_day else start.astimezone(tz).strftime(" at %I:%M %p").replace(" 0", " ")
            candidates.append(FocusItem(55 - 3 * away, "calendar", title, f"on your calendar {when}{clock}", d))

    # -- reminders ----------------------------------------------------
    if service is not None:
        try:
            for offset in range(days):
                day = today + timedelta(days=offset)
                items, _skipped = reminder_items(service, day, tz, now)
                for it in items:
                    if offset == 0 and it.overdue:
                        candidates.append(FocusItem(60, "reminder", it.text, "overdue", today))
                    elif not it.detail and not it.overdue and it.when is not None:
                        candidates.append(FocusItem(
                            25 - 2 * offset, "reminder", it.text,
                            f"reminder {day_label(day, today)}", day,
                        ))
        except Exception:
            logger.exception("week focus: reminder lookup failed.")
            focus.notes.append("I couldn't read your reminders.")

    # One entry per (source, text): a recurring reminder or a multi-day goal
    # mustn't fill the list with copies of itself.
    best: dict[tuple[str, str], FocusItem] = {}
    for c in candidates:
        key = (c.source, c.text.strip().lower())
        if key not in best or c.score > best[key].score:
            best[key] = c
    ranked = sorted(best.values(), key=lambda c: (-c.score, c.due or window_end, c.text.lower()))
    focus.considered = len(ranked)
    focus.items = ranked[: max(1, limit)]
    return focus


def format_week_focus(focus: WeekFocus) -> str:
    if not focus.items:
        text = (f"Nothing looks pressing over the next {focus.days} days: no overdue goals or "
                "reminders, no deadlines on your calendar, and no streaks about to break.")
        return " ".join([text, *focus.notes])
    n = len(focus.items)
    head = (f"Here {'is the one thing' if n == 1 else f'are the {n} things'} that actually "
            f"need attention over the next {focus.days} days")
    if focus.considered > n:
        head += f" (out of {focus.considered} I looked at)"
    lines = [head + "."]
    for i, item in enumerate(focus.items, 1):
        label = {"goal": "Goal", "habit": "Habit", "calendar": "Calendar",
                 "reminder": "Reminder"}.get(item.source, item.source.title())
        lines.append(f"{i}. {label}: {item.text} ({item.why}).")
    lines.extend(focus.notes)
    return " ".join(lines)
