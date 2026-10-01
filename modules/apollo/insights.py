"""
modules/apollo/insights.py

Pure, dependency-free helpers for Apollo's analytics (backlog #111, #112,
#116, #117, #120, #126, #161). Nothing in here touches the database, the
clock (every "today" is passed in), the network or an LLM, so every
function is independently testable and deterministic.

Design rules that apply to everything below
-------------------------------------------
- Reflect the user's own logged data back to them; never diagnose.
- Refuse to report a pattern when the sample is thin. Every comparison
  reports its sample sizes.
- Timestamps in the databases are UTC (SQLite CURRENT_TIMESTAMP). Day
  bucketing into the user's calendar days happens here, via ``local_date``.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any, Iterable, Optional

try:  # zoneinfo is stdlib on 3.9+, but tzdata may be missing on Windows.
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Time zones and day bucketing
# ---------------------------------------------------------------------------


def resolve_tz(name: Optional[str]) -> tzinfo:
    """Return a tzinfo for *name*, falling back to UTC on anything unusable."""
    if not name or ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(str(name))
    except Exception:
        return timezone.utc


def parse_ts(ts: Any) -> Optional[datetime]:
    """Parse a stored timestamp (SQLite ``YYYY-MM-DD HH:MM:SS`` = UTC)."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if not ts:
        return None
    text = str(ts).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def local_date(ts: Any, tz: tzinfo) -> Optional[date]:
    """The calendar date of *ts* as seen in *tz*."""
    dt = parse_ts(ts)
    return dt.astimezone(tz).date() if dt else None


def week_start(d: date) -> date:
    """Monday of the ISO week containing *d*."""
    return d - timedelta(days=d.weekday())


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Bed / wake time parsing (#111)
# ---------------------------------------------------------------------------


def _t(n: int) -> str:
    return (
        rf"(?P<h{n}>\d{{1,2}})(?::(?P<m{n}>\d{{2}}))?\s*(?P<ap{n}>[ap]\.?m\.?)?"
    )


_WINDOW_RE = re.compile(
    r"(?<![\d:.])" + _t(1) + r"\s*(?:to|until|till|-|\u2013|\u2014)\s*" + _t(2)
    + r"(?![\d:])",
    re.IGNORECASE,
)
_SINGLE_RE = re.compile(r"^\s*" + _t(1) + r"\s*$", re.IGNORECASE)


def _to_minutes(h: str, m: Optional[str], ap: Optional[str]) -> Optional[int]:
    """One clock time -> minutes after midnight, or None if ambiguous/invalid.

    A time is only accepted when it is self-sufficient: it has am/pm, or it
    has a ``:mm`` part (24-hour form). A bare "6" is ambiguous, so we don't
    guess.
    """
    hour = int(h)
    minute = int(m) if m is not None else 0
    if minute > 59:
        return None
    if ap:
        if not (1 <= hour <= 12):
            return None
        hour = hour % 12 + (12 if ap.lower().startswith("p") else 0)
    else:
        if m is None or hour > 23:
            return None
    return hour * 60 + minute


def min_to_hhmm(minutes: int) -> str:
    minutes %= 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def hhmm_to_min(text: Any) -> Optional[int]:
    """'23:30' -> 1410. Also accepts anything ``parse_time_of_day`` accepts."""
    if text is None:
        return None
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(text))
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        return h * 60 + m if h <= 23 and m <= 59 else None
    parsed = parse_time_of_day(text)
    return hhmm_to_min(parsed) if parsed else None


def parse_time_of_day(text: Any) -> Optional[str]:
    """Parse a single time such as '11pm', '6:30 am' or '23:15' -> 'HH:MM'."""
    match = _SINGLE_RE.match(str(text or ""))
    if not match:
        return None
    mins = _to_minutes(match.group("h1"), match.group("m1"), match.group("ap1"))
    return min_to_hhmm(mins) if mins is not None else None


def parse_sleep_window(text: Any) -> Optional[tuple[str, str]]:
    """Find a 'bed to wake' pair in free text, e.g. '11pm to 6:30am'.

    Returns ('23:00', '06:30') or None. Ambiguous phrasing such as
    'slept 7 to 8 hours' is deliberately not treated as a window.
    """
    for match in _WINDOW_RE.finditer(str(text or "")):
        bed = _to_minutes(match.group("h1"), match.group("m1"), match.group("ap1"))
        wake = _to_minutes(match.group("h2"), match.group("m2"), match.group("ap2"))
        if bed is not None and wake is not None:
            return min_to_hhmm(bed), min_to_hhmm(wake)
    return None


def window_hours(bed: str, wake: str) -> Optional[float]:
    """Hours between bed and wake, wrapping past midnight."""
    b, w = hhmm_to_min(bed), hhmm_to_min(wake)
    if b is None or w is None:
        return None
    diff = (w - b) % 1440
    return round(diff / 60, 2) if diff else None


# ---------------------------------------------------------------------------
# Sleep quality score (#111)
# ---------------------------------------------------------------------------

_SLEEP_WEIGHTS = {"duration": 0.5, "consistency": 0.3, "rating": 0.2}
_QUALITY_WORDS = {
    "terrible": 1, "awful": 1, "bad": 2, "poor": 2, "restless": 2,
    "ok": 3, "okay": 3, "fine": 3, "decent": 3, "average": 3,
    "good": 4, "great": 5, "excellent": 5, "amazing": 5, "restful": 5,
}


def circular_std_minutes(times_min: list[int]) -> Optional[float]:
    """Spread of clock times in minutes, correct across midnight.

    23:30 and 00:30 are an hour apart, not 23 hours. Uses the circular
    standard deviation of the times mapped onto a 24-hour circle.
    """
    if len(times_min) < 2:
        return None
    angles = [t / 1440.0 * 2 * math.pi for t in times_min]
    c = sum(math.cos(a) for a in angles) / len(angles)
    s = sum(math.sin(a) for a in angles) / len(angles)
    r = min(1.0, math.hypot(c, s))
    if r < 1e-9:
        return 720.0
    return math.sqrt(-2.0 * math.log(r)) * 1440.0 / (2 * math.pi)


def duration_score(hours: float) -> float:
    if hours < 7:
        return clamp((hours - 3) / 4 * 100, 0, 100)
    if hours <= 9:
        return 100.0
    return clamp(100 - (hours - 9) * 25, 40, 100)


def consistency_score(
    bed_times: list[str], wake_times: list[str], min_nights: int = 3
) -> Optional[float]:
    """0-100 from the spread of bed and wake times (100 = within ~15 min)."""
    spreads = []
    for series in (bed_times, wake_times):
        mins = [m for m in (hhmm_to_min(t) for t in series) if m is not None]
        if len(mins) >= min_nights:
            sd = circular_std_minutes(mins)
            if sd is not None:
                spreads.append(sd)
    if not spreads:
        return None
    sd = sum(spreads) / len(spreads)
    return clamp(100 * (1 - (sd - 15) / 105), 0, 100)


def rating_score(rating: Any = None, quality: Any = None) -> Optional[float]:
    """1-5 rating (or a quality word) -> 0-100; None when there's neither."""
    if rating not in (None, ""):
        try:
            r = float(rating)
            if 1 <= r <= 5:
                return (r - 1) / 4 * 100
        except (TypeError, ValueError):
            pass
    for word in re.findall(r"[a-z]+", str(quality or "").lower()):
        if word in _QUALITY_WORDS:
            return (_QUALITY_WORDS[word] - 1) / 4 * 100
    return None


def sleep_quality_score(
    hours: Optional[float],
    rating: Any = None,
    quality: Any = None,
    bed_times: Optional[list[str]] = None,
    wake_times: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Blend duration, 7-night consistency and the user's own rating.

    Components that can't be computed are left out and the remaining
    weights are renormalised, and the result says which were missing so the
    caller can tell the user instead of quietly scoring on less.
    """
    if hours is None:
        return {"score": None, "components": {}, "missing": list(_SLEEP_WEIGHTS)}
    components: dict[str, float] = {"duration": duration_score(hours)}
    cons = consistency_score(bed_times or [], wake_times or [])
    if cons is not None:
        components["consistency"] = cons
    rate = rating_score(rating, quality)
    if rate is not None:
        components["rating"] = rate
    total_w = sum(_SLEEP_WEIGHTS[k] for k in components)
    score = sum(_SLEEP_WEIGHTS[k] * v for k, v in components.items()) / total_w
    return {
        "score": int(round(score)),
        "components": {k: int(round(v)) for k, v in components.items()},
        "missing": [k for k in _SLEEP_WEIGHTS if k not in components],
    }


# ---------------------------------------------------------------------------
# Mood scoring and correlations (#112, #126)
# ---------------------------------------------------------------------------

_MOOD_WORDS: dict[str, float] = {
    # strongly positive
    "great": 2, "amazing": 2, "fantastic": 2, "excellent": 2, "awesome": 2,
    "happy": 2, "joyful": 2, "energetic": 2, "wonderful": 2, "excited": 2,
    # mildly positive
    "good": 1, "positive": 1, "calm": 1, "content": 1, "relaxed": 1,
    "productive": 1, "motivated": 1, "hopeful": 1, "fine": 0.5, "better": 1,
    "peaceful": 1, "grateful": 1, "optimistic": 1,
    # neutral
    "okay": 0, "ok": 0, "alright": 0, "neutral": 0, "meh": 0, "average": 0,
    # mildly negative
    "tired": -1, "low": -1, "down": -1, "sad": -1, "bored": -1, "off": -1,
    "worried": -1, "irritable": -1, "lonely": -1, "gloomy": -1, "sluggish": -1,
    "unmotivated": -1, "restless": -1, "drained": -1, "worse": -1,
    # strongly negative
    "stressed": -2, "anxious": -2, "depressed": -2, "awful": -2,
    "terrible": -2, "miserable": -2, "exhausted": -2, "overwhelmed": -2,
    "angry": -2, "hopeless": -2, "burnt": -2, "burned": -2, "panicked": -2,
}
_NEGATORS = frozenset({
    "not", "no", "never", "isn't", "isnt", "wasn't", "wasnt", "don't", "dont",
    "cant", "can't", "hardly",
})
_INTENSIFIERS = frozenset({"very", "so", "really", "that", "too", "quite"})


def mood_score(text: Any) -> Optional[float]:
    """Score a free-text mood on a -2..+2 scale, or None if unrecognised.

    Deliberately a small, auditable word list rather than a model: the
    scale is only used to compare the user's own days against each other.
    Anything it can't score is left out, not guessed at.
    """
    tokens = re.findall(r"[a-z']+", str(text or "").lower())
    scores: list[float] = []
    for i, tok in enumerate(tokens):
        if tok not in _MOOD_WORDS:
            continue
        s = float(_MOOD_WORDS[tok])
        prev = tokens[i - 1] if i > 0 else ""
        prev2 = tokens[i - 2] if i > 1 else ""
        if prev in _NEGATORS or (prev in _INTENSIFIERS and prev2 in _NEGATORS):
            s = -1.0 if s > 0 else (1.0 if s < 0 else 0.0)
        scores.append(s)
    return sum(scores) / len(scores) if scores else None


def mood_valence(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 0.5:
        return "positive"
    if score <= -0.5:
        return "negative"
    return "neutral"


def daily_mood(rows: list[dict], tz: tzinfo) -> dict[date, float]:
    """Mean scorable mood per calendar day (rows with unscorable text skipped)."""
    buckets: dict[date, list[float]] = defaultdict(list)
    for row in rows:
        s = mood_score(row.get("mood"))
        d = local_date(row.get("logged_at"), tz)
        if s is not None and d is not None:
            buckets[d].append(s)
    return {d: sum(v) / len(v) for d, v in buckets.items()}


def compare_groups(
    a: list[float],
    b: list[float],
    min_n: int = 4,
    min_diff: float = 0.5,
) -> dict[str, Any]:
    """Compare mean mood of two groups of days, refusing on thin data."""
    out: dict[str, Any] = {
        "n_a": len(a), "n_b": len(b), "min_n": min_n,
        "mean_a": None, "mean_b": None, "diff": None,
    }
    if len(a) < min_n or len(b) < min_n:
        out["status"] = "insufficient"
        return out
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    out.update(mean_a=round(ma, 2), mean_b=round(mb, 2), diff=round(ma - mb, 2))
    out["status"] = "ok" if abs(ma - mb) >= min_diff else "no_difference"
    return out


def mood_by_sleep(
    sleep_by_day: dict[date, float],
    mood_by_day: dict[date, float],
    short_below: float = 6.0,
    long_from: float = 7.0,
    min_n: int = 4,
) -> dict[str, Any]:
    """Mood on days with short sleep (a) vs long sleep (b). 6-7h is skipped."""
    short, long_ = [], []
    for d, hours in sleep_by_day.items():
        if d not in mood_by_day:
            continue
        if hours < short_below:
            short.append(mood_by_day[d])
        elif hours >= long_from:
            long_.append(mood_by_day[d])
    return compare_groups(short, long_, min_n=min_n)


def mood_by_workout(
    workout_days: set[date],
    mood_by_day: dict[date, float],
    min_n: int = 4,
) -> dict[str, Any]:
    """Mood on workout days (a) vs non-workout days (b)."""
    with_w = [m for d, m in mood_by_day.items() if d in workout_days]
    without = [m for d, m in mood_by_day.items() if d not in workout_days]
    return compare_groups(with_w, without, min_n=min_n)


def habit_rate_by_mood(
    habit_days: set[date],
    since: Optional[date],
    mood_by_day: dict[date, float],
    today: date,
    high: float = 0.5,
    low: float = -0.5,
) -> dict[str, Any]:
    """How often a habit was kept on high-mood vs low-mood days.

    Only days on/after *since* (when completion dates started being
    recorded) count; earlier days are unknown, not "missed".
    """
    kept_hi = n_hi = kept_lo = n_lo = 0
    for d, m in mood_by_day.items():
        if since is None or d < since or d > today:
            continue
        if m >= high:
            n_hi += 1
            kept_hi += d in habit_days
        elif m <= low:
            n_lo += 1
            kept_lo += d in habit_days
    return {
        "n_high": n_hi, "kept_high": kept_hi,
        "n_low": n_lo, "kept_low": kept_lo,
        "rate_high": (kept_hi / n_hi) if n_hi else None,
        "rate_low": (kept_lo / n_lo) if n_lo else None,
    }


# ---------------------------------------------------------------------------
# Workout streaks (#116)
# ---------------------------------------------------------------------------


def consecutive_days(days: set[date], today: date) -> int:
    """Run of consecutive days ending today (or yesterday, so a streak isn't
    'broken' just because today's session hasn't happened yet)."""
    cursor = today if today in days else today - timedelta(days=1)
    n = 0
    while cursor in days:
        n += 1
        cursor -= timedelta(days=1)
    return n


def consecutive_weeks(sessions: list[date], today: date, min_sessions: int = 1) -> int:
    """Run of consecutive ISO weeks with >= *min_sessions* sessions.

    The current week is in progress, so if it hasn't met the bar yet it is
    skipped rather than counted as a break.
    """
    counts = Counter(week_start(d) for d in sessions)
    cursor = week_start(today)
    if counts[cursor] < min_sessions:
        cursor -= timedelta(days=7)
    n = 0
    while counts[cursor] >= min_sessions:
        n += 1
        cursor -= timedelta(days=7)
    return n


def type_streaks(
    workouts: list[dict], tz: tzinfo, today: date, min_sessions: int = 2
) -> dict[str, dict[str, Any]]:
    """Per exercise type: consecutive days and consecutive qualifying weeks."""
    by_type: dict[str, list[date]] = defaultdict(list)
    for w in workouts:
        d = local_date(w.get("logged_at"), tz)
        if d is None:
            continue
        kind = (w.get("type") or "general").strip().lower() or "general"
        by_type[kind].append(d)
    out: dict[str, dict[str, Any]] = {}
    for kind, dates in by_type.items():
        out[kind] = {
            "days": consecutive_days(set(dates), today),
            "weeks": consecutive_weeks(dates, today, min_sessions),
            "sessions": len(dates),
            "last": max(dates).isoformat(),
        }
    return out


# ---------------------------------------------------------------------------
# Pain / injury (#117)
# ---------------------------------------------------------------------------

_RED_FLAGS: tuple[tuple[str, str], ...] = (
    (r"\bchest (pain|pressure|tightness)\b", "chest pain or pressure"),
    (r"\b(can'?t|cannot|hard to|difficulty|trouble) breath", "trouble breathing"),
    (r"\bshort(ness)? of breath\b", "shortness of breath"),
    (r"\bnumb(ness)?\b|\bcan'?t feel\b", "numbness or loss of feeling"),
    (r"\b(can'?t|unable to|cannot) (move|walk|bear weight|lift)\b", "loss of movement"),
    (r"\b(faint(ed|ing)?|passed out|blacked out|blackout)\b", "fainting"),
    (r"\bsudden(ly)? (and )?(severe|intense|terrible)\b", "sudden severe pain"),
    (r"\bworst (pain|headache)\b", "the worst pain you've had"),
    (r"\b(hit|hurt|injured) (my )?head\b|\bhead injury\b", "a head injury"),
    (r"\b(blood in|coughing (up )?blood|vomiting blood)\b", "bleeding"),
)

_BODY_AREAS = (
    "head", "neck", "shoulder", "back", "lower back", "upper back", "chest",
    "stomach", "abdomen", "hip", "knee", "ankle", "foot", "heel", "wrist",
    "elbow", "hand", "arm", "leg", "thigh", "calf", "shin", "groin", "jaw",
    "tooth", "ear", "eye", "hamstring", "quad", "glute", "rib",
)


def red_flags(text: Any) -> list[str]:
    """Phrases that should prompt a 'see a clinician' message."""
    lowered = str(text or "").lower()
    return [label for pattern, label in _RED_FLAGS if re.search(pattern, lowered)]


def guess_body_area(text: Any) -> Optional[str]:
    lowered = str(text or "").lower()
    for area in sorted(_BODY_AREAS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(area)}s?\b", lowered):
            return area
    return None


def pain_summary(
    rows: list[dict], tz: tzinfo, today: date
) -> list[dict[str, Any]]:
    """Per-area trend: last 7 days vs the 7 before, once there's enough data."""
    by_area: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for r in rows:
        d = local_date(r.get("logged_at"), tz)
        if d is None or r.get("severity") is None:
            continue
        area = (r.get("area") or "unspecified").strip().lower()
        by_area[area].append((d, float(r["severity"])))

    out: list[dict[str, Any]] = []
    for area, entries in by_area.items():
        entries.sort()
        recent = [s for d, s in entries if (today - d).days < 7]
        prior = [s for d, s in entries if 7 <= (today - d).days < 14]
        if len(recent) >= 2 and len(prior) >= 2:
            diff = sum(recent) / len(recent) - sum(prior) / len(prior)
            trend = "worse" if diff >= 1 else "better" if diff <= -1 else "steady"
        else:
            diff, trend = None, "not_enough_data"
        last28 = [(d, s) for d, s in entries if (today - d).days < 28]
        persisting = (
            len(last28) >= 3
            and (today - min(d for d, _ in last28)).days >= 21
        )
        out.append({
            "area": area,
            "n": len(entries),
            "latest": entries[-1][1],
            "latest_date": entries[-1][0].isoformat(),
            "avg_recent": round(sum(recent) / len(recent), 1) if recent else None,
            "avg_prior": round(sum(prior) / len(prior), 1) if prior else None,
            "diff": round(diff, 1) if diff is not None else None,
            "trend": trend,
            "persisting_weeks": persisting,
        })
    out.sort(key=lambda x: (-x["latest"], x["area"]))
    return out


def clinician_advice(summary: dict[str, Any]) -> Optional[str]:
    """A 'see a clinician' nudge for a single area's summary, or None."""
    area = summary["area"]
    if summary["latest"] >= 9:
        return (
            f"Pain that severe in your {area} is worth getting looked at "
            "promptly — please contact a doctor or urgent care."
        )
    if summary["persisting_weeks"]:
        return (
            f"Your {area} pain has been logged for three weeks or more. "
            "Pain that lasts that long is worth having a clinician look at."
        )
    if summary["trend"] == "worse" and summary["latest"] >= 6:
        return (
            f"Your {area} pain is trending up and is at {summary['latest']:g}/10. "
            "It would be sensible to have a clinician check it."
        )
    return None


# ---------------------------------------------------------------------------
# Goal pace (#120)
# ---------------------------------------------------------------------------


def linear_slope(points: list[tuple[float, float]]) -> Optional[float]:
    """Least-squares slope of y over x (units of y per unit of x)."""
    if len(points) < 2:
        return None
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    den = sum((x - mx) ** 2 for x, _ in points)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in points) / den


def goal_pace(
    current: float,
    target: float,
    slope_per_day: Optional[float],
    start: Optional[float] = None,
    days_left: Optional[int] = None,
    reached_tolerance: float = 0.1,
) -> dict[str, Any]:
    """Distance to target, rate, ETA and on/behind pace.

    ``slope_per_day`` is the recent trend in the metric's own units (e.g.
    kg/day). ``days_left`` comes from an optional deadline. Status is one of
    reached, on_pace, behind, moving_away, no_trend, or steady_no_deadline.
    """
    remaining = target - current
    out: dict[str, Any] = {
        "current": current, "target": target, "start": start,
        "remaining": remaining, "rate_per_day": slope_per_day,
        "eta_days": None, "days_left": days_left, "required_per_day": None,
        "progress_pct": None,
    }
    if start is not None and start != target:
        out["progress_pct"] = round(
            clamp((current - start) / (target - start) * 100, 0, 100), 0
        )
    if abs(remaining) <= reached_tolerance:
        out["status"] = "reached"
        return out
    if days_left is not None and days_left > 0:
        out["required_per_day"] = remaining / days_left
    if slope_per_day is None:
        out["status"] = "no_trend"
        return out
    toward = slope_per_day * remaining > 0
    if abs(slope_per_day) < 1e-9:
        out["status"] = "no_trend"
        return out
    if not toward:
        out["status"] = "moving_away"
        return out
    out["eta_days"] = int(math.ceil(remaining / slope_per_day))
    if days_left is None:
        out["status"] = "steady_no_deadline"
    elif days_left <= 0:
        out["status"] = "behind"
    else:
        out["status"] = "on_pace" if out["eta_days"] <= days_left else "behind"
    return out


# ---------------------------------------------------------------------------
# Spending stress proxy (#161)
# ---------------------------------------------------------------------------


def weekly_spend_spike(
    expenses: list[dict], today: date, tz: tzinfo,
    ratio_threshold: float = 1.5, min_baseline_weeks: int = 3,
) -> Optional[dict[str, Any]]:
    """Last 7 days of spend vs the average of the 4 weeks before.

    Returns None when there isn't enough history to call anything a spike.
    A weak signal by design: spending rises for many ordinary reasons.
    """
    weeks: dict[int, float] = defaultdict(float)
    for e in expenses:
        d = local_date(e.get("logged_at"), tz)
        try:
            amount = float(e.get("amount") or 0)
        except (TypeError, ValueError):
            continue
        if d is None or d > today:
            continue
        idx = (today - d).days // 7  # 0 = the last 7 days
        if idx <= 4:
            weeks[idx] += amount
    baseline_weeks = [weeks[i] for i in range(1, 5) if weeks.get(i, 0) > 0]
    if len(baseline_weeks) < min_baseline_weeks:
        return None
    baseline = sum(baseline_weeks) / len(baseline_weeks)
    if baseline <= 0:
        return None
    recent = weeks.get(0, 0.0)
    ratio = recent / baseline
    return {
        "recent": round(recent, 2), "baseline": round(baseline, 2),
        "ratio": round(ratio, 2), "spike": ratio >= ratio_threshold,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

KG_PER_LB = 0.45359237


def fmt_weight(kg: float, unit: str) -> str:
    return f"{kg / KG_PER_LB:.1f} lb" if unit == "lb" else f"{kg:.1f} kg"


def build_weekly_summary(d: dict[str, Any]) -> str:
    """Deterministic weekly summary text from pre-computed numbers.

    No LLM: this is sent unprompted, so it has to be reproducible and must
    never editorialise about weight or food.
    """
    unit = d.get("weight_unit", "kg")
    lines = ["Your week in health (last 7 days):"]

    if d.get("sleep_n"):
        line = f"- Sleep: {d['sleep_avg']:.1f} h/night on average over {d['sleep_n']} logged night(s)"
        if d.get("sleep_score_avg") is not None:
            line += f", average sleep score {d['sleep_score_avg']:.0f}/100"
        lines.append(line + ".")
    else:
        lines.append("- Sleep: nothing logged.")

    if d.get("workouts"):
        lines.append(
            f"- Workouts: {d['workouts']} session(s) across {d['active_days']} day(s)."
        )
    else:
        lines.append("- Workouts: none logged.")

    if d.get("weight_n", 0) >= 2:
        delta = d["weight_last"] - d["weight_first"]
        sign = "+" if delta > 0 else "-" if delta < 0 else ""
        shown = abs(delta / KG_PER_LB) if unit == "lb" else abs(delta)
        line = (
            f"- Weight: {fmt_weight(d['weight_first'], unit)} to "
            f"{fmt_weight(d['weight_last'], unit)} "
            f"({sign}{shown:.1f} {unit}) over {d['weight_n']} logs."
        )
        if abs(delta) > 1.0:
            line += (
                " That's a fairly quick change; if it isn't intended, "
                "it's worth mentioning to a clinician."
            )
        lines.append(line)
    elif d.get("weight_n") == 1:
        lines.append(f"- Weight: one log ({fmt_weight(d['weight_last'], unit)}).")
    else:
        lines.append("- Weight: nothing logged.")

    if d.get("water_days_logged"):
        lines.append(
            f"- Water: goal of {d['water_goal_ml']:.0f} ml met on "
            f"{d['water_days_met']} of {d['water_days_logged']} day(s) you logged."
        )
    else:
        lines.append("- Water: nothing logged.")

    if d.get("steps_days"):
        lines.append(
            f"- Steps: {d['steps_avg']:.0f}/day on average over {d['steps_days']} day(s) of imported data."
        )
    if d.get("mood_n"):
        lines.append(
            f"- Mood: {d['mood_n']} entr{'y' if d['mood_n'] == 1 else 'ies'} logged"
            + (f", mostly {d['mood_label']}." if d.get("mood_label") else ".")
        )
    return "\n".join(lines)
