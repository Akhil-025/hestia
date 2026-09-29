"""
modules/chronos/recurrence.py

Pure (no I/O, no DB, no network) helpers for Chronos's recurring reminders.

Covers three backlog items that share one data model:

  #81  Recurring reminders (daily / weekly / monthly / yearly / every-N-units
       / cron-like), not just one-shot.
  #84  Natural-language recurring-rule parsing ("every weekday at 7am").
  #85  Per-reminder timezone override ("at 9am London time", "9am EST").

Design
------
A rule is a small frozen ``Recurrence`` dataclass, serialised to JSON for
storage in the ``reminders.recurrence`` column and convertible to/from an
iCalendar ``RRULE`` (used by the ICS export/import, backlog #90).

``next_after(rule, after, tz)`` is the single scheduling primitive: given an
aware datetime it returns the first occurrence strictly after it, computed on
the *local wall clock of the rule's timezone* so "every day at 7am" keeps
firing at 7am local across a DST change instead of drifting by an hour.

Everything here is deterministic given its inputs, which is what makes it
cheap to test exhaustively.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREQ_MINUTELY = "minutely"
FREQ_HOURLY = "hourly"
FREQ_DAILY = "daily"
FREQ_WEEKLY = "weekly"
FREQ_MONTHLY = "monthly"
FREQ_YEARLY = "yearly"
FREQ_CRON = "cron"

_VALID_FREQS = frozenset({
    FREQ_MINUTELY, FREQ_HOURLY, FREQ_DAILY, FREQ_WEEKLY,
    FREQ_MONTHLY, FREQ_YEARLY, FREQ_CRON,
})

# Upper bound on how far ahead we search for the next occurrence. Five years
# covers a Feb-29 yearly rule (worst realistic case) with room to spare, and
# bounds the loop so a pathological cron expression can never spin forever.
_MAX_SEARCH_DAYS = 366 * 5

# Sanity caps so a typo ("every 100000 days") can't produce a nonsense rule.
_MAX_INTERVAL = 1000

DEFAULT_HOUR = 9      # matches chronos.engine._TIME_OF_DAY["morning"]
DEFAULT_MINUTE = 0

_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
              "Saturday", "Sunday")
_ICS_DAYS = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")

_WEEKDAY_LOOKUP: dict[str, int] = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_WEEKDAY_WORD = (
    r"(?:monday|mon|tuesday|tues|tue|wednesday|wed|thursday|thurs|thur|thu"
    r"|friday|fri|saturday|sat|sunday|sun)s?\b"
)

_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "sixty": 60,
}

# "every morning" etc. -> (hour, minute). Kept in step with
# chronos.engine._TIME_OF_DAY (the one-shot equivalent).
_PART_OF_DAY: dict[str, tuple[int, int]] = {
    "morning": (9, 0),
    "afternoon": (14, 0),
    "evening": (18, 0),
    "night": (21, 0),
}

_UNIT_TO_FREQ: dict[str, str] = {
    "minute": FREQ_MINUTELY, "min": FREQ_MINUTELY,
    "hour": FREQ_HOURLY, "hr": FREQ_HOURLY,
    "day": FREQ_DAILY,
    "week": FREQ_WEEKLY,
    "month": FREQ_MONTHLY,
    "year": FREQ_YEARLY,
}


# ---------------------------------------------------------------------------
# Rule model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recurrence:
    """
    A recurrence rule.

    Attributes
    ----------
    freq:
        One of the ``FREQ_*`` constants.
    interval:
        Every *interval* units of ``freq`` (>= 1). Ignored for ``cron``.
    byday:
        Weekday numbers, 0 = Monday .. 6 = Sunday. For ``weekly`` the set of
        weekdays to fire on (empty = the anchor's weekday). For ``daily`` an
        optional filter ("every weekday" is daily-or-weekly restricted to
        Mon-Fri; it is stored as weekly + byday).
    bymonthday:
        Day of month for ``monthly`` rules (clamped to the month's length,
        so "the 31st" fires on the 30th/28th in shorter months).
    hour, minute:
        Local wall-clock time of day. Unused for minutely/hourly.
    cron:
        The 5-field cron expression when ``freq == "cron"``.
    anchor:
        ISO datetime (UTC) of the first occurrence. Needed to make
        ``interval > 1`` well defined ("every 2 weeks" - which weeks?) and
        as the phase reference for minutely/hourly rules.
    until:
        Inclusive ISO date (YYYY-MM-DD, in the rule's timezone) after which
        the rule stops. ``None`` = forever.
    count:
        Remaining number of occurrences including the next one. ``None`` =
        unlimited. Decremented by ``consume()`` each time one fires.
    """

    freq: str
    interval: int = 1
    byday: tuple[int, ...] = ()
    bymonthday: Optional[int] = None
    hour: int = DEFAULT_HOUR
    minute: int = DEFAULT_MINUTE
    cron: Optional[str] = None
    anchor: Optional[str] = None
    until: Optional[str] = None
    count: Optional[int] = None

    # -- validation ------------------------------------------------------

    def __post_init__(self) -> None:
        if self.freq not in _VALID_FREQS:
            raise ValueError(f"Unknown recurrence freq {self.freq!r}.")
        if not (1 <= self.interval <= _MAX_INTERVAL):
            raise ValueError(f"interval must be 1..{_MAX_INTERVAL}, got {self.interval}.")
        if not (0 <= self.hour <= 23 and 0 <= self.minute <= 59):
            raise ValueError(f"Invalid time of day {self.hour}:{self.minute}.")
        if any(not (0 <= d <= 6) for d in self.byday):
            raise ValueError(f"byday entries must be 0..6, got {self.byday!r}.")
        if self.bymonthday is not None and not (1 <= self.bymonthday <= 31):
            raise ValueError(f"bymonthday must be 1..31, got {self.bymonthday}.")
        if self.freq == FREQ_CRON:
            _parse_cron(self.cron or "")  # raises ValueError if malformed
        if self.count is not None and self.count < 0:
            raise ValueError("count must be >= 0.")

    # -- serialisation ---------------------------------------------------

    def to_json(self) -> str:
        payload: dict[str, Any] = {"freq": self.freq}
        if self.interval != 1:
            payload["interval"] = self.interval
        if self.byday:
            payload["byday"] = list(self.byday)
        if self.bymonthday is not None:
            payload["bymonthday"] = self.bymonthday
        if self.freq not in (FREQ_MINUTELY, FREQ_HOURLY, FREQ_CRON):
            payload["hour"] = self.hour
            payload["minute"] = self.minute
        if self.cron:
            payload["cron"] = self.cron
        if self.anchor:
            payload["anchor"] = self.anchor
        if self.until:
            payload["until"] = self.until
        if self.count is not None:
            payload["count"] = self.count
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: Optional[str]) -> Optional["Recurrence"]:
        """Inverse of ``to_json``. Returns ``None`` for empty/garbage input
        rather than raising - a corrupt stored rule must never take down the
        scheduler loop."""
        if not raw:
            return None
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None
            return cls(
                freq=data["freq"],
                interval=int(data.get("interval", 1)),
                byday=tuple(int(d) for d in data.get("byday", ())),
                bymonthday=data.get("bymonthday"),
                hour=int(data.get("hour", DEFAULT_HOUR)),
                minute=int(data.get("minute", DEFAULT_MINUTE)),
                cron=data.get("cron"),
                anchor=data.get("anchor"),
                until=data.get("until"),
                count=data.get("count"),
            )
        except (ValueError, KeyError, TypeError):
            return None

    # -- helpers ---------------------------------------------------------

    def with_anchor(self, first_occurrence: datetime) -> "Recurrence":
        """Return a copy anchored at *first_occurrence* (aware datetime)."""
        return replace(self, anchor=first_occurrence.astimezone(timezone.utc).isoformat())

    def consume(self) -> Optional["Recurrence"]:
        """
        Record that one occurrence has fired. Returns the updated rule, or
        ``None`` if that was the last one (count exhausted).
        """
        if self.count is None:
            return self
        remaining = self.count - 1
        if remaining <= 0:
            return None
        return replace(self, count=remaining)

    def describe(self, tz: Any = None) -> str:
        """Human/TTS-friendly description, e.g. "every weekday at 7:00 AM".

        ``tz`` is only needed for yearly rules, whose month and day live in
        the anchor (stored in UTC) and must be read back in the reminder's own
        zone. Callers add any time-zone note themselves.
        """
        text = _describe_core(self, tz)
        if self.until:
            text += f" until {self.until}"
        if self.count is not None:
            text += f" ({self.count} more time{'s' if self.count != 1 else ''})"
        return text


# ---------------------------------------------------------------------------
# Time-zone helpers  (#85)
# ---------------------------------------------------------------------------

# City / region -> IANA zone. Deliberately mirrors the cities in
# chronos.engine._CITY_COORDS so anywhere the user can ask for weather they
# can also pin a reminder's time zone, plus a few common travel hubs.
_CITY_TZ: dict[str, str] = {
    "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata", "bengaluru": "Asia/Kolkata",
    "chennai": "Asia/Kolkata", "kolkata": "Asia/Kolkata",
    "hyderabad": "Asia/Kolkata", "pune": "Asia/Kolkata", "india": "Asia/Kolkata",
    "london": "Europe/London", "uk": "Europe/London",
    "paris": "Europe/Paris", "berlin": "Europe/Berlin", "madrid": "Europe/Madrid",
    "rome": "Europe/Rome", "amsterdam": "Europe/Amsterdam", "zurich": "Europe/Zurich",
    "moscow": "Europe/Moscow", "istanbul": "Europe/Istanbul",
    "new york": "America/New_York", "nyc": "America/New_York",
    "boston": "America/New_York", "toronto": "America/Toronto",
    "chicago": "America/Chicago", "denver": "America/Denver",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles",
    "vancouver": "America/Vancouver", "sao paulo": "America/Sao_Paulo",
    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo", "seoul": "Asia/Seoul",
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong", "singapore": "Asia/Singapore",
    "bangkok": "Asia/Bangkok", "dubai": "Asia/Dubai", "abu dhabi": "Asia/Dubai",
    "karachi": "Asia/Karachi", "dhaka": "Asia/Dhaka", "kathmandu": "Asia/Kathmandu",
    "colombo": "Asia/Colombo",
    "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "perth": "Australia/Perth", "auckland": "Pacific/Auckland",
    "johannesburg": "Africa/Johannesburg", "cairo": "Africa/Cairo",
    "nairobi": "Africa/Nairobi", "lagos": "Africa/Lagos",
}

# Abbreviation -> IANA zone. Abbreviations are inherently ambiguous (CST is
# US Central *and* China Standard); we pick the interpretation that matches
# the user's likely context and, more importantly, always pick a zone that
# observes DST for the "S/D" pairs so "EST" in July isn't silently an hour off.
_TZ_ABBREVIATIONS: dict[str, str] = {
    "ist": "Asia/Kolkata",
    "utc": "UTC", "gmt": "UTC", "zulu": "UTC",
    "bst": "Europe/London",
    "est": "America/New_York", "edt": "America/New_York", "et": "America/New_York",
    "cst": "America/Chicago", "cdt": "America/Chicago", "ct": "America/Chicago",
    "mst": "America/Denver", "mdt": "America/Denver", "mt": "America/Denver",
    "pst": "America/Los_Angeles", "pdt": "America/Los_Angeles",
    "pt": "America/Los_Angeles",
    "cet": "Europe/Paris", "cest": "Europe/Paris",
    "eet": "Europe/Athens", "eest": "Europe/Athens",
    "jst": "Asia/Tokyo", "kst": "Asia/Seoul",
    "sgt": "Asia/Singapore", "hkt": "Asia/Hong_Kong",
    "gst": "Asia/Dubai", "pkt": "Asia/Karachi",
    "aest": "Australia/Sydney", "aedt": "Australia/Sydney",
    "acst": "Australia/Adelaide", "awst": "Australia/Perth",
    "nzst": "Pacific/Auckland", "nzdt": "Pacific/Auckland",
}

_IANA_RE = re.compile(r"\b([A-Za-z]+(?:/[A-Za-z_\-]+){1,2})\b")


def resolve_timezone_name(name: Optional[str]) -> Optional[str]:
    """
    Resolve a user-supplied zone *name* to a valid IANA zone string.

    Accepts an IANA name (``"Europe/Berlin"``), an abbreviation (``"EST"``)
    or a city (``"Tokyo"``). Returns ``None`` if it can't be resolved -
    callers must treat that as "no override", never guess.
    """
    if not name:
        return None
    cleaned = str(name).strip()
    if not cleaned or cleaned.lower() in {"none", "null"}:
        return None
    key = cleaned.lower()
    # Strip a trailing "time"/"timezone"/"time zone" ("London time" -> "london").
    key = re.sub(r"\s*(?:time\s*zone|timezone|time)$", "", key).strip()
    if key in _TZ_ABBREVIATIONS:
        return _TZ_ABBREVIATIONS[key]
    if key in _CITY_TZ:
        return _CITY_TZ[key]
    # IANA names are case-sensitive on some platforms; try the given spelling
    # then a Title_Case normalisation ("europe/london" -> "Europe/London").
    for candidate in (cleaned, "/".join(p.title() for p in cleaned.split("/"))):
        if "/" not in candidate and candidate.upper() != "UTC":
            continue
        try:
            ZoneInfo(candidate)
            return candidate
        except (ZoneInfoNotFoundError, KeyError, ValueError):
            continue
    return None


# Matches "London time", "in Tokyo time", "9am EST", "PST", "Europe/Berlin".
_TZ_PHRASE_RE = re.compile(
    r"\b(?:(?:in|at|for)\s+)?"
    r"(?P<city>[A-Za-z][A-Za-z ]{1,20}?)\s+time(?:\s*zone)?\b",
    re.IGNORECASE,
)
_TZ_ABBR_RE = re.compile(
    r"\b(?P<abbr>" + "|".join(sorted(_TZ_ABBREVIATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def extract_timezone(text: str) -> tuple[Optional[str], str]:
    """
    Find a per-reminder timezone in *text*.

    Returns ``(iana_or_None, text_with_the_tz_phrase_removed)``. The removal
    matters: the leftover text feeds task extraction and date parsing, and a
    stray "London time" would otherwise end up inside the task label.

    Only phrases that resolve to a real zone are consumed; "quality time" or
    "free time" resolve to nothing and are left untouched.
    """
    if not text:
        return None, text or ""

    # 1. Explicit IANA name.
    m = _IANA_RE.search(text)
    if m:
        resolved = resolve_timezone_name(m.group(1))
        if resolved:
            return resolved, _squash(text[: m.start()] + text[m.end():])

    # 2. "<city> time" / "<city> time zone". Try the longest suffix of the
    #    captured words first so "call john at 3pm new york time" resolves
    #    "new york" rather than failing on "3pm new york".
    for m in _TZ_PHRASE_RE.finditer(text):
        words = m.group("city").split()
        city_start = m.start("city")
        for i in range(len(words)):
            candidate = " ".join(words[i:])
            resolved = resolve_timezone_name(candidate)
            if not resolved:
                continue
            # Offset of the first kept city word inside the original text.
            span_start = city_start + len(m.group("city")) - len(candidate)
            # Words before the city are task text; keep them. Drop a
            # directly-preceding "in"/"at"/"for" along with the city.
            head = text[:span_start].rstrip()
            head = re.sub(r"\s+(?:in|at|for)$", "", head, flags=re.IGNORECASE) \
                if i > 0 or m.start() < city_start else head
            return resolved, _squash(head + " " + text[m.end():])

    # 3. Bare abbreviation, but only when it sits next to a clock time, so
    #    the pronoun-like "et"/"pt"/"mt" or the word "gst" in prose isn't
    #    mistaken for a zone.
    for m in _TZ_ABBR_RE.finditer(text):
        before = text[: m.start()].rstrip()
        after = text[m.end():].lstrip()
        near_clock = bool(
            re.search(r"(?:\d|am|pm|noon|midnight)$", before, re.IGNORECASE)
            or re.match(r"^(?:tomorrow|today|tonight|on\b|\d)", after, re.IGNORECASE)
        )
        if near_clock:
            return _TZ_ABBREVIATIONS[m.group("abbr").lower()], _squash(
                text[: m.start()] + text[m.end():]
            )
    return None, text


def _squash(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


def to_zoneinfo(name: Optional[str], default: Any) -> Any:
    """ZoneInfo for *name*, or *default* if it's empty/invalid."""
    if not name:
        return default
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Cron  (#81 "cron-like")
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CronSpec:
    minutes: frozenset[int]
    hours: frozenset[int]
    doms: frozenset[int]
    months: frozenset[int]
    dows: frozenset[int]          # 0 = Monday .. 6 = Sunday (Python's weekday())
    dom_star: bool
    dow_star: bool


_CRON_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_CRON_DOWS = {d: i for i, d in enumerate(("mon", "tue", "wed", "thu", "fri", "sat", "sun"))}


def _cron_field(field: str, lo: int, hi: int, names: Optional[dict[str, int]] = None) -> frozenset[int]:
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip().lower()
        if not part:
            raise ValueError(f"Empty item in cron field {field!r}.")
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            if not step_s.isdigit() or int(step_s) < 1:
                raise ValueError(f"Bad step in cron field {field!r}.")
            step = int(step_s)
        if part in ("*", "?"):
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = _cron_num(a, names), _cron_num(b, names)
        else:
            start = _cron_num(part, names)
            end = hi if step > 1 else start
        if start < lo or end > hi or start > end:
            raise ValueError(f"Cron value out of range in {field!r} (allowed {lo}-{hi}).")
        values.update(range(start, end + 1, step))
    return frozenset(values)


def _cron_num(token: str, names: Optional[dict[str, int]]) -> int:
    token = token.strip().lower()
    if names and token in names:
        return names[token]
    if not token.isdigit():
        raise ValueError(f"Bad cron token {token!r}.")
    return int(token)


def _parse_cron(expr: str) -> _CronSpec:
    parts = (expr or "").split()
    if len(parts) != 5:
        raise ValueError(f"Cron expression needs 5 fields, got {len(parts)}: {expr!r}.")
    minute_f, hour_f, dom_f, month_f, dow_f = parts
    # Standard cron numbers Sunday as 0 (or 7); Python's weekday() has Monday=0.
    # Translate cron numerics into Python numbering; names are already Python-based.
    dow_f_norm = _cron_dow_numbers_to_python(dow_f)
    return _CronSpec(
        minutes=_cron_field(minute_f, 0, 59),
        hours=_cron_field(hour_f, 0, 23),
        doms=_cron_field(dom_f, 1, 31),
        months=_cron_field(month_f, 1, 12, _CRON_MONTHS),
        dows=_cron_field(dow_f_norm, 0, 6, _CRON_DOWS),
        dom_star=dom_f.strip() in ("*", "?"),
        dow_star=dow_f.strip() in ("*", "?"),
    )


def _cron_dow_numbers_to_python(field: str) -> str:
    """Rewrite numeric cron weekdays (0/7 = Sun, 1 = Mon .. 6 = Sat) to
    Python numbering (0 = Mon .. 6 = Sun), leaving names/steps/stars alone."""
    def convert_token(tok: str) -> str:
        return str((int(tok) - 1) % 7) if tok.isdigit() else tok

    out_parts = []
    for part in field.split(","):
        step = ""
        if "/" in part:
            part, step = part.split("/", 1)
            step = "/" + step
        if "-" in part:
            a, b = part.split("-", 1)
            # "5-7" (Fri..Sun) and "0-6" style ranges: expand explicitly so the
            # 7 -> Sunday wrap doesn't produce an inverted range.
            if a.isdigit() and b.isdigit():
                lo, hi = int(a), int(b)
                if lo > hi:
                    raise ValueError(f"Inverted cron weekday range {part!r}.")
                days = sorted({convert_token(str(n)) for n in range(lo, hi + 1)}, key=int)
                if step:
                    # Steps over an expanded set: apply to the original numeric run.
                    s = int(step[1:])
                    days = sorted({convert_token(str(n)) for n in range(lo, hi + 1, s)}, key=int)
                out_parts.append(",".join(days))
                continue
            out_parts.append(f"{convert_token(a)}-{convert_token(b)}{step}")
        else:
            out_parts.append(convert_token(part) + step)
    return ",".join(out_parts)


def parse_cron(expr: str, *, anchor: Optional[datetime] = None) -> Recurrence:
    """Build a ``cron`` Recurrence from a 5-field expression (validated)."""
    rule = Recurrence(freq=FREQ_CRON, cron=" ".join(expr.split()))
    return rule.with_anchor(anchor) if anchor else rule


# ---------------------------------------------------------------------------
# Next-occurrence engine
# ---------------------------------------------------------------------------

def _month_index(d: date) -> int:
    return d.year * 12 + (d.month - 1)


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def _anchor_local(rule: Recurrence, tz: Any) -> Optional[datetime]:
    if not rule.anchor:
        return None
    try:
        dt = datetime.fromisoformat(rule.anchor)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz)


def _date_matches(rule: Recurrence, d: date, anchor_d: Optional[date]) -> bool:
    """Does calendar date *d* satisfy the rule's date constraints?"""
    if rule.freq == FREQ_DAILY:
        if rule.byday and d.weekday() not in rule.byday:
            return False
        if rule.interval > 1 and anchor_d is not None:
            return (d - anchor_d).days % rule.interval == 0
        return True

    if rule.freq == FREQ_WEEKLY:
        days = rule.byday or ((anchor_d.weekday(),) if anchor_d else ())
        if not days:
            days = tuple(range(7))
        if d.weekday() not in days:
            return False
        if rule.interval > 1 and anchor_d is not None:
            week_a = anchor_d - timedelta(days=anchor_d.weekday())
            week_d = d - timedelta(days=d.weekday())
            return ((week_d - week_a).days // 7) % rule.interval == 0
        return True

    if rule.freq == FREQ_MONTHLY:
        wanted = rule.bymonthday or (anchor_d.day if anchor_d else 1)
        if d.day != min(wanted, _last_day_of_month(d.year, d.month)):
            return False
        if rule.interval > 1 and anchor_d is not None:
            return (_month_index(d) - _month_index(anchor_d)) % rule.interval == 0
        return True

    if rule.freq == FREQ_YEARLY:
        if anchor_d is None:
            return False
        if d.month != anchor_d.month:
            return False
        if d.day != min(anchor_d.day, _last_day_of_month(d.year, d.month)):
            return False
        return (d.year - anchor_d.year) % rule.interval == 0

    return False


def _cron_date_matches(spec: _CronSpec, d: date) -> bool:
    if d.month not in spec.months:
        return False
    dom_ok = d.day in spec.doms
    dow_ok = d.weekday() in spec.dows
    # Vixie-cron semantics: when both day fields are restricted the day
    # matches if EITHER does; when only one is restricted, that one decides.
    if not spec.dom_star and not spec.dow_star:
        return dom_ok or dow_ok
    if not spec.dom_star:
        return dom_ok
    if not spec.dow_star:
        return dow_ok
    return True


def _combine(d: date, hour: int, minute: int, tz: Any) -> datetime:
    """
    Aware datetime for local wall-clock *d hour:minute* in *tz*.

    A time that doesn't exist (spring-forward gap) is nudged forward by
    round-tripping through UTC, which is what most calendar apps do.
    """
    naive = datetime(d.year, d.month, d.day, hour, minute)
    aware = naive.replace(tzinfo=tz)
    return aware.astimezone(timezone.utc).astimezone(tz)


def next_after(rule: Recurrence, after: datetime, tz: Any) -> Optional[datetime]:
    """
    First occurrence of *rule* strictly after *after*, as an aware datetime in
    *tz*, or ``None`` if the rule has ended (``until`` passed / count spent /
    nothing within the search horizon).
    """
    if after.tzinfo is None:
        after = after.replace(tzinfo=tz)
    after_local = after.astimezone(tz)
    after_utc = after.astimezone(timezone.utc)

    if rule.count is not None and rule.count <= 0:
        return None

    until_date: Optional[date] = None
    if rule.until:
        try:
            until_date = date.fromisoformat(rule.until)
        except ValueError:
            until_date = None

    def within_until(dt: datetime) -> bool:
        return until_date is None or dt.astimezone(tz).date() <= until_date

    anchor_local = _anchor_local(rule, tz)

    # ---- sub-daily: stepping from the anchor keeps a stable phase ----------
    if rule.freq in (FREQ_MINUTELY, FREQ_HOURLY):
        step = timedelta(
            minutes=rule.interval if rule.freq == FREQ_MINUTELY else 0,
            hours=rule.interval if rule.freq == FREQ_HOURLY else 0,
        )
        base = (anchor_local or after_local).astimezone(timezone.utc)
        if after_utc < base:
            candidate = base
        else:
            n = int((after_utc - base) / step) + 1
            candidate = base + n * step
        candidate = candidate.astimezone(tz)
        return candidate if within_until(candidate) else None

    # ---- cron ---------------------------------------------------------------
    if rule.freq == FREQ_CRON:
        spec = _parse_cron(rule.cron or "")
        hours = sorted(spec.hours)
        minutes = sorted(spec.minutes)
        d = after_local.date()
        for _ in range(_MAX_SEARCH_DAYS):
            if _cron_date_matches(spec, d):
                for h in hours:
                    for mi in minutes:
                        cand = _combine(d, h, mi, tz)
                        if cand.astimezone(timezone.utc) > after_utc:
                            return cand if within_until(cand) else None
            d += timedelta(days=1)
            if until_date is not None and d > until_date:
                return None
        return None

    # ---- calendar-based rules ----------------------------------------------
    anchor_d = anchor_local.date() if anchor_local else None
    d = after_local.date()
    for _ in range(_MAX_SEARCH_DAYS):
        if _date_matches(rule, d, anchor_d):
            cand = _combine(d, rule.hour, rule.minute, tz)
            if cand.astimezone(timezone.utc) > after_utc:
                return cand if within_until(cand) else None
        d += timedelta(days=1)
        if until_date is not None and d > until_date:
            return None
    return None


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------

def _fmt_time(hour: int, minute: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    h12 = hour % 12 or 12
    return f"{h12}:{minute:02d} {suffix}"


def _describe_days(days: tuple[int, ...]) -> str:
    s = tuple(sorted(set(days)))
    if s == (0, 1, 2, 3, 4):
        return "weekday"
    if s == (5, 6):
        return "weekend day"
    if s == tuple(range(7)):
        return "day"
    return ", ".join(_DAY_NAMES[d] for d in s)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _anchor_date_for_describe(rule: Recurrence, tz: Any = None) -> Optional[date]:
    """Calendar date a yearly rule repeats on: its anchor, read in *tz*
    (UTC when no zone is given)."""
    if not rule.anchor:
        return None
    try:
        dt = datetime.fromisoformat(rule.anchor)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz if tz is not None else timezone.utc).date()


def _describe_core(rule: Recurrence, tz: Any = None) -> str:
    n = rule.interval
    if rule.freq == FREQ_MINUTELY:
        return "every minute" if n == 1 else f"every {n} minutes"
    if rule.freq == FREQ_HOURLY:
        return "every hour" if n == 1 else f"every {n} hours"
    if rule.freq == FREQ_CRON:
        return f"on schedule '{rule.cron}'"

    at = f" at {_fmt_time(rule.hour, rule.minute)}"
    if rule.freq == FREQ_DAILY:
        if rule.byday:
            return f"every {_describe_days(rule.byday)}{at}"
        return ("every day" if n == 1 else f"every {n} days") + at
    if rule.freq == FREQ_WEEKLY:
        prefix = "every week" if n == 1 else f"every {n} weeks"
        if rule.byday:
            d = _describe_days(rule.byday)
            if d in ("weekday", "weekend day") and n == 1:
                return f"every {d}{at}"
            return f"{prefix} on {d}{at}" if n > 1 else f"every {d}{at}"
        return f"{prefix}{at}"
    if rule.freq == FREQ_MONTHLY:
        prefix = "every month" if n == 1 else f"every {n} months"
        if rule.bymonthday:
            return f"{prefix} on the {_ordinal(rule.bymonthday)}{at}"
        return f"{prefix}{at}"
    if rule.freq == FREQ_YEARLY:
        prefix = "every year" if n == 1 else f"every {n} years"
        anchor = _anchor_date_for_describe(rule, tz)
        if anchor is not None:
            return f"{prefix} on {anchor.strftime('%B')} {anchor.day}{at}"
        return prefix + at
    return "on a schedule"


# ---------------------------------------------------------------------------
# Natural-language parsing  (#84)
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(
    r"\b(?:at|@|around|by)?\s*"
    r"(?:(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)"   # 7am / 7:30 pm
    r"|(?P<h24>[01]?\d|2[0-3]):(?P<m24>[0-5]\d)"                          # 19:00
    r"|(?P<word>noon|midnight))\b",
    re.IGNORECASE,
)

_MONTHS_BY_ABBR: dict[str, int] = {
    m: i + 1 for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    )
}
_MONTH_WORD = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
)
_TRAILING_MONTHDAY_RE = re.compile(
    r"\bon\s+the\s+(?P<dom>\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE,
)
_TRAILING_YEARDATE_RE = re.compile(
    r"\bon\s+(?:the\s+)?(?:(?P<mon1>" + _MONTH_WORD + r")\s+(?P<day1>\d{1,2})(?:st|nd|rd|th)?"
    r"|(?P<day2>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mon2>" + _MONTH_WORD + r"))\b",
    re.IGNORECASE,
)

# "first monday of every month", "last friday of the month": a recurrence
# Chronos cannot represent (its rules have no "nth weekday"). Detected so the
# caller can say so instead of quietly making a plain weekly reminder.
_NTH_WEEKDAY_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\s+"
    + _WEEKDAY_WORD + r"\s+of\s+(?:every|each|the)\s+month\b"
    r"|\b(?:every|each)\s+(?:first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\s+"
    + _WEEKDAY_WORD + r"\b",
    re.IGNORECASE,
)


def unsupported_recurrence_reason(text: str) -> Optional[str]:
    """Explain why *text* asks for a repeat pattern Chronos can't do, or
    return ``None`` when nothing in it is unsupported."""
    if text and _NTH_WEEKDAY_RE.search(text):
        return (
            "I can't do 'nth weekday of the month' repeats (like the first Monday) yet. "
            "I can repeat on a fixed date, like the 1st of every month, or every week."
        )
    return None


_UNTIL_RE = re.compile(
    r"\b(?:until|till|through|up\s+to)\s+(?P<when>[A-Za-z0-9,\-/ ]{3,30}?)(?=$|\s+(?:at|to|and)\b|[,.;])",
    re.IGNORECASE,
)
_FOR_N_RE = re.compile(
    r"\bfor\s+(?:the\s+next\s+)?(?P<n>\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")\s+"
    r"(?P<unit>minutes?|hours?|days?|weeks?|months?|years?|times?)\b",
    re.IGNORECASE,
)

_WORD_NUM = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))


def parse_time_of_day(text: str) -> Optional[tuple[int, int, tuple[int, int]]]:
    """
    Find a clock time in *text*. Returns ``(hour, minute, (span_start, span_end))``
    or ``None``. Handles ``7am``, ``7:30 pm``, ``19:00``, ``noon``, ``midnight``.
    """
    m = _TIME_RE.search(text)
    if not m:
        return None
    if m.group("word"):
        hour = 12 if m.group("word").lower() == "noon" else 0
        minute = 0
    elif m.group("h24") is not None:
        hour, minute = int(m.group("h24")), int(m.group("m24"))
    else:
        hour = int(m.group("h"))
        minute = int(m.group("m") or 0)
        ap = m.group("ap").lower().replace(".", "")
        if not (1 <= hour <= 12) or minute > 59:
            return None
        if ap == "pm" and hour != 12:
            hour += 12
        elif ap == "am" and hour == 12:
            hour = 0
    return hour, minute, (m.start(), m.end())


def _num(token: str) -> int:
    token = token.lower()
    return int(token) if token.isdigit() else _NUMBER_WORDS.get(token, 1)


def _parse_weekdays(text: str) -> tuple[int, ...]:
    found = []
    for tok in re.findall(_WEEKDAY_WORD, text, flags=re.IGNORECASE):
        key = tok.lower().rstrip("s") if tok.lower() not in _WEEKDAY_LOOKUP else tok.lower()
        if key in _WEEKDAY_LOOKUP:
            found.append(_WEEKDAY_LOOKUP[key])
        elif key.rstrip("s") in _WEEKDAY_LOOKUP:  # "tues" -> handled above; "thurs"
            found.append(_WEEKDAY_LOOKUP[key.rstrip("s")])
    return tuple(sorted(set(found)))


# Each entry: (compiled regex, builder(match) -> partial-rule kwargs).
# Order matters: more specific patterns first. The FIRST match wins and its
# span is removed from the text.
_NL_RULES: list[tuple["re.Pattern[str]", Any]] = []


def _rule(pattern: str):
    def deco(fn):
        _NL_RULES.append((re.compile(pattern, re.IGNORECASE), fn))
        return fn
    return deco


@_rule(r"\bevery\s+(?:(?P<n>\d{1,3}|" + _WORD_NUM + r")\s+)?(?P<unit>minutes?|mins?|hours?|hrs?)\b")
def _r_subdaily(m):
    n = _num(m.group("n")) if m.group("n") else 1
    unit = m.group("unit").lower().rstrip("s")
    return {"freq": _UNIT_TO_FREQ[unit], "interval": n}


@_rule(r"\bevery\s+(?:other|second)\s+(?P<unit>day|week|month|year)\b")
def _r_other(m):
    return {"freq": _UNIT_TO_FREQ[m.group("unit").lower()], "interval": 2}


@_rule(
    r"\bevery\s+(?P<n>\d{1,3}|" + _WORD_NUM + r")\s+(?P<unit>days?|weeks?|months?|years?)"
    r"(?:\s+on\s+(?P<days>(?:" + _WEEKDAY_WORD + r"(?:\s*(?:,|and|&)\s*)?)+))?"
)
def _r_every_n(m):
    unit = m.group("unit").lower().rstrip("s")
    out: dict[str, Any] = {"freq": _UNIT_TO_FREQ[unit], "interval": _num(m.group("n"))}
    if m.group("days"):
        out["freq"] = FREQ_WEEKLY
        out["byday"] = _parse_weekdays(m.group("days"))
    return out


# NOTE: requires "every"/"each". A bare "on the 15th" is a ONE-SHOT reminder
# for the next 15th, not a monthly rule, so it must not match here.
@_rule(r"\b(?:every|each)\s+(?:the\s+)?(?P<dom>\d{1,2})(?:st|nd|rd|th)\b(?!\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december))")
def _r_monthday(m):
    day = int(m.group("dom"))
    if not 1 <= day <= 31:
        return None
    return {"freq": FREQ_MONTHLY, "bymonthday": day}


@_rule(r"\b(?:on\s+the\s+)?(?P<dom>\d{1,2})(?:st|nd|rd|th)\s+of\s+(?:every|each)\s+month\b")
def _r_monthday_of(m):
    day = int(m.group("dom"))
    return {"freq": FREQ_MONTHLY, "bymonthday": day} if 1 <= day <= 31 else None


@_rule(r"\b(?:every|each)\s+weekdays?\b|\bon\s+weekdays\b|\bweekdays\b|\bmonday\s+(?:to|through|-)\s+friday\b|\bmon\s*-\s*fri\b|\bworking\s+days?\b")
def _r_weekdays(m):
    return {"freq": FREQ_WEEKLY, "byday": (0, 1, 2, 3, 4)}


@_rule(r"\b(?:every|each)\s+weekends?\b|\bon\s+weekends\b|\bweekends\b")
def _r_weekends(m):
    return {"freq": FREQ_WEEKLY, "byday": (5, 6)}


@_rule(
    r"\bevery\s+(?P<days>(?:" + _WEEKDAY_WORD + r")(?:\s*(?:,|and|&|/)\s*(?:" + _WEEKDAY_WORD + r"))*)"
    r"(?:\s+(?P<part>morning|afternoon|evening|night))?"
)
def _r_named_days(m):
    out: dict[str, Any] = {"freq": FREQ_WEEKLY, "byday": _parse_weekdays(m.group("days"))}
    if m.group("part"):
        out["_part"] = m.group("part").lower()
    return out


@_rule(r"\bevery\s+(?P<part>morning|afternoon|evening|night)\b")
def _r_part_of_day(m):
    return {"freq": FREQ_DAILY, "_part": m.group("part").lower()}


@_rule(r"\b(?:every\s+(?:single\s+)?day|everyday|daily|each\s+day)\b")
def _r_daily(m):
    return {"freq": FREQ_DAILY}


@_rule(r"\b(?:every\s+week|weekly|each\s+week)\b")
def _r_weekly(m):
    return {"freq": FREQ_WEEKLY}


@_rule(r"\b(?:every\s+month|monthly|each\s+month)\b")
def _r_monthly(m):
    return {"freq": FREQ_MONTHLY}


@_rule(r"\b(?:every\s+year|yearly|annually|each\s+year)\b")
def _r_yearly(m):
    return {"freq": FREQ_YEARLY}


_CRON_LITERAL_RE = re.compile(
    r"(?:^|\s|cron\s*[:=]?\s*)((?:[\d\*/,\-?]+|[A-Za-z]{3}(?:[-,][A-Za-z]{3})*)"
    r"(?:\s+(?:[\d\*/,\-?]+|[A-Za-z]{3}(?:[-,][A-Za-z]{3})*)){4})(?=\s|$)"
)


def extract_recurrence(
    text: str,
    now_local: datetime,
    tz: Any,
) -> tuple[Optional[Recurrence], str]:
    """
    Parse a recurrence rule out of free text.

    Returns ``(rule_or_None, text_with_rule_phrases_removed)``. The rule is
    *anchored* at its first occurrence (the first slot strictly after
    *now_local*), so ``next_after`` is well defined for ``interval > 1``.

    Never raises: unparseable input yields ``(None, text)`` unchanged, so the
    caller falls back to one-shot reminder handling.
    """
    if not text or not text.strip():
        return None, text or ""

    remaining = text

    # Raw cron literal wins ("0 7 * * 1-5"): unambiguous, no NL guessing.
    cm = _CRON_LITERAL_RE.search(remaining)
    if cm:
        try:
            rule = parse_cron(cm.group(1))
        except ValueError:
            rule = None
        if rule is not None:
            remaining = _squash(remaining[: cm.start()] + " " + remaining[cm.end():])
            first = next_after(rule, now_local, tz)
            if first is None:
                return None, text
            return rule.with_anchor(first), remaining

    fields: Optional[dict[str, Any]] = None
    span: Optional[tuple[int, int]] = None
    for pattern, builder in _NL_RULES:
        m = pattern.search(remaining)
        if not m:
            continue
        built = builder(m)
        if built is None:
            continue
        fields, span = built, m.span()
        break
    if fields is None or span is None:
        return None, text

    remaining = _squash(remaining[: span[0]] + " " + remaining[span[1]:])

    # "every other week on friday", "weekly on monday and wednesday": the
    # weekday list trails the base phrase rather than being part of it.
    if fields["freq"] == FREQ_WEEKLY and not fields.get("byday"):
        dm = re.search(
            r"\bon\s+(?P<days>(?:" + _WEEKDAY_WORD + r")(?:\s*(?:,|and|&|/)\s*(?:" + _WEEKDAY_WORD + r"))*)\b",
            remaining, flags=re.IGNORECASE,
        )
        if dm:
            fields["byday"] = _parse_weekdays(dm.group("days"))
            remaining = _squash(remaining[: dm.start()] + " " + remaining[dm.end():])

    # "every month on the 31st": the day of month trails the base phrase.
    if fields["freq"] == FREQ_MONTHLY and not fields.get("bymonthday"):
        mm = _TRAILING_MONTHDAY_RE.search(remaining)
        if mm:
            day = int(mm.group("dom"))
            if 1 <= day <= 31:
                fields["bymonthday"] = day
                remaining = _squash(remaining[: mm.start()] + " " + remaining[mm.end():])

    # "every year on march 5": the calendar date trails the base phrase.
    yearly_date: Optional[tuple[int, int]] = None
    if fields["freq"] == FREQ_YEARLY:
        ym = _TRAILING_YEARDATE_RE.search(remaining)
        if ym:
            month_name = (ym.group("mon1") or ym.group("mon2")).lower()[:3]
            day = int(ym.group("day1") or ym.group("day2"))
            month = _MONTHS_BY_ABBR.get(month_name)
            if month is not None and 1 <= day <= _last_day_of_month(2024, month):
                yearly_date = (month, day)
                remaining = _squash(remaining[: ym.start()] + " " + remaining[ym.end():])

    # Optional end condition: "for 5 days", "until Friday", "until 2026-12-31".
    count: Optional[int] = None
    until: Optional[str] = None
    fm = _FOR_N_RE.search(remaining)
    if fm:
        n = _num(fm.group("n"))
        unit = fm.group("unit").lower().rstrip("s")
        remaining = _squash(remaining[: fm.start()] + " " + remaining[fm.end():])
        if unit == "time":
            count = n
        else:
            # "for 5 days" -> stop after now + 5 days; resolved to a date.
            per = {"minute": timedelta(minutes=n), "hour": timedelta(hours=n),
                   "day": timedelta(days=n), "week": timedelta(weeks=n),
                   "month": timedelta(days=30 * n), "year": timedelta(days=365 * n)}[unit]
            until = (now_local + per).date().isoformat()
    else:
        um = _UNTIL_RE.search(remaining)
        if um:
            parsed = _parse_until_date(um.group("when"), now_local)
            if parsed:
                until = parsed
                remaining = _squash(remaining[: um.start()] + " " + remaining[um.end():])

    # Time of day (only meaningful for calendar-based frequencies).
    hour, minute = DEFAULT_HOUR, DEFAULT_MINUTE
    part = fields.pop("_part", None)
    if part:
        hour, minute = _PART_OF_DAY[part]
    tm = parse_time_of_day(remaining)
    if tm:
        hour, minute, (s, e) = tm
        remaining = _squash(remaining[:s] + " " + remaining[e:])

    freq = fields["freq"]
    try:
        rule = Recurrence(
            freq=freq,
            interval=fields.get("interval", 1),
            byday=tuple(fields.get("byday", ())),
            bymonthday=fields.get("bymonthday"),
            hour=hour,
            minute=minute,
            until=until,
            count=count,
        )
    except ValueError:
        return None, text

    # Sub-daily rules with no explicit clock time start one interval from now.
    if freq in (FREQ_MINUTELY, FREQ_HOURLY):
        step = timedelta(
            minutes=rule.interval if freq == FREQ_MINUTELY else 0,
            hours=rule.interval if freq == FREQ_HOURLY else 0,
        )
        first = (now_local + step).replace(second=0, microsecond=0)
        return rule.with_anchor(first), remaining

    if yearly_date is not None:
        month, day = yearly_date
        for year in (now_local.year, now_local.year + 1, now_local.year + 2, now_local.year + 4):
            if day > _last_day_of_month(year, month):
                continue
            cand = _combine(date(year, month, day), hour, minute, tz)
            if cand > now_local and (until is None or cand.date().isoformat() <= until):
                return rule.with_anchor(cand), remaining
        return None, text

    # Anchor at the first real occurrence, then re-derive it from the anchor
    # so phase (for interval > 1) is consistent with what next_after will use.
    provisional = rule.with_anchor(now_local)
    first = next_after(provisional, now_local, tz)
    if first is None:
        return None, text
    return rule.with_anchor(first), remaining


def _parse_until_date(phrase: str, now_local: datetime) -> Optional[str]:
    """Best-effort 'until <date>' -> ISO date. Uses dateparser lazily so this
    module stays importable (and testable) without it."""
    phrase = phrase.strip(" ,.")
    if not phrase:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", phrase):
        return phrase
    try:
        import dateparser
    except ImportError:  # pragma: no cover - dateparser is a hard dependency
        return None
    parsed = dateparser.parse(
        phrase,
        languages=["en"],
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": now_local.replace(tzinfo=None),
        },
    )
    return parsed.date().isoformat() if parsed else None


# ---------------------------------------------------------------------------
# Durations (snooze, #83)
# ---------------------------------------------------------------------------

# The number word alternatives need a "not preceded by a letter" guard, or the
# article "a" matches inside ordinary words ("bla|h" -> "a hour"). Digits
# don't, so compact forms like "1h30m" still parse. The unit is closed with
# "not followed by a letter" rather than \b for the same reason: "1h" is
# followed by the digit in "1h30m", which \b treats as no boundary at all.
_DURATION_PART_RE = re.compile(
    r"(?P<n>\d+(?:\.\d+)?|(?<![a-z])(?:" + _WORD_NUM + r"|half\s+an?|quarter\s+of\s+an?))(?:(?<=\d)\s*|\s+)"
    r"(?P<unit>hours?|hrs?|h|minutes?|mins?|m|days?|d|weeks?|w|seconds?|secs?)(?![a-z])",
    re.IGNORECASE,
)


def parse_duration(text: str) -> Optional[timedelta]:
    """
    Parse a relative duration: ``"10 minutes"``, ``"an hour"``, ``"half an
    hour"``, ``"1h30m"``, ``"2 hours 15 minutes"``, ``"1.5 hours"``.

    Returns ``None`` when nothing duration-like is found (so callers can fall
    back to a default or an absolute time), and a non-positive result is
    treated as "not a duration".
    """
    if not text:
        return None
    total = timedelta()
    found = False
    lowered = text.lower().replace("&", " and ")
    # "an hour and a half" -> 1.5 hours
    if re.search(r"\b(?:an?|one)\s+hour\s+and\s+a\s+half\b", lowered):
        return timedelta(minutes=90)
    for m in _DURATION_PART_RE.finditer(lowered):
        raw = re.sub(r"\s+", " ", m.group("n").strip())
        if raw.startswith("half"):
            qty = 0.5
        elif raw.startswith("quarter"):
            qty = 0.25
        elif raw in _NUMBER_WORDS:
            qty = float(_NUMBER_WORDS[raw])
        else:
            try:
                qty = float(raw)
            except ValueError:
                continue
        unit = m.group("unit")
        if unit.startswith(("hour", "hr")) or unit == "h":
            total += timedelta(hours=qty)
        elif unit.startswith(("min",)) or unit == "m":
            total += timedelta(minutes=qty)
        elif unit.startswith("day") or unit == "d":
            total += timedelta(days=qty)
        elif unit.startswith("week") or unit == "w":
            total += timedelta(weeks=qty)
        elif unit.startswith("sec"):
            total += timedelta(seconds=qty)
        else:
            continue
        found = True
    if not found or total <= timedelta(0):
        return None
    return total


_DURATION_LEAD_RE = re.compile(r"\b(?:for|by|another|an\s+extra|extra)\s*$", re.IGNORECASE)


def strip_duration(text: str) -> str:
    """
    Remove relative-duration phrases (and a dangling "for" / "by" / "another"
    in front of them) so what is left can be used as a reminder-name hint.

    ``"snooze the gym reminder for 10 minutes"`` -> ``"snooze the gym reminder"``
    ``"snooze 1h30m"`` -> ``"snooze"``
    """
    if not text:
        return ""
    lowered = text.replace("&", " and ")
    # "an hour and a half" is not matched by the part regex on its own.
    lowered = re.sub(
        r"\b(?:an?|one)\s+hour\s+and\s+a\s+half\b", " ", lowered, flags=re.IGNORECASE
    )
    out: list[str] = []
    last = 0
    for m in _DURATION_PART_RE.finditer(lowered):
        chunk = lowered[last:m.start()]
        chunk = _DURATION_LEAD_RE.sub("", chunk)
        out.append(chunk)
        last = m.end()
    out.append(lowered[last:])
    cleaned = " ".join("".join(out).split())
    # A joining "and" left between two removed parts ("2 hours and 15 minutes").
    cleaned = re.sub(r"\band\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


# ---------------------------------------------------------------------------
# iCalendar RRULE bridge  (#90)
# ---------------------------------------------------------------------------

def to_rrule(rule: Recurrence, tz: Any) -> Optional[str]:
    """
    Convert *rule* to an RFC 5545 ``RRULE`` value (without the ``RRULE:``
    prefix), or ``None`` if it has no faithful RRULE form (complex cron).

    ``UNTIL`` is emitted as a UTC date-time at end of the local day, which is
    the form every calendar app accepts for a DTSTART with a time component.
    """
    parts: list[str] = []
    n = rule.interval

    if rule.freq == FREQ_MINUTELY:
        parts = ["FREQ=MINUTELY"]
    elif rule.freq == FREQ_HOURLY:
        parts = ["FREQ=HOURLY"]
    elif rule.freq == FREQ_DAILY:
        parts = ["FREQ=DAILY"]
        if rule.byday:
            parts = ["FREQ=WEEKLY"]
            n = 1
    elif rule.freq == FREQ_WEEKLY:
        parts = ["FREQ=WEEKLY"]
    elif rule.freq == FREQ_MONTHLY:
        parts = ["FREQ=MONTHLY"]
    elif rule.freq == FREQ_YEARLY:
        parts = ["FREQ=YEARLY"]
    elif rule.freq == FREQ_CRON:
        simple = _simple_cron_to_rrule(rule)
        if simple is None:
            return None
        parts = simple
        n = 1
    if n != 1:
        parts.append(f"INTERVAL={n}")
    if rule.byday and rule.freq != FREQ_CRON:
        parts.append("BYDAY=" + ",".join(_ICS_DAYS[d] for d in sorted(rule.byday)))
    if rule.freq == FREQ_MONTHLY and rule.bymonthday:
        parts.append(f"BYMONTHDAY={rule.bymonthday}")
    if rule.count is not None:
        parts.append(f"COUNT={rule.count}")
    elif rule.until:
        try:
            end_local = datetime.combine(
                date.fromisoformat(rule.until), datetime.max.time().replace(microsecond=0)
            ).replace(tzinfo=tz)
            parts.append("UNTIL=" + end_local.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        except ValueError:
            pass
    return ";".join(parts)


def _simple_cron_to_rrule(rule: Recurrence) -> Optional[list[str]]:
    """Only the cron shapes that map cleanly: single minute+hour, with either
    all days, a weekday list, or a single day-of-month."""
    spec = _parse_cron(rule.cron or "")
    if len(spec.minutes) != 1 or len(spec.hours) != 1 or len(spec.months) != 12:
        return None
    if spec.dom_star and spec.dow_star:
        return ["FREQ=DAILY"]
    if spec.dom_star and not spec.dow_star:
        return ["FREQ=WEEKLY", "BYDAY=" + ",".join(_ICS_DAYS[d] for d in sorted(spec.dows))]
    if not spec.dom_star and spec.dow_star and len(spec.doms) == 1:
        return ["FREQ=MONTHLY", f"BYMONTHDAY={next(iter(spec.doms))}"]
    return None


def from_rrule(
    rrule: str,
    dtstart_local: datetime,
    tz: Any,
) -> Recurrence:
    """
    Build a ``Recurrence`` from an RFC 5545 RRULE *value* and its DTSTART.

    Raises ``ValueError`` for rules Chronos can't represent faithfully
    (e.g. ``BYDAY=1MO`` "first Monday of the month", ``BYSETPOS``, ``BYWEEKNO``)
    - the importer catches that and falls back to a one-shot reminder with a
    warning rather than silently importing a *different* schedule.
    """
    fields: dict[str, str] = {}
    for chunk in rrule.strip().split(";"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            fields[k.strip().upper()] = v.strip()
    freq_raw = fields.get("FREQ", "").upper()
    freq_map = {
        "MINUTELY": FREQ_MINUTELY, "HOURLY": FREQ_HOURLY, "DAILY": FREQ_DAILY,
        "WEEKLY": FREQ_WEEKLY, "MONTHLY": FREQ_MONTHLY, "YEARLY": FREQ_YEARLY,
    }
    if freq_raw not in freq_map:
        raise ValueError(f"Unsupported RRULE FREQ {freq_raw!r}.")
    unsupported = {"BYSETPOS", "BYWEEKNO", "BYYEARDAY", "BYHOUR", "BYMINUTE",
                   "BYSECOND", "BYMONTH"}
    bad = unsupported & set(fields)
    if bad:
        raise ValueError(f"Unsupported RRULE parts: {sorted(bad)}.")

    byday: list[int] = []
    if "BYDAY" in fields:
        for tok in fields["BYDAY"].split(","):
            tok = tok.strip().upper()
            if tok not in _ICS_DAYS:  # e.g. "1MO", "-1FR" -> nth weekday of month
                raise ValueError(f"Unsupported BYDAY value {tok!r}.")
            byday.append(_ICS_DAYS.index(tok))

    bymonthday: Optional[int] = None
    if "BYMONTHDAY" in fields:
        try:
            vals = [int(x) for x in fields["BYMONTHDAY"].split(",")]
        except ValueError as exc:
            raise ValueError("Bad BYMONTHDAY.") from exc
        if len(vals) != 1 or vals[0] < 1:
            raise ValueError("Only a single positive BYMONTHDAY is supported.")
        bymonthday = vals[0]

    interval = int(fields.get("INTERVAL", "1") or 1)
    count = int(fields["COUNT"]) if "COUNT" in fields else None
    until: Optional[str] = None
    if "UNTIL" in fields:
        u = fields["UNTIL"]
        try:
            if len(u) == 8:
                until = datetime.strptime(u, "%Y%m%d").date().isoformat()
            else:
                stamp = datetime.strptime(u.rstrip("Z"), "%Y%m%dT%H%M%S")
                if u.endswith("Z"):
                    stamp = stamp.replace(tzinfo=timezone.utc).astimezone(tz)
                until = stamp.date().isoformat()
        except ValueError as exc:
            raise ValueError(f"Bad UNTIL {u!r}.") from exc

    freq = freq_map[freq_raw]
    if freq == FREQ_WEEKLY and byday and len(byday) == 7 and interval == 1:
        freq, byday = FREQ_DAILY, []

    rule = Recurrence(
        freq=freq,
        interval=interval,
        byday=tuple(sorted(set(byday))),
        bymonthday=bymonthday,
        hour=dtstart_local.hour,
        minute=dtstart_local.minute,
        until=until,
        count=count,
    )
    return rule.with_anchor(dtstart_local)