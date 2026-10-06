"""
modules/chronos/engine.py

ChronosEngine: time, date, weather, and reminder module.

Design notes
------------
- All datetime operations use timezone-aware objects (UTC internally,
  local-tz for display) to avoid DST ambiguity.
- Weather fetching is isolated in pure helpers; the engine delegates and
  handles failures without coupling to HTTP internals.
- Reminder parsing is decomposed into focused private methods: task
  extraction, datetime parsing, and validation are each independently
  testable.
- City coordinates live in a typed constant; callers can extend it by
  subclassing or by injecting a config dict.
- Every public method conforms to the BaseModule response contract:
  {response: str, data: dict, confidence: float}.

Extended reminders (backlog #81-#90)
------------------------------------
The engine is a thin intent router. The rules live in sibling modules so
they can be tested without HTTP, threads or a real clock:

  recurrence.py  recurring rules, natural-language parsing, time zones
  reminders.py   ReminderService: firing, snooze, holidays, location, ICS
  agenda.py      "what's on my plate" aggregation + weather concerns
  ics.py         iCalendar reading/writing

The extended behaviour needs the Mnemosyne database. Without it (no memory
object, or one that predates the extended schema) the engine keeps its
original one-shot behaviour.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser
from dateparser.search import search_dates
import requests

from modules.base import BaseModule
from core.free_apis import FreeAPIError, is_public_holiday, public_holidays

from . import agenda as agenda_mod
from .recurrence import (
    Recurrence,
    extract_recurrence,
    extract_timezone,
    parse_duration,
    parse_time_of_day,
    resolve_timezone_name,
    strip_duration,
    to_zoneinfo,
    unsupported_recurrence_reason,
)
from .reminders import (
    DEFAULT_RADIUS_M,
    ReminderService,
    describe_when,
    parse_iso,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOCATION = "Mumbai"
_DEFAULT_LAT = 19.0760
_DEFAULT_LON = 72.8777
_REQUEST_TIMEOUT = 10  # seconds
_MIN_TASK_LEN = 3
_TASK_FALLBACK = "your task"
_DEFAULT_COUNTRY = "IN"  # ISO 3166-1 alpha-2, matches _DEFAULT_LOCATION (Mumbai)

# WMO weather interpretation codes → human-readable label
_WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "showers", 82: "heavy showers",
    95: "thunderstorm",
}

# (latitude, longitude) for commonly requested cities
CityCoords = dict[str, tuple[float, float]]

_CITY_COORDS: CityCoords = {
    "mumbai":       (19.0760,   72.8777),
    "delhi":        (28.6139,   77.2090),
    "bangalore":    (12.9716,   77.5946),
    "bengaluru":    (12.9716,   77.5946),
    "chennai":      (13.0827,   80.2707),
    "kolkata":      (22.5726,   88.3639),
    "hyderabad":    (17.3850,   78.4867),
    "pune":         (18.5204,   73.8567),
    "london":       (51.5074,   -0.1278),
    "new york":     (40.7128,  -74.0060),
    "los angeles":  (34.0522, -118.2437),
    "tokyo":        (35.6762,  139.6503),
    "paris":        (48.8566,    2.3522),
    "sydney":       (-33.8688,  151.2093),
    "dubai":        (25.2048,   55.2708),
}

# Natural-language time-of-day → (hour, minute, delta_days)
_TIME_OF_DAY: dict[str, tuple[int, int, int]] = {
    "morning":   (9,  0, 1),
    "afternoon": (14, 0, 0),
    "evening":   (18, 0, 0),
    "night":     (21, 0, 0),
    "tonight":   (21, 0, 0),
    "midnight":  (0,  0, 1),
    "noon":      (12, 0, 0),
}

# Verbs / noise words stripped when extracting a task from raw text
_TASK_NOISE = re.compile(
    r"\b(remind me|set a reminder|reminder|please|can you|could you)\b",
    flags=re.IGNORECASE,
)
_TIME_SUFFIX = re.compile(
    r"\b(at|in|on|next|this coming|tomorrow|today|tonight|morning|afternoon"
    r"|evening|night|midnight|noon|monday|tuesday|wednesday|thursday|friday"
    r"|saturday|sunday|\d{1,2}[:\s]\d{2}|\d{1,2}\s?(?:am|pm))\b.*",
    flags=re.IGNORECASE,
)

# --- scheduling phrases pulled out of a reminder request -------------------

# Explicit "don't fire on holidays" wording. These phrases are *removed* from
# the text so they never leak into the task label.
_SKIP_HOLIDAYS_RE = re.compile(
    r"\b(?:but\s+)?(?:skip(?:ping)?|except(?:\s+on)?|excluding|not\s+on)\s+"
    r"(?:my\s+|the\s+|any\s+|all\s+)?(?:holidays?|days?\s+off|non[- ]?working\s+days?)\b"
    r"|\bunless\s+(?:it'?s|its)\s+a\s+holiday\b",
    flags=re.IGNORECASE,
)
# Wording that *implies* it ("every working day"). Left in place: the
# recurrence parser turns it into Monday-Friday.
_WORKING_DAY_RE = re.compile(
    r"\b(?:working|business|school|work)\s+days?\b", flags=re.IGNORECASE
)

# Places that can be named without a preposition ("when I get home").
_BARE_PLACES = r"home|work|office|school|college|campus|gym|the\s+office|the\s+gym"
_PLACE_END = r"(?=$|[,.;!?]|\s+(?:and|then|so|remind|please)\b)"
_PLACE_CLAUSE_RES = (
    # "when I get/reach/arrive/come/return/go (back) to|at|in <place>"
    re.compile(
        r"\b(?:when|once|as\s+soon\s+as|the\s+moment|whenever)\s+i\s+"
        r"(?:get|got|reach|reached|arrive|arrived|come|came|return|returned|go|went|walk\s+into)\s+"
        r"(?:back\s+)?(?:to|at|in|into)\s+(?:the\s+|my\s+)?(?P<p>[a-z][a-z' ]{1,28}?)" + _PLACE_END,
        flags=re.IGNORECASE,
    ),
    # "when I get home", "when I reach work"
    re.compile(
        r"\b(?:when|once|as\s+soon\s+as|the\s+moment|whenever)\s+i\s+"
        r"(?:get|got|reach|reached|arrive|arrived|come|came|return|returned|go|went)\s+"
        r"(?:back\s+)?(?P<p>" + _BARE_PLACES + r")\b",
        flags=re.IGNORECASE,
    ),
    # "when I'm at the gym", "when I am home"
    re.compile(
        r"\b(?:when|once|as\s+soon\s+as|whenever)\s+i(?:'m|\s+am)\s+"
        r"(?:(?:at|in)\s+(?:the\s+|my\s+)?(?P<p>[a-z][a-z' ]{1,28}?)" + _PLACE_END
        + r"|(?P<p2>" + _BARE_PLACES + r")\b)",
        flags=re.IGNORECASE,
    ),
)
# Words that make a captured "place" obviously not a place.
_NOT_A_PLACE_RE = re.compile(
    r"\b(?:the\s+mood|a\s+chance|time|trouble|ready|done|bored|free|hungry|tired|"
    r"back|there|here|late|early|sad|happy|able|going|stuck)\b|\d",
    re.IGNORECASE,
)
_PLACE_ALIASES: dict[str, str] = {"the office": "office", "the gym": "gym"}
# "work" and "office" are usually the same place to the user; look up either.
_PLACE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "work": ("work", "office"), "office": ("office", "work"),
    "college": ("college", "campus", "school"), "campus": ("campus", "college"),
    "school": ("school", "college"),
}

# Reminders whose task reads like something you'd not do on a day off skip
# user-marked holidays automatically - but only when they REPEAT (a one-off
# "remind me to study at 7pm" is something you asked for explicitly).
# Deliberately narrow: a false positive means a reminder the user wanted
# silently doesn't fire.
_HOLIDAY_SENSITIVE_RE = re.compile(
    r"\b(?:study|studying|homework|revision|revise|lecture|class|classes|"
    r"office|standup|stand-up)\b",
    flags=re.IGNORECASE,
)

_ICS_MAX_BYTES = 1_000_000
_ICS_PATH_RE = re.compile(
    r"(?P<path>(?:~|/|[A-Za-z]:\\|\.{1,2}/)?[^\s'\"]+\.ics)\b", re.IGNORECASE
)
_DEFAULT_EXPORTS_DIR = "data/exports"
_SCHEDULER_INTERVAL_S = 30.0
_MAX_HOLIDAY_RANGE_DAYS = 60
_SPOKEN_REMINDER_LIMIT = 8
_WEATHER_RETRY_MINUTES = 30
_WEATHER_MAX_ATTEMPTS = 3
_PROACTIVE_WEATHER_FROM_HOUR = 7

# Command words stripped from the *front* of a management request
# ("cancel the dentist reminder") and reminder words from its *end*.
_MGMT_NOISE = frozenset({
    "please", "can", "you", "could", "would", "hestia", "hey", "snooze", "postpone",
    "delay", "push", "back", "cancel", "delete", "remove", "clear", "stop", "forget",
    "drop", "the", "my", "that", "this", "it", "reminder", "reminders", "remind",
    "me", "again", "about", "for", "to", "of", "by", "another", "until", "in", "all",
    "everything", "every", "a", "an", "off", "out", "of", "on", "up", "with", "and",
    "number", "no", "next", "set", "existing", "scheduled", "pending", "still",
})
_MISSED_RE = re.compile(r"\b(?:missed|miss|overdue|slipped|forgot|skipped)\b", re.IGNORECASE)
_ALL_RE = re.compile(r"\b(?:all|everything|every)\b", re.IGNORECASE)
_INDEX_RE = re.compile(r"(?:\breminder\s+|#|\bnumber\s+|\bno\.?\s*)(\d{1,3})\b", re.IGNORECASE)

# A fragment returned by dateparser's `search_dates()` is only trusted as an
# actual date/time reference if it contains a digit or one of these
# unambiguous temporal keywords. Without this filter, `search_dates()`
# regularly misfires on ordinary words inside a reminder task — e.g. a task
# like "email May about the report" or "text August" gets a chunk of the
# task's own text ("May", "August", ...) parsed as a date, silently
# producing a *wrong* reminder time instead of no match at all.
_TEMPORAL_KEYWORD_RE = re.compile(
    r"\d|\b(today|tomorrow|tonight|yesterday|noon|midnight|morning|afternoon"
    r"|evening|night|next|this coming|monday|tuesday|wednesday|thursday"
    r"|friday|saturday|sunday|week|weekend|month|year|hour|hours|minute"
    r"|minutes|sec|second|seconds|am|pm|o'clock)\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ChronosError(Exception):
    """Base exception for ChronosEngine failures."""


class WeatherFetchError(ChronosError):
    """Raised when the weather API call fails."""


class ReminderParseError(ChronosError):
    """Raised when a reminder time cannot be parsed."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ChronosEngine(BaseModule):
    """
    Time, date, weather, and reminder module.

    Parameters
    ----------
    memory:
        Optional Mnemosyne memory engine for persisting reminders and
        reading user preferences (location, timezone).
    city_coords:
        Optional mapping of ``{city_name_lower: (lat, lon)}`` that extends
        or overrides ``_CITY_COORDS``.
    local_tz:
        IANA timezone string for local-time display (e.g. ``"Asia/Kolkata"``).
        Defaults to UTC when omitted or invalid.
    hermes, artemis, dionysus:
        Optional sibling modules used by the "what's on my plate" aggregator
        (backlog #86) and weather-triggered suggestions (#88). They are
        usually attached after construction with ``attach_sources`` because
        Chronos is registered before they exist.
    notify:
        ``callable(text)`` used to announce fired reminders. Defaults to
        emitting ``speak`` on the event bus.
    clock:
        Zero-arg callable returning the current aware datetime (tests).
    skip_public_holidays:
        When True, reminders that skip holidays also skip *public* holidays
        (looked up via Nager.Date); user-marked days always count.
    holiday_country:
        ISO country code for public-holiday lookups (default ``IN``).
    exports_dir:
        Where ICS exports are written (default ``data/exports``).
    default_snooze_minutes:
        Length of a snooze when the user doesn't say.
    proactive_weather:
        When True the scheduler warns once each morning if rain threatens an
        outdoor plan in today's agenda.
    """

    name = "chronos"

    _INTENTS: frozenset[str] = frozenset(
        {
            "get_time", "get_date", "get_weather", "set_reminder", "get_holiday",
            # Reminder management (backlog #81-#90)
            "list_reminders", "cancel_reminder", "snooze_reminder",
            "get_agenda", "mark_holiday", "unmark_holiday", "save_place",
            "export_calendar", "import_calendar", "weather_plan",
            # Backlog #163: the few things that need attention this week
            "weekly_focus",
        }
    )

    def __init__(
        self,
        memory: Any = None,
        city_coords: Optional[CityCoords] = None,
        local_tz: Optional[str] = None,
        *,
        hermes: Any = None,
        artemis: Any = None,
        dionysus: Any = None,
        notify: Optional[Callable[[str], None]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        skip_public_holidays: bool = False,
        holiday_country: Optional[str] = None,
        exports_dir: Optional[str] = None,
        default_snooze_minutes: float = 10,
        proactive_weather: bool = False,
    ) -> None:
        self._memory = memory
        self._coords: CityCoords = {**_CITY_COORDS, **(city_coords or {})}
        self._tz = _resolve_tz(local_tz)
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))
        self._hermes = hermes
        self._artemis = artemis
        self._dionysus = dionysus
        self._notify = notify or _default_notify
        self._country = (holiday_country or _DEFAULT_COUNTRY).upper()
        self._exports_dir = Path(exports_dir or _DEFAULT_EXPORTS_DIR)
        self._proactive_weather = proactive_weather
        self._weather_checked_on: Optional[date] = None
        self._weather_attempts = 0
        self._weather_next_try: Optional[datetime] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # The extended reminder features need the Mnemosyne database. A
        # memory object without it (or no memory at all) keeps the original
        # one-shot behaviour. The check is made on the *class* so that a
        # MagicMock stand-in (which invents any attribute you ask it for)
        # is not mistaken for the real thing.
        self._svc: Optional[ReminderService] = None
        store = getattr(memory, "db", None)
        if store is not None and callable(getattr(type(store), "add_reminder_full", None)):
            lookup = None
            if skip_public_holidays:
                lookup = lambda iso: is_public_holiday(iso, self._country)  # noqa: E731
            self._svc = ReminderService(
                store,
                self._tz,
                clock=self._clock,
                public_holiday_lookup=lookup,
                default_snooze=timedelta(minutes=default_snooze_minutes or 10),
            )
            # Device-location updates (browser GPS / Telegram) feed straight
            # into location-triggered reminders (#82).
            adder = getattr(memory, "add_location_listener", None)
            if callable(adder):
                try:
                    adder(self.on_location)
                except Exception:
                    logger.exception("Could not register the location listener.")
        logger.info(
            "ChronosEngine ready (tz=%s, cities=%d, reminders=%s).",
            self._tz,
            len(self._coords),
            "full" if self._svc else "basic",
        )

    def attach_sources(
        self, hermes: Any = None, artemis: Any = None, dionysus: Any = None,
    ) -> None:
        """Late-bind the sibling modules the aggregator/weather features use."""
        if hermes is not None:
            self._hermes = hermes
        if artemis is not None:
            self._artemis = artemis
        if dionysus is not None:
            self._dionysus = dionysus

    @property
    def extended(self) -> bool:
        """True when the database-backed reminder features are available."""
        return self._svc is not None

    def _now_in(self, tz: Any) -> datetime:
        return self._clock().astimezone(tz)

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Dispatch an intent to the appropriate handler.

        Never raises; all errors produce a graceful response dict.
        """
        try:
            return self._dispatch(intent, entities or {}, context or {})
        except Exception:
            logger.exception(
                "ChronosEngine.handle() raised for intent=%s.", intent
            )
            return _err("Something went wrong in the time module.")

    def get_context(self) -> dict:
        """Return a lightweight time context for NLU enrichment."""
        try:
            now = self._now_in(self._tz)
            return {
                "current_time": now.strftime("%I:%M %p"),
                "current_date": now.strftime("%A, %B %d, %Y"),
                "hour": now.hour,
                "day_of_week": now.strftime("%A"),
                "timezone": str(self._tz),
            }
        except Exception:
            logger.exception("get_context() failed.")
            return {}

    # ------------------------------------------------------------------
    # Private – dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, entities: dict, context: dict) -> dict:
        if intent == "get_time":
            # The NLU has been observed to return "get_time" (high confidence)
            # for phrasings like "what is todays date" that are clearly
            # asking for the date, not the clock time — e.g. it fires on
            # "what is" / "todays" and never notices "date" is the actual
            # subject. A raw_query containing "date" but not "time"/"clock"
            # is unambiguous enough to safely redirect rather than answer
            # the wrong question with high confidence.
            raw = (entities.get("raw_query") or "").lower()
            if "date" in raw and "time" not in raw and "clock" not in raw:
                return self._get_date()
            return self._get_time()
        if intent == "get_date":
            return self._get_date()
        if intent == "get_weather":
            return self._get_weather(entities, context)
        if intent == "set_reminder":
            return self._set_reminder(entities)
        if intent == "get_holiday":
            return self._get_holiday(entities)
        if intent == "list_reminders":
            return self._list_reminders(entities)
        if intent == "cancel_reminder":
            return self._cancel_reminder(entities)
        if intent == "snooze_reminder":
            return self._snooze_reminder(entities)
        if intent == "get_agenda":
            return self._get_agenda(entities)
        if intent == "weekly_focus":
            return self._weekly_focus(entities)
        if intent == "mark_holiday":
            return self._mark_holiday(entities, add=True)
        if intent == "unmark_holiday":
            return self._mark_holiday(entities, add=False)
        if intent == "save_place":
            return self._save_place(entities)
        if intent == "export_calendar":
            return self._export_calendar(entities)
        if intent == "import_calendar":
            return self._import_calendar(entities)
        if intent == "weather_plan":
            return self._weather_plan(entities)
        return _err(f"Unknown intent: {intent!r}")

    # ------------------------------------------------------------------
    # Private – time / date
    # ------------------------------------------------------------------

    def _get_time(self) -> dict:
        now = self._now_in(self._tz)
        return _ok(
            f"It's {now.strftime('%I:%M %p')}.",
            data={"time": now.isoformat()},
            confidence=1.0,
        )

    def _get_date(self) -> dict:
        now = self._now_in(self._tz)
        return _ok(
            f"Today is {now.strftime('%A, %B %d, %Y')}.",
            data={"date": now.date().isoformat()},
            confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Private – weather
    # ------------------------------------------------------------------

    def _get_weather(self, entities: dict, context: dict) -> dict:
        location = (
            entities.get("location")
            or (self._memory.get_preference("location") if self._memory else None)
            or _DEFAULT_LOCATION
        )
        location = location.strip()
        coords = self._coords.get(location.lower())

        using_fallback_location = coords is None
        if using_fallback_location:
            logger.info("Weather: location %r not recognized, using default.", location)
            coords = (_DEFAULT_LAT, _DEFAULT_LON)
        lat, lon = coords

        try:
            weather = _fetch_weather(lat, lon)
        except WeatherFetchError:
            logger.exception("Weather fetch failed for location=%r.", location)
            return _err("I couldn't fetch the weather right now.")

        condition = _WMO_CODES.get(
            weather.get("weathercode", weather.get("weather_code", -1)), ""
        )
        condition_str = f", {condition}" if condition else ""
        temp = weather.get("temperature", "?")
        wind = weather.get("windspeed", "?")

        if using_fallback_location:
            return _ok(
                f"I don't have coordinates for {location!r}, so here's the weather "
                f"for {_DEFAULT_LOCATION} instead: {temp}°C{condition_str}, wind {wind} km/h.",
                data={"location": _DEFAULT_LOCATION, "weather": weather, "requested_location": location},
                confidence=0.5,
            )

        return _ok(
            f"Currently in {location}: {temp}°C{condition_str}, "
            f"wind {wind} km/h.",
            data={"location": location, "weather": weather},
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # Private – creating reminders (#81, #82, #84, #85, #87)
    # ------------------------------------------------------------------

    def _set_reminder(self, entities: dict) -> dict:
        if self._svc is None:
            return self._set_reminder_basic(entities)

        raw: str = entities.get("raw_query") or ""
        task_hint = entities.get("task")
        date_hint: str = entities.get("date") or ""
        time_hint: str = entities.get("time") or ""
        # The NLU sometimes gives structured fields but no raw text.
        probe = raw or " ".join(
            str(x) for x in (task_hint, date_hint, time_hint) if x and str(x).lower() != "none"
        )

        # -- per-reminder time zone (#85) --------------------------------
        tz_name = resolve_timezone_name(entities.get("timezone") or entities.get("tz"))
        found_tz, work = extract_timezone(probe)
        tz_name = tz_name or found_tz
        rtz = to_zoneinfo(tz_name, self._tz)
        now = self._now_in(rtz)

        # -- holiday wording (#87) ---------------------------------------
        skip_holidays = False
        if _SKIP_HOLIDAYS_RE.search(work):
            skip_holidays = True
            work = _squash(_SKIP_HOLIDAYS_RE.sub(" ", work))
        if _WORKING_DAY_RE.search(work):
            skip_holidays = True

        # -- "when I get home" (#82) -------------------------------------
        place_label, work_no_place = _extract_place(work)

        # -- recurrence (#81, #84) ---------------------------------------
        unsupported = unsupported_recurrence_reason(work_no_place)
        if unsupported:
            return _err(unsupported)
        rule, work_rec = extract_recurrence(work_no_place, now, rtz)
        if rule is not None and time_hint and not _has_clock_time(work_no_place):
            # "every weekday" + a separate time entity ("7am")
            if parse_time_of_day(time_hint) is not None:
                rule2, work_rec2 = extract_recurrence(f"{work_no_place} at {time_hint}", now, rtz)
                if rule2 is not None:
                    rule, work_rec = rule2, work_rec2

        if place_label and rule is not None:
            return _clarify(
                "I can remind you either when you arrive somewhere or on a repeating "
                "schedule, but not both in one reminder yet. Which would you like?"
            )

        if place_label:
            task = _extract_task(_squash(work_no_place), _clean_task_hint(task_hint, now, rtz))
            return self._create_place_reminder(task, place_label)

        if rule is not None:
            return self._create_recurring(
                rule, work_rec, task_hint, now, rtz, tz_name, skip_holidays,
            )
        return self._create_one_shot(
            work_no_place, task_hint, date_hint, time_hint, now, rtz, tz_name, skip_holidays,
        )

    def _set_reminder_basic(self, entities: dict) -> dict:
        """The original one-shot behaviour (no extended database)."""
        raw: str = entities.get("raw_query") or ""
        task = _extract_task(raw=raw, task_hint=entities.get("task"))
        date_hint: str = entities.get("date") or ""
        time_hint: str = entities.get("time") or ""
        now = self._now_in(self._tz)

        if raw and unsupported_recurrence_reason(raw):
            return _err(unsupported_recurrence_reason(raw))
        if raw and extract_recurrence(raw, now, self._tz)[0] is not None:
            return _clarify(
                "I can't set repeating reminders without my memory module running."
            )

        try:
            due_dt = _parse_reminder_time(
                raw=raw, date_hint=date_hint, time_hint=time_hint, base=now,
            )
        except ReminderParseError as exc:
            logger.warning("_set_reminder: time parse failed: %s", exc)
            return _clarify(
                "I couldn't understand the reminder time. "
                "Try something like 'remind me to call John at 3 PM tomorrow'."
            )

        due_iso = due_dt.isoformat()
        if self._memory:
            try:
                self._memory.add_reminder(task, due_iso)
            except Exception:
                logger.exception("_set_reminder: failed to persist reminder (task=%r).", task)
                return _err("I understood the reminder but couldn't save it.")

        readable = due_dt.strftime("%A, %B %d at %I:%M %p")
        logger.info("Reminder set: task=%r due=%s", task, due_iso)
        return _ok(
            f"Reminder set: {task!r} on {readable}.",
            data={"task": task, "due": due_iso},
            confidence=0.95,
        )

    def _create_one_shot(
        self, work: str, task_hint: Any, date_hint: str, time_hint: str,
        now: datetime, rtz: Any, tz_name: Optional[str], skip_holidays: bool,
    ) -> dict:
        assert self._svc is not None
        try:
            due = _parse_reminder_time(raw=work, date_hint=date_hint, time_hint=time_hint, base=now)
        except ReminderParseError as exc:
            logger.warning("_set_reminder: time parse failed: %s", exc)
            return _clarify(
                "I couldn't understand the reminder time. "
                "Try something like 'remind me to call John at 3 PM tomorrow'."
            )
        task = _extract_task(raw=work, task_hint=_clean_task_hint(task_hint, now, rtz))

        rolled_note = ""
        if skip_holidays:
            new_due, rolled = self._svc.next_non_holiday(due, rtz)
            if rolled:
                label = self._svc.holiday_label(due.astimezone(rtz).date()) or "a holiday"
                rolled_note = (
                    f" That day is {label}, so I moved it to "
                    f"{describe_when(new_due, rtz, now)}."
                )
                due = new_due

        if self._svc.store.reminder_exists(task, due.replace(microsecond=0).isoformat()):
            return _ok(
                f"You already have a reminder for {task!r} at that time.",
                data={"task": task, "due": due.isoformat(), "duplicate": True},
                confidence=0.9,
            )
        try:
            rid = self._svc.create(task, due, tz_name=tz_name, skip_holidays=skip_holidays)
        except Exception:
            logger.exception("_set_reminder: failed to persist reminder (task=%r).", task)
            return _err("I understood the reminder but couldn't save it.")

        readable = due.strftime("%A, %B %d at %I:%M %p")
        tz_note = f" ({_zone_label(tz_name)})" if tz_name else ""
        holiday_note = " I'll skip it on holidays." if skip_holidays and not rolled_note else ""
        logger.info("Reminder set: task=%r due=%s", task, due.isoformat())
        return _ok(
            f"Reminder set: {task!r} on {readable}{tz_note}.{rolled_note}{holiday_note}",
            data={"id": rid, "task": task, "due": due.isoformat(), "timezone": tz_name,
                  "skip_holidays": skip_holidays},
            confidence=0.95,
        )

    def _create_recurring(
        self, rule: Recurrence, remaining: str, task_hint: Any, now: datetime,
        rtz: Any, tz_name: Optional[str], skip_holidays: bool,
    ) -> dict:
        assert self._svc is not None
        task = _extract_task(raw=remaining, task_hint=_clean_task_hint(task_hint, now, rtz))
        first = parse_iso(rule.anchor)
        if first is None:
            return _clarify("I couldn't work out when that should repeat first.")
        first = first.astimezone(rtz)

        # Study-like repeating reminders skip user-marked days off by default.
        auto_skip = False
        if not skip_holidays and _HOLIDAY_SENSITIVE_RE.search(task):
            skip_holidays = auto_skip = True

        if self._svc.store.reminder_exists(task, first.replace(microsecond=0).isoformat()):
            return _ok(
                f"You already have a repeating reminder for {task!r}.",
                data={"task": task, "duplicate": True}, confidence=0.9,
            )
        try:
            rid = self._svc.create(
                task, first, tz_name=tz_name, rule=rule, skip_holidays=skip_holidays,
            )
        except Exception:
            logger.exception("_set_reminder: failed to persist recurring reminder (task=%r).", task)
            return _err("I understood the reminder but couldn't save it.")

        when = describe_when(first, rtz, now)
        tz_note = f" ({_zone_label(tz_name)})" if tz_name else ""
        extra = ""
        if skip_holidays:
            extra = (
                " I'll skip days you've marked as holidays."
                if not auto_skip
                else " I'll skip days you've marked as holidays, since it sounds like a work or study day."
            )
        logger.info("Recurring reminder set: task=%r rule=%s first=%s", task, rule.to_json(), first.isoformat())
        return _ok(
            f"Okay, I'll remind you to {task} {rule.describe(rtz)}{tz_note}. "
            f"The first one is {when}.{extra}",
            data={
                "id": rid, "task": task, "due": first.isoformat(), "recurring": True,
                "recurrence": rule.describe(rtz), "timezone": tz_name,
                "skip_holidays": skip_holidays,
            },
            confidence=0.95,
        )

    def _create_place_reminder(self, task: str, label: str) -> dict:
        assert self._svc is not None
        place = self._lookup_place(label)
        if place is None:
            return _clarify(
                f"I don't know where {label!r} is yet. When you're there, say "
                f"'this is {label}' and I'll remember it, then ask me again."
            )
        matched_label, lat, lon = place
        try:
            rid = self._svc.create(
                task, None,
                place={"label": matched_label, "lat": lat, "lon": lon,
                       "radius_m": DEFAULT_RADIUS_M, "armed": False},
            )
        except Exception:
            logger.exception("_set_reminder: failed to persist location reminder (task=%r).", task)
            return _err("I understood the reminder but couldn't save it.")

        armed = False
        loc = self._device_location()
        if loc is not None:
            armed = self._svc.arm_if_outside(rid, loc[0], loc[1])
        if armed:
            note = ""
        elif loc is None:
            note = (
                " I don't know where you are right now, so it will start watching once "
                f"I see you away from {matched_label}."
            )
        else:
            note = f" You're at {matched_label} now, so I'll wait until you've left and come back."
        return _ok(
            f"Okay, I'll remind you to {task} when you {_arrival_phrase(matched_label)}.{note}",
            data={"id": rid, "task": task, "place": matched_label, "armed": armed},
            confidence=0.9,
        )

    def _lookup_place(self, label: str) -> Optional[tuple[str, float, float]]:
        assert self._svc is not None
        for cand in _PLACE_SYNONYMS.get(label, (label,)):
            try:
                row = self._svc.store.get_place(cand)
            except Exception:
                logger.exception("place lookup failed for %r", cand)
                row = None
            if row:
                return row["label"], float(row["lat"]), float(row["lon"])
        return None

    def _device_location(self) -> Optional[tuple[float, float]]:
        getter = getattr(self._memory, "get_device_location", None)
        if not callable(getter):
            return None
        try:
            loc = getter()
        except Exception:
            return None
        if not isinstance(loc, dict):
            return None
        try:
            return float(loc["lat"]), float(loc["lon"])
        except (KeyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Private – listing / cancelling / snoozing (#83)
    # ------------------------------------------------------------------

    def _need_extended(self, what: str) -> dict:
        return _err(
            f"I can't {what} because my memory module isn't running, so I have "
            "no stored reminders to work with."
        )

    def _list_reminders(self, entities: dict) -> dict:
        if self._svc is None:
            return self._need_extended("list your reminders")
        raw = entities.get("raw_query") or ""
        now = self._now_in(self._tz)

        if _MISSED_RE.search(raw):
            rows = self._svc.recently_missed(now=now)
            if not rows:
                return _ok("You haven't missed any reminders in the past week.",
                           data={"missed": []}, confidence=0.9)
            lines = [f"{r['text']} ({describe_when(parse_iso(r.get('due_time') or r.get('fired_at')), self._svc.zone_for(r), now)})"
                     for r in rows[:_SPOKEN_REMINDER_LIMIT]]
            n = len(rows)
            return _ok(
                f"You missed {n} reminder{'s' if n != 1 else ''} recently: " + "; ".join(lines) + ".",
                data={"missed": [_row_summary(r) for r in rows]}, confidence=0.9,
            )

        rows = self._svc.list_pending()
        hint = _command_hint(raw, entities.get("task"), extra_noise=_LIST_NOISE)
        if hint:
            rows = [r for r in rows if hint.lower() in (r.get("text") or "").lower()]
        if not rows:
            msg = f"You have no reminders about {hint!r}." if hint else "You have no reminders set."
            return _ok(msg, data={"reminders": []}, confidence=0.9)
        n = len(rows)
        parts = [f"{i}. {self._svc.describe_row(r, now)}"
                 for i, r in enumerate(rows[:_SPOKEN_REMINDER_LIMIT], start=1)]
        more = f" And {n - _SPOKEN_REMINDER_LIMIT} more." if n > _SPOKEN_REMINDER_LIMIT else ""
        return _ok(
            f"You have {n} reminder{'s' if n != 1 else ''}. " + " ".join(p + "." for p in parts) + more,
            data={"reminders": [_row_summary(r) for r in rows]}, confidence=0.95,
        )

    def _cancel_reminder(self, entities: dict) -> dict:
        if self._svc is None:
            return self._need_extended("cancel reminders")
        raw = entities.get("raw_query") or ""
        now = self._now_in(self._tz)
        pending = self._svc.list_pending()
        if not pending:
            return _ok("You have no reminders to cancel.", data={"cancelled": []}, confidence=0.9)

        # "cancel reminder 2" refers to the numbering `list_reminders` speaks.
        m = _INDEX_RE.search(raw)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= len(pending):
                return self._cancel_rows([pending[idx - 1]], now)
            return _clarify(f"You only have {len(pending)} reminder{'s' if len(pending) != 1 else ''}.")

        hint = _command_hint(raw, entities.get("task"))
        wants_all = bool(_ALL_RE.search(raw)) and not entities.get("task")
        if not hint:
            if wants_all:
                return self._cancel_rows(pending, now)
            if len(pending) == 1:
                return self._cancel_rows(pending, now)
            return self._ask_which(pending, "cancel", now)

        matches = self._svc.find_pending(hint)
        if not matches:
            # Fall back to the longest single word ("cancel my dentist appointment reminder").
            for word in sorted(set(hint.split()), key=len, reverse=True):
                if len(word) >= 4:
                    matches = self._svc.find_pending(word)
                    if matches:
                        break
        if not matches:
            return _ok(f"I couldn't find a reminder matching {hint!r}.",
                       data={"cancelled": []}, confidence=0.6)
        if len(matches) > 1 and not wants_all:
            return self._ask_which(matches, "cancel", now)
        return self._cancel_rows(matches, now)

    def _cancel_rows(self, rows: list[dict[str, Any]], now: datetime) -> dict:
        assert self._svc is not None
        done = [r for r in rows if self._svc.cancel(r["id"])]
        if not done:
            return _err("I couldn't cancel that reminder.")
        if len(done) == 1:
            r = done[0]
            rule = ReminderService._rule_of(r)
            kind = "repeating reminder" if rule is not None else "reminder"
            return _ok(f"Cancelled the {kind} {r['text']!r}.",
                       data={"cancelled": [r["id"]]}, confidence=0.95)
        names = ", ".join(repr(r["text"]) for r in done[:5])
        more = f" and {len(done) - 5} more" if len(done) > 5 else ""
        return _ok(f"Cancelled {len(done)} reminders: {names}{more}.",
                   data={"cancelled": [r["id"] for r in done]}, confidence=0.95)

    def _ask_which(self, rows: list[dict[str, Any]], verb: str, now: datetime) -> dict:
        assert self._svc is not None
        lines = [f"{r['text']} ({describe_when(self._svc._due_of(r), self._svc.zone_for(r), now)})"
                 for r in rows[:5]]
        return _clarify(
            f"Which one do you want to {verb}? " + "; ".join(lines) + ". "
            "You can say part of its name."
        )

    def _snooze_reminder(self, entities: dict) -> dict:
        if self._svc is None:
            return self._need_extended("snooze reminders")
        raw = entities.get("raw_query") or ""
        now = self._now_in(self._tz)
        duration = parse_duration(str(entities.get("duration") or "")) or parse_duration(raw)
        hint = _command_hint(strip_duration(raw), entities.get("task"))
        result = self._svc.snooze(text_hint=hint or None, duration=duration, now=now)
        if not result.get("ok"):
            reason = result.get("reason")
            if reason == "location":
                msg = "That's a place-based reminder, so there's no time to snooze."
            elif reason == "cancelled":
                msg = "That reminder was cancelled, so there's nothing to snooze."
            elif hint:
                msg = f"I couldn't find a reminder matching {hint!r} to snooze."
            else:
                msg = "There's nothing to snooze. No reminder has gone off recently."
            return _ok(msg, data={"snoozed": False, "reason": reason}, confidence=0.6)
        tz = result["tz"]
        when = describe_when(result["due"], tz, now)
        span = _fmt_duration(result["duration"])
        if result["mode"] == "postponed":
            msg = f"Pushed {result['text']!r} back by {span}. It's now due {when}."
        else:
            msg = f"Snoozed {result['text']!r} for {span}. I'll remind you again {when}."
        return _ok(
            msg,
            data={"snoozed": True, "id": result["id"], "text": result["text"],
                  "due": result["due"].isoformat(), "mode": result["mode"],
                  "snooze_count": result["snooze_count"]},
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # Private – "what's on my plate" (#86)
    # ------------------------------------------------------------------

    def _get_agenda(self, entities: dict) -> dict:
        now = self._now_in(self._tz)
        day = _resolve_day(entities.get("raw_query") or "", entities.get("date") or "", now)
        agenda = agenda_mod.build_agenda(
            self._svc, day, self._tz, now, hermes=self._hermes, artemis=self._artemis,
        )
        text = agenda_mod.format_agenda(agenda, now.date(), self._tz)
        return _ok(
            text,
            data={
                "date": day.isoformat(),
                "holiday": agenda.holiday,
                "items": [i.to_dict() for i in agenda.items],
                "notes": agenda.notes,
            },
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # Private – what needs attention this week (#163)
    # ------------------------------------------------------------------

    def _weekly_focus(self, entities: dict) -> dict:
        """Rank the next seven days' goals, deadlines, reminders and streaks
        into the few that need attention. Read-only; works with whichever
        sources are attached."""
        now = self._now_in(self._tz)
        focus = agenda_mod.build_week_focus(
            self._svc, self._tz, now, hermes=self._hermes, artemis=self._artemis,
        )
        return _ok(
            agenda_mod.format_week_focus(focus),
            data={
                "start": focus.start.isoformat(),
                "days": focus.days,
                "considered": focus.considered,
                "items": [i.to_dict() for i in focus.items],
                "notes": focus.notes,
            },
            confidence=0.9,
        )

    # ------------------------------------------------------------------
    # Private – holidays the user marks (#87)
    # ------------------------------------------------------------------

    def _mark_holiday(self, entities: dict, *, add: bool) -> dict:
        if self._svc is None:
            return self._need_extended("track holidays")
        raw = entities.get("raw_query") or ""
        now = self._now_in(self._tz)
        days = _extract_day_range(raw, entities.get("date") or "", now)
        if days is None:
            upcoming = self._svc.store.list_holidays(now.date().isoformat())
            if upcoming and not add:
                pass
            if not add or not upcoming:
                return _clarify("Which day should I " + ("mark as a holiday?" if add else "unmark?"))
            names = ", ".join(_day_phrase(date.fromisoformat(h["day"]), now.date()) for h in upcoming[:5])
            return _ok(f"Your upcoming marked holidays: {names}.",
                       data={"holidays": upcoming}, confidence=0.7)
        if len(days) > _MAX_HOLIDAY_RANGE_DAYS:
            return _clarify(f"That's more than {_MAX_HOLIDAY_RANGE_DAYS} days. Give me a shorter range.")

        label = _holiday_label(raw, entities.get("label"))
        today = now.date()
        span = (_day_phrase(days[0], today) if len(days) == 1
                else f"{_day_phrase(days[0], today)} through {_day_phrase(days[-1], today)}")
        if add:
            new = [d for d in days if self._svc.store.add_holiday(d.isoformat(), label)]
            tag = f" ({label})" if label else ""
            if not new:
                return _ok(f"{span.capitalize()} {'was' if len(days) == 1 else 'were'} already marked as a holiday.",
                           data={"marked": [], "days": [d.isoformat() for d in days]}, confidence=0.9)
            what = "as a holiday" if len(days) == 1 else f"as holidays ({len(days)} days)"
            return _ok(
                f"Marked {span} {what}{tag}. Repeating study or work reminders, and any set to skip "
                "holidays, won't fire then.",
                data={"marked": [d.isoformat() for d in new], "label": label}, confidence=0.95,
            )
        removed = [d for d in days if self._svc.store.remove_holiday(d.isoformat())]
        if not removed:
            return _ok(f"{span.capitalize()} wasn't marked as a holiday.",
                       data={"unmarked": []}, confidence=0.8)
        return _ok(f"Unmarked {span}. Reminders will fire normally again.",
                   data={"unmarked": [d.isoformat() for d in removed]}, confidence=0.95)

    # ------------------------------------------------------------------
    # Private – named places (#82)
    # ------------------------------------------------------------------

    def _save_place(self, entities: dict) -> dict:
        if self._svc is None:
            return self._need_extended("remember places")
        raw = entities.get("raw_query") or ""
        label = _place_label_from_text(
            str(entities.get("place") or entities.get("label") or entities.get("name") or ""), raw,
        )
        if not label:
            return _clarify("What should I call this place? For example, 'this is home'.")

        coords: Optional[tuple[float, float]] = None
        try:
            if entities.get("lat") is not None and entities.get("lon") is not None:
                coords = (float(entities["lat"]), float(entities["lon"]))
        except (TypeError, ValueError):
            coords = None
        if coords is None:
            coords = self._device_location()
        if coords is None:
            return _err(
                "I don't know where you are right now. Share your location from the web app "
                "or Telegram, then tell me again."
            )
        try:
            self._svc.store.set_place(label, coords[0], coords[1])
        except Exception:
            logger.exception("save_place failed for %r", label)
            return _err("I couldn't save that place.")
        return _ok(
            f"Got it. I'll remember this spot as {label}. Now you can say "
            f"'remind me to ... when I get {label}'.",
            data={"place": label, "lat": coords[0], "lon": coords[1]}, confidence=0.95,
        )

    # ------------------------------------------------------------------
    # Private – ICS export / import (#90)
    # ------------------------------------------------------------------

    def _export_calendar(self, entities: dict) -> dict:
        if self._svc is None:
            return self._need_extended("export reminders")
        text, count, notes = self._svc.export_ics()
        if count == 0:
            extra = " " + " ".join(notes) if notes else ""
            return _ok("You have no exportable reminders." + extra,
                       data={"count": 0, "notes": notes}, confidence=0.8)
        stamp = self._now_in(self._tz).strftime("%Y%m%d_%H%M%S")
        path = self._exports_dir / f"hestia_reminders_{stamp}.ics"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        except OSError:
            logger.exception("export_calendar: could not write %s", path)
            return _err("I couldn't write the calendar file.")
        msg = (
            f"Exported {count} reminder{'s' if count != 1 else ''} to {path}. "
            "You can import that file into any calendar app."
        )
        if notes:
            msg += " Note: " + " ".join(notes[:3])
        return _ok(msg, data={"path": str(path), "count": count, "notes": notes}, confidence=0.95)

    def _import_calendar(self, entities: dict) -> dict:
        if self._svc is None:
            return self._need_extended("import a calendar")
        text = entities.get("ics_text")
        source = "the calendar data"
        if not text:
            raw = entities.get("raw_query") or ""
            candidate = entities.get("path") or entities.get("file")
            if not candidate:
                m = _ICS_PATH_RE.search(raw)
                candidate = m.group("path") if m else None
            if not candidate:
                return _clarify("Which calendar file? Give me the path to a .ics file.")
            path = Path(os.path.expanduser(str(candidate).strip().strip("'\"")))
            if path.suffix.lower() != ".ics":
                return _err("I can only import .ics calendar files.")
            try:
                if not path.is_file():
                    return _err(f"I couldn't find a file at {path}.")
                if path.stat().st_size > _ICS_MAX_BYTES:
                    return _err("That calendar file is too large to import (over 1 MB).")
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.exception("import_calendar: could not read %s", path)
                return _err("I couldn't read that calendar file.")
            source = path.name
        elif len(str(text).encode("utf-8", errors="ignore")) > _ICS_MAX_BYTES:
            return _err("That calendar data is too large to import (over 1 MB).")

        result = self._svc.import_ics(str(text))
        n, skipped = len(result.created), len(result.skipped)
        if n == 0 and skipped == 0:
            return _ok(f"I didn't find any reminders I could import from {source}.",
                       data={"created": [], "skipped": [], "warnings": result.warnings}, confidence=0.7)
        parts = [f"Imported {n} reminder{'s' if n != 1 else ''} from {source}."]
        if skipped:
            parts.append(f"Skipped {skipped}: " + " ".join(result.skipped[:3]))
        if result.warnings:
            parts.append("Note: " + " ".join(result.warnings[:3]))
        return _ok(" ".join(parts),
                   data={"created": result.created, "skipped": result.skipped,
                         "warnings": result.warnings}, confidence=0.95)

    # ------------------------------------------------------------------
    # Private – weather-triggered suggestions (#88)
    # ------------------------------------------------------------------

    def _weather_plan(self, entities: dict) -> dict:
        raw = entities.get("raw_query") or ""
        now = self._now_in(self._tz)
        day = _resolve_day(raw, entities.get("date") or "", now)
        activity = str(entities.get("activity") or entities.get("task") or "").strip()

        items = agenda_mod.build_agenda(
            self._svc, day, self._tz, now, hermes=self._hermes, artemis=self._artemis,
        ).items
        adhoc = None
        if activity or agenda_mod.is_outdoor(raw):
            text = activity or _activity_from_text(raw)
            adhoc = agenda_mod.AgendaItem(None, "plan", text or "your plan")
            items = [*items, adhoc]

        try:
            forecast = self._forecast_for(entities)
        except WeatherFetchError:
            logger.exception("weather_plan: forecast fetch failed.")
            return _err("I couldn't fetch the forecast right now.")

        concerns = agenda_mod.assess_agenda(items, forecast, self._tz)
        when_label = agenda_mod.day_label(day, now.date())
        outlook = agenda_mod.rain_outlook(forecast, day, self._tz)
        data: dict[str, Any] = {
            "date": day.isoformat(),
            "concerns": [c.to_dict() for c in concerns],
            "rain_outlook": {"probability": outlook[0], "peak": outlook[1]} if outlook else None,
        }
        if outlook is None:
            return _ok(f"The forecast doesn't cover {when_label} yet.", data=data, confidence=0.6)

        if not concerns:
            outdoor = [i for i in items if agenda_mod.is_outdoor(i.text)]
            if outdoor:
                msg = (f"No rain worries for {when_label}. The peak chance is {outlook[0]}% "
                       f"around {outlook[1]}, so {outdoor[0].text!r} looks fine.")
            else:
                msg = (f"You've no outdoor plans {when_label}. For what it's worth, the peak rain "
                       f"chance is {outlook[0]}% around {outlook[1]}.")
            return _ok(msg, data=data, confidence=0.9)

        msg = agenda_mod.format_concerns(concerns, self._tz)
        wants_alt = bool(re.search(r"\b(?:indoor|indoors|instead|alternative|else|rain\s*plan)\b", raw, re.I))
        alt = self._indoor_alternative(concerns[0]) if wants_alt else None
        if alt:
            msg += f" Here's an indoor idea: {alt}"
            data["alternative"] = alt
        else:
            msg += " Want me to suggest something indoors instead?" if self._dionysus is not None \
                else " You may want to move it or bring an umbrella."
        return _ok(msg, data=data, confidence=0.9)

    def _indoor_alternative(self, concern: "agenda_mod.WeatherConcern") -> Optional[str]:
        """Ask Dionysus's outing planner for an indoor alternative."""
        if self._dionysus is None:
            return None
        topic = f"an indoor alternative to {concern.item.text}, because of {concern.condition}"
        try:
            result = self._dionysus.handle(
                "plan_outing", {"topic": topic, "raw_query": topic}, {},
            )
        except Exception:
            logger.exception("weather_plan: Dionysus outing planner failed.")
            return None
        if not isinstance(result, dict) or float(result.get("confidence") or 0) < 0.5:
            return None
        return (result.get("response") or "").strip() or None

    def _forecast_for(self, entities: dict) -> dict[str, Any]:
        _, lat, lon = self._resolve_coords(entities)
        tz_key = getattr(self._tz, "key", None) or "UTC"
        return _fetch_forecast(lat, lon, tz_key)

    def _resolve_coords(self, entities: dict) -> tuple[str, float, float]:
        loc = (entities.get("location") or "").strip()
        if not loc and self._memory is not None:
            getter = getattr(self._memory, "get_preference", None)
            if callable(getter):
                try:
                    loc = str(getter("location") or "").strip()
                except Exception:
                    loc = ""
        coords = self._coords.get(loc.lower()) if loc else None
        if coords is not None:
            return loc, coords[0], coords[1]
        dev = self._device_location()
        if dev is not None:
            return "your location", dev[0], dev[1]
        return _DEFAULT_LOCATION, _DEFAULT_LAT, _DEFAULT_LON

    def _maybe_proactive_weather(self, now: Optional[datetime] = None) -> Optional[str]:
        """Once each morning, warn if rain threatens an outdoor plan today."""
        if not self._proactive_weather:
            return None
        now = now or self._now_in(self._tz)
        today = now.date()
        if self._weather_checked_on == today or now.hour < _PROACTIVE_WEATHER_FROM_HOUR:
            return None
        if self._weather_next_try is not None and now < self._weather_next_try:
            return None
        try:
            forecast = self._forecast_for({})
        except WeatherFetchError:
            self._weather_attempts += 1
            if self._weather_attempts >= _WEATHER_MAX_ATTEMPTS:
                self._weather_checked_on, self._weather_attempts = today, 0
                self._weather_next_try = None
            else:
                self._weather_next_try = now + timedelta(minutes=_WEATHER_RETRY_MINUTES)
            return None
        self._weather_checked_on, self._weather_attempts, self._weather_next_try = today, 0, None
        try:
            agenda = agenda_mod.build_agenda(
                self._svc, today, self._tz, now, hermes=self._hermes, artemis=self._artemis,
            )
            concerns = agenda_mod.assess_agenda(agenda.items, forecast, self._tz)
        except Exception:
            logger.exception("proactive weather check failed.")
            return None
        if not concerns:
            return None
        msg = "Heads up: " + agenda_mod.format_concerns(concerns, self._tz)
        if self._dionysus is not None:
            msg += " Ask me for an indoor alternative if you like."
        self._safe_notify(msg)
        return msg

    # ------------------------------------------------------------------
    # Scheduler, announcements, location feed (#81, #82, #89)
    # ------------------------------------------------------------------

    def poll_once(self) -> list:
        """Fire everything that is due right now and announce it. Anything
        overdue by more than the grace period (Hestia was off) is announced
        once, flagged as missed, rather than dropped (#89)."""
        if self._svc is None:
            return []
        fired = self._svc.tick()
        self._announce(fired)
        try:
            self._maybe_proactive_weather()
        except Exception:
            logger.exception("proactive weather raised.")
        return fired

    def on_location(self, lat: float, lon: float, *_ignored: Any) -> list:
        """Feed a device position in (registered as a Mnemosyne location
        listener). Fires location reminders whose place was just reached."""
        if self._svc is None:
            return []
        try:
            fired = self._svc.on_location(float(lat), float(lon))
        except Exception:
            logger.exception("on_location failed.")
            return []
        self._announce(fired)
        return fired

    def _announce(self, fired: list) -> None:
        if not fired or self._svc is None:
            return
        for line in self._svc.format_fired(fired):
            self._safe_notify(line)
        for f in fired:
            if getattr(f, "series_ended", False):
                self._safe_notify(f"That was the last repeat of {f.text!r}.")

    def _safe_notify(self, text: str) -> None:
        try:
            self._notify(text)
        except Exception:
            logger.exception("notify callback failed for %r", text)

    def start_scheduler(self, interval: Optional[float] = None) -> bool:
        """Start the background reminder loop. Returns False when there's
        nothing to schedule (no database) or it's already running. The first
        pass runs immediately, which is the startup catch-up (#89)."""
        if self._svc is None or (self._thread is not None and self._thread.is_alive()):
            return False
        step = float(interval or _SCHEDULER_INTERVAL_S)
        self._stop.clear()

        def _loop() -> None:
            while True:
                try:
                    self.poll_once()
                except Exception:
                    logger.exception("scheduler pass failed.")
                if self._stop.wait(step):
                    return

        self._thread = threading.Thread(target=_loop, name="chronos-scheduler", daemon=True)
        self._thread.start()
        logger.info("Chronos scheduler started (every %.0fs).", step)
        return True

    def stop_scheduler(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)
        self._thread = None

    @property
    def scheduler_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Private – holidays (backed by core.free_apis / Nager.Date, no key)
    # ------------------------------------------------------------------

    def _get_holiday(self, entities: dict) -> dict:
        """
        Answer "is <date> a public holiday" or "what holidays are there
        in <country>". Defaults to today's local date and
        ``_DEFAULT_COUNTRY`` when not specified in entities.
        """
        raw: str = entities.get("raw_query") or ""
        country = (entities.get("country") or _DEFAULT_COUNTRY).strip().upper()

        date_hint = entities.get("date")
        settings = {"RELATIVE_BASE": self._now_in(self._tz).replace(tzinfo=None)}
        iso_date: Optional[str] = None
        if date_hint:
            try:
                parsed = dateparser.parse(date_hint, settings=settings)
            except Exception:
                parsed = None
            if parsed:
                iso_date = parsed.date().isoformat()
        elif raw:
            try:
                found = search_dates(raw, settings=settings)
            except Exception:
                found = None
            if found:
                iso_date = found[0][1].date().isoformat()

        # Broad request ("what holidays are there in France this year")
        # with no specific date resolved -> list the whole year.
        if not iso_date:
            year = self._now_in(self._tz).year
            try:
                holidays = public_holidays(year, country)
            except FreeAPIError:
                logger.warning("_get_holiday: public_holidays(%s, %s) failed.", year, country)
                return _err(
                    f"I couldn't reach the holiday calendar for {country} right now."
                )
            if not holidays:
                return _ok(
                    f"I couldn't find any public holidays for {country} in {year}.",
                    data={"country": country, "year": year, "holidays": []},
                    confidence=0.7,
                )
            names = ", ".join(h.get("localName") or h.get("name", "") for h in holidays[:5])
            more = f" and {len(holidays) - 5} more" if len(holidays) > 5 else ""
            return _ok(
                f"{country} has {len(holidays)} public holidays in {year}, including {names}{more}.",
                data={"country": country, "year": year, "holidays": holidays},
                confidence=0.9,
            )

        try:
            name = is_public_holiday(iso_date, country)
        except FreeAPIError:
            logger.warning("_get_holiday: is_public_holiday(%s, %s) failed.", iso_date, country)
            return _err(
                f"I couldn't reach the holiday calendar for {country} right now."
            )

        if name:
            return _ok(
                f"Yes — {iso_date} is {name} in {country}.",
                data={"date": iso_date, "country": country, "is_holiday": True, "name": name},
                confidence=0.95,
            )
        return _ok(
            f"No, {iso_date} is not a public holiday in {country}.",
            data={"date": iso_date, "country": country, "is_holiday": False},
            confidence=0.9,
        )


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

_LIST_NOISE = frozenset({
    "what", "whats", "what's", "which", "show", "list", "tell", "give", "do", "i", "have",
    "are", "is", "there", "any", "set", "got", "upcoming", "pending", "current", "active",
    "coming", "reminder", "reminders", "my", "the", "me", "please", "about", "for", "on",
    "regarding", "concerning", "with", "all", "of", "a", "an", "to", "and", "hestia",
    "scheduled", "next", "how", "many", "you", "can", "could", "see",
})


def _now(tz: Any) -> datetime:
    """Return the current moment as a timezone-aware datetime."""
    return datetime.now(tz)


def _default_notify(text: str) -> None:
    """Announce *text* through the event bus (the assistant speaks it)."""
    try:
        from core.event_bus import bus
        bus.emit("speak", {"text": text})
    except Exception:
        logger.exception("Could not announce %r on the event bus.", text)


def _resolve_tz(tz_str: Optional[str]) -> Any:
    """
    Return a ZoneInfo object for *tz_str*, falling back to UTC on failure.
    """
    if not tz_str:
        return timezone.utc
    try:
        return ZoneInfo(tz_str)
    except (ZoneInfoNotFoundError, KeyError):
        logger.warning("Unknown timezone %r; defaulting to UTC.", tz_str)
        return timezone.utc


def _squash(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


def _zone_label(tz_name: Optional[str]) -> str:
    """'Europe/London' -> 'London time' for speech."""
    if not tz_name:
        return ""
    city = tz_name.split("/")[-1].replace("_", " ")
    return "UTC" if tz_name.upper() == "UTC" else f"{city} time"


def _has_clock_time(text: str) -> bool:
    if parse_time_of_day(text) is not None:
        return True
    return bool(re.search(r"\bat\s+\d{1,2}(?::\d{2})?\b", text, re.IGNORECASE))


def _fmt_duration(td: timedelta) -> str:
    total = int(td.total_seconds())
    if total >= 86400 and total % 86400 == 0:
        d = total // 86400
        return f"{d} day{'s' if d != 1 else ''}"
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes or not hours:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts)


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "text", "due_time", "status", "tz", "recurrence", "skip_holidays",
            "place_label", "fired_at", "missed", "snooze_count")
    return {k: row.get(k) for k in keys}


def _clean_task_hint(hint: Any, now: datetime, tz: Any) -> Optional[str]:
    """An NLU task hint may still carry scheduling words ("study every
    weekday"); strip them so they don't end up in the label."""
    if not hint or str(hint).strip().lower() in {"", "none", "null"}:
        return None
    text = str(hint).strip()
    _, text = extract_timezone(text)
    _, text = extract_recurrence(text, now, tz)
    _, text = _extract_place(text)
    text = _squash(_SKIP_HOLIDAYS_RE.sub(" ", text))
    return text or None


def _arrival_phrase(label: str) -> str:
    """"home" -> "get home", "office" -> "get to the office", else "get to X"."""
    label = (label or "").strip()
    if label == "home":
        return "get home"
    if label in ("office", "gym", "station", "airport", "library"):
        return f"get to the {label}"
    return f"get to {label}"


def _normalise_place(label: str) -> str:
    cleaned = re.sub(r"^(?:the|my)\s+", "", label.strip().lower())
    cleaned = _PLACE_ALIASES.get(cleaned, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" '")


def _extract_place(text: str) -> tuple[Optional[str], str]:
    """Find a "when I get <place>" clause. Returns ``(label, text_without_it)``."""
    for pattern in _PLACE_CLAUSE_RES:
        m = pattern.search(text)
        if not m:
            continue
        raw_label = m.groupdict().get("p") or m.groupdict().get("p2") or ""
        label = _normalise_place(raw_label)
        if not label or _NOT_A_PLACE_RE.search(label):
            continue
        return label, _squash(text[: m.start()] + " " + text[m.end():])
    return None, text


def _place_label_from_text(entity: str, raw: str) -> Optional[str]:
    """Name for a place being saved: an explicit entity, else phrasing like
    'this is home', 'save this location as the gym', 'I'm at the office'."""
    if entity.strip():
        label = _normalise_place(entity)
        return label or None
    patterns = (
        r"\b(?:save|remember|store|set|mark)\b.*?\b(?:as|to be)\s+(?:the\s+|my\s+)?(?P<p>[a-z][a-z' ]{1,24}?)\s*[.!?]?$",
        r"\bthis\s+is\s+(?:the\s+|my\s+)?(?P<p>[a-z][a-z' ]{1,24}?)\s*[.!?]?$",
        r"\bi(?:'m|\s+am)\s+(?:at|in)\s+(?:the\s+|my\s+)?(?P<p>[a-z][a-z' ]{1,24}?)\s*[.!?]?$",
        r"\bi(?:'m|\s+am)\s+(?P<p>home|at\s+home)\s*[.!?]?$",
        r"\bset\s+(?:my\s+)?(?P<p>[a-z][a-z' ]{1,24}?)\s+(?:location|address|place)\b",
        r"\bmy\s+(?P<p>[a-z][a-z' ]{1,24}?)\s+(?:location|address)\b",
    )
    for pat in patterns:
        m = re.search(pat, raw.strip(), re.IGNORECASE)
        if m:
            label = _normalise_place(re.sub(r"^at\s+", "", m.group("p")))
            if label and not _NOT_A_PLACE_RE.search(label) and label not in {"location", "place", "spot", "here"}:
                return label
    return None


def _command_hint(
    raw: str, task_entity: Any = None, extra_noise: frozenset[str] = frozenset(),
) -> str:
    """Isolate which reminder a management command refers to: the NLU's task
    entity if usable, else *raw* with command/filler words trimmed off both ends."""
    if task_entity and str(task_entity).strip().lower() not in {"", "none", "null"}:
        return str(task_entity).strip()
    noise = _MGMT_NOISE | extra_noise
    tokens = re.findall(r"[\w'’\-]+", raw or "")
    while tokens and tokens[0].lower() in noise:
        tokens.pop(0)
    while tokens and tokens[-1].lower() in noise:
        tokens.pop()
    return " ".join(tokens)


# -- dates ---------------------------------------------------------------

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _day_phrase(d: date, today: date) -> str:
    return agenda_mod.day_label(d, today)


def _resolve_day(raw: str, date_hint: str, now: datetime) -> date:
    """Which calendar day a request is about; today when nothing is said."""
    today = now.date()
    text = f"{raw} {date_hint}".lower()
    if re.search(r"\bday after tomorrow\b", text):
        return today + timedelta(days=2)
    if re.search(r"\btomorrow\b", text):
        return today + timedelta(days=1)
    if re.search(r"\byesterday\b", text):
        return today - timedelta(days=1)
    if re.search(r"\b(?:today|tonight|this\s+(?:morning|afternoon|evening))\b", text):
        return today
    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": now.replace(tzinfo=None),
        "RETURN_AS_TIMEZONE_AWARE": False,
    }
    if date_hint.strip():
        try:
            parsed = dateparser.parse(date_hint, languages=["en"], settings=settings)
        except Exception:
            parsed = None
        if parsed:
            return parsed.date()
    for fragment in re.findall(r"\b(?:next\s+)?(?:%s)\b" % "|".join(_WEEKDAYS), text):
        try:
            parsed = dateparser.parse(fragment, languages=["en"], settings=settings)
        except Exception:
            parsed = None
        if parsed:
            return parsed.date()
    if raw:
        try:
            found = search_dates(raw, languages=["en"], settings=settings)
        except Exception:
            found = None
        for fragment, when in found or []:
            if _TEMPORAL_KEYWORD_RE.search(fragment) and not re.fullmatch(r"\W*\d{1,2}\s*(?:am|pm)?\W*", fragment):
                return when.date()
    return today


_RANGE_WORD_RE = re.compile(r"\b(?:to|through|thru|until|till)\b|\s-\s|\d-\d", re.IGNORECASE)


def _extract_day_range(raw: str, date_hint: str, now: datetime) -> Optional[list[date]]:
    """Every calendar day a holiday request covers, or ``None`` if no date
    could be found. Handles a single day, "from X to Y", "next week" and
    "this weekend"."""
    today = now.date()
    text = f"{raw} {date_hint}".strip()
    low = text.lower()

    m = re.search(r"\b(this|next)\s+week\b", low)
    if m:
        monday = today - timedelta(days=today.weekday())
        if m.group(1) == "next":
            start = monday + timedelta(days=7)
        else:
            start = today
        end = monday + timedelta(days=(13 if m.group(1) == "next" else 6))
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if re.search(r"\b(?:this\s+|the\s+)?weekend\b", low):
        sat = today + timedelta(days=(5 - today.weekday()) % 7)
        if today.weekday() == 6:
            sat = today - timedelta(days=1)
        return [sat, sat + timedelta(days=1)]

    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": now.replace(tzinfo=None),
        "RETURN_AS_TIMEZONE_AWARE": False,
    }
    dates: list[date] = []
    try:
        found = search_dates(text, languages=["en"], settings=settings) if text else None
    except Exception:
        found = None
    for fragment, _ in found or []:
        if not _TEMPORAL_KEYWORD_RE.search(fragment):
            continue
        if re.fullmatch(r"\W*(?:am|pm|now|year|month|week)\W*", fragment.strip(), re.IGNORECASE):
            continue
        try:
            parsed = dateparser.parse(fragment, languages=["en"], settings=settings)
        except Exception:
            parsed = None
        if parsed and parsed.date() not in dates:
            dates.append(parsed.date())
    if not dates:
        d = _resolve_day(raw, date_hint, now) if re.search(r"\b(?:today|tomorrow|tonight)\b", low) else None
        return [d] if d else None
    if len(dates) >= 2 and _RANGE_WORD_RE.search(text):
        lo, hi = min(dates), max(dates)
        return [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
    return sorted(set(dates))[:1] if len(dates) == 1 else sorted(set(dates))


def _holiday_label(raw: str, entity: Any = None) -> Optional[str]:
    if entity and str(entity).strip().lower() not in {"", "none", "null"}:
        return str(entity).strip()[:60]
    m = re.search(
        r"\b(?:for|because\s+of|due\s+to|it'?s)\s+(?P<l>[A-Za-z][A-Za-z' ]{2,30}?)\s*[.!?]?$", raw or "",
    )
    if not m:
        return None
    label = m.group("l").strip()
    if _TEMPORAL_KEYWORD_RE.search(label) or label.lower() in {"a holiday", "holiday", "me", "my"}:
        return None
    return label


def _activity_from_text(raw: str) -> str:
    m = agenda_mod._OUTDOOR_RE.search(raw or "")
    return m.group(0).lower() if m else ""


# -- weather ---------------------------------------------------------------

def _fetch_weather(lat: float, lon: float) -> dict[str, Any]:
    """
    Call the Open-Meteo API and return the ``current_weather`` dict.

    Raises
    ------
    WeatherFetchError
        On any network, HTTP, or JSON-parse error.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true&wind_speed_unit=kmh"
    )
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data: dict = resp.json()
    except requests.RequestException as exc:
        raise WeatherFetchError(f"HTTP request failed: {exc}") from exc
    except ValueError as exc:
        raise WeatherFetchError(f"Response is not valid JSON: {exc}") from exc

    weather = data.get("current_weather")
    if not isinstance(weather, dict):
        raise WeatherFetchError("'current_weather' key missing from API response.")

    return weather


def _fetch_forecast(lat: float, lon: float, tz_name: str = "UTC", days: int = 3) -> dict[str, Any]:
    """
    Hourly rain probability + weather code from Open-Meteo, with timestamps
    expressed in *tz_name* (the shape ``agenda.assess_agenda`` reads).

    Raises
    ------
    WeatherFetchError
        On any network, HTTP, or JSON-parse error, or an unusable response.
    """
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "precipitation_probability,weathercode",
                "timezone": tz_name, "forecast_days": max(1, min(days, 7)),
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data: dict = resp.json()
    except requests.RequestException as exc:
        raise WeatherFetchError(f"HTTP request failed: {exc}") from exc
    except ValueError as exc:
        raise WeatherFetchError(f"Response is not valid JSON: {exc}") from exc
    hourly = data.get("hourly")
    if not isinstance(hourly, dict) or not hourly.get("time"):
        raise WeatherFetchError("'hourly' forecast missing from API response.")
    return hourly


# -- one-shot reminder parsing ---------------------------------------------

def _search_raw_datetime(
    raw: str, naive_base: datetime, settings: dict[str, Any]
) -> Optional[datetime]:
    """
    Find a date/time fragment anywhere inside *raw* and return the
    datetime it parses to, or ``None`` if nothing trustworthy is found.

    ``search_dates()`` is used only to *locate* candidate fragments; each
    candidate is re-parsed on its own with ``dateparser.parse()`` since
    the datetimes ``search_dates()`` attaches to embedded matches are
    unreliable (e.g. it can silently drop an explicit time like "9am"
    and substitute the current time). Candidates are also required to
    contain a digit or an unambiguous temporal keyword
    (``_TEMPORAL_KEYWORD_RE``) — without that filter, ordinary words in
    the reminder's task text (a name like "May" or "August") are
    regularly mistaken for dates.

    Among valid candidates, the first one that re-parses to a moment
    strictly after *naive_base* is returned, so a stray false-positive
    fragment earlier in the sentence doesn't shadow a real, later one.
    """
    try:
        found = search_dates(raw, languages=["en"], settings=settings)
    except Exception:
        logger.exception("search_dates() failed for raw=%r.", raw)
        return None
    if not found:
        return None

    for fragment, _ in found:
        if not _TEMPORAL_KEYWORD_RE.search(fragment):
            continue
        candidate = dateparser.parse(fragment, languages=["en"], settings=settings)
        if candidate is not None and candidate > naive_base:
            return candidate
    return None


def _extract_task(raw: str, task_hint: Optional[str]) -> str:
    """
    Derive a human-readable task label from the raw query or NLU entity.

    Strategy
    --------
    1. Use *task_hint* if it is a valid, non-trivial string.
    2. Extract the phrase after the word "to" in *raw* (``"remind me to X"``).
    3. Strip noise words and time suffixes from *raw*.
    4. Fall back to ``_TASK_FALLBACK``.
    """
    # 1 – NLU-provided hint
    if task_hint and str(task_hint).strip().lower() not in {"", "none", "null"}:
        cleaned = str(task_hint).strip()
        if len(cleaned) >= _MIN_TASK_LEN:
            return cleaned

    # 2 – "to <task>" pattern
    match = re.search(r"\bto\s+(.+)", raw, flags=re.IGNORECASE)
    if match:
        candidate = _TIME_SUFFIX.sub("", match.group(1)).strip()
        if len(candidate) >= _MIN_TASK_LEN:
            return candidate

    # 3 – Strip noise and time suffix from full raw query
    stripped = _TASK_NOISE.sub("", raw)
    stripped = _TIME_SUFFIX.sub("", stripped).strip()
    if len(stripped) >= _MIN_TASK_LEN:
        return stripped

    return _TASK_FALLBACK


def _parse_reminder_time(
    raw: str,
    date_hint: str,
    time_hint: str,
    base: datetime,
) -> datetime:
    """
    Parse a due-datetime for a reminder from available string inputs.

    Resolution order
    ----------------
    1. Locate a date/time-bearing fragment anywhere inside the full *raw*
       query and parse that fragment on its own (handles sentences like
       "remind me to call John at 3pm tomorrow", where the surrounding
       task text stops ``dateparser.parse()`` from matching the string
       as a whole).
    2. ``dateparser`` on the combination of *date_hint* and *time_hint*.
    3. Named time-of-day matching against *time_hint* (morning, evening …).

    Returns a timezone-aware datetime in the same timezone as *base*.

    Raises
    ------
    ReminderParseError
        When no strategy can produce a valid future datetime.
    """
    naive_base = base.replace(tzinfo=None)  # dateparser wants naïve
    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": naive_base,
        "RETURN_AS_TIMEZONE_AWARE": False,
    }

    # Strategy 1: find the date/time fragment inside the full sentence.
    # ``dateparser.parse()`` on the whole sentence almost never matches
    # once there's surrounding task text (names, verbs, etc.), so we
    # first use ``search_dates()`` to locate the temporal fragment(s),
    # then re-parse each fragment on its own. ``search_dates()``'s own
    # returned datetimes are unreliable when the match is embedded in a
    # longer string (it can drop an explicit time and substitute the
    # current time instead), so the fragment text is always re-parsed
    # rather than trusted directly.
    parsed = _search_raw_datetime(raw, naive_base, settings)

    # Strategy 2: explicit date + time entities
    if parsed is None and (date_hint or time_hint):
        combined = f"{date_hint} {time_hint}".strip()
        parsed = dateparser.parse(combined, settings=settings)

    # Strategy 3: named time-of-day fallback
    if parsed is None and time_hint:
        for label, (hour, minute, delta_days) in _TIME_OF_DAY.items():
            if label in time_hint.lower():
                candidate = base.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                ) + timedelta(days=delta_days)
                if candidate > base:
                    parsed = candidate.replace(tzinfo=None)
                    break

    if parsed is None:
        raise ReminderParseError(
            f"Could not parse a reminder time from raw={raw!r} "
            f"date_hint={date_hint!r} time_hint={time_hint!r}."
        )

    # Re-attach the caller's timezone
    tz = base.tzinfo or timezone.utc
    aware = parsed.replace(tzinfo=tz)

    if aware <= base:
        raise ReminderParseError(
            f"Parsed time {aware.isoformat()} is in the past (base={base.isoformat()})."
        )

    return aware


def _ok(
    response: str,
    data: Optional[dict[str, Any]] = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict[str, Any]:
    return {"response": response, "data": {}, "confidence": 0.0}


def _clarify(question: str) -> dict[str, Any]:
    return {
        "response": question,
        "data": {"needs_clarification": True},
        "confidence": 0.5,
    }