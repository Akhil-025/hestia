"""
modules/chronos/engine.py

ChronosEngine: time, date, weather, and reminder module.

Design notes
------------
- All datetime operations use timezone-aware objects (UTC internally,
  local-tz for display) to avoid DST ambiguity.
- Weather fetching is isolated in a pure helper; the engine delegates and
  handles failures without coupling to HTTP internals.
- Reminder parsing is decomposed into focused private methods: task
  extraction, datetime parsing, and validation are each independently
  testable.
- City coordinates live in a typed constant; callers can extend it by
  subclassing or by injecting a config dict.
- Every public method conforms to the BaseModule response contract:
  {response: str, data: dict, confidence: float}.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser
from dateparser.search import search_dates
import requests

from modules.base import BaseModule
from core.free_apis import FreeAPIError, is_public_holiday, public_holidays

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
    """

    name = "chronos"

    _INTENTS: frozenset[str] = frozenset(
        {"get_time", "get_date", "get_weather", "set_reminder", "get_holiday"}
    )

    def __init__(
        self,
        memory: Any = None,
        city_coords: Optional[CityCoords] = None,
        local_tz: Optional[str] = None,
    ) -> None:
        self._memory = memory
        self._coords: CityCoords = {**_CITY_COORDS, **(city_coords or {})}
        self._tz = _resolve_tz(local_tz)
        logger.info(
            "ChronosEngine ready (tz=%s, cities=%d).",
            self._tz,
            len(self._coords),
        )

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
            return self._dispatch(intent, entities, context)
        except Exception:
            logger.exception(
                "ChronosEngine.handle() raised for intent=%s.", intent
            )
            return _err("Something went wrong in the time module.")

    def get_context(self) -> dict:
        """Return a lightweight time context for NLU enrichment."""
        try:
            now = _now(self._tz)
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
        return _err(f"Unknown intent: {intent!r}")

    # ------------------------------------------------------------------
    # Private – time / date
    # ------------------------------------------------------------------

    def _get_time(self) -> dict:
        now = _now(self._tz)
        return _ok(
            f"It's {now.strftime('%I:%M %p')}.",
            data={"time": now.isoformat()},
            confidence=1.0,
        )

    def _get_date(self) -> dict:
        now = _now(self._tz)
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

        condition = _WMO_CODES.get(weather.get("weathercode", -1), "")
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
    # Private – reminders
    # ------------------------------------------------------------------

    def _set_reminder(self, entities: dict) -> dict:
        raw: str = entities.get("raw_query") or ""
        task = _extract_task(
            raw=raw,
            task_hint=entities.get("task"),
        )
        date_hint: str = entities.get("date") or ""
        time_hint: str = entities.get("time") or ""

        try:
            due_dt = _parse_reminder_time(
                raw=raw,
                date_hint=date_hint,
                time_hint=time_hint,
                base=_now(self._tz),
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
                logger.exception(
                    "_set_reminder: failed to persist reminder (task=%r).", task
                )
                return _err("I understood the reminder but couldn't save it.")

        readable = due_dt.strftime("%A, %B %d at %I:%M %p")
        logger.info("Reminder set: task=%r due=%s", task, due_iso)
        return _ok(
            f"Reminder set: {task!r} on {readable}.",
            data={"task": task, "due": due_iso},
            confidence=0.95,
        )

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
        settings = {"RELATIVE_BASE": _now(self._tz).replace(tzinfo=None)}
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
            year = _now(self._tz).year
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

def _now(tz: Any) -> datetime:
    """Return the current moment as a timezone-aware datetime."""
    return datetime.now(tz)


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