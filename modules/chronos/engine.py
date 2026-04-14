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
import requests

from modules.base import BaseModule

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
    r"\b(at|in|on|tomorrow|today|tonight|morning|afternoon|evening|night"
    r"|midnight|noon|\d{1,2}[:\s]\d{2}|\d{1,2}\s?(?:am|pm))\b.*",
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
        {"get_time", "get_date", "get_weather", "set_reminder"}
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
            return self._get_time()
        if intent == "get_date":
            return self._get_date()
        if intent == "get_weather":
            return self._get_weather(entities, context)
        if intent == "set_reminder":
            return self._set_reminder(entities)
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
        lat, lon = self._coords.get(location.lower(), (_DEFAULT_LAT, _DEFAULT_LON))

        try:
            weather = _fetch_weather(lat, lon)
        except WeatherFetchError:
            logger.exception("Weather fetch failed for location=%r.", location)
            return _err("I couldn't fetch the weather right now.")

        condition = _WMO_CODES.get(weather.get("weathercode", -1), "")
        condition_str = f", {condition}" if condition else ""
        temp = weather.get("temperature", "?")
        wind = weather.get("windspeed", "?")

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
    1. ``dateparser`` on the full *raw* query (most context).
    2. ``dateparser`` on the combination of *date_hint* and *time_hint*.
    3. Named time-of-day matching against *time_hint* (morning, evening …).

    Returns a timezone-aware datetime in the same timezone as *base*.

    Raises
    ------
    ReminderParseError
        When no strategy can produce a valid future datetime.
    """
    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": base.replace(tzinfo=None),  # dateparser wants naïve
        "RETURN_AS_TIMEZONE_AWARE": False,
    }

    # Strategy 1: full raw string
    parsed = dateparser.parse(raw, settings=settings)

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