"""
modules/hermes/engine.py

HermesEngine: Gmail and Google Calendar integration module.

Design notes
------------
- The Google agent is injected at construction time and validated before
  every intent; a clear 503-style response is returned when it is absent
  or unauthenticated rather than raising AttributeError deep in a handler.
- Date/time parsing is isolated in a pure helper so it can be unit-tested
  without an engine instance and extended (e.g. natural-language dates)
  without touching business logic.
- Every handler is a private method; `handle` owns only dispatch and the
  top-level auth guard.
- All response dicts conform to the BaseModule contract:
  {response: str, data: dict, confidence: float}.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, time, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from modules.base import BaseModule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EMAIL_COUNT = 5
_DEFAULT_DAYS_AHEAD = 7
_DEFAULT_SUBJECT = "Message from Hestia"
_DEFAULT_EVENT_TIME = "09:00"
_MAX_EMAIL_COUNT = 50
_MAX_DAYS_AHEAD = 90
_MAX_EVENT_DELETE = 50

_NOT_CONNECTED = "Communication services are not connected. Ask me to reconnect Google."
_UNHANDLED = "I can't handle that communication request."


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HermesError(Exception):
    """Base exception for HermesEngine failures."""


class DateTimeParseError(HermesError):
    """Raised when a date/time string cannot be interpreted."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HermesEngine(BaseModule):
    """
    Gmail and Google Calendar integration module.

    Parameters
    ----------
    google_agent:
        A ``HestiaGoogleAgent`` instance (or compatible duck-typed object).
        Injected by the orchestrator at startup.
    """

    name = "hermes"

    _INTENTS: frozenset[str] = frozenset(
        {
            "read_email",
            "send_email",
            "list_events",
            "create_event",
            "delete_events",
        }
    )

    # NLU models don't always emit these exact canonical names — "check my
    # email" or "what's on my calendar" can plausibly come back as
    # "get_email" or "get_calendar_event" instead of "read_email" /
    # "list_events". can_handle()/handle() accept both the canonical and
    # alias spellings so a reasonably-named intent still reaches Hermes
    # rather than being rejected and falling back to chat (see also
    # HestiaOrchestrator._find_alternate_module, which relies on
    # can_handle() covering every name it might plausibly be asked about).
    _INTENT_ALIASES: dict[str, str] = {
        "get_email": "read_email",
        "check_email": "read_email",
        "fetch_email": "read_email",
        "check_mail": "read_email",
        "get_mail": "read_email",
        "check_inbox": "read_email",
        "gmail": "read_email",
        "get_calendar_event": "list_events",
        "get_calendar_events": "list_events",
        "get_events": "list_events",
        "check_calendar": "list_events",
        "get_calendar": "list_events",
        "schedule_event": "create_event",
        "add_event": "create_event",
        "add_calendar_event": "create_event",
        "clear_events": "delete_events",
        "clear_schedule": "delete_events",
        "clear_calendar": "delete_events",
        "cancel_events": "delete_events",
        "cancel_event": "delete_events",
        "remove_events": "delete_events",
    }

    def __init__(self, google_agent: Any = None, timezone_name: str = "UTC") -> None:
        self._google = google_agent
        # The user's local IANA timezone (e.g. "Asia/Kolkata"). Used to
        # resolve "today"/"tomorrow" against local wall-clock time and to
        # build naive local datetimes for create_event — see _parse_datetime
        # for why these must NOT carry a UTC tzinfo.
        try:
            self._tz = ZoneInfo(timezone_name)
        except Exception:
            logger.warning(
                "Unrecognised timezone %r; falling back to UTC.", timezone_name
            )
            self._tz = ZoneInfo("UTC")
        logger.info(
            "HermesEngine ready (google_agent=%s, timezone=%s).",
            type(google_agent).__name__ if google_agent else "None",
            timezone_name,
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS or intent in self._INTENT_ALIASES

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Dispatch an intent to the appropriate handler.

        Returns a "not connected" response when the Google agent is absent
        or unauthenticated.  Never raises.
        """
        if not self._is_ready():
            logger.warning(
                "handle(%r): Google agent not ready.", intent
            )
            return _err(_NOT_CONNECTED)

        try:
            return self._dispatch(intent, entities)
        except Exception:
            logger.exception(
                "HermesEngine.handle() raised for intent=%s.", intent
            )
            return _err("Something went wrong in the communication module.")

    def get_context(self) -> dict:
        return {
            "hermes_connected": self._is_ready(),
        }

    # ------------------------------------------------------------------
    # Private – readiness
    # ------------------------------------------------------------------

    def _is_ready(self) -> bool:
        """Return True when the Google agent exists and is authenticated."""
        return bool(self._google and self._google.is_authenticated())

    # ------------------------------------------------------------------
    # Private – dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, entities: dict) -> dict:
        intent = self._INTENT_ALIASES.get(intent, intent)
        if intent == "read_email":
            return self._read_email(entities)
        if intent == "send_email":
            return self._send_email(entities)
        if intent == "list_events":
            return self._list_events(entities)
        if intent == "create_event":
            return self._create_event(entities)
        if intent == "delete_events":
            return self._delete_events(entities)
        return _err(_UNHANDLED)

    # ------------------------------------------------------------------
    # Private – intent handlers
    # ------------------------------------------------------------------

    def _read_email(self, entities: dict) -> dict:
        """Fetch recent emails and return a TTS-ready summary."""
        count = _clamp_int(entities.get("count", _DEFAULT_EMAIL_COUNT), 1, _MAX_EMAIL_COUNT)

        try:
            emails = self._google.read_emails(max_results=count)
        except Exception:
            logger.exception("read_emails() failed.")
            return _err("I couldn't fetch your emails right now.")

        summary = self._google.format_emails_for_tts(emails)
        logger.info("read_email: fetched %d email(s).", len(emails))
        return _ok(summary, data={"emails": [_email_to_dict(e) for e in emails]})

    def _send_email(self, entities: dict) -> dict:
        """
        Validate recipients / body and send a plain-text email.

        Two-phase, gated by the orchestrator's confirmation mechanism (see
        HestiaOrchestrator._resolve_pending): the first call — entities has
        no "_confirmed" flag yet — validates the request and returns a
        preview + confirmation question WITHOUT calling Google's API. Only
        the second call, made by the orchestrator itself after the user's
        next reply reads as a clear "yes", actually sends anything. This
        matters specifically because Hestia is voice-driven: a single
        misheard recipient or body should never be enough to put a real
        message in someone's inbox.
        """
        to: str = (entities.get("to") or "").strip()
        subject: str = (entities.get("subject") or _DEFAULT_SUBJECT).strip()
        body: str = (
            entities.get("body") or entities.get("message") or ""
        ).strip()

        if not to:
            return _clarify("Who should I send it to?")
        if not body:
            return _clarify("What should the email say?")

        if not entities.get("_confirmed"):
            preview = _truncate(body, 120)
            return {
                "response": (
                    f'Send an email to {to}, subject "{subject}", saying '
                    f'"{preview}"? Say yes to send it.'
                ),
                "data": {"to": to, "subject": subject, "body": body},
                "confidence": 0.9,
                "needs_confirmation": True,
                "confirm_intent": "send_email",
                "confirm_entities": {"to": to, "subject": subject, "body": body},
                "confirm_label": f"send that email to {to}",
            }

        try:
            success = self._google.send_email(to, subject, body)
        except Exception:
            logger.exception("send_email() raised for to=%r.", to)
            return _err("I couldn't send that email due to an unexpected error.")

        if success:
            logger.info("send_email: message sent to %r.", to)
            return _ok(f"Email sent to {to}.", confidence=0.9)

        logger.warning("send_email: send_email() returned False for to=%r.", to)
        return _err("I couldn't send that email.")

    def _list_events(self, entities: dict) -> dict:
        """Fetch upcoming calendar events and return a TTS-ready summary."""
        days = _clamp_int(entities.get("days", _DEFAULT_DAYS_AHEAD), 1, _MAX_DAYS_AHEAD)

        try:
            events = self._google.list_events(days_ahead=days)
        except Exception:
            logger.exception("list_events() failed.")
            return _err("I couldn't fetch your calendar right now.")

        summary = self._google.format_events_for_tts(events)
        logger.info("list_events: fetched %d event(s).", len(events))
        return _ok(summary, data={"events": [_event_to_dict(e) for e in events]})

    # Fallback for when the NLU returns an empty entities dict but the raw
    # text clearly names the event (e.g. "create an event tomorrow at 5pm
    # called Gym"). Mirrors the take_note raw-text fallback in CoreModule.
    _EVENT_TITLE_RE = re.compile(
        r"\b(?:called|titled|named)\s+(.+?)\s*$", re.IGNORECASE
    )

    def _create_event(self, entities: dict) -> dict:
        """Parse entities, build a datetime, and create a calendar event."""
        title: str = (
            entities.get("task")
            or entities.get("title")
            or entities.get("event")
            or ""
        ).strip()

        if not title:
            raw = (entities.get("raw_query") or "").strip()
            match = self._EVENT_TITLE_RE.search(raw)
            if match:
                title = match.group(1).strip(" .!?\"'")

        if not title:
            return _clarify("What should I call the event?")

        date_str: str = (entities.get("date") or "today").strip()
        time_str: str = (entities.get("time") or _DEFAULT_EVENT_TIME).strip()

        try:
            start_dt = _parse_datetime(date_str, time_str, self._tz)
        except DateTimeParseError as exc:
            logger.warning("_create_event: datetime parse failed: %s", exc)
            return _clarify(
                "I couldn't understand that date/time. Could you say it "
                "differently? (e.g. 'tomorrow at 3pm' or '2024-12-25 at 09:00')"
            )

        try:
            success = self._google.create_event(title=title, start_dt=start_dt)
        except Exception:
            logger.exception("create_event() raised for title=%r.", title)
            return _err("I couldn't create that event due to an unexpected error.")

        if success:
            # %-d is glibc/macOS-only and raises ValueError on Windows
            # (Python's Windows strftime doesn't support the "-" no-pad
            # flag). Build the "day month" part manually so this works on
            # every platform.
            readable = f"{start_dt:%A} {start_dt.day} {start_dt:%B} at {start_dt:%H:%M}"
            logger.info("create_event: %r created at %s.", title, start_dt.isoformat())
            return _ok(
                f"Done. {title!r} added to your calendar for {readable}.",
                confidence=0.9,
            )

        logger.warning("create_event: create_event() returned False for title=%r.", title)
        return _err("I couldn't create that event.")

    def _delete_events(self, entities: dict) -> dict:
        """
        Clear events for a date window and report how many were removed.

        Google Calendar's list endpoint (as wrapped by list_events) only
        supports "the next N days starting now", not an arbitrary specific
        date. A naive implementation of "today"/"tomorrow"/an explicit date
        would have to pick a days-ahead window and trust that everything in
        it belongs to the target day — which is only true for "today". Any
        other target ("tomorrow", "2026-12-25", ...) needs a wider window to
        reach that far, and without filtering, that wider window's *entire*
        contents get deleted: "clear tomorrow's schedule" would wipe out a
        full week, not just tomorrow.

        To avoid that, a recognised single-day target (today / tomorrow /
        an explicit date literal) is resolved to a concrete date, events are
        fetched over a window wide enough to include it, and the result is
        filtered down to just that day before anything is deleted. Only a
        bare `days` entity (e.g. "clear the next 3 days") skips the
        single-day filter and clears the whole window, since that's an
        explicit multi-day request rather than a single ambiguous date.
        """
        target_date: Optional[date] = None

        if "days" in entities:
            # Explicit multi-day window ("clear the next 3 days") — no
            # single target date to filter to.
            days = _clamp_int(entities["days"], 1, _MAX_DAYS_AHEAD)
        else:
            date_str = (entities.get("date") or "today").strip()
            try:
                target_date = _resolve_date(date_str, self._tz)
            except DateTimeParseError as exc:
                logger.warning("_delete_events: date parse failed: %s", exc)
                return _clarify(
                    "I couldn't understand that date. Could you say it "
                    "differently? (e.g. 'today', 'tomorrow', or '2024-12-25')"
                )
            today = datetime.now(self._tz).date()
            days = _clamp_int((target_date - today).days + 1, 1, _MAX_DAYS_AHEAD)

        try:
            events = self._google.list_events(max_results=_MAX_EVENT_DELETE, days_ahead=days)
        except Exception:
            logger.exception("_delete_events: list_events() failed.")
            return _err("I couldn't fetch your calendar right now.")

        if target_date is not None:
            events = [e for e in events if _event_falls_on(e, target_date, self._tz)]

        if not events:
            return _ok("You don't have any events to clear.", confidence=0.9)

        deleted = 0
        for event in events:
            try:
                if self._google.delete_event(event.event_id):
                    deleted += 1
            except Exception:
                logger.exception(
                    "_delete_events: delete_event() failed for event_id=%r.",
                    event.event_id,
                )

        logger.info("_delete_events: cleared %d/%d event(s).", deleted, len(events))
        if deleted == 0:
            return _err("I couldn't clear your calendar.")

        return _ok(
            f"Cleared {deleted} event(s) from your calendar.",
            data={"deleted": deleted, "found": len(events)},
            confidence=0.9,
        )


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _parse_datetime(date_str: str, time_str: str, tz: "ZoneInfo") -> datetime:
    """
    Combine a date string and a time string into a naive local datetime.

    Supported date formats
    ----------------------
    - ``"today"`` / ``""``     → today's date (in *tz*)
    - ``"tomorrow"``           → tomorrow's date (in *tz*)
    - ``"YYYY-MM-DD"``         → ISO date literal
    - ``"DD/MM/YYYY"`` or ``"DD-MM-YYYY"`` → day-first literal (the format
      users type unprompted; the NLU doesn't always normalise this to ISO)

    Supported time format
    ---------------------
    - ``"HH:MM"`` (24-hour) or ``"3pm"``/``"3:30pm"`` (12-hour)

    Returns
    -------
    datetime
        A **naive** datetime representing the wall-clock time the user
        meant, e.g. "3pm tomorrow" → 15:00 on tomorrow's date, with no
        tzinfo attached. This is intentional: HestiaGoogleAgent.create_event
        sends this via isoformat() alongside an explicit "timeZone" field,
        and the Google Calendar API interprets an offset-less dateTime as
        local time *in that timeZone*. If we attached tzinfo=UTC here (as
        previously), isoformat() would embed a "+00:00" offset that the API
        treats as authoritative, silently shifting every event by the
        difference between UTC and the user's actual timezone — e.g. "3pm"
        in Asia/Kolkata (UTC+5:30) would be created as 3pm UTC = 8:30pm IST.

    Raises
    ------
    DateTimeParseError
        If either string cannot be interpreted.
    """
    base_date = _resolve_date(date_str, tz)
    event_time = _parse_time(time_str)
    if event_time is None:
        raise DateTimeParseError(
            f"Unrecognised time format: {time_str!r}. Use HH:MM or e.g. '3pm'."
        )

    return datetime.combine(base_date, event_time)


def _resolve_date(date_str: str, tz: "ZoneInfo") -> date:
    """
    Resolve ``"today"`` / ``"tomorrow"`` / an explicit date literal to a
    concrete ``date`` in *tz*.

    Shared by ``_parse_datetime`` (event creation) and ``HermesEngine.
    _delete_events`` (so "clear tomorrow's schedule" can filter to exactly
    that day instead of trusting a blind days-ahead window — see
    ``_delete_events`` for why that distinction matters).

    Raises
    ------
    DateTimeParseError
        If *date_str* cannot be interpreted.
    """
    today = datetime.now(tz).date()
    lower = date_str.lower().strip()

    if lower in ("today", ""):
        return today
    if lower == "tomorrow":
        return today + timedelta(days=1)

    literal = _parse_date_literal(date_str)
    if literal is None:
        raise DateTimeParseError(
            f"Unrecognised date format: {date_str!r}. "
            "Use YYYY-MM-DD or DD/MM/YYYY."
        )
    return literal


# Matches "3pm", "3 pm", "3:30pm", "3:30 p.m.", "11am" etc.
_TIME_12H_RE = re.compile(
    r'^\s*(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>[ap]\.?m\.?)\s*$',
    re.IGNORECASE,
)
# Matches "HH:MM" / "H:MM" 24-hour, e.g. "09:00", "9:00", "15:00".
_TIME_24H_RE = re.compile(r'^\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*$')


# Matches "25/11/2026" or "25-11-2026" (day-first, as typed by users —
# not necessarily normalised to ISO by the NLU).
_DATE_DMY_RE = re.compile(r'^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*$')


def _parse_date_literal(date_str: str) -> Optional[date]:
    """Parse a YYYY-MM-DD or DD/MM/YYYY (also DD-MM-YYYY) date literal.
    Returns None — never raises — if neither format matches, so callers can
    produce one consistent DateTimeParseError."""
    try:
        return date.fromisoformat(date_str.strip())
    except ValueError:
        pass

    m = _DATE_DMY_RE.match(date_str)
    if m:
        day, month, year = (int(g) for g in m.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def _parse_time(time_str: str) -> Optional[time]:
    """
    Parse a time string in either 24-hour ("HH:MM") or 12-hour
    ("3pm", "3:30 pm") format. Returns None if the string can't be
    interpreted, rather than raising, so callers can decide how to react.
    """
    s = time_str.strip()

    m = _TIME_12H_RE.match(s)
    if m:
        hour = int(m.group("hour"))
        minute = int(m.group("minute") or 0)
        meridiem = m.group("meridiem").lower().replace(".", "")
        if not (1 <= hour <= 12 and 0 <= minute <= 59):
            return None
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        return time(hour, minute)

    m = _TIME_24H_RE.match(s)
    if m:
        hour, minute = int(m.group("hour")), int(m.group("minute"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return time(hour, minute)

    return None


def _truncate(text: str, max_len: int) -> str:
    """Shorten *text* for a spoken/displayed confirmation preview."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    """Coerce *value* to int and clamp it to [lo, hi]."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo


def _email_to_dict(email: Any) -> dict[str, str]:
    """Convert an Email dataclass or dict to a plain dict."""
    if hasattr(email, "to_dict"):
        return email.to_dict()
    return dict(email) if isinstance(email, dict) else {}


def _event_to_dict(event: Any) -> dict[str, str]:
    """Convert a CalendarEvent dataclass or dict to a plain dict."""
    if hasattr(event, "to_dict"):
        return event.to_dict()
    return dict(event) if isinstance(event, dict) else {}


def _event_falls_on(event: Any, target: date, tz: "ZoneInfo") -> bool:
    """
    Return True if *event*'s start falls on *target* (in *tz*).

    Used by ``HermesEngine._delete_events`` to narrow a days-ahead fetch
    down to a single requested day. ``start`` is either an all-day date
    literal ("2026-08-07") or a full RFC 3339 datetime, per
    ``CalendarEvent.from_api`` / the Google Calendar API. Malformed or
    missing start values are treated as a non-match (excluded, not
    deleted) rather than raising, so one bad event can't abort the whole
    clear operation.
    """
    start = getattr(event, "start", None)
    if start is None and isinstance(event, dict):
        start = event.get("start")
    if not start:
        return False

    try:
        if len(start) <= 10:
            return date.fromisoformat(start) == target
        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(tz)
        return dt.date() == target
    except ValueError:
        logger.warning("_event_falls_on: could not parse event start %r.", start)
        return False


def _ok(
    response: str,
    data: Optional[dict[str, Any]] = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict[str, Any]:
    return {"response": response, "data": {}, "confidence": 0.0}


def _clarify(question: str) -> dict[str, Any]:
    return {"response": question, "data": {"needs_clarification": True}, "confidence": 0.5}