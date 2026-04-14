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
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional

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
        }
    )

    def __init__(self, google_agent: Any = None) -> None:
        self._google = google_agent
        logger.info(
            "HermesEngine ready (google_agent=%s).",
            type(google_agent).__name__ if google_agent else "None",
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

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
        if intent == "read_email":
            return self._read_email(entities)
        if intent == "send_email":
            return self._send_email(entities)
        if intent == "list_events":
            return self._list_events(entities)
        if intent == "create_event":
            return self._create_event(entities)
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
        """Validate recipients / body and send a plain-text email."""
        to: str = (entities.get("to") or "").strip()
        subject: str = (entities.get("subject") or _DEFAULT_SUBJECT).strip()
        body: str = (
            entities.get("body") or entities.get("message") or ""
        ).strip()

        if not to:
            return _clarify("Who should I send it to?")
        if not body:
            return _clarify("What should the email say?")

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

    def _create_event(self, entities: dict) -> dict:
        """Parse entities, build a datetime, and create a calendar event."""
        title: str = (
            entities.get("task")
            or entities.get("title")
            or entities.get("event")
            or ""
        ).strip()

        if not title:
            return _clarify("What should I call the event?")

        date_str: str = (entities.get("date") or "today").strip()
        time_str: str = (entities.get("time") or _DEFAULT_EVENT_TIME).strip()

        try:
            start_dt = _parse_datetime(date_str, time_str)
        except DateTimeParseError as exc:
            logger.warning(
                "_create_event: datetime parse failed (%s); defaulting to +1 h.", exc
            )
            start_dt = datetime.now(timezone.utc) + timedelta(hours=1)

        try:
            success = self._google.create_event(title=title, start_dt=start_dt)
        except Exception:
            logger.exception("create_event() raised for title=%r.", title)
            return _err("I couldn't create that event due to an unexpected error.")

        if success:
            readable = start_dt.strftime("%A %-d %B at %H:%M")
            logger.info("create_event: %r created at %s.", title, start_dt.isoformat())
            return _ok(
                f"Done. {title!r} added to your calendar for {readable}.",
                confidence=0.9,
            )

        logger.warning("create_event: create_event() returned False for title=%r.", title)
        return _err("I couldn't create that event.")


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _parse_datetime(date_str: str, time_str: str) -> datetime:
    """
    Combine a date string and a time string into a timezone-aware datetime.

    Supported date formats
    ----------------------
    - ``"today"`` / ``""``     → today's date
    - ``"tomorrow"``           → tomorrow's date
    - ``"YYYY-MM-DD"``         → ISO date literal

    Supported time format
    ---------------------
    - ``"HH:MM"`` (24-hour)

    Raises
    ------
    DateTimeParseError
        If either string cannot be interpreted.
    """
    today = datetime.now(timezone.utc).date()
    lower = date_str.lower().strip()

    if lower in ("today", ""):
        base_date = today
    elif lower == "tomorrow":
        base_date = today + timedelta(days=1)
    else:
        try:
            base_date = date.fromisoformat(date_str)
        except ValueError as exc:
            raise DateTimeParseError(
                f"Unrecognised date format: {date_str!r}. Use YYYY-MM-DD."
            ) from exc

    try:
        parts = time_str.replace(":", " ").split()
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("Hour or minute out of range.")
        event_time = time(hour, minute)
    except (IndexError, ValueError) as exc:
        raise DateTimeParseError(
            f"Unrecognised time format: {time_str!r}. Use HH:MM."
        ) from exc

    return datetime.combine(base_date, event_time, tzinfo=timezone.utc)


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