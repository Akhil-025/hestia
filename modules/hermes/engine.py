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
- Anything that sends mail or writes to the calendar in a way the user might
  not expect (a send, an overlapping event) is held behind the orchestrator's
  confirmation mechanism and only executes on the second, ``_confirmed`` call.
- Triage, inbox-zero and schedule-gap analysis are pure functions over the
  agent's results (no LLM needed); the LLM, if one is injected, is used only to
  word email drafts.
- Things that change the outside world (booking + inviting, archiving mail)
  are opt-in and always go through the confirm step first. Archiving also
  needs the Gmail modify scope, which is only requested when
  ``hermes.allow_mailbox_changes`` is on.
- Todoist (#91) is a separate connection from Google: its intents work with a
  Todoist token and no Google login, and vice versa.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from core.todoist_agent import (
    TodoistError,
    TodoistTask,
    find_matches,
    rank_tasks,
    select_due,
)
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
_DEFAULT_SEARCH_COUNT = 10
_DEFAULT_DIGEST_COUNT = 20
_DEFAULT_INBOX_ZERO_COUNT = 10
_MAX_INBOX_ZERO_COUNT = 25
_DEFAULT_BUFFER_MINUTES = 10
_DEFAULT_TRAVEL_MINUTES = 30
_DEFAULT_MEETING_MINUTES = 30
_DEFAULT_SLOT_DAYS = 3
_MAX_SLOT_DAYS = 14
_SLOT_STEP_MINUTES = 30
_MAX_SLOTS_OFFERED = 3
_DEFAULT_WORK_START = 9
_DEFAULT_WORK_END = 18
_SLOT_MEMORY_MINUTES = 30        # how long "book the first one" remembers proposed times
_DIGEST_WINDOW_HOURS = 6         # a missed morning digest isn't sent late in the evening
_MAX_ARCHIVE_BATCH = 25
_TODOIST_SPOKEN = 5

_TODOIST_INTENTS: frozenset[str] = frozenset(
    {"todoist_list_tasks", "todoist_add_task", "todoist_complete_task", "todoist_prioritize"}
)
_TODOIST_NOT_CONNECTED = (
    "Todoist isn't connected. Add your API token under todoist.api_token "
    "(or the TODOIST_API_TOKEN environment variable) and restart me."
)

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
            # backlog #92-#99
            "email_digest",
            "draft_email",
            "search_email",
            "check_schedule_gaps",
            "find_meeting_slot",
            "inbox_zero",
            # backlog #96 (booking + invites) and #91 (Todoist)
            "book_meeting_slot",
            "todoist_list_tasks",
            "todoist_add_task",
            "todoist_complete_task",
            "todoist_prioritize",
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
        "email_triage": "email_digest",
        "triage_email": "email_digest",
        "triage_inbox": "email_digest",
        "inbox_digest": "email_digest",
        "daily_digest": "email_digest",
        "compose_email": "draft_email",
        "write_email": "draft_email",
        "reply_email": "draft_email",
        "email_reply": "draft_email",
        "find_email": "search_email",
        "search_emails": "search_email",
        "search_mail": "search_email",
        "check_buffers": "check_schedule_gaps",
        "schedule_gaps": "check_schedule_gaps",
        "back_to_back": "check_schedule_gaps",
        "suggest_meeting_time": "find_meeting_slot",
        "find_meeting_time": "find_meeting_slot",
        "find_free_slot": "find_meeting_slot",
        "schedule_meeting": "find_meeting_slot",
        "book_slot": "book_meeting_slot",
        "schedule_meeting": "book_meeting_slot",
        "todoist_tasks": "todoist_list_tasks",
        "todoist_add": "todoist_add_task",
        "todoist_done": "todoist_complete_task",
        "todoist_complete": "todoist_complete_task",
        "todoist_priorities": "todoist_prioritize",
        "todoist_sort": "todoist_prioritize",
        "inbox_zero_mode": "inbox_zero",
        "clean_inbox": "inbox_zero",
        "process_inbox": "inbox_zero",
    }

    def __init__(
        self,
        google_agent: Any = None,
        timezone_name: str = "UTC",
        *,
        llm: Any = None,
        contacts: Optional[dict[str, str]] = None,
        vip_senders: Optional[list[str]] = None,
        work_hours: Optional[tuple[int, int]] = None,
        buffer_minutes: int = _DEFAULT_BUFFER_MINUTES,
        travel_minutes: int = _DEFAULT_TRAVEL_MINUTES,
        travel: Any = None,
        todoist: Any = None,
        allow_mailbox_changes: bool = False,
        digest_time: Optional[str] = None,
        state_path: Optional[str] = None,
    ) -> None:
        """
        Optional keyword settings (all default to the previous behaviour):

        llm            object with ``generate(prompt, fmt=None)``; only used to
                       word ``draft_email`` — a template draft is used when
                       absent or when it fails.
        contacts       ``{"john": "john@example.com"}``; lets "email John" work
                       without the address (names are matched case-insensitively).
        vip_senders    substrings of sender addresses/names always ranked high
                       in triage (#92).
        work_hours     ``(9, 18)``; the window ``find_meeting_slot`` offers (#96).
        buffer_minutes minimum gap between back-to-back events before
                       ``check_schedule_gaps`` flags them (#95).
        travel_minutes flat estimate added when consecutive events have
                       different locations (#95), used whenever ``travel``
                       can't give a better answer.
        travel         object with ``estimate(origin, destination)`` returning
                       something with ``.minutes`` / ``.source`` or ``None``
                       (``core.travel_time.TravelTimeEstimator``). Optional.
        todoist        ``core.todoist_agent.TodoistAgent`` (#91). Optional.
        allow_mailbox_changes
                       lets ``inbox_zero`` archive mail after a "yes" (#99).
                       The Google agent must also hold the modify scope.
        digest_time    ``"08:00"``: speak the email digest once a day from this
                       time via the heartbeat (#92). ``None`` = off.
        state_path     JSON file remembering the last digest date across
                       restarts. ``None`` keeps it in memory only.
        """
        self._google = google_agent
        self._travel = travel
        self._todoist = todoist
        self._allow_mailbox_changes = bool(allow_mailbox_changes)
        self._digest_time = _parse_clock_setting(digest_time)
        self._state_path = Path(state_path) if state_path else None
        self._digest_state: dict[str, Any] = self._load_state()
        self._last_slots: Optional[dict[str, Any]] = None
        self._todoist_error = ""
        self._llm = llm
        self._contacts = {
            str(k).strip().lower(): str(v).strip()
            for k, v in (contacts or {}).items()
            if k and v
        }
        self._vips = [str(v).strip().lower() for v in (vip_senders or []) if str(v).strip()]
        wh = work_hours or (_DEFAULT_WORK_START, _DEFAULT_WORK_END)
        try:
            ws, we = int(wh[0]), int(wh[1])
        except (TypeError, ValueError, IndexError):
            ws, we = _DEFAULT_WORK_START, _DEFAULT_WORK_END
        if not (0 <= ws < we <= 24):
            ws, we = _DEFAULT_WORK_START, _DEFAULT_WORK_END
        self._work_hours = (ws, we)
        self._buffer_minutes = _clamp_int(buffer_minutes, 0, 240)
        self._travel_minutes = _clamp_int(travel_minutes, 0, 480)
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
        if self._INTENT_ALIASES.get(intent, intent) in _TODOIST_INTENTS:
            if not self._todoist_ready():
                logger.warning("handle(%r): Todoist not configured.", intent)
                return _err(_TODOIST_NOT_CONNECTED)
        elif not self._is_ready():
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
            "todoist_connected": self._todoist_ready(),
        }

    # ------------------------------------------------------------------
    # Private – readiness
    # ------------------------------------------------------------------

    def _is_ready(self) -> bool:
        """Return True when the Google agent exists and is authenticated."""
        return bool(self._google and self._google.is_authenticated())

    def _todoist_ready(self) -> bool:
        return bool(self._todoist and self._todoist.is_ready())

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
        if intent == "email_digest":
            return self._email_digest(entities)
        if intent == "draft_email":
            return self._draft_email(entities)
        if intent == "search_email":
            return self._search_email(entities)
        if intent == "check_schedule_gaps":
            return self._check_schedule_gaps(entities)
        if intent == "find_meeting_slot":
            return self._find_meeting_slot(entities)
        if intent == "inbox_zero":
            return self._inbox_zero(entities)
        if intent == "book_meeting_slot":
            return self._book_meeting_slot(entities)
        if intent == "todoist_list_tasks":
            return self._todoist_list(entities)
        if intent == "todoist_add_task":
            return self._todoist_add(entities)
        if intent == "todoist_complete_task":
            return self._todoist_complete(entities)
        if intent == "todoist_prioritize":
            return self._todoist_prioritize(entities)
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
            return _clarify(
                "Who should I send it to?", slot="to", entities=entities
            )
        if not body:
            return _clarify(
                "What should the email say?", slot="body", entities=entities
            )

        # A misheard or half-said recipient ("John") must never reach
        # Google as an address. Resolve it against the configured contacts,
        # otherwise ask for the actual address (backlog #100).
        address = self._resolve_recipient(to)
        if address is None:
            return _clarify(
                f"What's the email address for {to}?", slot="to", entities=entities
            )
        to_label = to if address.lower() == to.lower() else f"{to} ({address})"
        to = address

        if not entities.get("_confirmed"):
            preview = _truncate(body, 120)
            return {
                "response": (
                    f'Send an email to {to_label}, subject "{subject}", saying '
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
            return _clarify(
                "What should I call the event?", slot="title", entities=entities
            )

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

        # Recurrence (#98). An unreadable repeat rule is a question, never a
        # silently-created one-off event.
        recurrence_text = _recurrence_text(entities)
        recurrence: Optional[list[str]] = None
        repeat_label = ""
        if recurrence_text:
            recurrence = _build_rrule(
                recurrence_text,
                count=entities.get("count"),
                until=entities.get("until"),
                start=start_dt,
                tz=self._tz,
            )
            if recurrence is None:
                return _clarify(
                    "I couldn't understand how that event repeats. Try "
                    "'every weekday', 'weekly', 'every Monday' or 'monthly'."
                )
            repeat_label = f", repeating {recurrence_text.strip().lower()}"

        # Conflict detection (#97): held behind the same confirm-then-execute
        # path as send_email, so an overlap is reported before it is created.
        duration = _parse_duration_minutes(
            entities.get("duration"), default=60
        )
        if not entities.get("_confirmed"):
            clashes = self._conflicts_for(start_dt, start_dt + timedelta(minutes=duration))
            if clashes:
                names = ", ".join(
                    f"{c.title!r} at {_clock(c.start, self._tz)}" for c in clashes[:3]
                )
                more = f" and {len(clashes) - 3} more" if len(clashes) > 3 else ""
                keep = {k: v for k, v in entities.items() if k != "_confirmed"}
                return {
                    "response": (
                        f"That overlaps with {names}{more}. "
                        f"Add {title!r} anyway? Say yes to add it."
                    ),
                    "data": {"conflicts": [_event_to_dict(c) for c in clashes]},
                    "confidence": 0.9,
                    "needs_confirmation": True,
                    "confirm_intent": "create_event",
                    "confirm_entities": keep,
                    "confirm_label": f"add {title} despite the overlap",
                }

        create_kwargs: dict[str, Any] = {"title": title, "start_dt": start_dt}
        if "duration" in entities and entities.get("duration") not in (None, ""):
            create_kwargs["end_dt"] = start_dt + timedelta(minutes=duration)
        if recurrence:
            create_kwargs["recurrence"] = recurrence

        try:
            success = self._google.create_event(**create_kwargs)
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
                f"Done. {title!r} added to your calendar for {readable}{repeat_label}.",
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

    # ------------------------------------------------------------------
    # Private – recipients, conflicts (backlog #97, #100)
    # ------------------------------------------------------------------

    def _resolve_recipient(self, to: str) -> Optional[str]:
        """Return a real address for *to* (an address or a configured
        contact name), or None if it can't be resolved safely."""
        to = (to or "").strip()
        if _EMAIL_RE.match(to):
            return to
        found = self._contacts.get(to.lower())
        if found and _EMAIL_RE.match(found):
            return found
        # "email John Smith" where the contact is stored as "john"
        first = to.lower().split(" ")[0] if to else ""
        found = self._contacts.get(first)
        if found and _EMAIL_RE.match(found):
            return found
        return None

    def _events_in_window(self, start: datetime, end: datetime) -> list[Any]:
        """Events overlapping [start, end). Uses the agent's explicit-window
        call when it has one, otherwise a days-ahead fetch (which can only
        look forward from now)."""
        start, end = _aware(start, self._tz), _aware(end, self._tz)
        fn = getattr(self._google, "list_events_between", None)
        if callable(fn):
            return list(fn(start, end, max_results=_MAX_EVENT_DELETE) or [])
        today = datetime.now(self._tz).date()
        days = _clamp_int((end.date() - today).days + 2, 1, _MAX_DAYS_AHEAD)
        return list(
            self._google.list_events(max_results=_MAX_EVENT_DELETE, days_ahead=days) or []
        )

    def _conflicts_for(self, start: datetime, end: datetime) -> list[Any]:
        """Timed events that overlap [start, end). All-day events are not
        treated as conflicts. A failed lookup returns [] (never blocks)."""
        try:
            events = self._events_in_window(start, end)
        except Exception:
            logger.exception("conflict check failed; creating without it.")
            return []
        s, e = _aware(start, self._tz), _aware(end, self._tz)
        clashes = []
        for ev in events:
            bounds = _event_bounds(ev, self._tz)
            if bounds and bounds[0] < e and bounds[1] > s:
                clashes.append(ev)
        return clashes

    # ------------------------------------------------------------------
    # Private – email triage / digest / inbox zero (backlog #92, #99)
    # ------------------------------------------------------------------

    def _fetch_unread(self, count: int) -> Optional[list[Any]]:
        try:
            return list(self._google.read_emails(max_results=count) or [])
        except Exception:
            logger.exception("read_emails() failed.")
            return None

    def _triage(self, emails: list[Any]) -> list[dict[str, Any]]:
        items = []
        for e in emails:
            t = _triage_email(e, self._vips)
            t["email"] = _email_to_dict(e)
            items.append(t)
        # Stable sort keeps the inbox's own newest-first order within a tier.
        items.sort(key=lambda i: -i["score"])
        return items

    def _email_digest(self, entities: dict) -> dict:
        """Rank unread mail by urgency and speak a short digest."""
        count = _clamp_int(
            entities.get("count", _DEFAULT_DIGEST_COUNT), 1, _MAX_EMAIL_COUNT
        )
        emails = self._fetch_unread(count)
        if emails is None:
            return _err("I couldn't fetch your emails right now.")
        if not emails:
            return _ok("Your inbox is clear — no unread emails.", data={"items": []})

        items = self._triage(emails)
        counts = {p: sum(1 for i in items if i["priority"] == p) for p in ("high", "normal", "low")}
        total = len(items)
        parts = [f"You have {total} unread {'email' if total == 1 else 'emails'}."]
        bits = []
        if counts["high"]:
            bits.append(f"{counts['high']} need attention")
        if counts["normal"]:
            bits.append(f"{counts['normal']} look routine")
        if counts["low"]:
            bits.append(f"{counts['low']} look like newsletters or notifications")
        parts.append(", ".join(bits).capitalize() + ".")
        lead = [i for i in items if i["priority"] == "high"] or items
        label = "Most urgent" if counts["high"] else "Newest"
        spoken = [
            f"{n}. From {_sender_name(i['email']['sender'])}: {i['email']['subject']}."
            for n, i in enumerate(lead[:3], start=1)
        ]
        parts.append(f"{label}: " + " ".join(spoken))
        logger.info("email_digest: %d email(s) triaged (%s).", total, counts)
        return _ok(" ".join(parts), data={"items": items, "counts": counts})

    def check_email_digest(self, now: Optional[datetime] = None) -> Optional[str]:
        """
        Heartbeat hook (#92): the spoken digest, at most once a day.

        Returns text to speak or ``None``. Quiet when the feature is off, it
        is before ``digest_time`` or more than six hours past it, today's
        digest was already given, Google isn't connected (retried on a later
        tick), or there is nothing unread. Safe to call on every tick.
        """
        if self._digest_time is None:
            return None
        if now is None:
            now = datetime.now(self._tz)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=self._tz)      # naive = already local wall-clock
        else:
            now = now.astimezone(self._tz)
        due = datetime.combine(now.date(), self._digest_time, tzinfo=self._tz)
        if now < due or now > due + timedelta(hours=_DIGEST_WINDOW_HOURS):
            return None
        today = now.date().isoformat()
        if self._digest_state.get("last_digest") == today:
            return None
        if not self._is_ready():
            return None
        emails = self._fetch_unread(_DEFAULT_DIGEST_COUNT)
        if emails is None:
            return None                     # fetch failed: try again next tick
        self._digest_state["last_digest"] = today
        self._save_state()
        if not emails:
            return None
        result = self._email_digest({"count": _DEFAULT_DIGEST_COUNT})
        text = result.get("response") if result.get("confidence", 0) > 0 else None
        return f"Morning email digest. {text}" if text else None

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path:
            return {}
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self._digest_state), encoding="utf-8")
        except OSError:
            logger.exception("Couldn't save Hermes state to %s.", self._state_path)

    def _inbox_zero(self, entities: dict) -> dict:
        """
        Batch plan for unread mail: a suggested action per message.

        By default this is a plan only. With ``hermes.allow_mailbox_changes``
        on (which also asks Google for the Gmail modify scope), "archive the
        low-priority ones" archives the messages triaged as ``archive`` —
        newsletters, promotions, receipts, automated mail — after a "yes".
        Nothing is ever deleted (archived mail stays in All Mail), and mail
        triaged as read/reply is never touched. Snooze stays a suggestion:
        Gmail's API has no snooze.
        """
        if entities.get("_confirmed") and entities.get("apply"):
            return self._archive_confirmed(entities)

        want_apply = _truthy(entities.get("apply")) or str(
            entities.get("action") or ""
        ).strip().lower() in ("archive", "apply", "do it", "clean")
        count = _clamp_int(
            entities.get("count", _DEFAULT_INBOX_ZERO_COUNT), 1, _MAX_INBOX_ZERO_COUNT
        )
        emails = self._fetch_unread(count)
        if emails is None:
            return _err("I couldn't fetch your emails right now.")
        if not emails:
            return _ok("Inbox zero already — nothing unread.", data={"items": []})

        items = self._triage(emails)
        by_action: dict[str, int] = {}
        for i in items:
            by_action[i["action"]] = by_action.get(i["action"], 0) + 1

        if want_apply:
            return self._archive_preview(items)

        order = ("read", "reply", "snooze", "archive")
        summary = ", ".join(
            f"{by_action[a]} to {a}" for a in order if by_action.get(a)
        )
        lines = [
            f"Here's a plan for your {len(items)} unread: {summary}."
        ]
        for n, i in enumerate([x for x in items if x["action"] in ("read", "reply")][:3], 1):
            lines.append(
                f"{n}. From {_sender_name(i['email']['sender'])}: "
                f"{i['email']['subject']} — {i['action']}."
            )
        if self._can_change_mailbox() and by_action.get("archive"):
            lines.append(
                f"Say \"archive the low-priority ones\" and I'll archive the "
                f"{by_action['archive']} marked archive, after you confirm."
            )
        else:
            lines.append("I haven't changed anything in your mailbox.")
        return _ok(" ".join(lines), data={"items": items, "by_action": by_action})

    def _can_change_mailbox(self) -> bool:
        return bool(
            self._allow_mailbox_changes
            and callable(getattr(self._google, "archive_emails", None))
            and getattr(self._google, "can_modify_mailbox", True)
        )

    def _archive_preview(self, items: list[dict[str, Any]]) -> dict:
        if not self._can_change_mailbox():
            return _ok(
                "I can only suggest actions right now. To let me archive mail, set "
                "hermes.allow_mailbox_changes to true in your config and reconnect Google "
                "once so it can ask for that permission.",
                data={"items": items}, confidence=0.7,
            )
        targets = [
            i for i in items
            if i["action"] == "archive" and i["email"].get("message_id")
        ][:_MAX_ARCHIVE_BATCH]
        if not targets:
            return _ok(
                "Nothing in your unread looks safe to archive.",
                data={"items": items}, confidence=0.8,
            )
        senders = []
        for t in targets:
            name = _sender_name(t["email"]["sender"])
            if name not in senders:
                senders.append(name)
        sample = ", ".join(senders[:3]) + (f" and {len(senders) - 3} more" if len(senders) > 3 else "")
        ids = [t["email"]["message_id"] for t in targets]
        n = len(ids)
        return {
            "response": (
                f"Archive {n} {'email' if n == 1 else 'emails'} (from {sample})? "
                "They stay in All Mail and nothing is deleted. Say yes to archive."
            ),
            "data": {"ids": ids, "count": n},
            "confidence": 0.9,
            "needs_confirmation": True,
            "confirm_intent": "inbox_zero",
            "confirm_entities": {"apply": True, "ids": ids},
            "confirm_label": f"archive {n} low-priority {'email' if n == 1 else 'emails'}",
        }

    def _archive_confirmed(self, entities: dict) -> dict:
        if not self._can_change_mailbox():
            return _err("Mailbox changes aren't enabled, so I archived nothing.")
        ids = [
            str(i) for i in (entities.get("ids") or [])
            if isinstance(i, (str, int)) and str(i).strip()
        ][:_MAX_ARCHIVE_BATCH]
        if not ids:
            return _err("I lost track of which emails to archive. Ask me again.")
        try:
            done = int(self._google.archive_emails(ids))
        except Exception:
            logger.exception("archive_emails() raised.")
            return _err("I couldn't archive those emails due to an unexpected error.")
        if done <= 0:
            return _err("I couldn't archive those emails.")
        if done < len(ids):
            return _ok(f"Archived {done} of {len(ids)} emails; the rest didn't go through.",
                       data={"archived": done}, confidence=0.7)
        logger.info("inbox_zero: archived %d message(s).", done)
        return _ok(
            f"Archived {done} {'email' if done == 1 else 'emails'}. "
            "They're still in All Mail if you need them.",
            data={"archived": done}, confidence=0.9,
        )

    # ------------------------------------------------------------------
    # Private – drafting (backlog #93)
    # ------------------------------------------------------------------

    def _draft_email(self, entities: dict) -> dict:
        """
        Turn a short instruction into a draft and hand it to the normal
        send flow, so a draft can never go out without a spoken "yes".
        """
        to = (entities.get("to") or "").strip()
        greet = (entities.get("_greet") or to or "").strip()

        if entities.get("_drafted") and entities.get("body"):
            subject = (entities.get("subject") or _DEFAULT_SUBJECT).strip()
            body = str(entities["body"]).strip()
        else:
            instruction = (
                entities.get("instruction")
                or entities.get("body")
                or entities.get("message")
                or entities.get("raw_query")
                or ""
            ).strip()
            if not instruction:
                return _clarify(
                    "What should the email say?", slot="instruction", entities=entities
                )
            name = "" if _EMAIL_RE.match(greet) else greet
            subject, body = self._compose(
                instruction, name, (entities.get("subject") or "").strip(),
                (entities.get("tone") or "").strip(),
            )

        preview = _truncate(body, 120)
        keep = dict(entities)
        keep.update({"subject": subject, "body": body, "_drafted": True, "_greet": greet})

        if not to:
            return _clarify(
                f'Here\'s a draft with subject "{subject}": "{preview}" '
                "Who should I send it to?",
                slot="to", entities=keep,
            )
        if self._resolve_recipient(to) is None:
            return _clarify(
                f'Here\'s a draft with subject "{subject}": "{preview}" '
                f"What's the email address for {to}?",
                slot="to", entities=keep,
            )

        result = self._send_email({"to": to, "subject": subject, "body": body})
        result["response"] = "Here's my draft. " + result["response"]
        result.setdefault("data", {})["draft"] = {"subject": subject, "body": body}
        return result

    def _compose(self, instruction: str, name: str, subject: str, tone: str) -> tuple[str, str]:
        """LLM wording when available and valid, otherwise a template."""
        if self._llm is not None:
            try:
                prompt = (
                    "Write a short, polite email for the user based on this "
                    f"instruction: {instruction!r}.\n"
                    + (f"Recipient first name: {name}.\n" if name else "")
                    + (f"Tone: {tone}.\n" if tone else "")
                    + "Do not invent facts, dates or names that are not in the "
                    "instruction. Reply ONLY with JSON: "
                    '{"subject": "...", "body": "..."}'
                )
                raw = self._llm.generate(prompt, fmt="json")
                data = json.loads(raw) if isinstance(raw, str) else raw
                body = str(data.get("body") or "").strip()
                if body:
                    return (str(data.get("subject") or subject or _DEFAULT_SUBJECT).strip(), body)
            except Exception:
                logger.warning("LLM draft failed; using the template.", exc_info=True)
        return _template_draft(instruction, name, subject)

    # ------------------------------------------------------------------
    # Private – email search (backlog #94)
    # ------------------------------------------------------------------

    def _search_email(self, entities: dict) -> dict:
        """Search by sender / subject / date range, not just 'unread'."""
        search = getattr(self._google, "search_emails", None)
        if not callable(search):
            return _err("Email search isn't available with this Google connection.")

        parts: list[str] = []
        desc: list[str] = []

        sender = (entities.get("sender") or entities.get("from") or "").strip()
        if sender:
            resolved = self._resolve_recipient(sender) or sender
            parts.append(f"from:{_gmail_quote(resolved)}")
            desc.append(f"from {sender}")

        subject = (entities.get("subject") or "").strip()
        if subject:
            parts.append(f"subject:{_gmail_quote(subject)}")
            desc.append(f'about "{subject}"')

        try:
            after, before = _search_date_range(entities, self._tz)
        except DateTimeParseError:
            return _clarify(
                "I couldn't understand that date range. Try 'yesterday', "
                "'last week', or a date like 2026-03-01."
            )
        if after:
            parts.append(f"after:{after:%Y/%m/%d}")
            desc.append(f"since {after.day} {after:%B}")
        if before:
            parts.append(f"before:{before:%Y/%m/%d}")
            desc.append(f"before {before.day} {before:%B}")

        free = (entities.get("query") or entities.get("keywords") or "").strip()
        if free:
            parts.append(free)
            desc.append(f'containing "{free}"')

        if not parts:
            return _clarify(
                "What should I search for? You can give a sender, a subject "
                "or a date range.",
                slot="query", entities=entities,
            )

        count = _clamp_int(entities.get("count", _DEFAULT_SEARCH_COUNT), 1, _MAX_EMAIL_COUNT)
        query = " ".join(parts)
        try:
            emails = list(search(query, max_results=count) or [])
        except Exception:
            logger.exception("search_emails() failed.")
            return _err("I couldn't search your emails right now.")

        what = " ".join(desc)
        if not emails:
            return _ok(
                f"I didn't find any emails {what}.",
                data={"emails": [], "query": query}, confidence=0.9,
            )
        n = len(emails)
        lines = [f"I found {n} {'email' if n == 1 else 'emails'} {what}."]
        for i, e in enumerate(emails[:3], start=1):
            d = _email_to_dict(e)
            when = _short_date(d.get("date", ""))
            lines.append(
                f"{i}. From {_sender_name(d.get('sender', ''))}: {d.get('subject', '')}"
                + (f" ({when})." if when else ".")
            )
        if n > 3:
            lines.append(f"And {n - 3} more.")
        return _ok(" ".join(lines), data={"emails": [_email_to_dict(e) for e in emails], "query": query})

    # ------------------------------------------------------------------
    # Private – schedule gaps (backlog #95)
    # ------------------------------------------------------------------

    def _check_schedule_gaps(self, entities: dict) -> dict:
        """
        Flag overlapping and back-to-back events on one day.

        When two consecutive events have different locations, travel time is
        added to the buffer: a driving-time lookup when a travel provider is
        configured (``hermes.travel.provider``), otherwise (or if the lookup
        fails) a flat ``travel_minutes`` allowance. The reply says which.
        """
        date_str = (entities.get("date") or "today").strip()
        try:
            day = _resolve_date(date_str, self._tz)
        except DateTimeParseError:
            return _clarify(
                "I couldn't understand that date. Try 'today', 'tomorrow' or '2026-12-25'."
            )
        buffer_min = _parse_duration_minutes(
            entities.get("buffer"), default=self._buffer_minutes, lo=0, hi=240
        )
        start = datetime.combine(day, time(0, 0))
        try:
            events = self._events_in_window(start, start + timedelta(days=1))
        except Exception:
            logger.exception("check_schedule_gaps: event fetch failed.")
            return _err("I couldn't fetch your calendar right now.")

        timed = []
        for ev in events:
            b = _event_bounds(ev, self._tz)
            if b and b[0].date() == day:
                timed.append((b[0], b[1], ev))
        timed.sort(key=lambda t: t[0])
        label = _day_label(day, self._tz)

        if len(timed) < 2:
            return _ok(
                f"You have {len(timed)} timed {'event' if len(timed) == 1 else 'events'} "
                f"{label}, so there's nothing back-to-back.",
                data={"flags": [], "events": len(timed)}, confidence=0.9,
            )

        flags = []
        for (_, a_end, a), (b_start, _, b) in zip(timed, timed[1:]):
            gap = int((b_start - a_end).total_seconds() // 60)
            a_loc = (getattr(a, "location", "") or "").strip().lower()
            b_loc = (getattr(b, "location", "") or "").strip().lower()
            travel = bool(a_loc and b_loc and a_loc != b_loc)
            travel_min, travel_src = 0, ""
            if travel:
                est = self._lookup_travel(a_loc, b_loc)
                if est is not None:
                    travel_min, travel_src = est
                else:
                    travel_min, travel_src = self._travel_minutes, "flat"
            needed = buffer_min + travel_min
            if gap >= needed:
                continue
            if gap < 0:
                kind = "overlap"
            elif travel:
                kind = "travel"
            else:
                kind = "back-to-back"
            flags.append({
                "first": a.title, "second": b.title, "gap_minutes": gap,
                "needed_minutes": needed, "kind": kind,
                "travel_minutes": travel_min, "travel_source": travel_src,
            })

        if not flags:
            return _ok(
                f"Your {len(timed)} events {label} all have enough breathing room.",
                data={"flags": [], "events": len(timed)}, confidence=0.9,
            )

        def _sentence(f: dict) -> str:
            if f["kind"] == "overlap":
                return f"{f['first']!r} and {f['second']!r} overlap by {-f['gap_minutes']} minutes"
            if f["kind"] == "travel":
                if f["travel_source"] not in ("", "flat"):
                    return (
                        f"only {f['gap_minutes']} minutes between {f['first']!r} and "
                        f"{f['second']!r}, which are about {f['travel_minutes']} minutes "
                        f"apart by car"
                    )
                return (
                    f"only {f['gap_minutes']} minutes between {f['first']!r} and "
                    f"{f['second']!r}, which are in different places"
                )
            return f"only {f['gap_minutes']} minutes between {f['first']!r} and {f['second']!r}"

        n = len(flags)
        lines = [f"{n} tight {'spot' if n == 1 else 'spots'} {label}."]
        for i, f in enumerate(flags[:3], start=1):
            lines.append(f"{i}. {_sentence(f).capitalize()}.")
        if n > 3:
            lines.append(f"And {n - 3} more.")
        if any(f["kind"] == "travel" and f["travel_source"] in ("", "flat") for f in flags):
            lines.append(
                f"Travel is a flat {self._travel_minutes}-minute estimate, not a route lookup."
            )
        sources = sorted({
            f["travel_source"] for f in flags
            if f["kind"] == "travel" and f["travel_source"] not in ("", "flat")
        })
        if sources:
            lines.append(
                f"Drive times come from {' and '.join(sources)} and don't include traffic."
            )
        return _ok(" ".join(lines), data={"flags": flags, "events": len(timed)})

    def _lookup_travel(self, origin: str, destination: str) -> Optional[tuple[int, str]]:
        """(minutes, source) from the travel estimator, or None to use the flat
        allowance. Never raises."""
        if self._travel is None:
            return None
        try:
            est = self._travel.estimate(origin, destination)
        except Exception:
            logger.exception("travel estimate failed; using the flat allowance.")
            return None
        if est is None:
            return None
        try:
            return max(0, int(est.minutes)), str(est.source)
        except (AttributeError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Private – meeting slots (backlog #96)
    # ------------------------------------------------------------------

    def _find_meeting_slot(self, entities: dict) -> dict:
        """Propose up to three free slots inside working hours, using free/busy
        for you and any attendees whose calendars are visible to you."""
        duration = _parse_duration_minutes(
            entities.get("duration"), default=_DEFAULT_MEETING_MINUTES, lo=5, hi=480
        )

        names = _split_attendees(
            entities.get("attendees") or entities.get("attendee") or entities.get("with")
        )
        emails: list[str] = []
        unresolved: list[str] = []
        for n in names:
            addr = self._resolve_recipient(n)
            (emails if addr else unresolved).append(addr or n)

        explicit_date = (entities.get("date") or "").strip()
        try:
            first_day = _resolve_date(explicit_date, self._tz) if explicit_date else datetime.now(self._tz).date()
        except DateTimeParseError:
            return _clarify(
                "I couldn't understand that date. Try 'tomorrow' or '2026-12-25'."
            )
        if "days" in entities and entities.get("days") not in (None, ""):
            days = _clamp_int(entities["days"], 1, _MAX_SLOT_DAYS)
        else:
            days = 1 if explicit_date else _DEFAULT_SLOT_DAYS

        win_start = datetime.combine(first_day, time(0, 0))
        win_end = win_start + timedelta(days=days)

        busy: list[tuple[datetime, datetime]] = []
        unknown: list[str] = []
        fb = getattr(self._google, "free_busy", None)
        result = None
        if callable(fb):
            try:
                result = fb(_aware(win_start, self._tz), _aware(win_end, self._tz), emails)
            except Exception:
                logger.exception("free_busy() failed.")
        if result and result.get("primary") is not None:
            busy.extend(result["primary"])
            for addr in emails:
                got = result.get(addr)
                if got is None:
                    unknown.append(addr)
                else:
                    busy.extend(got)
        else:
            # No free/busy (or it failed): fall back to your own events only.
            unknown = list(emails)
            try:
                for ev in self._events_in_window(win_start, win_end):
                    b = _event_bounds(ev, self._tz)
                    if b:
                        busy.append(b)
            except Exception:
                logger.exception("find_meeting_slot: event fetch failed.")
                return _err("I couldn't read your calendar right now.")

        busy = [(_aware(s, self._tz), _aware(e, self._tz)) for s, e in busy]
        now = datetime.now(self._tz)
        ws, we = self._work_hours
        slots: list[datetime] = []
        for offset in range(days):
            d = first_day + timedelta(days=offset)
            if d.weekday() >= 5 and not explicit_date:
                continue
            t = datetime.combine(d, time(ws, 0), tzinfo=self._tz)
            day_end = datetime.combine(d, time(0, 0), tzinfo=self._tz) + timedelta(hours=we)
            while t + timedelta(minutes=duration) <= day_end and len(slots) < _MAX_SLOTS_OFFERED:
                end = t + timedelta(minutes=duration)
                if t >= now and not any(s < end and e > t for s, e in busy):
                    slots.append(t)
                    t = end  # offer non-overlapping options
                else:
                    t += timedelta(minutes=_SLOT_STEP_MINUTES)
            if len(slots) >= _MAX_SLOTS_OFFERED:
                break

        notes = []
        if unknown:
            notes.append(
                "I couldn't see the calendar for " + ", ".join(unknown)
                + ", so those times only account for yours."
            )
        if unresolved:
            notes.append(
                "I don't have an email address for " + ", ".join(unresolved)
                + ", so I didn't check them."
            )

        who = "you" if not emails else "you and " + ", ".join(emails)
        if not slots:
            msg = (
                f"I couldn't find a {duration}-minute slot between {ws}:00 and "
                f"{we}:00 over {'that day' if days == 1 else f'the next {days} days'}."
            )
            return _ok(" ".join([msg] + notes), data={"slots": []}, confidence=0.8)

        lines = [f"Here are {len(slots)} {duration}-minute times that work for {who}."]
        for i, t in enumerate(slots, start=1):
            lines.append(f"{i}. {t:%A} {t.day} {t:%B} at {t:%H:%M}.")
        lines.extend(notes)
        # Remembered briefly so "book the first one" can follow (#96).
        self._last_slots = {
            "slots": list(slots), "duration": duration, "attendees": list(emails),
            "unchecked": list(unknown), "at": datetime.now(self._tz),
        }
        lines.append("Say \"book the first one\" and I'll add it to your calendar"
                     + (" and invite them." if emails else "."))
        return _ok(
            " ".join(lines),
            data={"slots": [t.isoformat() for t in slots], "duration_minutes": duration,
                  "unchecked": unknown + unresolved},
        )


    # ------------------------------------------------------------------
    # Private – Todoist (backlog #91)
    # ------------------------------------------------------------------

    def _todoist_tasks(self) -> Optional[list[TodoistTask]]:
        try:
            return list(self._todoist.list_tasks())
        except TodoistError as exc:
            logger.warning("Todoist list failed: %s", exc)
            self._todoist_error = str(exc)
            return None
        except Exception:
            logger.exception("Todoist list raised.")
            self._todoist_error = "Something went wrong talking to Todoist."
            return None

    def _todoist_fail(self) -> dict:
        msg = self._todoist_error or "I couldn't reach Todoist right now."
        return _err(msg if msg.startswith(("Todoist", "Something", "I ")) else f"Todoist: {msg}")

    def _today(self) -> date:
        return datetime.now(self._tz).date()

    @staticmethod
    def _task_phrase(t: TodoistTask, today: date) -> str:
        bits = []
        if t.due_date and t.due_date < today:
            days = (today - t.due_date).days
            bits.append("overdue" if days <= 1 else f"{days} days overdue")
        elif t.due_date == today:
            bits.append("due today")
        elif t.due_date:
            bits.append(f"due {t.due_date:%A} {t.due_date.day} {t.due_date:%B}")
        if t.priority >= 3:
            bits.append(f"p{t.ui_priority}")
        return f"{t.content}" + (f" ({', '.join(bits)})" if bits else "")

    def _todoist_list(self, entities: dict) -> dict:
        scope = str(
            entities.get("scope") or entities.get("when") or entities.get("filter")
            or entities.get("date") or "today"
        ).strip().lower()
        tasks = self._todoist_tasks()
        if tasks is None:
            return self._todoist_fail()
        today = self._today()
        chosen = rank_tasks(select_due(tasks, today, scope), today)
        label = {"overdue": "overdue", "week": "due this week", "all": "open"}.get(
            scope, "due today or overdue"
        )
        if not chosen:
            extra = f" You have {len(tasks)} open in total." if tasks and scope != "all" else ""
            return _ok(f"Nothing {label} in Todoist.{extra}",
                       data={"tasks": [], "open_total": len(tasks)})
        n = len(chosen)
        lines = [f"{n} {'task' if n == 1 else 'tasks'} {label}."]
        for i, t in enumerate(chosen[:_TODOIST_SPOKEN], start=1):
            lines.append(f"{i}. {self._task_phrase(t, today)}.")
        if n > _TODOIST_SPOKEN:
            lines.append(f"And {n - _TODOIST_SPOKEN} more.")
        return _ok(" ".join(lines), data={
            "tasks": [t.to_dict() for t in chosen], "open_total": len(tasks)})

    def _todoist_prioritize(self, entities: dict) -> dict:
        """Rank everything open: overdue, then due today, then by Todoist
        priority. Read-only: it doesn't change anything in Todoist."""
        tasks = self._todoist_tasks()
        if tasks is None:
            return self._todoist_fail()
        if not tasks:
            return _ok("Your Todoist is empty. Nothing to prioritise.", data={"tasks": []})
        today = self._today()
        ranked = rank_tasks(tasks, today)
        top = ranked[:3]
        overdue = sum(1 for t in tasks if t.due_date and t.due_date < today)
        due_today = sum(1 for t in tasks if t.due_date == today)
        head = f"You have {len(tasks)} open tasks"
        if overdue or due_today:
            parts = []
            if overdue:
                parts.append(f"{overdue} overdue")
            if due_today:
                parts.append(f"{due_today} due today")
            head += f", {' and '.join(parts)}"
        lines = [head + ". Start with:"]
        for i, t in enumerate(top, start=1):
            lines.append(f"{i}. {self._task_phrase(t, today)}.")
        return _ok(" ".join(lines), data={
            "tasks": [t.to_dict() for t in ranked[:10]],
            "overdue": overdue, "due_today": due_today, "open_total": len(tasks)})

    def _todoist_add(self, entities: dict) -> dict:
        content = str(
            entities.get("task") or entities.get("content") or entities.get("title")
            or entities.get("text") or ""
        ).strip()
        if not content:
            return _clarify("What should the task say?", slot="task", entities=entities)
        due = str(entities.get("due") or entities.get("date") or entities.get("when") or "").strip()
        priority = _parse_todoist_priority(entities.get("priority"))
        try:
            task = self._todoist.add_task(content, due_string=due, priority=priority)
        except TodoistError as exc:
            logger.warning("Todoist add failed: %s", exc)
            return _err(str(exc))
        except Exception:
            logger.exception("Todoist add raised.")
            return _err("I couldn't add that task due to an unexpected error.")
        when = f", due {task.due_string or due}" if (task.due_string or due) else ""
        pr = f", priority {task.ui_priority}" if task.priority >= 2 else ""
        return _ok(f"Added {task.content!r} to Todoist{when}{pr}.",
                   data={"task": task.to_dict()}, confidence=0.9)

    def _todoist_complete(self, entities: dict) -> dict:
        query = str(
            entities.get("task") or entities.get("content") or entities.get("name")
            or entities.get("title") or entities.get("text") or ""
        ).strip()
        if not query:
            return _clarify("Which task did you finish?", slot="task", entities=entities)
        tasks = self._todoist_tasks()
        if tasks is None:
            return self._todoist_fail()
        matches = find_matches(tasks, query)
        if not matches:
            return _ok(f"I couldn't find an open Todoist task matching {query!r}.",
                       data={"matches": []}, confidence=0.6)
        if len(matches) > 1:
            names = "; ".join(repr(m.content) for m in matches[:3])
            more = f" and {len(matches) - 3} more" if len(matches) > 3 else ""
            return _clarify(
                f"That matches {len(matches)} tasks: {names}{more}. Which one?",
                slot="task", entities=entities,
            )
        target = matches[0]
        try:
            self._todoist.complete_task(target.id)
        except TodoistError as exc:
            logger.warning("Todoist complete failed: %s", exc)
            return _err(str(exc))
        except Exception:
            logger.exception("Todoist complete raised.")
            return _err("I couldn't complete that task due to an unexpected error.")
        return _ok(f"Marked {target.content!r} as done in Todoist.",
                   data={"task": target.to_dict()}, confidence=0.9)

    def _book_meeting_slot(self, entities: dict) -> dict:
        """
        Put one of the times ``find_meeting_slot`` just proposed on the
        calendar and invite the attendees (backlog #96).

        Two-phase like ``send_email``: the first call previews, only the
        confirmed second call creates the event (which makes Google email the
        invitations). The slot is re-checked against your calendar first.
        """
        if entities.get("_confirmed"):
            return self._book_confirmed(entities)

        mem = self._last_slots
        now = datetime.now(self._tz)
        if (
            not mem or not mem.get("slots")
            or (now - mem["at"]) > timedelta(minutes=_SLOT_MEMORY_MINUTES)
        ):
            return _ok(
                "I don't have any proposed times to book. Ask me to find a time first.",
                confidence=0.6,
            )
        slots = mem["slots"]
        raw_choice = next(
            (entities[k] for k in ("choice", "slot", "number", "which", "option")
             if entities.get(k) not in (None, "")),
            None,
        )
        if raw_choice is None:
            if len(slots) > 1:
                return _clarify(
                    f"Which one — 1 to {len(slots)}?", slot="choice", entities=entities
                )
            idx = 0
        else:
            idx = _parse_choice(raw_choice, len(slots))
            if idx is None:
                return _clarify(
                    f"Which one — 1 to {len(slots)}?", slot="choice", entities=entities
                )

        start: datetime = slots[idx]
        duration: int = int(mem["duration"])
        attendees: list[str] = list(mem["attendees"])
        title = (
            entities.get("title") or entities.get("task") or entities.get("event") or ""
        ).strip() or ("Meeting" if not attendees else f"Meeting with {', '.join(attendees)}")
        when = f"{start:%A} {start.day} {start:%B} at {start:%H:%M}"
        invite = f" and invite {', '.join(attendees)}" if attendees else ""
        caveat = ""
        if mem.get("unchecked"):
            caveat = (
                " I couldn't see " + ", ".join(mem["unchecked"])
                + "'s calendar, so I don't know if that time suits them."
            )
        return {
            "response": (
                f"Book {title!r} for {when} ({duration} minutes){invite}? "
                f"Say yes to book it.{caveat}"
            ),
            "data": {"start": start.isoformat(), "duration": duration, "attendees": attendees},
            "confidence": 0.9,
            "needs_confirmation": True,
            "confirm_intent": "book_meeting_slot",
            "confirm_entities": {
                "start": start.isoformat(), "duration": duration,
                "attendees": attendees, "title": title,
            },
            "confirm_label": f"book {title} on {when}",
        }

    def _book_confirmed(self, entities: dict) -> dict:
        try:
            start = datetime.fromisoformat(str(entities.get("start")))
            duration = int(entities.get("duration"))
        except (TypeError, ValueError):
            return _err("I lost track of which time you picked. Ask me to find a time again.")
        if not 5 <= duration <= 480:
            return _err("That meeting length doesn't look right. Ask me to find a time again.")
        attendees = [
            a for a in (entities.get("attendees") or [])
            if isinstance(a, str) and _EMAIL_RE.match(a.strip())
        ]
        title = str(entities.get("title") or "Meeting").strip() or "Meeting"
        end = start + timedelta(minutes=duration)

        clashes = self._conflicts_for(start, end)
        if clashes:
            names = ", ".join(f"{c.title!r}" for c in clashes[:2])
            return _err(f"That time has since filled up ({names}). Ask me to find another.")

        kwargs: dict[str, Any] = {
            "title": title,
            "start_dt": start.replace(tzinfo=None) if start.tzinfo else start,
            "end_dt": (end.replace(tzinfo=None) if end.tzinfo else end),
        }
        if attendees:
            kwargs["attendees"] = attendees
        try:
            ok = self._google.create_event(**kwargs)
        except Exception:
            logger.exception("book_meeting_slot: create_event raised for %r.", title)
            return _err("I couldn't book that meeting due to an unexpected error.")
        if not ok:
            return _err("I couldn't book that meeting.")
        self._last_slots = None
        when = f"{start:%A} {start.day} {start:%B} at {start:%H:%M}"
        sent = f" Invitations are on their way to {', '.join(attendees)}." if attendees else ""
        logger.info("book_meeting_slot: %r booked at %s (%d invitee(s)).",
                    title, start.isoformat(), len(attendees))
        return _ok(f"Booked {title!r} for {when}.{sent}", confidence=0.9)


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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "y", "on", "apply", "do it")


def _parse_clock_setting(value: Any) -> Optional[time]:
    """'08:30' / '8:30' / '8' -> time; anything unreadable disables the feature."""
    if value in (None, "", False):
        return None
    m = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*", str(value))
    if not m:
        logger.warning("Ignoring unreadable time setting %r (use HH:MM).", value)
        return None
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        logger.warning("Ignoring out-of-range time setting %r.", value)
        return None
    return time(h, mi)


_ORDINALS = {
    "first": 0, "1st": 0, "one": 0, "1": 0,
    "second": 1, "2nd": 1, "two": 1, "2": 1,
    "third": 2, "3rd": 2, "three": 2, "3": 2,
}


def _parse_choice(value: Any, n: int) -> Optional[int]:
    """'first', 'the second one', 2, 'last' -> zero-based index within n slots."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value - 1 if 1 <= value <= n else None
    t = str(value).strip().lower()
    if re.search(r"\blast\b", t):
        return n - 1
    for word in re.findall(r"[a-z0-9]+", t):
        if word in _ORDINALS:
            idx = _ORDINALS[word]
            return idx if idx < n else None
    return None


def _parse_todoist_priority(value: Any) -> Optional[int]:
    """Words or p1-p4 (as shown in the Todoist app) -> API priority (4 = urgent).
    Unknown or missing -> None (leave Todoist's default)."""
    if value in (None, ""):
        return None
    t = str(value).strip().lower()
    table = {
        "p1": 4, "1": 4, "urgent": 4, "highest": 4, "critical": 4,
        "p2": 3, "2": 3, "high": 3, "important": 3,
        "p3": 2, "3": 2, "medium": 2, "normal": 2,
        "p4": 1, "4": 1, "low": 1, "lowest": 1,
    }
    return table.get(t)


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


def _clarify(
    question: str,
    *,
    slot: Optional[str] = None,
    entities: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Ask for one missing piece of information.

    `slot`/`entities` are optional (backlog #29): when supplied, the
    orchestrator holds this as a PendingSlotFill and feeds the user's
    verbatim next reply into `entities[slot]` before re-dispatching
    straight back to this same handler — so "Who should I send it to?"
    followed by "raj@example.com" actually completes the email instead of
    the reply vanishing into a fresh, unrelated classification. Omit both
    for a clarification that doesn't cleanly reduce to one entity key
    (e.g. "I couldn't understand that date" — the fix isn't a single
    verbatim slot value in the same way).
    """
    data: dict[str, Any] = {"needs_clarification": True}
    if slot:
        data["missing_slot"] = slot
        data["slot_entities"] = dict(entities or {})
    return {"response": question, "data": data, "confidence": 0.5}


# ---------------------------------------------------------------------------
# Module-level pure helpers — drafting, triage, recurrence, durations
# (backlog #92-#99)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[A-Za-z]{2,}$")


def _aware(dt: datetime, tz: "ZoneInfo") -> datetime:
    """Naive datetimes are read as wall-clock time in *tz*."""
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


def _parse_iso(value: str, tz: "ZoneInfo") -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=tz)


def _event_bounds(event: Any, tz: "ZoneInfo") -> Optional[tuple[datetime, datetime]]:
    """Aware (start, end) of a *timed* event; None for all-day or malformed
    events. A missing end is taken as one hour after the start."""
    start = getattr(event, "start", None)
    end = getattr(event, "end", None)
    if isinstance(event, dict):
        start = event.get("start", start)
        end = event.get("end", end)
    if not start or len(start) <= 10:
        return None
    s = _parse_iso(start, tz)
    if s is None:
        return None
    e = _parse_iso(end, tz) if end and len(end) > 10 else None
    if e is None or e <= s:
        e = s + timedelta(hours=1)
    return s, e


def _clock(iso: str, tz: "ZoneInfo") -> str:
    dt = _parse_iso(iso, tz)
    return f"{dt:%H:%M}" if dt else "an unknown time"


def _day_label(day: date, tz: "ZoneInfo") -> str:
    today = datetime.now(tz).date()
    if day == today:
        return "today"
    if day == today + timedelta(days=1):
        return "tomorrow"
    return f"on {day:%A} {day.day} {day:%B}"


def _sender_name(sender: str) -> str:
    name = (sender or "").split("<")[0].strip().strip('"')
    return name or (sender or "Unknown").strip("<> ")


def _short_date(header: str) -> str:
    if not header:
        return ""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(header)
        return f"{dt.day} {dt:%b}"
    except Exception:
        return ""


def _gmail_quote(value: str) -> str:
    value = value.replace('"', "").strip()
    return f'"{value}"' if re.search(r"\s", value) else value


def _split_attendees(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = re.split(r"[,;]|\band\b|&", str(value), flags=re.IGNORECASE)
    return [i.strip() for i in items if i and i.strip()]


# -- durations --------------------------------------------------------------

def _parse_duration_minutes(
    value: Any, *, default: int, lo: int = 5, hi: int = 720
) -> int:
    """'30', '45 minutes', '1 hour', '1h30', 'half an hour', 'an hour' → minutes.
    Unreadable or out-of-range values give *default*."""
    if value is None or value == "":
        return default
    minutes: Optional[float] = None
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        minutes = float(value)
    else:
        t = str(value).strip().lower()
        if "half an hour" in t or "half hour" in t:
            minutes = 30
        elif re.fullmatch(r"(an|1|one)\s*(hour|hr)", t):
            minutes = 60
        else:
            h = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", t)
            m = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes)\b", t)
            h_compact = re.fullmatch(r"(\d+)h(\d{1,2})", t)
            if h_compact:
                minutes = int(h_compact.group(1)) * 60 + int(h_compact.group(2))
            elif h or m:
                minutes = (float(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)
            elif re.fullmatch(r"\d+(?:\.\d+)?", t):
                minutes = float(t)
    if minutes is None or not (lo <= minutes <= hi):
        return default
    return int(round(minutes))


# -- search date range (#94) ------------------------------------------------

def _search_date_range(
    entities: dict, tz: "ZoneInfo"
) -> tuple[Optional[date], Optional[date]]:
    """(after, before) as dates; *before* is exclusive, as in Gmail's
    ``before:``. Raises DateTimeParseError for text it can't read."""
    today = datetime.now(tz).date()
    after = before = None

    when = (entities.get("date") or entities.get("when") or "").strip().lower()
    if when:
        monday = today - timedelta(days=today.weekday())
        if when == "today":
            after, before = today, today + timedelta(days=1)
        elif when == "yesterday":
            after, before = today - timedelta(days=1), today
        elif when in ("this week",):
            after, before = monday, monday + timedelta(days=7)
        elif when in ("last week", "past week"):
            after, before = monday - timedelta(days=7), monday
        elif when == "this month":
            after = today.replace(day=1)
            before = (after + timedelta(days=32)).replace(day=1)
        elif when == "last month":
            before = today.replace(day=1)
            after = (before - timedelta(days=1)).replace(day=1)
        else:
            d = _resolve_date(when, tz)  # raises DateTimeParseError
            after, before = d, d + timedelta(days=1)

    for key, is_after in (("after", True), ("since", True), ("before", False), ("until", False)):
        raw = (entities.get(key) or "").strip()
        if not raw:
            continue
        low = raw.lower()
        d = (
            today - timedelta(days=1) if low == "yesterday" else _resolve_date(raw, tz)
        )
        if is_after:
            after = d
        else:
            before = d
    if after and before and before <= after:
        raise DateTimeParseError("'before' is not later than 'after'.")
    return after, before


# -- drafting (#93) ---------------------------------------------------------

def _sentence(text: str) -> str:
    text = text.strip().strip("\"'").rstrip(".!? ")
    return (text[:1].upper() + text[1:] + ".") if text else ""


def _template_draft(instruction: str, name: str, subject: str = "") -> tuple[str, str]:
    """Deterministic fallback draft. Recognises a few common intents and
    otherwise restates the instruction politely; it never invents details."""
    low = instruction.lower()
    first = name.strip().split(" ")[0].capitalize() if name.strip() else ""
    greeting = f"Hi {first}," if first else "Hi,"

    alt = None
    m = re.search(r"\bsuggest(?:ing)?\s+(.+)$", low) or re.search(
        r"\b(?:reschedul\w*\s+(?:to|for)|how about|instead on|instead)\s+(.+)$", low
    )
    if m:
        alt = re.sub(r"\s+instead$", "", m.group(1).strip(" .!?"))
        alt = alt[:1].upper() + alt[1:]

    if re.search(r"\b(can'?t|cannot|won'?t be able to|unable to|not able to)\s+(?:make|attend|come|join|do)\b", low):
        lines = ["Unfortunately I can't make it."]
        if alt:
            lines.append(f"Would {alt} work for you instead?")
        else:
            lines.append("Sorry for the inconvenience.")
        return (subject or "Can't make it", _assemble(greeting, lines))
    if re.search(r"\b(running late|be late|will be late|i'?m late)\b", low):
        return (subject or "Running late", _assemble(greeting, ["I'm running a little late. Sorry about that — I'll be there as soon as I can."]))
    if re.search(r"\bthank", low):
        return (subject or "Thank you", _assemble(greeting, ["Thank you — I really appreciate it."]))
    if re.search(r"\b(follow(?:ing)? up|checking in|any update)\b", low):
        return (subject or "Following up", _assemble(greeting, ["I wanted to follow up and see if there's any update. Thanks!"]))

    # Unrecognised: drop a leading "reply/email/tell/say(ing)" so we keep the message.
    core = re.sub(
        r"^(?:please\s+)?(?:(?:reply|respond|write|send|email|draft)(?:\s+an?\s+email)?"
        r"(?:\s+to\s+\S+)?|tell(?:\s+\S+)?)\s*(?:saying|that|and say|to say)?\s*",
        "", instruction.strip(), flags=re.IGNORECASE,
    ) or instruction.strip()
    return (subject or _DEFAULT_SUBJECT, _assemble(greeting, [_sentence(core)]))


def _assemble(greeting: str, lines: list[str]) -> str:
    return f"{greeting}\n\n" + " ".join(lines) + "\n\nBest regards"


# -- triage (#92, #99) ------------------------------------------------------

_STRONG_RE = re.compile(
    r"\b(urgent|asap|immediately|action required|deadline|due (?:today|tomorrow|soon)|"
    r"overdue|final notice|expires? (?:today|tomorrow|soon)|expiring|interview|"
    r"offer letter|response needed|please respond|payment (?:due|failed)|"
    r"security alert|suspicious (?:activity|sign))\b"
)
_MILD_RE = re.compile(
    r"\b(reminder|invoice|payment|meeting|rsvp|approval|approve|review|request(?:ed)?|"
    r"follow(?:ing)? up|question|schedule|confirm)\b"
)
_AUTOMATED_RE = re.compile(
    r"(no-?reply|do-?not-?reply|notifications?@|newsletter|mailer|marketing|promo|"
    r"offers?@|updates?@|news@|digest)"
)
_PROMO_RE = re.compile(
    r"(\d+% off|\bsale\b|discount|\bdeals?\b|coupon|unsubscribe|newsletter|webinar|"
    r"weekly digest|new arrivals|limited time|free shipping|offer ends)"
)
_TRANSACTIONAL_RE = re.compile(
    r"\b(receipt|order (?:confirmed|shipped|#\w+)|shipped|delivered|verification code|"
    r"one[- ]time (?:password|code)|your otp)\b"
)


def _triage_email(email: Any, vips: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Score one email for urgency from its sender, subject and snippet.

    A transparent keyword heuristic (no model, no network): ``priority`` is
    ``high`` / ``normal`` / ``low``, ``reasons`` says why, and ``action`` is
    the suggested next step — ``read``, ``reply``, ``snooze`` or ``archive``.
    """
    d = _email_to_dict(email)
    sender = (d.get("sender") or "").lower()
    text = f"{d.get('subject', '')} {d.get('snippet', '')}".lower()
    score = 0
    reasons: list[str] = []

    if any(v and v in sender for v in (vips or [])):
        score += 4
        reasons.append("VIP sender")
    if _STRONG_RE.search(text):
        score += 3
        reasons.append("urgent wording")
    elif _MILD_RE.search(text):
        score += 1
        reasons.append("action-related wording")
    asks = "?" in text
    if asks:
        score += 1
        reasons.append("asks a question")
    automated = bool(_AUTOMATED_RE.search(sender))
    if automated:
        score -= 2
        reasons.append("automated sender")
    promo = bool(_PROMO_RE.search(text))
    if promo:
        score -= 2
        reasons.append("promotional")
    transactional = bool(_TRANSACTIONAL_RE.search(text))
    if transactional:
        score -= 1
        reasons.append("transactional")

    if score >= 3:
        priority = "high"
    elif score <= -2:
        priority = "low"
    else:
        priority = "normal"

    if priority == "low" or (transactional and not asks and priority != "high"):
        action = "archive"
    elif priority == "high":
        action = "reply" if asks else "read"
    else:
        action = "reply" if asks else "snooze"
    return {"score": score, "priority": priority, "reasons": reasons, "action": action}


# -- recurrence (#98) -------------------------------------------------------

_REPEAT_RE = re.compile(
    r"\b(every\s+(?:other\s+)?(?:day|weekday|week|month|year|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)|daily|weekly|monthly|yearly|annually|weekdays|"
    r"biweekly|fortnightly)\b",
    re.IGNORECASE,
)
_NO_REPEAT = {"", "none", "no", "once", "false", "never", "no repeat", "one time", "one-off"}
_DAY_CODES = {
    "mon": "MO", "tue": "TU", "wed": "WE", "thu": "TH",
    "fri": "FR", "sat": "SA", "sun": "SU",
}
_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tues?|wed|thu(?:rs?)?|fri|sat|sun)s?\b"
)


def _recurrence_text(entities: dict) -> str:
    """The user's repeat phrase, from an entity or (fallback) the raw query."""
    for key in ("recurrence", "repeat", "repeats", "frequency"):
        val = entities.get(key)
        if val is not None and not isinstance(val, bool):
            val = str(val).strip()
            if val.lower() not in _NO_REPEAT:
                return val
    m = _REPEAT_RE.search(str(entities.get("raw_query") or ""))
    return m.group(1) if m else ""


def _build_rrule(
    text: str,
    *,
    count: Any = None,
    until: Any = None,
    start: Optional[datetime] = None,
    tz: Optional["ZoneInfo"] = None,
) -> Optional[list[str]]:
    """Translate a phrase like 'every weekday' into ``["RRULE:..."]``, or None
    when it can't be understood. ``count`` wins over ``until`` (RFC 5545 forbids both)."""
    t = text.lower().strip()
    interval = 2 if re.search(r"every other|biweekly|fortnight|every (?:two|2) weeks", t) else 1
    byday: list[str] = []

    if re.search(r"\bweekdays?\b", t):
        freq, byday = "WEEKLY", ["MO", "TU", "WE", "TH", "FR"]
    elif _WEEKDAY_RE.search(t):
        freq = "WEEKLY"
        for m in _WEEKDAY_RE.finditer(t):
            code = _DAY_CODES[m.group(1)[:3]]
            if code not in byday:
                byday.append(code)
    elif re.search(r"\b(daily|every day|each day|everyday|day)\b", t):
        freq = "DAILY"
    elif re.search(r"\b(weekly|biweekly|fortnightly|fortnight|weeks?)\b", t):
        freq = "WEEKLY"
    elif re.search(r"\b(monthly|months?)\b", t):
        freq = "MONTHLY"
    elif re.search(r"\b(yearly|annually|annual|years?)\b", t):
        freq = "YEARLY"
    else:
        return None

    parts = [f"FREQ={freq}"]
    if interval > 1:
        parts.append(f"INTERVAL={interval}")
    if byday:
        parts.append("BYDAY=" + ",".join(byday))

    if count not in (None, "", 0, "0"):
        try:
            parts.append(f"COUNT={max(1, min(730, int(count)))}")
        except (TypeError, ValueError):
            pass
    elif until not in (None, "") and tz is not None:
        try:
            end_day = _resolve_date(str(until), tz)
            end_local = datetime.combine(end_day, time(23, 59, 59), tzinfo=tz)
            from datetime import timezone as _tz
            parts.append("UNTIL=" + end_local.astimezone(_tz.utc).strftime("%Y%m%dT%H%M%SZ"))
        except DateTimeParseError:
            logger.warning("Ignoring unreadable repeat end date %r.", until)
    return ["RRULE:" + ";".join(parts)]
