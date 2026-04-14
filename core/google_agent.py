"""
agents/google/agent.py

HestiaGoogleAgent: Gmail and Google Calendar operations via Google OAuth2 APIs.

Design notes
------------
- All Google SDK imports are deferred so the module loads even when the
  optional google-api-python-client packages are absent.
- Authentication state is validated before every API call; token refresh
  is handled transparently.
- Every public method returns a typed result and never raises; errors are
  logged and surfaced as empty/False returns with structured log context.
- Timezone is configurable at construction time rather than hard-coded.
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CREDENTIALS_PATH = Path("config/google_credentials.json")
_DEFAULT_TOKEN_PATH = Path("data/google_token.json")
_DEFAULT_TIMEZONE = "UTC"

_GMAIL_SCOPES = frozenset(
    {
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
    }
)
_CALENDAR_SCOPES = frozenset(
    {
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    }
)
SCOPES: list[str] = sorted(_GMAIL_SCOPES | _CALENDAR_SCOPES)

_MAX_EMAIL_RESULTS = 50
_MAX_EVENT_RESULTS = 50
_MAX_DAYS_AHEAD = 365

_TTS_EMAIL_PREVIEW = 3
_TTS_EVENT_PREVIEW = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GoogleAgentError(Exception):
    """Base exception for HestiaGoogleAgent failures."""


class AuthenticationError(GoogleAgentError):
    """Raised when OAuth2 authentication cannot be completed."""


class CredentialsFileNotFoundError(AuthenticationError):
    """Raised when the OAuth2 client-secrets file is missing."""


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Email:
    message_id: str
    subject: str
    sender: str
    snippet: str
    date: str

    @classmethod
    def from_api(cls, msg_id: str, detail: dict[str, Any]) -> "Email":
        headers = {
            h["name"]: h["value"]
            for h in detail.get("payload", {}).get("headers", [])
        }
        return cls(
            message_id=msg_id,
            subject=headers.get("Subject") or "(no subject)",
            sender=headers.get("From") or "Unknown",
            snippet=detail.get("snippet") or "",
            date=headers.get("Date") or "",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "subject": self.subject,
            "sender": self.sender,
            "snippet": self.snippet,
            "date": self.date,
        }


@dataclass(frozen=True)
class CalendarEvent:
    event_id: str
    title: str
    start: str
    end: str
    location: str
    description: str

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "CalendarEvent":
        start = raw.get("start", {})
        end = raw.get("end", {})
        return cls(
            event_id=raw.get("id") or "",
            title=raw.get("summary") or "(no title)",
            start=start.get("dateTime") or start.get("date") or "",
            end=end.get("dateTime") or end.get("date") or "",
            location=raw.get("location") or "",
            description=raw.get("description") or "",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "location": self.location,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class HestiaGoogleAgent:
    """
    Handles Gmail and Google Calendar operations via Google OAuth2 APIs.

    Parameters
    ----------
    credentials_path:
        Path to the OAuth2 client-secrets JSON downloaded from Google Cloud
        Console.
    token_path:
        Path where the granted access/refresh token will be persisted.
    timezone:
        IANA timezone string used for calendar event creation (e.g.
        ``"America/New_York"``).  Defaults to ``"UTC"``.

    Usage
    -----
    ::

        agent = HestiaGoogleAgent()
        agent.authenticate()          # opens browser on first run
        emails = agent.read_emails()
        events = agent.list_events()
    """

    def __init__(
        self,
        credentials_path: str | Path = _DEFAULT_CREDENTIALS_PATH,
        token_path: str | Path = _DEFAULT_TOKEN_PATH,
        timezone: str = _DEFAULT_TIMEZONE,
    ) -> None:
        self._credentials_path = Path(credentials_path).resolve()
        self._token_path = Path(token_path).resolve()
        self._timezone = timezone
        self._creds: Any = None          # google.oauth2.credentials.Credentials
        self._gmail: Any = None          # googleapiclient Resource
        self._calendar: Any = None       # googleapiclient Resource

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> None:
        """
        Run the OAuth2 flow if required, cache the token, and build API
        service clients.

        On first call (no cached token) this opens a browser window.
        Subsequent calls reuse or silently refresh the cached token.

        Raises
        ------
        AuthenticationError
            If authentication cannot be completed for any reason.
        GoogleAgentError
            If the API service clients cannot be built after auth.
        """
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise AuthenticationError(
                "Google API packages are not installed. "
                "Run: pip install google-api-python-client google-auth-oauthlib"
            ) from exc

        creds = self._load_cached_token(Credentials)
        creds = self._ensure_valid(creds, Request, InstalledAppFlow)
        self._persist_token(creds)
        self._creds = creds

        try:
            self._gmail = build("gmail", "v1", credentials=creds)
            self._calendar = build("calendar", "v3", credentials=creds)
        except Exception as exc:
            raise GoogleAgentError(
                f"Failed to build Google API service clients: {exc}"
            ) from exc

        logger.info("Google API authentication successful.")

    def is_authenticated(self) -> bool:
        """Return ``True`` if both Gmail and Calendar clients are ready."""
        return self._gmail is not None and self._calendar is not None

    # ------------------------------------------------------------------
    # Gmail
    # ------------------------------------------------------------------

    def read_emails(
        self,
        max_results: int = 5,
        unread_only: bool = True,
    ) -> list[Email]:
        """
        Fetch recent emails from the Gmail inbox.

        Parameters
        ----------
        max_results:
            Maximum number of messages to return (capped at
            ``_MAX_EMAIL_RESULTS``).
        unread_only:
            When ``True`` the query is filtered to unread messages only.

        Returns
        -------
        list[Email]
            Empty list on error or when not authenticated.
        """
        self._require_auth()
        max_results = _clamp(max_results, 1, _MAX_EMAIL_RESULTS)

        try:
            query = "is:unread in:inbox" if unread_only else "in:inbox"
            result: dict = (
                self._gmail.users()
                .messages()
                .list(userId="me", maxResults=max_results, q=query)
                .execute()
            )
            messages: list[dict] = result.get("messages") or []

            emails: list[Email] = []
            for msg in messages:
                detail = (
                    self._gmail.users()
                    .messages()
                    .get(
                        userId="me",
                        id=msg["id"],
                        format="metadata",
                        metadataHeaders=["Subject", "From", "Date"],
                    )
                    .execute()
                )
                emails.append(Email.from_api(msg["id"], detail))

            logger.info("read_emails: fetched %d message(s).", len(emails))
            return emails

        except Exception:
            logger.exception("read_emails failed.")
            return []

    def send_email(self, to: str, subject: str, body: str) -> bool:
        """
        Send a plain-text email via Gmail.

        Parameters
        ----------
        to:
            Recipient address.
        subject:
            Email subject line.
        body:
            Plain-text message body.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.

        Raises
        ------
        ValueError
            If *to*, *subject*, or *body* are empty.
        """
        self._require_auth()
        _require_non_empty(to=to, subject=subject, body=body)

        try:
            mime = MIMEText(body, _charset="utf-8")
            mime["to"] = to
            mime["subject"] = subject
            raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()

            self._gmail.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()

            logger.info("send_email: message sent to %r.", to)
            return True

        except Exception:
            logger.exception("send_email to %r failed.", to)
            return False

    # ------------------------------------------------------------------
    # Google Calendar
    # ------------------------------------------------------------------

    def list_events(
        self,
        max_results: int = 5,
        days_ahead: int = 7,
    ) -> list[CalendarEvent]:
        """
        Fetch upcoming calendar events.

        Parameters
        ----------
        max_results:
            Maximum number of events (capped at ``_MAX_EVENT_RESULTS``).
        days_ahead:
            How many days into the future to look (capped at
            ``_MAX_DAYS_AHEAD``).

        Returns
        -------
        list[CalendarEvent]
            Empty list on error or when not authenticated.
        """
        self._require_auth()
        max_results = _clamp(max_results, 1, _MAX_EVENT_RESULTS)
        days_ahead = _clamp(days_ahead, 1, _MAX_DAYS_AHEAD)

        try:
            now_utc = datetime.now(timezone.utc)
            time_min = _to_rfc3339(now_utc)
            time_max = _to_rfc3339(now_utc + timedelta(days=days_ahead))

            result: dict = (
                self._calendar.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = [CalendarEvent.from_api(e) for e in result.get("items") or []]
            logger.info("list_events: fetched %d event(s).", len(events))
            return events

        except Exception:
            logger.exception("list_events failed.")
            return []

    def create_event(
        self,
        title: str,
        start_dt: datetime,
        end_dt: Optional[datetime] = None,
        location: str = "",
        description: str = "",
    ) -> bool:
        """
        Create a new calendar event.

        Parameters
        ----------
        title:
            Event title / summary.
        start_dt:
            Start datetime (timezone-aware recommended).
        end_dt:
            End datetime; defaults to ``start_dt + 1 hour``.
        location:
            Optional venue string.
        description:
            Optional event description.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.

        Raises
        ------
        ValueError
            If *title* is empty or *start_dt* is not a datetime.
        """
        self._require_auth()

        if not title or not title.strip():
            raise ValueError("title must be a non-empty string.")
        if not isinstance(start_dt, datetime):
            raise ValueError("start_dt must be a datetime instance.")

        effective_end = end_dt if end_dt is not None else start_dt + timedelta(hours=1)

        if effective_end <= start_dt:
            raise ValueError("end_dt must be after start_dt.")

        body: dict[str, Any] = {
            "summary": title.strip(),
            "location": location,
            "description": description,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": self._timezone},
            "end": {"dateTime": effective_end.isoformat(), "timeZone": self._timezone},
        }

        try:
            self._calendar.events().insert(
                calendarId="primary", body=body
            ).execute()
            logger.info(
                "create_event: %r created at %s (%s).",
                title,
                start_dt.isoformat(),
                self._timezone,
            )
            return True

        except Exception:
            logger.exception("create_event %r failed.", title)
            return False

    def delete_event(self, event_id: str) -> bool:
        """
        Delete a calendar event by its API ID.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.
        """
        self._require_auth()
        if not event_id or not event_id.strip():
            raise ValueError("event_id must be a non-empty string.")

        try:
            self._calendar.events().delete(
                calendarId="primary", eventId=event_id
            ).execute()
            logger.info("delete_event: event %r deleted.", event_id)
            return True
        except Exception:
            logger.exception("delete_event %r failed.", event_id)
            return False

    # ------------------------------------------------------------------
    # TTS formatting
    # ------------------------------------------------------------------

    def format_emails_for_tts(self, emails: list[Email]) -> str:
        """Return a natural spoken summary of *emails*."""
        if not emails:
            return "Your inbox is clear — no unread emails."

        count = len(emails)
        noun = "email" if count == 1 else "emails"
        lines = [f"You have {count} unread {noun}."]

        for i, email in enumerate(emails[:_TTS_EMAIL_PREVIEW], start=1):
            sender = email.sender.split("<")[0].strip()
            lines.append(f"{i}. From {sender}: {email.subject}.")

        remainder = count - _TTS_EMAIL_PREVIEW
        if remainder > 0:
            lines.append(f"And {remainder} more.")

        return " ".join(lines)

    def format_events_for_tts(self, events: list[CalendarEvent]) -> str:
        """Return a natural spoken summary of *events*."""
        if not events:
            return "You have no upcoming events."

        count = len(events)
        noun = "event" if count == 1 else "events"
        lines = [f"You have {count} upcoming {noun}."]

        for i, evt in enumerate(events[:_TTS_EVENT_PREVIEW], start=1):
            time_str = _format_event_time(evt.start)
            lines.append(f"{i}. {evt.title} on {time_str}.")

        return " ".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_auth(self) -> None:
        """Raise if the API clients are not ready."""
        if not self.is_authenticated():
            raise AuthenticationError(
                "Not authenticated. Call authenticate() first."
            )

    def _load_cached_token(self, Credentials: type) -> Any:
        """Return cached Credentials or None if absent / unreadable."""
        if not self._token_path.exists():
            return None
        try:
            return Credentials.from_authorized_user_file(
                str(self._token_path), SCOPES
            )
        except Exception:
            logger.warning(
                "Cached token at %s could not be loaded; re-authorising.",
                self._token_path,
            )
            return None

    def _ensure_valid(
        self,
        creds: Any,
        Request: type,
        InstalledAppFlow: type,
    ) -> Any:
        """Refresh or re-run the OAuth flow until creds are valid."""
        if creds and creds.valid:
            return creds

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.debug("Token refreshed successfully.")
                return creds
            except Exception:
                logger.warning("Token refresh failed; re-authorising.")

        # Full OAuth flow
        if not self._credentials_path.exists():
            raise CredentialsFileNotFoundError(
                f"OAuth2 credentials file not found at {self._credentials_path!r}. "
                "Download it from https://console.cloud.google.com/."
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(self._credentials_path), SCOPES
        )
        creds = flow.run_local_server(port=0)
        logger.info("OAuth2 authorisation completed.")
        return creds

    def _persist_token(self, creds: Any) -> None:
        """Atomically write the token JSON to disk with secure permissions."""
        self._token_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fd, tmp = tempfile.mkstemp(
                dir=self._token_path.parent,
                suffix=".tmp"
            )

            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(creds.to_json())

                # 🔒 Restrict permissions BEFORE move
                os.chmod(tmp, 0o600)

                # Atomic replace
                os.replace(tmp, self._token_path)

                # 🔒 Ensure final file is also restricted (important on some FS)
                os.chmod(self._token_path, 0o600)

            except Exception:
                os.unlink(tmp)
                raise

        except Exception:
            logger.exception("Failed to persist token to %s.", self._token_path)
            raise

# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _to_rfc3339(dt: datetime) -> str:
    """Return an RFC 3339 string suitable for the Google Calendar API."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_event_time(iso_start: str) -> str:
    """Convert an ISO-8601 start string to a spoken time label."""
    if not iso_start:
        return "an unspecified time"
    try:
        dt = datetime.fromisoformat(iso_start.replace("Z", "+00:00"))
        return dt.strftime("%A at %I:%M %p").lstrip("0")
    except ValueError:
        return iso_start


def _require_non_empty(**kwargs: str) -> None:
    """Raise ValueError if any keyword argument is empty."""
    for name, value in kwargs.items():
        if not value or not value.strip():
            raise ValueError(f"{name!r} must be a non-empty string.")