import os
import sys
import datetime
import json
from typing import Optional

class HestiaGoogleAgent:
    """Handles Gmail and Google Calendar operations via Google API."""

    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    ]

    def __init__(self, credentials_path: str = "config/google_credentials.json",
                 token_path: str = "data/google_token.json"):
        """
        Args:
            credentials_path: Path to OAuth2 client credentials JSON from Google Cloud Console.
            token_path: Path where the access/refresh token will be stored.
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self._creds = None
        self._gmail = None
        self._calendar = None

    def authenticate(self) -> bool:
        """
        Run OAuth2 flow if needed, cache token to token_path.
        Returns True if authenticated successfully, False otherwise.
        """
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError:
            print("[GoogleAgent] google API packages not installed.", file=sys.stderr)
            return False

        creds = None

        if os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
            except Exception:
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None

            if not creds:
                if not os.path.exists(self.credentials_path):
                    print(
                        f"[GoogleAgent] credentials file not found at '{self.credentials_path}'. "
                        "Download it from Google Cloud Console.",
                        file=sys.stderr
                    )
                    return False
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
            with open(self.token_path, "w") as f:
                f.write(creds.to_json())

        self._creds = creds
        try:
            self._gmail = build("gmail", "v1", credentials=creds)
            self._calendar = build("calendar", "v3", credentials=creds)
            return True
        except Exception as e:
            print(f"[GoogleAgent] Failed to build service: {e}", file=sys.stderr)
            return False

    def is_authenticated(self) -> bool:
        """Return True if Gmail and Calendar services are ready."""
        return self._gmail is not None and self._calendar is not None

    def read_emails(self, max_results: int = 5, unread_only: bool = True) -> list[dict]:
        """
        Fetch recent emails from Gmail inbox.
        Returns list of dicts: {subject, sender, snippet, date, message_id}
        """
        if not self.is_authenticated():
            return []
        try:
            query = "is:unread" if unread_only else ""
            result = self._gmail.users().messages().list(
                userId="me", maxResults=max_results, q=query
            ).execute()
            messages = result.get("messages", [])
            emails = []
            for msg in messages:
                detail = self._gmail.users().messages().get(
                    userId="me", id=msg["id"], format="metadata",
                    metadataHeaders=["Subject", "From", "Date"]
                ).execute()
                headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
                emails.append({
                    "subject": headers.get("Subject", "(no subject)"),
                    "sender": headers.get("From", "Unknown"),
                    "snippet": detail.get("snippet", ""),
                    "date": headers.get("Date", ""),
                    "message_id": msg["id"]
                })
            return emails
        except Exception as e:
            print(f"[GoogleAgent] read_emails error: {e}", file=sys.stderr)
            return []

    def send_email(self, to: str, subject: str, body: str) -> bool:
        """
        Send a plain text email via Gmail.
        Returns True on success.
        """
        if not self.is_authenticated():
            return False
        try:
            import base64
            from email.mime.text import MIMEText
            message = MIMEText(body)
            message["to"] = to
            message["subject"] = subject
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            self._gmail.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
            return True
        except Exception as e:
            print(f"[GoogleAgent] send_email error: {e}", file=sys.stderr)
            return False

    def list_events(self, max_results: int = 5, days_ahead: int = 7) -> list[dict]:
        """
        Fetch upcoming calendar events.
        Returns list of dicts: {title, start, end, location, description}
        """
        if not self.is_authenticated():
            return []
        try:
            now = datetime.datetime.utcnow().isoformat() + "Z"
            until = (
                datetime.datetime.utcnow() + datetime.timedelta(days=days_ahead)
            ).isoformat() + "Z"
            result = self._calendar.events().list(
                calendarId="primary",
                timeMin=now,
                timeMax=until,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime"
            ).execute()
            events = []
            for e in result.get("items", []):
                start = e.get("start", {})
                end = e.get("end", {})
                events.append({
                    "title": e.get("summary", "(no title)"),
                    "start": start.get("dateTime", start.get("date", "")),
                    "end": end.get("dateTime", end.get("date", "")),
                    "location": e.get("location", ""),
                    "description": e.get("description", "")
                })
            return events
        except Exception as e:
            print(f"[GoogleAgent] list_events error: {e}", file=sys.stderr)
            return []

    def create_event(self, title: str, start_dt: datetime.datetime,
                     end_dt: datetime.datetime = None,
                     location: str = "", description: str = "") -> bool:
        """
        Create a new calendar event.
        If end_dt is None, defaults to start_dt + 1 hour.
        Returns True on success.
        """
        if not self.is_authenticated():
            return False
        try:
            if end_dt is None:
                end_dt = start_dt + datetime.timedelta(hours=1)
            event = {
                "summary": title,
                "location": location,
                "description": description,
                "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Kolkata"},
                "end":   {"dateTime": end_dt.isoformat(),   "timeZone": "Asia/Kolkata"},
            }
            self._calendar.events().insert(
                calendarId="primary", body=event
            ).execute()
            return True
        except Exception as e:
            print(f"[GoogleAgent] create_event error: {e}", file=sys.stderr)
            return False

    def format_emails_for_tts(self, emails: list[dict]) -> str:
        """Convert email list to a natural spoken summary."""
        if not emails:
            return "Your inbox is clear — no unread emails."
        lines = [f"You have {len(emails)} unread email{'s' if len(emails) > 1 else ''}."]
        for i, e in enumerate(emails[:3], 1):
            sender = e["sender"].split("<")[0].strip()
            lines.append(f"{i}. From {sender}: {e['subject']}.")
        if len(emails) > 3:
            lines.append(f"And {len(emails) - 3} more.")
        return " ".join(lines)

    def format_events_for_tts(self, events: list[dict]) -> str:
        """Convert event list to a natural spoken summary."""
        if not events:
            return "You have no upcoming events."
        lines = [f"You have {len(events)} upcoming event{'s' if len(events) > 1 else ''}."]
        for i, e in enumerate(events[:3], 1):
            start = e["start"]
            try:
                dt = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
                time_str = dt.strftime("%A at %I:%M %p")
            except Exception:
                time_str = start
            lines.append(f"{i}. {e['title']} on {time_str}.")
        return " ".join(lines)
