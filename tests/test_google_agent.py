"""
tests/test_google_agent.py

Covers core/google_agent.py's HestiaGoogleAgent, its Email/CalendarEvent
dataclasses, and its module-level pure helpers.

google-api-python-client / google-auth-oauthlib are imported lazily inside
authenticate(), so most tests here exercise the Gmail/Calendar methods by
injecting mock `_gmail`/`_calendar` clients directly (bypassing OAuth
entirely) — this covers the actual business logic without needing any
real Google credentials. A dedicated TestAuthenticate class separately
fakes the OAuth library imports to cover authenticate() itself.
"""
import sys
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

import core.google_agent as ga


# ---------------------------------------------------------------------------
# Email dataclass
# ---------------------------------------------------------------------------

class TestEmail:
    def test_from_api_extracts_headers(self):
        detail = {
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Re: Dinner"},
                    {"name": "From", "value": "Alice <alice@example.com>"},
                    {"name": "Date", "value": "Mon, 1 Sep 2026 10:00:00 +0000"},
                ]
            },
            "snippet": "Sounds great, see you then",
        }
        email = ga.Email.from_api("msg1", detail)
        assert email.message_id == "msg1"
        assert email.subject == "Re: Dinner"
        assert email.sender == "Alice <alice@example.com>"
        assert email.snippet == "Sounds great, see you then"

    def test_from_api_defaults_missing_fields(self):
        email = ga.Email.from_api("msg2", {})
        assert email.subject == "(no subject)"
        assert email.sender == "Unknown"
        assert email.snippet == ""
        assert email.date == ""

    def test_to_dict_roundtrip(self):
        email = ga.Email("id1", "Subj", "From", "snip", "date")
        assert email.to_dict() == {
            "message_id": "id1",
            "subject": "Subj",
            "sender": "From",
            "snippet": "snip",
            "date": "date",
        }


# ---------------------------------------------------------------------------
# CalendarEvent dataclass
# ---------------------------------------------------------------------------

class TestCalendarEvent:
    def test_from_api_prefers_datetime_over_date(self):
        raw = {
            "id": "evt1",
            "summary": "Standup",
            "start": {"dateTime": "2026-09-15T09:00:00Z"},
            "end": {"dateTime": "2026-09-15T09:30:00Z"},
            "location": "Room 4",
            "description": "Daily sync",
        }
        evt = ga.CalendarEvent.from_api(raw)
        assert evt.start == "2026-09-15T09:00:00Z"
        assert evt.title == "Standup"

    def test_from_api_falls_back_to_all_day_date(self):
        raw = {"id": "evt2", "start": {"date": "2026-09-15"}, "end": {"date": "2026-09-16"}}
        evt = ga.CalendarEvent.from_api(raw)
        assert evt.start == "2026-09-15"
        assert evt.title == "(no title)"

    def test_to_dict_roundtrip(self):
        evt = ga.CalendarEvent("id", "title", "start", "end", "loc", "desc")
        assert evt.to_dict()["event_id"] == "id"


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

class TestPureHelpers:
    def test_clamp_within_range(self):
        assert ga._clamp(5, 1, 10) == 5

    def test_clamp_below_min(self):
        assert ga._clamp(-5, 1, 10) == 1

    def test_clamp_above_max(self):
        assert ga._clamp(500, 1, 10) == 10

    def test_to_rfc3339_format(self):
        dt = datetime(2026, 9, 15, 12, 30, 0, tzinfo=timezone.utc)
        assert ga._to_rfc3339(dt) == "2026-09-15T12:30:00Z"

    def test_format_event_time_with_valid_iso(self):
        result = ga._format_event_time("2026-09-15T09:00:00Z")
        assert "9:00 AM" in result

    def test_format_event_time_empty_string(self):
        assert ga._format_event_time("") == "an unspecified time"

    def test_format_event_time_unparseable_returns_raw(self):
        assert ga._format_event_time("not-a-date") == "not-a-date"

    def test_require_non_empty_passes_for_valid_strings(self):
        ga._require_non_empty(to="a@b.com", subject="hi")  # must not raise

    def test_require_non_empty_raises_on_blank(self):
        with pytest.raises(ValueError, match="subject"):
            ga._require_non_empty(to="a@b.com", subject="   ")

    def test_require_non_empty_raises_on_missing(self):
        with pytest.raises(ValueError, match="body"):
            ga._require_non_empty(body="")


# ---------------------------------------------------------------------------
# is_authenticated / _require_auth
# ---------------------------------------------------------------------------

class TestAuthState:
    def test_not_authenticated_before_authenticate(self):
        agent = ga.HestiaGoogleAgent()
        assert agent.is_authenticated() is False

    def test_authenticated_when_both_clients_set(self):
        agent = ga.HestiaGoogleAgent()
        agent._gmail = MagicMock()
        agent._calendar = MagicMock()
        assert agent.is_authenticated() is True

    def test_partial_clients_not_authenticated(self):
        agent = ga.HestiaGoogleAgent()
        agent._gmail = MagicMock()
        assert agent.is_authenticated() is False

    def test_require_auth_raises_when_not_authenticated(self):
        agent = ga.HestiaGoogleAgent()
        with pytest.raises(ga.AuthenticationError):
            agent.read_emails()


# ---------------------------------------------------------------------------
# Gmail methods (client mocked directly, no OAuth involved)
# ---------------------------------------------------------------------------

def _authed_agent():
    agent = ga.HestiaGoogleAgent()
    agent._gmail = MagicMock()
    agent._calendar = MagicMock()
    return agent


class TestReadEmails:
    def test_returns_emails_for_each_message(self):
        agent = _authed_agent()
        agent._gmail.users().messages().list().execute.return_value = {
            "messages": [{"id": "m1"}, {"id": "m2"}]
        }
        agent._gmail.users().messages().get().execute.return_value = {
            "payload": {"headers": [{"name": "Subject", "value": "Hi"}]},
            "snippet": "snip",
        }
        emails = agent.read_emails(max_results=2)
        assert len(emails) == 2
        assert all(isinstance(e, ga.Email) for e in emails)

    def test_returns_empty_list_on_api_exception(self):
        agent = _authed_agent()
        agent._gmail.users.side_effect = Exception("api down")
        assert agent.read_emails() == []

    def test_unread_only_uses_unread_query(self):
        agent = _authed_agent()
        list_mock = agent._gmail.users().messages().list
        list_mock.return_value.execute.return_value = {"messages": []}
        agent.read_emails(unread_only=True)
        _, kwargs = list_mock.call_args
        assert kwargs["q"] == "is:unread in:inbox"

    def test_all_mail_query_when_unread_only_false(self):
        agent = _authed_agent()
        list_mock = agent._gmail.users().messages().list
        list_mock.return_value.execute.return_value = {"messages": []}
        agent.read_emails(unread_only=False)
        _, kwargs = list_mock.call_args
        assert kwargs["q"] == "in:inbox"

    def test_max_results_clamped(self):
        agent = _authed_agent()
        list_mock = agent._gmail.users().messages().list
        list_mock.return_value.execute.return_value = {"messages": []}
        agent.read_emails(max_results=999)
        _, kwargs = list_mock.call_args
        assert kwargs["maxResults"] == ga._MAX_EMAIL_RESULTS


class TestSendEmail:
    def test_success_returns_true(self):
        agent = _authed_agent()
        result = agent.send_email("to@example.com", "Subject", "Body text")
        assert result is True
        agent._gmail.users().messages().send.assert_called()

    def test_raises_on_empty_recipient(self):
        agent = _authed_agent()
        with pytest.raises(ValueError):
            agent.send_email("", "Subject", "Body")

    def test_returns_false_on_api_exception(self):
        agent = _authed_agent()
        agent._gmail.users.side_effect = Exception("quota exceeded")
        assert agent.send_email("to@example.com", "s", "b") is False


# ---------------------------------------------------------------------------
# Calendar methods
# ---------------------------------------------------------------------------

class TestListEvents:
    def test_returns_parsed_events(self):
        agent = _authed_agent()
        agent._calendar.events().list().execute.return_value = {
            "items": [{"id": "e1", "summary": "Meeting", "start": {"dateTime": "2026-09-15T10:00:00Z"}, "end": {}}]
        }
        events = agent.list_events()
        assert len(events) == 1
        assert events[0].title == "Meeting"

    def test_returns_empty_list_on_exception(self):
        agent = _authed_agent()
        agent._calendar.events.side_effect = Exception("boom")
        assert agent.list_events() == []


class TestCreateEvent:
    def test_success(self):
        agent = _authed_agent()
        start = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
        result = agent.create_event("Dentist", start)
        assert result is True
        agent._calendar.events().insert.assert_called()

    def test_default_end_is_one_hour_after_start(self):
        agent = _authed_agent()
        start = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
        agent.create_event("Dentist", start)
        _, kwargs = agent._calendar.events().insert.call_args
        end_iso = kwargs["body"]["end"]["dateTime"]
        assert end_iso == (start + timedelta(hours=1)).isoformat()

    def test_raises_on_empty_title(self):
        agent = _authed_agent()
        with pytest.raises(ValueError, match="title"):
            agent.create_event("   ", datetime.now(timezone.utc))

    def test_raises_when_start_dt_not_datetime(self):
        agent = _authed_agent()
        with pytest.raises(ValueError, match="start_dt"):
            agent.create_event("Dentist", "not-a-datetime")

    def test_raises_when_end_before_start(self):
        agent = _authed_agent()
        start = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
        end = start - timedelta(hours=1)
        with pytest.raises(ValueError, match="end_dt must be after start_dt"):
            agent.create_event("Dentist", start, end_dt=end)

    def test_returns_false_on_api_exception(self):
        agent = _authed_agent()
        agent._calendar.events.side_effect = Exception("boom")
        result = agent.create_event("Dentist", datetime.now(timezone.utc))
        assert result is False


class TestDeleteEvent:
    def test_success(self):
        agent = _authed_agent()
        assert agent.delete_event("evt1") is True

    def test_raises_on_empty_id(self):
        agent = _authed_agent()
        with pytest.raises(ValueError):
            agent.delete_event("")

    def test_returns_false_on_exception(self):
        agent = _authed_agent()
        agent._calendar.events.side_effect = Exception("boom")
        assert agent.delete_event("evt1") is False


# ---------------------------------------------------------------------------
# TTS formatting
# ---------------------------------------------------------------------------

class TestFormatEmailsForTts:
    def test_empty_inbox_message(self):
        agent = ga.HestiaGoogleAgent()
        assert agent.format_emails_for_tts([]) == "Your inbox is clear — no unread emails."

    def test_singular_noun_for_one_email(self):
        agent = ga.HestiaGoogleAgent()
        email = ga.Email("1", "Hi", "Bob <b@x.com>", "", "")
        result = agent.format_emails_for_tts([email])
        assert "1 unread email." in result
        assert "From Bob:" in result

    def test_plural_and_remainder_count(self):
        agent = ga.HestiaGoogleAgent()
        emails = [ga.Email(str(i), f"Subj{i}", f"S{i}", "", "") for i in range(5)]
        result = agent.format_emails_for_tts(emails)
        assert "5 unread emails." in result
        assert "And 2 more." in result


class TestFormatEventsForTts:
    def test_no_events_message(self):
        agent = ga.HestiaGoogleAgent()
        assert agent.format_events_for_tts([]) == "You have no upcoming events."

    def test_singular_event(self):
        agent = ga.HestiaGoogleAgent()
        evt = ga.CalendarEvent("1", "Standup", "2026-09-15T09:00:00Z", "", "", "")
        result = agent.format_events_for_tts([evt])
        assert "1 upcoming event." in result
        assert "Standup" in result


# ---------------------------------------------------------------------------
# authenticate() — OAuth libraries faked
# ---------------------------------------------------------------------------

def _install_fake_google_oauth_libs(monkeypatch, credentials_cls=None, build_fn=None):
    fake_request_mod = types.ModuleType("google.auth.transport.requests")
    fake_request_mod.Request = MagicMock(name="Request")

    fake_creds_mod = types.ModuleType("google.oauth2.credentials")
    fake_creds_mod.Credentials = credentials_cls or MagicMock(name="Credentials")

    fake_flow_mod = types.ModuleType("google_auth_oauthlib.flow")
    fake_flow_mod.InstalledAppFlow = MagicMock(name="InstalledAppFlow")

    fake_discovery_mod = types.ModuleType("googleapiclient.discovery")
    fake_discovery_mod.build = build_fn or MagicMock(name="build")

    for name, mod in {
        "google": types.ModuleType("google"),
        "google.auth": types.ModuleType("google.auth"),
        "google.auth.transport": types.ModuleType("google.auth.transport"),
        "google.auth.transport.requests": fake_request_mod,
        "google.oauth2": types.ModuleType("google.oauth2"),
        "google.oauth2.credentials": fake_creds_mod,
        "google_auth_oauthlib": types.ModuleType("google_auth_oauthlib"),
        "google_auth_oauthlib.flow": fake_flow_mod,
        "googleapiclient": types.ModuleType("googleapiclient"),
        "googleapiclient.discovery": fake_discovery_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return fake_request_mod, fake_creds_mod, fake_flow_mod, fake_discovery_mod


class TestAuthenticate:
    def test_raises_authentication_error_when_packages_missing(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _raise(name, *a, **kw):
            if name.startswith("google"):
                raise ImportError("not installed")
            return real_import(name, *a, **kw)

        for mod in list(sys.modules):
            if mod.startswith("google"):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setattr(builtins, "__import__", _raise)

        agent = ga.HestiaGoogleAgent()
        with pytest.raises(ga.AuthenticationError, match="not installed"):
            agent.authenticate()

    def test_uses_cached_valid_token_without_full_oauth_flow(self, monkeypatch, tmp_path):
        cached_creds = MagicMock()
        cached_creds.valid = True
        cached_creds.to_json.return_value = "{}"
        creds_cls = MagicMock()
        creds_cls.from_authorized_user_file.return_value = cached_creds

        built_service = MagicMock()
        _, _, flow_mod, discovery_mod = _install_fake_google_oauth_libs(
            monkeypatch, credentials_cls=creds_cls, build_fn=MagicMock(return_value=built_service)
        )

        token_path = tmp_path / "token.json"
        token_path.write_text("{}")
        agent = ga.HestiaGoogleAgent(
            credentials_path=tmp_path / "creds.json", token_path=token_path
        )
        agent.authenticate()

        flow_mod.InstalledAppFlow.from_client_secrets_file.assert_not_called()
        assert agent.is_authenticated() is True

    def test_refreshes_expired_token_with_refresh_token(self, monkeypatch, tmp_path):
        expired_creds = MagicMock()
        expired_creds.valid = False
        expired_creds.expired = True
        expired_creds.refresh_token = "rt-123"
        expired_creds.to_json.return_value = "{}"
        creds_cls = MagicMock()
        creds_cls.from_authorized_user_file.return_value = expired_creds

        _install_fake_google_oauth_libs(monkeypatch, credentials_cls=creds_cls)

        token_path = tmp_path / "token.json"
        token_path.write_text("{}")
        agent = ga.HestiaGoogleAgent(
            credentials_path=tmp_path / "creds.json", token_path=token_path
        )
        agent.authenticate()

        expired_creds.refresh.assert_called_once()

    def test_raises_credentials_file_not_found_when_full_flow_needed(self, monkeypatch, tmp_path):
        creds_cls = MagicMock()
        creds_cls.from_authorized_user_file.side_effect = Exception("no cached token")

        _install_fake_google_oauth_libs(monkeypatch, credentials_cls=creds_cls)

        agent = ga.HestiaGoogleAgent(
            credentials_path=tmp_path / "does_not_exist.json",
            token_path=tmp_path / "token.json",
        )
        with pytest.raises(ga.CredentialsFileNotFoundError):
            agent.authenticate()

    def test_persists_token_with_restricted_permissions(self, monkeypatch, tmp_path):
        creds = MagicMock()
        creds.valid = True
        creds.to_json.return_value = '{"token": "abc"}'
        creds_cls = MagicMock()
        creds_cls.from_authorized_user_file.return_value = creds

        _install_fake_google_oauth_libs(monkeypatch, credentials_cls=creds_cls)

        token_dir = tmp_path / "sub"
        token_dir.mkdir()
        token_path = token_dir / "token.json"
        token_path.write_text("{}")  # pre-existing cached token -> skips full OAuth flow
        agent = ga.HestiaGoogleAgent(
            credentials_path=tmp_path / "creds.json", token_path=token_path
        )
        agent.authenticate()

        assert token_path.exists()
        assert token_path.read_text() == '{"token": "abc"}'
        mode = token_path.stat().st_mode & 0o777
        assert mode == 0o600
