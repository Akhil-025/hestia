# tests/test_hermes.py
"""
Regression tests for modules/hermes/engine.py (HermesEngine).

Run with:  pytest tests/test_hermes.py -v
(or:       python -m pytest tests/test_hermes.py -v
 or:       python3 tests/test_hermes.py)

Note on history: this file previously contained a copy-pasted duplicate of
tests/test_hestia.py's CoreModule/orchestrator tests instead of anything
exercising HermesEngine — there was no dedicated coverage for Hermes at all.
Replaced with real HermesEngine tests below.

Main focus: send_email's confirmation gating. Hestia is voice-driven, and
send_email used to call Google's API the instant "to" and "body" were
non-empty — one misheard recipient or body would have been enough to put a
real message in someone's inbox. It's now two-phase: the first call
validates and previews without sending, and only a call carrying
entities["_confirmed"] = True (sent only by HestiaOrchestrator's pending-
confirmation mechanism, after the user's next reply reads as a clear "yes")
actually sends anything. See modules/hestia/orchestrator.py's "Confirmation
gating" section and tests/test_hestia.py's
test_orchestrator_holds_delete_notes_pending_until_confirmed for the
end-to-end version of the same mechanism.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.hermes.engine import HermesEngine


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeGoogleAgent:
    """Minimal stand-in for core.google_agent.HestiaGoogleAgent covering
    only what HermesEngine actually calls."""

    def __init__(self, authenticated=True):
        self._authenticated = authenticated
        self.sent_emails = []  # list of (to, subject, body)
        self.send_should_succeed = True

    def is_authenticated(self):
        return self._authenticated

    def send_email(self, to, subject, body):
        self.sent_emails.append((to, subject, body))
        return self.send_should_succeed

    def read_emails(self, max_results=5):
        return []

    def format_emails_for_tts(self, emails):
        return "No emails."

    def list_events(self, max_results=5, days_ahead=7):
        return []

    def format_events_for_tts(self, events):
        return "No events."

    def create_event(self, title, start_dt, end_dt=None, location="", description=""):
        return True

    def delete_event(self, event_id):
        return True


_UNSET = object()


def make_hermes(agent=_UNSET, timezone_name="Asia/Kolkata"):
    """Build a HermesEngine. Pass agent=None explicitly to simulate no
    Google agent configured at all (distinct from an unauthenticated one)."""
    if agent is _UNSET:
        agent = FakeGoogleAgent()
    return HermesEngine(google_agent=agent, timezone_name=timezone_name)


# ---------------------------------------------------------------------------
# Readiness / not-connected guard
# ---------------------------------------------------------------------------

def test_handle_returns_not_connected_when_agent_missing():
    hermes = make_hermes(agent=None)
    r = hermes.handle("send_email", {"to": "bob@example.com", "body": "hi"}, {})
    assert "not connected" in r["response"].lower()


def test_handle_returns_not_connected_when_unauthenticated():
    hermes = make_hermes(agent=FakeGoogleAgent(authenticated=False))
    r = hermes.handle("send_email", {"to": "bob@example.com", "body": "hi"}, {})
    assert "not connected" in r["response"].lower()


# ---------------------------------------------------------------------------
# send_email: confirmation gating
# ---------------------------------------------------------------------------

def test_send_email_missing_recipient_asks_who_without_confirmation_flow():
    agent = FakeGoogleAgent()
    hermes = make_hermes(agent)
    r = hermes.handle("send_email", {"body": "hi"}, {})
    assert "who" in r["response"].lower()
    assert r.get("needs_confirmation", False) is False
    assert agent.sent_emails == []


def test_send_email_missing_body_asks_what_without_confirmation_flow():
    agent = FakeGoogleAgent()
    hermes = make_hermes(agent)
    r = hermes.handle("send_email", {"to": "bob@example.com"}, {})
    assert "what" in r["response"].lower()
    assert r.get("needs_confirmation", False) is False
    assert agent.sent_emails == []


def test_send_email_first_call_asks_for_confirmation_and_sends_nothing():
    agent = FakeGoogleAgent()
    hermes = make_hermes(agent)

    r = hermes.handle(
        "send_email",
        {"to": "bob@example.com", "subject": "Hi", "body": "See you at 5pm"},
        {},
    )

    assert r.get("needs_confirmation") is True
    assert "bob@example.com" in r["response"]
    assert "See you at 5pm" in r["response"]
    assert r["confirm_intent"] == "send_email"
    assert r["confirm_entities"] == {
        "to": "bob@example.com", "subject": "Hi", "body": "See you at 5pm",
    }
    assert agent.sent_emails == []  # nothing actually sent yet


def test_send_email_confirmed_call_actually_sends():
    agent = FakeGoogleAgent()
    hermes = make_hermes(agent)

    preview = hermes.handle(
        "send_email", {"to": "bob@example.com", "body": "hi"}, {},
    )
    assert preview.get("needs_confirmation") is True

    r = hermes.handle(
        "send_email",
        {"to": "bob@example.com", "subject": "Message from Hestia", "body": "hi", "_confirmed": True},
        {},
    )

    assert r["confidence"] > 0
    assert "sent" in r["response"].lower()
    assert agent.sent_emails == [("bob@example.com", "Message from Hestia", "hi")]


def test_send_email_confirmed_call_reports_failure_without_crashing():
    agent = FakeGoogleAgent()
    agent.send_should_succeed = False
    hermes = make_hermes(agent)

    r = hermes.handle(
        "send_email",
        {"to": "bob@example.com", "subject": "Hi", "body": "hi", "_confirmed": True},
        {},
    )
    assert r["confidence"] == 0.0
    assert "couldn't send" in r["response"].lower()
    assert agent.sent_emails == [("bob@example.com", "Hi", "hi")]  # attempted once


def test_send_email_confirmed_call_survives_agent_exception():
    class _ExplodingAgent(FakeGoogleAgent):
        def send_email(self, to, subject, body):
            raise RuntimeError("network down")

    hermes = make_hermes(_ExplodingAgent())
    r = hermes.handle(
        "send_email",
        {"to": "bob@example.com", "subject": "Hi", "body": "hi", "_confirmed": True},
        {},
    )
    assert r["confidence"] == 0.0
    assert r["response"]  # graceful message, not a raised exception


def test_send_email_preview_truncates_long_body_for_readability():
    agent = FakeGoogleAgent()
    hermes = make_hermes(agent)
    long_body = "x" * 500
    r = hermes.handle("send_email", {"to": "bob@example.com", "body": long_body}, {})
    assert r.get("needs_confirmation") is True
    assert len(r["response"]) < len(long_body)
    assert "…" in r["response"]
    # The FULL body is preserved in confirm_entities for the actual send,
    # not the truncated preview text.
    assert r["confirm_entities"]["body"] == long_body


# ---------------------------------------------------------------------------
# Baseline sanity: intents / aliases / dispatch contract
# ---------------------------------------------------------------------------

def test_can_handle_covers_all_declared_intents_and_aliases():
    hermes = make_hermes()
    for intent in ("read_email", "send_email", "list_events", "create_event", "delete_events"):
        assert hermes.can_handle(intent)
    for alias in ("check_email", "get_calendar_event", "schedule_event", "clear_calendar"):
        assert hermes.can_handle(alias)
    assert not hermes.can_handle("totally_unknown_intent")


def test_unhandled_intent_returns_graceful_response_not_a_crash():
    hermes = make_hermes()
    r = hermes.handle("totally_unknown_intent", {}, {})
    assert r["confidence"] == 0.0
    assert r["response"]


def test_read_email_summarises_via_agent():
    agent = FakeGoogleAgent()
    hermes = make_hermes(agent)
    r = hermes.handle("read_email", {}, {})
    assert r["response"] == "No emails."


def test_engine_never_raises_when_agent_read_emails_throws():
    class _ExplodingAgent(FakeGoogleAgent):
        def read_emails(self, max_results=5):
            raise RuntimeError("boom")

    hermes = make_hermes(_ExplodingAgent())
    r = hermes.handle("read_email", {}, {})
    assert r["confidence"] == 0.0
    assert r["response"]


if __name__ == "__main__":
    failures = []
    tests = [
        (name, fn) for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures.append(name)
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failures.append(name)
            print(f"ERROR {name}: {e!r}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        sys.exit(1)