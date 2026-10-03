# tests/test_hermes_backlog.py
"""
Tests for the Hermes backlog pass (#92-#100): email digest/triage, drafting,
search, schedule gaps, meeting slots, conflict detection, recurring events,
inbox-zero plan, and recipient validation before send.

All Google access goes through a fake agent; nothing touches the network.
"""
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.google_agent import CalendarEvent, Email  # noqa: E402
from modules.hecate.intent_registry import module_for_intent  # noqa: E402
from modules.hermes.engine import (  # noqa: E402
    HermesEngine,
    _build_rrule,
    _parse_duration_minutes,
    _triage_email,
)

TZ = ZoneInfo("Asia/Kolkata")


def _email(subject, sender="Someone <someone@example.com>", snippet="", date="Tue, 14 Oct 2026 09:00:00 +0530"):
    return Email(message_id=subject, subject=subject, sender=sender, snippet=snippet, date=date)


def _event(title, start, end=None, location=""):
    end = end or start + timedelta(hours=1)
    return CalendarEvent(
        event_id=title, title=title, start=start.isoformat(), end=end.isoformat(),
        location=location, description="",
    )


def _tomorrow_at(h, m=0):
    d = datetime.now(TZ).date() + timedelta(days=1)
    return datetime(d.year, d.month, d.day, h, m, tzinfo=TZ)


class Agent:
    def __init__(self):
        self.unread = []
        self.found = []
        self.events = []
        self.busy = {"primary": []}
        self.searched = []
        self.created = []
        self.sent = []

    def is_authenticated(self):
        return True

    def read_emails(self, max_results=5):
        return self.unread[:max_results]

    def search_emails(self, query, max_results=10):
        self.searched.append(query)
        return self.found[:max_results]

    def format_emails_for_tts(self, emails):
        return "ok"

    def list_events(self, max_results=5, days_ahead=7):
        return self.events

    def list_events_between(self, start, end, max_results=50):
        return self.events

    def format_events_for_tts(self, events):
        return "ok"

    def free_busy(self, start, end, emails=None):
        return {k: v for k, v in self.busy.items()}

    def create_event(self, title, start_dt, end_dt=None, location="", description="", recurrence=None):
        self.created.append({"title": title, "start": start_dt, "end": end_dt, "recurrence": recurrence})
        return True

    def delete_event(self, event_id):
        return True

    def send_email(self, to, subject, body):
        self.sent.append((to, subject, body))
        return True


def make(agent=None, **kw):
    return HermesEngine(agent or Agent(), timezone_name="Asia/Kolkata", **kw)


# ---------------------------------------------------------------- registry

def test_new_intents_are_registered_to_hermes():
    for intent in ("email_digest", "draft_email", "search_email",
                   "check_schedule_gaps", "find_meeting_slot", "inbox_zero"):
        assert module_for_intent(intent) == "hermes"
        assert make().can_handle(intent)


# ---------------------------------------------------------------- #92 triage

def test_triage_ranks_urgent_over_newsletter():
    urgent = _triage_email(_email("Action required: invoice overdue", snippet="Please respond today?"))
    promo = _triage_email(_email("50% off sale", sender="Shop <no-reply@shop.com>", snippet="unsubscribe"))
    assert urgent["priority"] == "high"
    assert promo["priority"] == "low" and promo["action"] == "archive"
    assert urgent["score"] > promo["score"]


def test_vip_sender_is_high_priority():
    r = _triage_email(_email("lunch", sender="Boss <boss@corp.com>"), vips=["boss@corp.com"])
    assert r["priority"] == "high"


def test_digest_counts_and_leads_with_urgent():
    a = Agent()
    a.unread = [
        _email("Weekly digest", sender="News <newsletter@x.com>", snippet="unsubscribe"),
        _email("URGENT: payment failed", sender="Bank <alerts@bank.com>"),
        _email("hello"),
    ]
    r = make(a).handle("email_digest", {}, {})
    assert "3 unread" in r["response"]
    assert "Most urgent" in r["response"] and "payment failed" in r["response"]
    assert r["data"]["counts"]["high"] == 1 and r["data"]["counts"]["low"] == 1


def test_digest_empty_inbox():
    r = make().handle("email_digest", {}, {})
    assert "clear" in r["response"].lower()


def test_digest_survives_agent_failure():
    class Boom(Agent):
        def read_emails(self, max_results=5):
            raise RuntimeError("x")
    r = make(Boom()).handle("email_digest", {}, {})
    assert r["confidence"] == 0.0


# ---------------------------------------------------------------- #99 inbox zero

def test_inbox_zero_plans_actions_and_changes_nothing():
    a = Agent()
    a.unread = [
        _email("Quick question about Friday?"),
        _email("Sale", sender="Shop <no-reply@shop.com>", snippet="50% off"),
    ]
    r = make(a).handle("inbox_zero", {}, {})
    assert r["data"]["by_action"].get("archive") == 1
    assert "haven't changed anything" in r["response"]
    assert a.sent == [] and a.created == []


# ---------------------------------------------------------------- #93 drafting

def test_draft_with_address_goes_through_send_confirmation():
    a = Agent()
    r = make(a).handle(
        "draft_email",
        {"to": "priya@example.com", "instruction": "I can't make it, suggest Thursday instead"}, {},
    )
    assert r["needs_confirmation"] is True
    assert r["confirm_intent"] == "send_email"
    assert "Thursday" in r["confirm_entities"]["body"]
    assert a.sent == []  # nothing goes out on the draft turn


def test_draft_without_recipient_asks_who_and_keeps_the_draft():
    r = make().handle("draft_email", {"instruction": "thank them for the report"}, {})
    assert r["data"]["missing_slot"] == "to"
    kept = r["data"]["slot_entities"]
    assert kept["_drafted"] is True and kept["body"]
    # Slot-fill re-dispatch with an address reuses the SAME draft.
    kept = dict(kept, to="sam@example.com")
    r2 = make().handle("draft_email", kept, {})
    assert r2["needs_confirmation"] is True
    assert r2["confirm_entities"]["body"] == kept["body"]


def test_draft_uses_llm_when_available_and_falls_back_when_it_breaks():
    class LLM:
        def generate(self, prompt, fmt=None):
            return '{"subject": "Hello", "body": "Custom body."}'
    r = make(llm=LLM()).handle("draft_email", {"to": "a@b.co", "instruction": "say hi"}, {})
    assert r["confirm_entities"]["body"] == "Custom body."

    class Broken:
        def generate(self, prompt, fmt=None):
            raise RuntimeError("down")
    r = make(llm=Broken()).handle("draft_email", {"to": "a@b.co", "instruction": "say hi"}, {})
    assert r["needs_confirmation"] is True and r["confirm_entities"]["body"]


def test_draft_without_instruction_asks():
    r = make().handle("draft_email", {}, {})
    assert r["data"]["missing_slot"] == "instruction"


# ---------------------------------------------------------------- #100 recipients

def test_send_email_to_a_bare_name_asks_for_the_address():
    a = Agent()
    r = make(a).handle("send_email", {"to": "John", "body": "hi"}, {})
    assert r["data"]["missing_slot"] == "to"
    assert "John" in r["response"]
    assert a.sent == []


def test_send_email_resolves_a_configured_contact_and_still_confirms():
    a = Agent()
    h = make(a, contacts={"John": "john@example.com"})
    r = h.handle("send_email", {"to": "john", "body": "hi"}, {})
    assert r["needs_confirmation"] is True
    assert r["confirm_entities"]["to"] == "john@example.com"
    assert "john@example.com" in r["response"]
    h.handle("send_email", dict(r["confirm_entities"], _confirmed=True), {})
    assert a.sent[0][0] == "john@example.com"


# ---------------------------------------------------------------- #94 search

def test_search_builds_a_gmail_query():
    a = Agent()
    a.found = [_email("Invoice #4", sender="Raj <raj@x.com>")]
    r = make(a).handle("search_email", {"sender": "Raj", "subject": "invoice", "date": "2026-03-01"}, {})
    q = a.searched[0]
    assert "from:Raj" in q and "subject:invoice" in q
    assert "after:2026/03/01" in q and "before:2026/03/02" in q
    assert "Invoice #4" in r["response"]


def test_search_uses_contact_address_for_sender():
    a = Agent()
    make(a, contacts={"raj": "raj@x.com"}).handle("search_email", {"sender": "Raj"}, {})
    assert "from:raj@x.com" in a.searched[0]


def test_search_with_nothing_to_search_asks():
    r = make().handle("search_email", {}, {})
    assert r["data"]["missing_slot"] == "query"


def test_search_no_results_and_bad_date():
    r = make().handle("search_email", {"sender": "nobody"}, {})
    assert "didn't find" in r["response"]
    r = make().handle("search_email", {"sender": "x", "date": "sometime"}, {})
    assert r["data"]["needs_clarification"] is True


def test_search_unsupported_agent_is_graceful():
    class Old:
        def is_authenticated(self):
            return True
    r = HermesEngine(Old(), "Asia/Kolkata").handle("search_email", {"sender": "x"}, {})
    assert r["confidence"] == 0.0


# ---------------------------------------------------------------- #95 gaps

def test_gaps_flags_back_to_back_overlap_and_travel():
    a = Agent()
    a.events = [
        _event("A", _tomorrow_at(9), _tomorrow_at(10), location="Office"),
        _event("B", _tomorrow_at(10), _tomorrow_at(11), location="Office"),   # back-to-back
        _event("C", _tomorrow_at(11, 15), _tomorrow_at(12), location="Cafe"),  # 15 min + travel
        _event("D", _tomorrow_at(11, 45), _tomorrow_at(12, 30)),               # overlap
    ]
    r = make(a).handle("check_schedule_gaps", {"date": "tomorrow"}, {})
    kinds = [f["kind"] for f in r["data"]["flags"]]
    assert kinds == ["back-to-back", "travel", "overlap"]
    assert "flat" in r["response"]  # the estimate caveat is stated


def test_gaps_ok_when_there_is_room():
    a = Agent()
    a.events = [_event("A", _tomorrow_at(9), _tomorrow_at(10)), _event("B", _tomorrow_at(11), _tomorrow_at(12))]
    r = make(a).handle("check_schedule_gaps", {"date": "tomorrow"}, {})
    assert r["data"]["flags"] == [] and "breathing room" in r["response"]


def test_gaps_ignores_all_day_events():
    a = Agent()
    a.events = [CalendarEvent("x", "Holiday", "2026-10-14", "2026-10-15", "", ""),
                _event("A", _tomorrow_at(9))]
    r = make(a).handle("check_schedule_gaps", {"date": "tomorrow"}, {})
    assert r["data"]["events"] <= 1


def test_gaps_custom_buffer():
    a = Agent()
    a.events = [_event("A", _tomorrow_at(9), _tomorrow_at(10)), _event("B", _tomorrow_at(10, 20), _tomorrow_at(11))]
    assert make(a).handle("check_schedule_gaps", {"date": "tomorrow"}, {})["data"]["flags"] == []
    assert len(make(a).handle("check_schedule_gaps", {"date": "tomorrow", "buffer": "30 minutes"}, {})["data"]["flags"]) == 1


# ---------------------------------------------------------------- #96 slots

def test_slots_skip_busy_time_and_respect_work_hours():
    a = Agent()
    a.busy = {"primary": [(_tomorrow_at(9), _tomorrow_at(11))]}
    r = make(a).handle("find_meeting_slot", {"date": "tomorrow", "duration": "30 minutes"}, {})
    slots = [datetime.fromisoformat(s) for s in r["data"]["slots"]]
    assert slots and slots[0] == _tomorrow_at(11)
    assert all(9 <= s.hour < 18 for s in slots)
    assert len(slots) == 3


def test_slots_include_attendee_busy_and_flag_unreadable_calendars():
    a = Agent()
    a.busy = {
        "primary": [],
        "sam@example.com": [(_tomorrow_at(9), _tomorrow_at(12))],
        "lee@example.com": None,
    }
    r = make(a).handle(
        "find_meeting_slot",
        {"date": "tomorrow", "attendees": "sam@example.com, lee@example.com"}, {},
    )
    first = datetime.fromisoformat(r["data"]["slots"][0])
    assert first == _tomorrow_at(12)
    assert "lee@example.com" in r["response"] and "only account for yours" in r["response"]


def test_slots_unknown_attendee_name_is_reported_not_guessed():
    r = make().handle("find_meeting_slot", {"date": "tomorrow", "attendees": "Zed"}, {})
    assert "Zed" in r["response"] and "didn't check" in r["response"]


def test_slots_none_free():
    a = Agent()
    a.busy = {"primary": [(_tomorrow_at(0), _tomorrow_at(0) + timedelta(days=1))]}
    r = make(a).handle("find_meeting_slot", {"date": "tomorrow"}, {})
    assert r["data"]["slots"] == [] and "couldn't find" in r["response"]


def test_slots_fall_back_to_own_events_without_free_busy():
    class NoFB(Agent):
        free_busy = None
    a = NoFB()
    a.events = [_event("Busy", _tomorrow_at(9), _tomorrow_at(10))]
    r = make(a).handle("find_meeting_slot", {"date": "tomorrow"}, {})
    assert datetime.fromisoformat(r["data"]["slots"][0]) == _tomorrow_at(10)


def test_slots_custom_work_hours():
    r = make(work_hours=(14, 16)).handle("find_meeting_slot", {"date": "tomorrow", "duration": 60}, {})
    slots = [datetime.fromisoformat(s) for s in r["data"]["slots"]]
    assert [s.hour for s in slots] == [14, 15]


# ---------------------------------------------------------------- #97 conflicts

def _create_entities(**kw):
    base = {"title": "Dentist", "date": "tomorrow", "time": "3pm"}
    base.update(kw)
    return base


def test_create_event_with_conflict_asks_before_creating():
    a = Agent()
    a.events = [_event("Team call", _tomorrow_at(15), _tomorrow_at(16))]
    r = make(a).handle("create_event", _create_entities(), {})
    assert r["needs_confirmation"] is True and r["confirm_intent"] == "create_event"
    assert "Team call" in r["response"]
    assert a.created == []
    r2 = make(a).handle("create_event", dict(r["confirm_entities"], _confirmed=True), {})
    assert len(a.created) == 1 and "added" in r2["response"]


def test_create_event_without_conflict_creates_immediately():
    a = Agent()
    a.events = [_event("Earlier", _tomorrow_at(9), _tomorrow_at(10))]
    r = make(a).handle("create_event", _create_entities(), {})
    assert not r.get("needs_confirmation") and len(a.created) == 1


def test_adjacent_events_are_not_conflicts():
    a = Agent()
    a.events = [_event("Before", _tomorrow_at(14), _tomorrow_at(15))]
    make(a).handle("create_event", _create_entities(), {})
    assert len(a.created) == 1


def test_conflict_check_failure_does_not_block_creation():
    class Flaky(Agent):
        def list_events_between(self, *a, **k):
            raise RuntimeError("api down")
    a = Flaky()
    make(a).handle("create_event", _create_entities(), {})
    assert len(a.created) == 1


def test_create_event_works_with_an_agent_lacking_the_window_call():
    class Old(Agent):
        list_events_between = None
    a = Old()
    a.events = [_event("Team call", _tomorrow_at(15), _tomorrow_at(16))]
    assert make(a).handle("create_event", _create_entities(), {}).get("needs_confirmation")


# ---------------------------------------------------------------- #98 recurrence

def test_recurring_event_passes_rrule_to_agent():
    a = Agent()
    r = make(a).handle("create_event", _create_entities(recurrence="every weekday", count=10), {})
    assert a.created[0]["recurrence"] == ["RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;COUNT=10"]
    assert "repeating every weekday" in r["response"]


def test_recurrence_read_from_raw_query_when_entity_missing():
    a = Agent()
    make(a).handle("create_event", _create_entities(raw_query="add gym every monday at 6am"), {})
    assert a.created[0]["recurrence"] == ["RRULE:FREQ=WEEKLY;BYDAY=MO"]


def test_unreadable_recurrence_asks_instead_of_creating_a_one_off():
    a = Agent()
    r = make(a).handle("create_event", _create_entities(recurrence="whenever"), {})
    assert r["data"]["needs_clarification"] is True and a.created == []


def test_plain_event_has_no_recurrence():
    a = Agent()
    make(a).handle("create_event", _create_entities(recurrence="none"), {})
    assert a.created[0]["recurrence"] is None


def test_duration_sets_event_end():
    a = Agent()
    make(a).handle("create_event", _create_entities(duration="90 minutes"), {})
    c = a.created[0]
    assert c["end"] - c["start"] == timedelta(minutes=90)


def test_rrule_helpers():
    assert _build_rrule("every other week")[0] == "RRULE:FREQ=WEEKLY;INTERVAL=2"
    assert _build_rrule("monthly", count=3)[0].endswith("COUNT=3")
    assert _build_rrule("every monday and wednesday")[0] == "RRULE:FREQ=WEEKLY;BYDAY=MO,WE"
    assert _build_rrule("every month")[0] == "RRULE:FREQ=MONTHLY"  # 'mon' must not read as Monday
    assert _build_rrule("gibberish") is None
    until = _build_rrule("daily", until="2026-12-31", tz=TZ)[0]
    assert "UNTIL=20261231T182959Z" in until
    both = _build_rrule("daily", count=2, until="2026-12-31", tz=TZ)[0]
    assert "COUNT=2" in both and "UNTIL" not in both


def test_duration_parser():
    assert _parse_duration_minutes("1h30", default=60) == 90
    assert _parse_duration_minutes("half an hour", default=60) == 30
    assert _parse_duration_minutes("nonsense", default=45) == 45
    assert _parse_duration_minutes(None, default=45) == 45
    assert _parse_duration_minutes(99999, default=45) == 45


# ---------------------------------------------------------------- not connected

def test_new_intents_respect_the_not_connected_guard():
    h = HermesEngine(None, "Asia/Kolkata")
    for intent in ("email_digest", "draft_email", "search_email",
                   "check_schedule_gaps", "find_meeting_slot", "inbox_zero"):
        assert "not connected" in h.handle(intent, {}, {})["response"].lower()
