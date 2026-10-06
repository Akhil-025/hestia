# tests/test_hermes_remaining.py
"""
Tests for the items left open in backlog #51-#100:

  #91  Todoist tasks (core/todoist_agent.py + Hermes todoist_* intents)
  #92  scheduled email digest (HermesEngine.check_email_digest + heartbeat hook)
  #95  driving-time lookup in the back-to-back check (core/travel_time.py)
  #96  booking a proposed meeting slot and inviting attendees
  #99  inbox-zero that archives after a "yes" (opt-in)
  #88  weather-code field-name tolerance

Everything runs against fakes; nothing touches the network or a real account.
"""
import os
import sys
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.google_agent as ga  # noqa: E402
from core.heartbeat import HestiaHeartbeat  # noqa: E402
from core.todoist_agent import (  # noqa: E402
    TodoistAgent,
    TodoistError,
    TodoistTask,
    find_matches,
    rank_tasks,
    select_due,
)
from core.travel_time import TravelTimeEstimator  # noqa: E402
from modules.chronos import agenda  # noqa: E402
from modules.hecate.intent_registry import module_for_intent  # noqa: E402
from modules.hermes.engine import HermesEngine, _parse_choice  # noqa: E402

TZ = ZoneInfo("Asia/Kolkata")


# =============================================================== helpers

def _email(subject, sender="Someone <someone@example.com>", snippet="", mid=None):
    return ga.Email(message_id=mid or subject, subject=subject, sender=sender,
                    snippet=snippet, date="Tue, 14 Oct 2026 09:00:00 +0530")


def _event(title, start, end=None, location=""):
    end = end or start + timedelta(hours=1)
    return ga.CalendarEvent(event_id=title, title=title, start=start.isoformat(),
                            end=end.isoformat(), location=location, description="")


def _tomorrow_at(h, m=0):
    d = datetime.now(TZ).date() + timedelta(days=1)
    return datetime(d.year, d.month, d.day, h, m, tzinfo=TZ)


class Agent:
    def __init__(self, modify=True):
        self.unread, self.events, self.busy = [], [], {"primary": []}
        self.created, self.archived = [], []
        self.can_modify_mailbox = modify

    def is_authenticated(self):
        return True

    def read_emails(self, max_results=5):
        return self.unread[:max_results]

    def list_events(self, max_results=5, days_ahead=7):
        return self.events

    def list_events_between(self, start, end, max_results=50):
        return self.events

    def free_busy(self, start, end, emails=None):
        return dict(self.busy)

    def create_event(self, title, start_dt, end_dt=None, location="", description="",
                     recurrence=None, attendees=None):
        self.created.append({"title": title, "start": start_dt, "end": end_dt,
                             "attendees": attendees})
        return True

    def archive_emails(self, ids):
        self.archived.extend(ids)
        return len(ids)


def make(agent=None, **kw):
    return HermesEngine(agent or Agent(), timezone_name="Asia/Kolkata", **kw)


# ================================================================ registry

@pytest.mark.parametrize("intent", [
    "book_meeting_slot", "todoist_list_tasks", "todoist_add_task",
    "todoist_complete_task", "todoist_prioritize",
])
def test_new_intents_registered_to_hermes(intent):
    assert module_for_intent(intent) == "hermes"
    assert make().can_handle(intent)


def test_todoist_intents_say_not_connected_without_a_token():
    h = HermesEngine(Agent(), "Asia/Kolkata")          # Google fine, no Todoist
    for intent in ("todoist_list_tasks", "todoist_add_task",
                   "todoist_complete_task", "todoist_prioritize"):
        out = h.handle(intent, {"task": "x"}, {})
        assert "todoist isn't connected" in out["response"].lower()
        assert out["confidence"] == 0.0


def test_todoist_works_without_google():
    fake = FakeTodoist([_t("1", "Pay rent", 4, date.today())])
    h = HermesEngine(None, "Asia/Kolkata", todoist=fake)   # no Google at all
    assert "pay rent" in h.handle("todoist_list_tasks", {}, {})["response"].lower()
    # ...while Google intents still report Google as missing.
    assert "not connected" in h.handle("read_email", {}, {})["response"].lower()


# =========================================================== Todoist agent

class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, method, url, headers, params, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": headers,
                           "params": params, "body": body})
        return self.replies.pop(0)


def _raw(i, content, priority=1, due=None):
    r = {"id": str(i), "content": content, "priority": priority, "project_id": "p"}
    if due:
        r["due"] = {"date": due, "string": "x"}
    return r


def test_agent_not_ready_without_token(monkeypatch):
    monkeypatch.delenv("TODOIST_API_TOKEN", raising=False)
    assert TodoistAgent().is_ready() is False
    with pytest.raises(TodoistError):
        TodoistAgent(transport=FakeTransport([])).list_tasks()


def test_agent_reads_token_from_environment(monkeypatch):
    monkeypatch.setenv("TODOIST_API_TOKEN", "abc")
    assert TodoistAgent().is_ready() is True


def test_list_tasks_uses_v1_url_bearer_header_and_follows_cursor():
    tr = FakeTransport([
        (200, {"results": [_raw(1, "A")], "next_cursor": "c2"}),
        (200, {"results": [_raw(2, "B")], "next_cursor": None}),
    ])
    tasks = TodoistAgent("tok", transport=tr).list_tasks()
    assert [t.content for t in tasks] == ["A", "B"]
    assert tr.calls[0]["url"] == "https://api.todoist.com/api/v1/tasks"
    assert tr.calls[0]["headers"]["Authorization"] == "Bearer tok"
    assert tr.calls[1]["params"]["cursor"] == "c2"
    assert "/rest/" not in tr.calls[0]["url"]


def test_list_tasks_accepts_a_bare_list_and_skips_junk():
    tr = FakeTransport([(200, [_raw(1, "A"), {"id": "", "content": "no id"}, "junk"])])
    assert [t.content for t in TodoistAgent("t", transport=tr).list_tasks()] == ["A"]


@pytest.mark.parametrize("status,needle", [(401, "rejected"), (403, "rejected"),
                                           (404, "couldn't find"), (429, "rate-limiting"),
                                           (500, "HTTP 500")])
def test_http_errors_become_todoist_errors(status, needle):
    tr = FakeTransport([(status, None)])
    with pytest.raises(TodoistError, match=needle):
        TodoistAgent("t", transport=tr).list_tasks()


def test_add_task_sends_body_and_clamps_priority():
    tr = FakeTransport([(200, _raw(9, "Buy ink", 4, "2026-10-07"))])
    task = TodoistAgent("t", transport=tr).add_task(
        "  Buy ink ", due_string="tomorrow", priority=9)
    body = tr.calls[0]["body"]
    assert body == {"content": "Buy ink", "due_string": "tomorrow", "priority": 4}
    assert tr.calls[0]["method"] == "POST"
    assert task.id == "9" and task.ui_priority == 1
    with pytest.raises(ValueError):
        TodoistAgent("t", transport=tr).add_task("   ")


def test_complete_task_posts_to_close_and_quotes_the_id():
    tr = FakeTransport([(204, None)])
    TodoistAgent("t", transport=tr).complete_task("a/b")
    assert tr.calls[0]["method"] == "POST"
    assert tr.calls[0]["url"].endswith("/tasks/a%2Fb/close")


def test_task_parsing_handles_datetimes_and_bad_priority():
    t = TodoistTask.from_api({"id": 1, "content": "x", "priority": "oops",
                              "due": {"date": "2026-10-07T09:30:00Z"}})
    assert t.priority == 1 and t.due_date == date(2026, 10, 7)
    assert TodoistTask.from_api({"id": 1, "content": "x", "due": None}).due_date is None


# ============================================================ ranking helpers

TODAY = date(2026, 10, 6)


def _t(i, content, priority=1, due=None):
    return TodoistTask(id=str(i), content=content, priority=priority, due_date=due)


def test_rank_overdue_then_today_then_future_then_undated():
    tasks = [
        _t(1, "undated p1", 4),
        _t(2, "future", 4, TODAY + timedelta(days=3)),
        _t(3, "today p1", 4, TODAY),
        _t(4, "today p4", 1, TODAY),
        _t(5, "overdue recent", 1, TODAY - timedelta(days=1)),
        _t(6, "overdue old", 1, TODAY - timedelta(days=9)),
    ]
    order = [t.content for t in rank_tasks(tasks, TODAY)]
    assert order == ["overdue old", "overdue recent", "today p1", "today p4",
                     "future", "undated p1"]


def test_select_due_scopes():
    tasks = [_t(1, "old", due=TODAY - timedelta(days=2)), _t(2, "now", due=TODAY),
             _t(3, "soon", due=TODAY + timedelta(days=4)),
             _t(4, "later", due=TODAY + timedelta(days=30)), _t(5, "none")]
    names = lambda s: sorted(t.content for t in select_due(tasks, TODAY, s))  # noqa: E731
    assert names("today") == ["now", "old"]
    assert names("overdue") == ["old"]
    assert names("week") == ["now", "old", "soon"]
    assert len(select_due(tasks, TODAY, "all")) == 5


def test_find_matches_prefers_exact_then_all_words():
    tasks = [_t(1, "Call the bank"), _t(2, "Call the bank about card"), _t(3, "Email Sam")]
    assert [t.id for t in find_matches(tasks, "call the bank")] == ["1"]
    assert [t.id for t in find_matches(tasks, "bank card")] == ["2"]
    assert find_matches(tasks, "zzz") == [] and find_matches(tasks, "") == []


# ============================================================ Todoist intents

class FakeTodoist:
    def __init__(self, tasks=None, fail=None):
        self.tasks = list(tasks or [])
        self.added, self.completed, self.fail = [], [], fail

    def is_ready(self):
        return True

    def list_tasks(self, project_id=None):
        if self.fail:
            raise self.fail
        return list(self.tasks)

    def add_task(self, content, *, due_string="", priority=None, project_id=""):
        self.added.append((content, due_string, priority))
        return TodoistTask(id="n1", content=content, priority=priority or 1,
                           due_string=due_string)

    def complete_task(self, task_id):
        self.completed.append(task_id)


def _hermes_with(todoist):
    return HermesEngine(Agent(), "Asia/Kolkata", todoist=todoist)


def test_list_tasks_today_ranks_and_speaks():
    today = datetime.now(TZ).date()
    todo = FakeTodoist([_t(1, "Pay rent", 4, today), _t(2, "File taxes", 1, today - timedelta(days=3)),
                        _t(3, "Next month thing", 1, today + timedelta(days=30))])
    out = _hermes_with(todo).handle("todoist_list_tasks", {}, {})
    r = out["response"]
    assert "2 tasks due today or overdue" in r
    assert r.index("File taxes") < r.index("Pay rent")       # overdue first
    assert "3 days overdue" in r and "Next month thing" not in r
    assert out["data"]["open_total"] == 3


def test_list_tasks_empty_mentions_open_total():
    out = _hermes_with(FakeTodoist([_t(1, "Later", due=datetime.now(TZ).date() + timedelta(days=9))])
                       ).handle("todoist_list_tasks", {}, {})
    assert "Nothing due today or overdue" in out["response"]
    assert "1 open in total" in out["response"]


def test_prioritize_names_the_top_three_and_changes_nothing():
    today = datetime.now(TZ).date()
    todo = FakeTodoist([_t(i, f"task {i}", 1, today) for i in range(5)] +
                       [_t(9, "urgent one", 4, today)])
    out = _hermes_with(todo).handle("todoist_prioritize", {}, {})
    assert out["response"].startswith("You have 6 open tasks, 6 due today. Start with:")
    assert "1. urgent one (due today, p1)" in out["response"]
    assert not todo.added and not todo.completed


def test_prioritize_empty():
    out = _hermes_with(FakeTodoist()).handle("todoist_prioritize", {}, {})
    assert "empty" in out["response"].lower()


def test_add_task_maps_priority_words_and_due():
    todo = FakeTodoist()
    h = _hermes_with(todo)
    out = h.handle("todoist_add_task", {"task": "call the bank", "due": "tomorrow",
                                        "priority": "p1"}, {})
    assert todo.added == [("call the bank", "tomorrow", 4)]       # p1 in the app = API 4
    assert "Added 'call the bank' to Todoist, due tomorrow, priority 1" in out["response"]
    h.handle("todoist_add_task", {"task": "x", "priority": "low"}, {})
    h.handle("todoist_add_task", {"task": "y", "priority": "gibberish"}, {})
    assert todo.added[1][2] == 1 and todo.added[2][2] is None


def test_add_task_without_text_asks_with_slot_fill():
    out = _hermes_with(FakeTodoist()).handle("todoist_add_task", {}, {})
    assert out["data"]["missing_slot"] == "task"


def test_complete_task_unique_ambiguous_and_missing():
    todo = FakeTodoist([_t(1, "Call the bank about card"), _t(2, "Call the plumber"),
                        _t(3, "Email Sam")])
    h = _hermes_with(todo)
    assert "Marked 'Email Sam' as done" in h.handle(
        "todoist_complete_task", {"task": "email sam"}, {})["response"]
    assert todo.completed == ["3"]
    amb = h.handle("todoist_complete_task", {"task": "call"}, {})
    assert "matches 2 tasks" in amb["response"] and todo.completed == ["3"]   # nothing completed
    assert amb["data"]["missing_slot"] == "task"
    miss = h.handle("todoist_complete_task", {"task": "zebra"}, {})
    assert "couldn't find" in miss["response"] and todo.completed == ["3"]


def test_todoist_failure_is_spoken_not_raised():
    out = _hermes_with(FakeTodoist(fail=TodoistError("Todoist rejected the API token."))
                       ).handle("todoist_list_tasks", {}, {})
    assert out["response"] == "Todoist rejected the API token."
    boom = _hermes_with(FakeTodoist(fail=RuntimeError("kaboom"))).handle(
        "todoist_prioritize", {}, {})
    assert "went wrong" in boom["response"].lower()


# =================================================================== #95 travel

class FakeGet:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, params, headers, timeout):
        self.calls.append((url, dict(params)))
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value(params) if callable(value) else value
        raise AssertionError(f"unexpected url {url}")


def _geo(params):
    pts = {"office": ("19.10", "72.85"), "airport": ("19.09", "72.87")}
    q = params["q"]
    return [{"lat": pts[q][0], "lon": pts[q][1]}] if q in pts else []


def test_flat_provider_never_calls_out():
    get = FakeGet({})
    est = TravelTimeEstimator("flat", get_json=get)
    assert est.estimate("a", "b") is None and get.calls == []
    assert est.enabled is False


def test_google_without_a_key_falls_back_to_flat(caplog):
    assert TravelTimeEstimator("google").provider == "flat"
    assert TravelTimeEstimator("nonsense").provider == "flat"


def test_osrm_geocodes_then_routes_with_lon_lat_order_and_rounds_up():
    get = FakeGet({"nominatim": _geo,
                   "router.project-osrm.org": {"code": "Ok", "routes": [{"duration": 1501.0}]}})
    est = TravelTimeEstimator("osrm", get_json=get, sleep=lambda s: None)
    out = est.estimate("Office", "Airport")
    assert out.minutes == 26 and out.source == "osrm"           # 1501 s -> 25.02 min -> 26
    route_url = [u for u, _ in get.calls if "osrm" in u][0]
    assert "72.850000,19.100000;72.870000,19.090000" in route_url      # lon,lat
    n = len(get.calls)
    assert est.estimate("office", "AIRPORT").minutes == 26             # cache, case-insensitive
    assert len(get.calls) == n


def test_osrm_geocoding_is_throttled_to_one_a_second():
    sleeps = []
    t = [0.0]
    get = FakeGet({"nominatim": _geo,
                   "router.project-osrm.org": {"code": "Ok", "routes": [{"duration": 60}]}})
    est = TravelTimeEstimator("osrm", get_json=get, sleep=sleeps.append, clock=lambda: t[0])
    est.estimate("office", "airport")
    assert sleeps and sleeps[0] == pytest.approx(1.0)       # second geocode waited


@pytest.mark.parametrize("osrm_reply", [
    {"code": "NoRoute"}, {"code": "Ok", "routes": []},
    {"code": "Ok", "routes": [{"duration": "x"}]}, "garbage",
])
def test_osrm_bad_replies_give_none(osrm_reply):
    get = FakeGet({"nominatim": _geo, "router.project-osrm.org": osrm_reply})
    assert TravelTimeEstimator("osrm", get_json=get, sleep=lambda s: None
                               ).estimate("office", "airport") is None


def test_unknown_place_and_network_failure_give_none_and_failures_are_not_cached():
    get = FakeGet({"nominatim": _geo, "router.project-osrm.org": OSError("down")})
    est = TravelTimeEstimator("osrm", get_json=get, sleep=lambda s: None)
    assert est.estimate("office", "nowhere") is None
    assert est.estimate("office", "airport") is None                    # raised -> None
    get.routes["router.project-osrm.org"] = {"code": "Ok", "routes": [{"duration": 600}]}
    assert est.estimate("office", "airport").minutes == 10             # retried, not cached


def test_same_place_is_zero_minutes():
    est = TravelTimeEstimator("osrm", get_json=FakeGet({}), sleep=lambda s: None)
    assert est.estimate("Office ", "office").minutes == 0


def test_google_distance_matrix():
    ok = {"status": "OK", "rows": [{"elements": [{"status": "OK", "duration": {"value": 1800}}]}]}
    est = TravelTimeEstimator("google", google_api_key="k", get_json=FakeGet({"distancematrix": ok}))
    out = est.estimate("a", "b")
    assert (out.minutes, out.source) == (30, "google")
    bad = {"status": "OK", "rows": [{"elements": [{"status": "NOT_FOUND"}]}]}
    assert TravelTimeEstimator("google", google_api_key="k",
                               get_json=FakeGet({"distancematrix": bad})).estimate("a", "b") is None


# ---- Hermes gap check using the estimator ------------------------------

class FakeTravel:
    def __init__(self, minutes=None, source="osrm", boom=False):
        self.minutes, self.source, self.boom, self.calls = minutes, source, boom, []

    def estimate(self, o, d):
        self.calls.append((o, d))
        if self.boom:
            raise RuntimeError("x")
        if self.minutes is None:
            return None
        return type("E", (), {"minutes": self.minutes, "source": self.source})()


def _gap_agent(gap_min=30):
    a = Agent()
    a.events = [_event("Standup", _tomorrow_at(10), _tomorrow_at(11), "Office"),
                _event("Lunch", _tomorrow_at(11, gap_min), _tomorrow_at(12, 30), "Airport")]
    return a


def test_gap_check_uses_real_travel_time_when_available():
    h = make(_gap_agent(30), travel=FakeTravel(minutes=45))
    out = h.handle("check_schedule_gaps", {"date": "tomorrow"}, {})
    f = out["data"]["flags"][0]
    assert f["kind"] == "travel" and f["travel_minutes"] == 45 and f["travel_source"] == "osrm"
    assert f["needed_minutes"] == 10 + 45
    assert "about 45 minutes apart by car" in out["response"]
    assert "come from osrm" in out["response"] and "flat" not in out["response"]


def test_a_short_real_drive_clears_what_the_flat_allowance_would_flag():
    # 30 min gap: flat (10 + 30 = 40) flags it, a real 5-minute drive (10 + 5) does not.
    assert make(_gap_agent(30)).handle("check_schedule_gaps", {"date": "tomorrow"}, {}
                                       )["data"]["flags"]
    clear = make(_gap_agent(30), travel=FakeTravel(minutes=5)).handle(
        "check_schedule_gaps", {"date": "tomorrow"}, {})
    assert clear["data"]["flags"] == []


@pytest.mark.parametrize("travel", [FakeTravel(minutes=None), FakeTravel(boom=True)])
def test_failed_lookup_falls_back_to_the_flat_allowance(travel):
    out = make(_gap_agent(30), travel=travel, travel_minutes=30).handle(
        "check_schedule_gaps", {"date": "tomorrow"}, {})
    f = out["data"]["flags"][0]
    assert f["travel_source"] == "flat" and f["travel_minutes"] == 30
    assert "flat 30-minute estimate" in out["response"]


def test_same_location_never_triggers_a_lookup():
    a = Agent()
    a.events = [_event("A", _tomorrow_at(10), _tomorrow_at(11), "Office"),
                _event("B", _tomorrow_at(11, 30), _tomorrow_at(12), "office")]
    tr = FakeTravel(minutes=99)
    make(a, travel=tr).handle("check_schedule_gaps", {"date": "tomorrow"}, {})
    assert tr.calls == []


# ================================================================ #96 booking

def _propose(h, agent, attendees="sam@example.com"):
    return h.handle("find_meeting_slot",
                    {"duration": "45 minutes", "attendees": attendees, "date": "tomorrow"}, {})


def test_find_slot_offers_to_book_and_remembers_the_times():
    a = Agent()
    h = make(a)
    out = _propose(h, a)
    assert 'Say "book the first one"' in out["response"] and "invite them" in out["response"]
    assert len(h._last_slots["slots"]) == 3


def test_booking_is_two_phase_and_invites_the_attendee():
    a = Agent()
    h = make(a)
    _propose(h, a)
    preview = h.handle("book_meeting_slot", {"choice": "second"}, {})
    assert preview["needs_confirmation"] is True and a.created == []
    assert "invite sam@example.com" in preview["response"]
    assert preview["confirm_intent"] == "book_meeting_slot"
    ce = preview["confirm_entities"]
    assert ce["attendees"] == ["sam@example.com"] and ce["duration"] == 45

    done = h.handle("book_meeting_slot", dict(ce, _confirmed=True), {})
    assert "Booked" in done["response"] and "sam@example.com" in done["response"]
    made = a.created[0]
    assert made["attendees"] == ["sam@example.com"]
    assert made["end"] - made["start"] == timedelta(minutes=45)
    assert made["start"].tzinfo is None                       # local wall-clock, like create_event
    assert h._last_slots is None                              # can't double-book the same offer


def test_booking_picks_the_requested_slot():
    a = Agent()
    h = make(a)
    _propose(h, a)
    slots = list(h._last_slots["slots"])
    pick = h.handle("book_meeting_slot", {"choice": "last"}, {})
    assert pick["confirm_entities"]["start"] == slots[-1].isoformat()


def test_booking_without_a_choice_asks_which_one():
    a = Agent()
    h = make(a)
    _propose(h, a)
    out = h.handle("book_meeting_slot", {}, {})
    assert out["data"]["missing_slot"] == "choice" and "Which one" in out["response"]
    bad = h.handle("book_meeting_slot", {"choice": "ninth"}, {})
    assert bad["data"].get("needs_clarification")


def test_booking_with_nothing_proposed_or_after_expiry():
    a = Agent()
    h = make(a)
    assert "find a time first" in h.handle("book_meeting_slot", {"choice": 1}, {})["response"]
    _propose(h, a)
    h._last_slots["at"] -= timedelta(hours=2)
    assert "find a time first" in h.handle("book_meeting_slot", {"choice": 1}, {})["response"]
    assert a.created == []


def test_booking_rechecks_the_calendar_before_creating():
    a = Agent()
    h = make(a)
    _propose(h, a)
    ce = h.handle("book_meeting_slot", {"choice": 1}, {})["confirm_entities"]
    start = datetime.fromisoformat(ce["start"])
    a.events = [_event("Sudden meeting", start, start + timedelta(hours=1))]   # filled up since
    out = h.handle("book_meeting_slot", dict(ce, _confirmed=True), {})
    assert "filled up" in out["response"] and a.created == []


def test_confirmed_booking_rejects_garbage_and_non_email_attendees():
    h = make(Agent())
    assert h.handle("book_meeting_slot", {"_confirmed": True, "start": "nope", "duration": 30}, {}
                    )["confidence"] == 0.0
    a = Agent()
    h = make(a)
    start = _tomorrow_at(15).isoformat()
    h.handle("book_meeting_slot", {"_confirmed": True, "start": start, "duration": 30,
                                   "attendees": ["not-an-email", "ok@example.com"], "title": "T"}, {})
    assert a.created[0]["attendees"] == ["ok@example.com"]


def test_no_attendees_means_no_attendees_kwarg():
    a = Agent()
    h = make(a)
    h.handle("find_meeting_slot", {"duration": "30", "date": "tomorrow"}, {})
    ce = h.handle("book_meeting_slot", {"choice": 1}, {})["confirm_entities"]
    h.handle("book_meeting_slot", dict(ce, _confirmed=True), {})
    assert a.created[0]["attendees"] is None


def test_parse_choice():
    assert [_parse_choice(v, 3) for v in ("first", "the second one", 3, "last", "2nd")] == [0, 1, 2, 2, 1]
    assert _parse_choice("fourth", 3) is None and _parse_choice(7, 3) is None
    assert _parse_choice(True, 3) is None and _parse_choice("banana", 3) is None


# ============================================================ #99 archiving

def _inbox(agent):
    agent.unread = [
        _email("Action required: invoice overdue", snippet="please reply?", mid="m-urgent"),
        _email("50% off sale", sender="Shop <no-reply@shop.com>", snippet="unsubscribe", mid="m-promo"),
        _email("Your receipt", sender="Store <noreply@store.com>", snippet="order confirmation", mid="m-receipt"),
    ]


def test_plan_only_by_default_and_says_nothing_changed():
    a = Agent()
    _inbox(a)
    out = make(a).handle("inbox_zero", {}, {})
    assert "I haven't changed anything" in out["response"] and a.archived == []


def test_plan_offers_archiving_when_enabled():
    a = Agent()
    _inbox(a)
    out = make(a, allow_mailbox_changes=True).handle("inbox_zero", {}, {})
    assert "archive the low-priority ones" in out["response"]


def test_apply_when_not_enabled_explains_how_to_turn_it_on():
    a = Agent()
    _inbox(a)
    out = make(a).handle("inbox_zero", {"apply": True}, {})
    assert "allow_mailbox_changes" in out["response"] and a.archived == []
    assert "needs_confirmation" not in out


def test_apply_is_two_phase_and_only_archives_the_archive_items():
    a = Agent()
    _inbox(a)
    h = make(a, allow_mailbox_changes=True)
    preview = h.handle("inbox_zero", {"apply": True}, {})
    assert preview["needs_confirmation"] and a.archived == []
    ids = preview["confirm_entities"]["ids"]
    assert "m-urgent" not in ids and set(ids) == {"m-promo", "m-receipt"}
    assert "nothing is deleted" in preview["response"].lower()

    done = h.handle("inbox_zero", dict(preview["confirm_entities"], _confirmed=True), {})
    assert "Archived 2 emails" in done["response"] and set(a.archived) == {"m-promo", "m-receipt"}


def test_confirmed_archive_is_refused_if_the_feature_is_off():
    a = Agent()
    out = make(a).handle("inbox_zero", {"apply": True, "ids": ["x"], "_confirmed": True}, {})
    assert "aren't enabled" in out["response"] and a.archived == []


def test_agent_without_the_modify_capability_is_not_used():
    a = Agent(modify=False)
    _inbox(a)
    out = make(a, allow_mailbox_changes=True).handle("inbox_zero", {"apply": True}, {})
    assert "allow_mailbox_changes" in out["response"] and a.archived == []


def test_nothing_safe_to_archive():
    a = Agent()
    a.unread = [_email("Action required: invoice overdue", snippet="reply?")]
    out = make(a, allow_mailbox_changes=True).handle("inbox_zero", {"apply": True}, {})
    assert "Nothing in your unread looks safe to archive" in out["response"]


def test_partial_and_failed_archive_are_reported_honestly():
    a = Agent()
    a.archive_emails = lambda ids: 1
    h = make(a, allow_mailbox_changes=True)
    out = h.handle("inbox_zero", {"apply": True, "ids": ["a", "b"], "_confirmed": True}, {})
    assert "Archived 1 of 2" in out["response"]
    a.archive_emails = lambda ids: 0
    assert h.handle("inbox_zero", {"apply": True, "ids": ["a"], "_confirmed": True}, {}
                    )["confidence"] == 0.0


# ---- Google agent: scopes, archive, attendees --------------------------

def _ready_agent(**kw):
    agent = ga.HestiaGoogleAgent(**kw)
    agent._gmail, agent._calendar = MagicMock(), MagicMock()
    return agent


def test_modify_scope_is_opt_in():
    assert ga.GMAIL_MODIFY_SCOPE not in ga.HestiaGoogleAgent()._scopes
    assert ga.GMAIL_MODIFY_SCOPE not in ga.SCOPES
    assert ga.GMAIL_MODIFY_SCOPE in ga.HestiaGoogleAgent(allow_mailbox_changes=True)._scopes


def test_archive_refused_without_opt_in_and_never_touches_gmail():
    agent = _ready_agent()
    with pytest.raises(ga.AuthenticationError):
        agent.archive_emails(["a"])
    agent._gmail.users.assert_not_called()


def test_archive_batches_and_removes_only_the_inbox_label():
    agent = _ready_agent(allow_mailbox_changes=True)
    assert agent.archive_emails(["a", "b"]) == 2
    call = agent._gmail.users().messages().batchModify
    call.assert_called_once_with(userId="me", body={"ids": ["a", "b"], "removeLabelIds": ["INBOX"]})
    assert agent.archive_emails([]) == 0


def test_archive_failure_returns_count_done_not_an_exception():
    agent = _ready_agent(allow_mailbox_changes=True)
    agent._gmail.users().messages().batchModify().execute.side_effect = RuntimeError("quota")
    assert agent.archive_emails(["a"]) == 0


def test_mark_read_removes_unread_label():
    agent = _ready_agent(allow_mailbox_changes=True)
    agent.mark_emails_read(["a"])
    assert agent._gmail.users().messages().batchModify.call_args.kwargs["body"]["removeLabelIds"] == ["UNREAD"]


def test_create_event_with_attendees_sends_invitations():
    agent = _ready_agent()
    start = datetime(2026, 10, 7, 10, 0)
    assert agent.create_event("Sync", start, attendees=["sam@example.com", " ", "ana@example.com"])
    insert = agent._calendar.events().insert
    kw = insert.call_args.kwargs
    assert kw["sendUpdates"] == "all"
    assert kw["body"]["attendees"] == [{"email": "sam@example.com"}, {"email": "ana@example.com"}]


def test_create_event_without_attendees_is_unchanged():
    agent = _ready_agent()
    agent.create_event("Solo", datetime(2026, 10, 7, 10, 0))
    kw = agent._calendar.events().insert.call_args.kwargs
    assert "sendUpdates" not in kw and "attendees" not in kw["body"]


# ============================================================ #92 digest

def _digest_hermes(tmp_path=None, **kw):
    a = Agent()
    _inbox(a)
    path = str(tmp_path / "state.json") if tmp_path else None
    return make(a, digest_time="08:30", state_path=path, **kw), a


def _at(h, m=0, day=None):
    d = day or datetime.now(TZ).date()
    return datetime(d.year, d.month, d.day, h, m, tzinfo=TZ)


def test_digest_off_by_default():
    a = Agent()
    _inbox(a)
    assert make(a).check_email_digest(_at(9)) is None


def test_digest_waits_for_the_time_then_speaks_once_a_day():
    h, _ = _digest_hermes()
    assert h.check_email_digest(_at(8, 0)) is None
    text = h.check_email_digest(_at(8, 31))
    assert text.startswith("Morning email digest.") and "3 unread emails" in text
    assert h.check_email_digest(_at(9, 0)) is None            # already given today
    tomorrow = datetime.now(TZ).date() + timedelta(days=1)
    assert h.check_email_digest(_at(8, 40, tomorrow)) is not None


def test_digest_not_read_out_in_the_evening_after_a_late_start():
    h, _ = _digest_hermes()
    assert h.check_email_digest(_at(20, 0)) is None


def test_digest_is_silent_for_an_empty_inbox_but_counts_as_done():
    h, a = _digest_hermes()
    a.unread = []
    assert h.check_email_digest(_at(9)) is None
    a.unread = [_email("late arrival")]
    assert h.check_email_digest(_at(10)) is None


def test_digest_retries_when_google_or_the_fetch_fails():
    h, a = _digest_hermes()
    a.is_authenticated = lambda: False
    assert h.check_email_digest(_at(9)) is None
    a.is_authenticated = lambda: True
    a.read_emails = MagicMock(side_effect=RuntimeError("net"))
    assert h.check_email_digest(_at(9, 5)) is None
    a.read_emails = lambda max_results=5: [_email("hi")]
    assert h.check_email_digest(_at(9, 10)) is not None       # not marked done by the failures


def test_digest_state_survives_a_restart(tmp_path):
    h, a = _digest_hermes(tmp_path)
    assert h.check_email_digest(_at(9)) is not None
    h2 = make(a, digest_time="08:30", state_path=str(tmp_path / "state.json"))
    assert h2.check_email_digest(_at(9, 30)) is None


def test_naive_now_is_treated_as_local_time():
    h, _ = _digest_hermes()
    assert h.check_email_digest(datetime.now().replace(hour=9, minute=0, tzinfo=None)) is not None


@pytest.mark.parametrize("bad", ["25:00", "soon", "8:75", "-1"])
def test_unreadable_digest_time_disables_the_feature(bad):
    h, _ = _digest_hermes()
    h2 = make(h._google, digest_time=bad)
    assert h2.check_email_digest(_at(9)) is None


def test_heartbeat_speaks_the_digest_and_survives_hook_errors():
    from core.event_bus import bus
    hb = HestiaHeartbeat(hermes=MagicMock(check_email_digest=MagicMock(return_value="Hello digest")))
    seen = []
    orig = bus.emit
    bus.emit = lambda name, payload=None: seen.append((name, payload))
    try:
        hb._maybe_run_hermes_digest()
        assert seen == [("speak", {"text": "Hello digest"})]
        hb.hermes.check_email_digest.side_effect = RuntimeError("x")
        hb._maybe_run_hermes_digest()               # must not raise
        assert len(seen) == 1
        HestiaHeartbeat()._maybe_run_hermes_digest()   # no hermes: no-op
    finally:
        bus.emit = orig


# ===================================================================== #88

def test_forecast_parser_accepts_either_weather_code_field_name():
    base = {"time": ["2026-10-07T10:00"], "precipitation_probability": [0]}
    day = date(2026, 10, 7)
    legacy = dict(base, weathercode=[63])
    modern = dict(base, weather_code=[63])
    assert agenda.rain_outlook(legacy, day, TZ)[0] == 100
    assert agenda.rain_outlook(modern, day, TZ)[0] == 100          # rain code seen under either name
    assert agenda.rain_outlook(base, day, TZ)[0] == 0


# ============================================ end to end via the orchestrator
# Real HestiaOrchestrator + real Hecate routing + real HermesEngine. Proves the
# new confirm intents survive the pending-confirmation round trip: nothing
# happens on the first message, and "yes" performs exactly the previewed action.

def _orchestrator(hermes):
    from modules.hecate.engine import HecateEngine
    from modules.hestia.orchestrator import HestiaOrchestrator
    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(hermes)
    return orch


def _nlu(intent, **entities):
    return {"intent": intent, "entities": entities, "confidence": 0.95}


def test_e2e_find_book_confirm_creates_the_event_and_invites():
    a = Agent()
    orch = _orchestrator(make(a))
    found = orch.dispatch("find a time with sam", _nlu(
        "find_meeting_slot", duration="30 minutes", attendees="sam@example.com", date="tomorrow"))
    assert "Here are" in found
    ask = orch.dispatch("book the first one", _nlu("book_meeting_slot", choice="first"))
    assert "Say yes" in ask and a.created == []                 # nothing booked yet
    done = orch.dispatch("yes", _nlu("chat"))
    assert "Booked" in done
    assert a.created[0]["attendees"] == ["sam@example.com"]


def test_e2e_declining_the_booking_creates_nothing():
    a = Agent()
    orch = _orchestrator(make(a))
    orch.dispatch("find a time", _nlu("find_meeting_slot", duration="30", date="tomorrow"))
    orch.dispatch("book the first one", _nlu("book_meeting_slot", choice=1))
    orch.dispatch("no", _nlu("chat"))
    assert a.created == []


def test_e2e_archive_requires_yes_and_archives_only_what_was_previewed():
    a = Agent()
    _inbox(a)
    orch = _orchestrator(make(a, allow_mailbox_changes=True))
    ask = orch.dispatch("archive the low priority ones", _nlu("inbox_zero", apply=True))
    assert "Say yes" in ask and a.archived == []
    a.unread.append(_email("brand new promo", sender="Shop <no-reply@shop.com>",
                           snippet="unsubscribe", mid="m-late"))    # arrives after the preview
    done = orch.dispatch("yes", _nlu("chat"))
    assert "Archived 2 emails" in done
    assert set(a.archived) == {"m-promo", "m-receipt"}              # not the late one
