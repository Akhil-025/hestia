# tests/test_web_section17.py
"""
Backlog section 17 (web UI): login (#186), live updates (#187), the Today
dashboard (#180), the activity feed (#182), cross-module search (#184), the
settings page (#188), "why this answer" (#189) and data export (#190).

These exercise the Flask app through its test client with real Mnemosyne,
Artemis, Apollo and Chronos engines on temp files; only the pieces that need
heavy dependencies (Pluto's Redis/Qdrant, Athena's vector store, Iris's CLIP)
are small fakes.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("flask")

from core import web_auth, web_dashboard, web_export, web_search, web_settings  # noqa: E402
from core.event_bus import EventBus  # noqa: E402
from core.web_live import LiveHub  # noqa: E402
from modules.apollo.engine import ApolloEngine  # noqa: E402
from modules.artemis.engine import ArtemisEngine  # noqa: E402
from modules.artemis.tracker import ArtemisTracker  # noqa: E402
from modules.chronos.engine import ChronosEngine  # noqa: E402
from modules.mnemosyne.db import MnemosyneDB  # noqa: E402
from test_mnemosyne import make_engine as make_mnemosyne  # noqa: E402
from web_ui import HestiaWebUI  # noqa: E402


class _LLM:
    def generate(self, prompt):
        return "ok"


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def memory(tmp):
    eng, _ = make_mnemosyne(str(tmp))
    return eng


def _ui(memory, **kw):
    return HestiaWebUI(memory=memory, **kw)


# ===========================================================================
# #186  login
# ===========================================================================

def _login_ui(memory, tmp, **kw):
    kw.setdefault("session_key_file", str(tmp / "session.key"))
    return _ui(memory, password="correct horse", **kw)


def test_without_a_password_nothing_changes(memory):
    c = _ui(memory).app.test_client()
    assert c.get("/").status_code == 200
    assert c.get("/api/history").status_code == 200
    assert c.get("/login").status_code == 302          # nothing to log in to
    assert c.get("/api/auth/status").get_json() == {"mode": "none", "authenticated": True}


def test_password_mode_gates_the_page_and_the_api(memory, tmp):
    c = _login_ui(memory, tmp).app.test_client()
    r = c.get("/")
    assert r.status_code == 302 and r.headers["Location"].startswith("/login")
    r = c.get("/api/history")
    assert r.status_code == 401 and r.get_json()["login"] is True
    assert c.get("/login").status_code == 200
    assert c.get("/api/auth/status").get_json()["authenticated"] is False


def test_correct_password_signs_in_and_logout_signs_out(memory, tmp):
    c = _login_ui(memory, tmp).app.test_client()
    r = c.post("/login", data={"password": "correct horse", "next": "/"})
    assert r.status_code == 303 and r.headers["Location"] == "/"
    assert "HttpOnly" in r.headers["Set-Cookie"] and "SameSite=Lax" in r.headers["Set-Cookie"]
    assert c.get("/").status_code == 200
    assert c.get("/api/history").status_code == 200
    assert c.post("/logout").status_code == 303
    assert c.get("/api/history").status_code == 401


def test_wrong_password_is_refused_and_throttled(memory, tmp):
    ui = _login_ui(memory, tmp)
    c = ui.app.test_client()
    for _ in range(4):
        assert c.post("/login", data={"password": "nope"}).status_code == 401
    assert c.post("/login", data={"password": "nope"}).status_code == 401   # 5th: arms lockout
    r = c.post("/login", data={"password": "correct horse"})                # even the right one
    assert r.status_code == 429 and "Retry-After" in r.headers
    assert c.get("/api/history").status_code == 401


def test_hashed_password_works(memory, tmp):
    from werkzeug.security import generate_password_hash
    ui = _ui(memory, password_hash=generate_password_hash("s3cret-pass"),
             session_key_file=str(tmp / "k"))
    c = ui.app.test_client()
    assert c.post("/login", data={"password": "wrong"}).status_code == 401
    assert c.post("/login", data={"password": "s3cret-pass"}).status_code == 303


def test_login_redirect_cannot_leave_the_site(memory, tmp):
    c = _login_ui(memory, tmp).app.test_client()
    r = c.post("/login", data={"password": "correct horse", "next": "https://evil.example/x"})
    assert r.headers["Location"] == "/"
    c2 = _login_ui(memory, tmp).app.test_client()
    r = c2.post("/login", data={"password": "correct horse", "next": "//evil.example"})
    assert r.headers["Location"] == "/"


def test_cross_origin_post_is_refused_when_logged_in(memory, tmp):
    ui = _login_ui(memory, tmp, process_fn=lambda t: "hi")
    c = ui.app.test_client()
    c.post("/login", data={"password": "correct horse"})
    bad = c.post("/api/chat", json={"text": "hello"}, headers={"Origin": "https://evil.example"})
    assert bad.status_code == 403
    ok = c.post("/api/chat", json={"text": "hello"}, headers={"Origin": "http://localhost"})
    assert ok.status_code == 200                       # test client's host is "localhost"
    assert c.post("/api/chat", json={"text": "hello"}).status_code == 200   # no Origin: script


def test_allowed_origins_setting_admits_a_proxy(memory, tmp):
    ui = _login_ui(memory, tmp, process_fn=lambda t: "hi",
                   allowed_origins=["https://hestia.example.net"])
    c = ui.app.test_client()
    c.post("/login", data={"password": "correct horse"},
           headers={"Origin": "https://hestia.example.net"})
    r = c.post("/api/chat", json={"text": "x"}, headers={"Origin": "https://hestia.example.net"})
    assert r.status_code == 200


def test_api_key_still_works_beside_a_password(memory, tmp):
    c = _login_ui(memory, tmp, api_key="k" * 12).app.test_client()
    assert c.get("/api/history").status_code == 401
    assert c.get("/api/history", headers={"X-API-Key": "k" * 12}).status_code == 200
    assert c.get("/api/history", headers={"X-API-Key": "wrong"}).status_code == 401


def test_api_key_only_mode_is_unchanged(memory):
    c = _ui(memory, api_key="abc123").app.test_client()
    assert c.get("/").status_code == 200               # page open, as before
    assert c.get("/api/history").status_code == 401
    assert c.get("/api/history", headers={"X-API-Key": "abc123"}).status_code == 200


def test_lan_host_needs_a_password_or_key(memory, tmp):
    with pytest.raises(ValueError):
        _ui(memory, host="0.0.0.0")
    _ui(memory, host="0.0.0.0", password="pw-long-enough", session_key_file=str(tmp / "k"))
    _ui(memory, host="0.0.0.0", api_key="key")


def test_session_key_is_generated_once_and_reused(tmp):
    path = tmp / "sub" / "web.key"
    first = web_auth.resolve_secret_key(None, str(path))
    assert len(first) >= 32 and path.read_text().strip() == first
    assert web_auth.resolve_secret_key(None, str(path)) == first
    assert web_auth.resolve_secret_key("x" * 40, str(path)) == "x" * 40
    assert web_auth.resolve_secret_key("short", str(tmp / "other")) != "short"


def test_safe_next_rejects_everything_off_site():
    assert web_auth.safe_next("/settings?tab=1") == "/settings?tab=1"
    for bad in ("//evil.example", "https://evil.example", "/\\evil.example",
                "javascript:alert(1)", "", None, "/ok\r\nSet-Cookie: x=1"):
        assert web_auth.safe_next(bad) == "/"


def test_lockout_expires():
    now = [0.0]
    a = web_auth.PasswordAuth(password="pw", max_failures=2, lockout_seconds=30,
                              clock=lambda: now[0])
    a.record_failure("1.2.3.4")
    assert a.record_failure("1.2.3.4") == 30 and a.locked_for("1.2.3.4") > 0
    assert a.locked_for("5.6.7.8") == 0
    now[0] = 31
    assert a.locked_for("1.2.3.4") == 0


# ===========================================================================
# #187  live updates
# ===========================================================================

def test_hub_replays_missed_events_after_a_reconnect():
    hub = LiveHub()
    first = hub.publish("message", query="a")
    hub.publish("message", query="b")
    sub = hub.subscribe(last_id=first["id"])
    assert sub.get(0.1)["query"] == "b" and sub.get(0.05) is None
    sub.close()


def test_hub_caps_open_streams_and_frees_slots():
    hub = LiveHub(max_subscribers=2)
    a, b = hub.subscribe(), hub.subscribe()
    assert hub.subscribe() is None
    a.close()
    assert hub.subscribe() is not None
    assert hub.subscriber_count == 2
    b.close()


def test_slow_subscriber_is_told_to_resync_not_blocked():
    hub = LiveHub(queue_size=10)
    sub = hub.subscribe()
    for i in range(25):
        hub.publish("message", n=i)             # never blocks
    assert sub.get(0.1)["type"] == "resync"
    sub.close()


def test_bus_events_become_live_events():
    bus = EventBus()
    hub = LiveHub()
    hub.attach_bus(bus)
    hub.attach_bus(bus)                          # idempotent
    with hub.origin("web"):
        bus.emit_sync("interaction_logged",
                      {"query": "log my sleep", "response": "Logged.", "intent": "track_sleep"})
    bus.emit_sync("interaction_logged", {"query": "hi", "response": "hello", "intent": "chat"})
    bus.emit_sync("speak", {"text": "Reminder: call mum"})
    bus.emit_sync("speak", {"text": "Your weekly health summary is ready."})
    events = hub.recent(10)
    assert [e["type"] for e in events] == ["message", "message", "reminder", "notification"]
    assert events[0]["origin"] == "web" and events[1]["origin"] == "other"
    assert events[0]["module"] == "apollo" and events[1]["module"] == "core"
    hub.detach_bus()
    bus.emit_sync("speak", {"text": "Reminder: after detach"})
    assert len(hub.recent(10)) == 4


def test_a_broken_listener_cannot_fail_a_query():
    bus = EventBus()
    hub = LiveHub()
    hub.attach_bus(bus)
    hub.publish = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    bus.emit_sync("interaction_logged", {"query": "x", "response": "y", "intent": "chat"})
    bus.emit_sync("speak", {"text": "Reminder: z"})        # neither raises


def test_sse_endpoint_streams_hello_then_events(memory):
    ui = _ui(memory)
    ui.live_hub.publish("reminder", text="Reminder: stretch")
    resp = ui.app.test_client().get("/api/live/stream", buffered=False)
    assert resp.mimetype == "text/event-stream"
    it = iter(resp.response)
    body = ""
    for _ in range(3):
        chunk = next(it)
        body += chunk.decode() if isinstance(chunk, bytes) else chunk
    resp.close()
    assert "event: hello" in body
    assert ui.live_hub.subscriber_count == 0                # slot released on close


def test_sse_reconnect_resumes_after_last_event_id(memory):
    ui = _ui(memory)
    first = ui.live_hub.publish("message", query="old", response="", intent="chat", module="core")
    ui.live_hub.publish("message", query="new", response="", intent="chat", module="core")
    resp = ui.app.test_client().get("/api/live/stream", buffered=False,
                                    headers={"Last-Event-ID": str(first["id"])})
    it = iter(resp.response)
    body = ""
    for _ in range(3):
        chunk = next(it)
        body += chunk.decode() if isinstance(chunk, bytes) else chunk
    resp.close()
    assert '"query": "new"' in body and '"query": "old"' not in body


def test_sse_refuses_past_the_cap_and_needs_login(memory, tmp):
    ui = _ui(memory)
    ui.live_hub.max_subscribers = 1
    held = ui.live_hub.subscribe()
    assert ui.app.test_client().get("/api/live/stream").status_code == 429
    held.close()
    locked = _login_ui(memory, tmp).app.test_client()
    assert locked.get("/api/live/stream").status_code == 401


# ===========================================================================
# #189  why this answer
# ===========================================================================

TRACE = {"module": "apollo", "intent": "track_sleep", "confidence": 0.93, "source": "nlu",
         "reason": "intent registered to apollo", "latency_ms": 412.5, "request_id": "r-1",
         "checked": ["alias: no match", "registry: apollo"], "query": "private words",
         "script": "latin"}


def test_stream_carries_the_routing_record_in_a_header(memory):
    ui = _ui(memory, process_fn=lambda t: "x",
             process_traced_fn=lambda t: ("Slept 7 hours — logged.", TRACE))
    r = ui.app.test_client().post("/api/chat/stream", json={"text": "slept 7 hours"})
    assert r.get_data(as_text=True).strip() == "Slept 7 hours — logged."
    trace = json.loads(r.headers["X-Hestia-Trace"])
    assert trace["module"] == "apollo" and trace["confidence"] == 0.93
    assert trace["checked"] == ["alias: no match", "registry: apollo"]
    assert "query" not in trace and "script" not in trace    # only what the panel shows
    r.headers["X-Hestia-Trace"].encode("ascii")              # header-safe


def test_plain_chat_returns_the_trace_in_json(memory):
    ui = _ui(memory, process_fn=lambda t: "x", process_traced_fn=lambda t: ("hi", TRACE))
    body = ui.app.test_client().post("/api/chat", json={"text": "hello"}).get_json()
    assert body["response"] == "hi" and body["trace"]["intent"] == "track_sleep"


def test_chat_works_without_a_traced_pipeline(memory):
    ui = _ui(memory, process_fn=lambda t: "plain")
    c = ui.app.test_client()
    assert c.post("/api/chat", json={"text": "hello"}).get_json() == {"response": "plain", "trace": None}
    r = c.post("/api/chat/stream", json={"text": "hello"})
    assert "X-Hestia-Trace" not in r.headers


def test_a_failing_pipeline_is_a_500_not_a_200_with_junk(memory):
    def boom(t):
        raise RuntimeError("x")
    c = _ui(memory, process_fn=boom).app.test_client()
    assert c.post("/api/chat/stream", json={"text": "hello"}).status_code == 500
    assert c.post("/api/chat", json={"text": "hello"}).status_code == 500


def test_web_chat_turns_are_tagged_with_their_origin(memory):
    seen = []
    ui = _ui(memory)
    ui.process_fn = lambda t: seen.append(ui.live_hub.current_origin()) or "ok"
    ui.app.test_client().post("/api/chat", json={"text": "hello"})
    assert seen == ["web"] and ui.live_hub.current_origin() == "other"


def test_hestia_process_text_traced_returns_the_routing_record():
    """The real pipeline method, with its collaborators stubbed out."""
    import main
    h = main.Hestia.__new__(main.Hestia)
    rec = {"module": "chronos", "intent": "set_reminder", "confidence": 0.9}
    h.process_text = lambda text: (main.Hestia._trace_local.__setattr__("trace", rec) or "done")
    assert main.Hestia.process_text_traced(h, "remind me") == ("done", rec)


# ===========================================================================
# #182  activity feed
# ===========================================================================

def test_activity_lists_newest_first_with_module_and_utc_time(memory):
    memory.db.push_interaction("log my sleep", "Logged.", "track_sleep")
    memory.db.push_interaction("what time is it", "3pm", "get_time")
    memory.db.push_interaction("remember this", "Noted.", "take_note")
    ui = _ui(memory)
    data = ui.app.test_client().get("/api/activity?limit=10").get_json()
    assert [i["query"] for i in data["items"]] == ["remember this", "what time is it", "log my sleep"]
    assert {i["module"] for i in data["items"]} == {"core", "chronos", "apollo"}
    assert data["items"][0]["ts"].endswith("Z")
    assert set(data["modules"]) == {"core", "chronos", "apollo"}


def test_activity_can_be_filtered_by_module_and_merges_notifications(memory):
    memory.db.push_interaction("log my sleep", "Logged.", "track_sleep")
    memory.db.push_interaction("hello", "hi", "chat")
    ui = _ui(memory)
    ui.live_hub.publish("reminder", text="Reminder: stretch")
    c = ui.app.test_client()
    only = c.get("/api/activity?module=apollo").get_json()["items"]
    assert [i["query"] for i in only] == ["log my sleep"]
    kinds = [i["kind"] for i in c.get("/api/activity").get_json()["items"]]
    assert "reminder" in kinds and kinds.count("message") == 2
    assert c.get("/api/activity?limit=abc").status_code == 400


# ===========================================================================
# #180  Today dashboard
# ===========================================================================

TODAY = date(2026, 10, 7)


def test_goal_score_weighs_priority_urgency_and_progress():
    hi_today = web_dashboard.score_goal({"name": "Ship", "priority": "high", "due_date": "2026-10-07",
                                         "progress": 0.0}, TODAY)
    assert hi_today["score"] == 30 + 40 and "high-priority goal, due today" == hi_today["reason"]
    late = web_dashboard.score_goal({"name": "Tax", "priority": None, "due_date": "2026-10-04",
                                     "progress": 0.5}, TODAY)
    assert late["score"] == 15 + 50 - 5 and "overdue by 3 days" in late["reason"]
    far = web_dashboard.score_goal({"name": "Later", "priority": "low", "due_date": "2027-01-01",
                                    "progress": 0}, TODAY)
    assert far["score"] == 10
    assert web_dashboard.score_goal({"name": "x", "due_date": "garbage"}, TODAY)["score"] == 15


def test_top_priority_is_the_highest_score_and_explains_itself():
    now = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
    overdue_reminder = web_dashboard.score_reminder(
        {"source": "reminder", "text": "Pay rent", "overdue": True, "when": None}, now)
    soon = web_dashboard.score_reminder(
        {"source": "reminder", "text": "Call", "overdue": False,
         "when": "2026-10-07T11:00:00+00:00"}, now)
    later = web_dashboard.score_reminder(
        {"source": "reminder", "text": "Night", "overdue": False,
         "when": "2026-10-07T20:00:00+00:00"}, now)
    assert overdue_reminder["score"] == 45 and soon["score"] == 35 and later is None
    assert web_dashboard.score_reminder({"source": "calendar", "text": "x"}, now) is None
    streak = web_dashboard.score_habit("read", 9, False)
    assert streak["score"] == 28 and "9-day streak" in streak["reason"]
    assert web_dashboard.score_habit("read", 2, False) is None
    assert web_dashboard.score_habit("read", 9, True) is None
    best = web_dashboard.pick_top([None, streak, overdue_reminder, soon])
    assert best["title"] == "Pay rent"
    assert web_dashboard.pick_top([None, None]) is None


class _Chronos:
    def dashboard_data(self, n):
        return {"date": "2026-10-07", "holiday": None, "notes": [],
                "today": [{"source": "reminder", "text": "Pay rent", "overdue": True, "when": None}],
                "upcoming": [{"text": "Dentist", "day": "2026-10-09", "when": "2026-10-09T10:00:00+05:30"}]}


def _artemis(tmp):
    tracker = ArtemisTracker(tmp / "art.json")
    eng = ArtemisEngine(tracker=tracker, llm=_LLM())
    tracker.add_habit("read")
    tracker.add_habit("run")
    tracker.complete_habit("run", today=TODAY)
    tracker.add_goal("Finish report", due_date="2026-10-08", priority="high")
    tracker.add_goal("Learn guitar", due_date="2027-03-01", priority="low")
    return eng


def test_today_view_combines_every_available_module(memory, tmp):
    apollo = ApolloEngine(db_path=tmp / "a.db", llm=_LLM())
    apollo.db.log_water(500)
    pluto = SimpleNamespace(pf_manager=SimpleNamespace(db=SimpleNamespace(
        get_totals_between=lambda a, b: [{"category": "food", "total": 120.5, "count": 3},
                                         {"category": "travel", "total": 40, "count": 1}])))
    ui = _ui(memory, apollo=apollo, artemis=_artemis(tmp), pluto=pluto, chronos=_Chronos())
    data = ui.app.test_client().get("/api/dashboard/today").get_json()
    assert data["errors"] == {}
    # High-priority goal due tomorrow (30 + 25) outranks the overdue reminder (45).
    assert data["top_priority"]["title"] == "Finish report"
    assert data["top_priority"]["reason"] == "high-priority goal, due tomorrow"
    titles = [d["title"] for d in data["deadlines"]]
    assert titles == ["Finish report", "Dentist"]                # soonest first, far goal left out
    assert data["habits"]["total"] == 2
    assert data["finance"]["spent"] == 160.5 and data["finance"]["top_categories"][0]["category"] == "food"
    assert data["health"]["water_ml"] >= 0
    assert data["modules"] == {"chronos": True, "artemis": True, "apollo": True,
                               "pluto": True, "study": data["modules"]["study"]}


def test_today_view_survives_missing_and_broken_modules(memory):
    ui = _ui(memory)
    data = ui.app.test_client().get("/api/dashboard/today").get_json()
    assert data["top_priority"] is None and data["agenda"] is None and data["habits"] is None

    class Broken:
        def dashboard_data(self, n):
            raise RuntimeError("calendar down")
    ui2 = _ui(memory, chronos=Broken())
    data2 = ui2.app.test_client().get("/api/dashboard/today").get_json()
    assert data2["errors"] == {"chronos": "RuntimeError"} and data2["agenda"] is None


def test_today_view_is_cached_briefly_and_refreshable(memory):
    calls = []

    class C(_Chronos):
        def dashboard_data(self, n):
            calls.append(1)
            return super().dashboard_data(n)
    c = _ui(memory, chronos=C()).app.test_client()
    c.get("/api/dashboard/today")
    c.get("/api/dashboard/today")
    assert len(calls) == 1
    c.get("/api/dashboard/today?refresh=1")
    assert len(calls) == 2


def test_chronos_dashboard_data_lists_today_and_the_week_ahead(tmp):
    from test_chronos_engine import Rig
    rig = Rig(tmp)
    rig.db.add_reminder("Today thing", "2026-09-29T16:00:00+05:30")
    rig.db.add_reminder("Friday thing", "2026-10-02T09:00:00+05:30")
    rig.db.add_reminder("Next month", "2026-11-20T09:00:00+05:30")
    data = rig.engine.dashboard_data(7)
    assert data["date"] == "2026-09-29"
    assert [i["text"] for i in data["today"] if i["source"] == "reminder"] == ["Today thing"]
    assert [(u["text"], u["day"]) for u in data["upcoming"]] == [("Friday thing", "2026-10-02")]


# ===========================================================================
# #184  cross-module search
# ===========================================================================

class _Athena:
    def search_sources(self, q, n):
        return [{"text": "Entropy is a measure of disorder in a system. " * 6,
                 "file_name": "physics.pdf", "page": 12, "subject": "physics", "score": 0.8}]


class _Iris:
    def search_records(self, q, n):
        return [{"file_path": "C:\\Pictures\\beach\\IMG_1.jpg", "caption": "A beach at sunset",
                 "tags": "beach,sunset", "is_sensitive": 0}]


def test_search_covers_memory_documents_and_photos(memory):
    memory.db.set_fact("favourite_beach", "Palolem")
    memory.db.push_interaction("note: book beach trip", "Noted.", "take_note")
    memory.db.push_interaction("what is the weather", "Sunny at the beach.", "get_weather")
    ui = _ui(memory, athena=_Athena(), iris=_Iris())
    data = ui.app.test_client().get("/api/search?q=beach").get_json()
    g = data["groups"]
    assert all(v["ok"] for v in g.values())
    kinds = {i["kind"] for i in g["mnemosyne"]["items"]}
    assert {"fact", "note", "history"} <= kinds
    assert g["athena"]["items"][0]["meta"]["page"] == 12
    assert g["iris"]["items"][0]["title"] == "IMG_1.jpg"
    assert data["total"] == sum(len(v["items"]) for v in g.values())


def test_one_failing_source_does_not_blank_the_others(memory):
    class Bad:
        def search_sources(self, q, n):
            raise RuntimeError("index offline")
    memory.db.set_fact("pet", "a cat called Mo")
    data = _ui(memory, athena=Bad(), iris=_Iris()).app.test_client().get("/api/search?q=cat").get_json()
    assert data["groups"]["athena"] == {"ok": False, "items": [], "error": "RuntimeError"}
    assert data["groups"]["mnemosyne"]["ok"] and data["groups"]["iris"]["ok"]


def test_a_stuck_source_times_out_instead_of_hanging_the_page(memory):
    release = threading.Event()

    class Slow:
        def search_sources(self, q, n):
            release.wait(5)
            return []
    ui = _ui(memory, athena=Slow())
    out = web_search.search_all(ui, "anything", timeout=0.2)
    release.set()
    assert out["groups"]["athena"]["error"] == "timed out" and out["groups"]["mnemosyne"]["ok"]


def test_search_treats_percent_and_underscore_literally(memory):
    memory.db.set_fact("budget", "50% of salary")
    memory.db.set_fact("other", "5000 of salary")
    hits = memory.db.search_facts("50%")
    assert [h["key"] for h in hits] == ["budget"]
    assert memory.db.search_facts("a_b") == []


def test_search_validates_input_and_reports_unavailable_modules(memory):
    c = _ui(memory).app.test_client()
    assert c.get("/api/search?q=a").status_code == 400
    assert c.get("/api/search?q=" + "x" * 300).status_code == 400
    g = c.get("/api/search?q=hello").get_json()["groups"]
    assert g["athena"]["error"] == "not available" and g["iris"]["error"] == "not available"


def test_snippet_centres_on_the_match():
    text = "x" * 300 + " needle " + "y" * 300
    s = web_search.snippet(text, "needle", 100)
    assert "needle" in s and len(s) <= 104 and s.startswith("…")


# ===========================================================================
# #188  settings
# ===========================================================================

SAMPLE_YAML = (
    "# my config\r\n"
    "chronos:\r\n"
    "  timezone: 'Asia/Kolkata'\r\n"
    "  default_snooze_minutes: 10   # minutes\r\n"
    "  skip_public_holidays: false  # keep this comment\r\n"
    "tts:\r\n"
    "  rate: 175\r\n"
    "webui:\r\n"
    "  port: 5000\r\n"
)


@pytest.fixture
def cfg(tmp):
    p = tmp / "laptop_config.yaml"
    p.write_bytes(SAMPLE_YAML.encode())
    return p


def test_saving_a_setting_keeps_comments_and_line_endings(memory, cfg):
    c = _ui(memory, config_path=cfg).app.test_client()
    r = c.post("/api/settings", json={"changes": {"chronos.default_snooze_minutes": 20,
                                                   "chronos.skip_public_holidays": True}})
    assert r.status_code == 200 and r.get_json()["restart_required"] is True
    raw = cfg.read_bytes().decode()
    assert "default_snooze_minutes: 20   # minutes" in raw
    assert "skip_public_holidays: true  # keep this comment" in raw
    assert raw.count("\r\n") == SAMPLE_YAML.count("\r\n") and "\n" not in raw.replace("\r\n", "")
    assert (cfg.parent / "laptop_config.yaml.bak").read_bytes().decode() == SAMPLE_YAML
    shown = {s["key"]: s["value"] for s in c.get("/api/settings").get_json()["settings"]}
    assert shown["chronos.default_snooze_minutes"] == 20 and shown["tts.rate"] == 175


def test_a_missing_key_is_added_under_its_section(memory, cfg):
    c = _ui(memory, config_path=cfg).app.test_client()
    assert c.post("/api/settings", json={"changes": {"tts.volume": 0.5}}).status_code == 200
    assert "  volume: 0.5\r\n" in cfg.read_bytes().decode()
    assert c.post("/api/settings", json={"changes": {"whatif.enabled": False}}).status_code == 200
    import yaml
    assert yaml.safe_load(cfg.read_text())["whatif"] == {"enabled": False}


def test_only_whitelisted_valid_values_are_accepted(memory, cfg):
    c = _ui(memory, config_path=cfg).app.test_client()
    before = cfg.read_bytes()
    for bad in ({"webui.port": 1}, {"hermes.allow_mailbox_changes": True},
                {"tts.rate": 9999}, {"tts.rate": True}, {"tts.rate": "fast"},
                {"apollo.units.weight": "stone"}, {"chronos.skip_public_holidays": "yes"},
                {}, "nope"):
        r = c.post("/api/settings", json={"changes": bad})
        assert r.status_code == 400, bad
    assert cfg.read_bytes() == before                    # nothing was written


def test_settings_unavailable_without_a_file_and_needs_login(memory, tmp, cfg):
    assert _ui(memory).app.test_client().get("/api/settings").status_code == 503
    c = _login_ui(memory, tmp, config_path=cfg).app.test_client()
    assert c.get("/api/settings").status_code == 401
    assert c.post("/api/settings", json={"changes": {"tts.rate": 200}}).status_code == 401


def test_the_shipped_example_config_edits_cleanly():
    text = (Path(__file__).parent.parent / "config" / "laptop_config.example.yaml").read_text(encoding="utf-8")
    keys = {s.key: s for s in web_settings.SPEC}
    new, _ = web_settings.apply_changes(text, {"tts.rate": 200, "apollo.units.weight": "lb"})
    assert new != text and text.count("\n") == new.count("\n")
    # every whitelisted key is either present or addable, and re-parses
    import yaml
    for key, s in keys.items():
        value = (not s.default) if s.kind == "bool" else s.default
        web_settings.apply_changes(text, {key: value})
    assert yaml.safe_load(new)["tts"]["rate"] == 200


def test_no_whitelisted_setting_is_sensitive():
    forbidden = ("api_key", "token", "password", "path", "host", "port", "allow_mailbox",
                 "credentials", "secret", "travel", "faces", "camera", "dir")
    for s in web_settings.SPEC:
        assert not any(w in s.key for w in forbidden), s.key


# ===========================================================================
# #190  export
# ===========================================================================

def test_csv_neutralises_spreadsheet_formulas_but_keeps_numbers():
    csv_text = web_export.to_csv([{"name": "=HYPERLINK(\"http://x\")", "amount": -4.5, "note": "+1 call",
                                   "ok": True, "empty": None, "plain": "fine, with comma"}])
    assert csv_text.startswith("\ufeff")
    lines = csv_text.lstrip("\ufeff").split("\r\n")
    assert lines[0] == "name,amount,note,ok,empty,plain"
    assert lines[1].startswith("\"'=HYPERLINK(") and ",-4.5," in lines[1]
    assert "'+1 call" in lines[1] and ",true," in lines[1] and "\"fine, with comma\"" in lines[1]


def test_csv_columns_are_the_union_of_all_rows():
    out = web_export.to_csv([{"a": 1}, {"b": 2}]).lstrip("\ufeff").split("\r\n")
    assert out[0] == "a,b" and out[1] == "1," and out[2] == ",2"


def test_export_endpoint_serves_mnemosyne_and_artemis_tables(memory, tmp):
    memory.db.set_fact("city", "Mumbai")
    memory.db.push_interaction("note this", "Noted.", "take_note")
    memory.db.add_reminder("Pay rent", "2026-10-09T09:00:00+05:30")
    ui = _ui(memory, artemis=_artemis(tmp))
    c = ui.app.test_client()
    r = c.get("/api/export/mnemosyne/facts?format=csv")
    assert r.status_code == 200 and "attachment; filename=hestia_mnemosyne_facts_" in r.headers["Content-Disposition"]
    assert "city" in r.get_data(as_text=True) and "Mumbai" in r.get_data(as_text=True)
    j = c.get("/api/export/mnemosyne/notes?format=json").get_json()
    assert j["count"] == 1 and j["rows"][0]["query"] == "note this"
    assert c.get("/api/export/chronos/reminders?format=json").get_json()["rows"][0]["text"] == "Pay rent"
    habits = c.get("/api/export/artemis/habits?format=json").get_json()["rows"]
    assert {h["name"] for h in habits} == {"read", "run"}
    done = c.get("/api/export/artemis/completions?format=json").get_json()["rows"]
    assert done == [{"habit": "run", "date": "2026-10-07"}]
    assert c.get("/api/export/artemis/goals?format=json").get_json()["count"] == 2


def test_export_apollo_tables(memory, tmp):
    apollo = ApolloEngine(db_path=tmp / "a.db", llm=_LLM())
    apollo.db.log_weight(71.2, "")
    apollo.db.log_water(300)
    c = _ui(memory, apollo=apollo).app.test_client()
    assert c.get("/api/export/apollo/weight?format=json").get_json()["rows"][0]["weight_kg"] == 71.2
    assert c.get("/api/export/apollo/water?format=json").get_json()["count"] == 1
    assert c.get("/api/export/apollo/sleep?format=csv").status_code == 200   # empty is fine


def test_export_rejects_unknown_things_and_hides_missing_modules(memory):
    c = _ui(memory).app.test_client()
    assert c.get("/api/export/pluto/expenses").status_code == 404      # module not loaded
    assert c.get("/api/export/mnemosyne/passwords").status_code == 404
    assert c.get("/api/export/nope/x").status_code == 404
    assert c.get("/api/export/mnemosyne/facts?format=xml").status_code == 400
    mods = {m["module"] for m in c.get("/api/export/datasets").get_json()}
    assert mods == {"mnemosyne", "chronos"}


def test_export_needs_login(memory, tmp):
    c = _login_ui(memory, tmp).app.test_client()
    assert c.get("/api/export/mnemosyne/facts").status_code == 401


def test_export_pages_beyond_the_1000_row_cap(memory):
    for i in range(1500):
        memory.db.set_fact(f"k{i}", f"v{i}")
    assert len(web_export.get_rows(_ui(memory), "mnemosyne", "facts")) == 1500


# ===========================================================================
# retrieval-only engine methods behind the search (#184)
# ===========================================================================

def test_iris_search_records_merges_sources_without_duplicates():
    from modules.iris.iris_engine import IrisEngine
    eng = IrisEngine.__new__(IrisEngine)
    a = {"file_path": "/p/a.jpg", "caption": "beach"}
    b = {"file_path": "/p/b.jpg", "caption": "beach hut"}
    eng._semantic_matches = lambda q, n: [a]
    eng._object_matches = lambda q, n: []
    eng.db = SimpleNamespace(search_files_by_caption=lambda q, n: [a, b],
                             search_files_by_tags=lambda q, n: [b])
    assert [r["file_path"] for r in eng.search_records("beach", 10)] == ["/p/a.jpg", "/p/b.jpg"]
    assert len(eng.search_records("beach", 1)) == 1
    assert eng.search_records("   ") == []


def test_athena_search_sources_stops_before_the_language_model():
    from modules.athena.engine import AthenaEngine
    from modules.athena.models import SearchResults
    eng = AthenaEngine.__new__(AthenaEngine)
    calls = []

    class FakeRag:
        def search(self, q, n_results=5, **kw):
            calls.append((q, n_results))
            return object()           # opaque: from_rag_response is stubbed below
    eng.rag = FakeRag()
    # Retrieval is asked for exactly n results; no generation step is involved.
    orig = SearchResults.from_rag_response
    SearchResults.from_rag_response = staticmethod(lambda resp: SimpleNamespace(to_source_documents=lambda: []))
    try:
        assert eng.search_sources("entropy", 4) == []
        assert calls == [("entropy", 4)]
        assert eng.search_sources("  ") == []
    finally:
        SearchResults.from_rag_response = orig


# ===========================================================================
# the page itself
# ===========================================================================

def _template(name="index.html"):
    return (Path(__file__).parent.parent / "templates" / name).read_text(encoding="utf-8")


def test_new_sections_exist_and_every_nav_link_has_a_page():
    import re
    html = _template()
    sections = set(re.findall(r'<section id="(\w+)" class="section', html))
    nav = set(re.findall(r'<a [^>]*data-section="(\w+)"', html))
    assert {"today", "activity", "search"} <= sections
    assert nav <= sections
    assert html.count('class="section active show"') == 1       # exactly one default page
    assert 'id="today" class="section active show"' in html


def test_phone_tab_bar_is_not_hidden_by_a_later_rule():
    """The base 'display:none' must come BEFORE the media query that shows the
    bar. Declared after it, it won and phones had no navigation at all."""
    html = _template()
    base = html.index(".mobile-tabbar { display: none; }")
    media = html.index("@media (max-width: 700px) {\n      .sidebar { display: none; }")
    assert base < media
    assert html.count(".mobile-tabbar { display: none; }") == 1


def test_page_uses_the_viewport_and_theme_hooks_phones_need():
    html = _template()
    assert "viewport-fit=cover" in html and 'name="theme-color"' in html
    assert html.index("hestia_theme") < html.index("<style>")        # theme applied before first paint


def test_login_page_posts_to_login_and_escapes_the_error(memory, tmp):
    c = _login_ui(memory, tmp).app.test_client()
    html = c.get("/login").get_data(as_text=True)
    assert 'action="/login"' in html and 'type="password"' in html
    bad = c.post("/login", data={"password": "<script>x</script>"}).get_data(as_text=True)
    assert "<script>x</script>" not in bad


def test_logout_button_only_appears_when_a_password_is_set(memory, tmp):
    plain = _ui(memory).app.test_client().get("/").get_data(as_text=True)
    assert 'action="/logout"' not in plain and 'content="none"' in plain
    c = _login_ui(memory, tmp).app.test_client()
    c.post("/login", data={"password": "correct horse"})
    page = c.get("/").get_data(as_text=True)
    assert 'action="/logout"' in page and 'content="password"' in page
