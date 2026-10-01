# tests/test_artemis_extras.py
"""Focus sessions (#122), goal milestones (#124), badges (#127), smart nudges (#129), goal templates (#130)."""
import os
import sys
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import modules.artemis.engine as engine_mod
from core.heartbeat import HestiaHeartbeat
from modules.artemis import extras
from modules.artemis.engine import ArtemisEngine, _parse_steps
from modules.artemis.templates import GOAL_TEMPLATES, find_template
from modules.artemis.tracker import ArtemisTracker, Goal, GoalNotFoundError, Habit

UTC = timezone.utc
T0 = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


class Clock:
    """Controllable replacement for tracker.now()."""
    def __init__(self, at=T0):
        self.at = at

    def __call__(self):
        return self.at

    def advance(self, **kw):
        self.at += timedelta(**kw)


class LLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt):
        return self.reply


class FakeBus:
    def __init__(self):
        self.sent = []

    def emit(self, event, data=None):
        self.sent.append((event, data))


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make(tmp_path, clock, monkeypatch):
    bus = FakeBus()
    monkeypatch.setattr(engine_mod, "bus", bus)
    made = []

    def build(llm_reply='["Outline", "Draft", "Review", "Submit"]', nudges=None):
        t = ArtemisTracker(tmp_path / f"a{len(made)}.json", timezone_name="UTC")
        t.now = clock
        e = ArtemisEngine(tracker=t, llm=LLM(llm_reply), nudges=nudges)
        made.append(e)
        e.bus = bus
        return e
    yield build
    for e in made:
        e._cancel_focus_timer()


# --- pure logic ---------------------------------------------------------

def test_focus_finish_caps_minutes_at_planned():
    f = {}
    extras.focus_start(f, T0, 25)
    rec = extras.focus_finish(f, T0 + timedelta(minutes=90))
    assert rec["minutes"] == 25 and rec["completed"] is True and f["active"] is None


def test_focus_finish_early_and_nothing_running():
    f = {}
    assert extras.focus_finish(f, T0) is None
    extras.focus_start(f, T0, 25)
    rec = extras.focus_finish(f, T0 + timedelta(minutes=10))
    assert rec["minutes"] == 10 and rec["completed"] is False


def test_focus_stats_window():
    f = {"sessions": [
        {"start": (T0 - timedelta(days=10)).isoformat(), "minutes": 25, "completed": True},
        {"start": (T0 - timedelta(days=1)).isoformat(), "minutes": 25, "completed": True, "task": "thesis"},
        {"start": T0.isoformat(), "minutes": 10, "completed": False},
    ]}
    s = extras.focus_stats(f, T0)
    assert (s["minutes"], s["sessions"], s["completed"], s["total_minutes"]) == (35, 2, 1, 60)
    assert s["tasks"] == ["thesis"]


def test_typical_minute_needs_samples_and_uses_median():
    assert extras.typical_minute([480] * 4) is None
    assert extras.typical_minute([470, 480, 485, 900, 475]) == 480
    assert extras.fmt_minute(480) == "8:00 am" and extras.fmt_minute(13 * 60 + 5) == "1:05 pm"


def test_badges_thresholds():
    h = {"a": Habit("a", streak=0, best_streak=30, total_completions=60)}
    g = {"x": Goal("x", status="completed")}
    got = extras.evaluate_badges(h, g, {"sessions": [{"minutes": 600}]})
    assert {"streak_7", "streak_30", "done_50", "goal_first", "focus_10h"} <= got
    assert "streak_100" not in got and "habits_3" not in got


# --- milestones ---------------------------------------------------------

def test_milestones_drive_progress_and_complete_goal(make):
    t = make().tracker
    t.add_goal("paper")
    t.set_goal_milestones("paper", ["Outline", "Draft", "Submit"])
    r = t.complete_goal_milestone("paper", 1)
    assert (r["done"], r["total"]) == (1, 3) and round(r["progress"], 2) == 0.33
    t.complete_goal_milestone("paper", "draft")
    r = t.complete_goal_milestone("paper", "sub")
    assert r["goal_completed"] is True and t.get_goal("paper").status == "completed"


def test_milestone_no_match_and_missing_goal(make):
    t = make().tracker
    t.add_goal("paper", milestones=["Outline"])
    assert t.complete_goal_milestone("paper", "zzz") is None
    assert t.complete_goal_milestone("paper", 9) is None
    with pytest.raises(GoalNotFoundError):
        t.complete_goal_milestone("nope", 1)


def test_resetting_milestones_keeps_done_ones(make):
    t = make().tracker
    t.add_goal("g", milestones=["A", "B"])
    t.complete_goal_milestone("g", "A")
    t.set_goal_milestones("g", ["A", "C", "D"])
    assert [m["done"] for m in t.get_goal("g").milestones] == [True, False, False]


def test_legacy_goal_round_trips_without_milestones():
    raw = {"progress": 0.5, "status": "active", "created_at": "x", "updated_at": "y",
           "due_date": None, "priority": None}
    assert Goal.from_dict("g", raw).to_dict() == raw


def test_parse_steps_json_lines_and_junk():
    assert _parse_steps('Sure! ["a b", "c", "d"]') == ["a b", "c", "d"]
    assert _parse_steps("1. Plan\n2) Write\n- Review") == ["Plan", "Write", "Review"]
    assert _parse_steps("") == []


# --- templates ----------------------------------------------------------

def test_find_template_by_alias_and_longest_match():
    assert find_template("start the run a 5k template") == "run_5k"
    assert find_template("I want to write a paper") == "write_paper"
    assert find_template("something unrelated") is None


def test_every_template_is_valid():
    for key, t in GOAL_TEMPLATES.items():
        assert 3 <= len(t["milestones"]) <= 8 and t["days"] > 0 and t["priority"] in ("low", "medium", "high"), key


# --- engine: milestones & templates ------------------------------------

def test_decompose_creates_goal_with_steps(make):
    e = make()
    out = e.handle("decompose_goal", {"name": "publish the paper"}, {})
    assert "Outline" in out["response"]
    assert [m["title"] for m in e.tracker.get_goal("publish the paper").milestones] == ["Outline", "Draft", "Review", "Submit"]


def test_decompose_llm_failure_is_graceful(make):
    e = make(llm_reply="")
    out = e.handle("decompose_goal", {"name": "x"}, {})
    assert out["confidence"] < 0.5 and "x" not in e.tracker.get_goals()


def test_complete_milestone_by_text_and_next(make):
    e = make()
    e.handle("decompose_goal", {"name": "paper"}, {})
    out = e.handle("complete_milestone", {}, {"raw_query": "I finished step 2 of paper"})
    assert "Draft" in out["response"] and "1/4" in out["response"]
    assert "Outline" in e.handle("complete_milestone", {"name": "paper"}, {})["response"]   # next undone


def test_complete_milestone_without_steps(make):
    assert "no goals with steps" in make().handle("complete_milestone", {}, {})["response"]


def test_template_listing_and_adding(make, clock):
    e = make()
    assert "Run a 5K" in e.handle("list_goal_templates", {}, {})["response"]
    out = e.handle("add_goal_from_template", {}, {"raw_query": "start the run a 5k template"})
    g = e.tracker.get_goal("Run a 5K")
    assert len(g.milestones) == 5 and g.due_date == (clock().date() + timedelta(days=56)).isoformat()
    assert "already have" in e.handle("add_goal_from_template", {"template": "run a 5k"}, {})["response"]
    assert "Which template" in e.handle("add_goal_from_template", {}, {"raw_query": "from a template"})["response"]


# --- engine: focus ------------------------------------------------------

def test_start_focus_parses_minutes_and_task(make):
    e = make()
    out = e.handle("start_focus", {}, {"raw_query": "start a 40 minute focus session on my thesis"})
    assert "40 minutes" in out["response"] and "my thesis" in out["response"]
    assert e.tracker.active_focus()["planned"] == 40


def test_start_focus_default_and_already_running(make, clock):
    e = make()
    assert "25 minutes" in e.handle("start_focus", {}, {})["response"]
    clock.advance(minutes=5)
    assert "already running" in e.handle("start_focus", {}, {})["response"]
    assert "20 min left" in e.handle("focus_stats", {}, {})["response"]


def test_start_focus_bad_length(make):
    assert "1 to" in make().handle("start_focus", {"minutes": "999"}, {})["response"]


def test_stop_focus_early_and_when_idle(make, clock):
    e = make()
    assert "No focus session" in e.handle("stop_focus", {}, {})["response"]
    e.handle("start_focus", {"minutes": "25"}, {})
    clock.advance(minutes=10)
    assert "10 focused minutes" in e.handle("stop_focus", {}, {})["response"]
    assert e.tracker.active_focus() is None


def test_timer_callback_finishes_and_announces(make, clock):
    e = make()
    e.handle("start_focus", {"minutes": "25"}, {})
    start = e.tracker.active_focus()["start"]
    clock.advance(minutes=25)
    e._on_focus_timer(start)
    assert e.tracker.active_focus() is None and e.tracker.focus_stats()["completed"] == 1
    (event, data), = e.bus.sent
    assert event == "speak" and "break" in data["text"]


def test_stale_timer_does_not_touch_a_newer_session(make, clock):
    e = make()
    e.handle("start_focus", {"minutes": "25"}, {})
    old = e.tracker.active_focus()["start"]
    clock.advance(minutes=3)
    e.handle("stop_focus", {}, {})
    clock.advance(minutes=1)
    e.handle("start_focus", {"minutes": "25"}, {})
    e._on_focus_timer(old)
    assert e.tracker.active_focus() is not None and not e.bus.sent


def test_expired_session_is_logged_when_starting_a_new_one(make, clock):
    e = make()
    e.handle("start_focus", {"minutes": "10"}, {})
    clock.advance(minutes=60)                        # timer never fired (e.g. app restarted)
    e.handle("start_focus", {"minutes": "10"}, {})
    assert e.tracker.focus_stats()["completed"] == 1


def test_long_break_after_four_sessions(make, clock):
    e = make()
    for _ in range(4):
        e.handle("start_focus", {"minutes": "25"}, {})
        clock.advance(minutes=25)
        out = e.handle("stop_focus", {}, {})
        clock.advance(minutes=5)
    assert "15 minutes" in out["response"]


def test_productivity_summary_includes_focus(make, clock):
    e = make()
    e.handle("start_focus", {"minutes": "25"}, {})
    clock.advance(minutes=25)
    e.handle("stop_focus", {}, {})
    assert "Focus this week: 25 min" in e.handle("productivity_summary", {}, {})["response"]


# --- engine: badges -----------------------------------------------------

def test_badge_announced_once_on_streak(make):
    e = make()
    e.handle("add_habit", {"name": "read"}, {})
    d0 = date(2026, 8, 1)
    for n in range(6):
        e.tracker.complete_habit("read", today=d0 + timedelta(days=n))
    e.tracker.complete_habit("read", today=d0 + timedelta(days=6))
    first = e.handle("complete_habit", {"name": "read"}, {})          # live completion: resets, but best_streak is 7
    assert "Week Warrior" in first["response"] and first["data"]["new_badges"] == ["streak_7"]
    again = e.handle("complete_habit", {"name": "read"}, {})
    assert "badge" not in again["response"].lower()


def test_list_badges_empty_and_earned(make):
    e = make()
    assert "No badges yet" in e.handle("list_badges", {}, {})["response"]
    e.tracker.add_goal("g", milestones=["a"])
    e.tracker.complete_goal_milestone("g", 1)
    assert "Goal Getter" in e.handle("list_badges", {}, {})["response"]


# --- nudges -------------------------------------------------------------

def habit_with_times(e, name="meditate", minute=8 * 60, n=5):
    e.tracker.add_habit(name)
    base = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    for i in range(n):
        e.tracker.complete_habit(name, today=(base + timedelta(days=i)).date(),
                                 at=base + timedelta(days=i, minutes=minute))


def test_times_only_recorded_for_live_or_explicit_completions(make):
    e = make()
    e.tracker.add_habit("a")
    e.tracker.complete_habit("a", today=date(2026, 8, 1))            # back-dated: no time
    assert e.tracker.get_habit("a").times == []
    e.tracker.complete_habit("a", today=date(2026, 8, 2), at=datetime(2026, 8, 2, 7, 30, tzinfo=UTC))
    assert e.tracker.get_habit("a").times == [450]


def test_nudge_fires_once_when_late(make, clock):
    e = make()
    habit_with_times(e)
    clock.at = datetime(2026, 9, 1, 8, 30, tzinfo=UTC)              # only 30 min past usual
    assert e.check_habit_nudges() is None
    clock.at = datetime(2026, 9, 1, 9, 30, tzinfo=UTC)
    text = e.check_habit_nudges()
    assert "meditate" in text and "8:00 am" in text
    assert e.check_habit_nudges() is None                            # once per day
    clock.at = datetime(2026, 9, 2, 9, 30, tzinfo=UTC)
    assert e.check_habit_nudges() is not None                        # next day again


@pytest.mark.parametrize("setup", ["done", "paused", "late_night", "too_few", "disabled"])
def test_nudge_suppressed(make, clock, setup):
    e = make(nudges={"enabled": False} if setup == "disabled" else None)
    habit_with_times(e, n=3 if setup == "too_few" else 5)
    clock.at = datetime(2026, 9, 1, 23 if setup == "late_night" else 10, 0, tzinfo=UTC)
    if setup == "done":
        e.tracker.complete_habit("meditate", today=clock().date(), at=clock())
    if setup == "paused":
        e.tracker.pause_habit("meditate", today=clock().date())
    assert e.check_habit_nudges() is None


def test_nudge_picks_longest_streak_first(make, clock):
    e = make()
    habit_with_times(e, "short")
    habit_with_times(e, "long")
    e.tracker.complete_habit("long", today=date(2026, 8, 25))       # extends long's streak
    clock.at = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    assert "long" in e.check_habit_nudges() and "short" in e.check_habit_nudges()


def test_heartbeat_speaks_artemis_nudge(monkeypatch):
    import core.heartbeat as hb
    bus = FakeBus()
    monkeypatch.setattr(hb, "bus", bus)
    h = HestiaHeartbeat(artemis=SimpleNamespace(check_habit_nudges=lambda: "Gentle nudge: x"))
    h._maybe_run_artemis_checkins()
    assert bus.sent == [("speak", {"text": "Gentle nudge: x"})]
    HestiaHeartbeat(artemis=SimpleNamespace(check_habit_nudges=lambda: 1 / 0))._maybe_run_artemis_checkins()
    HestiaHeartbeat()._maybe_run_artemis_checkins()                  # no artemis: no-op
