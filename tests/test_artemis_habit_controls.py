# tests/test_artemis_habit_controls.py
"""Habit grace periods (#123), pause/resume (#128) and the weekly habit review (#125)."""
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.consensus import ConsensusEngine
from modules.artemis.engine import ArtemisEngine
from modules.artemis.tracker import ArtemisTracker, Habit, HabitNotFoundError, MAX_GRACE_DAYS

D0 = date(2026, 9, 1)


def d(n: int) -> date:
    return D0 + timedelta(days=n)


class NoLLM:
    def generate(self, prompt: str) -> str:
        return ""


@pytest.fixture
def tracker(tmp_path):
    return ArtemisTracker(tmp_path / "a.json")


def run_days(t, name, offsets):
    for n in offsets:
        t.complete_habit(name, today=d(n))


# --- grace periods ------------------------------------------------------

def test_strict_by_default_a_missed_day_resets(tracker):
    tracker.add_habit("read")
    run_days(tracker, "read", [0, 1, 2])
    assert tracker.complete_habit("read", today=d(4))["streak"] == 1


def test_per_habit_grace_keeps_streak_through_one_missed_day(tracker):
    tracker.add_habit("read")
    tracker.set_habit_grace("read", 1)
    run_days(tracker, "read", [0, 1, 2])
    r = tracker.complete_habit("read", today=d(4))
    assert r["streak"] == 4 and r["grace_used"] == 1


def test_grace_exceeded_resets(tracker):
    tracker.add_habit("read")
    tracker.set_habit_grace("read", 1)
    run_days(tracker, "read", [0, 1])
    assert tracker.complete_habit("read", today=d(5))["streak"] == 1


def test_default_grace_applies_and_habit_value_overrides(tmp_path):
    t = ArtemisTracker(tmp_path / "a.json", default_grace_days=2)
    t.add_habit("a"); t.add_habit("b")
    t.set_habit_grace("b", 0)
    run_days(t, "a", [0, 1]); run_days(t, "b", [0, 1])
    assert t.complete_habit("a", today=d(4))["streak"] == 3
    assert t.complete_habit("b", today=d(4))["streak"] == 1


def test_grace_validation_and_persistence(tracker):
    tracker.add_habit("read")
    with pytest.raises(ValueError):
        tracker.set_habit_grace("read", MAX_GRACE_DAYS + 1)
    with pytest.raises(HabitNotFoundError):
        tracker.set_habit_grace("nope", 1)
    tracker.set_habit_grace("read", 2)
    assert tracker.get_habit("read").grace_days == 2
    tracker.set_habit_grace("read", None)
    assert "grace_days" not in tracker.get_habit("read").to_dict()


# --- pauses -------------------------------------------------------------

def test_pause_protects_streak_across_the_break(tracker):
    tracker.add_habit("run")
    run_days(tracker, "run", [0, 1, 2])
    tracker.pause_habit("run", days=5, today=d(3))
    assert tracker.complete_habit("run", today=d(8))["streak"] == 4


def test_missed_day_after_pause_ends_still_breaks(tracker):
    tracker.add_habit("run")
    run_days(tracker, "run", [0, 1, 2])
    tracker.pause_habit("run", days=2, today=d(3))     # paused d3, d4
    assert tracker.complete_habit("run", today=d(6))["streak"] == 1   # d5 was a real miss


def test_open_ended_pause_and_resume(tracker):
    tracker.add_habit("run")
    run_days(tracker, "run", [0, 1])
    assert tracker.pause_habit("run", today=d(2))["until"] is None
    assert tracker.pause_habit("run", today=d(3))["already_paused"] is True
    assert tracker.resume_habit("run", today=d(10))["was_paused"] is True
    assert tracker.resume_habit("run", today=d(10))["was_paused"] is False
    assert tracker.complete_habit("run", today=d(10))["streak"] == 3


def test_completing_a_paused_habit_resumes_it(tracker):
    tracker.add_habit("run")
    run_days(tracker, "run", [0])
    tracker.pause_habit("run", today=d(1))
    r = tracker.complete_habit("run", today=d(5))
    assert r["resumed"] is True
    assert not tracker.get_habit("run").paused_on(d(6))


def test_pause_length_validated(tracker):
    tracker.add_habit("run")
    with pytest.raises(ValueError):
        tracker.pause_habit("run", days=0)


def test_legacy_habit_dict_round_trips_without_new_keys():
    raw = {"streak": 3, "last_done": "2026-09-01", "best_streak": 3,
           "total_completions": 3, "created_at": "2026-08-01T00:00:00+00:00"}
    assert Habit.from_dict("x", raw).to_dict() == raw


# --- consensus ----------------------------------------------------------

def test_streak_at_risk_logic():
    h = Habit("x", streak=5, last_done=d(0).isoformat())
    assert h.streak_ends_if_skipped_today(d(1)) is True
    assert h.streak_ends_if_skipped_today(d(1), default_grace=1) is False   # grace left
    h.pause(d(1))
    assert h.streak_ends_if_skipped_today(d(1)) is False                    # paused


def test_consensus_ignores_paused_and_graced_habits():
    def push(habit, grace=0):
        art = SimpleNamespace(tracker=SimpleNamespace(
            get_habits=lambda: {"m": habit}, default_grace_days=grace,
            get_at_risk_goals=lambda today=None: {}))
        return ConsensusEngine(None, art).push_signals(now=None.__class__ and __import__("datetime").datetime(2026, 9, 2, 12, tzinfo=__import__("datetime").timezone.utc))

    live = Habit("m", streak=10, last_done="2026-09-01")
    assert push(live)
    paused = Habit("m", streak=10, last_done="2026-09-01", pauses=[["2026-09-02", ""]])
    assert not push(paused)
    assert not push(Habit("m", streak=10, last_done="2026-09-01"), grace=2)


# --- weekly review ------------------------------------------------------

def test_weekly_review_counts_pauses_and_unknown_history_out(tracker):
    tracker.add_habit("run")
    run_days(tracker, "run", [0, 1, 2, 4, 5])
    tracker.pause_habit("run", days=1, today=d(3))
    rv = tracker.weekly_review(today=d(6))
    row = rv["habits"][0]
    assert row["done"] == 5 and row["possible"] == 5       # d3 paused; d6 (today) not yet over
    assert rv["overall_pct"] == 100


def test_weekly_review_today_counts_once_done(tracker):
    tracker.add_habit("run")
    run_days(tracker, "run", [0, 1, 2])
    row = tracker.weekly_review(today=d(2))["habits"][0]
    assert (row["done"], row["possible"]) == (3, 3)


def test_weekly_review_legacy_habit_has_no_pct(tmp_path):
    import json
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"habits": {"old": {"streak": 4, "last_done": "2026-09-01",
        "total_completions": 20, "created_at": "2026-06-01T00:00:00+00:00"}}, "goals": {}}))
    rv = ArtemisTracker(p).weekly_review(today=d(1))
    assert rv["habits"][0]["pct"] is None and rv["overall_pct"] is None


# --- engine intents -----------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    e = ArtemisEngine(tracker=ArtemisTracker(tmp_path / "a.json"), llm=NoLLM())
    e.handle("add_habit", {"name": "morning run"}, {})
    e.handle("add_habit", {"name": "reading"}, {})
    return e


def test_can_handle_new_intents(engine):
    for i in ("set_habit_grace", "pause_habit", "resume_habit", "weekly_habit_review"):
        assert engine.can_handle(i)


def test_pause_by_raw_query_with_weeks(engine):
    r = engine.handle("pause_habit", {}, {"raw_query": "pause my running habit for two weeks"})
    assert "morning run" in r["response"] and "Paused" in r["response"]
    assert engine.tracker.get_habit("morning run").paused_on(date.today())


def test_pause_unknown_and_ambiguous(engine):
    assert "don't have" in engine.handle("pause_habit", {"name": "yoga"}, {})["response"]
    assert "Which habit" in engine.handle("pause_habit", {}, {"raw_query": "pause it"})["response"]


def test_resume_when_not_paused(engine):
    assert "isn't paused" in engine.handle("resume_habit", {"name": "reading"}, {})["response"]


def test_set_grace_from_entities_and_text(engine):
    engine.handle("set_habit_grace", {"name": "reading", "days": "2"}, {})
    assert engine.tracker.get_habit("reading").grace_days == 2
    engine.handle("set_habit_grace", {}, {"raw_query": "turn off the grace period for reading"})
    assert engine.tracker.get_habit("reading").grace_days == 0
    assert "How many" in engine.handle("set_habit_grace", {"name": "reading"}, {})["response"]
    assert "between 0 and" in engine.handle("set_habit_grace", {"name": "reading", "days": "30"}, {})["response"]


def test_complete_mentions_grace_and_welcome_back(engine):
    t = engine.tracker
    t.set_habit_grace("reading", 1)
    today = date.today()
    for n in (4, 3, 2):                       # done 4, 3, 2 days ago; yesterday missed
        t.complete_habit("reading", today=today - timedelta(days=n))
    r = engine.handle("complete_habit", {"name": "reading"}, {})
    assert "grace" in r["response"]


def test_weekly_review_responses(engine):
    today = date.today()
    for n in (2, 1, 0):
        engine.tracker.complete_habit("morning run", today=today - timedelta(days=n))
    out = engine.handle("weekly_habit_review", {}, {})["response"]
    assert "consistent" in out


def test_list_habits_shows_pause_and_grace(engine):
    engine.tracker.set_habit_grace("reading", 2)
    engine.handle("pause_habit", {"name": "morning run"}, {})
    out = engine.handle("list_habits", {}, {})["response"]
    assert "paused" in out and "2-day grace" in out
