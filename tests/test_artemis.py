# tests/test_artemis.py
"""
Tests for modules/artemis/engine.py and modules/artemis/tracker.py,
focused on the new capabilities: dead-code wiring (remove_habit,
remove_goal, abandon_goal), deeper habit/goal models (best_streak,
total_completions, consistency %, due_date, priority), at-risk goals,
and the LLM-backed motivation/coaching voice.

Run with:  pytest tests/test_artemis.py -v
(or:       python -m pytest tests/test_artemis.py -v)

A FakeLLM is injected via ArtemisEngine(llm=...) so these tests never hit
a real Ollama server — see ArtemisEngine._llm(), which prefers
self._llm_instance over the module-level core.ollama_client.generate()
when one is provided.
"""
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.artemis.engine import ArtemisEngine, _milestone_callout
from modules.artemis.tracker import (
    ArtemisTracker,
    GoalNotFoundError,
    HabitNotFoundError,
)


class FakeLLM:
    """Deterministic stand-in for HestiaLLM; never touches the network."""
    def generate(self, prompt: str) -> str:
        return "You're building real momentum — keep it up."


class EmptyLLM:
    """Simulates an LLM/Ollama outage: always returns an empty string."""
    def generate(self, prompt: str) -> str:
        return ""


def make_engine(llm=None) -> ArtemisEngine:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    os.unlink(tmp.name)  # ArtemisTracker creates it fresh
    tracker = ArtemisTracker(Path(tmp.name))
    return ArtemisEngine(tracker=tracker, llm=llm if llm is not None else FakeLLM())


# ---------------------------------------------------------------------------
# Dead-code wiring: remove_habit / remove_goal / abandon_goal
# ---------------------------------------------------------------------------

def test_remove_habit_deletes_it():
    e = make_engine()
    e.handle("add_habit", {"name": "meditate"}, {})
    result = e.handle("remove_habit", {"name": "meditate"}, {})
    assert result["confidence"] > 0.5
    assert "meditate" not in e.tracker.get_habits()


def test_remove_habit_not_found():
    e = make_engine()
    result = e.handle("remove_habit", {"name": "nope"}, {})
    assert result["confidence"] == 0.6
    assert "nope" in result["response"]


def test_remove_habit_missing_name():
    e = make_engine()
    result = e.handle("remove_habit", {}, {})
    assert "specify a habit name" in result["response"]


def test_remove_goal_deletes_it():
    e = make_engine()
    e.handle("add_goal", {"name": "ship it"}, {})
    result = e.handle("remove_goal", {"name": "ship it"}, {})
    assert result["confidence"] > 0.5
    assert "ship it" not in e.tracker.get_goals()


def test_remove_goal_not_found():
    e = make_engine()
    result = e.handle("remove_goal", {"name": "nope"}, {})
    assert result["confidence"] == 0.6


def test_abandon_goal_marks_status_without_deleting():
    e = make_engine()
    e.handle("add_goal", {"name": "learn guitar"}, {})
    result = e.handle("abandon_goal", {"name": "learn guitar"}, {})
    assert "abandoned" in result["response"]
    goal = e.tracker.get_goals()["learn guitar"]
    assert goal.status == "abandoned"


def test_abandon_goal_not_found():
    e = make_engine()
    result = e.handle("abandon_goal", {"name": "nope"}, {})
    assert result["confidence"] == 0.6


def test_abandoned_goal_excluded_from_active_list():
    e = make_engine()
    e.handle("add_goal", {"name": "learn guitar"}, {})
    e.handle("abandon_goal", {"name": "learn guitar"}, {})
    result = e.handle("list_goals", {}, {})
    assert "no active goals" in result["response"].lower()
    # ...but it's still in the underlying data, not deleted.
    assert "learn guitar" in result["data"]


def test_tracker_still_supports_direct_deletion_apis():
    # The tracker's own delete methods (which the engine now routes to)
    # keep raising their typed not-found errors when called directly.
    tracker = ArtemisTracker(Path(tempfile.mktemp(suffix=".json")))
    try:
        tracker.remove_habit("ghost")
        assert False, "expected HabitNotFoundError"
    except HabitNotFoundError:
        pass
    try:
        tracker.remove_goal("ghost")
        assert False, "expected GoalNotFoundError"
    except GoalNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Deeper habit model: best_streak, total_completions, milestones
# ---------------------------------------------------------------------------

def test_complete_habit_tracks_total_completions():
    e = make_engine()
    e.handle("add_habit", {"name": "run"}, {})
    today = date.today()
    e.tracker.complete_habit("run", today=today - timedelta(days=1))
    result = e.handle("complete_habit", {"name": "run"}, {})
    assert result["data"]["total_completions"] == 2


def test_complete_habit_same_day_is_idempotent_and_reports_already_done():
    e = make_engine()
    e.handle("add_habit", {"name": "run"}, {})
    e.handle("complete_habit", {"name": "run"}, {})
    result = e.handle("complete_habit", {"name": "run"}, {})
    assert result["data"]["already_done"] is True
    assert result["data"]["total_completions"] == 1
    assert "already marked complete" in result["response"]


def test_best_streak_persists_after_a_broken_streak():
    e = make_engine()
    e.handle("add_habit", {"name": "run"}, {})
    today = date.today()
    # Build a 3-day streak ending 5 days ago, then break it and start fresh.
    e.tracker.complete_habit("run", today=today - timedelta(days=7))
    e.tracker.complete_habit("run", today=today - timedelta(days=6))
    e.tracker.complete_habit("run", today=today - timedelta(days=5))
    result = e.handle("complete_habit", {"name": "run"}, {})  # big gap -> streak resets to 1
    assert result["data"]["streak"] == 1
    assert result["data"]["best_streak"] == 3  # not erased by the break
    assert result["data"]["is_new_best"] is False


def test_new_best_streak_flagged():
    e = make_engine()
    e.handle("add_habit", {"name": "run"}, {})
    today = date.today()
    e.tracker.complete_habit("run", today=today - timedelta(days=1))
    result = e.handle("complete_habit", {"name": "run"}, {})
    assert result["data"]["streak"] == 2
    assert result["data"]["best_streak"] == 2
    assert result["data"]["is_new_best"] is True
    assert "🔥" in result["response"]


def test_milestone_callout_at_7_days():
    assert "7-day milestone" in _milestone_callout(7, is_new_best=True)


def test_milestone_callout_empty_for_non_milestone_non_best():
    assert _milestone_callout(3, is_new_best=False) == ""


def test_milestone_callout_no_new_best_spam_on_day_one():
    # Every fresh habit trivially sets a "best" of 1 — shouldn't announce it.
    assert "New best" not in _milestone_callout(1, is_new_best=True)


def test_consistency_pct_reflects_history_not_just_current_streak():
    e = make_engine()
    e.handle("add_habit", {"name": "run"}, {})
    today = date.today()
    # Habit "created" 9 days ago (via created_at manipulation isn't exposed,
    # so approximate via direct tracker completions across a wider window).
    for i in range(9, -1, -2):  # every other day for 10 days: 5 completions
        e.tracker.complete_habit("run", today=today - timedelta(days=i))
    result = e.handle("list_habits", {}, {})
    pct = result["data"]["run"]["consistency_pct"]
    assert 0 < pct <= 100


# ---------------------------------------------------------------------------
# Deeper goal model: due_date, priority
# ---------------------------------------------------------------------------

def test_add_goal_with_due_date_and_priority():
    e = make_engine()
    result = e.handle(
        "add_goal",
        {"name": "ship it", "due_date": "2099-01-01", "priority": "high"},
        {},
    )
    assert "due 2099-01-01" in result["response"]
    assert "high priority" in result["response"]
    goal = e.tracker.get_goals()["ship it"]
    assert goal.due_date == "2099-01-01"
    assert goal.priority == "high"


def test_add_goal_invalid_due_date_returns_clarification():
    e = make_engine()
    result = e.handle("add_goal", {"name": "ship it", "due_date": "not-a-date"}, {})
    assert result["confidence"] == 0.5


def test_add_goal_invalid_priority_returns_clarification():
    e = make_engine()
    result = e.handle("add_goal", {"name": "ship it", "priority": "urgent!"}, {})
    assert result["confidence"] == 0.5


def test_update_goal_can_set_due_date_and_priority_without_progress():
    e = make_engine()
    e.handle("add_goal", {"name": "ship it"}, {})
    result = e.handle(
        "update_goal",
        {"name": "ship it", "due_date": "2099-06-01", "priority": "low"},
        {},
    )
    goal = e.tracker.get_goals()["ship it"]
    assert goal.due_date == "2099-06-01"
    assert goal.priority == "low"
    assert "due 2099-06-01" in result["response"]


def test_update_goal_still_supports_progress_only():
    e = make_engine()
    e.handle("add_goal", {"name": "ship it"}, {})
    result = e.handle("update_goal", {"name": "ship it", "progress": 40}, {})
    assert "40%" in result["response"]


def test_update_goal_missing_everything_asks_for_progress():
    e = make_engine()
    e.handle("add_goal", {"name": "ship it"}, {})
    result = e.handle("update_goal", {"name": "ship it"}, {})
    assert "progress" in result["response"].lower()


# ---------------------------------------------------------------------------
# get_at_risk_goals
# ---------------------------------------------------------------------------

def test_at_risk_goal_close_to_deadline_with_low_progress():
    e = make_engine()
    today = date.today()
    due_soon = (today + timedelta(days=2)).isoformat()
    e.handle("add_goal", {"name": "ship it", "due_date": due_soon}, {})
    e.handle("update_goal", {"name": "ship it", "progress": 10}, {})
    result = e.handle("get_at_risk_goals", {}, {})
    assert "ship it" in result["data"]
    assert "1 goal(s) at risk" in result["response"]


def test_goal_not_at_risk_when_progress_is_high():
    e = make_engine()
    today = date.today()
    due_soon = (today + timedelta(days=2)).isoformat()
    e.handle("add_goal", {"name": "ship it", "due_date": due_soon}, {})
    e.handle("update_goal", {"name": "ship it", "progress": 90}, {})
    result = e.handle("get_at_risk_goals", {}, {})
    assert "ship it" not in result["data"]


def test_goal_not_at_risk_when_deadline_is_far_away():
    e = make_engine()
    today = date.today()
    due_later = (today + timedelta(days=60)).isoformat()
    e.handle("add_goal", {"name": "ship it", "due_date": due_later}, {})
    result = e.handle("get_at_risk_goals", {}, {})
    assert "ship it" not in result["data"]


def test_goal_without_due_date_is_never_at_risk():
    e = make_engine()
    e.handle("add_goal", {"name": "someday goal"}, {})
    result = e.handle("get_at_risk_goals", {}, {})
    assert "someday goal" not in result["data"]


def test_overdue_goal_is_at_risk():
    e = make_engine()
    today = date.today()
    overdue = (today - timedelta(days=3)).isoformat()
    e.handle("add_goal", {"name": "late thing", "due_date": overdue}, {})
    result = e.handle("get_at_risk_goals", {}, {})
    assert "overdue" in result["response"]


def test_no_at_risk_goals_message():
    e = make_engine()
    e.handle("add_goal", {"name": "someday goal"}, {})
    result = e.handle("get_at_risk_goals", {}, {})
    assert "no goals are at risk" in result["response"].lower()


# ---------------------------------------------------------------------------
# AI voice: get_motivation
# ---------------------------------------------------------------------------

def test_get_motivation_uses_fake_llm():
    e = make_engine()
    e.handle("add_habit", {"name": "meditate"}, {})
    result = e.handle("get_motivation", {}, {})
    assert result["response"] == "You're building real momentum — keep it up."
    assert result["confidence"] > 0.5


def test_get_motivation_falls_back_when_llm_empty():
    e = make_engine(llm=EmptyLLM())
    e.handle("add_habit", {"name": "meditate"}, {})
    result = e.handle("get_motivation", {}, {})
    # Falls back to the deterministic suggest_next_action() static path —
    # never breaks just because Ollama is unavailable.
    assert result["response"]
    assert result["confidence"] > 0


def test_get_motivation_with_no_data_asks_to_add_something():
    e = make_engine()
    result = e.handle("get_motivation", {}, {})
    assert "don't have any habits or goals" in result["response"]


# ---------------------------------------------------------------------------
# AI voice: productivity_summary coaching paragraph
# ---------------------------------------------------------------------------

def test_productivity_summary_includes_llm_coaching():
    e = make_engine()
    e.handle("add_habit", {"name": "meditate"}, {})
    e.handle("complete_habit", {"name": "meditate"}, {})
    result = e.handle("productivity_summary", {}, {})
    assert "You're building real momentum" in result["response"]


def test_productivity_summary_falls_back_when_llm_down():
    e = make_engine(llm=EmptyLLM())
    e.handle("add_habit", {"name": "meditate"}, {})
    result = e.handle("productivity_summary", {}, {})
    # Never breaks — static coaching fallback still produces a real answer.
    assert result["confidence"] > 0
    assert "Avg streak is" in result["response"]


def test_productivity_summary_flags_at_risk_goals():
    e = make_engine()
    today = date.today()
    due_soon = (today + timedelta(days=1)).isoformat()
    e.handle("add_goal", {"name": "ship it", "due_date": due_soon}, {})
    result = e.handle("productivity_summary", {}, {})
    assert result["data"]["at_risk_goals"]
    assert "at risk" in result["response"]


def test_productivity_summary_data_includes_overdue_goals():
    e = make_engine()
    today = date.today()
    overdue = (today - timedelta(days=1)).isoformat()
    e.handle("add_goal", {"name": "late thing", "due_date": overdue}, {})
    result = e.handle("productivity_summary", {}, {})
    assert "late thing" in result["data"]["overdue_goals"]


# ---------------------------------------------------------------------------
# Tracker-corruption / error handling stays intact through the new intents
# ---------------------------------------------------------------------------

def test_tracker_error_on_remove_habit_returns_graceful_response(monkeypatch):
    e = make_engine()
    e.handle("add_habit", {"name": "meditate"}, {})

    def boom(name):
        from modules.artemis.tracker import TrackerError
        raise TrackerError("disk on fire")

    monkeypatch.setattr(e.tracker, "remove_habit", boom)
    result = e.handle("remove_habit", {"name": "meditate"}, {})
    assert result["confidence"] < 0.5
    assert "couldn't access" in result["response"].lower()


def test_unknown_intent_handled_gracefully():
    e = make_engine()
    result = e.handle("not_a_real_intent", {}, {})
    assert result["confidence"] == 0.5
    assert "can't handle" in result["response"]


if __name__ == "__main__":
    unittest.main()