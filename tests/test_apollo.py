# tests/test_apollo.py
"""
Tests for modules/apollo/engine.py, focused on the new capabilities:
weight tracking, hydration tracking, health goals, and workout streaks.

Run with:  pytest tests/test_apollo.py -v
(or:       python -m pytest tests/test_apollo.py -v)

A FakeLLM is injected via ApolloEngine(llm=...) so these tests never hit a
real Ollama server — see ApolloEngine._llm(), which prefers self._llm_instance
over the module-level core.ollama_client.generate() when one is provided.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.apollo.engine import ApolloEngine, _compute_streak


class FakeLLM:
    """Deterministic stand-in for HestiaLLM; never touches the network."""
    def generate(self, prompt: str) -> str:
        return "Looking solid — keep it up."


def make_engine() -> ApolloEngine:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return ApolloEngine(db_path=Path(tmp.name), llm=FakeLLM())


# ---------------------------------------------------------------------------
# Weight tracking
# ---------------------------------------------------------------------------

def test_log_weight_kg():
    e = make_engine()
    result = e.handle("log_weight", {"weight": "72.5", "unit": "kg"}, {})
    assert result["confidence"] > 0
    assert result["data"]["weight_kg"] == 72.5
    assert "72.5 kg" in result["response"]


def test_log_weight_lb_converts_to_kg_for_storage():
    e = make_engine()
    result = e.handle("log_weight", {"weight": "154", "unit": "lb"}, {})
    # 154 lb ≈ 69.85 kg
    assert abs(result["data"]["weight_kg"] - 69.85) < 0.1
    assert "154.0 lb" in result["response"]


def test_log_weight_out_of_range_asks_for_clarification():
    e = make_engine()
    result = e.handle("log_weight", {"weight": "5", "unit": "kg"}, {})
    assert result["data"].get("needs_clarification") is True


def test_weight_query_without_data_asks_to_log():
    e = make_engine()
    result = e.handle("log_weight", {"raw_query": "what's my weight trend?"}, {})
    assert result["data"].get("needs_clarification") is True


def test_weight_trend_reported_after_multiple_logs():
    e = make_engine()
    e.handle("log_weight", {"weight": "80", "unit": "kg"}, {})
    e.handle("log_weight", {"weight": "78", "unit": "kg"}, {})
    result = e.handle("log_weight", {"raw_query": "what's my weight trend?"}, {})
    assert "down" in result["response"]


# ---------------------------------------------------------------------------
# Hydration tracking
# ---------------------------------------------------------------------------

def test_log_water_ml():
    e = make_engine()
    result = e.handle("log_water", {"amount": "500"}, {})
    assert result["data"]["amount_ml"] == 500
    assert result["data"]["total_today_ml"] == 500


def test_log_water_glasses_converted_to_ml():
    e = make_engine()
    result = e.handle("log_water", {"glasses": "2"}, {})
    assert result["data"]["amount_ml"] == 500  # 2 * 250ml


def test_water_totals_accumulate_same_day():
    e = make_engine()
    e.handle("log_water", {"amount": "300"}, {})
    result = e.handle("log_water", {"amount": "400"}, {})
    assert result["data"]["total_today_ml"] == 700


def test_water_query_with_no_logs_today():
    e = make_engine()
    result = e.handle("log_water", {"raw_query": "how much water have I had today?"}, {})
    assert result["data"]["total_today_ml"] == 0


def test_log_water_out_of_range_asks_for_clarification():
    e = make_engine()
    result = e.handle("log_water", {"amount": "99999"}, {})
    assert result["data"].get("needs_clarification") is True


# ---------------------------------------------------------------------------
# Health goals
# ---------------------------------------------------------------------------

def test_set_health_goal_workout_frequency():
    e = make_engine()
    result = e.handle("set_health_goal", {"goal_type": "workouts", "target": "4"}, {})
    assert result["data"]["goal_type"] == "workout_frequency"
    assert result["data"]["target"] == 4.0


def test_set_health_goal_unknown_type_asks_for_clarification():
    e = make_engine()
    result = e.handle("set_health_goal", {"goal_type": "flexibility", "target": "4"}, {})
    assert result["data"].get("needs_clarification") is True


def test_goal_progress_with_no_goals():
    e = make_engine()
    result = e.handle("get_goal_progress", {}, {})
    assert result["data"]["goals"] == []


def test_goal_progress_reports_workout_percentage():
    e = make_engine()
    e.handle("set_health_goal", {"goal_type": "workout_frequency", "target": "4"}, {})
    e.handle("log_workout", {"type": "run", "duration": "30"}, {})
    e.handle("log_workout", {"type": "run", "duration": "30"}, {})
    result = e.handle("get_goal_progress", {}, {})
    goal = next(g for g in result["data"]["goals"] if g["goal_type"] == "workout_frequency")
    assert goal["current"] == 2.0
    assert goal["target"] == 4.0


def test_water_goal_progress_shown_on_log():
    e = make_engine()
    e.handle("set_health_goal", {"goal_type": "water", "target": "2000"}, {})
    result = e.handle("log_water", {"amount": "1000"}, {})
    assert "50%" in result["response"]


# ---------------------------------------------------------------------------
# Streaks — _compute_streak is pure, so test it directly without a DB
# ---------------------------------------------------------------------------

def test_compute_streak_empty():
    assert _compute_streak([]) == 0


def test_compute_streak_consecutive_days():
    from datetime import date, timedelta
    today = date.today()
    dates = [(today - timedelta(days=i)).isoformat() for i in range(3)]
    assert _compute_streak(dates) == 3


def test_compute_streak_broken_by_gap():
    from datetime import date, timedelta
    today = date.today()
    # today and yesterday logged, then a gap before day 3 -> streak of 2
    dates = [
        today.isoformat(),
        (today - timedelta(days=1)).isoformat(),
        (today - timedelta(days=3)).isoformat(),
    ]
    assert _compute_streak(dates) == 2


def test_compute_streak_still_counts_if_today_not_yet_logged():
    from datetime import date, timedelta
    today = date.today()
    # Yesterday and the day before logged, nothing today yet -> streak of 2,
    # not reset to 0 just because today hasn't happened yet.
    dates = [
        (today - timedelta(days=1)).isoformat(),
        (today - timedelta(days=2)).isoformat(),
    ]
    assert _compute_streak(dates) == 2


def test_workout_log_reports_streak_in_data():
    e = make_engine()
    result = e.handle("log_workout", {"type": "run", "duration": "20"}, {})
    assert result["data"]["streak"] == 1


if __name__ == "__main__":
    unittest.main()