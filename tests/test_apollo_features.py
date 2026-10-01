# tests/test_apollo_features.py
"""
Engine-level tests for the Apollo improvement backlog (#111-#120, #126, #161,
#149 hooks, #183 data). They use a real temp-file SQLite DB and inject the
clock, so nothing depends on wall-clock time except that SQLite's own
``datetime('now')`` window queries (used by the DB layer) line up with
the real current date, which is why timestamps are built relative to it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.apollo.db import ApolloDB
from modules.apollo.engine import ApolloEngine

NOW = datetime.now(timezone.utc).replace(hour=15, minute=0, second=0, microsecond=0)


def ts(days_ago: int, hour: int = 8) -> str:
    d = (NOW - timedelta(days=days_ago)).replace(hour=hour)
    return d.strftime("%Y-%m-%d %H:%M:%S")


class FakeLLM:
    def generate(self, *a, **k):
        return "ok"


def make(tmp_path, **config):
    return ApolloEngine(
        ollama_cfg={}, db_path=Path(tmp_path) / "apollo.db",
        llm=FakeLLM(), config=config or None,
    )


def run(engine, intent, **entities):
    return engine.handle(intent, entities, {})


# ------------------------------------------------------------- migrations

class TestMigrations:
    def test_old_database_is_upgraded_without_losing_rows(self, tmp_path):
        path = str(tmp_path / "old.db")
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE sleep_logs (id INTEGER PRIMARY KEY AUTOINCREMENT,
                hours REAL NOT NULL, quality TEXT, notes TEXT,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE health_goals (goal_type TEXT PRIMARY KEY,
                target_value REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            INSERT INTO sleep_logs (hours, quality, notes) VALUES (7.0, 'ok', 'legacy');
            INSERT INTO health_goals (goal_type, target_value) VALUES ('weight_target', 70);
            """
        )
        conn.commit()
        conn.close()

        db = ApolloDB(path)
        assert db.schema_version == 1
        rows = db.get_sleep(7)
        assert rows[0]["notes"] == "legacy" and rows[0]["bed_time"] is None
        assert db.get_goal("weight_target")["start_value"] is None
        for table in ("profile", "meal_logs", "pain_logs", "step_logs", "reminder_state"):
            db._conn.execute(f"SELECT 1 FROM {table} LIMIT 1")

    def test_migration_is_idempotent(self, tmp_path):
        path = str(tmp_path / "a.db")
        ApolloDB(path).log_sleep(7, "good", "n", bed_time="23:00")
        again = ApolloDB(path)
        assert again.schema_version == 1
        assert again.get_sleep(7)[0]["bed_time"] == "23:00"

    def test_half_applied_migration_recovers(self, tmp_path):
        path = str(tmp_path / "a.db")
        db = ApolloDB(path)
        db._conn.execute("PRAGMA user_version = 0")   # pretend it never finished
        db._conn.commit()
        assert ApolloDB(path).schema_version == 1      # re-runs cleanly (columns exist)


# ------------------------------------------------------------------ #113

class TestUnits:
    def test_set_units_and_persist(self, tmp_path):
        e = make(tmp_path)
        r = run(e, "set_units", raw_query="switch me to pounds")
        assert "weight in lb" in r["response"]
        assert make(tmp_path)._weight_unit() == "lb"

    def test_set_units_clarifies_when_unclear(self, tmp_path):
        r = run(make(tmp_path), "set_units", raw_query="change stuff")
        assert r["data"].get("needs_clarification") or "Which units" in r["response"]

    def test_metric_and_imperial_keywords(self, tmp_path):
        e = make(tmp_path)
        run(e, "set_units", raw_query="go imperial")
        assert (e._weight_unit(), e._water_unit()) == ("lb", "oz")
        run(e, "set_units", raw_query="back to metric")
        assert (e._weight_unit(), e._water_unit()) == ("kg", "ml")

    def test_profile_unit_applies_when_message_has_none(self, tmp_path):
        e = make(tmp_path)
        run(e, "set_units", weight_unit="lb")
        r = run(e, "log_weight", weight=154)
        assert r["data"]["weight_kg"] == pytest.approx(69.85, abs=0.01)
        assert "154.0 lb" in r["response"]

    def test_explicit_unit_beats_profile(self, tmp_path):
        e = make(tmp_path)
        run(e, "set_units", weight_unit="lb")
        r = run(e, "log_weight", weight=70, unit="kg")
        assert r["data"]["weight_kg"] == 70
        r = run(e, "log_weight", weight=72, raw_query="log my weight as 72 kg")
        assert r["data"]["weight_kg"] == 72

    def test_default_is_unchanged_kg(self, tmp_path):
        r = run(make(tmp_path), "log_weight", weight=70)
        assert r["data"]["weight_kg"] == 70 and "70.0 kg" in r["response"]

    def test_water_oz_profile(self, tmp_path):
        e = make(tmp_path)
        run(e, "set_units", water_unit="oz")
        r = run(e, "log_water", amount=16)
        assert r["data"]["amount_ml"] == pytest.approx(473, abs=1)
        assert "16 oz" in r["response"]

    def test_weight_goal_in_pounds_is_stored_as_kg(self, tmp_path):
        e = make(tmp_path)
        run(e, "set_units", weight_unit="lb")
        r = run(e, "set_health_goal", goal_type="weight", target=150)
        assert r["data"]["target"] == pytest.approx(68.04, abs=0.01)
        assert "150 lb target weight" in r["response"]


# ------------------------------------------------------------------ #111

class TestSleepQuality:
    def test_window_in_message_logs_hours_and_times(self, tmp_path):
        e = make(tmp_path)
        r = run(e, "track_sleep", raw_query="slept 11pm to 6:30am")
        assert r["data"]["hours"] == 7.5
        assert (r["data"]["bed_time"], r["data"]["wake_time"]) == ("23:00", "06:30")
        assert "Sleep quality score" in r["response"]

    def test_missing_times_are_called_out(self, tmp_path):
        r = run(make(tmp_path), "track_sleep", hours=7, quality="good")
        assert "didn't have bed and wake times" in r["response"]
        assert r["data"]["quality_score"] is not None

    def test_consistency_appears_after_three_timed_nights(self, tmp_path):
        e = make(tmp_path)
        for d in (3, 2, 1):
            e.db.log_sleep(7.5, "good", "", bed_time="23:00", wake_time="06:30",
                           logged_at=ts(d))
        r = run(e, "track_sleep", raw_query="slept 11pm to 6:30am", quality="good")
        assert "consistency 100" in r["response"]

    def test_get_sleep_quality_explains_what_is_missing(self, tmp_path):
        e = make(tmp_path)
        assert "sleep logged" in run(e, "get_sleep_quality")["response"].lower() or \
            run(e, "get_sleep_quality")["data"].get("needs_clarification")
        run(e, "track_sleep", hours=7)
        r = run(e, "get_sleep_quality")
        assert "Consistency isn't counted yet" in r["response"]

    def test_existing_sleep_query_still_works(self, tmp_path):
        e = make(tmp_path)
        run(e, "track_sleep", hours=6.5)
        r = run(e, "track_sleep", raw_query="how did I sleep last night?")
        assert "6.5" in r["response"]


# ------------------------------------------------------------------ #112

class TestCorrelations:
    def _seed(self, e, short_mood, long_mood, n=5):
        for i in range(n):          # short-sleep days
            e.db.log_sleep(5.0, "", "", logged_at=ts(i + 1))
            e.db.log_mood(short_mood, "", logged_at=ts(i + 1, 20))
        for i in range(n):          # long-sleep days
            e.db.log_sleep(8.0, "", "", logged_at=ts(i + 1 + n))
            e.db.log_mood(long_mood, "", logged_at=ts(i + 1 + n, 20))

    def test_refuses_when_data_is_thin(self, tmp_path):
        e = make(tmp_path)
        e.db.log_mood("good", "")
        r = e._get_correlations({}, NOW)
        assert r["data"]["status"] == "insufficient"
        assert "can't say anything reliable" in r["response"]

    def test_reports_sleep_mood_pattern_with_n(self, tmp_path):
        e = make(tmp_path)
        self._seed(e, "tired", "good")
        r = e._get_correlations({}, NOW)
        assert "n=5" in r["response"]
        assert "lower on the first" in r["response"]
        assert "not proof" in r["response"]

    def test_workout_days_comparison(self, tmp_path):
        e = make(tmp_path)
        self._seed(e, "good", "good")
        for i in range(1, 6):
            e.db.log_workout("run", 30, "", logged_at=ts(i))
        r = e._get_correlations({}, NOW)
        # workout days and non-workout days are all +1 here: no difference, not a finding
        assert "no clear difference" in r["response"]

    def test_min_sample_is_configurable(self, tmp_path):
        e = make(tmp_path, min_sample=8)
        self._seed(e, "tired", "good", n=5)
        r = e._get_correlations({}, NOW)["response"]
        assert "at least 16 different days" in r      # 2 x min_sample


# ------------------------------------------------------------------ #116

class TestWorkoutStreaks:
    def test_per_type_days_and_weeks(self, tmp_path):
        e = make(tmp_path)
        for i in range(0, 4):
            e.db.log_workout("Run", 30, "", logged_at=ts(i))
        e.db.log_workout("yoga", 30, "", logged_at=ts(20))
        r = e._get_workout_streaks({"min_sessions": 1}, NOW)
        assert r["data"]["streaks"]["run"]["days"] == 4
        assert r["data"]["streaks"]["yoga"]["days"] == 0
        assert "run: 4 day(s) in a row" in r["response"]

    def test_filter_by_type(self, tmp_path):
        e = make(tmp_path)
        e.db.log_workout("run", 30, "", logged_at=ts(0))
        e.db.log_workout("yoga", 30, "", logged_at=ts(0))
        r = e._get_workout_streaks({"type": "yoga"}, NOW)
        assert list(r["data"]["streaks"]) == ["yoga"]

    def test_empty(self, tmp_path):
        assert "No workouts" in make(tmp_path)._get_workout_streaks({}, NOW)["response"]


# ------------------------------------------------------------------ #114

class TestMeals:
    def test_manual_kcal_is_logged_as_entered(self, tmp_path):
        e = make(tmp_path)
        r = e._log_meal({"food": "dal and rice", "kcal": 450}, NOW)
        assert "as you entered it" in r["response"]
        assert r["data"]["estimated"] is False and r["data"]["today_kcal"] == 450

    def test_lookup_is_labelled_an_estimate(self, tmp_path):
        e = make(tmp_path)
        e._food_lookup = lambda name, limit=3: [
            {"name": "Paneer", "calories_kcal_100g": 300, "protein_g_100g": 18}
        ]
        r = e._log_meal({"food": "paneer", "grams": 150}, NOW)
        assert r["data"]["kcal"] == 450 and r["data"]["estimated"] is True
        assert "estimate" in r["response"] and "rough" in r["response"]
        assert r["data"]["protein_g"] == pytest.approx(27)

    def test_assumed_100g_is_disclosed(self, tmp_path):
        e = make(tmp_path)
        e._food_lookup = lambda n, limit=3: [{"calories_kcal_100g": 200}]
        assert "assumed 100 g" in e._log_meal({"food": "oats"}, NOW)["response"]

    def test_failed_lookup_asks_instead_of_inventing(self, tmp_path):
        e = make(tmp_path)
        def boom(*a, **k): raise RuntimeError("offline")
        e._food_lookup = boom
        r = e._log_meal({"food": "mystery stew"}, NOW)
        assert r["data"].get("needs_clarification") or "Tell me the kcal" in r["response"]
        assert e.db.get_meals(1) == []

    def test_absurd_kcal_rejected(self, tmp_path):
        r = make(tmp_path)._log_meal({"food": "x", "kcal": 90000}, NOW)
        assert "doesn't look right" in r["response"]

    def test_no_target_shown_unless_user_set_one(self, tmp_path):
        e = make(tmp_path)
        r = e._log_meal({"food": "toast", "kcal": 200}, NOW)
        assert "goal" not in r["response"].lower()
        e.db.set_goal("calories_kcal", 2000)
        r = e._log_meal({"food": "toast", "kcal": 200}, NOW)
        assert "2000 kcal/day" in r["response"]

    def test_daily_total_and_summary(self, tmp_path):
        e = make(tmp_path)
        e._log_meal({"food": "a", "kcal": 300, "protein": 10}, NOW)
        e._log_meal({"food": "b", "kcal": 500, "protein": 20}, NOW)
        r = e._get_meal_summary({}, NOW)
        assert "Today: 800 kcal, 30 g protein" in r["response"]
        assert "estimates" in r["response"]

    def test_calorie_goal_below_floor_refused(self, tmp_path):
        e = make(tmp_path)
        r = run(e, "set_health_goal", goal_type="calories", target=900)
        assert r["data"]["refused"] and "dietitian" in r["response"]
        assert e.db.get_goal("calories_kcal") is None

    def test_calorie_goal_accepted_at_normal_level(self, tmp_path):
        e = make(tmp_path)
        run(e, "set_health_goal", goal_type="calories", target=2200)
        assert e.db.get_goal("calories_kcal")["target_value"] == 2200


# ------------------------------------------------------------------ #115

class TestHydration:
    def test_on_demand_status(self, tmp_path):
        e = make(tmp_path)
        r = e._hydration_status({}, NOW)            # 15:00, nothing drunk
        assert "0 ml of 2000 ml" in r["response"] and "behind" in r["response"]
        assert "default" in r["response"]

    def test_nudges_when_behind_then_respects_cooldown(self, tmp_path):
        e = make(tmp_path)
        e.db.set_goal("water_ml", 2000)
        assert "Hydration check" in e.check_hydration_nudge(NOW)
        assert e.check_hydration_nudge(NOW + timedelta(minutes=30)) is None   # cooldown
        assert e.check_hydration_nudge(NOW + timedelta(hours=2)) is not None

    def test_silent_when_on_pace(self, tmp_path):
        e = make(tmp_path)
        e.db.set_goal("water_ml", 2000)
        e.db.log_water(1200)
        assert e.check_hydration_nudge(NOW) is None

    def test_quiet_hours(self, tmp_path):
        e = make(tmp_path)
        e.db.set_goal("water_ml", 2000)
        assert e.check_hydration_nudge(NOW.replace(hour=23)) is None
        assert e.check_hydration_nudge(NOW.replace(hour=5)) is None

    def test_daily_cap(self, tmp_path):
        e = make(tmp_path, hydration={"daily_cap": 2, "cooldown_minutes": 1})
        e.db.set_goal("water_ml", 4000)
        sent = [e.check_hydration_nudge(NOW + timedelta(hours=i)) for i in range(5)]
        assert sum(1 for s in sent if s) == 2

    def test_does_not_nag_people_who_dont_track_water(self, tmp_path):
        assert make(tmp_path).check_hydration_nudge(NOW) is None

    def test_can_be_disabled(self, tmp_path):
        e = make(tmp_path, hydration={"enabled": False})
        e.db.set_goal("water_ml", 2000)
        assert e.check_hydration_nudge(NOW) is None

    def test_state_survives_restart(self, tmp_path):
        e = make(tmp_path)
        e.db.set_goal("water_ml", 2000)
        assert e.check_hydration_nudge(NOW)
        e2 = make(tmp_path)
        assert e2.check_hydration_nudge(NOW + timedelta(minutes=10)) is None


# ------------------------------------------------------------------ #118

class TestWeeklySummary:
    def _seed(self, e):
        for i in range(1, 6):
            e.db.log_sleep(7.0, "good", "", logged_at=ts(i))
        e.db.log_workout("run", 30, "", logged_at=ts(1))
        e.db.log_workout("run", 30, "", logged_at=ts(3))
        e.db.log_weight(80.0, "", logged_at=ts(6))
        e.db.log_weight(79.6, "", logged_at=ts(0))
        e.db.log_water(2200, logged_at=ts(1))
        e.db.log_water(800, logged_at=ts(2))

    def test_deterministic_content(self, tmp_path):
        e = make(tmp_path)
        self._seed(e)
        text, has = e.weekly_summary_text(NOW)
        assert has
        assert "5 logged night(s)" in text and "2 session(s)" in text
        assert "80.0 kg to 79.6 kg" in text
        assert "1 of 2 day(s)" in text
        assert e.weekly_summary_text(NOW)[0] == text          # no randomness

    def test_empty_week_has_no_data(self, tmp_path):
        assert make(tmp_path).weekly_summary_text(NOW)[1] is False
        r = make(tmp_path)._get_weekly_summary({}, NOW)
        assert "nothing to summarise" in r["response"]

    def test_sent_once_per_week_and_survives_restart(self, tmp_path):
        cfg = {"weekly_summary": {"weekday": NOW.weekday(), "hour": 0}}
        e = make(tmp_path, **cfg)
        self._seed(e)
        assert e.check_weekly_summary(NOW) is not None
        assert e.check_weekly_summary(NOW) is None
        e2 = make(tmp_path, **cfg)
        assert e2.check_weekly_summary(NOW) is None          # marker is in the DB
        assert e2.db.get_state("weekly_summary_sent")

    def test_not_sent_before_its_day_or_with_no_data(self, tmp_path):
        later = {"weekly_summary": {"weekday": (NOW.weekday() + 1) % 7 or 6, "hour": 0}}
        if NOW.weekday() < 6:
            e = make(tmp_path, weekly_summary={"weekday": 6, "hour": 0})
            self._seed(e)
            assert e.check_weekly_summary(NOW) is None
        e = make(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path,
                 weekly_summary={"weekday": NOW.weekday(), "hour": 0})
        assert e.check_weekly_summary(NOW) is None           # due, but nothing to say

    def test_can_be_disabled(self, tmp_path):
        e = make(tmp_path, weekly_summary={"enabled": False})
        self._seed(e)
        assert e.check_weekly_summary(NOW) is None


# ------------------------------------------------------------------ #117

class TestPainHandlers:
    def test_logs_with_disclaimer(self, tmp_path):
        e = make(tmp_path)
        r = e._log_pain({"raw_query": "my left knee hurts, 4 out of 10"}, NOW)
        assert r["data"]["area"] == "knee" and r["data"]["severity"] == 4
        assert "can't diagnose" in r["response"]
        assert not r["data"]["clinician_advised"]

    def test_red_flag_phrase_triggers_clinician_message(self, tmp_path):
        e = make(tmp_path)
        r = e._log_pain({"area": "chest", "severity": 5,
                         "raw_query": "chest pain when climbing stairs"}, NOW)
        assert r["data"]["red_flags"] and r["data"]["clinician_advised"]
        assert "doctor or urgent care" in r["response"]

    def test_asks_for_missing_area_or_severity(self, tmp_path):
        e = make(tmp_path)
        assert "Which part" in e._log_pain({"raw_query": "it hurts"}, NOW)["response"]
        assert "how bad" in e._log_pain({"area": "knee"}, NOW)["response"].lower()
        assert "0 (none) to 10" in e._log_pain({"area": "knee", "severity": 14}, NOW)["response"]
        assert e.db.get_pain(30) == []

    def test_trend_and_worsening_advice(self, tmp_path):
        e = make(tmp_path)
        for i, s in enumerate([3, 3, 4]):
            e.db.log_pain("knee", s, logged_at=ts(9 + i))
        for i, s in enumerate([7, 7, 6]):
            e.db.log_pain("knee", s, logged_at=ts(i))
        r = e._get_pain_trend({}, NOW)
        assert "worse" in r["response"] and "clinician" in r["response"]
        assert "can't diagnose" in r["response"]

    def test_not_enough_data_says_so(self, tmp_path):
        e = make(tmp_path)
        e.db.log_pain("hip", 3, logged_at=ts(0))
        assert "not enough data" in e._get_pain_trend({}, NOW)["response"]

    def test_pain_lasting_weeks(self, tmp_path):
        e = make(tmp_path)
        for d in (0, 11, 23):
            e.db.log_pain("hip", 3, logged_at=ts(d))
        assert "three weeks" in e._get_pain_trend({}, NOW)["response"]


# ------------------------------------------------------------------ #119

class TestStepImport:
    def _engine(self, tmp_path):
        folder = tmp_path / "imports"
        folder.mkdir()
        return make(tmp_path, import_dir=str(folder)), folder

    def test_not_configured(self, tmp_path):
        r = run(make(tmp_path), "import_steps")
        assert r["data"]["configured"] is False

    def test_csv_with_column_detection_and_dedupe(self, tmp_path):
        e, folder = self._engine(tmp_path)
        (folder / "steps.csv").write_text(
            "Start Date,Step Count\n2026-09-01,8000\n2026-09-02,\"10,500\"\nbad,row\n2026-09-03,-5\n"
        )
        r = run(e, "import_steps")
        assert r["data"]["inserted"] == 2 and r["data"]["skipped_rows"] == 2
        r2 = run(e, "import_steps")                        # idempotent
        assert r2["data"]["unchanged"] == 2 and not r2["data"].get("inserted")
        assert {s["day"]: s["steps"] for s in e.db.get_steps(400)} == {
            "2026-09-01": 8000, "2026-09-02": 10500}

    def test_per_record_rows_on_one_day_are_summed(self, tmp_path):
        e, folder = self._engine(tmp_path)
        (folder / "h.csv").write_text("date,steps\n2026-09-01,1000\n2026-09-01,2500\n")
        run(e, "import_steps")
        assert e.db.get_steps(400)[0]["steps"] == 3500

    def test_changed_value_updates(self, tmp_path):
        e, folder = self._engine(tmp_path)
        f = folder / "s.csv"
        f.write_text("date,steps\n2026-09-01,1000\n")
        run(e, "import_steps")
        f.write_text("date,steps\n2026-09-01,1200\n")
        assert run(e, "import_steps")["data"]["updated"] == 1

    @pytest.mark.parametrize("payload", [
        [{"date": "2026-09-01", "steps": 5000}],
        {"data": [{"day": "2026-09-01", "step_count": 5000}]},
        {"2026-09-01": 5000},
    ])
    def test_json_layouts(self, tmp_path, payload):
        e, folder = self._engine(tmp_path)
        (folder / "s.json").write_text(json.dumps(payload))
        assert run(e, "import_steps")["data"]["inserted"] == 1

    def test_path_traversal_cannot_leave_the_folder(self, tmp_path):
        e, folder = self._engine(tmp_path)
        outside = tmp_path / "secret.csv"
        outside.write_text("date,steps\n2026-09-01,1\n")
        r = run(e, "import_steps", file="../secret.csv")
        assert "can't find" in r["response"]
        assert e.db.get_steps(400) == []

    def test_symlink_escape_is_ignored(self, tmp_path):
        e, folder = self._engine(tmp_path)
        outside = tmp_path / "secret.csv"
        outside.write_text("date,steps\n2026-09-01,1\n")
        try:
            (folder / "link.csv").symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable")
        run(e, "import_steps")
        assert e.db.get_steps(400) == []

    def test_unrecognised_layout_reports_it(self, tmp_path):
        e, folder = self._engine(tmp_path)
        (folder / "x.csv").write_text("foo,bar\n1,2\n")
        r = run(e, "import_steps")
        assert "Skipped 1 row" in r["response"] and "date column" in r["response"]


# ------------------------------------------------------------------ #120

class TestGoalPace:
    def _weights(self, e, pairs):
        for days_ago, kg in pairs:
            e.db.log_weight(kg, "", logged_at=ts(days_ago))

    def test_goal_records_start_and_deadline(self, tmp_path):
        e = make(tmp_path)
        e.db.log_weight(80, "")
        r = run(e, "set_health_goal", goal_type="weight", target=70,
                raw_query="set a weight goal of 70 by 2099-01-01")
        assert r["data"]["deadline"] == "2099-01-01"
        g = e.db.get_goal("weight_target")
        assert g["start_value"] == 80 and g["deadline"] == "2099-01-01"

    def test_past_deadline_rejected(self, tmp_path):
        r = run(make(tmp_path), "set_health_goal", goal_type="weight", target=70,
                deadline="2001-01-01")
        assert "in the past" in r["response"]

    def test_relative_deadline(self, tmp_path):
        e = make(tmp_path)
        r = run(e, "set_health_goal", goal_type="weight", target=70,
                raw_query="lose weight, goal 70 kg in 8 weeks")
        assert r["data"]["deadline"]

    def test_on_pace(self, tmp_path):
        e = make(tmp_path)
        self._weights(e, [(28, 84.0), (14, 82.0), (0, 80.0)])      # -0.143 kg/day
        e.db.set_goal("weight_target", 75, start_value=84,
                      deadline=(NOW + timedelta(days=90)).date().isoformat())
        text = e._get_goal_pace({}, NOW)["response"]
        assert "kg to lose" in text and "before your" in text
        assert "Recent trend: -1.00 kg/week" in text

    def test_behind_pace(self, tmp_path):
        e = make(tmp_path)
        self._weights(e, [(28, 80.4), (14, 80.2), (0, 80.0)])      # slow
        e.db.set_goal("weight_target", 70, start_value=80.4,
                      deadline=(NOW + timedelta(days=30)).date().isoformat())
        assert "after your" in e._get_goal_pace({}, NOW)["response"]

    def test_fast_loss_is_not_cheered(self, tmp_path):
        e = make(tmp_path)
        self._weights(e, [(21, 90.0), (14, 86.0), (7, 82.0), (0, 78.0)])   # ~4 kg/week
        e.db.set_goal("weight_target", 70, start_value=90)
        text = e._get_goal_pace({}, NOW)["response"]
        assert "faster rate of change than is usually advised" in text
        assert "great" not in text.lower() and "keep it up" not in text.lower()

    def test_unrealistic_deadline_is_flagged(self, tmp_path):
        e = make(tmp_path)
        self._weights(e, [(28, 80.2), (0, 80.0)])
        e.db.set_goal("weight_target", 70, start_value=80.2,
                      deadline=(NOW + timedelta(days=14)).date().isoformat())
        assert "faster than is generally recommended" in e._get_goal_pace({}, NOW)["response"]

    def test_no_trend_yet(self, tmp_path):
        e = make(tmp_path)
        e.db.log_weight(80, "")
        e.db.set_goal("weight_target", 70, start_value=80)
        assert "don't have a trend yet" in e._get_goal_pace({}, NOW)["response"]

    def test_no_goal(self, tmp_path):
        assert "weight goal" in make(tmp_path)._get_goal_pace({}, NOW)["response"]

    def test_reminders_off_by_default_and_respect_cadence(self, tmp_path):
        e = make(tmp_path)
        self._weights(e, [(28, 84.0), (14, 82.0), (0, 80.0)])
        e.db.set_goal("weight_target", 75, start_value=84)
        assert e.check_goal_pace_reminder(NOW) is None               # opt-in only
        e2 = make(tmp_path, goal_reminders={"enabled": True, "every_days": 7})
        assert e2.check_goal_pace_reminder(NOW).startswith("Goal check-in")
        assert e2.check_goal_pace_reminder(NOW + timedelta(days=3)) is None
        assert e2.check_goal_pace_reminder(NOW + timedelta(days=8)) is not None

    def test_reminder_can_be_turned_off_with_zero(self, tmp_path):
        e = make(tmp_path, goal_reminders={"enabled": True, "every_days": 0})
        self._weights(e, [(28, 84.0), (0, 80.0)])
        e.db.set_goal("weight_target", 75)
        assert e.check_goal_pace_reminder(NOW) is None


# ------------------------------------------------------------ #126 / #161

def fake_artemis(history):
    return SimpleNamespace(tracker=SimpleNamespace(habit_history=lambda: history))


class TestHabitMood:
    def test_no_artemis(self, tmp_path):
        assert "can't see your habits" in make(tmp_path)._habit_mood_correlation({}, NOW)["response"]

    def test_not_enough_data_until_history_exists(self, tmp_path):
        e = make(tmp_path)
        e.attach_artemis(fake_artemis({"meditate": {"dates": [], "since": "", "streak": 4, "last_done": ""}}))
        r = e._habit_mood_correlation({}, NOW)
        assert r["data"]["status"] == "no_history" and "Not enough data yet" in r["response"]

    def test_compares_high_and_low_mood_days(self, tmp_path):
        e = make(tmp_path)
        today = NOW.date()
        for i in range(1, 7):
            e.db.log_mood("great", "", logged_at=ts(i))            # high days
        for i in range(7, 13):
            e.db.log_mood("sad", "", logged_at=ts(i))              # low days
        kept = [(today - timedelta(days=i)).isoformat() for i in range(1, 7)]
        e.attach_artemis(fake_artemis({
            "meditate": {"dates": kept, "since": (today - timedelta(days=13)).isoformat(),
                         "streak": 6, "last_done": kept[0]}}))
        r = e._habit_mood_correlation({}, NOW)
        assert "100% of high-mood days (n=6)" in r["response"]
        assert "0% of low-mood days (n=6)" in r["response"]
        assert "not a cause" in r["response"]

    def test_days_before_history_are_not_counted_as_misses(self, tmp_path):
        e = make(tmp_path)
        today = NOW.date()
        for i in range(1, 13):
            e.db.log_mood("sad", "", logged_at=ts(i))
        e.attach_artemis(fake_artemis({
            "walk": {"dates": [today.isoformat()], "since": today.isoformat(),
                     "streak": 1, "last_done": today.isoformat()}}))
        assert "Not enough data yet" in e._habit_mood_correlation({}, NOW)["response"]


class TestBurnout:
    def _low_sleep(self, e, n=4, hours=5.0):
        for i in range(1, n + 1):
            e.db.log_sleep(hours, "", "", logged_at=ts(i))

    def _low_mood(self, e, n=4, mood="exhausted"):
        for i in range(1, n + 1):
            e.db.log_mood(mood, "", logged_at=ts(i, 20))

    def test_needs_two_sources(self, tmp_path):
        e = make(tmp_path)
        self._low_sleep(e)
        res = e.burnout_assessment(NOW)
        assert res["level"] is None and res["n_sources"] == 1
        assert "at least two areas" in e._burnout_check({}, NOW)["response"]

    def test_elevated_lists_each_signal_and_is_not_diagnostic(self, tmp_path):
        e = make(tmp_path)
        self._low_sleep(e)
        self._low_mood(e)
        r = e._burnout_check({}, NOW)
        assert r["data"]["level"] == "elevated"
        assert "sleep averaged 5.0 h" in r["response"] and "mood has been low" in r["response"]
        assert "not a diagnosis" in r["response"]

    def test_healthy_data_is_low(self, tmp_path):
        e = make(tmp_path)
        self._low_sleep(e, hours=8.0)
        self._low_mood(e, mood="great")
        assert e.burnout_assessment(NOW)["level"] == "low"

    def test_habit_drop_signal(self, tmp_path):
        e = make(tmp_path)
        self._low_sleep(e, hours=8.0)
        today = NOW.date()
        prior = [(today - timedelta(days=i)).isoformat() for i in range(7, 13)]
        e.attach_artemis(fake_artemis({"gym": {
            "dates": prior, "since": (today - timedelta(days=20)).isoformat(),
            "streak": 0, "last_done": prior[0]}}))
        res = e.burnout_assessment(NOW)
        assert "habits" in res["sources"]
        assert any("habit completions dropped" in s["text"] for s in res["signals"])

    def test_spending_spike_is_a_weak_signal_only(self, tmp_path):
        e = make(tmp_path)
        self._low_sleep(e, hours=8.0)
        rows = [{"amount": a, "logged_at": ts(2 + 7 * i)} for i, a in enumerate([500, 100, 100, 100, 100])]
        e.attach_pluto(SimpleNamespace(pf_manager=SimpleNamespace(
            db=SimpleNamespace(get_expenses=lambda n: rows))))
        res = e.burnout_assessment(NOW)
        assert "spending" in res["sources"]
        assert res["level"] == "low" and res["score"] == 0.5   # spending alone never raises a flag

    def test_weekly_check_only_speaks_when_flagged_and_marks_week(self, tmp_path):
        cfg = {"burnout": {"weekday": NOW.weekday(), "hour": 0}}
        e = make(tmp_path, **cfg)
        self._low_sleep(e)
        self._low_mood(e)
        msg = e.check_burnout(NOW)
        assert msg.startswith("Weekly check-in")
        assert e.check_burnout(NOW) is None                      # once per week
        quiet = make(tmp_path / "q" if (tmp_path / "q").mkdir() is None else tmp_path, **cfg)
        assert quiet.check_burnout(NOW) is None                  # nothing flagged: silent

    def test_sustained_elevated_suggests_talking_to_someone(self, tmp_path):
        e = make(tmp_path)
        self._low_sleep(e)
        self._low_mood(e)
        res = e.burnout_assessment(NOW)
        assert "talking it over" in e._burnout_text(res, sustained=True)
        assert "talking it over" not in e._burnout_text(res, sustained=False)

    def test_three_elevated_weeks_marks_sustained(self, tmp_path):
        e = make(tmp_path, burnout={"weekday": 0, "hour": 0})
        self._low_sleep(e)
        self._low_mood(e)
        e.db.set_state("burnout_history", json.dumps(["elevated", "elevated"]))
        assert "talking it over" in e.check_burnout(NOW)


# ------------------------------------------------ #149 / #159 / #183 hooks

class TestHooksAndDashboard:
    def test_recent_mood_context(self, tmp_path):
        e = make(tmp_path)
        assert e.recent_mood_context(NOW) is None
        e.db.log_mood("tired", "")
        ctx = e.recent_mood_context(NOW)
        assert ctx["mood"] == "tired" and ctx["low_trend"] is False

    def test_stale_mood_is_ignored(self, tmp_path):
        e = make(tmp_path)
        e.db.log_mood("tired", "", logged_at=ts(3))
        assert e.recent_mood_context(NOW) is None

    def test_low_trend(self, tmp_path):
        e = make(tmp_path)
        for i, m in enumerate(["sad", "tired", "low", "good"]):
            e.db.log_mood(m, "", logged_at=ts(3 - i if i < 3 else 0, 12 + i))
        assert e.recent_mood_context(NOW + timedelta(hours=1))["low_trend"] is True

    def test_rest_signals(self, tmp_path):
        e = make(tmp_path)
        for i in range(3):
            e.db.log_sleep(4.5, "", "", logged_at=ts(i))
        e.db.log_mood("exhausted", "")
        reasons = " ".join(s["reason"] for s in e.rest_signals(NOW))
        assert "4.5h of sleep" in reasons and "exhausted" in reasons
        assert all(s["direction"] == "rest" for s in e.rest_signals(NOW))

    def test_mood_entries_have_the_fields_the_tab_reads(self, tmp_path):
        e = make(tmp_path)
        e.db.log_mood("tired", "")
        row = e.mood_entries(7)[0]
        assert row["valence"] == "negative" and row["timestamp"] and row["score"] == -1

    def test_dashboard_shape_and_units(self, tmp_path):
        e = make(tmp_path)
        run(e, "set_units", weight_unit="lb")
        e.db.log_sleep(7.5, "good", "", bed_time="23:00", wake_time="06:30", logged_at=ts(1))
        e.db.log_weight(70.0, "", logged_at=ts(1))
        e.db.log_water(1500, logged_at=ts(1))
        e.db.log_mood("good", "", logged_at=ts(1))
        e.db.log_workout("run", 30, "", logged_at=ts(1))
        e.db.upsert_steps((NOW - timedelta(days=1)).date().isoformat(), 9000)
        d = e.dashboard_data(30, NOW)
        assert d["units"]["weight"] == "lb"
        assert d["weight"][0]["value"] == pytest.approx(154.32, abs=0.01)
        assert d["sleep"][0]["hours"] == 7.5 and d["sleep"][0]["score"] is not None
        assert d["water"][0]["ml"] == 1500 and d["mood"][0]["score"] == 1.0
        assert d["streaks"]["run"]["sessions"] == 1 and d["steps"][0]["steps"] == 9000

    def test_dashboard_empty_is_safe(self, tmp_path):
        d = make(tmp_path).dashboard_data(30, NOW)
        assert d["sleep"] == [] and d["weight"] == [] and d["streaks"] == {}


class TestRegistryAndContract:
    NEW = ["set_units", "get_sleep_quality", "get_correlations", "log_meal",
           "get_meal_summary", "hydration_status", "get_workout_streaks", "log_pain",
           "get_pain_trend", "get_weekly_summary", "import_steps", "get_goal_pace",
           "habit_mood_correlation", "burnout_check"]

    def test_engine_and_registry_agree(self, tmp_path):
        from modules.hecate.intent_registry import INTENT_MODULE_MAP
        e = make(tmp_path)
        for name in self.NEW:
            assert e.can_handle(name), name
            assert INTENT_MODULE_MAP[f"apollo_{name}"] == "apollo"

    def test_every_new_intent_dispatches_without_crashing(self, tmp_path):
        e = make(tmp_path)
        for name in self.NEW:
            r = e.handle(name, {}, {})
            assert isinstance(r["response"], str) and r["response"], name
