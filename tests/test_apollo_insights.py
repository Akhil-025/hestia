# tests/test_apollo_insights.py
"""
Pure-logic tests for modules/apollo/insights.py and schedule.py
(backlog #111, #112, #116, #117, #120, #126, #161, #115, #118).

Nothing here touches a database, the clock or the network.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.apollo import insights as I
from modules.apollo import schedule as S

UTC = timezone.utc


# ---------------------------------------------------------------- #111 sleep

class TestSleepWindowParsing:
    @pytest.mark.parametrize("text,expected", [
        ("slept 11pm to 6:30am", ("23:00", "06:30")),
        ("from 10:30 pm to 6 am", ("22:30", "06:00")),
        ("23:15 - 06:45", ("23:15", "06:45")),
        ("12am until 8am", ("00:00", "08:00")),
    ])
    def test_reads_common_phrasings(self, text, expected):
        assert I.parse_sleep_window(text) == expected

    @pytest.mark.parametrize("text", [
        "slept 7 to 8 hours",      # a duration, not a window
        "I slept 11pm to 6",        # wake time is ambiguous: don't guess
        "slept well",
        "",
        None,
    ])
    def test_refuses_ambiguous_or_missing(self, text):
        assert I.parse_sleep_window(text) is None

    def test_duration_wraps_past_midnight(self):
        assert I.window_hours("23:00", "06:30") == 7.5
        assert I.window_hours("01:00", "09:00") == 8.0
        assert I.window_hours("06:00", "06:00") is None

    def test_single_time(self):
        assert I.parse_time_of_day("11pm") == "23:00"
        assert I.parse_time_of_day("6:30 am") == "06:30"
        assert I.parse_time_of_day("banana") is None


class TestSleepScore:
    def test_midnight_wrap_counts_as_consistent(self):
        # 23:30 / 00:00 / 00:30 straddle midnight but are within an hour.
        sd = I.circular_std_minutes([23 * 60 + 30, 0, 30])
        assert sd < 40
        naive = I.consistency_score(["23:30", "00:00", "00:30"], ["06:30"] * 3)
        assert naive > 60

    def test_steady_beats_erratic(self):
        steady = I.consistency_score(["23:00"] * 5, ["07:00"] * 5)
        erratic = I.consistency_score(
            ["21:00", "01:30", "23:00", "03:00", "22:00"],
            ["05:00", "10:00", "07:00", "11:30", "06:00"],
        )
        assert steady == 100
        assert erratic < 30

    def test_needs_three_nights(self):
        assert I.consistency_score(["23:00", "23:10"], ["07:00", "07:10"]) is None

    def test_duration_only_reports_missing_components(self):
        r = I.sleep_quality_score(7.5)
        assert r["score"] == 100
        assert set(r["missing"]) == {"consistency", "rating"}

    def test_blend_uses_all_available_components(self):
        r = I.sleep_quality_score(
            7.5, rating=5, bed_times=["23:00"] * 4, wake_times=["06:30"] * 4
        )
        assert r["score"] == 100 and r["missing"] == []

    def test_short_sleep_scores_low(self):
        assert I.sleep_quality_score(4.0)["score"] < 40

    def test_no_hours_means_no_score(self):
        assert I.sleep_quality_score(None)["score"] is None

    def test_quality_word_is_used_when_no_rating(self):
        good = I.sleep_quality_score(7.5, quality="great")["components"]["rating"]
        bad = I.sleep_quality_score(7.5, quality="terrible")["components"]["rating"]
        assert good > bad


# --------------------------------------------------------------- #112 / #126

class TestMoodScoring:
    @pytest.mark.parametrize("text,sign", [
        ("great", 1), ("feeling tired", -1), ("so stressed", -1),
        ("not good", -1), ("not sad", 1), ("okay", 0),
    ])
    def test_signs(self, text, sign):
        s = I.mood_score(text)
        assert s is not None
        assert (s > 0) - (s < 0) == sign

    def test_unrecognised_is_none_not_zero(self):
        assert I.mood_score("purple") is None
        assert I.mood_score("") is None

    def test_daily_mood_skips_unscorable(self):
        rows = [
            {"mood": "great", "logged_at": "2026-09-01 08:00:00"},
            {"mood": "purple", "logged_at": "2026-09-01 09:00:00"},
            {"mood": "tired", "logged_at": "2026-09-02 08:00:00"},
        ]
        out = I.daily_mood(rows, UTC)
        assert out == {date(2026, 9, 1): 2.0, date(2026, 9, 2): -1.0}

    def test_local_day_bucketing_respects_timezone(self):
        # 22:00 UTC is already the next day in Kolkata (+5:30).
        rows = [{"mood": "good", "logged_at": "2026-09-01 22:00:00"}]
        ist = I.resolve_tz("Asia/Kolkata")
        assert list(I.daily_mood(rows, ist)) == [date(2026, 9, 2)]
        assert list(I.daily_mood(rows, UTC)) == [date(2026, 9, 1)]


class TestCompareGroups:
    def test_refuses_thin_data(self):
        r = I.compare_groups([1, 1, 1], [-1] * 6, min_n=4)
        assert r["status"] == "insufficient" and r["diff"] is None
        assert (r["n_a"], r["n_b"]) == (3, 6)

    def test_reports_difference_with_n(self):
        r = I.compare_groups([-1.0] * 5, [1.0] * 5, min_n=4)
        assert r["status"] == "ok" and r["diff"] == -2.0

    def test_small_difference_is_not_a_finding(self):
        assert I.compare_groups([0.1] * 5, [0.2] * 5)["status"] == "no_difference"

    def test_mood_by_sleep_skips_middle_band(self):
        d = lambda n: date(2026, 9, n)
        sleep = {d(1): 5.0, d(2): 6.5, d(3): 8.0}
        mood = {d(1): -1.0, d(2): 0.0, d(3): 1.0}
        r = I.mood_by_sleep(sleep, mood, min_n=1)
        assert (r["n_a"], r["n_b"]) == (1, 1)  # the 6.5h day is in neither group

    def test_mood_by_workout_splits_days(self):
        d = lambda n: date(2026, 9, n)
        mood = {d(1): 1.0, d(2): 1.0, d(3): -1.0, d(4): -1.0}
        r = I.mood_by_workout({d(1), d(2)}, mood, min_n=2)
        assert r["mean_a"] == 1.0 and r["mean_b"] == -1.0

    def test_habit_rate_ignores_days_before_history(self):
        d = lambda n: date(2026, 9, n)
        mood = {d(1): 1.0, d(2): 1.0, d(10): 1.0, d(11): -1.0}
        r = I.habit_rate_by_mood({d(10)}, since=d(10), mood_by_day=mood, today=d(30))
        assert (r["n_high"], r["n_low"]) == (1, 1)
        assert r["rate_high"] == 1.0 and r["rate_low"] == 0.0


# ---------------------------------------------------------------------- #116

class TestStreaks:
    def test_days_streak_allows_today_not_yet_done(self):
        today = date(2026, 9, 30)
        days = {today - timedelta(days=i) for i in (1, 2, 3)}
        assert I.consecutive_days(days, today) == 3
        assert I.consecutive_days(days | {today}, today) == 4
        assert I.consecutive_days({today - timedelta(days=3)}, today) == 0

    def test_weeks_streak_needs_min_sessions_and_skips_partial_week(self):
        today = date(2026, 9, 30)                       # a Wednesday
        this_mon = I.week_start(today)
        sessions = []
        for w in (1, 2, 3):                              # 3 full previous weeks
            monday = this_mon - timedelta(days=7 * w)
            sessions += [monday, monday + timedelta(days=2)]
        assert I.consecutive_weeks(sessions, today, min_sessions=2) == 3
        assert I.consecutive_weeks(sessions, today, min_sessions=3) == 0

    def test_a_missed_week_breaks_it(self):
        today = date(2026, 9, 30)
        mon = I.week_start(today)
        sessions = [mon - timedelta(days=7), mon - timedelta(days=21)]
        assert I.consecutive_weeks(sessions, today, 1) == 1

    def test_per_type(self):
        today = date(2026, 9, 30)
        rows = [{"type": "Run", "logged_at": f"2026-09-{d} 08:00:00"} for d in (28, 29, 30)]
        rows.append({"type": "yoga", "logged_at": "2026-09-01 08:00:00"})
        out = I.type_streaks(rows, UTC, today, min_sessions=1)
        assert out["run"]["days"] == 3 and out["yoga"]["days"] == 0


# ---------------------------------------------------------------------- #117

class TestPain:
    @pytest.mark.parametrize("text", [
        "chest pain when I run", "I can't breathe properly", "my arm is numb",
        "I passed out", "sudden severe headache",
    ])
    def test_red_flags_detected(self, text):
        assert I.red_flags(text)

    def test_ordinary_soreness_is_not_a_red_flag(self):
        assert I.red_flags("my knee is a bit sore after leg day") == []

    def test_guess_area(self):
        assert I.guess_body_area("my lower back hurts") == "lower back"
        assert I.guess_body_area("everything is fine") is None

    def _rows(self, today, recent, prior, area="knee"):
        rows = [{"area": area, "severity": s,
                 "logged_at": (today - timedelta(days=i)).strftime("%Y-%m-%d 09:00:00")}
                for i, s in enumerate(recent)]
        rows += [{"area": area, "severity": s,
                  "logged_at": (today - timedelta(days=8 + i)).strftime("%Y-%m-%d 09:00:00")}
                 for i, s in enumerate(prior)]
        return rows

    def test_trend_needs_enough_data(self):
        today = date(2026, 9, 30)
        out = I.pain_summary(self._rows(today, [5], []), UTC, today)[0]
        assert out["trend"] == "not_enough_data"

    def test_worsening_and_advice(self):
        today = date(2026, 9, 30)
        out = I.pain_summary(self._rows(today, [7, 7, 6], [3, 3, 4]), UTC, today)[0]
        assert out["trend"] == "worse"
        assert "clinician" in I.clinician_advice(out)

    def test_improving_gets_no_advice(self):
        today = date(2026, 9, 30)
        out = I.pain_summary(self._rows(today, [2, 2, 3], [6, 6, 5]), UTC, today)[0]
        assert out["trend"] == "better" and I.clinician_advice(out) is None

    def test_pain_lasting_weeks_gets_advice(self):
        today = date(2026, 9, 30)
        rows = [{"area": "hip", "severity": 3,
                 "logged_at": (today - timedelta(days=d)).strftime("%Y-%m-%d 09:00:00")}
                for d in (0, 10, 24)]
        out = I.pain_summary(rows, UTC, today)[0]
        assert out["persisting_weeks"]
        assert "three weeks" in I.clinician_advice(out)

    def test_very_severe_pain_gets_urgent_advice(self):
        today = date(2026, 9, 30)
        out = I.pain_summary(self._rows(today, [9], []), UTC, today)[0]
        assert "promptly" in I.clinician_advice(out)


# ---------------------------------------------------------------------- #120

class TestGoalPace:
    def test_reached(self):
        assert I.goal_pace(68.05, 68.0, -0.1)["status"] == "reached"

    def test_on_pace_with_deadline(self):
        r = I.goal_pace(72, 68, -0.05, start=80, days_left=100)   # 80 days to go
        assert r["status"] == "on_pace" and r["eta_days"] == 80
        assert r["progress_pct"] == 67

    def test_behind_with_deadline(self):
        r = I.goal_pace(72, 68, -0.02, start=80, days_left=100)   # 200 days to go
        assert r["status"] == "behind" and r["eta_days"] == 200

    def test_moving_away(self):
        assert I.goal_pace(72, 68, +0.05)["status"] == "moving_away"

    def test_no_trend(self):
        assert I.goal_pace(72, 68, None)["status"] == "no_trend"

    def test_eta_without_deadline(self):
        assert I.goal_pace(72, 68, -0.1)["status"] == "steady_no_deadline"

    def test_slope(self):
        assert I.linear_slope([(0, 80), (10, 79), (20, 78)]) == pytest.approx(-0.1)
        assert I.linear_slope([(0, 80)]) is None
        assert I.linear_slope([(1, 80), (1, 79)]) is None


# ---------------------------------------------------------------------- #161

class TestSpendSpike:
    def _exp(self, today, weeks):
        rows = []
        for idx, amount in enumerate(weeks):
            d = today - timedelta(days=7 * idx + 2)
            rows.append({"amount": amount, "logged_at": d.strftime("%Y-%m-%d 12:00:00")})
        return rows

    def test_spike_detected(self):
        today = date(2026, 9, 30)
        r = I.weekly_spend_spike(self._exp(today, [300, 100, 100, 100, 100]), today, UTC)
        assert r["spike"] and r["ratio"] == 3.0

    def test_normal_week_is_not_a_spike(self):
        today = date(2026, 9, 30)
        assert not I.weekly_spend_spike(self._exp(today, [110, 100, 100, 100, 100]), today, UTC)["spike"]

    def test_thin_history_returns_none(self):
        today = date(2026, 9, 30)
        assert I.weekly_spend_spike(self._exp(today, [300, 100]), today, UTC) is None


class TestWeeklySummaryText:
    def test_reads_cleanly_and_flags_fast_change_without_cheering(self):
        text = I.build_weekly_summary({
            "weight_unit": "kg", "sleep_n": 5, "sleep_avg": 7.1, "sleep_score_avg": 82,
            "workouts": 3, "active_days": 3, "weight_n": 3,
            "weight_first": 80.0, "weight_last": 78.5,
            "water_goal_ml": 2000, "water_days_logged": 6, "water_days_met": 4,
        })
        assert "7.1 h/night" in text and "3 session(s)" in text
        assert "4 of 6" in text
        assert "quick change" in text or "fairly quick" in text
        assert "great job" not in text.lower()

    def test_empty_sections_say_nothing_logged(self):
        text = I.build_weekly_summary({"weight_unit": "kg"})
        assert text.count("nothing logged") + text.count("none logged") >= 3


# ------------------------------------------------------------------ schedule

class TestSchedule:
    WAKE, SLEEP = 7 * 60, 22 * 60

    def test_expected_fraction(self):
        assert S.expected_fraction(6 * 60, self.WAKE, self.SLEEP) == 0
        assert S.expected_fraction(self.SLEEP, self.WAKE, self.SLEEP) == 1
        assert S.expected_fraction(14 * 60 + 30, self.WAKE, self.SLEEP) == pytest.approx(0.5)

    def test_status_behind_and_ahead(self):
        st = S.hydration_status(500, 2000, 14 * 60 + 30, self.WAKE, self.SLEEP)
        assert st["behind_ml"] == pytest.approx(500) and st["phase"] == "active"
        st = S.hydration_status(1500, 2000, 14 * 60 + 30, self.WAKE, self.SLEEP)
        assert st["behind_ml"] == 0 and st["ahead_ml"] == pytest.approx(500)

    def _now(self, h, m=0):
        return datetime(2026, 9, 30, h, m, tzinfo=UTC)

    def test_nudges_only_when_behind(self):
        n = self._now(15)
        behind = S.hydration_status(0, 2000, 15 * 60, self.WAKE, self.SLEEP)
        ok, why, st = S.nudge_decision(behind, {}, n)
        assert ok and st["count"] == 1
        fine = S.hydration_status(1400, 2000, 15 * 60, self.WAKE, self.SLEEP)
        assert S.nudge_decision(fine, {}, n)[:2] == (False, "on_pace")

    def test_quiet_hours_never_nudge(self):
        n = self._now(23)
        st = S.hydration_status(0, 2000, 23 * 60, self.WAKE, self.SLEEP)
        assert S.nudge_decision(st, {}, n)[:2] == (False, "quiet_hours")
        st = S.hydration_status(0, 2000, 5 * 60, self.WAKE, self.SLEEP)
        assert S.nudge_decision(st, {}, self._now(5))[:2] == (False, "quiet_hours")

    def test_cooldown_and_daily_cap(self):
        st = S.hydration_status(0, 2000, 15 * 60, self.WAKE, self.SLEEP)
        ok, _, state = S.nudge_decision(st, {}, self._now(15))
        assert S.nudge_decision(st, state, self._now(15, 30))[:2] == (False, "cooldown")
        assert S.nudge_decision(st, state, self._now(17))[0] is True
        capped = {"date": "2026-09-30", "count": 4, "last": None}
        assert S.nudge_decision(st, capped, self._now(18))[:2] == (False, "daily_cap")

    def test_counter_resets_next_day(self):
        st = S.hydration_status(0, 2000, 15 * 60, self.WAKE, self.SLEEP)
        yesterday = {"date": "2026-09-29", "count": 4, "last": "2026-09-29T20:00:00+00:00"}
        assert S.nudge_decision(st, yesterday, self._now(15))[0] is True

    def test_weekly_due_once_per_iso_week(self):
        sun_evening = datetime(2026, 9, 27, 19, 0)        # a Sunday
        assert S.weekly_due(sun_evening, None)
        assert not S.weekly_due(sun_evening, S.iso_week_key(sun_evening.date()))
        assert not S.weekly_due(datetime(2026, 9, 27, 9, 0), None)   # too early
        assert not S.weekly_due(datetime(2026, 9, 30, 12, 0), None)  # midweek
        # week rolled over: due again on the next Sunday
        assert S.weekly_due(datetime(2026, 10, 4, 19, 0), S.iso_week_key(sun_evening.date()))

    def test_every_n_days(self):
        today = date(2026, 9, 30)
        assert not S.every_n_days_due(today, None, 0)          # off
        assert S.every_n_days_due(today, None, 7)
        assert not S.every_n_days_due(today, "2026-09-25", 7)
        assert S.every_n_days_due(today, "2026-09-23", 7)

    def test_parse_hhmm_falls_back(self):
        assert S.parse_hhmm("07:30", 0) == 450
        assert S.parse_hhmm("bogus", 99) == 99
