# tests/test_chronos_weather.py
"""
Tests for weather-triggered suggestions (backlog #88): rain assessment in
agenda.py, the weather_plan intent and the once-a-day proactive warning.

The forecast is always a stub - no network. Note the live Open-Meteo call
itself (`_fetch_forecast`) is not exercised here.

Run with:  pytest tests/test_chronos_weather.py -v
"""
import os
import sys
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chronos import agenda as ag  # noqa: E402
from modules.chronos.engine import ChronosEngine, WeatherFetchError  # noqa: E402
from modules.mnemosyne.db import MnemosyneDB  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 9, 29)


def forecast(day="2026-09-29", probs=None, codes=None):
    """Hourly forecast for one day; probs/codes map hour -> value."""
    probs, codes = probs or {}, codes or {}
    return {
        "time": [f"{day}T{h:02d}:00" for h in range(24)],
        "precipitation_probability": [probs.get(h, 0) for h in range(24)],
        "weathercode": [codes.get(h, 0) for h in range(24)],
    }


def item(text, hour=None):
    when = datetime(2026, 9, 29, hour, 0, tzinfo=IST) if hour is not None else None
    return ag.AgendaItem(when, "reminder", text)


# -- pure assessment ---------------------------------------------------------

@pytest.mark.parametrize("text, outdoor", [
    ("morning run", True), ("go for a hike", True), ("cricket match", True),
    ("dentist appointment", False), ("call mom", False), ("", False),
])
def test_is_outdoor(text, outdoor):
    assert ag.is_outdoor(text) is outdoor


def test_rain_during_a_timed_outdoor_item_is_a_concern():
    c = ag.assess_item(item("evening run", 18), forecast(probs={18: 80, 19: 70}), IST)
    assert c is not None and c.probability == 80 and c.window == "6:00 PM"


def test_rain_before_the_item_is_not_a_concern():
    assert ag.assess_item(item("evening run", 18), forecast(probs={9: 90}), IST) is None


def test_below_threshold_is_not_a_concern():
    assert ag.assess_item(item("evening run", 18), forecast(probs={18: 49}), IST) is None


def test_wet_weather_code_counts_even_with_low_probability():
    c = ag.assess_item(item("evening run", 18), forecast(probs={18: 10}, codes={18: 95}), IST)
    assert c is not None and c.condition == "thunderstorms"


def test_untimed_item_is_checked_across_the_day():
    c = ag.assess_item(item("garden work"), forecast(probs={14: 75}), IST)
    assert c is not None and c.window == "2:00 PM"


def test_no_forecast_data_means_no_concern():
    assert ag.assess_item(item("run", 18), {}, IST) is None


def test_assess_agenda_ignores_indoor_items_and_goals():
    goal = ag.AgendaItem(None, "goal", "run a marathon")
    concerns = ag.assess_agenda(
        [item("dentist", 18), goal, item("evening run", 18)], forecast(probs={18: 90}), IST)
    assert [c.item.text for c in concerns] == ["evening run"]


def test_rain_outlook_reports_peak_between_8_and_8():
    assert ag.rain_outlook(forecast(probs={3: 99, 14: 60, 16: 40}), DAY, IST) == (60, "2:00 PM")
    assert ag.rain_outlook(forecast(day="2026-10-05"), DAY, IST) is None


def test_format_concerns_mentions_probability_and_item():
    c = ag.assess_item(item("evening run", 18), forecast(probs={18: 80}), IST)
    text = ag.format_concerns([c], IST)
    assert "80%" in text and "evening run" in text and "6:00 PM" in text


# -- weather_plan intent -----------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    mem = MagicMock()
    mem.db = MnemosyneDB(str(tmp_path / "w.db"))
    mem.get_device_location.return_value = None
    now = datetime(2026, 9, 29, 6, 0, tzinfo=IST).astimezone(timezone.utc)
    eng = ChronosEngine(memory=mem, local_tz="Asia/Kolkata", clock=lambda: now,
                        notify=MagicMock(), exports_dir=str(tmp_path),
                        proactive_weather=True)
    eng.state_now = now
    return eng


def plan(engine, raw, fc, **ent):
    with patch("modules.chronos.engine._fetch_forecast", return_value=fc):
        return engine.handle("weather_plan", {"raw_query": raw, **ent}, {})


def test_weather_plan_warns_about_rain_on_an_outdoor_plan(engine):
    r = plan(engine, "is it going to rain during my run today",
             forecast(probs={17: 85}), activity="run")
    assert r["data"]["concerns"] and "85%" in r["response"]
    assert "indoors" in r["response"] or "umbrella" in r["response"]


def test_weather_plan_all_clear(engine):
    r = plan(engine, "will the rain ruin my run today", forecast(probs={17: 10}), activity="run")
    assert not r["data"]["concerns"] and "No rain worries" in r["response"]


def test_weather_plan_with_no_outdoor_plans(engine):
    r = plan(engine, "will the rain ruin my plans today", forecast(probs={17: 10}))
    assert "no outdoor plans" in r["response"]


def test_weather_plan_forecast_not_covering_the_day(engine):
    r = plan(engine, "rain tomorrow for my run", forecast(), activity="run", date="tomorrow")
    assert "doesn't cover" in r["response"]


def test_weather_plan_fetch_failure_is_reported_not_raised(engine):
    with patch("modules.chronos.engine._fetch_forecast", side_effect=WeatherFetchError("x")):
        r = engine.handle("weather_plan", {"raw_query": "rain today"}, {})
    assert r["confidence"] == 0.0 and "couldn't fetch" in r["response"]


def test_weather_plan_offers_dionysus_indoor_idea_when_asked(engine):
    dion = MagicMock()
    dion.handle.return_value = {"response": "Try the science museum.", "confidence": 0.9}
    engine.attach_sources(dionysus=dion)
    r = plan(engine, "rain during my run today, suggest something indoors instead",
             forecast(probs={17: 85}), activity="run")
    assert "science museum" in r["response"]
    assert r["data"]["alternative"] == "Try the science museum."


def test_dionysus_failure_falls_back_gracefully(engine):
    dion = MagicMock()
    dion.handle.side_effect = RuntimeError("down")
    engine.attach_sources(dionysus=dion)
    r = plan(engine, "rain during my run today, something indoors instead",
             forecast(probs={17: 85}), activity="run")
    assert r["data"]["concerns"] and "alternative" not in r["data"]


# -- proactive morning warning ----------------------------------------------

def test_proactive_warning_only_when_enabled(tmp_path):
    eng = ChronosEngine(memory=None, local_tz="Asia/Kolkata", proactive_weather=False)
    assert eng._maybe_proactive_weather() is None


def test_proactive_warning_fires_once_per_day(engine):
    engine.handle("set_reminder", {"raw_query": "remind me to go for a run at 6pm today",
                                   "task": "go for a run"}, {})
    fc = forecast(probs={18: 90, 19: 90})
    late = datetime(2026, 9, 29, 8, 0, tzinfo=IST)
    with patch("modules.chronos.engine._fetch_forecast", return_value=fc):
        first = engine._maybe_proactive_weather(late)
        second = engine._maybe_proactive_weather(late)
    assert first and first.startswith("Heads up")
    assert second is None


def test_proactive_warning_retries_after_a_failed_fetch(engine):
    late = datetime(2026, 9, 29, 8, 0, tzinfo=IST)
    with patch("modules.chronos.engine._fetch_forecast", side_effect=WeatherFetchError("x")):
        assert engine._maybe_proactive_weather(late) is None
    assert engine._weather_attempts == 1 and engine._weather_next_try is not None
