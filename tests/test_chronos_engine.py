# tests/test_chronos_engine.py
"""
End-to-end tests for ChronosEngine's extended reminder features
(backlog #81-#90) against a real SQLite database and a fake clock.

Run with:  pytest tests/test_chronos_engine.py -v
"""
import os
import sys
import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chronos.engine import ChronosEngine  # noqa: E402
from modules.mnemosyne import schema  # noqa: E402
from modules.mnemosyne.db import MnemosyneDB  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")


class FakeMemory:
    """Just enough of MnemosyneEngine for Chronos."""

    def __init__(self, db):
        self.db = db
        self.listeners = []
        self.loc = None

    def add_location_listener(self, cb):
        self.listeners.append(cb)

    def get_device_location(self):
        return self.loc

    def add_reminder(self, text, due):
        self.db.add_reminder(text, due)


class Rig:
    def __init__(self, tmp_path):
        self.db = MnemosyneDB(str(tmp_path / "t.db"))
        self.mem = FakeMemory(self.db)
        self.now = datetime(2026, 9, 29, 6, 0, tzinfo=IST).astimezone(timezone.utc)
        self.said = []
        self.engine = ChronosEngine(
            memory=self.mem, local_tz="Asia/Kolkata", clock=lambda: self.now,
            notify=self.said.append, exports_dir=str(tmp_path),
        )

    def set_time(self, *args):
        self.now = datetime(*args, tzinfo=IST).astimezone(timezone.utc)

    def ask(self, intent, raw="", **entities):
        return self.engine.handle(intent, {"raw_query": raw, **entities}, {})


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def test_engine_is_extended_with_a_database(rig):
    assert rig.engine.extended
    for intent in ("list_reminders", "cancel_reminder", "snooze_reminder", "get_agenda",
                   "mark_holiday", "unmark_holiday", "save_place", "export_calendar",
                   "import_calendar", "weather_plan"):
        assert rig.engine.can_handle(intent)


def test_location_listener_is_registered(rig):
    assert rig.engine.on_location in rig.mem.listeners or rig.mem.listeners


def test_recurring_reminder_fires_and_advances(rig):
    r = rig.ask("set_reminder", "remind me to take vitamins every weekday at 7am", task="take vitamins")
    assert "every weekday at 7:00 AM" in r["response"]
    rig.set_time(2026, 9, 29, 7, 0, 30)
    fired = rig.engine.poll_once()
    assert [f.text for f in fired] == ["take vitamins"]
    assert rig.said == ["Reminder: take vitamins"]
    listing = rig.ask("list_reminders", "list my reminders")["response"]
    assert "take vitamins" in listing and "tomorrow at 7:00 AM" in listing
    assert rig.engine.poll_once() == []          # never announced twice


def test_recurring_reminder_describes_yearly_dates(rig):
    r = rig.ask("set_reminder", "remind me to renew the domain every year on march 5", task="renew the domain")
    assert "March 5" in r["response"]


def test_unsupported_nth_weekday_is_refused_not_guessed(rig):
    r = rig.ask("set_reminder", "remind me to plan the first monday of every month", task="plan")
    assert r["confidence"] == 0.0 or "can't" in r["response"].lower()
    assert rig.db.list_reminders() == []


def test_time_zone_reminder(rig):
    r = rig.ask("set_reminder", "remind me to call the bank at 9am London time",
                task="call the bank", time="09:00")
    assert "London" in r["response"]


def test_snooze_after_fire_creates_a_linked_copy(rig):
    rig.ask("set_reminder", "remind me to stretch in 10 minutes")
    rig.now += timedelta(minutes=11)
    assert [f.text for f in rig.engine.poll_once()] == ["stretch"]
    r = rig.ask("snooze_reminder", "snooze that for 5 minutes")
    assert r["data"]["snoozed"] is True and r["data"]["mode"] == "new"
    rows = rig.db.list_reminders()
    assert len(rows) == 1 and rows[0]["snooze_count"] == 1


def test_snooze_with_nothing_to_snooze(rig):
    r = rig.ask("snooze_reminder", "snooze")
    assert r["data"]["snoozed"] is False


def test_missed_reminder_is_surfaced_once(rig):
    rig.ask("set_reminder", "remind me to stretch in 10 minutes")
    rig.now += timedelta(hours=1)
    fired = rig.engine.poll_once()
    assert len(fired) == 1 and fired[0].missed
    assert "missed" in rig.said[0].lower()
    assert rig.engine.poll_once() == []


def test_holiday_marking_skips_flagged_reminders(rig):
    rig.ask("set_reminder", "remind me to study every evening but not on holidays", task="study")
    rig.ask("mark_holiday", "mark today as a holiday", date="today")
    rig.set_time(2026, 9, 29, 18, 1)
    assert rig.engine.poll_once() == []
    rig.ask("unmark_holiday", "unmark today as a holiday", date="today")


def test_agenda_lists_todays_reminders(rig):
    rig.ask("set_reminder", "remind me to take vitamins every weekday at 7am", task="take vitamins")
    r = rig.ask("get_agenda", "what's on my plate today")
    assert "take vitamins" in r["response"]


def test_agenda_does_not_call_todays_occurrence_overdue_tomorrow(rig):
    rig.ask("set_reminder", "remind me to take vitamins every weekday at 7am", task="take vitamins")
    r = rig.ask("get_agenda", "what does my day look like tomorrow", date="tomorrow")
    assert "Overdue" not in r["response"]


def test_location_reminder_arms_then_fires_on_arrival(rig):
    rig.ask("save_place", "save my location as home", place="home", lat=19.07, lon=72.87)
    r = rig.ask("set_reminder", "remind me to buy milk when I get home", task="buy milk")
    assert "when you get home" in r["response"]
    assert rig.engine.on_location(19.20, 72.99) == []        # away: arms it
    fired = rig.engine.on_location(19.0701, 72.8701)          # back home
    assert [f.text for f in fired] == ["buy milk"]
    assert rig.engine.on_location(19.0701, 72.8701) == []     # once only


def test_ics_export_then_import_does_not_duplicate(rig):
    rig.ask("set_reminder", "remind me to study every evening", task="study")
    out = rig.ask("export_calendar", "export my reminders")
    path = out["data"]["path"]
    text = open(path, encoding="utf-8").read()
    assert "BEGIN:VCALENDAR" in text and "RRULE:FREQ=DAILY" in text
    back = rig.ask("import_calendar", "import my reminders", ics_text=text)
    assert "Imported 0" in back["response"]
    assert len(rig.db.list_reminders()) == 1


def test_cancel_one_and_cancel_all(rig):
    rig.ask("set_reminder", "remind me to go to the gym in 2 hours")
    rig.ask("set_reminder", "remind me to call mom in 3 hours")
    r = rig.ask("cancel_reminder", "cancel the gym reminder")
    assert "gym" in r["response"]
    assert len(rig.db.list_reminders()) == 1
    rig.ask("cancel_reminder", "cancel all my reminders")
    assert rig.db.list_reminders() == []


def test_scheduler_start_and_stop(rig):
    assert rig.engine.start_scheduler(interval=0.05) is True
    assert rig.engine.start_scheduler(interval=0.05) is False    # already running
    assert rig.engine.scheduler_running
    rig.engine.stop_scheduler()
    assert not rig.engine.scheduler_running


def test_old_database_is_migrated_and_keeps_its_rows(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE reminders (id INTEGER PRIMARY KEY, text TEXT, due_time TIMESTAMP, "
                 "status TEXT DEFAULT 'pending', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("INSERT INTO reminders(text, due_time) VALUES ('x', '2030-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()
    schema.init_db(path)
    schema.init_db(path)                                       # idempotent
    conn = sqlite3.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(reminders)")}
    assert {"recurrence", "tz", "skip_holidays", "place_lat", "armed", "snooze_of",
            "snooze_count", "fired_at", "skipped_count", "missed", "ics_uid"} <= cols
    assert conn.execute("SELECT count(*) FROM reminders").fetchone()[0] == 1
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"user_holidays", "places"} <= tables
    conn.close()
