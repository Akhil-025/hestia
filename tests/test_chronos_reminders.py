# tests/test_chronos_reminders.py
"""
Standalone tests for modules/chronos/reminders.py's ReminderService
(backlog #81-#83, #85, #87, #89, #90), driven directly - no ChronosEngine -
against a real SQLite MnemosyneDB and an injected fake clock.

Run with:  pytest tests/test_chronos_reminders.py -v
"""
import os
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chronos.recurrence import Recurrence, next_after  # noqa: E402
from modules.chronos.reminders import (  # noqa: E402
    ARM_DISTANCE_FACTOR,
    DEFAULT_RADIUS_M,
    MAX_SNOOZE,
    MIN_SNOOZE,
    Fired,
    ReminderService,
    describe_when,
    haversine_m,
    parse_iso,
    to_iso,
)
from modules.mnemosyne.db import MnemosyneDB  # noqa: E402

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 9, 29, 6, 0, tzinfo=IST)          # Tuesday, 06:00 IST


class Rig:
    def __init__(self, tmp_path, public=None):
        self.db = MnemosyneDB(str(tmp_path / "r.db"))
        self.now = T0
        self.svc = ReminderService(self.db, IST, clock=lambda: self.now,
                                   public_holiday_lookup=public)

    def add(self, text, due, **kw):
        return self.svc.create(text, due, **kw)

    def daily(self, hour=7, **kw):
        rule = Recurrence(freq="daily", hour=hour, anchor=to_iso(datetime(2026, 9, 29, hour, 0, tzinfo=IST)),
                          **kw)
        return rule


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_parse_iso_variants():
    assert parse_iso("2026-09-29T06:00:00+05:30") == T0
    assert parse_iso("2026-09-29T00:30:00Z") == T0
    assert parse_iso("2026-09-29T06:00:00", IST) == T0            # naive -> default zone
    assert parse_iso(T0) is T0
    assert parse_iso("") is None and parse_iso(None) is None and parse_iso("garbage") is None


def test_to_iso_drops_microseconds():
    assert to_iso(T0.replace(microsecond=999)) == "2026-09-29T06:00:00+05:30"


def test_haversine_known_distances():
    assert haversine_m(19.0, 72.0, 19.0, 72.0) == 0
    assert 111_000 < haversine_m(0, 0, 1, 0) < 111_400        # one degree of latitude
    assert haversine_m(19.076, 72.8777, 28.6139, 77.2090) == pytest.approx(1_150_000, rel=0.03)


@pytest.mark.parametrize("delta, expected", [
    (timedelta(hours=3), "today at 9:00 AM"),
    (timedelta(days=1, hours=1), "tomorrow at 10:00 AM"),
    (timedelta(days=-1, hours=3), "yesterday at 9:00 AM"),
    (timedelta(days=3, hours=3), "Friday at 9:00 AM"),
    (timedelta(days=10, hours=3), "Friday, October 9 at 9:00 AM"),
])
def test_describe_when(delta, expected):
    base = T0
    target = base + delta
    if delta == timedelta(days=1, hours=1):
        target = datetime(2026, 9, 30, 10, 0, tzinfo=IST)
    assert describe_when(target, IST, base) == expected


def test_describe_when_none_and_other_zone():
    assert describe_when(None, IST, T0) == "no set time"
    assert describe_when(T0, NY, T0) == "today at 8:30 PM"      # both dates read in the target zone


# ---------------------------------------------------------------------------
# zones
# ---------------------------------------------------------------------------

def test_zone_for_uses_row_zone_or_default(rig):
    assert rig.svc.zone_for({"tz": "America/New_York"}) == NY
    assert rig.svc.zone_for({}) == IST
    assert rig.svc.zone_for({"tz": "Not/AZone"}) == IST
    assert rig.svc.tz_name_for({"tz": "Europe/London"}) == "Europe/London"
    assert rig.svc.tz_name_for({"tz": "bogus"}) == "Asia/Kolkata"


def test_service_default_zone_name_falls_back_to_utc(tmp_path):
    svc = ReminderService(MnemosyneDB(str(tmp_path / "u.db")))
    assert svc.default_tz_name == "UTC"


# ---------------------------------------------------------------------------
# tick: one-shots
# ---------------------------------------------------------------------------

def test_one_shot_fires_once_when_due(rig):
    rid = rig.add("stretch", T0 + timedelta(minutes=10))
    assert rig.svc.tick(T0) == []
    fired = rig.svc.tick(T0 + timedelta(minutes=10, seconds=5))
    assert [f.id for f in fired] == [rid] and not fired[0].missed and not fired[0].recurring
    assert rig.db.get_reminder(rid)["status"] == "done"
    assert rig.svc.tick(T0 + timedelta(hours=1)) == []


def test_reminder_just_within_grace_is_not_missed(rig):
    rig.add("a", T0)
    (f,) = rig.svc.tick(T0 + timedelta(minutes=14))
    assert f.missed is False


def test_reminder_past_grace_is_flagged_missed_and_stored(rig):
    rid = rig.add("a", T0)
    (f,) = rig.svc.tick(T0 + timedelta(minutes=16))
    assert f.missed is True
    assert rig.db.get_reminder(rid)["missed"] == 1
    assert [r["id"] for r in rig.svc.recently_missed(now=T0 + timedelta(minutes=20))] == [rid]


def test_recently_missed_respects_the_window(rig):
    rig.add("a", T0)
    rig.svc.tick(T0 + timedelta(hours=1))
    assert rig.svc.recently_missed(days=7, now=T0 + timedelta(days=6))
    assert rig.svc.recently_missed(days=7, now=T0 + timedelta(days=8)) == []


def test_custom_grace_period(tmp_path):
    r = Rig(tmp_path)
    r.svc.missed_grace = timedelta(minutes=1)
    r.add("a", T0)
    (f,) = r.svc.tick(T0 + timedelta(minutes=2))
    assert f.missed


def test_tick_survives_a_store_read_failure(rig, monkeypatch):
    monkeypatch.setattr(rig.db, "get_due_reminders_full", lambda *_: (_ for _ in ()).throw(RuntimeError("db")))
    assert rig.svc.tick(T0) == []


def test_tick_isolates_a_failing_reminder(rig, monkeypatch):
    bad = rig.add("bad", T0)
    good = rig.add("good", T0)
    real = rig.db.claim_reminder

    def flaky(rid, *a, **k):
        if rid == bad:
            raise RuntimeError("boom")
        return real(rid, *a, **k)

    monkeypatch.setattr(rig.db, "claim_reminder", flaky)
    assert [f.id for f in rig.svc.tick(T0 + timedelta(minutes=1))] == [good]


def test_lost_claim_race_returns_nothing(rig, monkeypatch):
    rig.add("a", T0)
    monkeypatch.setattr(rig.db, "claim_reminder", lambda *a, **k: False)
    assert rig.svc.tick(T0 + timedelta(minutes=1)) == []


def test_reminder_with_unreadable_due_time_is_ignored(rig):
    rig.db.add_reminder_full("odd", "not-a-date")
    assert rig.svc.tick(T0 + timedelta(days=400)) == []


# ---------------------------------------------------------------------------
# tick: recurring
# ---------------------------------------------------------------------------

def test_recurring_fires_and_advances_by_one_day(rig):
    rid = rig.add("vitamins", datetime(2026, 9, 29, 7, 0, tzinfo=IST), rule=rig.daily(7))
    (f,) = rig.svc.tick(datetime(2026, 9, 29, 7, 0, 30, tzinfo=IST))
    assert f.recurring and not f.series_ended
    row = rig.db.get_reminder(rid)
    assert row["status"] == "pending"
    assert parse_iso(row["due_time"]) == datetime(2026, 9, 30, 7, 0, tzinfo=IST)


def test_recurring_after_downtime_fires_once_then_resumes_on_schedule(rig):
    rid = rig.add("vitamins", datetime(2026, 9, 29, 7, 0, tzinfo=IST), rule=rig.daily(7))
    woke = datetime(2026, 10, 2, 12, 0, tzinfo=IST)
    (f,) = rig.svc.tick(woke)
    assert f.missed
    assert parse_iso(rig.db.get_reminder(rid)["due_time"]) == datetime(2026, 10, 3, 7, 0, tzinfo=IST)
    assert rig.svc.tick(woke) == []                       # no backlog of 3 days of reminders


def test_count_limited_series_ends_and_is_flagged(rig):
    rid = rig.add("x", datetime(2026, 9, 29, 7, 0, tzinfo=IST), rule=rig.daily(7, count=2))
    (a,) = rig.svc.tick(datetime(2026, 9, 29, 7, 1, tzinfo=IST))
    assert not a.series_ended
    (b,) = rig.svc.tick(datetime(2026, 9, 30, 7, 1, tzinfo=IST))
    assert b.series_ended
    assert rig.db.get_reminder(rid)["status"] == "done"
    assert rig.svc.tick(datetime(2026, 10, 5, tzinfo=IST)) == []


def test_until_limited_series_ends_after_last_day(rig):
    rid = rig.add("x", datetime(2026, 9, 29, 7, 0, tzinfo=IST), rule=rig.daily(7, until="2026-09-30"))
    rig.svc.tick(datetime(2026, 9, 29, 7, 1, tzinfo=IST))
    (last,) = rig.svc.tick(datetime(2026, 9, 30, 7, 1, tzinfo=IST))
    assert last.series_ended and rig.db.get_reminder(rid)["status"] == "done"


def test_recurring_uses_the_reminders_own_time_zone(rig):
    rule = Recurrence(freq="daily", hour=9, anchor=to_iso(datetime(2026, 9, 29, 9, 0, tzinfo=NY)))
    rid = rig.add("ny standup", datetime(2026, 9, 29, 9, 0, tzinfo=NY), rule=rule, tz_name="America/New_York")
    rig.svc.tick(datetime(2026, 9, 29, 9, 1, tzinfo=NY))
    nxt = parse_iso(rig.db.get_reminder(rid)["due_time"])
    assert nxt.astimezone(NY) == datetime(2026, 9, 30, 9, 0, tzinfo=NY)


def test_unreadable_recurrence_is_treated_as_one_shot(rig):
    rid = rig.db.add_reminder_full("x", to_iso(T0), recurrence="{not json")
    (f,) = rig.svc.tick(T0 + timedelta(minutes=1))
    assert not f.recurring and rig.db.get_reminder(rid)["status"] == "done"


# ---------------------------------------------------------------------------
# holidays (#87)
# ---------------------------------------------------------------------------

def test_holiday_label_from_user_marks_and_public_lookup(tmp_path):
    r = Rig(tmp_path, public=lambda iso: "Gandhi Jayanti" if iso == "2026-10-02" else None)
    r.db.add_holiday("2026-09-30", "leave")
    r.db.add_holiday("2026-10-01")                               # marked without a label
    assert r.svc.holiday_label(date(2026, 9, 30)) == "leave"
    assert r.svc.holiday_label(date(2026, 10, 1)) == "a day off"
    assert r.svc.holiday_label(date(2026, 10, 2)) == "Gandhi Jayanti"
    assert r.svc.holiday_label(date(2026, 10, 3)) is None


def test_public_lookup_failure_means_not_a_holiday(tmp_path):
    def boom(_):
        raise RuntimeError("network")
    r = Rig(tmp_path, public=boom)
    assert r.svc.holiday_label(date(2026, 10, 2)) is None


def test_recurring_holiday_occurrence_is_skipped_but_series_continues(rig):
    rig.db.add_holiday("2026-09-29")
    rid = rig.add("study", datetime(2026, 9, 29, 18, 0, tzinfo=IST), rule=rig.daily(18), skip_holidays=True)
    assert rig.svc.tick(datetime(2026, 9, 29, 18, 1, tzinfo=IST)) == []
    row = rig.db.get_reminder(rid)
    assert row["skipped_count"] == 1 and row["fired_at"] is None
    assert parse_iso(row["due_time"]) == datetime(2026, 9, 30, 18, 0, tzinfo=IST)


def test_skipping_a_holiday_does_not_use_up_the_count(rig):
    rig.db.add_holiday("2026-09-29")
    rid = rig.add("study", datetime(2026, 9, 29, 18, 0, tzinfo=IST),
                  rule=rig.daily(18, count=1), skip_holidays=True)
    rig.svc.tick(datetime(2026, 9, 29, 18, 1, tzinfo=IST))
    assert Recurrence.from_json(rig.db.get_reminder(rid)["recurrence"]).count == 1
    (f,) = rig.svc.tick(datetime(2026, 9, 30, 18, 1, tzinfo=IST))
    assert f.series_ended


def test_one_shot_rolls_past_consecutive_holidays(rig):
    rig.db.add_holiday("2026-09-29")
    rig.db.add_holiday("2026-09-30")
    rid = rig.add("file report", datetime(2026, 9, 29, 9, 0, tzinfo=IST), skip_holidays=True)
    assert rig.svc.tick(datetime(2026, 9, 29, 9, 1, tzinfo=IST)) == []
    row = rig.db.get_reminder(rid)
    assert parse_iso(row["due_time"]) == datetime(2026, 10, 1, 9, 0, tzinfo=IST)
    assert row["skipped_count"] == 1 and row["status"] == "pending"


def test_reminders_not_flagged_skip_holidays_still_fire_on_holidays(rig):
    rig.db.add_holiday("2026-09-29")
    rig.add("call mom", datetime(2026, 9, 29, 9, 0, tzinfo=IST))
    assert len(rig.svc.tick(datetime(2026, 9, 29, 9, 1, tzinfo=IST))) == 1


def test_next_non_holiday_is_a_noop_on_a_working_day(rig):
    due = datetime(2026, 9, 29, 9, 0, tzinfo=IST)
    assert rig.svc.next_non_holiday(due, IST) == (due, 0)


# ---------------------------------------------------------------------------
# location (#82)
# ---------------------------------------------------------------------------

HOME = {"label": "home", "lat": 19.0760, "lon": 72.8777, "radius_m": 150.0}
AWAY = (19.20, 72.99)                                            # ~17 km away


def place_reminder(rig, **over):
    return rig.add("buy milk", None, place={**HOME, **over})


def test_arm_if_outside_only_when_far_enough(rig):
    rid = place_reminder(rig)
    assert rig.svc.arm_if_outside(rid, HOME["lat"], HOME["lon"]) is False
    assert rig.svc.arm_if_outside(rid, *AWAY) is True
    assert rig.db.get_reminder(rid)["armed"] == 1


def test_arm_if_outside_ignores_non_location_reminders(rig):
    rid = rig.add("x", T0 + timedelta(hours=1))
    assert rig.svc.arm_if_outside(rid, *AWAY) is False
    assert rig.svc.arm_if_outside(9999, *AWAY) is False


def test_unarmed_reminder_does_not_fire_when_already_home(rig):
    place_reminder(rig)
    assert rig.svc.on_location(HOME["lat"], HOME["lon"], now=T0) == []


def test_leave_then_arrive_fires_once(rig):
    rid = place_reminder(rig)
    assert rig.svc.on_location(*AWAY, now=T0) == []
    assert rig.db.get_reminder(rid)["armed"] == 1
    (f,) = rig.svc.on_location(HOME["lat"], HOME["lon"], now=T0 + timedelta(hours=1))
    assert f.place == "home" and f.text == "buy milk"
    assert rig.svc.on_location(HOME["lat"], HOME["lon"], now=T0 + timedelta(hours=2)) == []


def test_gps_jitter_at_the_edge_does_not_arm_or_fire(rig):
    rid = place_reminder(rig)
    edge_lat = HOME["lat"] + (HOME["radius_m"] * 1.2) / 111_320     # inside 1.5 x radius
    assert rig.svc.on_location(edge_lat, HOME["lon"], now=T0) == []
    assert rig.db.get_reminder(rid)["armed"] == 0


def test_armed_reminder_between_radius_and_arm_distance_waits(rig):
    rid = place_reminder(rig)
    rig.db.set_reminder_armed(rid, True)
    mid = HOME["lat"] + (HOME["radius_m"] * 1.3) / 111_320
    assert rig.svc.on_location(mid, HOME["lon"], now=T0) == []
    assert rig.db.get_reminder(rid)["status"] == "pending"


def test_default_radius_applies_when_none_stored(rig):
    rid = rig.add("x", None, place={"label": "gym", "lat": 19.0, "lon": 72.0, "armed": True})
    near = 19.0 + (DEFAULT_RADIUS_M * 0.5) / 111_320
    assert len(rig.svc.on_location(near, 72.0, now=T0)) == 1


def test_on_location_survives_store_failure(rig, monkeypatch):
    monkeypatch.setattr(rig.db, "get_location_reminders", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert rig.svc.on_location(*AWAY) == []


# ---------------------------------------------------------------------------
# snooze (#83)
# ---------------------------------------------------------------------------

def fire(rig, text="stretch", when=None, **kw):
    rid = rig.add(text, (when or T0) + timedelta(minutes=1), **kw)
    rig.svc.tick((when or T0) + timedelta(minutes=2))
    return rid


def test_snooze_of_a_fired_reminder_creates_a_linked_copy(rig):
    rid = fire(rig)
    now = T0 + timedelta(minutes=5)
    res = rig.svc.snooze(duration=timedelta(minutes=10), now=now)
    assert res["ok"] and res["mode"] == "new" and res["snooze_count"] == 1
    new = rig.db.get_reminder(res["id"])
    assert new["snooze_of"] == rid and new["status"] == "pending"
    assert parse_iso(new["due_time"]) == now + timedelta(minutes=10)


def test_repeated_snoozes_link_to_the_root_and_count_up(rig):
    root = fire(rig)
    a = rig.svc.snooze(duration=timedelta(minutes=5), now=T0 + timedelta(minutes=3))
    rig.svc.tick(T0 + timedelta(minutes=9))
    b = rig.svc.snooze(duration=timedelta(minutes=5), now=T0 + timedelta(minutes=10))
    assert rig.db.get_reminder(b["id"])["snooze_of"] == root
    assert b["snooze_count"] == 2 and a["snooze_count"] == 1


def test_snooze_of_a_waiting_reminder_postpones_it(rig):
    rid = rig.add("dentist", T0 + timedelta(hours=1))
    res = rig.svc.snooze(text_hint="dentist", duration=timedelta(minutes=30), now=T0)
    assert res["ok"] and res["mode"] == "postponed" and res["id"] == rid
    assert parse_iso(rig.db.get_reminder(rid)["due_time"]) == T0 + timedelta(hours=1, minutes=30)


def test_snooze_by_id(rig):
    rid = rig.add("dentist", T0 + timedelta(hours=1))
    assert rig.svc.snooze(reminder_id=rid, duration=timedelta(minutes=5), now=T0)["mode"] == "postponed"


def test_snooze_uses_default_and_clamps_duration(rig):
    rid = rig.add("x", T0 + timedelta(hours=1))
    base = parse_iso(rig.db.get_reminder(rid)["due_time"])
    res = rig.svc.snooze(reminder_id=rid, now=T0)
    assert res["duration"] == rig.svc.default_snooze
    assert rig.svc.snooze(reminder_id=rid, duration=timedelta(seconds=1), now=T0)["duration"] == MIN_SNOOZE
    assert rig.svc.snooze(reminder_id=rid, duration=timedelta(days=99), now=T0)["duration"] == MAX_SNOOZE
    assert parse_iso(rig.db.get_reminder(rid)["due_time"]) > base


def test_snooze_with_nothing_fired_recently(rig):
    assert rig.svc.snooze(now=T0) == {"ok": False, "reason": "none"}


def test_snooze_window_expires_after_12_hours(rig):
    fire(rig)
    assert rig.svc.snooze(now=T0 + timedelta(hours=13))["reason"] == "none"


def test_snooze_of_a_cancelled_reminder_is_refused(rig):
    rid = rig.add("x", T0 + timedelta(hours=1))
    rig.svc.cancel(rid)
    assert rig.svc.snooze(reminder_id=rid, now=T0) == {"ok": False, "reason": "cancelled"}


def test_snooze_of_a_location_reminder_is_refused(rig):
    rid = place_reminder(rig)
    assert rig.svc.snooze(reminder_id=rid, now=T0)["reason"] == "location"


def test_snoozing_a_recurring_reminder_leaves_the_series_alone(rig):
    rid = rig.add("vitamins", datetime(2026, 9, 29, 7, 0, tzinfo=IST), rule=rig.daily(7))
    rig.svc.tick(datetime(2026, 9, 29, 7, 1, tzinfo=IST))
    res = rig.svc.snooze(text_hint="vitamins", duration=timedelta(minutes=10),
                         now=datetime(2026, 9, 29, 7, 2, tzinfo=IST))
    assert res["mode"] == "new"
    series = rig.db.get_reminder(rid)
    assert parse_iso(series["due_time"]) == datetime(2026, 9, 30, 7, 0, tzinfo=IST)


# ---------------------------------------------------------------------------
# listing, cancelling, describing, announcing
# ---------------------------------------------------------------------------

def test_cancel_and_find_pending(rig):
    a = rig.add("gym session", T0 + timedelta(hours=1))
    rig.add("call mom", T0 + timedelta(hours=2))
    assert [r["id"] for r in rig.svc.find_pending("gym")] == [a]
    assert rig.svc.cancel(a) is True
    assert rig.svc.find_pending("gym") == []
    assert [r["text"] for r in rig.svc.list_pending()] == ["call mom"]
    assert rig.svc.cancel(9999) is False


def test_describe_row_variants(rig):
    one = rig.db.get_reminder(rig.add("call mom", datetime(2026, 9, 30, 9, 0, tzinfo=IST)))
    assert rig.svc.describe_row(one, T0) == "call mom (tomorrow at 9:00 AM)"

    hol = rig.db.get_reminder(rig.add("study", datetime(2026, 9, 29, 18, 0, tzinfo=IST), skip_holidays=True))
    assert rig.svc.describe_row(hol, T0).endswith("(skips holidays)")

    zoned = rig.db.get_reminder(rig.add("bank", datetime(2026, 9, 29, 9, 0, tzinfo=NY), tz_name="America/New_York"))
    assert "(America/New_York)" in rig.svc.describe_row(zoned, T0)

    rec = rig.db.get_reminder(rig.add("vitamins", datetime(2026, 9, 29, 7, 0, tzinfo=IST), rule=rig.daily(7)))
    assert rig.svc.describe_row(rec, T0) == "vitamins (every day at 7:00 AM; next today at 7:00 AM)"

    loc = rig.db.get_reminder(place_reminder(rig))
    assert rig.svc.describe_row(loc, T0) == "buy milk (when you get to home)"


def _fired(text, due, missed=False, place=None):
    return Fired(id=1, text=text, fired_at=T0, tz=IST, due=due, missed=missed, place=place)


def test_format_fired_on_time_and_arrival(rig):
    lines = rig.svc.format_fired([_fired("stretch", T0), _fired("milk", None, place="home")], T0)
    assert lines == ["Reminder: stretch", "You've arrived at home. Reminder: milk"]


def test_format_fired_single_missed(rig):
    (line,) = rig.svc.format_fired([_fired("stretch", T0 - timedelta(hours=1), missed=True)], T0)
    assert line == "You missed a reminder from today at 5:00 AM: stretch."


def test_format_fired_digest_of_many_missed_caps_at_five(rig):
    fired = [_fired(f"r{i}", T0 - timedelta(hours=i + 1), missed=True) for i in range(7)]
    (line,) = rig.svc.format_fired(fired, T0)
    assert line.startswith("While you were away, you missed 7 reminders: ")
    assert "r4" in line and "r5" not in line and line.endswith("and 2 more.")


def test_format_fired_orders_on_time_before_missed_digest(rig):
    lines = rig.svc.format_fired(
        [_fired("late", T0 - timedelta(hours=2), missed=True), _fired("now", T0)], T0)
    assert lines[0] == "Reminder: now" and "missed" in lines[1]


def test_fired_to_dict(rig):
    d = _fired("x", T0).to_dict()
    assert d["text"] == "x" and d["due"] == T0.isoformat() and d["place"] is None


# ---------------------------------------------------------------------------
# ICS glue (#90)
# ---------------------------------------------------------------------------

def test_export_lists_pending_only_and_notes_location_reminders(rig):
    rig.add("one-off", T0 + timedelta(hours=1))
    gone = rig.add("cancelled", T0 + timedelta(hours=2))
    rig.svc.cancel(gone)
    place_reminder(rig)
    text, count, notes = rig.svc.export_ics(T0)
    assert count == 1 and "SUMMARY:one-off" in text and "cancelled" not in text
    assert any("location reminder" in n for n in notes)


def test_export_keeps_wall_clock_zone_for_recurring_and_zoned_reminders(rig):
    rig.add("vitamins", datetime(2026, 9, 29, 7, 0, tzinfo=IST), rule=rig.daily(7))
    rig.add("bank", datetime(2026, 9, 29, 9, 0, tzinfo=NY), tz_name="America/New_York")
    text, _, _ = rig.svc.export_ics(T0)
    assert "DTSTART;TZID=Asia/Kolkata:20260929T070000" in text and "RRULE:FREQ=DAILY" in text
    assert "DTSTART;TZID=America/New_York:20260929T090000" in text


def test_export_notes_schedules_with_no_calendar_equivalent(rig):
    from modules.chronos.recurrence import parse_cron
    cron = parse_cron("*/5 9-17 * * 1-5").with_anchor(datetime(2026, 9, 29, 9, 0, tzinfo=IST))
    rig.add("busy", datetime(2026, 9, 29, 9, 5, tzinfo=IST), rule=cron)
    text, count, notes = rig.svc.export_ics(T0)
    assert count == 1 and "RRULE" not in text
    assert any("no calendar equivalent" in n for n in notes)


def _import(rig, *props, alarm=None, now=T0):
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "BEGIN:VEVENT", *props]
    if alarm:
        lines += ["BEGIN:VALARM", alarm, "END:VALARM"]
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return rig.svc.import_ics("\r\n".join(lines) + "\r\n", now=now)


def test_import_future_utc_event(rig):
    res = _import(rig, "UID:ext-1", "SUMMARY:Dentist", "DTSTART:20261001T043000Z")
    assert [c["text"] for c in res.created] == ["Dentist"] and res.skipped == []
    row = rig.db.list_reminders()[0]
    assert parse_iso(row["due_time"]) == datetime(2026, 10, 1, 4, 30, tzinfo=UTC)
    assert row["ics_uid"] == "ext-1"


def test_import_floating_time_uses_the_default_zone(rig):
    _import(rig, "SUMMARY:x", "DTSTART:20261001T090000")
    assert parse_iso(rig.db.list_reminders()[0]["due_time"]) == datetime(2026, 10, 1, 9, 0, tzinfo=IST)


def test_import_tzid_is_stored_on_the_reminder(rig):
    _import(rig, "SUMMARY:x", "DTSTART;TZID=America/New_York:20261001T090000")
    row = rig.db.list_reminders()[0]
    assert row["tz"] == "America/New_York"
    assert parse_iso(row["due_time"]).astimezone(NY).hour == 9


def test_import_all_day_reminds_at_nine_with_a_warning(rig):
    res = _import(rig, "SUMMARY:Holiday", "DTSTART;VALUE=DATE:20261001")
    assert any("all-day" in w for w in res.warnings)
    assert parse_iso(rig.db.list_reminders()[0]["due_time"]) == datetime(2026, 10, 1, 9, 0, tzinfo=IST)


def test_import_applies_alarm_lead_time(rig):
    _import(rig, "SUMMARY:Flight", "DTSTART:20261001T090000Z", alarm="TRIGGER:-PT2H")
    assert parse_iso(rig.db.list_reminders()[0]["due_time"]) == datetime(2026, 10, 1, 7, 0, tzinfo=UTC)


def test_import_past_one_shot_is_skipped(rig):
    res = _import(rig, "SUMMARY:Old", "DTSTART:20250101T090000Z")
    assert res.created == [] and "in the past" in res.skipped[0]


def test_import_recurring_event_starts_at_the_next_occurrence(rig):
    res = _import(rig, "SUMMARY:Standup", "DTSTART;TZID=Asia/Kolkata:20250101T100000",
                  "RRULE:FREQ=DAILY")
    assert len(res.created) == 1
    row = rig.db.list_reminders()[0]
    assert parse_iso(row["due_time"]) == datetime(2026, 9, 29, 10, 0, tzinfo=IST)
    assert Recurrence.from_json(row["recurrence"]).freq == "daily"


def test_import_unsupported_rrule_falls_back_to_single_reminder_with_warning(rig):
    res = _import(rig, "SUMMARY:Monthly review", "DTSTART:20261005T090000Z",
                  "RRULE:FREQ=MONTHLY;BYDAY=1MO")
    assert len(res.created) == 1 and res.created[0]["recurrence"] is None
    assert any("can't represent" in w for w in res.warnings)
    assert rig.db.list_reminders()[0]["recurrence"] is None


def test_import_alert_days_before_repeating_event_is_dropped_with_warning(rig):
    res = _import(rig, "SUMMARY:x", "DTSTART;TZID=Asia/Kolkata:20261001T090000", "RRULE:FREQ=WEEKLY",
                  alarm="TRIGGER:-P2D")
    assert any("more than a day" in w for w in res.warnings)
    assert parse_iso(rig.db.list_reminders()[0]["due_time"]) == datetime(2026, 10, 1, 9, 0, tzinfo=IST)


def test_import_finished_series_is_skipped(rig):
    res = _import(rig, "SUMMARY:Old series", "DTSTART;TZID=Asia/Kolkata:20250101T100000",
                  "RRULE:FREQ=DAILY;UNTIL=20250201T000000Z")
    assert res.created == [] and "already finished" in res.skipped[0]


def test_import_twice_by_uid_or_by_text_and_time_does_not_duplicate(rig):
    props = ("UID:ext-9", "SUMMARY:Dentist", "DTSTART:20261001T043000Z")
    _import(rig, *props)
    again = _import(rig, *props)
    assert again.created == [] and "already in your reminders" in again.skipped[0]
    no_uid = _import(rig, "SUMMARY:dentist", "DTSTART:20261001T043000Z")
    assert no_uid.created == [] and len(rig.db.list_reminders()) == 1


def test_import_own_export_does_not_duplicate(rig):
    rig.add("vitamins", datetime(2026, 9, 29, 7, 0, tzinfo=IST), rule=rig.daily(7))
    text, _, _ = rig.svc.export_ics(T0)
    res = rig.svc.import_ics(text, now=T0)
    assert res.created == [] and len(rig.db.list_reminders()) == 1


def test_import_garbage_reports_a_warning_and_creates_nothing(rig):
    res = rig.svc.import_ics("this is not a calendar", now=T0)
    assert res.created == [] and res.warnings


def test_import_one_bad_entry_does_not_sink_the_file(rig):
    text = ("BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\nSUMMARY:bad\r\nDTSTART:nonsense\r\nEND:VEVENT\r\n"
            "BEGIN:VEVENT\r\nSUMMARY:good\r\nDTSTART:20261001T090000Z\r\nEND:VEVENT\r\n"
            "END:VCALENDAR\r\n")
    res = rig.svc.import_ics(text, now=T0)
    assert [c["text"] for c in res.created] == ["good"] and res.warnings


def test_import_skip_holidays_flag_round_trips(rig):
    _import(rig, "SUMMARY:study", "DTSTART:20261001T090000Z", "X-HESTIA-SKIP-HOLIDAYS:TRUE")
    assert rig.db.list_reminders()[0]["skip_holidays"] == 1
