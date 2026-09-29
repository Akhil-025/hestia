# tests/test_chronos_recurrence.py
"""
Tests for modules/chronos/recurrence.py (backlog #81 recurring reminders,
#84 natural-language rules, #85 per-reminder time zones).

Everything here is pure: no database, no network, no real clock.

Run with:  pytest tests/test_chronos_recurrence.py -v
"""
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chronos.recurrence import (  # noqa: E402
    Recurrence,
    extract_recurrence,
    extract_timezone,
    from_rrule,
    next_after,
    parse_cron,
    parse_duration,
    parse_time_of_day,
    resolve_timezone_name,
    strip_duration,
    to_rrule,
    unsupported_recurrence_reason,
)

IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")
# Tuesday 29 Sep 2026, 06:00 IST.
NOW = datetime(2026, 9, 29, 6, 0, tzinfo=IST)


def parse(text, now=NOW, tz=IST):
    return extract_recurrence(text, now, tz)


@pytest.mark.parametrize(
    "phrase, description",
    [
        ("every weekday at 7am", "every weekday at 7:00 AM"),
        ("every day at 9am", "every day at 9:00 AM"),
        ("daily at 6pm", "every day at 6:00 PM"),
        ("every monday and thursday at 6:30pm", "every Monday, Thursday at 6:30 PM"),
        ("every 2 weeks on monday at 8pm", "every 2 weeks on Monday at 8:00 PM"),
        ("every other day", "every 2 days at 9:00 AM"),
        ("every 3 days at 8am", "every 3 days at 8:00 AM"),
        ("every weekend at 10am", "every weekend day at 10:00 AM"),
        ("every 15 minutes", "every 15 minutes"),
        ("every hour", "every hour"),
        ("every morning", "every day at 9:00 AM"),
        ("on the 1st of every month at 9am", "every month on the 1st at 9:00 AM"),
    ],
)
def test_natural_language_rules(phrase, description):
    rule, rest = parse(f"remind me to stretch {phrase}")
    assert rule is not None, phrase
    assert rule.describe(IST) == description
    assert rest == "remind me to stretch"


def test_no_recurrence_leaves_text_alone():
    rule, rest = parse("remind me to call mom tomorrow at 5pm")
    assert rule is None
    assert rest == "remind me to call mom tomorrow at 5pm"


def test_empty_text_is_safe():
    assert parse("") == (None, "")
    assert parse("   ")[0] is None


def test_month_is_not_read_as_monday():
    rule, rest = parse("remind me to pay rent every month on the 31st")
    assert rule is not None
    assert rule.freq == "monthly"
    assert rule.bymonthday == 31
    assert rest == "remind me to pay rent"


def test_every_month_without_a_day_uses_todays_date():
    rule, _ = parse("remind me to review budget every month")
    assert rule.freq == "monthly"
    assert next_after(rule, NOW, IST).day == 29


def test_yearly_rule_keeps_its_date():
    rule, rest = parse("remind me to renew the domain every year on march 5")
    assert rule.freq == "yearly"
    assert rule.describe(IST) == "every year on March 5 at 9:00 AM"
    assert rest == "remind me to renew the domain"
    first = next_after(rule, NOW, IST)
    assert (first.year, first.month, first.day) == (2027, 3, 5)
    second = next_after(rule, first, IST)
    assert (second.year, second.month, second.day) == (2028, 3, 5)


def test_yearly_day_before_month_and_time():
    rule, _ = parse("every year on 5th of march at 8am")
    first = next_after(rule, NOW, IST)
    assert (first.month, first.day, first.hour) == (3, 5, 8)


def test_yearly_feb_29_clamps_in_non_leap_years():
    rule, _ = parse("every year on feb 29")
    first = next_after(rule, NOW, IST)
    assert (first.year, first.month, first.day) == (2027, 2, 28)
    assert next_after(rule, first, IST).day == 29


def test_until_and_count_limits():
    rule, _ = parse("every day at 9am until 2026-10-01")
    assert rule.until == "2026-10-01"
    last = next_after(rule, datetime(2026, 10, 1, 8, 0, tzinfo=IST), IST)
    assert last.date().isoformat() == "2026-10-01"
    assert next_after(rule, last, IST) is None

    rule, _ = parse("every day at 9am for 3 times")
    assert rule.count == 3


def test_raw_cron_literal():
    rule, rest = parse("remind me to back up 0 7 * * 1-5")
    assert rule.freq == "cron"
    assert next_after(rule, datetime(2026, 10, 2, 7, 0, tzinfo=IST), IST).weekday() == 0
    assert rest == "remind me to back up"


@pytest.mark.parametrize(
    "text",
    [
        "remind me to plan every first monday of the month",
        "first monday of every month",
        "last friday of every month",
        "every 2nd friday",
    ],
)
def test_nth_weekday_is_flagged_as_unsupported(text):
    assert unsupported_recurrence_reason(text)


@pytest.mark.parametrize(
    "text",
    ["every monday", "every month on the 2nd", "every friday evening", "call mom on friday"],
)
def test_plain_rules_are_not_flagged(text):
    assert unsupported_recurrence_reason(text) is None


def test_weekday_rule_skips_the_weekend():
    rule, _ = parse("every weekday at 7am")
    friday = datetime(2026, 10, 2, 7, 0, tzinfo=IST)
    assert next_after(rule, friday, IST) == datetime(2026, 10, 5, 7, 0, tzinfo=IST)


def test_next_after_is_strictly_after():
    rule = Recurrence(freq="daily", hour=9)
    at = datetime(2026, 9, 29, 9, 0, tzinfo=IST)
    assert next_after(rule, at, IST) == at + timedelta(days=1)
    assert next_after(rule, at - timedelta(seconds=1), IST) == at


def test_monthly_31st_clamps_to_short_months():
    rule = Recurrence(freq="monthly", bymonthday=31, hour=9)
    assert next_after(rule, datetime(2026, 1, 31, 9, 0, tzinfo=IST), IST).date().isoformat() == "2026-02-28"
    assert next_after(rule, datetime(2026, 4, 1, 0, 0, tzinfo=IST), IST).date().isoformat() == "2026-04-30"


def test_wall_clock_time_survives_a_dst_change():
    rule = Recurrence(freq="daily", hour=7)
    before = datetime(2026, 10, 31, 7, 0, tzinfo=NY)
    after = next_after(rule, before, NY)
    assert after.hour == 7 and after.date().isoformat() == "2026-11-01"
    assert after.utcoffset() != before.utcoffset()


def test_count_is_consumed_and_ends_the_series():
    rule = Recurrence(freq="daily", hour=9, count=2)
    rule = rule.consume()
    assert rule.count == 1
    assert rule.consume() is None
    assert Recurrence(freq="daily", hour=9).consume().count is None
    assert next_after(Recurrence(freq="daily", hour=9, count=0), NOW, IST) is None


def test_json_round_trip():
    rule, _ = parse("every 2 weeks on monday at 8pm until 2027-01-01")
    assert Recurrence.from_json(rule.to_json()) == rule
    assert Recurrence.from_json(None) is None
    assert Recurrence.from_json("not json") is None


def test_invalid_rules_are_rejected():
    with pytest.raises(ValueError):
        Recurrence(freq="fortnightly")
    with pytest.raises(ValueError):
        Recurrence(freq="daily", hour=25)
    with pytest.raises(ValueError):
        Recurrence(freq="weekly", byday=(7,))
    with pytest.raises(ValueError):
        parse_cron("61 * * * *")


@pytest.mark.parametrize(
    "text, zone, rest",
    [
        ("call at 9am London time", "Europe/London", "call at 9am"),
        ("at 3pm EST", "America/New_York", "at 3pm"),
        ("meet at 9am PST please", "America/Los_Angeles", "meet at 9am please"),
    ],
)
def test_extract_timezone(text, zone, rest):
    assert extract_timezone(text) == (zone, rest)


def test_extract_timezone_none():
    assert extract_timezone("call mom at 9am")[0] is None


def test_resolve_timezone_name():
    assert resolve_timezone_name("Asia/Tokyo") == "Asia/Tokyo"
    assert resolve_timezone_name("Tokyo") == "Asia/Tokyo"
    assert resolve_timezone_name("nowhere-land") is None
    assert resolve_timezone_name(None) is None


@pytest.mark.parametrize(
    "text, hm",
    [("7am", (7, 0)), ("at 18:30", (18, 30)), ("noon", (12, 0)), ("12am", (0, 0)), ("7:15 pm", (19, 15))],
)
def test_parse_time_of_day(text, hm):
    h, m, _ = parse_time_of_day(text)
    assert (h, m) == hm


def test_parse_time_of_day_none():
    assert parse_time_of_day("no time here") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10 minutes", timedelta(minutes=10)),
        ("an hour", timedelta(hours=1)),
        ("half an hour", timedelta(minutes=30)),
        ("1h30m", timedelta(hours=1, minutes=30)),
        ("2 hours and 15 minutes", timedelta(hours=2, minutes=15)),
        ("an hour and a half", timedelta(minutes=90)),
        ("snooze for five minutes", timedelta(minutes=5)),
        ("3 days", timedelta(days=3)),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


def test_parse_duration_none():
    assert parse_duration("") is None
    assert parse_duration("snooze") is None
    assert parse_duration("0 minutes") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("snooze the gym reminder for 10 minutes", "snooze the gym reminder"),
        ("snooze 1h30m", "snooze"),
        ("snooze for 2 hours and 15 minutes", "snooze"),
        ("snooze the call an hour and a half", "snooze the call"),
        ("snooze", "snooze"),
        ("", ""),
    ],
)
def test_strip_duration(text, expected):
    assert strip_duration(text) == expected


def test_rrule_round_trip_weekly():
    rule, _ = parse("every monday and thursday at 6:30pm")
    rr = to_rrule(rule, IST)
    assert rr == "FREQ=WEEKLY;BYDAY=MO,TH"
    back = from_rrule(rr, datetime(2026, 10, 1, 18, 30, tzinfo=IST), IST)
    assert back.byday == (0, 3) and (back.hour, back.minute) == (18, 30)


def test_rrule_rejects_nth_weekday():
    with pytest.raises(ValueError):
        from_rrule("FREQ=MONTHLY;BYDAY=1MO", datetime(2026, 10, 5, 9, 0, tzinfo=IST), IST)


def test_complex_cron_has_no_rrule():
    assert to_rrule(parse_cron("*/5 9-17 * * 1-5"), IST) is None
