# tests/test_chronos_ics.py
"""
Standalone tests for modules/chronos/ics.py (backlog #90).

Pure text <-> dict/ParsedItem conversion: no database, no clock, no network.

Run with:  pytest tests/test_chronos_ics.py -v
"""
import os
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chronos import ics  # noqa: E402

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 29, 1, 30, tzinfo=UTC)


def wrap(*components):
    return "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n" + "\r\n".join(components) + "\r\nEND:VCALENDAR\r\n"


def event(*props, alarm=None):
    lines = ["BEGIN:VEVENT", *props]
    if alarm:
        lines += ["BEGIN:VALARM", "ACTION:DISPLAY", alarm, "END:VALARM"]
    lines.append("END:VEVENT")
    return "\r\n".join(lines)


def parse_one(*props, alarm=None):
    items, notes = ics.parse_ics(wrap(event(*props, alarm=alarm)))
    assert len(items) == 1, notes
    return items[0]


# -- text escaping -----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "plain", "a, b; c", "back\\slash", "line1\nline2", "mix, of; all\\ the\nthings", "",
])
def test_escape_unescape_round_trip(text):
    assert ics.unescape_text(ics.escape_text(text)) == text


def test_escape_specific_characters():
    assert ics.escape_text("a,b;c") == "a\\,b\\;c"
    assert ics.escape_text("x\ny") == "x\\ny"
    assert ics.escape_text(None) == ""


def test_unescape_handles_uppercase_n_and_trailing_backslash():
    assert ics.unescape_text("a\\Nb") == "a\nb"
    assert ics.unescape_text("end\\") == "end\\"


# -- folding -----------------------------------------------------------------

def test_short_lines_are_not_folded():
    assert ics.fold_line("SUMMARY:short") == "SUMMARY:short"


def test_long_lines_fold_at_75_octets_and_unfold_back():
    line = "SUMMARY:" + "x" * 300
    folded = ics.fold_line(line)
    segments = folded.split("\r\n")
    assert all(len(s.encode()) <= 75 for s in segments)
    assert all(s.startswith(" ") for s in segments[1:])
    assert ics.unfold(folded) == [line]


def test_folding_never_splits_a_multibyte_character():
    line = "SUMMARY:" + "नमस्ते " * 40 + "😀" * 30
    folded = ics.fold_line(line)
    for seg in folded.split("\r\n"):
        assert len(seg.encode("utf-8")) <= 75
        seg.encode("utf-8").decode("utf-8")           # would raise on a split character
    assert ics.unfold(folded) == [line]


def test_unfold_tolerates_lf_cr_and_tabs():
    assert ics.unfold("A:1\nB:2\n\tcont\rC:3") == ["A:1", "B:2cont", "C:3"]
    assert ics.unfold("") == []


# -- durations ---------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("-PT15M", -timedelta(minutes=15)),
    ("PT0S", timedelta(0)),
    ("-PT0S", timedelta(0)),
    ("P0D", timedelta(0)),
    ("-P1DT2H", -timedelta(days=1, hours=2)),
    ("P1W", timedelta(weeks=1)),
    ("+PT30S", timedelta(seconds=30)),
    ("PT1H30M", timedelta(hours=1, minutes=30)),
])
def test_parse_duration(value, expected):
    assert ics.parse_duration(value) == expected


@pytest.mark.parametrize("value", ["", "15 minutes", "PT", "banana", "P"])
def test_parse_duration_rejects_garbage(value):
    assert ics.parse_duration(value) is None


# -- property splitting ------------------------------------------------------

def test_split_property_with_params():
    name, params, value = ics._split_property("DTSTART;TZID=Asia/Kolkata:20260929T090000")
    assert (name, params, value) == ("DTSTART", {"TZID": "Asia/Kolkata"}, "20260929T090000")


def test_split_property_colon_inside_quotes_does_not_split():
    name, params, value = ics._split_property('ATTENDEE;CN="Doe: John":mailto:j@x.com')
    assert name == "ATTENDEE" and params["CN"] == "Doe: John" and value == "mailto:j@x.com"


def test_split_property_without_colon():
    assert ics._split_property("junk") == ("JUNK", {}, "")


# -- building ----------------------------------------------------------------

def _build(**item):
    base = {"uid": "hestia-reminder-1@hestia.local", "summary": "Stretch",
            "start": datetime(2026, 9, 29, 3, 30, tzinfo=UTC)}
    base.update(item)
    return ics.build_ics([base], now=NOW)


def test_build_has_calendar_envelope_and_crlf():
    text = _build()
    assert text.startswith("BEGIN:VCALENDAR\r\nVERSION:2.0\r\n")
    assert text.endswith("END:VCALENDAR\r\n")
    assert "\n" not in text.replace("\r\n", "")
    assert "PRODID:" + ics.PRODID in text
    assert "DTSTAMP:20260929T013000Z" in text


def test_build_utc_start_when_no_tzid():
    assert "DTSTART:20260929T033000Z" in _build()


def test_build_local_start_when_tzid_given():
    text = _build(tzid="Asia/Kolkata")
    assert "DTSTART;TZID=Asia/Kolkata:20260929T090000" in text


def test_build_includes_rrule_description_and_skip_flag():
    text = _build(rrule="FREQ=DAILY", description="a, b", skip_holidays=True)
    assert "RRULE:FREQ=DAILY" in text
    assert "DESCRIPTION:a\\, b" in text
    assert "X-HESTIA-SKIP-HOLIDAYS:TRUE" in text


def test_build_always_adds_a_display_alarm_at_the_start():
    text = _build()
    assert "BEGIN:VALARM" in text and "TRIGGER:PT0S" in text and "ACTION:DISPLAY" in text


def test_build_escapes_summary_and_defaults_empty_one():
    assert "SUMMARY:buy milk\\, eggs" in _build(summary="buy milk, eggs")
    assert "SUMMARY:Reminder" in _build(summary="")


def test_build_empty_calendar_is_still_valid():
    text = ics.build_ics([], now=NOW)
    items, notes = ics.parse_ics(text)
    assert items == [] and "no events" in notes[0]


def test_build_then_parse_round_trip():
    text = ics.build_ics([
        {"uid": ics.make_uid(7), "summary": "Pay rent, now; ok",
         "start": datetime(2026, 10, 1, 4, 0, tzinfo=UTC), "tzid": "Asia/Kolkata",
         "rrule": "FREQ=MONTHLY;BYMONTHDAY=1", "skip_holidays": True},
    ], now=NOW)
    (item,), notes = ics.parse_ics(text)
    assert notes == []
    assert item.summary == "Pay rent, now; ok"
    assert item.start == datetime(2026, 10, 1, 9, 30, tzinfo=IST)
    assert item.tzid == "Asia/Kolkata" and item.rrule == "FREQ=MONTHLY;BYMONTHDAY=1"
    assert item.skip_holidays is True
    assert item.alarm_offset == timedelta(0)
    assert ics.own_reminder_id(item.uid) == 7


# -- parsing: date forms -----------------------------------------------------

def test_parse_utc_datetime():
    it = parse_one("UID:a", "SUMMARY:x", "DTSTART:20260929T033000Z")
    assert it.start == datetime(2026, 9, 29, 3, 30, tzinfo=UTC) and it.utc and not it.all_day


def test_parse_tzid_datetime():
    it = parse_one("SUMMARY:x", "DTSTART;TZID=America/New_York:20260929T090000")
    assert it.tzid == "America/New_York" and it.start.utcoffset() == timedelta(hours=-4)


def test_parse_floating_datetime_is_naive():
    it = parse_one("SUMMARY:x", "DTSTART:20260929T090000")
    assert it.start.tzinfo is None and it.tzid is None and not it.utc


def test_parse_all_day_date_forms():
    a = parse_one("SUMMARY:x", "DTSTART;VALUE=DATE:20260929")
    b = parse_one("SUMMARY:x", "DTSTART:20260929")
    assert a.all_day and a.start == date(2026, 9, 29)
    assert b.all_day and b.start == date(2026, 9, 29)


def test_unknown_timezone_becomes_floating_with_a_warning():
    it = parse_one("SUMMARY:x", "DTSTART;TZID=Eastern Standard Time:20260929T090000")
    assert it.start.tzinfo is None and it.tzid is None
    assert any("Unknown time zone" in w for w in it.warnings)


def test_unreadable_date_skips_the_entry_with_a_note():
    items, notes = ics.parse_ics(wrap(event("SUMMARY:x", "DTSTART:garbage")))
    assert items == [] and any("unreadable date" in n for n in notes)


# -- parsing: what is skipped or flagged ------------------------------------

def test_cancelled_events_are_skipped_silently():
    items, notes = ics.parse_ics(wrap(event("SUMMARY:x", "DTSTART:20260929T090000Z", "STATUS:CANCELLED")))
    assert items == []


def test_recurrence_id_overrides_are_skipped():
    items, _ = ics.parse_ics(wrap(event(
        "SUMMARY:x", "DTSTART:20260929T090000Z", "RECURRENCE-ID:20260929T090000Z")))
    assert items == []


def test_event_without_title_or_start_is_reported():
    _, n1 = ics.parse_ics(wrap(event("DTSTART:20260929T090000Z")))
    _, n2 = ics.parse_ics(wrap(event("SUMMARY:x")))
    assert any("no title" in n for n in n1)
    assert any("no start time" in n for n in n2)


def test_description_is_used_when_summary_is_missing():
    it = parse_one("DESCRIPTION:call mom", "DTSTART:20260929T090000Z")
    assert it.summary == "call mom"


def test_multiline_summary_keeps_first_line_and_is_capped():
    it = parse_one("SUMMARY:first\\nsecond", "DTSTART:20260929T090000Z")
    assert it.summary == "first"
    assert len(parse_one("SUMMARY:" + "z" * 500, "DTSTART:20260929T090000Z").summary) == 300


def test_exdate_and_rdate_produce_warnings_but_still_import():
    it = parse_one("SUMMARY:x", "DTSTART:20260929T090000Z", "RRULE:FREQ=DAILY",
                   "EXDATE:20261001T090000Z", "RDATE:20261002T090000Z")
    assert it.rrule == "FREQ=DAILY"
    assert any("EXDATE" in w for w in it.warnings) and any("RDATE" in w for w in it.warnings)


def test_vtodo_prefers_due_over_dtstart():
    text = wrap("BEGIN:VTODO", "SUMMARY:file taxes", "DTSTART:20260901T090000Z",
                "DUE:20260930T090000Z", "END:VTODO")
    (it,), _ = ics.parse_ics(text)
    assert it.kind == "VTODO" and it.start == datetime(2026, 9, 30, 9, 0, tzinfo=UTC)


def test_skip_holidays_flag_is_read_back():
    assert parse_one("SUMMARY:x", "DTSTART:20260929T090000Z",
                     "X-HESTIA-SKIP-HOLIDAYS:TRUE").skip_holidays is True
    assert parse_one("SUMMARY:x", "DTSTART:20260929T090000Z").skip_holidays is False


# -- parsing: alarms ---------------------------------------------------------

def test_relative_alarm_before_start_becomes_lead_time():
    it = parse_one("SUMMARY:x", "DTSTART:20260929T090000Z", alarm="TRIGGER:-PT15M")
    assert it.alarm_offset == -timedelta(minutes=15)


def test_alarm_after_start_is_not_a_lead_time():
    it = parse_one("SUMMARY:x", "DTSTART:20260929T090000Z", alarm="TRIGGER:PT15M")
    assert it.alarm_offset is None


def test_absolute_and_end_related_alarms_are_ignored():
    a = parse_one("SUMMARY:x", "DTSTART:20260929T090000Z",
                  alarm="TRIGGER;VALUE=DATE-TIME:20260929T080000Z")
    b = parse_one("SUMMARY:x", "DTSTART:20260929T090000Z",
                  alarm="TRIGGER;RELATED=END:-PT15M")
    assert a.alarm_offset is None and b.alarm_offset is None


def test_earliest_of_several_alarms_wins():
    text = wrap("BEGIN:VEVENT", "SUMMARY:x", "DTSTART:20260929T090000Z",
                "BEGIN:VALARM", "TRIGGER:-PT10M", "END:VALARM",
                "BEGIN:VALARM", "TRIGGER:-PT1H", "END:VALARM", "END:VEVENT")
    (it,), _ = ics.parse_ics(text)
    assert it.alarm_offset == -timedelta(hours=1)


def test_alarm_properties_do_not_leak_into_the_event():
    it = parse_one("SUMMARY:real title", "DTSTART:20260929T090000Z",
                   alarm="DESCRIPTION:alarm text")
    assert it.summary == "real title"


# -- parsing: document level -------------------------------------------------

@pytest.mark.parametrize("text", ["", None, "hello world", "BEGIN:VEVENT\r\nEND:VEVENT"])
def test_non_calendars_are_rejected_without_raising(text):
    items, notes = ics.parse_ics(text)
    assert items == [] and "doesn't look like" in notes[0]


def test_one_bad_entry_does_not_sink_the_others():
    text = wrap(event("SUMMARY:bad", "DTSTART:nope"),
                event("SUMMARY:good", "DTSTART:20260929T090000Z"))
    items, notes = ics.parse_ics(text)
    assert [i.summary for i in items] == ["good"] and len(notes) == 1


def test_parses_lf_only_and_folded_input():
    text = ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:very lo\n ng title\n"
            "DTSTART:20260929T090000Z\nEND:VEVENT\nEND:VCALENDAR\n")
    (it,), _ = ics.parse_ics(text)
    assert it.summary == "very long title"


def test_lowercase_keywords_are_accepted():
    text = "begin:vcalendar\nbegin:vevent\nsummary:x\ndtstart:20260929T090000Z\nend:vevent\nend:vcalendar\n"
    (it,), _ = ics.parse_ics(text)
    assert it.summary == "x"


# -- uids --------------------------------------------------------------------

def test_uid_round_trip_and_rejection():
    assert ics.own_reminder_id(ics.make_uid(42)) == 42
    assert ics.own_reminder_id(f" {ics.make_uid(3)} ") == 3
    for bad in (None, "", "abc@example.com", "hestia-reminder-x@hestia.local",
                "hestia-reminder-5@other.local"):
        assert ics.own_reminder_id(bad) is None
