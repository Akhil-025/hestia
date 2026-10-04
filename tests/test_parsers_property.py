# tests/test_parsers_property.py
"""
Property-based tests (Hypothesis) for the parsing-heavy Apollo functions
(backlog #208).

Example tests check inputs somebody thought of. These generate thousands of
inputs, including hostile ones, and check things that must hold for ALL of
them:

* a parser never raises, whatever it is given;
* it returns either a valid value inside the documented range or an error
  message, never both and never neither;
* unit conversion is consistent (the same physical amount is accepted or
  rejected whatever unit it is written in);
* negative input is never accepted as positive.

The first run of these found two real bugs, both fixed in the same change:
"-30 min" was logged as a 30-minute workout (the sign was dropped), and a
400-digit number made ``_parse_duration`` raise OverflowError.
"""
from __future__ import annotations

import math
import os
import sys
from datetime import date

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import modules.apollo.engine as ap

SETTINGS = settings(max_examples=300, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])

# Anything the NLU could plausibly hand over: numbers, strings, None, bools,
# containers, and the usual float landmines.
junk = st.one_of(
    st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=40),
    st.sampled_from(["nan", "inf", "-inf", "1e999", "9" * 400, "-0", "", " ", "\x00", "٣", "½",
                     "−5", "--5", "7-8", "1,5", "5.", "+7", "0x10", "1_000"]),
    st.lists(st.integers(), max_size=3), st.dictionaries(st.text(max_size=3), st.integers(), max_size=2),
)
numberish = st.one_of(
    st.integers(min_value=-10**6, max_value=10**6),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
unit_words = st.sampled_from(["", " min", " minutes", " hours", " kg", " lb", " ml", " glasses"])


def _is_ok_or_error(result, ok_check):
    value, error = result
    if error is None:
        assert ok_check(value), result
    else:
        assert isinstance(error, str) and error, result
    return error is None


# ---------------------------------------------------------------- _extract_number

@SETTINGS
@given(junk, st.booleans())
def test_extract_number_never_raises_and_is_finite(raw, signed):
    out = ap._extract_number(raw, signed=signed)
    assert out is None or (isinstance(out, float) and math.isfinite(out))


@SETTINGS
@given(st.integers(min_value=0, max_value=10**6), unit_words)
def test_extract_number_round_trips_a_plain_number_with_trailing_words(n, tail):
    assert ap._extract_number(f"{n}{tail}") == float(n)
    assert ap._extract_number(f"-{n}{tail}", signed=True) == -float(n)


@SETTINGS
@given(st.integers(min_value=1, max_value=999), st.integers(min_value=1, max_value=999))
def test_a_range_is_not_read_as_a_negative_number(a, b):
    assert ap._extract_number(f"{a}-{b}", signed=True) == float(a)


# ------------------------------------------------------------------- durations

@SETTINGS
@given(junk)
def test_duration_never_raises_and_stays_in_range(raw):
    ok = _is_ok_or_error(
        ap._parse_duration(raw),
        lambda v: isinstance(v, int) and ap._MIN_WORKOUT_DURATION <= v <= ap._MAX_WORKOUT_DURATION,
    )
    assert ok or raw is not None


@SETTINGS
@given(st.integers(min_value=1, max_value=10**6))
def test_a_negative_duration_is_never_accepted(n):
    assert ap._parse_duration(f"-{n}")[1] is not None
    assert ap._parse_duration(f"-{n} min")[1] is not None


@SETTINGS
@given(st.integers(min_value=ap._MIN_WORKOUT_DURATION, max_value=ap._MAX_WORKOUT_DURATION), unit_words)
def test_every_in_range_duration_is_accepted_unchanged(n, tail):
    assert ap._parse_duration(f"{n}{tail}") == (n, None)


@SETTINGS
@given(st.integers(min_value=ap._MAX_WORKOUT_DURATION + 1, max_value=10**9))
def test_too_long_a_duration_is_rejected(n):
    assert ap._parse_duration(n)[1] is not None


def test_no_duration_means_the_default():
    assert ap._parse_duration(None) == (ap._DEFAULT_WORKOUT_DURATION, None)


# ----------------------------------------------------------------------- sleep

@SETTINGS
@given(junk)
def test_sleep_hours_never_raises_and_stays_in_range(raw):
    _is_ok_or_error(ap._parse_hours(raw),
                    lambda v: ap._MIN_SLEEP_HOURS <= v <= ap._MAX_SLEEP_HOURS)


@SETTINGS
@given(st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False))
def test_negative_sleep_is_never_accepted(x):
    assert ap._parse_hours(f"-{x:.2f}")[1] is not None


@SETTINGS
@given(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
def test_sleep_comment_is_always_one_of_the_known_comments(h):
    assert ap._sleep_comment(h) in ap._SLEEP_COMMENTS.values()


# ---------------------------------------------------------------------- weight

@SETTINGS
@given(junk, st.sampled_from(["kg", "lb"]))
def test_weight_never_raises_and_result_is_in_kg_range(raw, unit):
    _is_ok_or_error(ap._parse_weight(raw, unit),
                    lambda v: ap._MIN_WEIGHT_KG <= v <= ap._MAX_WEIGHT_KG)


@SETTINGS
@given(st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False))
def test_weight_limits_are_the_same_physical_amount_in_either_unit(kg):
    """Accepting 70 kg but rejecting 154.3 lb (or the reverse) would mean the
    range check depends on how the user phrased it."""
    lb = kg / ap._KG_PER_LB
    # stay clear of the boundary, where rounding of the repr could flip it
    if abs(kg - ap._MIN_WEIGHT_KG) < 0.01 or abs(kg - ap._MAX_WEIGHT_KG) < 0.01:
        return
    in_kg = ap._parse_weight(repr(kg), "kg")[1] is None
    in_lb = ap._parse_weight(repr(lb), "lb")[1] is None
    assert in_kg == in_lb


@SETTINGS
@given(st.floats(min_value=ap._MIN_WEIGHT_KG, max_value=ap._MAX_WEIGHT_KG,
                 allow_nan=False, allow_infinity=False))
def test_kg_to_unit_inverts_the_lb_conversion(kg):
    lb = ap._kg_to_unit(kg, "lb")
    assert ap._kg_to_unit(kg, "kg") == kg
    assert math.isclose(lb * ap._KG_PER_LB, kg, rel_tol=1e-9)


@SETTINGS
@given(st.text(max_size=12))
def test_weight_unit_is_always_kg_or_lb(raw):
    assert ap._normalize_weight_unit(raw) in ("kg", "lb")


@SETTINGS
@given(st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
       st.sampled_from(["kg", "lb"]))
def test_weight_delta_comment_is_text_with_a_sensible_direction(delta, unit):
    text = ap._weight_delta_comment(delta, unit)
    assert isinstance(text, str) and text
    if abs(delta) >= 0.05:
        assert (" down " in text) == (delta < 0) and (" up " in text) == (delta > 0)


@SETTINGS
@given(st.floats(allow_nan=False, allow_infinity=False))
def test_trend_word(delta):
    assert ap._trend_word(delta) == ("down" if delta < 0 else "up" if delta > 0 else "unchanged")


# ----------------------------------------------------------------------- water

@SETTINGS
@given(junk, st.sampled_from(["ml", "glasses", "glass", "oz", "ounces", "", None, "cups"]))
def test_water_never_raises_and_result_is_whole_ml_in_range(raw, unit):
    _is_ok_or_error(ap._parse_water(raw, unit),
                    lambda v: isinstance(v, int) and ap._MIN_WATER_ML <= v <= ap._MAX_WATER_ML)


@SETTINGS
@given(st.integers(min_value=1, max_value=20))
def test_glasses_convert_at_the_documented_rate(n):
    ml, err = ap._parse_water(n, "glasses")
    if n * ap._ML_PER_GLASS <= ap._MAX_WATER_ML:
        assert (ml, err) == (n * ap._ML_PER_GLASS, None)
    else:
        assert err is not None


@SETTINGS
@given(st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_negative_water_is_never_accepted(x):
    assert ap._parse_water(-x, "ml")[1] is not None


# ----------------------------------------------------------------------- goals

@SETTINGS
@given(junk)
def test_goal_target_never_raises_and_is_in_range(raw):
    _is_ok_or_error(ap._parse_goal_target(raw),
                    lambda v: ap._MIN_GOAL_TARGET <= v <= ap._MAX_GOAL_TARGET)


@SETTINGS
@given(st.text(max_size=30))
def test_goal_type_is_canonical_or_none(raw):
    out = ap._normalize_goal_type(raw)
    assert out is None or out in ap._GOAL_TYPES


# --------------------------------------------------------------------- ratings

@SETTINGS
@given(junk)
def test_rating_is_none_or_1_to_5(raw):
    out = ap._parse_rating(raw)
    assert out is None or (isinstance(out, int) and 1 <= out <= 5)


@SETTINGS
@given(st.integers(min_value=1, max_value=5))
def test_rating_accepts_every_valid_value(n):
    assert ap._parse_rating(n) == n and ap._parse_rating(f"{n} stars") == n


# ----------------------------------------------------------------------- dates

@SETTINGS
@given(junk)
def test_parse_day_never_raises_and_returns_a_date_or_none(raw):
    out = ap._parse_day(raw)
    assert out is None or isinstance(out, date)


@SETTINGS
@given(st.dates(min_value=date(1971, 1, 1), max_value=date(2100, 12, 31)),
       st.sampled_from(["%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"]))
def test_parse_day_round_trips_every_supported_format(d, fmt):
    assert ap._parse_day(d.strftime(fmt)) == d


@SETTINGS
@given(st.dates(min_value=date(1974, 1, 1), max_value=date(2100, 12, 31)))
def test_parse_day_reads_unix_seconds_and_milliseconds(d):
    # _parse_day treats anything above 1e11 as milliseconds, so the two units
    # are only distinguishable from March 1973 on; irrelevant for step data.
    from datetime import datetime, timezone
    secs = int(datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc).timestamp())
    assert ap._parse_day(secs) == d
    assert ap._parse_day(secs * 1000) == d


@SETTINGS
@given(junk)
def test_parse_step_value_is_none_or_a_sane_count(raw):
    out = ap._parse_step_value(raw)
    assert out is None or (isinstance(out, int) and 0 <= out <= ap._MAX_DAILY_STEPS)


# --------------------------------------------------------------------- streaks

@SETTINGS
@given(st.lists(st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 1, 1)), max_size=40))
def test_compute_streak_is_bounded_by_the_distinct_days_given(days):
    strs = sorted({d.isoformat() for d in days}, reverse=True)
    out = ap._compute_streak(strs)
    assert isinstance(out, int) and 0 <= out <= len(strs)
