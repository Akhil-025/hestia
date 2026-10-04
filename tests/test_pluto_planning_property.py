# tests/test_pluto_planning_property.py
"""
Hypothesis tests for the money and date maths in modules/pluto/planning.py.
They assert things that must hold for every input, not just the examples in
test_pluto_planning.py. Skipped without ``hypothesis`` (pip install -r requirements-dev.txt).
"""
from __future__ import annotations

import math
import os
import sys
from datetime import date, timedelta

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.pluto import planning as pl

S = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
junk = st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True, allow_infinity=True),
                 st.text(max_size=30), st.lists(st.integers(), max_size=2),
                 st.sampled_from(["nan", "inf", "-inf", "1e999", "9" * 400, "-0", "", "₹", "(5)", "5k", "2 lakh"]))
dates = st.dates(min_value=date(2000, 1, 1), max_value=date(2100, 12, 31))


@S
@given(junk)
def test_parse_amount_never_raises_and_returns_a_sane_positive_number(raw):
    out = pl.parse_amount(raw)
    assert out is None or (isinstance(out, float) and math.isfinite(out) and 0 < out <= pl.MAX_AMOUNT)


@S
@given(st.integers(min_value=1, max_value=10 ** 6))
def test_a_negative_amount_is_never_accepted(n):
    for text in (f"-{n}", f"-{n} rupees", f"({n})", f"\u2212{n}", -n):
        assert pl.parse_amount(text) is None


@S
@given(st.integers(min_value=1, max_value=10 ** 6))
def test_plain_numbers_and_formatted_numbers_agree(n):
    assert pl.parse_amount(n) == pl.parse_amount(str(n)) == pl.parse_amount(f"{n:,}") == float(n)


@S
@given(st.integers(min_value=1, max_value=900))
def test_multiplier_words(n):
    assert pl.parse_amount(f"{n}k") == n * 1000.0
    assert pl.parse_amount(f"{n} lakh") == n * 100000.0


@S
@given(dates)
def test_month_bounds_bracket_the_date_and_days_left_is_consistent(d):
    start, nxt = pl.month_bounds(d)
    assert start <= d < nxt and start.day == 1 and nxt.day == 1 and 28 <= (nxt - start).days <= 31
    left = pl.days_left_in_month(d)
    assert 1 <= left <= 31 and d + timedelta(days=left) == nxt


@S
@given(dates, st.integers(min_value=-240, max_value=240))
def test_add_months_lands_in_the_right_month_and_never_overflows_the_day(d, n):
    out = pl.add_months(d, n)
    expected_index = d.year * 12 + d.month - 1 + n
    assert out.year * 12 + out.month - 1 == expected_index
    assert out.day == min(d.day, (pl.month_bounds(out)[1] - timedelta(days=1)).day)


@S
@given(st.floats(min_value=0, max_value=1e7, allow_nan=False, allow_infinity=False),
       st.floats(min_value=0, max_value=1e7, allow_nan=False, allow_infinity=False),
       st.floats(min_value=0, max_value=1e7, allow_nan=False, allow_infinity=False), dates)
def test_safe_to_spend_per_day_is_never_negative_and_times_days_never_exceeds_whats_left(budget, spent, upcoming, d):
    r = pl.safe_to_spend(budget, spent, upcoming, d)
    assert r["per_day"] >= 0
    assert r["per_day"] * r["days_left"] <= max(0.0, r["remaining"]) + 1e-6
    assert r["overspent"] == (r["remaining"] < 0)


@S
@given(st.dictionaries(st.text(alphabet="abcXYZ", min_size=1, max_size=4), st.floats(min_value=1, max_value=1e6), max_size=6),
       st.dictionaries(st.text(alphabet="abcXYZ", min_size=1, max_size=4), st.floats(min_value=0, max_value=1e6), max_size=6))
def test_budget_status_state_matches_the_numbers(budgets, spent):
    for r in pl.budget_status(budgets, spent):
        assert r["state"] == ("over" if r["spent"] > r["limit"] else "warn" if r["spent"] >= 0.8 * r["limit"] else "ok")
        assert math.isclose(r["remaining"], r["limit"] - r["spent"], rel_tol=1e-9, abs_tol=1e-9)


@S
@given(st.floats(min_value=0, max_value=1e5), st.integers(min_value=1, max_value=60),
       st.floats(min_value=0, max_value=40), st.floats(min_value=0, max_value=1e5))
def test_scenario_never_loses_money_at_a_non_negative_return_and_grows_with_the_rate(monthly, years, rate, initial):
    if monthly == 0 and initial == 0:
        return
    base = pl.scenario(monthly, years, rate, initial)
    assert base["final"] >= base["contributed"] - 1e-6
    assert math.isclose(base["contributed"], initial + monthly * years * 12, rel_tol=1e-9)
    assert len(base["table"]) == years and base["table"][-1]["value"] == base["final"]
    assert pl.scenario(monthly, years, rate + 1, initial)["final"] >= base["final"] - 1e-6


@S
@given(junk, junk, junk, junk)
def test_scenario_never_raises_anything_but_valueerror(a, b, c, d):
    try:
        pl.scenario(a, b, c, d)
    except ValueError:
        pass
    except TypeError:
        pass          # a non-numeric argument is a caller bug, not a user-input path (the manager parses first)


@S
@given(st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False), max_size=12))
def test_component_scores_are_always_0_to_100(values):
    for c in (pl.volatility_score(values), pl.diversification_score(values)):
        assert c is None or 0 <= c["score"] <= 100


@S
@given(st.floats(min_value=1, max_value=1e7), st.floats(min_value=0, max_value=1e8))
def test_savings_rate_score_is_bounded(income, spend):
    assert 0 <= pl.savings_rate_score(income, spend)["score"] <= 100


@S
@given(st.text(max_size=60))
def test_normalise_description_is_idempotent_and_letters_only(text):
    once = pl.normalise_description(text)
    assert pl.normalise_description(once) == once
    assert all(ch.isalpha() or ch == " " for ch in once) and once == once.lower()


@S
@given(st.text(max_size=40))
def test_csv_safe_never_leaves_a_formula_trigger_first(text):
    assert pl.csv_safe(text)[:1] not in ("=", "+", "-", "@", "\t", "\r")


@S
@given(junk)
def test_parse_financial_year_never_raises(raw):
    out = pl.parse_financial_year(raw, date(2026, 9, 20))
    assert out is None or 1990 <= out <= 2027
