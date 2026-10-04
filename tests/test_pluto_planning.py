# tests/test_pluto_planning.py
"""
Pluto planning features (backlog #133, #134, #135, #137, #138, #266) and the
input hardening that came with them.

Pure logic is tested with plain data and an injected "today". The manager is
tested against a real SQLite database in a temp folder and a fake clock. One
test class runs the whole path (NLU -> Hecate -> PlutoEngine -> real database)
with only the model call scripted. Nothing here touches the network, Ollama,
Postgres/Redis/Qdrant, or the real clock.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from modules.pluto import planning as pl
from modules.pluto.db import PlutoDB
from modules.pluto.engine import PlutoEngine

TODAY = date(2026, 9, 20)


def exp(desc, amount, day, cat="Bills"):
    return {"description": desc, "amount": amount, "logged_at": f"{day} 09:00:00", "category": cat}


def monthly(desc, amount, months=((2026, 5), (2026, 6), (2026, 7), (2026, 8), (2026, 9)), d=5, cat="Entertainment"):
    return [exp(desc, amount, date(y, m, d).isoformat(), cat) for y, m in months]


# ------------------------------------------------------------------ amounts

class TestParseAmount:
    @pytest.mark.parametrize("raw,expected", [
        (500, 500.0), (12.5, 12.5), ("500", 500.0), ("₹1,500", 1500.0), ("1500 rupees", 1500.0),
        ("Rs. 2,50,000", 250000.0), ("5k", 5000.0), ("5 K", 5000.0), ("2.5 lakh", 250000.0),
        ("1 lac", 100000.0), ("3 crore", 30000000.0), ("2 thousand", 2000.0),
    ])
    def test_reads_what_people_say(self, raw, expected):
        assert pl.parse_amount(raw) == expected

    @pytest.mark.parametrize("raw", [
        None, True, False, "", "abc", 0, "0", -5, "-500", "(500)", "\u2212500", float("nan"),
        float("inf"), "nan", "inf", "1e999", 10 ** 12, "9" * 400, [], {},
    ])
    def test_rejects_what_it_should(self, raw):
        assert pl.parse_amount(raw) is None

    def test_kg_is_not_thousands(self):
        assert pl.parse_amount("5 kg") == 5.0

    def test_the_cap_is_inclusive(self):
        assert pl.parse_amount(pl.MAX_AMOUNT) == pl.MAX_AMOUNT
        assert pl.parse_amount(pl.MAX_AMOUNT + 1) is None

    def test_money_format(self):
        assert pl.money(12345.6) == "₹12,346" and pl.money(5, "$") == "$5"


# -------------------------------------------------------------------- dates

class TestDates:
    def test_month_bounds_including_december(self):
        assert pl.month_bounds(date(2026, 9, 20)) == (date(2026, 9, 1), date(2026, 10, 1))
        assert pl.month_bounds(date(2026, 12, 31)) == (date(2026, 12, 1), date(2027, 1, 1))

    @pytest.mark.parametrize("d,left", [(date(2026, 9, 20), 11), (date(2026, 9, 30), 1),
                                        (date(2026, 9, 1), 30), (date(2028, 2, 28), 2)])
    def test_days_left_includes_today(self, d, left):
        assert pl.days_left_in_month(d) == left

    def test_financial_year(self):
        assert pl.financial_year_bounds(2025) == (date(2025, 4, 1), date(2026, 4, 1))
        assert pl.current_financial_year_start(date(2026, 3, 31)) == 2025
        assert pl.current_financial_year_start(date(2026, 4, 1)) == 2026

    @pytest.mark.parametrize("raw,expected", [("2025-26", 2025), ("FY2025", 2025), ("2025/26", 2025),
                                              ("", 2026), (None, 2026), ("2024", 2024)])
    def test_parse_financial_year(self, raw, expected):
        assert pl.parse_financial_year(raw, date(2026, 9, 20)) == expected

    @pytest.mark.parametrize("raw", ["last year", "1850", "2099"])
    def test_unusable_years_are_none(self, raw):
        assert pl.parse_financial_year(raw, date(2026, 9, 20)) is None

    @pytest.mark.parametrize("raw,expected", [("2026-09-05 10:00:00", date(2026, 9, 5)), ("2026-09-05", date(2026, 9, 5)),
                                              ("2026-13-45", None), ("junk", None), (None, None)])
    def test_parse_day(self, raw, expected):
        assert pl.parse_day(raw) == expected


# ---------------------------------------------------------------- recurring

class TestRecurring:
    def test_a_monthly_subscription_is_found_with_its_cost(self):
        r = pl.detect_recurring(monthly("Netflix", 649), TODAY)
        assert len(r) == 1
        x = r[0]
        assert (x["cadence"], x["typical_amount"], x["count"], x["monthly_cost"]) == ("monthly", 649, 5, 649)
        assert x["last_date"] == "2026-09-05" and x["next_expected"] == "2026-10-05"
        assert x["active"] is True and x["price_changed"] is False

    def test_two_charges_are_not_a_pattern(self):
        assert pl.detect_recurring(monthly("Netflix", 649, months=((2026, 8), (2026, 9))), TODAY) == []

    def test_month_names_and_years_in_the_description_do_not_split_the_group(self):
        rows = [exp("Rent May 2026", 20000, "2026-05-01"), exp("Rent June", 20000, "2026-06-01"),
                exp("rent - july", 20000, "2026-07-01"), exp("RENT August", 20000, "2026-08-01")]
        r = pl.detect_recurring(rows, TODAY)
        assert len(r) == 1 and r[0]["key"] == "rent" and r[0]["count"] == 4

    def test_irregular_gaps_are_not_recurring(self):
        rows = [exp("Coffee", 150, d) for d in ("2026-09-01", "2026-09-02", "2026-09-11", "2026-09-19")]
        assert pl.detect_recurring(rows, TODAY) == []

    def test_wildly_different_amounts_are_not_one_subscription(self):
        rows = [exp("Electricity", a, f"2026-0{m}-05") for m, a in ((6, 400), (7, 1900), (8, 600), (9, 2500))]
        assert pl.detect_recurring(rows, TODAY) == []

    def test_a_price_rise_is_flagged(self):
        rows = monthly("Spotify", 119, months=((2026, 6), (2026, 7), (2026, 8))) + [exp("Spotify", 139, "2026-09-05")]
        r = pl.detect_recurring(rows, TODAY)[0]
        assert r["price_changed"] is True and r["latest_amount"] == 139 and r["typical_amount"] == 119

    def test_a_charge_that_stopped_is_inactive(self):
        r = pl.detect_recurring(monthly("Gym", 1500, months=((2026, 1), (2026, 2), (2026, 3))), TODAY)
        assert len(r) == 1 and r[0]["active"] is False

    def test_a_charge_a_few_days_late_is_still_active(self):
        r = pl.detect_recurring(monthly("Gym", 1500, months=((2026, 6), (2026, 7), (2026, 8))), TODAY)
        assert r[0]["active"] is True

    def test_weekly_charges(self):
        rows = [exp("Milk", 80, (date(2026, 8, 2) + timedelta(days=7 * i)).isoformat()) for i in range(6)]
        r = pl.detect_recurring(rows, TODAY)[0]
        assert r["cadence"] == "weekly" and r["monthly_cost"] == pytest.approx(80 * 52 / 12, abs=0.01)

    def test_several_charges_on_one_day_count_once(self):
        rows = monthly("Netflix", 649, months=((2026, 7), (2026, 8))) + [exp("Netflix", 649, "2026-09-05")] * 3
        assert len(pl.detect_recurring(rows, TODAY)) == 1

    def test_bad_rows_are_skipped_not_fatal(self):
        rows = monthly("Netflix", 649) + [{"description": "x", "amount": "abc", "logged_at": "2026-09-01"},
                                          {"description": "", "amount": 5, "logged_at": "2026-09-01"},
                                          {"description": "y", "amount": 5, "logged_at": "bad"},
                                          {"description": "z", "amount": float("nan"), "logged_at": "2026-09-01"}]
        assert len(pl.detect_recurring(rows, TODAY)) == 1

    @pytest.mark.parametrize("gap,name", [
        (5, None), (6, "weekly"), (8, "weekly"), (9, None), (12, None), (13, "fortnightly"), (16, "fortnightly"),
        (17, None), (26, None), (27, "monthly"), (33, "monthly"), (34, None), (84, None), (85, "quarterly"),
        (95, "quarterly"), (96, None), (359, None), (360, "yearly"), (370, "yearly"), (371, None)])
    def test_cadence_boundaries(self, gap, name):
        got = pl._cadence_for(gap)
        assert (got[0] if got else None) == name

    def test_cadence_charges_per_month(self):
        per = {n: pm for n, _, _, pm in pl._CADENCES}
        assert per["weekly"] == pytest.approx(52 / 12) and per["fortnightly"] == pytest.approx(26 / 12)
        assert per["monthly"] == 1 and per["quarterly"] == pytest.approx(1 / 3) and per["yearly"] == pytest.approx(1 / 12)

    def test_quarterly_charges(self):
        rows = [exp("Insurance", 3000, d) for d in ("2026-01-10", "2026-04-10", "2026-07-10")]
        r = pl.detect_recurring(rows, TODAY)[0]
        assert r["cadence"] == "quarterly" and r["monthly_cost"] == 1000 and r["next_expected"] == "2026-10-10"

    def test_yearly_charges(self):
        rows = [exp("Domain renewal", 1200, d) for d in ("2024-03-15", "2025-03-15", "2026-03-15")]
        r = pl.detect_recurring(rows, TODAY)[0]
        assert r["cadence"] == "yearly" and r["monthly_cost"] == 100 and r["next_expected"] == "2027-03-15"
        assert r["active"] is True

    def test_fortnightly_charges(self):
        rows = [exp("Cleaner", 600, (date(2026, 7, 1) + timedelta(days=14 * i)).isoformat()) for i in range(5)]
        r = pl.detect_recurring(rows, TODAY)[0]
        assert r["cadence"] == "fortnightly" and r["monthly_cost"] == pytest.approx(600 * 26 / 12, abs=0.01)

    def test_a_gap_just_outside_a_cadence_is_not_a_subscription(self):
        rows = [exp("Odd", 100, (date(2026, 6, 1) + timedelta(days=45 * i)).isoformat()) for i in range(4)]
        assert pl.detect_recurring(rows, TODAY) == []

    def test_active_first_then_costliest(self):
        rows = (monthly("Cheap", 100) + monthly("Pricey", 900)
                + monthly("Old", 5000, months=((2026, 1), (2026, 2), (2026, 3))))
        assert [r["key"] for r in pl.detect_recurring(rows, TODAY)] == ["pricey", "cheap", "old"]

    def test_upcoming_total_counts_only_active_charges_due_later_this_month(self):
        rec = [{"active": True, "next_expected": "2026-09-25", "typical_amount": 500, "category": "Bills"},
               {"active": True, "next_expected": "2026-09-20", "typical_amount": 111, "category": "Bills"},
               {"active": True, "next_expected": "2026-10-01", "typical_amount": 222, "category": "Bills"},
               {"active": False, "next_expected": "2026-09-28", "typical_amount": 333, "category": "Bills"},
               {"active": True, "next_expected": "2026-09-30", "typical_amount": 40, "category": "Food"}]
        assert pl.upcoming_recurring_total(rec, TODAY) == 540
        assert pl.upcoming_recurring_total(rec, TODAY, {"food"}) == 40


# ------------------------------------------------------------------ budgets

class TestBudgetStatus:
    def rows(self, spent, limit=1000):
        return pl.budget_status({"Food": limit}, {"food": spent})

    @pytest.mark.parametrize("spent,state", [(0, "ok"), (799.99, "ok"), (800, "warn"),
                                             (1000, "warn"), (1000.01, "over")])
    def test_boundaries(self, spent, state):
        assert self.rows(spent)[0]["state"] == state

    def test_numbers(self):
        r = self.rows(1250)[0]
        assert (r["percent"], r["remaining"], r["spent"]) == (125.0, -250, 1250)

    def test_case_insensitive_and_missing_spend_is_zero(self):
        r = pl.budget_status({"FOOD": 500, "Fun": 100}, {"food": 100})
        assert {x["category"]: x["spent"] for x in r} == {"FOOD": 100, "Fun": 0}

    def test_worst_first(self):
        r = pl.budget_status({"A": 100, "B": 100, "C": 100}, {"A": 10, "B": 150, "C": 90})
        assert [x["category"] for x in r] == ["B", "C", "A"]


class TestSafeToSpend:
    def test_arithmetic(self):
        r = pl.safe_to_spend(budget=30000, spent=12000, upcoming=3000, today=TODAY)   # 11 days left
        assert r["remaining"] == 15000 and r["days_left"] == 11
        assert r["per_day"] == pytest.approx(15000 / 11) and r["overspent"] is False

    def test_overspent_is_zero_per_day(self):
        r = pl.safe_to_spend(1000, 1500, 0, TODAY)
        assert r["per_day"] == 0 and r["overspent"] is True and r["remaining"] == -500

    def test_exactly_spent_is_not_overspent(self):
        assert pl.safe_to_spend(1000, 1000, 0, TODAY)["overspent"] is False

    def test_last_day_of_the_month_divides_by_one(self):
        assert pl.safe_to_spend(1000, 400, 0, date(2026, 9, 30))["per_day"] == 600


# ------------------------------------------------------------ health score

class TestHealthScore:
    def test_savings_rate_scale(self):
        assert pl.savings_rate_score(100000, 80000)["score"] == 100        # 20% saved
        assert pl.savings_rate_score(100000, 90000)["score"] == 50
        assert pl.savings_rate_score(100000, 120000)["score"] == 0         # overspending
        assert pl.savings_rate_score(0, 10) is None

    def test_volatility_needs_four_weeks_and_rewards_steadiness(self):
        assert pl.volatility_score([100, 100, 100]) is None
        assert pl.volatility_score([100, 100, 100, 100])["score"] == 100
        assert pl.volatility_score([0, 0, 0, 0]) is None
        spiky = pl.volatility_score([10, 400, 10, 400])["score"]
        assert 0 <= spiky < 50

    def test_diversification(self):
        assert pl.diversification_score([]) is None
        assert pl.diversification_score([100])["score"] == 0
        assert pl.diversification_score([100] * 5)["score"] == 100
        assert pl.diversification_score([100] * 10)["score"] == 100
        assert 0 < pl.diversification_score([900, 100])["score"] < 50
        assert pl.diversification_score([-5, float("nan"), 0]) is None

    def test_overall_is_a_weighted_average_of_what_exists(self):
        a = {"name": "savings rate", "score": 100}
        b = {"name": "diversification", "score": 0}
        r = pl.health_score([a, None, b])
        assert r["score"] == round((100 * 0.4 + 0 * 0.3) / 0.7) and r["missing"] == ["spending steadiness"]

    def test_one_component_is_not_a_score(self):
        r = pl.health_score([{"name": "savings rate", "score": 90}, None, None])
        assert r["score"] is None and r["label"] == "not enough data"

    @pytest.mark.parametrize("scores,label", [((90, 90), "strong"), ((60, 60), "decent"),
                                              ((40, 40), "needs attention"), ((10, 10), "weak")])
    def test_labels(self, scores, label):
        comps = [{"name": "savings rate", "score": scores[0]}, {"name": "spending steadiness", "score": scores[1]}]
        assert pl.health_score(comps)["label"] == label

    def test_weekly_totals_only_counts_weeks_since_tracking_began(self):
        rows = [exp("a", 100, (TODAY - timedelta(days=d)).isoformat(), "Food") for d in (1, 8, 15)]
        # The oldest expense sits in a week that began before tracking did: that
        # partial week is dropped rather than scored as a suspiciously quiet one.
        assert pl.weekly_totals(rows, TODAY) == [100, 100]

    def test_weekly_totals_full_weeks(self):
        rows = [exp("a", 50 * (w + 1), (TODAY - timedelta(days=7 * w + 1)).isoformat(), "Food") for w in range(8)]
        rows.append(exp("old", 1, (TODAY - timedelta(days=70)).isoformat(), "Food"))
        assert pl.weekly_totals(rows, TODAY) == [400, 350, 300, 250, 200, 150, 100, 50]

    @pytest.mark.parametrize("d,n,want", [(date(2026, 1, 31), 1, date(2026, 2, 28)), (date(2028, 1, 31), 1, date(2028, 2, 29)),
                                          (date(2026, 11, 15), 3, date(2027, 2, 15)), (date(2026, 12, 5), 1, date(2027, 1, 5)),
                                          (date(2026, 3, 31), 12, date(2027, 3, 31)), (date(2026, 5, 5), 0, date(2026, 5, 5))])
    def test_add_months(self, d, n, want):
        assert pl.add_months(d, n) == want

    def test_weekly_totals_empty(self):
        assert pl.weekly_totals([], TODAY) == []

    def test_monthly_average_uses_only_complete_months(self):
        rows = [exp("a", 1000, "2026-07-10"), exp("a", 3000, "2026-08-10"), exp("a", 99999, "2026-09-10")]
        assert pl.monthly_spend_average(rows, TODAY) == 2000
        assert pl.monthly_spend_average([exp("a", 5, "2026-09-10")], TODAY) is None


# ---------------------------------------------------------------- scenarios

class TestScenario:
    def test_matches_the_closed_form_annuity_due(self):
        r = pl.scenario(10000, 15, 12)
        i = 1.12 ** (1 / 12) - 1
        n = 180
        assert r["final"] == pytest.approx(10000 * ((1 + i) ** n - 1) / i * (1 + i), rel=1e-9)
        assert r["contributed"] == 10000 * 180 and r["gain"] == pytest.approx(r["final"] - r["contributed"])

    def test_zero_return_is_just_the_money_put_in(self):
        r = pl.scenario(5000, 10, 0, initial=20000)
        assert r["final"] == pytest.approx(20000 + 5000 * 120)

    def test_initial_lump_sum_compounds(self):
        r = pl.scenario(0, 10, 10, initial=100000)
        assert r["final"] == pytest.approx(100000 * 1.1 ** 10, rel=1e-9)

    def test_year_table(self):
        t = pl.scenario(1000, 3, 8)["table"]
        assert [x["year"] for x in t] == [1, 2, 3] and t[0]["contributed"] == 12000
        assert t[0]["value"] < t[1]["value"] < t[2]["value"]

    def test_a_loss_is_allowed_and_shrinks_the_pot(self):
        r = pl.scenario(1000, 5, -10)
        assert r["final"] < r["contributed"]

    @pytest.mark.parametrize("args", [(1000, 0, 10), (1000, 61, 10), (1000, 10, 101), (1000, 10, -51),
                                      (-1, 10, 10), (float("nan"), 10, 10), (1000, 10, float("inf")),
                                      (0, 10, 10), (1000, 2.5, 10), (pl.MAX_AMOUNT + 1, 10, 10)])
    def test_rejects_nonsense(self, args):
        with pytest.raises(ValueError):
            pl.scenario(*args)


# ---------------------------------------------------------------- tax export

class TestTaxExport:
    def test_csv_safe(self):
        assert pl.csv_safe("=SUM(A1)") == "'=SUM(A1)" and pl.csv_safe("@x") == "'@x"
        assert pl.csv_safe("+1") == "'+1" and pl.csv_safe("-1") == "'-1" and pl.csv_safe("\tx") == "'\tx"
        assert pl.csv_safe("lunch") == "lunch" and pl.csv_safe(None) == ""

    def test_files_contents_hints_and_formula_neutralising(self, tmp_path):
        expenses = [exp("=HYPERLINK(evil)", 100, "2026-05-01", "Other"),
                    exp("Apollo pharmacy", 450.5, "2026-06-02", "Health"),
                    exp("College fees", 20000, "2026-07-03", "Education")]
        invs = [{"name": "Axis ELSS fund", "type": "mutual fund", "quantity": 10, "buy_price": 55.5,
                 "logged_at": "2026-05-05 10:00:00"}]
        e_path, i_path = pl.write_tax_csvs(tmp_path / "exports", 2026, expenses, invs)
        assert e_path.name == "expenses_FY2026-27.csv" and i_path.name == "investments_FY2026-27.csv"
        rows = list(csv.DictReader(open(e_path, encoding="utf-8")))
        assert rows[0]["description"] == "'=HYPERLINK(evil)" and rows[1]["amount"] == "450.50"
        assert "80D" in rows[1]["tax_hint"] and "80C" in rows[2]["tax_hint"] and rows[0]["tax_hint"] == ""
        inv = list(csv.DictReader(open(i_path, encoding="utf-8")))[0]
        assert inv["cost"] == "555.00" and "80C" in inv["tax_hint"]


# ============================================================ manager + db

class Clock:
    def __init__(self, day=TODAY, hour=12):
        self.day, self.hour = day, hour

    def today(self):
        return self.day

    def now(self):
        return datetime(self.day.year, self.day.month, self.day.day, self.hour)


@pytest.fixture()
def env(tmp_path):
    db = PlutoDB(str(tmp_path / "pluto.db"))
    clock = Clock()
    pm = pl.PlanningManager(db, "₹", tmp_path / "exports", clock.today, clock.now)
    yield db, pm, clock
    db.close()


class TestDatabaseAdditions:
    def test_budget_round_trip_and_case_insensitive_upsert(self, env):
        db, _, _ = env
        db.set_budget("Food", 8000)
        db.set_budget("food", 9000)
        assert db.get_budgets() == {"Food": 9000.0}
        assert db.delete_budget("FOOD") is True and db.delete_budget("Food") is False

    def test_settings_and_alert_markers(self, env):
        db, _, _ = env
        assert db.get_setting("x") is None
        db.set_setting("x", "1"); db.set_setting("x", "2")
        assert db.get_setting("x") == "2"
        assert not db.alert_already_sent("k")
        db.mark_alert_sent("k"); db.mark_alert_sent("k")
        assert db.alert_already_sent("k")

    def test_range_queries_are_half_open(self, env):
        db, _, _ = env
        db.log_expense(10, "a", "Food", "2026-09-01 00:00:00")
        db.log_expense(20, "b", "Food", "2026-09-30 23:59:59")
        db.log_expense(40, "c", "Food", "2026-10-01 00:00:00")
        got = db.get_expenses_between("2026-09-01", "2026-10-01")
        assert [e["description"] for e in got] == ["a", "b"]
        assert db.get_totals_between("2026-09-01", "2026-10-01")[0]["total"] == 30

    def test_log_expense_still_stamps_the_time_itself(self, env):
        db, _, _ = env
        db.log_expense(5, "x", "Food")
        assert db.get_expenses(1)[0]["logged_at"]

    def test_an_existing_database_upgrades_in_place(self, tmp_path):
        import sqlite3
        path = str(tmp_path / "old.db")
        con = sqlite3.connect(path)
        con.executescript("CREATE TABLE expenses (id INTEGER PRIMARY KEY AUTOINCREMENT, amount REAL NOT NULL,"
                          "description TEXT NOT NULL, category TEXT NOT NULL, logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
                          "INSERT INTO expenses (amount, description, category) VALUES (99, 'old', 'Food');")
        con.commit(); con.close()
        db = PlutoDB(path)
        assert db.get_expenses()[0]["amount"] == 99 and db.get_budgets() == {}
        db.close()


class TestManagerBudgets:
    def test_set_status_remove(self, env):
        db, pm, _ = env
        r = pm.set_budget({"category": "groceries", "amount": "8k"})
        assert "Food" in r["response"] and "₹8,000" in r["response"] and db.get_budgets() == {"Food": 8000.0}
        db.log_expense(6500, "dinner", "Food", "2026-09-10 10:00:00")
        s = pm.budget_status({})
        assert "within" not in s["response"] and "close to its limit" in s["response"] and "₹1,500 left" in s["response"]
        assert pm.set_budget({"category": "food", "raw_query": "remove my food budget"})["response"] == "Removed your Food budget."
        assert db.get_budgets() == {}
        assert "don't have" in pm.set_budget({"category": "food", "raw_query": "remove food budget"})["response"]

    def test_over_budget_is_called_out(self, env):
        db, pm, _ = env
        pm.set_budget({"category": "food", "amount": 1000})
        db.log_expense(1400, "party", "Food", "2026-09-12 10:00:00")
        assert "over budget on Food" in pm.budget_status({})["response"]
        assert "₹400 over" in pm.budget_status({})["response"]

    def test_only_this_months_spending_counts(self, env):
        db, pm, _ = env
        pm.set_budget({"category": "food", "amount": 1000})
        db.log_expense(5000, "last month", "Food", "2026-08-31 23:00:00")
        assert "within every budget" in pm.budget_status({})["response"]

    def test_asks_for_what_is_missing(self, env):
        _, pm, _ = env
        assert pm.set_budget({})["confidence"] == 0.5
        assert pm.set_budget({"category": "food"})["response"].startswith("How much")
        assert pm.set_budget({"category": "food", "amount": "-5"})["response"].startswith("How much")
        assert "haven't set any budgets" in pm.budget_status({})["response"]

    def test_unknown_category_names_are_kept_tidy(self, env):
        db, pm, _ = env
        pm.set_budget({"category": "pet  care", "amount": 500})
        assert list(db.get_budgets()) == ["Pet Care"]

    def test_status_for_one_category(self, env):
        db, pm, _ = env
        pm.set_budget({"category": "food", "amount": 1000}); pm.set_budget({"category": "transport", "amount": 500})
        r = pm.budget_status({"category": "uber"})
        assert "Transport" in r["response"] and "Food" not in r["response"]


class TestManagerSafeToSpend:
    def test_needs_a_budget_or_income(self, env):
        assert "need a budget" in env[1].safe_to_spend()["response"]

    def test_from_budgets_with_recurring_bills_still_due(self, env):
        db, pm, _ = env
        pm.set_budget({"category": "entertainment", "amount": 3000})
        for e in monthly("Netflix", 649):
            db.log_expense(e["amount"], e["description"], "Entertainment", e["logged_at"])
        # Netflix was paid on the 5th; the next charge (Oct 5) is next month, so nothing is "still due".
        db.log_expense(1000, "movie", "Entertainment", "2026-09-10 10:00:00")
        r = pm.safe_to_spend()
        spent = 649 + 1000
        assert r["data"]["spent"] == spent and r["data"]["upcoming_recurring"] == 0
        assert r["data"]["per_day"] == pytest.approx((3000 - spent) / 11)

    def test_a_bill_due_later_this_month_is_set_aside(self, env):
        db, pm, _ = env
        pm.set_budget({"category": "bills", "amount": 10000})
        for m in (6, 7, 8):
            db.log_expense(1500, "Gym", "Bills", f"2026-0{m}-25 09:00:00")
        r = pm.safe_to_spend()
        assert r["data"]["upcoming_recurring"] == 1500 and r["data"]["remaining"] == 8500

    def test_spending_outside_budgeted_categories_is_ignored(self, env):
        db, pm, _ = env
        pm.set_budget({"category": "food", "amount": 1000})
        db.log_expense(90000, "laptop", "Shopping", "2026-09-02 10:00:00")
        assert pm.safe_to_spend()["data"]["spent"] == 0

    def test_falls_back_to_income_and_counts_everything(self, env):
        db, pm, _ = env
        pm.set_income({"amount": "50000"})
        db.log_expense(20000, "rent", "Bills", "2026-09-01 10:00:00")
        r = pm.safe_to_spend()
        assert r["data"]["budget"] == 50000 and r["data"]["spent"] == 20000 and "income" in r["response"]

    def test_overspent_says_so(self, env):
        db, pm, _ = env
        pm.set_budget({"category": "food", "amount": 1000})
        db.log_expense(1500, "feast", "Food", "2026-09-02 10:00:00")
        r = pm.safe_to_spend()
        assert "past" in r["response"] and "nothing more" in r["response"] and r["data"]["overspent"]


class TestManagerOthers:
    def test_set_income(self, env):
        db, pm, _ = env
        assert "₹85,000" in pm.set_income({"amount": "85000"})["response"]
        assert db.get_setting("monthly_income") == "85000.0"
        assert pm.set_income({"amount": "nope"})["confidence"] == 0.5

    def test_corrupt_income_setting_is_ignored(self, env):
        db, pm, _ = env
        for bad in ("abc", "nan", "-5", "inf"):
            db.set_setting("monthly_income", bad)
            assert pm._income() is None

    def test_recurring_report(self, env):
        db, pm, _ = env
        assert "don't see any" in pm.recurring_expenses()["response"]
        for e in monthly("Netflix", 649) + monthly("Spotify", 119):
            db.log_expense(e["amount"], e["description"], "Entertainment", e["logged_at"])
        r = pm.recurring_expenses()
        assert "2 regular" in r["response"] and "₹768 a month" in r["response"]
        assert "Netflix: ₹649 monthly" in r["response"]

    def test_health_score_needs_data_then_scores(self, env):
        db, pm, _ = env
        assert "can't score" in pm.financial_health()["response"] and "income" in pm.financial_health()["response"]
        pm.set_income({"amount": 100000})
        for m in (6, 7, 8):
            db.log_expense(70000, "life", "Bills", f"2026-0{m}-10 10:00:00")
        db.log_investment("Nifty index fund", "mutual_fund", 10, 100)
        db.log_investment("Reliance", "stock", 10, 100)
        r = pm.financial_health()
        assert "out of 100" in r["response"] and "savings rate" in r["response"] and "not financial advice" in r["response"]
        assert r["data"]["score"] is not None

    def test_scenario(self, env):
        _, pm, _ = env
        r = pm.scenario_plan({"amount": "10000", "years": "15", "rate": "12"})
        assert r["data"]["base"]["final"] == pytest.approx(pl.scenario(10000, 15, 12)["final"])
        assert "illustration" in r["response"] and r["data"]["low"] < r["data"]["base"]["final"] < r["data"]["high"]
        assert pm.scenario_plan({})["confidence"] == 0.5
        assert pm.scenario_plan({"amount": 100, "rate": 10})["response"] == "For how many years?"
        assert "yearly return" in pm.scenario_plan({"amount": 100, "years": 5})["response"]
        assert "can't run" in pm.scenario_plan({"amount": 100, "years": 500, "rate": 10})["response"]
        assert "can't run" in pm.scenario_plan({"amount": 100, "years": 5, "rate": 500})["response"]

    def test_tax_export_writes_the_financial_year_only(self, env, tmp_path):
        db, pm, _ = env
        db.log_expense(100, "inside", "Food", "2026-04-01 00:00:00")
        db.log_expense(200, "outside", "Food", "2026-03-31 23:59:59")
        r = pm.export_tax({"year": "2026-27"})
        files = [Path(f) for f in r["data"]["files"]]
        assert all(f.exists() for f in files) and "1 expense(s)" in r["response"]
        assert [x["description"] for x in csv.DictReader(open(files[0], encoding="utf-8"))] == ["inside"]
        assert "Nothing logged" in pm.export_tax({"year": "2010"})["response"]

    def test_tax_export_edge_cases(self, env):
        _, pm, _ = env
        assert "Nothing logged" in pm.export_tax({})["response"]
        assert "Which financial year" in pm.export_tax({"year": "last year"})["response"]
        pm.export_dir = None
        assert pm.export_tax({})["confidence"] == 0.0


class TestBudgetAlerts:
    def setup_budget(self, env):
        db, pm, clock = env
        pm.set_budget({"category": "food", "amount": 1000})
        return db, pm, clock

    def test_silent_until_a_threshold_is_crossed(self, env):
        db, pm, _ = self.setup_budget(env)
        assert pm.check_budget_alerts() is None
        db.log_expense(700, "x", "Food", "2026-09-10 10:00:00")
        assert pm.check_budget_alerts() is None

    def test_warning_once_then_over_once(self, env):
        db, pm, _ = self.setup_budget(env)
        db.log_expense(850, "x", "Food", "2026-09-10 10:00:00")
        first = pm.check_budget_alerts()
        assert first and "Food has used 85%" in first
        assert pm.check_budget_alerts() is None
        db.log_expense(300, "y", "Food", "2026-09-11 10:00:00")
        second = pm.check_budget_alerts()
        assert second and "over budget by ₹150" in second
        assert pm.check_budget_alerts() is None

    def test_going_straight_to_over_skips_the_warning(self, env):
        db, pm, _ = self.setup_budget(env)
        db.log_expense(1500, "x", "Food", "2026-09-10 10:00:00")
        assert "over budget" in pm.check_budget_alerts()
        db.log_expense(1, "z", "Food", "2026-09-11 10:00:00")
        assert pm.check_budget_alerts() is None

    def test_quiet_hours_hold_the_alert_without_losing_it(self, env):
        db, pm, clock = self.setup_budget(env)
        db.log_expense(900, "x", "Food", "2026-09-10 10:00:00")
        for hour in (22, 23, 0, 6):
            clock.hour = hour
            assert pm.check_budget_alerts() is None
        clock.hour = 7
        assert pm.check_budget_alerts() is not None

    def test_a_new_month_alerts_again(self, env):
        db, pm, clock = self.setup_budget(env)
        db.log_expense(900, "x", "Food", "2026-09-10 10:00:00")
        assert pm.check_budget_alerts()
        clock.day = date(2026, 10, 15)
        db.log_expense(900, "x", "Food", "2026-10-10 10:00:00")
        assert pm.check_budget_alerts()

    def test_several_categories_in_one_message(self, env):
        db, pm, _ = self.setup_budget(env)
        pm.set_budget({"category": "transport", "amount": 500})
        db.log_expense(900, "x", "Food", "2026-09-10 10:00:00")
        db.log_expense(600, "y", "Transport", "2026-09-10 10:00:00")
        text = pm.check_budget_alerts()
        assert "Food" in text and "Transport" in text and text.startswith("Heads up")

    def test_no_budgets_no_alerts(self, env):
        assert env[1].check_budget_alerts() is None


# ====================================================== hardening (found by these tests)

class TestInputHardening:
    @pytest.fixture()
    def pfm(self, tmp_path):
        from modules.pluto.config import PlutoConfig
        from modules.pluto.personal_finance import PersonalFinanceManager
        llm = MagicMock()
        llm.generate.return_value = {"category": "Other"}
        mgr = PersonalFinanceManager(PlutoConfig(db_path=tmp_path / "p.db"), db_manager=MagicMock(), llm_client=llm)
        yield mgr
        mgr.db.close()

    @pytest.mark.parametrize("amount", ["nan", float("nan"), "inf", float("inf"), "-5", 0, "0", "abc", None,
                                        "1e999", 10 ** 12])
    def test_a_bad_expense_amount_is_never_saved(self, pfm, amount):
        r = pfm.log_expense({"amount": amount, "description": "lunch"})
        assert pfm.db.get_expenses() == []
        assert r["response"]

    def test_nan_used_to_be_saved_as_an_expense(self, pfm):
        # Regression: float("nan") <= 0 is False, so "nan" slipped past the old check.
        pfm.log_expense({"amount": "nan", "description": "lunch"})
        assert pfm.db.get_grand_total() == 0

    def test_messy_but_real_amounts_are_accepted(self, pfm):
        for raw, want in (("₹1,500", 1500), ("2k", 2000), ("250 rupees", 250), (99.5, 99.5)):
            pfm.log_expense({"amount": raw, "description": "lunch"})
        assert sorted(e["amount"] for e in pfm.db.get_expenses()) == [99.5, 250, 1500, 2000]

    def test_zero_gets_its_specific_message(self, pfm):
        assert "greater than zero" in pfm.log_expense({"amount": 0, "description": "x"})["response"]
        assert "greater than zero" in pfm.log_expense({"amount": "-3", "description": "x"})["response"]

    @pytest.mark.parametrize("q,p", [("nan", 5), (5, "inf"), (-1, 5), (5, -1), ("1e999", 5)])
    def test_bad_investment_numbers_are_never_saved(self, pfm, q, p):
        r = pfm.track_investment({"type": "Reliance", "quantity": q, "buy_price": p})
        assert pfm.db.get_investments() == [] and r["confidence"] <= 0.5

    def test_convert_currency_rejects_nan(self, pfm):
        assert "How much" in pfm.convert_currency({"amount": "nan", "from_currency": "USD"})["response"]


# ================================================================ engine routing

def make_engine(tmp_path):
    from modules.pluto.config import PlutoConfig
    from modules.pluto.personal_finance import PersonalFinanceManager
    cfg = PlutoConfig(db_path=tmp_path / "pluto.db")
    llm = MagicMock()
    llm.generate.return_value = {"category": "Other"}
    pfm = PersonalFinanceManager(cfg, db_manager=MagicMock(), llm_client=llm)
    clock = Clock()
    eng = PlutoEngine(config=cfg, pf_manager=pfm, mi_manager=MagicMock(), db_manager=MagicMock(), llm_client=llm,
                      portfolio_optimizer=MagicMock(), backtester=MagicMock(), forecaster=MagicMock(), advisor=MagicMock())
    eng.planner.export_dir = tmp_path / "exports"
    eng.planner._today, eng.planner._now = clock.today, clock.now
    return eng, pfm, clock


class TestEngineRouting:
    PLAN = ["set_budget", "budget_status", "recurring_expenses", "safe_to_spend",
            "set_income", "financial_health", "scenario_plan", "export_tax"]

    def test_can_handle_every_planning_intent_and_no_stranger(self, tmp_path):
        eng, pfm, _ = make_engine(tmp_path)
        assert all(eng.can_handle(i) for i in self.PLAN) and not eng.can_handle("set_budgets")
        pfm.db.close()

    def test_each_intent_reaches_its_handler(self, tmp_path):
        eng, pfm, _ = make_engine(tmp_path)
        eng.planner = MagicMock()
        for intent in self.PLAN:
            getattr(eng.planner, intent).return_value = {"response": intent, "data": {}, "confidence": 1}
        for intent in self.PLAN:
            assert eng.handle(intent, {}, {})["response"] == intent, intent
        pfm.db.close()

    def test_handle_never_raises_when_planning_fails(self, tmp_path):
        eng, pfm, _ = make_engine(tmp_path)
        eng.planner = MagicMock(); eng.planner.safe_to_spend.side_effect = RuntimeError("boom")
        assert eng.handle("safe_to_spend", {}, {})["confidence"] == 0.0
        pfm.db.close()

    def test_heartbeat_hook_never_raises(self, tmp_path):
        eng, pfm, _ = make_engine(tmp_path)
        eng.planner = MagicMock(); eng.planner.check_budget_alerts.side_effect = RuntimeError("boom")
        assert eng.check_budget_alerts() is None
        pfm.db.close()

    def test_engine_without_a_database_still_builds(self):
        from modules.pluto.personal_finance import PersonalFinanceManager
        eng = PlutoEngine(pf_manager=MagicMock(spec=PersonalFinanceManager), mi_manager=MagicMock(),
                          db_manager=MagicMock(), llm_client=MagicMock(), portfolio_optimizer=MagicMock(),
                          backtester=MagicMock(), forecaster=MagicMock(), advisor=MagicMock())
        assert eng.can_handle("set_budget")


# ======================================================== heartbeat integration

class TestHeartbeatHook:
    def _hb(self, pluto):
        from core.heartbeat import HestiaHeartbeat
        return HestiaHeartbeat(pluto=pluto)

    def test_alert_text_is_spoken(self):
        from unittest.mock import patch
        pluto = MagicMock(); pluto.check_budget_alerts.return_value = "Heads up: Food is over budget by ₹150."
        with patch("core.heartbeat.bus.emit") as emit:
            self._hb(pluto)._maybe_run_pluto_budget_alerts()
        emit.assert_called_once_with("speak", {"text": "Heads up: Food is over budget by ₹150."})

    @pytest.mark.parametrize("value", [None, "", "   ", 5])
    def test_nothing_to_say_is_silent(self, value):
        from unittest.mock import patch
        pluto = MagicMock(); pluto.check_budget_alerts.return_value = value
        with patch("core.heartbeat.bus.emit") as emit:
            self._hb(pluto)._maybe_run_pluto_budget_alerts()
        emit.assert_not_called()

    def test_a_failing_hook_does_not_break_the_tick(self):
        pluto = MagicMock(); pluto.check_budget_alerts.side_effect = RuntimeError("x")
        self._hb(pluto)._maybe_run_pluto_budget_alerts()

    def test_no_pluto_is_fine(self):
        self._hb(None)._maybe_run_pluto_budget_alerts()


# ================================================================ full pipeline

class TestFullPipeline:
    """NLU -> Hecate -> PlutoEngine -> real SQLite, only the model scripted."""

    def _pipe(self, tmp_path, *replies):
        import core.nlu as nlu_mod
        from test_pipeline_integration import Pipeline
        eng, pfm, clock = make_engine(tmp_path)
        p = Pipeline(*replies, extra=(eng,))
        return p, eng, pfm

    def I(self, intent, entities=None):
        from test_pipeline_integration import ScriptedModel
        return ScriptedModel.intent(intent, entities)

    @pytest.fixture(autouse=True)
    def _fast(self, monkeypatch):
        import core.nlu as nlu_mod
        monkeypatch.setattr(nlu_mod.time, "sleep", lambda *_: None)

    def test_set_a_budget_by_voice_then_log_spend_then_ask_status(self, tmp_path):
        p, eng, pfm = self._pipe(tmp_path, self.I("pluto_set_budget", {"category": "food", "amount": "8000"}))
        assert "₹8,000" in p.ask("set a food budget of 8000 a month")
        assert pfm.db.get_budgets() == {"Food": 8000.0}
        p.model.replies = [self.I("pluto_log_expense", {"amount": "6500", "category": "groceries"})]
        p.ask("log 6500 for groceries")
        p.model.replies = [self.I("pluto_budget_status")]
        assert "Food" in p.ask("am I over budget?") and pfm.db.get_grand_total() == 6500
        pfm.db.close()

    def test_a_negative_amount_from_the_model_is_refused_end_to_end(self, tmp_path):
        p, eng, pfm = self._pipe(tmp_path, self.I("pluto_log_expense", {"amount": "-500", "category": "food"}))
        reply = p.ask("log minus 500 for food")
        assert "greater than zero" in reply or "didn't catch" in reply
        assert pfm.db.get_expenses() == []
        pfm.db.close()

    def test_scenario_by_voice(self, tmp_path):
        p, eng, pfm = self._pipe(tmp_path, self.I("pluto_scenario_plan", {"amount": "10000", "years": "15", "rate": "12"}))
        assert "grows to about" in p.ask("what if I invest 10000 a month for 15 years at 12 percent")
        pfm.db.close()

    def test_every_new_intent_is_routed_to_pluto_by_hecate(self, tmp_path):
        from modules.hecate import HecateEngine
        h = HecateEngine()
        for intent in ("set_budget", "budget_status", "recurring_expenses", "safe_to_spend", "set_income",
                       "financial_health", "scenario_plan", "export_tax"):
            d = h.decide("x", {"intent": f"pluto_{intent}", "confidence": 0.95}, ["core", "pluto"])
            assert (d["primary"], d["intent"]) == ("pluto", intent)


# ====================================================== gaps found by mutation testing

class TestPinnedByMutationTesting:
    """Each test here exists because scripts/mutation_check.py showed a change
    to planning.py that no earlier test noticed."""

    def test_the_smallest_valid_amounts(self):
        assert pl.parse_amount(1) == 1.0 and pl.parse_amount(0.5) == 0.5 and pl.parse_amount("0.01") == 0.01

    def test_weekly_charges_tolerate_three_days_of_jitter_but_not_four(self):
        base = date(2026, 7, 1)
        ok = [base, base + timedelta(days=7), base + timedelta(days=14), base + timedelta(days=24)]       # gaps 7,7,10
        bad = [base, base + timedelta(days=7), base + timedelta(days=14), base + timedelta(days=25)]      # gaps 7,7,11
        assert len(pl.detect_recurring([exp("Milk", 80, d.isoformat()) for d in ok], TODAY)) == 1
        assert pl.detect_recurring([exp("Milk", 80, d.isoformat()) for d in bad], TODAY) == []

    def test_weekly_next_expected_is_a_week_on(self):
        rows = [exp("Milk", 80, (date(2026, 8, 30) + timedelta(days=7 * i)).isoformat()) for i in range(4)]
        assert pl.detect_recurring(rows, TODAY)[0]["next_expected"] == "2026-09-27"

    def test_a_charge_logged_today_is_included(self, env):
        db, pm, _ = env
        for d in ("2026-07-20", "2026-08-20", "2026-09-20"):          # last one is TODAY
            db.log_expense(500, "Gym", "Health", f"{d} 08:00:00")
        assert "Gym" in pm.recurring_expenses()["response"]

    def test_diversification_formula_exact_values(self):
        assert pl.diversification_score([100, 300])["score"] == 47      # hhi 0.625 -> 0.375 / 0.8
        assert pl.diversification_score([500, 500])["score"] == 62
        assert pl.diversification_score([1, 1, 1, 1])["score"] == 94    # hhi 0.25

    @pytest.mark.parametrize("score,label", [(75, "strong"), (74, "decent"), (55, "decent"), (54, "needs attention"),
                                             (35, "needs attention"), (34, "weak"), (0, "weak"), (100, "strong")])
    def test_label_boundaries(self, score, label):
        comps = [{"name": "savings rate", "score": score}, {"name": "spending steadiness", "score": score}]
        assert pl.health_score(comps)["label"] == label

    def test_monthly_average_ignores_zero_negative_and_nan_rows(self):
        rows = [exp("a", 1000, "2026-08-10"), exp("b", 0, "2026-08-11"), exp("c", -500, "2026-08-12"),
                exp("d", float("nan"), "2026-08-13"), exp("e", "junk", "2026-08-14")]
        assert pl.monthly_spend_average(rows, TODAY) == 1000

    def test_weekly_totals_ignore_bad_rows_too(self):
        rows = [exp("anchor", 1, (TODAY - timedelta(days=9)).isoformat()),
                exp("a", 100, (TODAY - timedelta(days=1)).isoformat()), exp("b", 0, TODAY.isoformat()),
                exp("c", float("nan"), TODAY.isoformat()), exp("d", -5, TODAY.isoformat())]
        assert pl.weekly_totals(rows, TODAY, weeks=1) == [100]

    @pytest.mark.parametrize("name,type_,fragment", [("PPF account", "savings", "80C"), ("My NPS", "pension", "80CCD"),
                                                     ("Axis ELSS", "fund", "80C"), ("Reliance", "stock", "")])
    def test_investment_hints(self, name, type_, fragment):
        hint = pl.investment_hint(name, type_)
        assert (fragment in hint) if fragment else hint == ""

    def test_export_dates_are_plain_iso_days(self, tmp_path):
        e, i = pl.write_tax_csvs(tmp_path, 2026, [exp("x", 5, "2026-05-01")],
                                 [{"name": "n", "type": "t", "quantity": 1, "buy_price": 2, "logged_at": "2026-05-02 10:00:00"}])
        assert list(csv.DictReader(open(e, encoding="utf-8")))[0]["date"] == "2026-05-01"
        assert list(csv.DictReader(open(i, encoding="utf-8")))[0]["date"] == "2026-05-02"

    def test_a_long_list_of_subscriptions_is_capped_at_eight_with_a_count_of_the_rest(self, env):
        db, pm, _ = env
        for n in range(10):
            for m in (6, 7, 8, 9):
                db.log_expense(100 + n * 10, f"Service{chr(97 + n)}", "Bills", f"2026-0{m}-0{n % 9 + 1} 09:00:00")
        r = pm.recurring_expenses()["response"]
        assert "10 regular" in r and "...and 2 more." in r and r.count("monthly") == 8

    def test_stopped_subscriptions_are_mentioned_only_when_there_are_some(self, env):
        db, pm, _ = env
        for m in (6, 7, 8, 9):
            db.log_expense(649, "Netflix", "Entertainment", f"2026-0{m}-05 09:00:00")
        assert "older one" not in pm.recurring_expenses()["response"]
        for m in (1, 2, 3):
            db.log_expense(300, "Old gym", "Health", f"2026-0{m}-05 09:00:00")
        assert "1 older one(s) look like they've stopped." in pm.recurring_expenses()["response"]

    def test_income_without_a_complete_month_of_spending_does_not_crash_the_score(self, env):
        db, pm, _ = env
        pm.set_income({"amount": 100000})
        db.log_expense(500, "x", "Food", "2026-09-10 10:00:00")      # only the current, incomplete month
        db.log_investment("A", "stock", 1, 100); db.log_investment("B", "stock", 1, 100)
        r = pm.financial_health()
        assert "savings rate" not in r["response"].split("Not counted")[0].lower() or r["data"]["score"] is not None

    def test_missing_parts_are_named(self, env):
        db, pm, _ = env
        pm.set_income({"amount": 100000})
        for m in (6, 7, 8):
            db.log_expense(70000, "life", "Bills", f"2026-0{m}-10 10:00:00")
        for d in range(8):
            db.log_expense(1000, "w", "Food", (TODAY - timedelta(days=7 * d + 1)).isoformat() + " 10:00:00")
        r = pm.financial_health()
        assert r["data"]["score"] is not None and "Not counted (no data yet): diversification." in r["response"]

    def test_a_lump_sum_alone_is_a_valid_scenario(self, env):
        _, pm, _ = env
        r = pm.scenario_plan({"initial": 100000, "years": 10, "rate": 10})
        assert r["data"]["base"]["final"] == pytest.approx(100000 * 1.1 ** 10, rel=1e-9)
        assert "from a ₹100,000 start" in r["response"]

    def test_scenario_range_wording(self, env):
        _, pm, _ = env
        text = pm.scenario_plan({"amount": 1000, "years": 5, "rate": 12})["response"]
        assert "At 9% it's" in text and "at 15% it's" in text
        capped = pm.scenario_plan({"amount": 1000, "years": 5, "rate": 99})["response"]
        assert "at 100% it's" in capped
        floored = pm.scenario_plan({"amount": 1000, "years": 5, "rate": -49})["response"]
        assert "At -50% it's" in floored

    @pytest.mark.parametrize("rate,text", [("-5", "at -5% a year"), ("\u22125", "at -5% a year"), ("5 percent", "at 5% a year"),
                                           ("12%", "at 12% a year"), ("- 5", "at -5% a year")])
    def test_the_return_keeps_its_sign_and_tolerates_words(self, env, rate, text):
        r = env[1].scenario_plan({"amount": 1000, "years": 5, "rate": rate})
        assert text in r["response"], r["response"]
        if rate.strip().startswith(("-", "\u2212")):
            assert r["data"]["base"]["final"] < r["data"]["base"]["contributed"]

    @pytest.mark.parametrize("years", ["15", 15, "15 years", 15.0, "fifteen", "", None, "0", "-3"])
    def test_years_phrasing(self, env, years):
        r = env[1].scenario_plan({"amount": 1000, "years": years, "rate": 10})
        if str(years).startswith("15"):
            assert "15 year(s)" in r["response"]
        else:
            assert r["response"] == "For how many years?"

    def test_every_reply_has_a_confidence_between_zero_and_one(self, env):
        db, pm, _ = env
        replies = [
            pm.set_budget({}), pm.set_budget({"category": "food"}), pm.set_budget({"category": "food", "amount": 5}),
            pm.budget_status({}), pm.recurring_expenses(), pm.safe_to_spend(), pm.set_income({}), pm.set_income({"amount": 5}),
            pm.financial_health(), pm.scenario_plan({}), pm.scenario_plan({"amount": 5}),
            pm.scenario_plan({"amount": 5, "years": 2}), pm.scenario_plan({"amount": 5, "years": 2, "rate": 5}),
            pm.scenario_plan({"amount": 5, "years": 200, "rate": 5}), pm.export_tax({}), pm.export_tax({"year": "x"}),
        ]
        pm.set_budget({"category": "food", "amount": 100})
        db.log_expense(500, "x", "Food", "2026-09-02 10:00:00")
        replies += [pm.budget_status({}), pm.safe_to_spend(), pm.export_tax({"year": "2026"})]
        for r in replies:
            assert isinstance(r["response"], str) and r["response"]
            assert 0.0 <= r["confidence"] <= 1.0, r


class TestRemainingMutationGaps:
    def test_weekly_totals_count_sub_unit_amounts(self):
        rows = [exp("anchor", 0.5, (TODAY - timedelta(days=9)).isoformat()), exp("a", 0.25, (TODAY - timedelta(days=1)).isoformat())]
        assert pl.weekly_totals(rows, TODAY, weeks=1) == [0.25]

    def test_the_lump_sum_low_case_uses_the_same_lump_sum(self, env):
        r = env[1].scenario_plan({"initial": 100000, "years": 10, "rate": 10})
        assert r["data"]["low"] == pytest.approx(100000 * 1.07 ** 10, rel=1e-9)
        assert r["data"]["high"] == pytest.approx(100000 * 1.13 ** 10, rel=1e-9)

    def test_history_queries_run_through_the_end_of_today(self, env):
        db, pm, _ = env
        seen = []
        original = db.get_expenses_between
        db.get_expenses_between = lambda a, b: (seen.append((a, b)), original(a, b))[1]
        pm.recurring_expenses(); pm.financial_health()
        assert seen and all(b == (TODAY + timedelta(days=1)).isoformat() for _, b in seen)
