# tests/test_dionysus_group.py
"""Tests for Dionysus backlog #148: group / friends outing coordination."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.dionysus import group as g
from modules.dionysus.db import DionysusDB
from modules.dionysus.engine import DionysusEngine
from modules.hecate.intent_registry import INTENT_MODULE_MAP


class TestParseAvailability:
    def test_day_and_part(self):
        assert g.parse_availability("sat evening") == {("sat", "evening")}

    def test_day_alone_means_all_day(self):
        assert g.parse_availability("sunday") == {("sun", p) for p in g.PARTS}

    def test_clauses_and_weekend(self):
        s = g.parse_availability("sat evening, sun afternoon")
        assert s == {("sat", "evening"), ("sun", "afternoon")}
        assert g.parse_availability("weekend evening") == {("sat", "evening"), ("sun", "evening")}

    def test_weekdays_evenings_and_anytime(self):
        assert len(g.parse_availability("weekdays after work")) == 5
        assert g.parse_availability("evenings") == {(d, "evening") for d in g.DAYS}
        assert g.parse_availability("anytime") == g.ALL_SLOTS

    @pytest.mark.parametrize("bad", ["", None, 5, "no idea", "blah blah", ["sat"]])
    def test_unreadable_is_empty_not_guessed(self, bad):
        assert g.parse_availability(bad) == frozenset()

    def test_huge_input_is_capped(self):
        assert g.parse_availability("sat " * 100000)  # returns promptly, no crash


class TestOverlap:
    def _p(self, **kw):
        return {n: {"slots": g.parse_availability(a), "likes": g.parse_list(l), "dislikes": g.parse_list(d), "budget": b}
                for n, (a, l, d, b) in kw.items()}

    def test_everyone_free(self):
        r = g.overlap(self._p(a=("sat evening, sun", "", "", None), b=("sat, sun evening", "", "", None)))
        assert r["everyone"] == [("sat", "evening"), ("sun", "evening")]

    def test_no_common_time_shows_who_is_missing(self):
        r = g.overlap(self._p(a=("mon", "", "", None), b=("tue", "", "", None)))
        assert r["everyone"] == [] and r["partial"][0][2]
        assert "no time when everyone is free" in g.render_plan("x", self._p(a=("mon", "", "", None), b=("tue", "", "", None)), r)

    def test_unreadable_person_is_excluded_from_timing_and_named(self):
        r = g.overlap(self._p(a=("sat", "", "", None), b=("not sure yet", "", "", None)))
        assert r["unreadable"] == ["b"] and r["everyone"]

    def test_dislike_vetoes_a_like(self):
        r = g.overlap(self._p(a=("sat", "thai, bowling", "", None), b=("sat", "bowling", "loud places, bowling", None)))
        assert [i for i, _ in r["likes"]] == ["thai"] and r["vetoed"] == ["bowling"]

    def test_shared_likes_rank_by_count_and_budget_is_the_lowest(self):
        r = g.overlap(self._p(a=("sat", "thai, pizza", "", 1000), b=("sat", "thai", "", 600), c=("sat", "", "", None)))
        assert r["likes"][0] == ("thai", 2) and r["budget_cap"] == 600

    def test_empty_and_single_person(self):
        assert g.overlap({})["everyone"] == []
        t = g.render_plan("x", self._p(a=("sat", "", "", None)), g.overlap(self._p(a=("sat", "", "", None))))
        assert "Add your friends" in t


@pytest.fixture
def eng(tmp_path):
    e = DionysusEngine(ollama_cfg={}, llm=None)
    e.db = DionysusDB(str(tmp_path / "d.db"))
    return e


class TestEngine:
    def test_intents_registered_everywhere(self, eng):
        for i in ("set_outing_preferences", "plan_group_outing", "clear_group_outing"):
            assert eng.can_handle(i) and INTENT_MODULE_MAP[f"dionysus_{i}"] == "dionysus"

    def test_full_flow(self, eng):
        h = lambda i, e: eng.handle(i, e, {})
        assert "nobody" in h("plan_group_outing", {})["response"]
        h("set_outing_preferences", {"person": "Asha", "availability": "sat evening, sun afternoon", "likes": "thai and bowling", "budget": "800"})
        h("set_outing_preferences", {"person": "Ravi", "availability": "sat evening", "dislikes": "bowling", "budget": "under 1000"})
        r = h("plan_group_outing", {})
        assert r["data"]["everyone_free"] == ["Sat evening"]
        assert r["data"]["budget_cap"] == 800 and "thai" in r["response"] and "bowling" in r["response"]
        assert [i for i, _ in r["data"]["likes"]] == ["thai"]

    def test_update_merges_only_given_fields(self, eng):
        eng.handle("set_outing_preferences", {"person": "A", "availability": "sat", "likes": "thai"}, {})
        eng.handle("set_outing_preferences", {"person": "a", "budget": "500"}, {})
        row = eng.db.get_group("friends")
        assert len(row) == 1 and row[0]["availability"] == "sat" and row[0]["likes"] == "thai" and row[0]["budget"] == 500

    def test_groups_are_separate(self, eng):
        eng.handle("set_outing_preferences", {"group": "work", "person": "A", "availability": "fri"}, {})
        assert "nobody" in eng.handle("plan_group_outing", {"group": "family"}, {})["response"]
        assert eng.handle("plan_group_outing", {"group": "WORK"}, {})["data"]["everyone_free"]

    def test_bad_input_messages(self, eng):
        assert "What should I note" in eng.handle("set_outing_preferences", {"person": "A"}, {})["response"]
        assert "couldn't read that budget" in eng.handle("set_outing_preferences", {"person": "A", "budget": "lots"}, {})["response"]
        r = eng.handle("set_outing_preferences", {"person": "A", "availability": "blah"}, {})
        assert "couldn't read those times" in r["response"]

    def test_group_size_cap(self, eng):
        for i in range(g.MAX_PEOPLE):
            eng.handle("set_outing_preferences", {"person": f"p{i}", "likes": "x"}, {})
        assert "at most" in eng.handle("set_outing_preferences", {"person": "extra", "likes": "x"}, {})["response"]
        assert "noted" in eng.handle("set_outing_preferences", {"person": "p0", "likes": "y"}, {})["response"].lower()

    def test_clear_person_and_group(self, eng):
        eng.handle("set_outing_preferences", {"person": "A", "likes": "x"}, {})
        eng.handle("set_outing_preferences", {"person": "B", "likes": "x"}, {})
        assert "Removed A" in eng.handle("clear_group_outing", {"person": "A"}, {})["response"]
        assert "isn't in" in eng.handle("clear_group_outing", {"person": "A"}, {})["response"]
        assert "Cleared 'friends' (1 person)" in eng.handle("clear_group_outing", {}, {})["response"]
        assert "already empty" in eng.handle("clear_group_outing", {}, {})["response"]

    def test_sql_injection_in_names_is_just_text(self, eng):
        eng.handle("set_outing_preferences", {"person": "x'); DROP TABLE outing_people;--", "likes": "y"}, {})
        assert len(eng.db.get_group("friends")) == 1
