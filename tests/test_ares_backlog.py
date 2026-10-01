# tests/test_ares_backlog.py
"""
Tests for Ares backlog items #153-#157:

  #153  career_ranking       GATE/career ranking with weighted criteria
  #154  schedule_review      "revisit this decision" reminders
  #155  record_outcome / outcome_stats, and confidence calibration
  #156  simulate_outcomes    Monte Carlo (no LLM involved)
  #157  save/list/run/delete playbook

Uses a FakeLLM that records prompts and an in-memory AresDB, so nothing
touches disk or the network.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.ares import simulate as sim
from modules.ares.engine import AresEngine
from modules.hecate.intent_registry import INTENT_MODULE_MAP


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDB:
    def get_goals(self, status="active"):
        return []

    def get_by_intent(self, intent, limit):
        return []


class FakeMemory:
    def __init__(self):
        self.db = FakeDB()
        self.reminders = []
        self.learned = {}

    def get_top_facts_for_context(self, limit=8):
        return ""

    def remember(self, query, n=6):
        return ""

    def learn(self, key, value, source="user"):
        self.learned[key] = value

    def add_reminder(self, text, due_time):
        self.reminders.append((text, due_time))


class FakeLLM:
    def __init__(self, response="{}"):
        self.response = response
        self.prompts = []

    def generate(self, prompt, fmt=None, options=None):
        self.prompts.append(prompt)
        return self.response


def make(response="{}", memory=None, **kw):
    llm = FakeLLM(response)
    mem = memory if memory is not None else FakeMemory()
    return AresEngine(memory=mem, ollama_cfg={}, llm=llm, **kw), llm, mem


DECISION_JSON = json.dumps({
    "recommendation": "Take A.",
    "options_analysis": [
        {"option": "A", "pros": ["x"], "cons": ["y"], "score": 8},
        {"option": "B", "pros": ["x"], "cons": ["y"], "score": 5},
    ],
    "key_factors": ["f"],
    "next_step": "Call them",
})

CAREER_JSON = json.dumps({
    "options": [
        {"option": "M.Tech", "scores": {"feasibility at my expected score": 6,
                                         "career outcomes": 7, "learning and research value": 9,
                                         "job security": 6, "cost and time": 4,
                                         "alignment with my goals": 8},
         "strengths": ["research depth"], "risks": ["two years of low income"]},
        {"option": "PSU", "scores": {"feasibility at my expected score": 4,
                                      "career outcomes": 8, "learning and research value": 4,
                                      "job security": 10, "cost and time": 8,
                                      "alignment with my goals": 5},
         "strengths": ["stability"], "risks": ["cutoff competition"]},
    ],
    "deciding_factor": "Whether research is the long-term goal",
    "next_step": "Check last year's official cutoffs",
    "data_to_verify": ["Official PSU cutoffs"],
})


# ---------------------------------------------------------------------------
# Registry / contract
# ---------------------------------------------------------------------------

NEW_INTENTS = [
    "career_ranking", "schedule_review", "record_outcome", "outcome_stats",
    "simulate_outcomes", "save_playbook", "list_playbooks", "run_playbook",
    "delete_playbook",
]


@pytest.mark.parametrize("intent", NEW_INTENTS)
def test_new_intents_are_registered_and_handled(intent):
    engine, _, _ = make()
    assert INTENT_MODULE_MAP[f"ares_{intent}"] == "ares"
    assert engine.can_handle(intent)


def test_construction_does_not_open_a_database():
    engine, _, _ = make()
    assert engine._db_instance is None


# ---------------------------------------------------------------------------
# #153 career ranking
# ---------------------------------------------------------------------------

def test_career_ranking_ranks_by_weighted_score_computed_in_python():
    engine, llm, _ = make(CAREER_JSON)
    res = engine.handle("career_ranking",
                        {"topic": "post-GATE path", "options": "M.Tech, PSU"}, {})
    ranking = res["data"]["ranking"]
    assert [r["option"] for r in ranking] == ["M.Tech", "PSU"]
    # weights 5,4,3,3,3,5 over scores 6,7,9,6,4,8 -> 155/23
    assert ranking[0]["score"] == pytest.approx(155 / 23, abs=0.05)
    assert "RANKING" in res["response"]
    assert "VERIFY BEFORE DECIDING" in res["response"]


def test_career_ranking_gate_mode_is_detected_and_scopes_the_prompt():
    engine, llm, _ = make(CAREER_JSON)
    res = engine.handle("career_ranking",
                        {"topic": "GATE results", "options": "M.Tech, PSU"}, {})
    assert res["data"]["gate_mode"] is True
    assert "GATE-exam-related" in llm.prompts[0]
    assert "feasibility at my expected score" in llm.prompts[0]
    assert "(GATE mode)" in res["response"]


def test_career_ranking_forbids_invented_numbers_in_the_prompt():
    engine, llm, _ = make(CAREER_JSON)
    engine.handle("career_ranking", {"topic": "x", "options": "A, B"}, {})
    assert "Do NOT state specific cutoffs" in llm.prompts[0]


def test_career_ranking_uses_user_criteria_with_weights():
    engine, llm, _ = make(json.dumps({"options": [
        {"option": "A", "scores": {"salary": 9, "location": 2}},
        {"option": "B", "scores": {"salary": 5, "location": 9}},
    ]}))
    res = engine.handle("career_ranking",
                        {"topic": "offers", "options": "A, B",
                         "criteria": "salary:5, location:1"}, {})
    assert res["data"]["criteria"] == {"salary": 5, "location": 1}
    assert [r["option"] for r in res["data"]["ranking"]] == ["A", "B"]
    assert res["data"]["ranking"][0]["score"] == pytest.approx((45 + 2) / 6, abs=0.05)


def test_career_ranking_asks_for_options_when_missing():
    engine, llm, _ = make(CAREER_JSON)
    res = engine.handle("career_ranking", {"topic": "my future"}, {})
    assert "What are they" in res["response"]
    assert llm.prompts == []


def test_career_ranking_survives_garbage_scores():
    engine, _, _ = make(json.dumps({"options": [
        {"option": "A", "scores": {"salary": "high"}}]}))
    res = engine.handle("career_ranking", {"topic": "t", "options": "A, B"}, {})
    assert res["confidence"] <= 0.3
    assert "couldn't get usable scores" in res["response"]


def test_career_ranking_ignores_unmatched_score_keys_but_maps_close_ones():
    engine, _, _ = make(json.dumps({"options": [
        {"option": "A", "scores": {"Earning Potential": 8, "banana": 3}}]}))
    res = engine.handle("career_ranking", {"topic": "t", "options": "A, B"}, {})
    assert list(res["data"]["ranking"][0]["scores"]) == ["earning potential"]


# ---------------------------------------------------------------------------
# #154 revisit reminders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,days", [
    ("in 2 weeks", 14), ("in three months", 90), ("tomorrow", 1),
    ("next week", 7), ("in a month", 30), ("in 10 days", 10),
    ("whenever", None), ("in 9999 years", 730),
])
def test_parse_delay_days(text, days):
    assert AresEngine._parse_delay_days(text) == days


def test_schedule_review_creates_a_reminder_and_links_the_decision():
    engine, _, mem = make(DECISION_JSON)
    engine.handle("decision_support", {"topic": "job offer", "options": "A, B"}, {})
    res = engine.handle("schedule_review",
                        {"topic": "job offer", "when": "in 2 weeks"}, {})
    assert len(mem.reminders) == 2  # decision_support's own next-step + this one
    text, due = mem.reminders[-1]
    assert "job offer" in text
    delta = datetime.fromisoformat(due) - datetime.now(timezone.utc)
    assert timedelta(days=13, hours=23) < delta < timedelta(days=14, minutes=1)
    assert res["data"]["decision_id"] is not None
    assert "revisit 'job offer'" in res["response"]


def test_schedule_review_defaults_to_two_weeks():
    engine, _, mem = make()
    engine.handle("schedule_review", {"topic": "thesis plan"}, {})
    delta = datetime.fromisoformat(mem.reminders[-1][1]) - datetime.now(timezone.utc)
    assert timedelta(days=13) < delta < timedelta(days=15)


def test_schedule_review_for_unknown_topic_still_sets_reminder_and_says_so():
    engine, _, mem = make()
    res = engine.handle("schedule_review", {"topic": "moving abroad", "when": "next month"}, {})
    assert len(mem.reminders) == 1
    assert "don't have a saved analysis" in res["response"]


def test_schedule_review_with_nothing_saved_and_no_topic_asks():
    engine, _, mem = make()
    res = engine.handle("schedule_review", {}, {})
    assert mem.reminders == []
    assert "Which decision" in res["response"]


def test_schedule_review_without_memory_degrades_cleanly():
    engine = AresEngine(memory=None, ollama_cfg={}, llm=FakeLLM())
    res = engine.handle("schedule_review", {"topic": "x", "when": "in 2 days"}, {})
    assert "aren't available" in res["response"]


def test_auto_review_days_schedules_a_reminder_on_every_decision():
    engine, _, mem = make(DECISION_JSON, auto_review_days=21)
    res = engine.handle("decision_support", {"topic": "job offer", "options": "A, B"}, {})
    assert "Review reminder set for" in res["response"]
    review = [r for r in mem.reminders if "Revisit" in r[0]]
    assert len(review) == 1
    delta = datetime.fromisoformat(review[0][1]) - datetime.now(timezone.utc)
    assert timedelta(days=20) < delta < timedelta(days=22)


def test_no_auto_review_by_default():
    engine, _, mem = make(DECISION_JSON)
    res = engine.handle("decision_support", {"topic": "job offer", "options": "A, B"}, {})
    assert "Review reminder" not in res["response"]
    assert not [r for r in mem.reminders if "Revisit" in r[0]]


# ---------------------------------------------------------------------------
# #155 outcome tracking + calibration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("it worked out great", "success"),
    ("it didn't work", "failure"),
    ("went badly", "failure"),
    ("partially failed", "mixed"),
    ("so-so", "mixed"),
    ("no idea", None),
])
def test_parse_outcome(text, expected):
    assert AresEngine._parse_outcome(text) == expected


def test_analyses_are_tracked_automatically():
    engine, _, _ = make(DECISION_JSON)
    res = engine.handle("decision_support", {"topic": "job offer", "options": "A, B"}, {})
    assert res["data"]["decision_id"] == 1
    row = engine._db.get_decision(1)
    assert row["kind"] == "decision"
    assert row["predicted_confidence"] == pytest.approx(0.8)


def test_record_outcome_attaches_to_the_matching_decision():
    engine, _, _ = make(DECISION_JSON)
    engine.handle("decision_support", {"topic": "job offer in Pune", "options": "A, B"}, {})
    res = engine.handle("record_outcome",
                        {"topic": "the Pune job offer", "outcome": "failure"}, {})
    assert res["data"]["decision_id"] == 1
    assert engine._db.get_decision(1)["outcome"] == "failure"
    assert "failed" in res["response"]


def test_record_outcome_without_a_clear_outcome_asks():
    engine, _, _ = make()
    res = engine.handle("record_outcome", {"topic": "x", "raw_query": "update on x"}, {})
    assert "did it work" in res["response"]


def test_record_outcome_for_untracked_topic_is_logged_but_flagged():
    engine, _, _ = make()
    res = engine.handle("record_outcome",
                        {"topic": "buying the bike", "outcome": "success"}, {})
    assert "no saved analysis" in res["response"]
    assert engine._db.resolved_with_confidence() == []  # can't skew calibration


def test_record_outcome_overwrites_and_says_so():
    engine, _, _ = make(DECISION_JSON)
    engine.handle("decision_support", {"topic": "job offer", "options": "A, B"}, {})
    engine.handle("record_outcome", {"topic": "job offer", "outcome": "failure"}, {})
    res = engine.handle("record_outcome", {"topic": "job offer", "outcome": "success"}, {})
    assert "replaces the earlier outcome: failure" in res["response"]


def _seed_resolved(engine, outcomes, confidence=0.8):
    for i, outcome in enumerate(outcomes):
        did = engine._db.add_decision("decision", f"topic {i}", "", None, confidence)
        engine._db.record_outcome(did, outcome)


def test_no_calibration_below_minimum_sample():
    engine, _, _ = make(DECISION_JSON)
    _seed_resolved(engine, ["failure", "failure"])
    res = engine.handle("decision_support", {"topic": "new one", "options": "A, B"}, {})
    assert "TRACK RECORD" not in res["response"]
    assert "calibrated_confidence" not in res["data"]


def test_overconfident_history_lowers_confidence():
    engine, _, _ = make(DECISION_JSON)
    _seed_resolved(engine, ["failure", "failure", "mixed", "failure"], confidence=0.8)
    res = engine.handle("decision_support", {"topic": "new one", "options": "A, B"}, {})
    # mean predicted 0.8, actual 0.125, weight 0.4 -> 0.8 - 0.675*0.4 = 0.53
    assert res["data"]["calibrated_confidence"] == pytest.approx(0.53, abs=0.01)
    assert "TRACK RECORD" in res["response"]
    assert "scored higher than they turned out" in res["response"]


def test_underconfident_history_raises_confidence():
    engine, _, _ = make(DECISION_JSON)
    _seed_resolved(engine, ["success"] * 10, confidence=0.5)
    res = engine.handle("decision_support", {"topic": "new one", "options": "A, B"}, {})
    assert res["data"]["calibrated_confidence"] == pytest.approx(0.95, abs=0.01)  # clamped
    assert "turned out better than they scored" in res["response"]


def test_calibrated_confidence_is_clamped():
    engine, _, _ = make()
    cal = {"n": 10, "mean_predicted": 0.9, "actual_rate": 0.0, "gap": 0.9, "weight": 1.0}
    assert engine._calibrated(0.9, cal) == 0.05


def test_outcome_stats_empty_and_populated():
    engine, _, _ = make(DECISION_JSON)
    assert "haven't tracked" in engine.handle("outcome_stats", {}, {})["response"]
    engine.handle("decision_support", {"topic": "job offer", "options": "A, B"}, {})
    engine.handle("decision_support", {"topic": "flat hunt", "options": "A, B"}, {})
    engine.handle("record_outcome", {"topic": "flat hunt", "outcome": "success"}, {})
    res = engine.handle("outcome_stats", {}, {})
    assert "Tracked: 2" in res["response"]
    assert "Resolved: 1" in res["response"]
    assert "Waiting on an outcome: 1" in res["response"]
    assert res["data"]["counts"] == {"pending": 1, "success": 1}


def test_outcome_stats_lists_due_reviews():
    engine, _, _ = make()
    did = engine._db.add_decision("plan", "thesis plan", "")
    engine._db.set_review(did, (datetime.now(timezone.utc) - timedelta(days=1)).isoformat())
    res = engine.handle("outcome_stats", {}, {})
    assert "DUE FOR REVIEW" in res["response"]
    assert res["data"]["due_reviews"] == 1


def test_tracking_failure_never_breaks_the_analysis(monkeypatch):
    engine, _, _ = make(DECISION_JSON)
    monkeypatch.setattr(engine._db, "add_decision",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    res = engine.handle("decision_support", {"topic": "job offer", "options": "A, B"}, {})
    assert "RECOMMENDATION" in res["response"]
    assert "decision_id" not in res["data"]


# ---------------------------------------------------------------------------
# #156 Monte Carlo
# ---------------------------------------------------------------------------

def test_simulate_from_free_text_makes_no_llm_call():
    engine, llm, _ = make()
    res = engine.handle("simulate_outcomes", {
        "topic": "the bet",
        "raw_query": "simulate: 60% chance of gaining 10 lakh, 40% chance of losing 2 lakh",
        "seed": 7,
    }, {})
    assert llm.prompts == []
    best = res["data"]["options"][0]
    assert best["expected_value"] == pytest.approx(0.6 * 1_000_000 - 0.4 * 200_000)
    assert best["p_loss"] == pytest.approx(0.4, abs=0.03)
    assert "Chance of loss : " in res["response"]


def test_simulate_triangular_range():
    engine, _, _ = make()
    res = engine.handle("simulate_outcomes", {
        "topic": "project", "low": "50k", "likely": "100k", "high": "200k", "seed": 1}, {})
    r = res["data"]["options"][0]
    assert r["expected_value"] == pytest.approx((50_000 + 100_000 + 200_000) / 3)
    assert 50_000 <= r["worst"] and r["best"] <= 200_000
    assert r["p_loss"] == 0


def test_simulate_is_reproducible_with_a_seed():
    engine, _, _ = make()
    ent = {"topic": "t", "low": 0, "likely": 5, "high": 20, "seed": 42}
    a = engine.handle("simulate_outcomes", dict(ent), {})["data"]["options"][0]
    b = engine.handle("simulate_outcomes", dict(ent), {})["data"]["options"][0]
    assert a["median"] == b["median"] and a["stdev"] == b["stdev"]


def test_simulate_compares_options_and_reports_head_to_head():
    engine, _, _ = make()
    res = engine.handle("simulate_outcomes", {"topic": "A vs B", "seed": 3, "options": [
        {"name": "Safe", "scenarios": [{"probability": 1.0, "value": 100}]},
        {"name": "Risky", "scenarios": [{"probability": 0.5, "value": 400},
                                        {"probability": 0.5, "value": -100}]},
    ]}, {})
    names = [o["name"] for o in res["data"]["options"]]
    assert names == ["Risky", "Safe"]  # EV 150 vs 100
    assert res["data"]["p_top_beats_next"] == pytest.approx(0.5, abs=0.03)
    assert "higher chance of loss" in res["response"]


def test_simulate_percent_probabilities_and_cost():
    r = sim.run([{"scenarios": [{"probability": 50, "value": 300},
                                {"probability": 50, "value": 100}], "cost": 150}], seed=1)
    assert r["options"][0]["expected_value"] == pytest.approx(50)
    assert r["options"][0]["p_loss"] == pytest.approx(0.5, abs=0.03)


def test_simulate_probabilities_summing_below_one_get_an_explicit_note():
    r = sim.run([{"scenarios": [{"probability": 0.3, "value": 100}]}], seed=1)
    assert any("chance of nothing happening" in n for n in r["options"][0]["notes"])
    assert r["options"][0]["expected_value"] == pytest.approx(30)


def test_simulate_probabilities_summing_above_one_are_scaled_and_flagged():
    r = sim.run([{"scenarios": [{"probability": 0.8, "value": 100},
                                {"probability": 0.8, "value": 0}]}], seed=1)
    assert any("scaled" in n for n in r["options"][0]["notes"])


@pytest.mark.parametrize("entities", [
    {"topic": "t", "scenarios": [{"probability": 0.5}]},
    {"topic": "t", "low": "abc", "high": "10"},
    {"topic": "t", "low": 0, "likely": 99, "high": 10},
    {"topic": "t", "scenarios": [{"probability": -1, "value": 5}]},
])
def test_simulate_bad_input_is_a_friendly_message_not_a_crash(entities):
    engine, _, _ = make()
    res = engine.handle("simulate_outcomes", entities, {})
    assert res["confidence"] < 0.6
    assert "couldn't run that simulation" in res["response"]


def test_simulate_without_numbers_explains_what_it_needs():
    engine, _, _ = make()
    res = engine.handle("simulate_outcomes", {"topic": "t", "raw_query": "should I do it"}, {})
    assert "I need numbers" in res["response"]


def test_run_count_is_clamped():
    r = sim.run([{"low": 0, "high": 1}], runs=10**9, seed=1)
    assert r["runs"] == sim.MAX_RUNS


@pytest.mark.parametrize("text,value", [
    ("2 lakh", 200_000), ("1.5 crore", 15_000_000), ("75k", 75_000),
    ("₹1,50,000", 150_000), ("-3m", -3_000_000), ("$1,200", 1200),
])
def test_parse_amount(text, value):
    assert sim.parse_amount(text) == value


# ---------------------------------------------------------------------------
# #157 playbooks
# ---------------------------------------------------------------------------

def test_save_playbook_from_entities_and_list_it():
    engine, _, _ = make()
    res = engine.handle("save_playbook", {
        "name": "Job Offer", "analysis": "premortem",
        "criteria": "salary, growth, commute"}, {})
    assert "Saved playbook 'Job Offer' (premortem)" in res["response"]
    listing = engine.handle("list_playbooks", {}, {})
    assert "Job Offer: premortem (covers salary, growth, commute)" in listing["response"]


def test_save_playbook_parses_a_plain_sentence():
    engine, _, _ = make()
    res = engine.handle("save_playbook", {"raw_query": (
        "save a playbook called job offer that runs a premortem and always "
        "considers salary, growth and commute")}, {})
    assert res["data"]["name"] == "job offer"
    assert res["data"]["analysis"] == "premortem_analysis"
    assert "salary, growth and commute" in res["data"]["criteria"]


def test_save_playbook_name_containing_an_analysis_word_is_not_misread():
    engine, _, _ = make()
    res = engine.handle("save_playbook", {
        "raw_query": "save a playbook called risky bets that runs a swot"}, {})
    assert res["data"]["analysis"] == "swot_analysis"


def test_save_playbook_asks_when_name_or_analysis_missing():
    engine, _, _ = make()
    assert "call this playbook" in engine.handle("save_playbook", {"raw_query": "save a playbook"}, {})["response"]
    res = engine.handle("save_playbook", {"name": "x"}, {})
    assert "Which analysis" in res["response"]


def test_save_playbook_again_updates_instead_of_duplicating():
    engine, _, _ = make()
    engine.handle("save_playbook", {"name": "x", "analysis": "swot"}, {})
    res = engine.handle("save_playbook", {"name": "X", "analysis": "risk"}, {})
    assert res["data"]["updated"] is True
    assert len(engine._db.list_playbooks()) == 1
    assert engine._db.get_playbook("x")["analysis"] == "analyse_risk"


def test_run_playbook_runs_the_analysis_with_the_saved_criteria():
    engine, llm, mem = make(json.dumps({
        "scenario": "It failed", "failure_causes": [], "single_point_of_failure": "x",
        "confidence_in_success": "Low"}))
    engine.handle("save_playbook", {"name": "job offer", "analysis": "premortem",
                                    "criteria": "salary, growth, commute"}, {})
    res = engine.handle("run_playbook", {"name": "job offer", "topic": "Pune offer"}, {})
    assert res["response"].startswith("Playbook: job offer")
    assert "Premortem: Pune Offer" in res["response"]
    assert "salary, growth, commute" in llm.prompts[0]
    assert "Standing criteria" in llm.prompts[0]
    assert res["data"]["playbook"] == "job offer"
    assert engine._db.get_playbook("job offer")["uses"] == 1


def test_playbook_criteria_reach_the_prompt_even_without_memory():
    llm = FakeLLM(json.dumps({"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}))
    engine = AresEngine(memory=None, ollama_cfg={}, llm=llm)
    engine.handle("save_playbook", {"name": "p", "analysis": "swot", "criteria": "cost, time"}, {})
    engine.handle("run_playbook", {"name": "p", "topic": "move"}, {})
    assert "cost, time" in llm.prompts[0]


def test_criteria_do_not_leak_into_later_plain_analyses():
    engine, llm, _ = make(json.dumps({"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}))
    engine.handle("save_playbook", {"name": "p", "analysis": "swot", "criteria": "ZZZ-CRITERIA"}, {})
    engine.handle("run_playbook", {"name": "p", "topic": "move"}, {})
    engine.handle("swot_analysis", {"topic": "other"}, {})
    assert "ZZZ-CRITERIA" in llm.prompts[0]
    assert "ZZZ-CRITERIA" not in llm.prompts[1]


def test_run_playbook_finds_name_and_topic_in_a_sentence():
    engine, llm, _ = make(json.dumps({"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}))
    engine.handle("save_playbook", {"name": "job offer", "analysis": "swot"}, {})
    res = engine.handle("run_playbook",
                        {"raw_query": "run my job offer playbook on the Bangalore role"}, {})
    assert "SWOT Analysis: The Bangalore Role" in res["response"]


def test_run_playbook_uses_stored_topic_and_options_for_decisions():
    engine, llm, _ = make(DECISION_JSON)
    engine.handle("save_playbook", {"name": "laptop", "analysis": "decision",
                                    "topic": "which laptop", "options": "Mac, ThinkPad"}, {})
    res = engine.handle("run_playbook", {"name": "laptop"}, {})
    assert "Mac, ThinkPad" in llm.prompts[0]
    assert "Decision Support: Which Laptop" in res["response"]


def test_run_playbook_asks_for_a_topic_when_none_available():
    engine, llm, _ = make()
    engine.handle("save_playbook", {"name": "p", "analysis": "swot"}, {})
    res = engine.handle("run_playbook", {"name": "p"}, {})
    assert "What should I run" in res["response"]
    assert llm.prompts == []


def test_run_unknown_playbook_lists_what_exists():
    engine, _, _ = make()
    engine.handle("save_playbook", {"name": "alpha", "analysis": "swot"}, {})
    res = engine.handle("run_playbook", {"name": "beta", "topic": "x"}, {})
    assert "don't have a playbook called 'beta'" in res["response"]
    assert "alpha" in res["response"]


def test_delete_playbook():
    engine, _, _ = make()
    engine.handle("save_playbook", {"name": "alpha", "analysis": "swot"}, {})
    res = engine.handle("delete_playbook", {"name": "alpha"}, {})
    assert "Deleted playbook 'alpha'" in res["response"]
    assert engine._db.list_playbooks() == []
    assert "couldn't find" in engine.handle("delete_playbook", {"name": "alpha"}, {})["response"]


def test_list_playbooks_empty_gives_an_example():
    engine, _, _ = make()
    assert "haven't saved any playbooks" in engine.handle("list_playbooks", {}, {})["response"]


def test_playbooks_persist_in_a_real_database_file(tmp_path):
    path = str(tmp_path / "ares.db")
    first = AresEngine(memory=None, ollama_cfg={}, llm=FakeLLM(), db_path=path)
    first.handle("save_playbook", {"name": "keep me", "analysis": "swot"}, {})
    second = AresEngine(memory=None, ollama_cfg={}, llm=FakeLLM(), db_path=path)
    assert [p["name"] for p in second._db.list_playbooks()] == ["keep me"]
