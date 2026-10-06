# tests/test_shadow.py
"""Backlog #16: shadow mode for new intent handlers."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.shadow import (
    ShadowRecorder, ShadowRules, compare_replies,
)
from modules.base import BaseModule
from modules.hestia.orchestrator import HestiaOrchestrator


def _rule(**over):
    r = {"name": "w2", "intent": "get_weather", "candidate": {"module": "newmod", "intent": "weather_v2"},
         "mode": "shadow", "read_only": True, "min_samples": 3, "agreement": 0.8}
    r.update(over)
    return r


class Old(BaseModule):
    name = "oldmod"
    calls = 0
    def can_handle(self, i): return i == "get_weather"
    def handle(self, i, e, c):
        Old.calls += 1
        return {"response": "It is 30 degrees and sunny.", "data": {}, "confidence": 1.0}


class New(BaseModule):
    name = "newmod"
    reply = "It is 30 degrees and sunny."
    calls = 0
    boom = False
    def can_handle(self, i): return i == "weather_v2"
    def handle(self, i, e, c):
        New.calls += 1
        if New.boom:
            raise RuntimeError("candidate bug")
        return {"response": New.reply, "data": {}, "confidence": 1.0}


class FakeHecate:
    def decide(self, q, nlu, active):
        return {"primary": "oldmod", "secondary": [], "confidence": 1.0, "reason": "t",
                "synthesize": False, "intent": None, "conference": None, "checked": []}


def _setup(tmp_path, rule=None, sync=True):
    Old.calls = New.calls = 0
    New.reply, New.boom = "It is 30 degrees and sunny.", False
    o = HestiaOrchestrator()
    o._hecate = FakeHecate()
    o.register(Old())
    o.register(New())
    rules = ShadowRules([rule or _rule()], enabled=True)
    rec = ShadowRecorder(rules, tmp_path / "shadow.jsonl", synchronous=sync)
    o.attach_shadow(rec)
    return o, rec


def _ask(o):
    return o.dispatch("what's the weather", {"intent": "get_weather", "entities": {}, "confidence": 0.9})


# ----------------------------------------------------------- comparison

def test_identical_replies_agree_fully():
    assert compare_replies("Hello  World", "hello world") == {"identical": True, "similarity": 1.0, "agree": True}


def test_similar_replies_agree_above_threshold_and_not_below():
    assert compare_replies("it is 30 degrees and sunny", "it is 31 degrees and sunny")["agree"]
    assert not compare_replies("it is sunny", "your calendar is empty today")["agree"]


# ----------------------------------------------------------- rule validation

def test_shadow_rule_must_declare_read_only():
    r = ShadowRules([_rule(read_only=False)], enabled=True)
    assert r.rules == [] and "read_only" in r.rejected[0]


def test_live_mode_does_not_need_read_only():
    r = ShadowRules([_rule(mode="live", read_only=False)], enabled=True)
    assert len(r.rules) == 1


def test_bad_rules_are_rejected_with_reasons():
    r = ShadowRules([{"name": "x"}, _rule(mode="sideways"), "junk", _rule(candidate={"module": "a"})], enabled=True)
    assert r.rules == [] and len(r.rejected) == 4


def test_disabled_rules_never_match():
    assert ShadowRules([_rule()], enabled=False).rule_for("get_weather") is None


def test_from_config_defaults_to_disabled():
    assert ShadowRules.from_config(None).enabled is False
    assert ShadowRules.from_config({"rules": [_rule()]}).enabled is False


# ----------------------------------------------------------- orchestrator modes

def test_shadow_mode_serves_baseline_and_records_candidate(tmp_path):
    o, rec = _setup(tmp_path)
    assert _ask(o) == "It is 30 degrees and sunny."
    assert Old.calls == 1 and New.calls == 1
    r = rec.records()[0]
    assert r["served_by"] == "baseline" and r["agree"] and r["identical"]
    assert r["served"] == "oldmod.get_weather" and r["other"] == "newmod.weather_v2"


def test_shadow_mode_never_shows_the_candidates_reply(tmp_path):
    o, rec = _setup(tmp_path)
    New.reply = "TOTALLY DIFFERENT"
    assert _ask(o) == "It is 30 degrees and sunny."
    r = rec.records()[0]
    assert not r["agree"] and r["other_reply"] == "TOTALLY DIFFERENT"


def test_candidate_error_is_recorded_and_user_still_gets_the_baseline(tmp_path):
    o, rec = _setup(tmp_path)
    New.boom = True
    assert _ask(o) == "It is 30 degrees and sunny."
    r = rec.records()[0]
    assert r["candidate_error"] and not r["agree"]


def test_canary_mode_serves_candidate_and_compares_baseline(tmp_path):
    o, rec = _setup(tmp_path, _rule(mode="canary"))
    New.reply = "Candidate says sunny."
    assert _ask(o) == "Candidate says sunny."
    r = rec.records()[0]
    assert r["served_by"] == "candidate" and r["other"] == "oldmod.get_weather"
    assert r["baseline_error"] is None


def test_live_mode_serves_candidate_and_runs_nothing_else(tmp_path):
    o, rec = _setup(tmp_path, _rule(mode="live", read_only=False))
    New.reply = "Live reply."
    assert _ask(o) == "Live reply."
    assert Old.calls == 0 and rec.records() == []


def test_intent_without_a_rule_is_untouched(tmp_path):
    o, rec = _setup(tmp_path)
    class Other(BaseModule):
        name = "oldmod2"
        def can_handle(self, i): return i == "other"
        def handle(self, i, e, c): return {"response": "plain", "data": {}, "confidence": 1}
    o.register(Other())
    o._hecate = type("H", (), {"decide": lambda s, q, n, a: {
        "primary": "oldmod2", "secondary": [], "confidence": 1.0, "reason": "", "synthesize": False,
        "intent": None, "conference": None, "checked": []}})()
    assert o.dispatch("x", {"intent": "other", "entities": {}, "confidence": 0.9}) == "plain"
    assert rec.records() == [] and New.calls == 0


def test_no_shadow_attached_is_the_old_behaviour():
    Old.calls = 0
    o = HestiaOrchestrator()
    o._hecate = FakeHecate()
    o.register(Old())
    assert _ask(o) == "It is 30 degrees and sunny." and Old.calls == 1


def test_missing_candidate_module_is_recorded_as_unavailable(tmp_path):
    o, rec = _setup(tmp_path, _rule(candidate={"module": "ghost", "intent": "x"}))
    assert _ask(o) == "It is 30 degrees and sunny."
    assert "unavailable" in rec.records()[0]["candidate_error"]


def test_async_mode_does_not_block_the_reply(tmp_path):
    o, rec = _setup(tmp_path, sync=False)
    t0 = time.time()
    assert _ask(o)
    assert time.time() - t0 < 1.0
    end = time.time() + 2
    while time.time() < end and not rec.records():
        time.sleep(0.02)
    assert rec.records()


# ----------------------------------------------------------- reporting

def test_report_says_ready_only_after_enough_agreeing_samples(tmp_path):
    o, rec = _setup(tmp_path)
    for _ in range(2):
        _ask(o)
    assert rec.report()["w2"]["ready_to_promote"] is False
    assert "not ready (2/3 samples)" in rec.summary()
    _ask(o)
    rep = rec.report()["w2"]
    assert rep["samples"] == 3 and rep["agreement_rate"] == 1.0 and rep["ready_to_promote"]
    assert "READY" in rec.summary() and "still has to change the mode" in rec.summary()


def test_one_candidate_error_blocks_promotion(tmp_path):
    o, rec = _setup(tmp_path)
    for _ in range(3):
        _ask(o)
    New.boom = True
    _ask(o)
    rep = rec.report()["w2"]
    assert rep["candidate_errors"] == 1 and not rep["ready_to_promote"]


def test_low_agreement_blocks_promotion(tmp_path):
    o, rec = _setup(tmp_path)
    New.reply = "nothing alike at all"
    for _ in range(4):
        _ask(o)
    assert rec.report()["w2"]["agreement_rate"] == 0.0
    assert len(rec.disagreements()) == 4


def test_summary_with_no_rules_and_with_rejected_rules(tmp_path):
    assert "No shadow rules" in ShadowRecorder(ShadowRules([], True), tmp_path / "a.jsonl").summary()
    s = ShadowRecorder(ShadowRules([_rule(read_only=False)], True), tmp_path / "b.jsonl").summary()
    assert "REJECTED" in s


def test_log_lines_are_valid_json(tmp_path):
    o, rec = _setup(tmp_path)
    _ask(o)
    for line in (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["rule"] == "w2"
