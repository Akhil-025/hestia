# tests/test_decision_engine.py
"""
Tests for the "Hecate — Decision Engine" backlog items (section 14):

  #158  multi-module conference     (core/conference.py + Hecate's conference tier)
  #160  what-if simulator           (core/whatif.py)
  #162  routing audit trail         (Hecate's ``checked`` + Diagnostics.audit_last)
  #163  weekly critical-path list   (Chronos build_week_focus / weekly_focus)

Everything here uses small fakes; nothing touches a database, a model or the
network.
"""
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.conference import LENSES, MAX_PERSPECTIVE_CHARS, Conference, _trim
from core.whatif import (
    LOW_SLEEP_HOURS, WhatIfEngine, parse_amount_per_month, parse_sleep_change,
)
from modules.hecate import HecateEngine
from modules.hestia.orchestrator import HestiaOrchestrator
from modules.base import BaseModule

ALL = ["core", "pluto", "ares", "apollo", "artemis", "chronos"]
TODAY = date(2026, 10, 5)  # a Monday


def decide(query, intent="chat", conf=0.9, active=None, entities=None):
    nlu = {"intent": intent, "confidence": conf, "entities": entities or {}}
    return HecateEngine().decide(query, nlu, active if active is not None else ALL)


# ===========================================================================
# #162 — audit trail
# ===========================================================================

class TestAuditTrail:
    def test_every_decision_has_a_nonempty_trail(self):
        for intent in ("chat", "pluto_log_expense", "weekly_focus", "nonsense_intent"):
            d = decide("hello there", intent)
            assert isinstance(d["checked"], list) and d["checked"]
            assert all(isinstance(s, str) and s for s in d["checked"])

    def test_trail_starts_with_what_nlu_said(self):
        d = decide("log my expense", "pluto_log_expense", 0.93)
        assert "pluto_log_expense" in d["checked"][0] and "93%" in d["checked"][0]

    def test_trail_records_the_deciding_registry_hit(self):
        d = decide("log my expense", "pluto_log_expense")
        assert d["primary"] == "pluto"
        assert any("registry" in s and "pluto" in s for s in d["checked"])

    def test_trail_records_checks_that_did_not_decide(self):
        d = decide("just chatting about the weather", "chat", 0.6)
        text = " ".join(d["checked"])
        assert "text triggers" in text and "none matched" in text
        assert "prefix fallback" in text and "keyword match" in text

    def test_trail_notes_an_inactive_owning_module(self):
        d = decide("log my expense", "pluto_log_expense", 0.9, active=["core"])
        assert d["primary"] == "core"
        assert any("pluto" in s and "isn't active" in s for s in d["checked"])

    def test_trail_records_the_confidence_gate(self):
        d = decide("something", "pluto_log_expense", 0.2)
        assert d["intent"] == "clarify_intent"
        assert any("confidence gate" in s and "floor" in s for s in d["checked"])

    def test_trail_records_text_trigger_that_matched(self):
        d = decide("what did we talk about yesterday", "chat", 0.5)
        assert any("text trigger matched" in s for s in d["checked"])

    def test_trails_are_independent_between_calls(self):
        h = HecateEngine()
        nlu = {"intent": "chat", "confidence": 0.6, "entities": {}}
        a = h.decide("hi", nlu, ALL)["checked"]
        b = h.decide("hi", nlu, ALL)["checked"]
        assert a == b and a is not b

    def test_decision_keys_are_complete(self):
        d = decide("hi")
        assert {"primary", "secondary", "confidence", "reason", "synthesize",
                "intent", "conference", "checked"} <= set(d)


@pytest.fixture
def diag(tmp_path, monkeypatch):
    """Diagnostics with its log directory pointed at tmp_path (same pattern
    as tests/test_observability.py)."""
    monkeypatch.setenv("HESTIA_LOG_DIR", str(tmp_path))
    import importlib, logging
    import core.observability as obs
    obs = importlib.reload(obs)
    yield obs.Diagnostics()
    for name in ("hestia.routing", "hestia.feedback"):
        log = logging.getLogger(name)
        for h in list(log.handlers):
            h.close()
            log.removeHandler(h)


class TestDiagnosticsAudit:
    def _rec(self, diag, query, intent, **kw):
        return diag.record_classification(
            query=query, intent=intent, confidence=0.9, module="pluto",
            reason="r", **kw)

    def test_checked_is_stored_on_the_record(self, diag):
        rec = self._rec(diag, "log", "pluto_log_expense", checked=["a", "b"])
        assert rec["checked"] == ["a", "b"]

    def test_record_without_checked_has_no_key(self, diag):
        assert "checked" not in self._rec(diag, "log", "pluto_log_expense")

    def test_checked_is_bounded(self, diag):
        rec = self._rec(diag, "q", "chat", checked=["x" * 1000] * 100)
        assert len(rec["checked"]) == 30 and all(len(s) <= 300 for s in rec["checked"])

    def test_audit_lists_steps_in_order(self, diag):
        self._rec(diag, "log my lunch", "pluto_log_expense",
                  checked=["NLU said X", "registry hit"])
        out = diag.audit_last()
        assert "log my lunch" in out
        assert out.index("1. NLU said X") < out.index("2. registry hit")
        assert "'pluto'" in out

    def test_audit_with_nothing_routed(self, diag):
        assert "nothing to audit" in diag.audit_last()

    def test_audit_without_a_recorded_trail_is_honest(self, diag):
        self._rec(diag, "old query", "chat")
        out = diag.audit_last()
        assert "didn't keep a step-by-step record" in out and "pluto" in out

    def test_audit_skips_meta_questions(self, diag):
        self._rec(diag, "log my lunch", "pluto_log_expense", checked=["step one"])
        self._rec(diag, "what did you check", "audit_routing", checked=["meta"])
        self._rec(diag, "why did you route that", "explain_routing", checked=["meta2"])
        assert "log my lunch" in diag.audit_last()
        assert "log my lunch" in diag.explain_last()

    def test_core_module_audit_intent(self, diag):
        from modules.hestia.core_module import CoreModule
        self._rec(diag, "log my lunch", "pluto_log_expense", checked=["step one"])
        core = CoreModule(None, {}, diagnostics=diag)
        res = core.handle("audit_routing", {}, {})
        assert "step one" in res["response"] and res["data"]["checked"] == ["step one"]

    def test_core_module_audit_without_diagnostics(self):
        from modules.hestia.core_module import CoreModule
        res = CoreModule(None, {}).handle("audit_routing", {}, {})
        assert res["response"] and res["confidence"] < 0.5


# ===========================================================================
# #158 — Hecate's conference tier
# ===========================================================================

class TestConferenceDecision:
    def test_money_and_health_topic_seats_both_sides(self):
        d = decide("should I buy a laptop, my budget is tight and I'm stressed",
                   "conference")
        assert d["primary"] == "core" and d["intent"] == "conference"
        assert d["conference"] and len(d["conference"]) >= 2
        assert "pluto" in d["conference"] and "apollo" in d["conference"]
        assert d["secondary"] == []

    def test_seats_are_capped_at_three(self):
        d = decide("my budget, my sleep, my exam, my habits", "conference")
        assert len(d["conference"]) <= 3

    def test_explicit_modules_win(self):
        d = decide("weigh it up", "conference",
                   entities={"modules": ["artemis", "pluto"]})
        assert d["conference"] == ["artemis", "pluto"]

    def test_explicit_modules_as_string(self):
        d = decide("weigh it up", "conference",
                   entities={"modules": "pluto and apollo"})
        assert d["conference"] == ["pluto", "apollo"]

    def test_unknown_and_inactive_modules_are_dropped(self):
        d = decide("my budget and sleep", "conference", active=["core", "pluto", "apollo"],
                   entities={"modules": ["pluto", "ghost", "ares", "apollo", "core"]})
        assert d["conference"] == ["pluto", "apollo"]

    def test_duplicates_in_explicit_list_collapse(self):
        d = decide("x", "conference",
                   entities={"modules": ["pluto", "pluto", "apollo"]})
        assert d["conference"] == ["pluto", "apollo"]

    def test_fewer_than_two_modules_does_not_convene(self):
        d = decide("my budget", "conference", active=["core", "pluto"])
        assert d["conference"] is None and d["intent"] == "conference"
        assert any("nothing to convene" in s for s in d["checked"])

    def test_no_topic_words_does_not_convene(self):
        d = decide("hmm what do you think", "conference")
        assert d["conference"] is None

    def test_low_confidence_conference_asks_first(self):
        d = decide("budget and sleep", "conference", conf=0.2)
        assert d["intent"] == "clarify_intent" and d["conference"] is None

    def test_other_intents_never_convene(self):
        assert decide("my budget and my sleep", "pluto_log_expense")["conference"] is None
        assert decide("my budget and my sleep", "chat")["conference"] is None

    def test_conference_needs_core_active(self):
        d = decide("my budget and sleep", "conference", active=["pluto", "apollo"])
        assert d["conference"] is None

    def test_conference_trail_names_the_seats(self):
        d = decide("my budget and sleep", "conference")
        assert any(s.startswith("conference: topic touches") for s in d["checked"])

    def test_seat_order_is_stable(self):
        a = decide("my budget and sleep", "conference")["conference"]
        b = decide("my budget and sleep", "conference")["conference"]
        assert a == b


# ===========================================================================
# #158 — Conference (gathering and merging)
# ===========================================================================

def _caller(table):
    """A fake orchestrator.call_module backed by {module: result-or-exception}."""
    calls = []

    def call(name, intent, entities, context):
        calls.append((name, intent, entities))
        out = table.get(name)
        if isinstance(out, Exception):
            raise out
        return out
    call.calls = calls
    return call


def _view(text, data=True, conf=0.9):
    return {"response": text, "data": {"k": 1} if data else {}, "confidence": conf}


class _Consensus:
    def __init__(self, found=True, boom=False):
        self.found, self.boom = found, boom

    def rest_signals(self): return ["r"]
    def push_signals(self): return ["p"]

    def evaluate(self, signals):
        if self.boom:
            raise RuntimeError("x")
        return ["t"] if self.found else []

    def describe(self, found): return "Your sleep and your streak pull in opposite directions."


class TestConference:
    def test_lens_table_matches_real_modules(self):
        # Read the engine source rather than importing it, so this runs
        # without each module's heavy dependencies (pydantic, psycopg...).
        # An intent that is renamed or removed no longer appears quoted in
        # its module's engine file, which is what this catches.
        root = os.path.join(os.path.dirname(__file__), "..", "modules")
        for mod, (intent, label) in LENSES.items():
            with open(os.path.join(root, mod, "engine.py"), encoding="utf-8") as fh:
                assert f'"{intent}"' in fh.read(), f"{mod} no longer offers {intent!r}"
            assert label

    def test_no_lens_writes_data(self):
        # A conference must be read-only. These names are the writing ones
        # the lenses must never drift towards.
        assert "decision_support" not in {i for i, _ in LENSES.values()}
        for intent, _ in LENSES.values():
            assert not intent.startswith(("log_", "set_", "add_", "delete_", "create_"))

    def test_merges_views_side_by_side(self):
        call = _caller({"pluto": _view("You have 12,000 spare."),
                        "apollo": _view("You average 5.5h sleep.")})
        out = Conference(call).convene("can I afford this", ["pluto", "apollo"])
        assert out["data"]["convened"] is True and out["confidence"] >= 0.8
        assert "- Money (Pluto): You have 12,000 spare." in out["response"]
        assert "- Health (Apollo): You average 5.5h sleep." in out["response"]

    def test_asks_each_module_its_own_lens(self):
        call = _caller({"pluto": _view("a"), "artemis": _view("b")})
        Conference(call).convene("q", ["pluto", "artemis"])
        assert [(n, i) for n, i, _ in call.calls] == [
            ("pluto", "financial_health"), ("artemis", "productivity_summary")]
        assert all(e == {"raw_query": "q"} for _, _, e in call.calls)

    def test_a_module_that_raises_just_loses_its_seat(self):
        call = _caller({"pluto": RuntimeError("db down"), "apollo": _view("a"),
                        "artemis": _view("b")})
        out = Conference(call).convene("q", ["pluto", "apollo", "artemis"])
        assert out["data"]["convened"] is True
        assert "Money (Pluto)" in out["response"] and "didn't answer" in out["response"]

    def test_module_returning_none_is_silent(self):
        call = _caller({"pluto": None, "apollo": _view("a"), "artemis": _view("b")})
        out = Conference(call).convene("q", ["pluto", "apollo", "artemis"])
        assert "Nothing to add from" in out["response"]

    def test_reply_with_no_data_is_not_a_voice(self):
        call = _caller({"pluto": _view("Nothing tracked yet.", data=False, conf=0.5),
                        "apollo": _view("a"), "artemis": _view("b")})
        out = Conference(call).convene("q", ["pluto", "apollo", "artemis"])
        assert "nothing recorded yet" in out["response"]
        assert "- Money (Pluto)" not in out["response"]

    def test_too_few_voices_says_so_instead_of_inventing(self):
        call = _caller({"pluto": _view("a"), "apollo": None})
        out = Conference(call).convene("q", ["pluto", "apollo"])
        assert out["data"]["convened"] is False and out["confidence"] <= 0.5
        assert "enough recorded" in out["response"]

    def test_unknown_seat_is_reported_not_crashed(self):
        call = _caller({"pluto": _view("a"), "apollo": _view("b")})
        out = Conference(call).convene("q", ["pluto", "apollo", "dionysus"])
        assert out["data"]["convened"] is True

    def test_summary_uses_llm_when_available(self):
        seen = {}

        def synth(prompt):
            seen["p"] = prompt
            return "  Pluto and Apollo agree on caution.  "
        call = _caller({"pluto": _view("p"), "apollo": _view("a")})
        out = Conference(call, synthesize=synth).convene("q?", ["pluto", "apollo"])
        assert out["response"].startswith("Pluto and Apollo agree on caution.")
        assert "[Money (Pluto)] p" in seen["p"] and "invent nothing" in seen["p"]

    @pytest.mark.parametrize("bad", [None, "", "   "])
    def test_summary_falls_back_when_llm_gives_nothing(self, bad):
        call = _caller({"pluto": _view("p"), "apollo": _view("a")})
        out = Conference(call, synthesize=lambda p: bad).convene("q", ["pluto", "apollo"])
        assert "side by side" in out["response"]

    def test_summary_falls_back_when_llm_raises(self):
        def boom(p): raise TimeoutError()
        call = _caller({"pluto": _view("p"), "apollo": _view("a")})
        out = Conference(call, synthesize=boom).convene("q", ["pluto", "apollo"])
        assert "side by side" in out["response"] and out["data"]["convened"]

    def test_tension_is_appended_when_both_sides_are_seated(self):
        call = _caller({"apollo": _view("a"), "artemis": _view("b")})
        out = Conference(call, consensus=_Consensus()).convene("q", ["apollo", "artemis"])
        assert "pull in opposite directions" in out["response"]
        assert out["data"]["tension"]

    def test_no_tension_without_both_sides(self):
        call = _caller({"pluto": _view("a"), "apollo": _view("b")})
        out = Conference(call, consensus=_Consensus()).convene("q", ["pluto", "apollo"])
        assert out["data"]["tension"] is None

    def test_no_tension_when_signals_agree(self):
        call = _caller({"apollo": _view("a"), "artemis": _view("b")})
        out = Conference(call, consensus=_Consensus(found=False)).convene("q", ["apollo", "artemis"])
        assert out["data"]["tension"] is None

    def test_consensus_failure_is_swallowed(self):
        call = _caller({"apollo": _view("a"), "artemis": _view("b")})
        out = Conference(call, consensus=_Consensus(boom=True)).convene("q", ["apollo", "artemis"])
        assert out["data"]["convened"] is True and out["data"]["tension"] is None

    def test_long_views_are_trimmed(self):
        long = "A fact. " * 200
        out = Conference(_caller({"pluto": _view(long), "apollo": _view("b")})).convene(
            "q", ["pluto", "apollo"])
        line = next(l for l in out["response"].splitlines() if l.startswith("- Money"))
        assert len(line) < MAX_PERSPECTIVE_CHARS + 60 and line.endswith("…")

    def test_trim_short_text_untouched(self):
        assert _trim("  hello   world ") == "hello world"

    def test_convene_never_raises(self):
        def call(*a): raise KeyboardInterrupt if False else ValueError("x")
        out = Conference(call).convene("q", ["pluto", "apollo"])
        assert out["data"]["convened"] is False

    def test_convene_survives_garbage_seats(self):
        out = Conference(_caller({})).convene("q", None)
        assert out["data"]["convened"] is False


class _CoreStub(BaseModule):
    name = "core"

    def can_handle(self, intent):
        return intent in ("conference", "what_if")

    def handle(self, intent, entities, context):
        return {"response": "core couldn't", "data": {}, "confidence": 0.3}


class _Seat(BaseModule):
    def __init__(self, name, intent, text, raises=False):
        self.name, self._i, self._t, self._r = name, intent, text, raises

    def can_handle(self, intent): return intent == self._i

    def handle(self, intent, entities, context):
        if self._r:
            raise RuntimeError("boom")
        return {"response": self._t, "data": {"k": 1}, "confidence": 0.9}


def _orch(*mods):
    o = HestiaOrchestrator()
    o.register_hecate(HecateEngine())
    o.register(_CoreStub())
    for m in mods:
        o.register(m)
    return o


class TestOrchestratorWiring:
    def test_call_module_reaches_a_module(self):
        o = _orch(_Seat("pluto", "financial_health", "ok"))
        assert o.call_module("pluto", "financial_health", {}, {})["response"] == "ok"

    def test_call_module_unknown_module_or_intent(self):
        o = _orch(_Seat("pluto", "financial_health", "ok"))
        assert o.call_module("ghost", "x", {}, {}) is None
        assert o.call_module("pluto", "other", {}, {}) is None

    def test_call_module_swallows_errors_and_trips_the_breaker(self):
        o = _orch(_Seat("pluto", "financial_health", "x", raises=True))
        for _ in range(10):
            assert o.call_module("pluto", "financial_health", {}, {}) is None

    def test_end_to_end_conference(self):
        o = _orch(_Seat("pluto", "financial_health", "Money is fine."),
                  _Seat("apollo", "get_health_summary", "Sleep is short."))
        o.attach_conference(Conference(o.call_module))
        out = o.dispatch("should I buy this, my budget and my sleep",
                         {"intent": "conference", "entities": {}, "confidence": 0.95})
        assert "Money is fine." in out and "Sleep is short." in out
        assert "core couldn't" not in out

    def test_conference_without_the_layer_keeps_core_reply(self):
        o = _orch(_Seat("pluto", "financial_health", "a"), _Seat("apollo", "get_health_summary", "b"))
        out = o.dispatch("my budget and my sleep",
                         {"intent": "conference", "entities": {}, "confidence": 0.95})
        assert out == "core couldn't"

    def test_what_if_end_to_end(self):
        class W:
            def project(self, q, e): return {"response": "projected!", "data": {}, "confidence": 0.9}
        o = _orch()
        o.attach_whatif(W())
        out = o.dispatch("what if I cancel netflix",
                         {"intent": "what_if", "entities": {}, "confidence": 0.95})
        assert out == "projected!"

    def test_what_if_layer_failure_keeps_core_reply(self):
        class W:
            def project(self, q, e): raise RuntimeError("x")
        o = _orch()
        o.attach_whatif(W())
        out = o.dispatch("what if I cancel netflix",
                         {"intent": "what_if", "entities": {}, "confidence": 0.95})
        assert out == "core couldn't"

    def test_attach_none_turns_it_off(self):
        o = _orch()
        o.attach_whatif(None); o.attach_conference(None)
        out = o.dispatch("what if I cancel netflix",
                         {"intent": "what_if", "entities": {}, "confidence": 0.95})
        assert out == "core couldn't"


# ===========================================================================
# #160 — what-if
# ===========================================================================

class _Habit:
    def __init__(self, streak=12, best=20, total=80, done=24, possible=30):
        self.streak, self.best_streak, self.total_completions = streak, best, total
        self._d = (done, possible)

    def window_stats(self, start, end, today): return self._d


class _Goal:
    def __init__(self, progress=0.4, status="active", due=None):
        self.progress, self.status, self._due = progress, status, due

    def days_until_due(self, today):
        return None if self._due is None else (self._due - today).days


class _Tracker:
    def __init__(self, habits=None, goals=None):
        self._h, self._g = habits or {}, goals or {}
        self.default_grace_days = 0

    def get_habits(self): return self._h
    def get_goals(self): return self._g


class _Artemis:
    def __init__(self, **kw): self.tracker = _Tracker(**kw)


class _Db:
    def __init__(self, avg=7.0, workouts=3):
        self._a, self._w = avg, workouts

    def avg_sleep(self, days): return self._a
    def workout_count(self, days): return self._w


class _Apollo:
    def __init__(self, **kw): self.db = _Db(**kw)


class _Pluto:
    def __init__(self, rows=None, sts=None, boom=False):
        self.rows, self.sts, self.boom = rows or [], sts, boom

    def handle(self, intent, entities, context):
        if self.boom:
            raise RuntimeError("x")
        if intent == "recurring_expenses":
            return {"response": "", "data": {"recurring": self.rows}, "confidence": 0.9}
        if intent == "safe_to_spend":
            return {"response": "", "data": self.sts or {}, "confidence": 0.9}
        return None


NETFLIX = {"key": "netflix", "description": "Netflix", "typical_amount": 649.0,
           "cadence": "monthly", "monthly_cost": 649.0, "next_expected": "2026-10-20",
           "active": True}
GYM = {"key": "gym membership", "description": "Gym membership", "typical_amount": 1500.0,
       "cadence": "monthly", "monthly_cost": 1500.0, "next_expected": "2026-11-02",
       "active": True}
STS = {"budget": 30000.0, "remaining": 10000.0, "per_day": 500.0, "days_left": 26}


def wi(**kw):
    return WhatIfEngine(today=lambda: TODAY, **kw)


class TestParsers:
    @pytest.mark.parametrize("text,expected", [
        ("cut 500 a month", 500), ("save 1,200 per week", 1200 * 52 / 12),
        ("spend 50 less every day", 50 * 365 / 12), ("6000 a year", 500),
        ("cut 2k", 2000), ("drop 700", 700),
    ])
    def test_amounts(self, text, expected):
        assert parse_amount_per_month(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["cancel 3 subscriptions", "no numbers", "", None])
    def test_not_amounts(self, text):
        assert parse_amount_per_month(text) is None

    @pytest.mark.parametrize("text,expected", [
        ("what if I slept an hour less", {"delta": -1.0}),
        ("what if I sleep 2 hours more", {"delta": 2.0}),
        ("what if I only slept 5 hours", {"target": 5.0}),
        ("what if I slept half an hour less", {"delta": -0.5}),
        ("what if I sleep 1.5 hours extra", {"delta": 1.5}),
    ])
    def test_sleep_changes(self, text, expected):
        assert parse_sleep_change(text) == expected

    @pytest.mark.parametrize("text", ["what if I eat less", "I walked 3 hours", "sleep", ""])
    def test_not_sleep_changes(self, text):
        assert parse_sleep_change(text) is None

    def test_threshold_matches_apollos(self):
        import modules.apollo.engine as apollo_engine
        assert LOW_SLEEP_HOURS == apollo_engine._LOW_SLEEP_THRESHOLD


class TestWhatIfExpense:
    def test_cancelling_a_detected_subscription(self):
        out = wi(pluto=_Pluto([NETFLIX], STS)).project("what if I cancel netflix")
        d = out["data"]
        assert d["kind"] == "cut_expense" and d["matched_recurring"] is True
        assert d["monthly_saving"] == 649 and d["saving_12m"] == 649 * 12
        assert "Netflix" in out["response"] and "7,788" in out["response"]

    def test_safe_to_spend_moves_when_still_due_this_month(self):
        d = wi(pluto=_Pluto([NETFLIX], STS)).project("what if I cancel netflix")["data"]
        assert d["safe_per_day_before"] == 500
        assert d["safe_per_day_after"] == pytest.approx((10000 + 649) / 26, abs=0.01)

    def test_safe_to_spend_unchanged_when_due_next_month(self):
        out = wi(pluto=_Pluto([GYM], STS)).project("what if I cancel my gym membership")
        assert "safe_per_day_after" not in out["data"]
        assert "wouldn't move" in out["response"]

    def test_budget_share_is_reported(self):
        d = wi(pluto=_Pluto([NETFLIX], STS)).project("what if I cancel netflix")["data"]
        assert d["budget_share"] == pytest.approx(649 / 30000, abs=1e-4)

    def test_stated_amount_without_a_match(self):
        out = wi(pluto=_Pluto([NETFLIX], STS)).project("what if I cut 3000 a month on eating out")
        assert out["data"]["monthly_saving"] == 3000 and out["data"]["matched_recurring"] is False

    def test_stated_amount_via_entities(self):
        out = wi(pluto=_Pluto([], None)).project("what if I cut my spending", {"amount": "1,000"})
        assert out["data"]["monthly_saving"] == 1000

    def test_weekly_amount_is_normalised(self):
        out = wi(pluto=_Pluto([], None)).project("what if I cut 100 per week")
        assert out["data"]["monthly_saving"] == pytest.approx(100 * 52 / 12, abs=0.01)

    def test_inactive_recurring_charge_is_not_matched(self):
        dead = dict(NETFLIX, active=False)
        out = wi(pluto=_Pluto([dead], STS)).project("what if I cancel netflix")
        assert out["data"]["projected"] is False

    def test_unknown_item_with_no_amount_is_not_invented(self):
        out = wi(pluto=_Pluto([NETFLIX], STS)).project("what if I cancel disney")
        assert out["data"]["projected"] is False and "can't project" in out["response"]

    def test_pluto_failure_is_survivable(self):
        out = wi(pluto=_Pluto(boom=True)).project("what if I cut 500 a month")
        assert out["data"]["monthly_saving"] == 500

    def test_no_pluto_at_all(self):
        assert wi().project("what if I cancel netflix")["data"]["projected"] is False

    def test_needs_a_cutting_word(self):
        out = wi(pluto=_Pluto([NETFLIX], STS)).project("what is netflix costing me")
        assert out["data"]["projected"] is False

    def test_zero_budget_does_not_divide(self):
        sts = dict(STS, budget=0)
        out = wi(pluto=_Pluto([NETFLIX], sts)).project("what if I cancel netflix")
        assert "budget_share" not in out["data"]


class TestWhatIfHabit:
    def _eng(self, **hk):
        return wi(artemis=_Artemis(habits={"meditation": _Habit(**hk), "running": _Habit()}),
                  apollo=_Apollo())

    def test_stopping_a_habit_reports_streak_and_rate(self):
        out = self._eng().project("what if I stop meditating")
        d = out["data"]
        assert d["habit"] == "meditation" and d["streak"] == 12 and d["best_streak"] == 20
        assert d["last_30_done"] == 24 and d["completion_rate"] == pytest.approx(0.8)
        assert "12-day" in out["response"] and "best is 20" in out["response"]
        assert "not a prediction" in out["response"]

    def test_best_streak_wording(self):
        out = self._eng(streak=20, best=20).project("what if I stop meditating")
        assert "your best so far" in out["response"]

    def test_no_live_streak(self):
        out = self._eng(streak=0, best=5).project("what if I stop meditating")
        assert "no live streak" in out["response"]

    def test_fitness_habit_pulls_in_apollo_workouts(self):
        out = self._eng().project("what if I stop running")
        assert out["data"]["linked"] and "3 workout" in out["data"]["linked"][0]

    def test_related_goal_is_mentioned(self):
        eng = wi(artemis=_Artemis(habits={"meditation": _Habit()},
                                  goals={"meditation retreat": _Goal(0.5)}))
        out = eng.project("what if I stop meditating")
        assert any("50% done" in l for l in out["data"]["linked"])

    def test_completed_goal_is_not_mentioned(self):
        eng = wi(artemis=_Artemis(habits={"meditation": _Habit()},
                                  goals={"meditation retreat": _Goal(1.0, status="completed")}))
        assert self._linked(eng) == []

    @staticmethod
    def _linked(eng):
        return eng.project("what if I stop meditating")["data"]["linked"]

    def test_unknown_habit_falls_through_to_unsupported(self):
        out = self._eng().project("what if I stop knitting")
        assert out["data"]["projected"] is False

    def test_requires_a_quitting_word(self):
        out = self._eng().project("how is my meditation going")
        assert out["data"]["projected"] is False

    def test_explicit_habit_entity(self):
        out = self._eng().project("what if", {"habit": "meditation"})
        assert out["data"]["habit"] == "meditation"

    def test_no_history_still_answers(self):
        eng = wi(artemis=_Artemis(habits={"meditation": _Habit(done=0, possible=0, total=4)}))
        out = eng.project("what if I stop meditating")
        assert out["data"]["completion_rate"] is None and "4 time(s)" in out["response"]

    def test_longest_habit_name_wins(self):
        eng = wi(artemis=_Artemis(habits={"run": _Habit(streak=1), "run club": _Habit(streak=9)}))
        assert eng.project("what if I stop run club")["data"]["habit"] == "run club"


class TestWhatIfSleep:
    def test_dropping_below_the_line(self):
        out = wi(apollo=_Apollo(avg=6.5)).project("what if I slept an hour less")
        d = out["data"]
        assert d["current_avg"] == 6.5 and d["new_avg"] == 5.5 and d["crosses_low_sleep_line"]
        assert "burnout" in out["response"] and "7.0h less over a week" in out["response"]

    def test_staying_above_the_line(self):
        out = wi(apollo=_Apollo(avg=8.0)).project("what if I slept an hour less")
        assert "stay above" in out["response"] and not out["data"]["crosses_low_sleep_line"]

    def test_already_below(self):
        out = wi(apollo=_Apollo(avg=5.0)).project("what if I slept an hour less")
        assert "already" in out["response"]

    def test_recovering_above_the_line(self):
        out = wi(apollo=_Apollo(avg=5.0)).project("what if I slept 2 hours more")
        assert "back above" in out["response"]

    def test_target_hours(self):
        d = wi(apollo=_Apollo(avg=7.0)).project("what if I only slept 5 hours")["data"]
        assert d["new_avg"] == 5.0 and d["weekly_change_hours"] == -14.0

    def test_no_change(self):
        out = wi(apollo=_Apollo(avg=7.0)).project("what if I only slept 7 hours")
        assert "about what you already get" in out["response"]

    def test_no_sleep_data(self):
        out = wi(apollo=_Apollo(avg=None)).project("what if I slept an hour less")
        assert out["data"]["projected"] is False and "Log a few nights" in out["response"]

    def test_never_negative(self):
        d = wi(apollo=_Apollo(avg=0.5)).project("what if I slept 3 hours less")["data"]
        assert d["new_avg"] == 0.0


class TestWhatIfGeneral:
    def test_unsupported_lists_what_is_supported(self):
        out = wi().project("what if I move to Berlin")
        assert out["data"]["projected"] is False and "netflix" in out["response"].lower()

    @pytest.mark.parametrize("q", ["", None])
    def test_empty_query(self, q):
        assert wi().project(q)["data"]["projected"] is False

    def test_never_raises(self):
        class Bad:
            tracker = property(lambda s: (_ for _ in ()).throw(RuntimeError("x")))
        out = wi(artemis=Bad()).project("what if I stop meditating")
        assert out["response"]

    def test_read_only(self):
        pluto = _Pluto([NETFLIX], STS)
        seen = []
        orig = pluto.handle
        pluto.handle = lambda i, e, c: (seen.append(i), orig(i, e, c))[1]
        wi(pluto=pluto).project("what if I cancel netflix")
        assert set(seen) <= {"recurring_expenses", "safe_to_spend"}


# ===========================================================================
# #163 — weekly critical path
# ===========================================================================

class _Hermes:
    def __init__(self, events=None, connected=True, boom=False):
        self.events, self.connected, self.boom = events or [], connected, boom

    def handle(self, intent, entities, context):
        if self.boom:
            raise RuntimeError("x")
        return {"response": "", "data": {"events": self.events} if self.connected else {},
                "confidence": 0.9}


@pytest.fixture
def agenda():
    pytest.importorskip("yaml")
    try:
        import modules.chronos.agenda as a
    except ImportError as e:
        pytest.skip(str(e))
    return a


NOW = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)


def focus(agenda, **kw):
    return agenda.build_week_focus(None, timezone.utc, NOW, **kw)


def ev(day, title, start="10:00", end="11:00"):
    d = TODAY + timedelta(days=day)
    return {"title": title, "start": f"{d}T{start}:00+00:00", "end": f"{d}T{end}:00+00:00"}


def allday(day, title):
    return {"title": title, "start": str(TODAY + timedelta(days=day)), "end": str(TODAY + timedelta(days=day + 1))}


class TestWeekFocus:
    def test_nothing_pressing(self, agenda):
        f = focus(agenda)
        assert f.items == [] and "Nothing looks pressing" in agenda.format_week_focus(f)

    def test_overdue_goal_outranks_a_due_goal(self, agenda):
        art = _Artemis(goals={"tax filing": _Goal(0.1, due=TODAY - timedelta(days=2)),
                              "report": _Goal(0.2, due=TODAY + timedelta(days=2))})
        f = focus(agenda, artemis=art)
        assert [i.text for i in f.items] == ["tax filing", "report"]
        assert "overdue by 2 days" in f.items[0].why

    def test_less_finished_goal_ranks_higher(self, agenda):
        art = _Artemis(goals={"a": _Goal(0.9, due=TODAY + timedelta(days=3)),
                              "b": _Goal(0.1, due=TODAY + timedelta(days=3))})
        assert [i.text for i in focus(agenda, artemis=art).items] == ["b", "a"]

    def test_goal_outside_the_window_is_ignored(self, agenda):
        art = _Artemis(goals={"later": _Goal(0.0, due=TODAY + timedelta(days=30)),
                              "undated": _Goal(0.0)})
        assert focus(agenda, artemis=art).items == []

    def test_inactive_goal_is_ignored(self, agenda):
        art = _Artemis(goals={"done": _Goal(0.1, status="completed", due=TODAY + timedelta(days=1))})
        assert focus(agenda, artemis=art).items == []

    def test_busy_calendar_raises_a_goals_urgency(self, agenda):
        art = _Artemis(goals={"report": _Goal(0.3, due=TODAY + timedelta(days=3))})
        free = focus(agenda, artemis=art, hermes=_Hermes([ev(6, "dentist")])).items[0]
        busy = focus(agenda, artemis=art, hermes=_Hermes(
            [allday(0, "offsite"), allday(1, "travel"), allday(2, "wedding")])).items[0]
        assert busy.open_days < free.open_days and busy.score > free.score
        assert "open day" in busy.why

    def test_long_meetings_make_a_day_busy(self, agenda):
        art = _Artemis(goals={"report": _Goal(0.3, due=TODAY + timedelta(days=1))})
        h = _Hermes([ev(0, "workshop", "09:00", "14:00"), ev(1, "board", "09:00", "15:00")])
        assert focus(agenda, artemis=art, hermes=h).items[0].open_days == 0

    def test_calendar_deadline_is_picked_up(self, agenda):
        f = focus(agenda, hermes=_Hermes([ev(2, "GATE exam"), ev(2, "lunch")]))
        assert [i.text for i in f.items] == ["GATE exam"] and f.items[0].source == "calendar"

    def test_deadline_outside_window_ignored(self, agenda):
        assert focus(agenda, hermes=_Hermes([ev(10, "exam")])).items == []

    def test_streak_ending_today_is_surfaced(self, agenda):
        class H(_Habit):
            def streak_ends_if_skipped_today(self, today, grace): return True
        f = focus(agenda, artemis=_Artemis(habits={"meditation": H(streak=10)}))
        assert f.items and f.items[0].source == "habit" and "10-day" in f.items[0].why

    def test_short_streaks_are_not_surfaced(self, agenda):
        class H(_Habit):
            def streak_ends_if_skipped_today(self, today, grace): return True
        assert focus(agenda, artemis=_Artemis(habits={"x": H(streak=2)})).items == []

    def test_limit_and_considered_count(self, agenda):
        goals = {f"g{i}": _Goal(0.1 * i / 10, due=TODAY + timedelta(days=1 + i % 5)) for i in range(9)}
        f = focus(agenda, artemis=_Artemis(goals=goals), limit=3)
        assert len(f.items) == 3 and f.considered == 9
        assert "out of 9" in agenda.format_week_focus(f)

    def test_ranking_is_deterministic(self, agenda):
        goals = {n: _Goal(0.5, due=TODAY + timedelta(days=3)) for n in ("b", "a", "c")}
        a = [i.text for i in focus(agenda, artemis=_Artemis(goals=goals)).items]
        b = [i.text for i in focus(agenda, artemis=_Artemis(goals=dict(reversed(list(goals.items()))))).items]
        assert a == b == ["a", "b", "c"]

    def test_duplicates_collapse(self, agenda):
        h = _Hermes([ev(1, "exam"), ev(1, "exam", "14:00", "15:00")])
        assert len(focus(agenda, hermes=h).items) == 1

    def test_disconnected_calendar_is_noted_not_fatal(self, agenda):
        art = _Artemis(goals={"report": _Goal(0.2, due=TODAY + timedelta(days=2))})
        f = focus(agenda, artemis=art, hermes=_Hermes(connected=False))
        assert f.items and any("calendar" in n.lower() for n in f.notes)

    def test_failing_calendar_is_noted_not_fatal(self, agenda):
        art = _Artemis(goals={"report": _Goal(0.2, due=TODAY + timedelta(days=2))})
        f = focus(agenda, artemis=art, hermes=_Hermes(boom=True))
        assert f.items and f.notes

    def test_window_is_clamped(self, agenda):
        assert focus(agenda, days=99).days == 14 and focus(agenda, days=0).days == 1

    def test_single_item_wording(self, agenda):
        art = _Artemis(goals={"report": _Goal(0.2, due=TODAY + timedelta(days=2))})
        assert "is the one thing" in agenda.format_week_focus(focus(agenda, artemis=art))

    def test_to_dict_is_json_friendly(self, agenda):
        import json
        art = _Artemis(goals={"report": _Goal(0.2, due=TODAY + timedelta(days=2))})
        json.dumps([i.to_dict() for i in focus(agenda, artemis=art).items])

    def test_chronos_intent_is_declared_and_registered(self):
        from modules.hecate.intent_registry import module_for_intent
        assert module_for_intent("weekly_focus") == "chronos"
        for name in ("audit_routing", "conference", "what_if"):
            assert module_for_intent(name) == "core"
