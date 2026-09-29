# tests/test_orchestrator_resilience.py
"""
Tests for the circuit-breaker integration in
modules/hestia/orchestrator.py._dispatch_primary (backlog #7), and for
Hecate's confidence-weighted clarification tier (backlog #2).

These sit in their own file rather than tests/test_hestia.py because they
need a breaker's threshold to actually trip, which means several
dispatch() calls per test — a different shape from the existing
single-call orchestrator tests there.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.base import BaseModule
from modules.hecate import HecateEngine
from modules.hecate.intent_registry import INTENT_MODULE_MAP
from modules.hestia.orchestrator import HestiaOrchestrator


class _FlakyModule(BaseModule):
    """A module whose handle() can be told to raise on demand."""

    name = "pluto"

    def __init__(self):
        self.calls = 0
        self.should_raise = True

    def can_handle(self, intent):
        return True

    def handle(self, intent, entities, context):
        self.calls += 1
        if self.should_raise:
            raise RuntimeError("postgres pool exhausted")
        return {"response": "ok", "data": {}, "confidence": 0.9}


def make_orchestrator(module):
    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(module)
    return orch


def dispatch_chat(orch, module_name="pluto"):
    # A registered, high-confidence intent so Hecate's Tier 1 routes
    # straight to the module under test.
    intent = next(k for k, v in INTENT_MODULE_MAP.items() if v == module_name)
    return orch.dispatch("x", {"intent": intent, "entities": {}, "confidence": 0.95})


# ---------------------------------------------------------------------------
# Circuit breaker integration (#7)
# ---------------------------------------------------------------------------

def test_repeated_failures_eventually_stop_calling_the_module():
    module = _FlakyModule()
    orch = make_orchestrator(module)

    # Threshold is 3 (core/circuit_breaker.py default); the third failing
    # dispatch trips the breaker.
    for _ in range(3):
        dispatch_chat(orch)
    assert module.calls == 3

    # A fourth query routes to the same module but must be short-circuited
    # — the breaker, not the module, produces this response.
    response = dispatch_chat(orch)
    assert module.calls == 3          # handle() was NOT called again
    assert "pluto" in response
    assert "break" in response.lower() or "again" in response.lower()


def test_a_single_failure_does_not_trip_the_breaker():
    # Regression guard: this must not regress the existing
    # test_orchestrator_falls_back_gracefully_when_module_raises behaviour
    # — one bad call is normal operation, not a reason to stop calling a
    # module for a minute.
    module = _FlakyModule()
    orch = make_orchestrator(module)
    dispatch_chat(orch)
    dispatch_chat(orch)
    assert module.calls == 2          # both calls actually reached handle()


def test_success_resets_the_breaker_after_partial_failures():
    module = _FlakyModule()
    orch = make_orchestrator(module)
    dispatch_chat(orch)
    dispatch_chat(orch)
    module.should_raise = False
    dispatch_chat(orch)               # succeeds, resets consecutive count
    module.should_raise = True
    dispatch_chat(orch)
    dispatch_chat(orch)
    # Two more failures after the reset is still below threshold=3.
    assert module.calls == 5


def test_circuit_breaker_status_is_exposed_on_the_orchestrator():
    module = _FlakyModule()
    orch = make_orchestrator(module)
    for _ in range(3):
        dispatch_chat(orch)
    status = orch.circuit_breaker_status
    assert status["pluto"]["state"] == "open"
    assert status["pluto"]["total_failures"] == 3


def test_a_module_that_never_fails_never_appears_open():
    module = _FlakyModule()
    module.should_raise = False
    orch = make_orchestrator(module)
    for _ in range(5):
        dispatch_chat(orch)
    assert orch.circuit_breaker_status.get("pluto", {}).get("state") != "open"


def test_breaker_is_per_module_not_global():
    class _AlwaysOk(BaseModule):
        name = "core"

        def can_handle(self, intent):
            return True

        def handle(self, intent, entities, context):
            return {"response": "fine", "data": {}, "confidence": 0.9}

    flaky = _FlakyModule()
    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(flaky)
    orch.register(_AlwaysOk())

    for _ in range(3):
        dispatch_chat(orch, "pluto")
    assert orch.circuit_breaker_status["pluto"]["state"] == "open"

    # core must be entirely unaffected by pluto's breaker.
    response = orch.dispatch("hi", {"intent": "chat", "entities": {}, "confidence": 0.9})
    assert response == "fine"


def test_notimplementederror_never_counts_toward_the_breaker():
    # A module that deliberately doesn't implement a path (e.g. Hecate's
    # own handle()) must not accumulate toward its breaker — no amount of
    # retrying fixes a call shape the module refuses on purpose.
    class _AlwaysUnimplemented(BaseModule):
        name = "pluto"

        def can_handle(self, intent):
            return True

        def handle(self, intent, entities, context):
            raise NotImplementedError

    orch = make_orchestrator(_AlwaysUnimplemented())
    for _ in range(5):
        dispatch_chat(orch)
    assert orch.circuit_breaker_status.get("pluto", {}).get("state") != "open"


# ---------------------------------------------------------------------------
# Confidence-weighted clarification (#2)
# ---------------------------------------------------------------------------

def make_engine():
    return HecateEngine()


def test_low_confidence_registered_intent_asks_for_clarification():
    h = make_engine()
    nlu = {"intent": "pluto_log_expense", "entities": {}, "confidence": 0.2}
    decision = h.decide("uh spent something at the place", nlu, ["core", "pluto"])
    assert decision["primary"] == "core"
    assert decision["intent"] == "clarify_intent"


def test_high_confidence_registered_intent_still_dispatches_normally():
    h = make_engine()
    nlu = {"intent": "pluto_log_expense", "entities": {}, "confidence": 0.95}
    decision = h.decide("i spent 200 on lunch", nlu, ["core", "pluto"])
    assert decision["primary"] == "pluto"
    assert decision["intent"] != "clarify_intent"


def test_low_confidence_chat_is_not_intercepted():
    # A low-confidence "chat" already has a correct home — Tier 6's force-
    # chat fallback — with nothing concrete to confirm first.
    h = make_engine()
    nlu = {"intent": "chat", "entities": {}, "confidence": 0.2}
    decision = h.decide("mumble mumble", nlu, ["core"])
    assert decision["intent"] != "clarify_intent"


def test_confidence_right_at_the_threshold_is_not_intercepted():
    h = make_engine()
    nlu = {
        "intent": "pluto_log_expense", "entities": {},
        "confidence": h._CLARIFY_CONFIDENCE_THRESHOLD,
    }
    decision = h.decide("i spent 200", nlu, ["core", "pluto"])
    assert decision["intent"] != "clarify_intent"


def test_clarify_intent_end_to_end_through_the_orchestrator():
    from modules.hestia.core_module import CoreModule

    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(CoreModule(memory=None, ollama_cfg={}))
    response = orch.dispatch(
        "uh spent something",
        {"intent": "pluto_log_expense", "entities": {}, "confidence": 0.15},
    )
    assert "?" in response
    assert "spent" not in response.lower()  # no implementation leakage
