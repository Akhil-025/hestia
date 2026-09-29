# tests/test_session_ttl.py
"""
Tests for OrchestratorContext.maybe_expire_session and its wiring into
HestiaOrchestrator.dispatch (backlog #12).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.base import BaseModule
from modules.hecate import HecateEngine
from modules.hestia.orchestrator import HestiaOrchestrator, OrchestratorContext


class _EchoModule(BaseModule):
    name = "core"

    def can_handle(self, intent):
        return True

    def handle(self, intent, entities, context):
        return {"response": "ok", "data": {}, "confidence": 0.9}


def make_orchestrator(ttl):
    orch = HestiaOrchestrator(session_ttl_seconds=ttl)
    orch.register_hecate(HecateEngine())
    orch.register(_EchoModule())
    return orch


# ---------------------------------------------------------------------------
# OrchestratorContext.maybe_expire_session directly
# ---------------------------------------------------------------------------

def test_first_call_never_expires():
    ctx = OrchestratorContext()
    ctx.entities = {"topic": "sleep"}
    assert ctx.maybe_expire_session(now=1000.0, ttl_seconds=60) is False
    assert ctx.entities == {"topic": "sleep"}


def test_within_ttl_does_not_expire():
    ctx = OrchestratorContext()
    ctx.maybe_expire_session(now=1000.0, ttl_seconds=60)
    ctx.entities = {"topic": "sleep"}
    assert ctx.maybe_expire_session(now=1030.0, ttl_seconds=60) is False
    assert ctx.entities == {"topic": "sleep"}


def test_beyond_ttl_clears_conversational_state():
    ctx = OrchestratorContext()
    ctx.maybe_expire_session(now=1000.0, ttl_seconds=60)
    ctx.entities = {"topic": "sleep"}
    ctx.recent_intents = ["apollo_track_sleep"]
    ctx.time_context = {"last_query_time": "10:00"}
    ctx.memory_context = {"facts": ["likes coffee"]}
    expired = ctx.maybe_expire_session(now=2000.0, ttl_seconds=60)
    assert expired is True
    assert ctx.entities == {}
    assert ctx.recent_intents == []
    assert ctx.time_context == {}
    assert ctx.memory_context == {}


def test_expiry_never_touches_active_modules():
    ctx = OrchestratorContext()
    ctx.active_modules = ["core", "apollo"]
    ctx.maybe_expire_session(now=1000.0, ttl_seconds=60)
    ctx.maybe_expire_session(now=2000.0, ttl_seconds=60)
    # active_modules is process-lifetime registration state, not
    # conversation state — a session gap must never deregister modules.
    assert ctx.active_modules == ["core", "apollo"]


def test_zero_ttl_disables_expiry():
    ctx = OrchestratorContext()
    ctx.maybe_expire_session(now=1000.0, ttl_seconds=0)
    ctx.entities = {"topic": "sleep"}
    assert ctx.maybe_expire_session(now=999999.0, ttl_seconds=0) is False
    assert ctx.entities == {"topic": "sleep"}


def test_exactly_at_the_ttl_boundary_does_not_expire():
    ctx = OrchestratorContext()
    ctx.maybe_expire_session(now=1000.0, ttl_seconds=60)
    ctx.entities = {"topic": "sleep"}
    assert ctx.maybe_expire_session(now=1060.0, ttl_seconds=60) is False
    assert ctx.entities == {"topic": "sleep"}


def test_last_active_advances_on_every_call():
    ctx = OrchestratorContext()
    ctx.maybe_expire_session(now=1000.0, ttl_seconds=60)
    assert ctx.last_active == 1000.0
    ctx.maybe_expire_session(now=1010.0, ttl_seconds=60)
    assert ctx.last_active == 1010.0


# ---------------------------------------------------------------------------
# Wired into HestiaOrchestrator.dispatch
# ---------------------------------------------------------------------------

def test_dispatch_default_ttl_does_not_expire_a_normal_conversation():
    orch = make_orchestrator(ttl=1800)
    orch.dispatch("hello", {"intent": "chat", "entities": {}, "confidence": 0.9})
    orch._ctx.entities = {"topic": "sleep"}
    orch.dispatch("more", {"intent": "chat", "entities": {}, "confidence": 0.9})
    assert orch._ctx.entities == {"topic": "sleep"}


def test_dispatch_expires_context_after_a_real_gap(monkeypatch):
    import modules.hestia.orchestrator as orch_module

    orch = make_orchestrator(ttl=60)
    fake_now = [1000.0]
    monkeypatch.setattr(orch_module.time, "time", lambda: fake_now[0])

    orch.dispatch("hello", {"intent": "chat", "entities": {}, "confidence": 0.9})
    orch._ctx.entities = {"topic": "sleep"}

    fake_now[0] = 2000.0  # 1000s gap, well past the 60s ttl
    orch.dispatch("what was I just doing", {"intent": "chat", "entities": {}, "confidence": 0.9})
    assert orch._ctx.entities == {}


def test_dispatch_with_ttl_disabled_never_expires(monkeypatch):
    import modules.hestia.orchestrator as orch_module

    orch = make_orchestrator(ttl=0)
    fake_now = [1000.0]
    monkeypatch.setattr(orch_module.time, "time", lambda: fake_now[0])

    orch.dispatch("hello", {"intent": "chat", "entities": {}, "confidence": 0.9})
    orch._ctx.entities = {"topic": "sleep"}
    fake_now[0] = 99999.0
    orch.dispatch("later", {"intent": "chat", "entities": {}, "confidence": 0.9})
    assert orch._ctx.entities == {"topic": "sleep"}
