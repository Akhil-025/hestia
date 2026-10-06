# tests/test_module_events.py
"""Backlog #10: module-to-module events through the bus, no direct calls."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus
from core.module_events import (
    EventLedger, ModuleEventPort, ModuleEventWiring, make_envelope, topic_matches, valid_pattern,
)
from modules.base import BaseModule
from modules.hestia.orchestrator import HestiaOrchestrator


def _wait(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Publisher(BaseModule):
    name = "pluto"

    def can_handle(self, intent):
        return intent in ("log", "fail")

    def handle(self, intent, entities, context):
        if intent == "fail":
            raise RuntimeError("nope")
        self.events.publish("expense_logged", {"amount": 5})
        return {"response": "logged", "data": {}, "confidence": 1.0}


class Subscriber(BaseModule):
    name = "artemis"
    EVENT_SUBSCRIPTIONS = {"pluto.expense_logged": "on_expense", "intent.*": "on_intent"}

    def __init__(self):
        self.expenses, self.intents = [], []

    def can_handle(self, intent):
        return False

    def handle(self, intent, entities, context):
        return {}

    def on_expense(self, env):
        self.expenses.append(env)

    def on_intent(self, env):
        self.intents.append(env["payload"])


def _orch():
    o = HestiaOrchestrator()
    bus = EventBus()
    o.attach_event_bus(bus)
    return o, bus


def _dispatch(o, name, intent):
    return o._dispatch_primary(primary_name=name, intent=intent, raw_intent=intent, entities={},
                               context={}, raw_query="q", nlu_result={})


# -------------------------------------------------------------- patterns

def test_topic_matching():
    assert topic_matches("pluto.expense_logged", "pluto.expense_logged")
    assert topic_matches("pluto.*", "pluto.anything")
    assert not topic_matches("pluto.*", "plutonium.x")
    assert topic_matches("*", "whatever")
    assert not topic_matches("pluto.a", "pluto.b")


def test_valid_patterns():
    assert valid_pattern("pluto.expense_logged") and valid_pattern("pluto.*") and valid_pattern("*")
    assert not valid_pattern("nodot") and not valid_pattern("Bad.Name") and not valid_pattern("")


# -------------------------------------------------------------- publish/subscribe

def test_one_module_reacts_to_anothers_event_without_a_direct_call():
    o, _ = _orch()
    pub, sub = Publisher(), Subscriber()
    o.register(pub)
    o.register(sub)
    _dispatch(o, "pluto", "log")
    assert _wait(lambda: sub.expenses)
    env = sub.expenses[0]
    assert env["topic"] == "pluto.expense_logged" and env["source"] == "pluto"
    assert env["payload"] == {"amount": 5}
    assert not hasattr(sub, "pluto")             # it got data, never the publisher


def test_publisher_does_not_receive_its_own_events():
    o, _ = _orch()

    class Echo(Publisher):
        EVENT_SUBSCRIPTIONS = {"pluto.*": "on_any"}
        got = []
        def on_any(self, env): self.got.append(env)
    mod = Echo()
    o.register(mod)
    _dispatch(o, "pluto", "log")
    time.sleep(0.2)
    assert [e for e in mod.got if e["topic"].startswith("pluto.")] == []


def test_orchestrator_publishes_intent_handled():
    o, _ = _orch()
    sub = Subscriber()
    o.register(Publisher())
    o.register(sub)
    _dispatch(o, "pluto", "log")
    assert _wait(lambda: sub.intents)
    p = sub.intents[0]
    assert p["module"] == "pluto" and p["intent"] == "log" and p["ok"] is True and p["ms"] >= 0


def test_failed_handler_publishes_ok_false():
    o, _ = _orch()
    sub = Subscriber()
    o.register(Publisher())
    o.register(sub)
    _dispatch(o, "pluto", "fail")
    assert _wait(lambda: sub.intents)
    assert sub.intents[0]["ok"] is False


def test_modules_registered_before_attach_are_wired_too():
    o = HestiaOrchestrator()
    pub, sub = Publisher(), Subscriber()
    o.register(pub)
    o.register(sub)
    o.attach_event_bus(EventBus())
    _dispatch(o, "pluto", "log")
    assert _wait(lambda: sub.expenses)


def test_unregister_stops_delivery():
    o, _ = _orch()
    sub = Subscriber()
    o.register(Publisher())
    o.register(sub)
    o.unregister("artemis")
    _dispatch(o, "pluto", "log")
    time.sleep(0.2)
    assert sub.expenses == []


def test_no_bus_means_nothing_published_and_no_crash():
    o = HestiaOrchestrator()
    o.register(Publisher())
    assert o.event_ledger is None
    # a module's .events is absent without a bus; publishing is the module's concern,
    # so dispatching a module that doesn't publish must still work
    class Quiet(BaseModule):
        name = "quiet"
        def can_handle(self, i): return True
        def handle(self, i, e, c): return {"response": "ok", "data": {}, "confidence": 1}
    o.register(Quiet())
    assert _dispatch(o, "quiet", "x").response == "ok"


# -------------------------------------------------------------- failures

def test_failing_subscriber_goes_to_dead_letters_and_trips_its_breaker():
    o, _ = _orch()

    class Bad(Subscriber):
        name = "bad"
        # only the failing subscription, so a success elsewhere can't reset the breaker
        EVENT_SUBSCRIPTIONS = {"pluto.expense_logged": "on_expense"}
        def on_expense(self, env): raise RuntimeError("subscriber bug")
    o.register(Publisher())
    o.register(Bad())
    for _ in range(8):
        _dispatch(o, "pluto", "log")
    assert _wait(lambda: o.circuit_breaker_status.get("bad", {}).get("total_failures", 0) >= 5)
    assert _wait(lambda: len(o.event_ledger.dead_letters()) >= 3)
    dl = o.event_ledger.dead_letters()[0]
    assert dl["subscriber"] == "bad" and "subscriber bug" in dl["error"]
    assert o.circuit_breaker_status["bad"]["state"] != "closed"


def test_failing_subscriber_does_not_break_the_publisher_or_siblings():
    o, _ = _orch()

    class Bad(Subscriber):
        name = "bad"
        def on_expense(self, env): raise RuntimeError("x")
    good = Subscriber()
    o.register(Publisher())
    o.register(Bad())
    o.register(good)
    assert _dispatch(o, "pluto", "log").response == "logged"
    assert _wait(lambda: good.expenses)


def test_bad_declarations_are_skipped_not_fatal():
    o, _ = _orch()

    class Odd(BaseModule):
        name = "odd"
        EVENT_SUBSCRIPTIONS = {"nodot": "m", "ok.topic": "missing_method", "ok.other": "real"}
        seen = []
        def can_handle(self, i): return False
        def handle(self, i, e, c): return {}
        def real(self, env): self.seen.append(env)
    mod = Odd()
    o.register(mod)
    assert o._events.subscriptions()["odd"] == ["ok.other"]


def test_invalid_event_name_is_ignored_and_returns_none():
    bus = EventBus()
    port = ModuleEventPort(bus, "pluto", EventLedger())
    assert port.publish("Bad Name") is None
    assert port.publish("good_name", {"a": 1}) is not None


def test_non_module_bus_events_are_not_delivered_to_module_handlers():
    o, bus = _orch()
    sub = Subscriber()
    o.register(sub)
    bus.emit("speak", {"text": "hi"})          # the pre-existing kind of event
    time.sleep(0.2)
    assert sub.expenses == [] and sub.intents == []


def test_ledger_counts_and_summary():
    led = EventLedger()
    assert "No module events" in led.summary()
    led.record_published(make_envelope("a.b", "a", {}))
    led.record_published(make_envelope("a.b", "a", {}))
    led.record_published(make_envelope("c.d", "c", {}))
    assert led.counts() == {"a.b": 2, "c.d": 1}
    assert "3 module event(s)" in led.summary()
    assert len(led.recent()) == 3


def test_envelope_ids_increase():
    a, b = make_envelope("x.y", "x", None), make_envelope("x.y", "x", None)
    assert b["id"] > a["id"] and a["payload"] == {}
