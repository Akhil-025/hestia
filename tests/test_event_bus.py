# tests/test_event_bus.py
"""
Regression tests for core/event_bus.py.

Run with:  pytest tests/test_event_bus.py -v

core/ previously had almost no direct test coverage even though it's the
layer every module is wired through (main.py's HestiaBuilder, Hecate's
routing, the heartbeat, the web UI). EventBus specifically is used for
cross-module notifications (e.g. "speak", "morning_brief_requested") where
a silent regression (wrong priority order, a one-shot listener that never
unregisters, a swallowed callback exception) would be very hard to notice
from the outside.

These tests use a fresh EventBus() per test rather than the module-level
`bus` singleton, so tests can't leak subscriptions into each other.
"""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus, CallbackError


def make_bus(**kwargs) -> EventBus:
    return EventBus(**kwargs)


# ---------------------------------------------------------------------------
# Basic on/emit_sync
# ---------------------------------------------------------------------------

def test_emit_sync_calls_subscriber_with_payload():
    bus = make_bus()
    received = []
    bus.on("ping", lambda data: received.append(data))

    bus.emit_sync("ping", {"n": 1})

    assert received == [{"n": 1}]


def test_emit_sync_with_no_subscribers_is_a_noop():
    bus = make_bus()
    # Should not raise even though nothing is subscribed.
    bus.emit_sync("nothing_listens_to_this")


def test_multiple_subscribers_all_fire():
    bus = make_bus()
    calls = []
    bus.on("event", lambda d: calls.append("a"))
    bus.on("event", lambda d: calls.append("b"))

    bus.emit_sync("event")

    assert sorted(calls) == ["a", "b"]


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

def test_higher_priority_fires_first():
    bus = make_bus()
    order = []
    bus.on("event", lambda d: order.append("low"), priority=0)
    bus.on("event", lambda d: order.append("high"), priority=10)
    bus.on("event", lambda d: order.append("mid"), priority=5)

    bus.emit_sync("event")

    assert order == ["high", "mid", "low"]


# ---------------------------------------------------------------------------
# One-shot listeners
# ---------------------------------------------------------------------------

def test_once_fires_only_a_single_time():
    bus = make_bus()
    calls = []
    bus.once("event", lambda d: calls.append(d))

    bus.emit_sync("event", "first")
    bus.emit_sync("event", "second")

    assert calls == ["first"]


def test_one_shot_removal_does_not_affect_persistent_listeners():
    bus = make_bus()
    persistent_calls = []
    once_calls = []
    bus.on("event", lambda d: persistent_calls.append(d))
    bus.once("event", lambda d: once_calls.append(d))

    bus.emit_sync("event")
    bus.emit_sync("event")

    assert len(persistent_calls) == 2
    assert len(once_calls) == 1


# ---------------------------------------------------------------------------
# Wildcard subscriptions
# ---------------------------------------------------------------------------

def test_wildcard_subscriber_receives_every_event():
    bus = make_bus()
    seen = []
    bus.on("*", lambda d: seen.append(d))

    bus.emit_sync("event_a", "a")
    bus.emit_sync("event_b", "b")

    assert seen == ["a", "b"]


def test_wildcard_and_specific_subscriber_both_fire_without_duplication():
    bus = make_bus()
    calls = []
    bus.on("event", lambda d: calls.append("specific"))
    bus.on("*", lambda d: calls.append("wildcard"))

    bus.emit_sync("event")

    assert sorted(calls) == ["specific", "wildcard"]


def test_same_callback_registered_for_event_and_wildcard_fires_once():
    bus = make_bus()
    calls = []

    def cb(d):
        calls.append(d)

    bus.on("event", cb)
    bus.on("*", cb)

    bus.emit_sync("event")

    # _collect() dedupes by callback identity across the (event, "*")
    # buckets, so a callback registered on both should only fire once.
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# off() / clear()
# ---------------------------------------------------------------------------

def test_off_unsubscribes_callback():
    bus = make_bus()
    calls = []

    def cb(d):
        calls.append(d)

    bus.on("event", cb)
    bus.off("event", cb)
    bus.emit_sync("event")

    assert calls == []


def test_off_on_unregistered_callback_is_a_silent_noop():
    bus = make_bus()
    bus.off("never_registered", lambda d: None)  # must not raise


def test_clear_single_event_leaves_others_intact():
    bus = make_bus()
    a_calls, b_calls = [], []
    bus.on("a", lambda d: a_calls.append(d))
    bus.on("b", lambda d: b_calls.append(d))

    bus.clear("a")
    bus.emit_sync("a")
    bus.emit_sync("b")

    assert a_calls == []
    assert b_calls == [None]


def test_clear_all_removes_every_subscription():
    bus = make_bus()
    bus.on("a", lambda d: None)
    bus.on("b", lambda d: None)

    bus.clear()

    assert bus.subscriber_count() == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_on_rejects_empty_event_name():
    bus = make_bus()
    with pytest.raises(ValueError):
        bus.on("", lambda d: None)


def test_on_rejects_non_callable_callback():
    bus = make_bus()
    with pytest.raises(TypeError):
        bus.on("event", "not_a_function")


def test_emit_rejects_empty_event_name():
    bus = make_bus()
    with pytest.raises(ValueError):
        bus.emit("")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_emit_sync_raises_callback_error_when_subscriber_throws():
    bus = make_bus()
    bus.on("event", lambda d: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(CallbackError):
        bus.emit_sync("event")


def test_emit_sync_still_runs_remaining_callbacks_after_one_fails():
    bus = make_bus()
    calls = []
    bus.on("event", lambda d: (_ for _ in ()).throw(RuntimeError("boom")), priority=10)
    bus.on("event", lambda d: calls.append("ran"), priority=0)

    with pytest.raises(CallbackError):
        bus.emit_sync("event")

    assert calls == ["ran"]


def test_custom_error_handler_is_invoked_instead_of_default_logging():
    handled = []
    bus = make_bus(error_handler=lambda event, exc: handled.append((event, str(exc))))
    bus.on("event", lambda d: (_ for _ in ()).throw(RuntimeError("boom")))

    # A custom error handler still lets emit_sync raise CallbackError after
    # invoking it — only the *handling* is customized, not the propagation.
    with pytest.raises(CallbackError):
        bus.emit_sync("event")

    assert handled == [("event", "boom")]


def test_async_emit_does_not_propagate_callback_exceptions_to_caller():
    bus = make_bus()
    # emit() dispatches on background threads; a raising callback must not
    # crash the calling thread.
    bus.on("event", lambda d: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.emit("event")  # must not raise
    bus.shutdown()


# ---------------------------------------------------------------------------
# Async emit()
# ---------------------------------------------------------------------------

def test_async_emit_eventually_invokes_subscriber():
    bus = make_bus()
    done = threading.Event()
    received = []

    def cb(data):
        received.append(data)
        done.set()

    bus.on("event", cb)
    bus.emit("event", "payload")

    assert done.wait(timeout=2.0), "callback was not invoked within timeout"
    assert received == ["payload"]
    bus.shutdown()


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def test_listeners_for_returns_snapshot():
    bus = make_bus()

    def cb(d):
        pass

    bus.on("event", cb)
    assert bus.listeners_for("event") == [cb]
    assert bus.listeners_for("other_event") == []


def test_events_property_only_lists_events_with_subscribers():
    bus = make_bus()
    bus.on("a", lambda d: None)

    assert bus.events == ["a"]


def test_subscriber_count_total_and_per_event():
    bus = make_bus()
    bus.on("a", lambda d: None)
    bus.on("a", lambda d: None)
    bus.on("b", lambda d: None)

    assert bus.subscriber_count("a") == 2
    assert bus.subscriber_count("b") == 1
    assert bus.subscriber_count() == 3
