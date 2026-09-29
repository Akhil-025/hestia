# tests/test_circuit_breaker.py
"""
Tests for core/circuit_breaker.py (backlog #7).
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.circuit_breaker import (
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
)


def make_registry(threshold=3, cooldown=0.1):
    return CircuitBreakerRegistry(failure_threshold=threshold, cooldown_seconds=cooldown)


# ---------------------------------------------------------------------------
# Closed state (normal operation)
# ---------------------------------------------------------------------------

def test_new_module_starts_closed():
    reg = make_registry()
    assert reg.state_of("pluto") == CircuitState.CLOSED
    reg.before_call("pluto")  # must not raise


def test_success_keeps_it_closed():
    reg = make_registry(threshold=2)
    reg.before_call("pluto")
    reg.record_success("pluto")
    assert reg.state_of("pluto") == CircuitState.CLOSED


def test_a_single_failure_below_threshold_stays_closed():
    reg = make_registry(threshold=3)
    reg.record_failure("pluto")
    assert reg.state_of("pluto") == CircuitState.CLOSED
    reg.before_call("pluto")  # still allowed through


def test_success_resets_the_failure_count():
    reg = make_registry(threshold=3)
    reg.record_failure("pluto")
    reg.record_failure("pluto")
    reg.record_success("pluto")
    reg.record_failure("pluto")
    # The two failures before the success must not carry over.
    assert reg.state_of("pluto") == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Tripping open
# ---------------------------------------------------------------------------

def test_reaching_the_threshold_opens_the_circuit():
    reg = make_registry(threshold=3)
    for _ in range(3):
        reg.record_failure("pluto")
    assert reg.state_of("pluto") == CircuitState.OPEN


def test_open_circuit_raises_on_before_call():
    reg = make_registry(threshold=1, cooldown=10)
    reg.record_failure("pluto")
    with pytest.raises(CircuitBreakerOpen) as exc:
        reg.before_call("pluto")
    assert exc.value.module == "pluto"
    assert exc.value.retry_after > 0


def test_modules_are_independent():
    reg = make_registry(threshold=1)
    reg.record_failure("pluto")
    assert reg.state_of("pluto") == CircuitState.OPEN
    assert reg.state_of("athena") == CircuitState.CLOSED
    reg.before_call("athena")  # unaffected


def test_trip_count_and_total_failures_are_tracked():
    reg = make_registry(threshold=2)
    reg.record_failure("pluto")
    reg.record_failure("pluto")   # trips
    snap = reg.snapshot()["pluto"]
    assert snap["total_failures"] == 2
    assert snap["total_trips"] == 1


# ---------------------------------------------------------------------------
# Cooldown and half-open probing
# ---------------------------------------------------------------------------

def test_before_call_still_raises_within_the_cooldown():
    reg = make_registry(threshold=1, cooldown=10)
    reg.record_failure("pluto")
    reg.before_call.__self__  # sanity: bound method
    with pytest.raises(CircuitBreakerOpen):
        reg.before_call("pluto")
    assert reg.state_of("pluto") == CircuitState.OPEN


def test_after_cooldown_transitions_to_half_open_and_allows_one_call():
    reg = make_registry(threshold=1, cooldown=0.05)
    reg.record_failure("pluto")
    time.sleep(0.08)
    reg.before_call("pluto")  # must not raise — this is the probe
    assert reg.state_of("pluto") == CircuitState.HALF_OPEN


def test_successful_probe_closes_the_circuit():
    reg = make_registry(threshold=1, cooldown=0.05)
    reg.record_failure("pluto")
    time.sleep(0.08)
    reg.before_call("pluto")
    reg.record_success("pluto")
    assert reg.state_of("pluto") == CircuitState.CLOSED
    reg.before_call("pluto")  # normal calls allowed again


def test_failed_probe_reopens_immediately():
    reg = make_registry(threshold=1, cooldown=0.05)
    reg.record_failure("pluto")
    time.sleep(0.08)
    reg.before_call("pluto")
    reg.record_failure("pluto")   # the probe itself failed
    assert reg.state_of("pluto") == CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpen):
        reg.before_call("pluto")


def test_reopening_resets_the_cooldown_clock():
    reg = make_registry(threshold=1, cooldown=0.1)
    reg.record_failure("pluto")
    time.sleep(0.12)
    reg.before_call("pluto")        # half-open probe
    reg.record_failure("pluto")     # reopens
    with pytest.raises(CircuitBreakerOpen) as exc:
        reg.before_call("pluto")    # cooldown clock restarted, still open
    assert exc.value.retry_after > 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_one_module():
    reg = make_registry(threshold=1)
    reg.record_failure("pluto")
    reg.record_failure("athena")
    reg.reset("pluto")
    assert reg.state_of("pluto") == CircuitState.CLOSED
    assert reg.state_of("athena") == CircuitState.OPEN


def test_reset_all_modules():
    reg = make_registry(threshold=1)
    reg.record_failure("pluto")
    reg.record_failure("athena")
    reg.reset()
    assert reg.snapshot() == {}


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def test_snapshot_is_empty_for_a_fresh_registry():
    assert make_registry().snapshot() == {}


def test_snapshot_only_includes_modules_that_have_been_touched():
    reg = make_registry()
    reg.before_call("pluto")
    assert set(reg.snapshot()) == {"pluto"}


def test_open_error_message_names_the_module_and_retry_time():
    reg = make_registry(threshold=1, cooldown=30)
    reg.record_failure("pluto")
    with pytest.raises(CircuitBreakerOpen) as exc:
        reg.before_call("pluto")
    assert "pluto" in str(exc.value)
    assert "retry" in str(exc.value)
