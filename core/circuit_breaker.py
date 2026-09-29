"""
core/circuit_breaker.py

Per-module circuit breakers (backlog #7).

Why
---
Before this, a module whose `handle()` kept raising — Pluto's Postgres
pool exhausted, Athena's Ollama call timing out on every request — was
still called fresh on every single query that routed to it. Each call
paid the full cost of failing (a network timeout, a DB connection
attempt) before `_dispatch_primary`'s existing try/except caught it and
fell back to a generic error string. A crashing module didn't take down
dispatch, but it did make every query that happened to route there slow
and useless for as long as the module stayed broken.

Design
------
Standard three-state breaker, one instance per module name:

  CLOSED     — normal. Calls go through. N consecutive failures opens it.
  OPEN       — calls are skipped entirely and a fast, honest fallback
               response is returned instead, until `cooldown_seconds`
               has elapsed.
  HALF_OPEN  — after cooldown, exactly one call is let through as a
               probe. Success closes the breaker; failure reopens it and
               resets the cooldown clock.

Deliberately per-module, not global: Pluto being down says nothing about
whether Athena is down, and `HestiaOrchestrator._dispatch_primary`'s
existing `_find_alternate_module` recovery path already assumes modules
fail independently.

This wraps `_dispatch_primary`'s try/except; it does not replace it. An
exception from `mod.handle()` is still caught there and turned into
`_GENERIC_ERROR` — the breaker's job is purely to avoid making that call
in the first place once a module has proven itself unreliable.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_FAILURE_THRESHOLD = 3
_DEFAULT_COOLDOWN_SECONDS = 60.0


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _ModuleBreaker:
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    total_failures: int = 0
    total_trips: int = 0


class CircuitBreakerOpen(Exception):
    """
    Raised by :meth:`CircuitBreakerRegistry.guard` instead of calling
    through, when the named module's breaker is open.
    """

    def __init__(self, module: str, retry_after: float) -> None:
        self.module = module
        self.retry_after = round(max(retry_after, 0.0), 1)
        super().__init__(
            f"circuit open for module {module!r}; retry in "
            f"{self.retry_after:.0f}s"
        )


class CircuitBreakerRegistry:
    """
    Owns one breaker per module name. Thread-safe: dispatch can be called
    concurrently (see HestiaOrchestrator's own thread-safety note).
    """

    def __init__(
        self,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self.failure_threshold = int(failure_threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self._breakers: dict[str, _ModuleBreaker] = {}
        self._lock = threading.Lock()

    def _get(self, module: str) -> _ModuleBreaker:
        breaker = self._breakers.get(module)
        if breaker is None:
            breaker = _ModuleBreaker()
            self._breakers[module] = breaker
        return breaker

    def before_call(self, module: str) -> None:
        """
        Call immediately before invoking *module*'s handler.

        Raises :class:`CircuitBreakerOpen` if the call should be skipped.
        Transitions OPEN -> HALF_OPEN once the cooldown has elapsed,
        letting exactly one probe call through (the caller must report its
        outcome via :meth:`record_success`/:meth:`record_failure`).
        """
        with self._lock:
            breaker = self._get(module)

            if breaker.state == CircuitState.OPEN:
                elapsed = time.monotonic() - breaker.opened_at
                if elapsed < self.cooldown_seconds:
                    raise CircuitBreakerOpen(
                        module, self.cooldown_seconds - elapsed
                    )
                # Cooldown elapsed — allow exactly one probe through.
                breaker.state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit for module %r half-open after %.0fs; "
                    "probing with the next call.", module, elapsed,
                )

    def record_success(self, module: str) -> None:
        """Call after a successful handle() — resets and closes the breaker."""
        with self._lock:
            breaker = self._get(module)
            if breaker.state != CircuitState.CLOSED:
                logger.info("Circuit for module %r closed (recovered).", module)
            breaker.state = CircuitState.CLOSED
            breaker.consecutive_failures = 0

    def record_failure(self, module: str) -> None:
        """
        Call after a failed handle() (an exception, per
        `_dispatch_primary`'s existing except clause).

        A failure during HALF_OPEN reopens immediately — the probe
        failed, so there's no reason to let more real queries through
        before the cooldown runs again.
        """
        with self._lock:
            breaker = self._get(module)
            breaker.total_failures += 1
            breaker.consecutive_failures += 1

            if breaker.state == CircuitState.HALF_OPEN:
                self._trip(module, breaker)
                return

            if (
                breaker.state == CircuitState.CLOSED
                and breaker.consecutive_failures >= self.failure_threshold
            ):
                self._trip(module, breaker)

    def _trip(self, module: str, breaker: _ModuleBreaker) -> None:
        breaker.state = CircuitState.OPEN
        breaker.opened_at = time.monotonic()
        breaker.total_trips += 1
        logger.warning(
            "Circuit OPEN for module %r after %d consecutive failure(s); "
            "skipping calls to it for %.0fs.",
            module, breaker.consecutive_failures, self.cooldown_seconds,
        )

    def state_of(self, module: str) -> CircuitState:
        with self._lock:
            return self._get(module).state

    def reset(self, module: Optional[str] = None) -> None:
        """Force-close one breaker, or every breaker if *module* is None."""
        with self._lock:
            if module is None:
                self._breakers.clear()
            else:
                self._breakers.pop(module, None)

    def snapshot(self) -> dict[str, dict]:
        """
        Per-module state for diagnostics (surfaced via
        Diagnostics.module_status() as an extra field, and readable from
        modules_status).
        """
        with self._lock:
            return {
                name: {
                    "state": breaker.state.value,
                    "consecutive_failures": breaker.consecutive_failures,
                    "total_failures": breaker.total_failures,
                    "total_trips": breaker.total_trips,
                }
                for name, breaker in self._breakers.items()
            }
