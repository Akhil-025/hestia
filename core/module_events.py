"""
core/module_events.py

Module-to-module events through the bus (backlog #10), without modules ever
calling each other.

What was there before
---------------------
``core/event_bus.py`` existed and was used, but only for notifications aimed at
the front-end: ``speak``, ``wake_detected``, ``morning_brief_requested`` and a
couple of heartbeat events. Modules did not publish facts about what they had
done and nothing could subscribe to them, so "when I log a workout, Artemis
should know" could only be done by one module importing another, which the
architecture forbids.

The contract
------------
* A module PUBLISHES with ``self.events.publish("expense_logged", {...})``. The
  orchestrator injects ``module.events`` when the module registers. The topic on
  the bus is ``"<module>.<event>"`` (``"pluto.expense_logged"``).
* A module SUBSCRIBES by declaring a class attribute::

      EVENT_SUBSCRIPTIONS = {"pluto.expense_logged": "on_expense_logged"}

  and a method ``on_expense_logged(self, envelope: dict) -> None``. A pattern
  ending in ``.*`` (``"pluto.*"``) matches every event from that module.
* The subscriber is given a plain-data *envelope* (``id``, ``topic``,
  ``source``, ``ts``, ``payload``), never a reference to the publisher.
* Delivery is asynchronous (the bus's worker pool), so a slow or failing
  subscriber can never delay or break the publisher, and a module is never
  handed its own events back. A failing subscriber counts toward its module's
  circuit breaker (the same breakers a normal dispatch uses), and the failure
  is recorded in the ledger as a dead letter instead of vanishing into a log.
* The orchestrator itself publishes ``intent.handled`` for every dispatched
  intent: ``{"module", "intent", "ok", "ms"}``.

Subscriber methods run on bus worker threads: they must be thread-safe.
"""
from __future__ import annotations

import itertools
import logging
import re
import threading
import time
from collections import Counter, deque
from typing import Any, Callable, Deque, Optional

logger = logging.getLogger(__name__)

_EVENT_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")

_ids = itertools.count(1)


def make_envelope(topic: str, source: str, payload: Any) -> dict:
    return {
        "id": next(_ids), "topic": topic, "source": source,
        "ts": time.time(), "payload": payload if payload is not None else {},
    }


class EventLedger:
    """Bounded record of what was published and what failed to be delivered.

    Subscribes to the bus wildcard, so it sees every event from every source
    (including the audio/heartbeat ones), but only keeps the envelope metadata
    for module events and counts the rest.
    """

    def __init__(self, size: int = 200) -> None:
        self._lock = threading.Lock()
        self._recent: Deque[dict] = deque(maxlen=size)
        self._dead: Deque[dict] = deque(maxlen=size)
        self._counts: Counter = Counter()

    def __call__(self, data: Any) -> None:       # bus wildcard callback
        pass

    def record_published(self, envelope: dict) -> None:
        with self._lock:
            self._counts[envelope["topic"]] += 1
            self._recent.append({k: envelope[k] for k in ("id", "topic", "source", "ts")})

    def record_dead_letter(self, envelope: dict, subscriber: str, error: BaseException) -> None:
        with self._lock:
            self._dead.append({
                "id": envelope.get("id"), "topic": envelope.get("topic"),
                "subscriber": subscriber, "error": f"{type(error).__name__}: {error}",
                "ts": time.time(),
            })

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(self._recent)[-limit:]

    def dead_letters(self) -> list[dict]:
        with self._lock:
            return list(self._dead)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def summary(self) -> str:
        counts, dead = self.counts(), self.dead_letters()
        if not counts:
            return "No module events have been published yet."
        top = ", ".join(f"{t} x{n}" for t, n in Counter(counts).most_common(5))
        tail = f" {len(dead)} delivery failure(s) recorded." if dead else ""
        return f"{sum(counts.values())} module event(s) published; most common: {top}.{tail}"


class ModuleEventPort:
    """The publishing handle a module gets as ``self.events``."""

    def __init__(self, bus, module: str, ledger: Optional[EventLedger] = None) -> None:
        self._bus = bus
        self._module = module
        self._ledger = ledger

    def publish(self, event: str, payload: Optional[dict] = None) -> Optional[int]:
        """Publish ``<module>.<event>``. Returns the event id, or None when the
        name is invalid (logged, never raised: publishing must not break a
        handler that has already done its real work)."""
        if not isinstance(event, str) or not _EVENT_RE.match(event):
            logger.warning("%s: ignoring event with invalid name %r.", self._module, event)
            return None
        topic = f"{self._module}.{event}"
        env = make_envelope(topic, self._module, payload)
        publish_envelope(self._bus, self._ledger, env)
        return env["id"]


def publish_envelope(bus, ledger: Optional[EventLedger], env: dict) -> None:
    if ledger is not None:
        ledger.record_published(env)
    try:
        bus.emit(env["topic"], env)
    except Exception:
        logger.exception("Could not publish %s.", env.get("topic"))


def topic_matches(pattern: str, topic: str) -> bool:
    if pattern == "*":
        return True
    if pattern.endswith(".*"):
        return topic.startswith(pattern[:-1])
    return pattern == topic


def _is_wild(pattern: str) -> bool:
    return pattern == "*" or pattern.endswith(".*")


def valid_pattern(pattern: str) -> bool:
    if pattern == "*":
        return True
    base = pattern[:-2] if pattern.endswith(".*") else pattern
    return bool(_EVENT_RE.match(base)) and "." in pattern


class ModuleEventWiring:
    """Connects modules' ``EVENT_SUBSCRIPTIONS`` to the bus.

    Owned by the orchestrator. ``guard`` is the orchestrator's circuit-breaker
    hook: ``guard.before(module)`` raises when the module's breaker is open,
    ``guard.success(module)`` / ``guard.failure(module)`` record the outcome.
    """

    def __init__(self, bus, guard, ledger: Optional[EventLedger] = None) -> None:
        self.bus = bus
        self.guard = guard
        self.ledger = ledger or EventLedger()
        self._subs: dict[str, list[tuple[str, Callable]]] = {}
        self._lock = threading.Lock()

    def attach(self, module) -> list[str]:
        """Give *module* its publishing port and subscribe its handlers.
        Returns the patterns subscribed. Bad declarations are skipped with a
        warning rather than failing registration."""
        name = module.name
        try:
            module.events = ModuleEventPort(self.bus, name, self.ledger)
        except Exception:        # a module using __slots__, say
            logger.debug("Could not set .events on %s.", name)
        declared = getattr(module, "EVENT_SUBSCRIPTIONS", None) or {}
        self.detach(name)
        patterns: list[str] = []
        for pattern, method_name in declared.items():
            handler = getattr(module, method_name, None)
            if not valid_pattern(pattern) or not callable(handler):
                logger.warning("%s: skipping subscription %r -> %r (invalid pattern or missing method).",
                               name, pattern, method_name)
                continue
            cb = self._make_callback(name, pattern, handler)
            self.bus.on("*" if _is_wild(pattern) else pattern, cb)
            with self._lock:
                self._subs.setdefault(name, []).append((pattern, cb))
            patterns.append(pattern)
        return patterns

    def detach(self, name: str) -> None:
        with self._lock:
            subs = self._subs.pop(name, [])
        for pattern, cb in subs:
            try:
                self.bus.off("*" if _is_wild(pattern) else pattern, cb)
            except Exception:
                pass

    def _make_callback(self, module: str, pattern: str, handler: Callable) -> Callable:
        def _deliver(data: Any) -> None:
            if not isinstance(data, dict) or "topic" not in data or "source" not in data:
                return                     # not a module envelope (e.g. "speak")
            if not topic_matches(pattern, data["topic"]) or data["source"] == module:
                return
            try:
                self.guard.before(module)
            except Exception:
                self.ledger.record_dead_letter(data, module, RuntimeError("module breaker open"))
                return
            try:
                handler(data)
            except Exception as exc:
                logger.exception("%s failed handling %s.", module, data.get("topic"))
                self.guard.failure(module)
                self.ledger.record_dead_letter(data, module, exc)
                return
            self.guard.success(module)
        _deliver.__qualname__ = f"{module}.event[{pattern}]"
        return _deliver

    def subscriptions(self) -> dict[str, list[str]]:
        with self._lock:
            return {m: [p for p, _ in subs] for m, subs in self._subs.items()}
