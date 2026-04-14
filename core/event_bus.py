"""
core/event_bus.py

A thread-safe, priority-aware event bus with sync and async dispatch,
one-shot listeners, wildcard subscriptions, per-event error handling,
and structured logging.
"""
from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WILDCARD = "*"
_DEFAULT_PRIORITY = 0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EventBusError(Exception):
    """Base exception for EventBus failures."""


class CallbackError(EventBusError):
    """Raised (in sync mode) when a callback raises and no error handler is set."""


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _Subscription:
    """
    A registered callback with associated metadata.

    Ordering is by priority (higher = earlier), so the list can be sorted
    once after each `on()` call rather than on every `emit()`.
    """

    # Negated so that higher priority sorts first with the default ascending sort.
    _sort_key: int = field(init=False, repr=False)
    priority: int = field(default=_DEFAULT_PRIORITY)
    callback: Callable[..., Any] = field(compare=False)
    one_shot: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        self._sort_key = -self.priority


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Thread-safe event bus with sync and async dispatch.

    Features
    --------
    - **Priority ordering**: callbacks with higher priority fire first.
    - **One-shot listeners**: auto-unregister after the first firing.
    - **Wildcard subscriptions**: subscribe to ``"*"`` to receive every event.
    - **Error handler**: register a per-bus error callback; defaults to
      logging the exception without swallowing it in sync mode.
    - **Thread naming**: async threads are named for easier debugging.
    - **`listeners_for`**: introspect registered callbacks without locking
      internals.

    Thread-safety
    -------------
    All mutations to the subscription registry are guarded by a single
    ``RLock`` (re-entrant so that error handlers can safely call `emit`).
    The callback list is copied before iteration so that callbacks may
    safely call `on` / `off` without deadlocking.
    """

    def __init__(
        self,
        error_handler: Optional[Callable[[str, BaseException], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        error_handler:
            Optional callable ``(event_name, exception) -> None`` invoked
            whenever a callback raises.  If *None*, exceptions are logged
            at ERROR level.  In sync mode the exception is also re-raised
            as a ``CallbackError`` after all callbacks have been attempted.
        """
        self._subscriptions: dict[str, list[_Subscription]] = {}
        self._lock = threading.RLock()
        self._error_handler = error_handler or self._default_error_handler
        self._thread_counter = 0  # used to produce unique thread names
        self._executor = ThreadPoolExecutor(max_workers=16)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on(
        self,
        event: str,
        callback: Callable[..., Any],
        *,
        priority: int = _DEFAULT_PRIORITY,
        one_shot: bool = False,
    ) -> None:
        """
        Subscribe *callback* to *event*.

        Parameters
        ----------
        event:
            The event name, or ``"*"`` to receive all events.
        callback:
            Any callable; receives a single positional argument (the payload).
        priority:
            Higher values fire earlier.  Default is 0.
        one_shot:
            If ``True``, the subscription is removed after the first firing.

        Raises
        ------
        ValueError
            If *event* is an empty string.
        TypeError
            If *callback* is not callable.
        """
        _validate_event(event)
        _validate_callback(callback)

        sub = _Subscription(priority=priority, callback=callback, one_shot=one_shot)

        with self._lock:
            bucket = self._subscriptions.setdefault(event, [])
            bucket.append(sub)
            bucket.sort()  # maintain priority order; list is usually short

        logger.debug(
            "Subscribed %r to event %r (priority=%d, one_shot=%s).",
            _callback_name(callback),
            event,
            priority,
            one_shot,
        )

    def once(
        self,
        event: str,
        callback: Callable[..., Any],
        *,
        priority: int = _DEFAULT_PRIORITY,
    ) -> None:
        """Convenience wrapper for ``on(..., one_shot=True)``."""
        self.on(event, callback, priority=priority, one_shot=True)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        """
        Unregister *callback* from *event*.

        No-ops silently if the callback is not registered.

        Raises
        ------
        ValueError
            If *event* is an empty string.
        """
        _validate_event(event)

        with self._lock:
            bucket = self._subscriptions.get(event)
            if bucket is None:
                return
            before = len(bucket)
            self._subscriptions[event] = [
                s for s in bucket if s.callback is not callback
            ]
            removed = before - len(self._subscriptions[event])

        if removed:
            logger.debug(
                "Unsubscribed %r from event %r (%d subscription(s) removed).",
                _callback_name(callback),
                event,
                removed,
            )

    def clear(self, event: Optional[str] = None) -> None:
        """
        Remove all subscriptions for *event*, or for every event if *None*.

        Parameters
        ----------
        event:
            Target event name, or ``None`` to clear everything.
        """
        with self._lock:
            if event is None:
                count = sum(len(v) for v in self._subscriptions.values())
                self._subscriptions.clear()
                logger.debug("Cleared all subscriptions (%d total).", count)
            else:
                _validate_event(event)
                count = len(self._subscriptions.pop(event, []))
                logger.debug(
                    "Cleared %d subscription(s) for event %r.", count, event
                )

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event: str, data: Any = None) -> None:
        """
        Fire all callbacks for *event* asynchronously (one daemon thread each).

        Wildcard subscribers registered with ``"*"`` are also notified.
        One-shot subscriptions are removed before the thread is spawned.

        Raises
        ------
        ValueError
            If *event* is an empty string.
        """
        _validate_event(event)
        callbacks = self._collect(event, remove_one_shots=True)

        for cb in callbacks:
            self._spawn(event, cb, data)

    def emit_sync(self, event: str, data: Any = None) -> None:
        """
        Fire all callbacks for *event* synchronously in the caller's thread.

        Raises
        ------
        ValueError
            If *event* is an empty string.
        CallbackError
            If one or more callbacks raised and no custom error handler is set.
            All callbacks are attempted before the error is raised.
        """
        _validate_event(event)
        callbacks = self._collect(event, remove_one_shots=True)
        errors: list[tuple[Callable, BaseException]] = []

        for cb in callbacks:
            try:
                cb(data)
            except Exception as exc:
                self._error_handler(event, exc)
                errors.append((cb, exc))

        if errors:
            names = ", ".join(_callback_name(cb) for cb, _ in errors)
            raise CallbackError(
                f"{len(errors)} callback(s) failed for event {event!r}: {names}"
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def listeners_for(self, event: str) -> list[Callable[..., Any]]:
        """Return a snapshot of callbacks registered for *event*."""
        _validate_event(event)
        with self._lock:
            return [s.callback for s in self._subscriptions.get(event, [])]

    @property
    def events(self) -> list[str]:
        """Return a snapshot of all event names that have at least one subscriber."""
        with self._lock:
            return [e for e, subs in self._subscriptions.items() if subs]

    def subscriber_count(self, event: Optional[str] = None) -> int:
        """
        Return the number of subscriptions for *event*, or the total if *None*.
        """
        with self._lock:
            if event is not None:
                return len(self._subscriptions.get(event, []))
            return sum(len(v) for v in self._subscriptions.values())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect(
        self, event: str, *, remove_one_shots: bool
    ) -> list[Callable[..., Any]]:
        """
        Return a deduplicated, ordered list of callbacks for *event* plus
        wildcard subscribers.

        One-shot subscriptions are atomically removed from the registry
        before the list is returned.
        """
        with self._lock:
            callbacks: list[Callable[..., Any]] = []
            seen: set[int] = set()  # track by id to avoid duplicates

            for bucket_key in (event, _WILDCARD):
                bucket = self._subscriptions.get(bucket_key)
                if not bucket:
                    continue

                keep: list[_Subscription] = []
                for sub in bucket:
                    if id(sub.callback) not in seen:
                        callbacks.append(sub.callback)
                        seen.add(id(sub.callback))
                    if not (remove_one_shots and sub.one_shot):
                        keep.append(sub)

                self._subscriptions[bucket_key] = keep

        return callbacks

    def _spawn(self, event: str, callback: Callable[..., Any], data: Any) -> None:
        def _run() -> None:
            try:
                callback(data)
            except Exception as exc:
                self._error_handler(event, exc)

        self._executor.submit(_run)

    def shutdown(self) -> None:
       self._executor.shutdown(wait=True)  

    @staticmethod
    def _default_error_handler(event: str, exc: BaseException) -> None:
        """Log the exception; do not swallow it in sync paths."""
        logger.error(
            "Unhandled exception in callback for event %r: %s",
            event,
            exc,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _validate_event(event: str) -> None:
    if not event or not event.strip():
        raise ValueError("Event name must be a non-empty string.")


def _validate_callback(callback: Callable[..., Any]) -> None:
    if not callable(callback):
        raise TypeError(f"callback must be callable, got {type(callback).__name__!r}.")


def _callback_name(callback: Callable[..., Any]) -> str:
    """Return a human-readable name for *callback* for use in log messages."""
    return getattr(callback, "__qualname__", None) or repr(callback)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

bus = EventBus()