"""
core/web_live.py - live updates for the web UI (backlog #187, feeds #182).

What this is
------------
A small in-memory hub. Things that happen in Hestia (a message was handled, a
reminder fired, a module announced something) are published to it; browsers
that have the dashboard open receive them over one long-lived HTTP response
(``GET /api/live/stream``, Server-Sent Events) instead of polling.

Why Server-Sent Events and not a WebSocket
------------------------------------------
The backlog item says "WebSocket". The web UI is served by waitress, which
cannot upgrade a connection to a WebSocket, and everything the page needs is
one-way (server -> browser): browsers send their messages through the normal
``POST /api/chat`` routes. SSE does exactly that job over plain HTTP, works
with waitress and Flask's dev server unchanged, reconnects by itself, and
resumes from the last event it saw. If two-way traffic is ever needed, this
hub's ``publish`` / ``subscribe`` interface is the part to put a WebSocket
transport in front of.

Limits (deliberate)
-------------------
* Each open stream holds one server thread, so the number of simultaneous
  streams is capped (``max_subscribers``); the (n+1)th gets HTTP 429 and the
  page falls back to polling.
* The hub remembers only the last ``max_events`` events, in memory. It is a
  live view, not a log: the interaction history is still Mnemosyne's job.
* A slow browser never blocks the rest of Hestia: each subscriber has a
  bounded queue and, when it overflows, the oldest events are dropped and the
  browser is told to refresh (``resync``).
"""
from __future__ import annotations

import collections
import contextlib
import json
import logging
import queue
import re
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Iterator, Optional

logger = logging.getLogger(__name__)

# Text longer than this is cut in a live event. The full text is still in the
# interaction history; the live feed only needs enough to show a line.
_MAX_TEXT = 4000

_REMINDER_RE = re.compile(r"\b(reminder|you missed)\b", re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def module_for(intent: str) -> str:
    """Which module owns *intent* (``"core"`` when unknown).

    Read from the intent registry so the feed uses the same names as ``/help``
    and the routing log. Imported lazily: the registry package pulls in the
    routing engine, which a test or a stripped-down install may not have.
    """
    intent = (intent or "").strip()
    if not intent:
        return "core"
    try:
        from modules.hecate.intent_registry import MODULE_PREFIXES, module_for_intent
        # The registry keys most intents with a module prefix
        # ("apollo_track_sleep"), but an interaction may be logged under the
        # bare name ("track_sleep") - try it as given, then with each prefix.
        found = module_for_intent(intent)
        for prefix in MODULE_PREFIXES:
            if found:
                break
            found = module_for_intent(prefix + intent)
        return found or "core"
    except Exception:
        return "core"


class Subscription:
    """One browser's view of the hub. Always ``close()`` it (use try/finally)."""

    def __init__(self, hub: "LiveHub", backlog: list[dict], queue_size: int) -> None:
        self._hub = hub
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=queue_size)
        self._dropped = False
        self._closed = False
        for ev in backlog:
            self._offer(ev)

    # called by the hub, under its lock
    def _offer(self, ev: dict) -> None:
        try:
            self._q.put_nowait(ev)
        except queue.Full:
            # Make room by dropping the oldest, and remember that something
            # was lost so the browser can reload instead of showing a gap.
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._dropped = True
            try:
                self._q.put_nowait(ev)
            except queue.Full:
                pass

    def get(self, timeout: float) -> Optional[dict]:
        """Next event, or None if nothing arrived within *timeout* seconds."""
        if self._dropped:
            self._dropped = False
            return {"id": None, "type": "resync", "ts": _now_iso()}
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._hub._unsubscribe(self)


class LiveHub:
    def __init__(
        self,
        max_events: int = 200,
        max_subscribers: int = 6,
        queue_size: int = 100,
    ) -> None:
        self._events: Deque[dict] = collections.deque(maxlen=max(10, int(max_events)))
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()
        self._next_id = 1
        self.max_subscribers = max(1, int(max_subscribers))
        self._queue_size = max(10, int(queue_size))
        self._origin = threading.local()
        self._bus: Any = None
        self._handlers: list[tuple[str, Callable[[Any], None]]] = []

    # -- who caused an event -------------------------------------------

    @contextlib.contextmanager
    def origin(self, name: str) -> Iterator[None]:
        """Tag every interaction logged by *this thread* inside the block.

        The web chat route wraps its call to the pipeline in
        ``with hub.origin("web")``. ``interaction_logged`` is emitted
        synchronously in the thread that handled the query, so the handler
        below can read the tag back. That lets the Chat tab show messages that
        came from the voice loop or Telegram without echoing its own.
        """
        previous = getattr(self._origin, "name", None)
        self._origin.name = name
        try:
            yield
        finally:
            self._origin.name = previous

    def current_origin(self) -> str:
        return getattr(self._origin, "name", None) or "other"

    # -- publishing -----------------------------------------------------

    def publish(self, type_: str, **payload: Any) -> dict:
        """Record an event and hand it to every open stream. Never raises."""
        with self._lock:
            ev = {"id": self._next_id, "type": type_, "ts": _now_iso(), **payload}
            self._next_id += 1
            self._events.append(ev)
            for sub in list(self._subs):
                sub._offer(ev)
        return ev

    def recent(self, limit: int = 50, types: Optional[tuple[str, ...]] = None) -> list[dict]:
        """Newest-last copies of the remembered events."""
        with self._lock:
            items = [dict(e) for e in self._events if not types or e["type"] in types]
        return items[-limit:] if limit > 0 else items

    # -- subscribing ----------------------------------------------------

    def subscribe(self, last_id: Any = None) -> Optional[Subscription]:
        """Open a stream. *last_id* replays events newer than it (reconnects).

        Returns None when the connection cap is reached.
        """
        try:
            after = int(last_id) if last_id not in (None, "") else None
        except (TypeError, ValueError):
            after = None
        with self._lock:
            if len(self._subs) >= self.max_subscribers:
                return None
            backlog = [e for e in self._events if after is not None and e["id"] > after]
            sub = Subscription(self, backlog, self._queue_size)
            self._subs.append(sub)
            return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    # -- Server-Sent Events framing --------------------------------------

    @staticmethod
    def sse(ev: dict) -> str:
        data = json.dumps(ev, ensure_ascii=False, default=str)
        head = f"id: {ev['id']}\n" if ev.get("id") is not None else ""
        return f"{head}event: {ev['type']}\ndata: {data}\n\n"

    def stream(
        self,
        sub: Subscription,
        heartbeat: float = 15.0,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Iterator[str]:
        """The body of the HTTP response: SSE frames until the client leaves.

        A comment line every *heartbeat* seconds keeps proxies from closing an
        idle connection and is also how a vanished client is noticed (the
        write fails, the generator is closed, ``finally`` frees the slot).
        """
        try:
            yield "retry: 3000\n\n"
            yield self.sse({"id": None, "type": "hello", "ts": _now_iso(),
                            "subscribers": self.subscriber_count})
            while not (should_stop and should_stop()):
                ev = sub.get(timeout=heartbeat)
                yield self.sse(ev) if ev is not None else ": keep-alive\n\n"
        finally:
            sub.close()

    # -- wiring to the event bus ------------------------------------------

    def attach_bus(self, bus: Any) -> None:
        """Listen to the bus for ``interaction_logged`` and ``speak``.

        Idempotent. Handlers swallow every exception: ``emit_sync`` raises if a
        listener does, and a broken live feed must never fail a user's query.
        """
        if self._bus is not None:
            return
        self._bus = bus

        def on_interaction(data: Any) -> None:
            try:
                data = data or {}
                intent = str(data.get("intent") or "chat")
                self.publish(
                    "message",
                    origin=self.current_origin(),
                    query=str(data.get("query") or "")[:_MAX_TEXT],
                    response=str(data.get("response") or "")[:_MAX_TEXT],
                    intent=intent,
                    module=module_for(intent),
                )
            except Exception:
                logger.debug("live hub: interaction handler failed", exc_info=True)

        def on_speak(data: Any) -> None:
            try:
                data = data or {}
                text = str(data.get("text") or "").strip()
                if not text:
                    return
                kind = "reminder" if _REMINDER_RE.search(text) else "notification"
                self.publish(
                    kind, text=text[:_MAX_TEXT],
                    module=str(data.get("module") or "") or None,
                )
            except Exception:
                logger.debug("live hub: speak handler failed", exc_info=True)

        for name, fn in (("interaction_logged", on_interaction), ("speak", on_speak)):
            bus.on(name, fn)
            self._handlers.append((name, fn))

    def detach_bus(self) -> None:
        if self._bus is None:
            return
        for name, fn in self._handlers:
            try:
                self._bus.off(name, fn)
            except Exception:
                pass
        self._handlers.clear()
        self._bus = None
