"""Simple event bus for decoupled communication with thread safety and sync/async emission."""

import sys
import threading
from typing import Callable, Dict, List, Optional

class EventBus:
    """Thread-safe event bus allowing async or sync callback dispatch."""

    def __init__(self):
        """Initialize empty listener registry and lock."""
        self._listeners: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        with self._lock:
            self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister a specific callback for an event. Does nothing if not found."""
        with self._lock:
            if event in self._listeners:
                try:
                    self._listeners[event].remove(callback)
                except ValueError:
                    pass  # callback not in list, ignore

    def emit(self, event: str, data=None) -> None:
        """Fire all callbacks for event asynchronously in daemon threads."""
        with self._lock:
            callbacks = self._listeners.get(event, [])[:]  # copy to avoid mutation issues

        for cb in callbacks:
            def wrapper(callback=cb):
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error in event callback for '{event}': {e}", file=sys.stderr)
            t = threading.Thread(target=wrapper, daemon=True)
            t.start()

    def emit_sync(self, event: str, data=None) -> None:
        """Fire all callbacks for event synchronously in the caller's thread (for testing)."""
        with self._lock:
            callbacks = self._listeners.get(event, [])[:]

        for cb in callbacks:
            try:
                cb(data)
            except Exception as e:
                print(f"Error in sync callback for '{event}': {e}", file=sys.stderr)

    def clear(self, event: Optional[str] = None) -> None:
        """Remove all callbacks for an event, or all events if event is None."""
        with self._lock:
            if event is None:
                self._listeners.clear()
            else:
                self._listeners.pop(event, None)

# Module‑level singleton
bus = EventBus()

if __name__ == "__main__":
    # Smoke test
    test_event = "test_event"
    results = []

    def cb1(data):
        results.append(("cb1", data["msg"]))

    def cb2(data):
        results.append(("cb2", data["msg"]))

    bus.on(test_event, cb1)
    bus.on(test_event, cb2)

    # Async emit – need to wait briefly
    import time
    bus.emit(test_event, {"msg": "hello"})
    time.sleep(0.1)  # allow daemon threads to run
    assert len(results) == 2, "Both callbacks should have fired"
    assert ("cb1", "hello") in results
    assert ("cb2", "hello") in results

    # Test off()
    bus.off(test_event, cb1)
    results.clear()
    bus.emit_sync(test_event, {"msg": "world"})
    assert len(results) == 1 and results[0] == ("cb2", "world"), "cb1 should be removed"

    # Test clear(event)
    bus.clear(test_event)
    results.clear()
    bus.emit_sync(test_event, {"msg": "gone"})
    assert results == [], "No callbacks should remain"

    # Test clear(None)
    bus.on("a", lambda x: None)
    bus.on("b", lambda x: None)
    bus.clear()
    with bus._lock:
        assert bus._listeners == {}, "All events cleared"

    print("All smoke tests passed.")