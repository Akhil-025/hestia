"""
modules/pluto/throttle.py

Visibility into market-data API throttling (backlog #144).

Pluto's price lookups (Yahoo Finance charts, CoinGecko, Yahoo news) are
unauthenticated, best-effort endpoints that rate-limit without notice. Before
this module a 429 just looked like any other failure: the retry loop spent
its attempts, logged a warning nobody reads, and the user got "couldn't fetch
price history". ThrottleMonitor records what happened per source so Pluto can
say "Yahoo Finance is rate-limiting requests, try again in about 40 seconds"
instead.

What it tracks, per source name (e.g. "yahoo_chart", "coingecko", "yahoo_news"):

* requests, successes, throttled (HTTP 429), other errors, retries
* the server's Retry-After, when it sent one, as a deadline
* the most recent outcome, which decides ``is_throttled``

It only observes. It never sleeps and never blocks; the one thing callers do
with it is ``cooldown_remaining()``, to skip a request that is certain to be
refused while a Retry-After is still running.

Thread-safe (the web UI and the heartbeat both call into Pluto). The module
level ``MONITOR`` is shared so every fetcher reports to one place.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Optional

# A throttle with no Retry-After header is treated as "recent" for this long.
DEFAULT_COOLDOWN_S = 60.0
# Ignore a Retry-After beyond this; a voice assistant shouldn't promise
# "try again in 3 hours" off one header, and it bounds a hostile/garbled value.
MAX_RETRY_AFTER_S = 900.0
# Recent-event ring buffer per source.
MAX_EVENTS = 50


def parse_retry_after(value: Any) -> Optional[float]:
    """Seconds from a Retry-After header (delta-seconds form), else None.

    The HTTP-date form is rare for these APIs and is ignored rather than
    guessed at. Negative, non-finite or non-numeric values give None.
    """
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_S)


class _SourceStats:
    __slots__ = ("requests", "successes", "throttled", "errors", "retries",
                 "last_outcome", "last_at", "last_throttled_at",
                 "retry_until", "last_error", "events")

    def __init__(self) -> None:
        self.requests = 0
        self.successes = 0
        self.throttled = 0
        self.errors = 0
        self.retries = 0
        self.last_outcome: Optional[str] = None     # "ok" | "throttled" | "error"
        self.last_at: Optional[float] = None
        self.last_throttled_at: Optional[float] = None
        self.retry_until: Optional[float] = None
        self.last_error: Optional[str] = None
        self.events: deque = deque(maxlen=MAX_EVENTS)


class ThrottleMonitor:
    def __init__(self, cooldown_s: float = DEFAULT_COOLDOWN_S,
                 clock: Callable[[], float] = time.time) -> None:
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._sources: dict[str, _SourceStats] = {}

    # ---- recording ----------------------------------------------------

    def _stats(self, source: str) -> _SourceStats:
        st = self._sources.get(source)
        if st is None:
            st = self._sources[source] = _SourceStats()
        return st

    def record_success(self, source: str) -> None:
        now = self._clock()
        with self._lock:
            st = self._stats(source)
            st.requests += 1
            st.successes += 1
            st.last_outcome, st.last_at = "ok", now
            st.retry_until = None            # the source is answering again
            st.events.append((now, "ok", ""))

    def record_throttled(self, source: str, retry_after: Optional[float] = None,
                         detail: str = "") -> None:
        now = self._clock()
        with self._lock:
            st = self._stats(source)
            st.requests += 1
            st.throttled += 1
            st.last_outcome, st.last_at, st.last_throttled_at = "throttled", now, now
            wait = retry_after if retry_after is not None else self.cooldown_s
            st.retry_until = now + min(wait, MAX_RETRY_AFTER_S)
            st.events.append((now, "throttled", detail[:120]))

    def record_error(self, source: str, detail: str = "") -> None:
        now = self._clock()
        with self._lock:
            st = self._stats(source)
            st.requests += 1
            st.errors += 1
            st.last_outcome, st.last_at = "error", now
            st.last_error = detail[:200] or None
            st.events.append((now, "error", detail[:120]))

    def record_retry(self, source: str) -> None:
        with self._lock:
            self._stats(source).retries += 1

    def note_http(self, source: str, status_code: Any, headers: Any = None) -> bool:
        """Record an HTTP response. Returns True when it was a 429.

        Only a real int 429 counts, so a test double whose ``status_code`` is
        a Mock is read as "not throttled" rather than guessed at.
        """
        if isinstance(status_code, int) and status_code == 429:
            retry_after = None
            try:
                retry_after = parse_retry_after(headers.get("Retry-After")) if headers else None
            except Exception:
                retry_after = None
            self.record_throttled(source, retry_after, "HTTP 429")
            return True
        return False

    # ---- reading ------------------------------------------------------

    def cooldown_remaining(self, source: str) -> float:
        """Seconds until a known Retry-After/cooldown ends; 0 when none."""
        now = self._clock()
        with self._lock:
            st = self._sources.get(source)
            if not st or st.retry_until is None:
                return 0.0
            return max(0.0, st.retry_until - now)

    def is_throttled(self, source: str) -> bool:
        """True while the latest outcome is a 429 and its cooldown is running."""
        with self._lock:
            st = self._sources.get(source)
            if not st or st.last_outcome != "throttled":
                return False
        return self.cooldown_remaining(source) > 0

    def snapshot(self) -> dict[str, dict]:
        now = self._clock()
        out: dict[str, dict] = {}
        with self._lock:
            for name, st in self._sources.items():
                remaining = max(0.0, st.retry_until - now) if st.retry_until else 0.0
                out[name] = {
                    "requests": st.requests,
                    "successes": st.successes,
                    "throttled": st.throttled,
                    "errors": st.errors,
                    "retries": st.retries,
                    "last_outcome": st.last_outcome,
                    "last_error": st.last_error,
                    "seconds_since_last_throttle": (
                        round(now - st.last_throttled_at, 1)
                        if st.last_throttled_at is not None else None),
                    "cooldown_remaining_s": round(remaining, 1),
                    "is_throttled": st.last_outcome == "throttled" and remaining > 0,
                }
        return out

    def reset(self) -> None:
        with self._lock:
            self._sources.clear()

    def describe(self, retry_settings: Optional[dict] = None) -> str:
        """Plain-language status for the chat reply."""
        snap = self.snapshot()
        if not snap:
            lines = ["No market-data requests have been made since Pluto started, so there is nothing to report yet."]
        else:
            lines = ["Market-data sources since Pluto started:"]
            for name in sorted(snap):
                s = snap[name]
                if s["is_throttled"]:
                    state = f"THROTTLED, cooling down for about {s['cooldown_remaining_s']:.0f}s"
                elif s["throttled"]:
                    state = "was throttled earlier, answering now" if s["last_outcome"] == "ok" else "was throttled earlier"
                else:
                    state = "no throttling seen"
                lines.append(
                    f"  {name:12} {s['requests']} request(s), {s['throttled']} throttled, "
                    f"{s['errors']} other error(s), {s['retries']} retry(ies): {state}"
                )
        if retry_settings:
            lines.append(
                "Retry settings: " + ", ".join(f"{k}={v}" for k, v in sorted(retry_settings.items()))
                + ". A throttled (429) request is not retried; Pluto waits out the cooldown instead."
            )
        return "\n".join(lines)


MONITOR = ThrottleMonitor()


def get_monitor() -> ThrottleMonitor:
    return MONITOR
