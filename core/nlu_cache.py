"""
core/nlu_cache.py

Short-TTL cache for NLU classification results (backlog #28).

Why
---
Classification runs on every query, costs an Ollama round-trip (plus up to
three retries), and is by far the most repeated call during development:
re-running the same test phrase twenty times while tuning a module pays
the full classification cost twenty times for an answer that cannot have
changed. Same for the web UI and Telegram bot, where re-sending an
identical message is a normal thing for a user to do.

Design
------
- **Short TTL by default (90 s).** Classification is *not* a pure function
  of the query text: ``HestiaNLU._build_prompt`` folds in recent
  conversation context and memory facts, so the same words can legitimately
  classify differently ten minutes apart. A short TTL captures the "I just
  sent that" case without pinning a stale answer for the session.
- **Context-aware key.** The cache key includes a fingerprint of the
  context turns passed in, so a repeat of the same text in a *different*
  conversational state is a miss, not a wrong hit.
- **Never caches failures.** A result is only stored when it looks like a
  real classification (an intent that isn't the generic error fallback).
  Caching "Sorry, I had trouble understanding that" would turn one bad
  Ollama moment into 90 seconds of guaranteed failure.
- **Bounded.** LRU eviction at ``max_entries`` so a long session can't grow
  it without limit.
- **Thread-safe.** The web UI, Telegram bot and heartbeat all call
  ``understand()`` from different threads in the single process.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Iterable, Optional

_DEFAULT_TTL_SECONDS = 90.0
_DEFAULT_MAX_ENTRIES = 256

# Intents/responses that indicate the pipeline failed rather than
# classified. These must never be cached — see module docstring.
_NON_CACHEABLE_MARKERS = (
    "Sorry, I had trouble understanding that.",
    "My backend isn't responding right now.",
)


def _fingerprint_context(context: Optional[Iterable[Any]]) -> str:
    """
    Stable short hash of the context passed to the NLU.

    Uses a hash rather than the raw context so the key stays small and no
    conversation text is held in a second place in memory.
    """
    if not context:
        return "none"
    try:
        blob = json.dumps(context, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(context)
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:12]


class NLUCache:
    """Bounded, TTL-expiring, thread-safe cache of classification results."""

    def __init__(
        self,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.ttl = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._store: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # -- keys -----------------------------------------------------------

    @staticmethod
    def make_key(text: str, context: Optional[Iterable[Any]] = None) -> str:
        normalised = " ".join((text or "").lower().split())
        return f"{normalised}|{_fingerprint_context(context)}"

    # -- api ------------------------------------------------------------

    def get(self, text: str, context: Optional[Iterable[Any]] = None) -> Optional[dict[str, Any]]:
        """Return a cached result for *text*, or None on miss/expiry."""
        if self.ttl <= 0:
            return None
        key = self.make_key(text, context)
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, result = entry
            if now - stored_at > self.ttl:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)  # LRU touch
            self.hits += 1
            # Copy: callers mutate the result (entity normalisation,
            # learn_fact repair), and a mutated dict must not leak back
            # into the cache.
            return dict(result)

    def put(
        self,
        text: str,
        result: dict[str, Any],
        context: Optional[Iterable[Any]] = None,
    ) -> bool:
        """
        Store *result* unless it represents a failure. Returns True if stored.
        """
        if self.ttl <= 0 or not isinstance(result, dict):
            return False
        if not self.is_cacheable(result):
            return False

        key = self.make_key(text, context)
        with self._lock:
            self._store[key] = (time.monotonic(), dict(result))
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)  # evict least-recently-used
        return True

    @staticmethod
    def is_cacheable(result: dict[str, Any]) -> bool:
        """False for the NLU's error/fallback shapes (see module docstring)."""
        intent = result.get("intent")
        if not isinstance(intent, str) or not intent:
            return False
        response = result.get("response") or ""
        if any(marker in response for marker in _NON_CACHEABLE_MARKERS):
            return False
        # A zero-confidence answer is the unreachable-backend shape.
        try:
            if float(result.get("confidence", 0)) <= 0.0:
                return False
        except (TypeError, ValueError):
            return False
        return True

    def invalidate(self) -> None:
        """Drop everything (used by config/prompt reloads and tests)."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            size = len(self._store)
        total = self.hits + self.misses
        return {
            "entries": size,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
