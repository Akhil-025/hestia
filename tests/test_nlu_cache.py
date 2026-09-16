# tests/test_nlu_cache.py
"""
Tests for core/nlu_cache.py (backlog #28).

The risky part of caching classification isn't the cache mechanics, it's
the correctness conditions:

  - classification is context-sensitive (HestiaNLU._build_prompt folds in
    recent turns and memory facts), so the key must include the context;
  - failure shapes must never be cached, or one unreachable-Ollama moment
    becomes a guaranteed-failure window;
  - callers mutate the returned dict (entity normalisation, learn_fact
    repair), so the cache must hand out copies.

Each of those has a test below.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.nlu_cache import NLUCache


def good_result(intent="apollo_track_sleep", confidence=0.93):
    return {
        "intent": intent,
        "entities": {"hours": 7},
        "response": "",
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Basic hit/miss
# ---------------------------------------------------------------------------

def test_miss_on_an_empty_cache():
    assert NLUCache().get("log my sleep") is None


def test_hit_after_put():
    cache = NLUCache()
    cache.put("log my sleep", good_result())
    assert cache.get("log my sleep")["intent"] == "apollo_track_sleep"


def test_key_is_case_and_whitespace_insensitive():
    cache = NLUCache()
    cache.put("Log  My   Sleep", good_result())
    assert cache.get("log my sleep") is not None


def test_different_queries_do_not_collide():
    cache = NLUCache()
    cache.put("log my sleep", good_result("apollo_track_sleep"))
    cache.put("log my workout", good_result("apollo_log_workout"))
    assert cache.get("log my sleep")["intent"] == "apollo_track_sleep"
    assert cache.get("log my workout")["intent"] == "apollo_log_workout"


# ---------------------------------------------------------------------------
# Context sensitivity
# ---------------------------------------------------------------------------

def test_same_text_with_different_context_is_a_miss():
    cache = NLUCache()
    cache.put("yes", good_result(), context=[{"query": "send the email?"}])
    assert cache.get("yes", context=[{"query": "delete my notes?"}]) is None


def test_same_text_with_identical_context_is_a_hit():
    cache = NLUCache()
    ctx = [{"query": "send the email?", "response": "send it?"}]
    cache.put("yes", good_result(), context=ctx)
    assert cache.get("yes", context=list(ctx)) is not None


def test_context_ordering_is_significant():
    cache = NLUCache()
    cache.put("yes", good_result(), context=[{"a": 1}, {"b": 2}])
    assert cache.get("yes", context=[{"b": 2}, {"a": 1}]) is None


def test_no_context_differs_from_some_context():
    cache = NLUCache()
    cache.put("yes", good_result())
    assert cache.get("yes", context=[{"a": 1}]) is None


def test_unserialisable_context_still_produces_a_stable_key():
    cache = NLUCache()

    class Weird:
        def __repr__(self):
            return "<weird>"

    ctx = [Weird()]
    cache.put("hello", good_result(), context=ctx)
    assert cache.get("hello", context=ctx) is not None


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

def test_entry_expires_after_the_ttl():
    cache = NLUCache(ttl_seconds=0.05)
    cache.put("log my sleep", good_result())
    time.sleep(0.08)
    assert cache.get("log my sleep") is None


def test_entry_survives_within_the_ttl():
    cache = NLUCache(ttl_seconds=5)
    cache.put("log my sleep", good_result())
    assert cache.get("log my sleep") is not None


def test_zero_ttl_disables_the_cache_entirely():
    # This is what the eval script uses so cached hits can't inflate
    # measured accuracy.
    cache = NLUCache(ttl_seconds=0)
    assert cache.put("log my sleep", good_result()) is False
    assert cache.get("log my sleep") is None


def test_expired_entry_is_evicted_not_just_ignored():
    cache = NLUCache(ttl_seconds=0.05)
    cache.put("q", good_result())
    time.sleep(0.08)
    cache.get("q")
    assert cache.stats()["entries"] == 0


# ---------------------------------------------------------------------------
# What must not be cached
# ---------------------------------------------------------------------------

def test_the_parse_failure_fallback_is_not_cached():
    cache = NLUCache()
    result = {
        "intent": "chat",
        "entities": {},
        "response": "Sorry, I had trouble understanding that.",
        "confidence": 0.5,
    }
    assert cache.put("gibberish", result) is False
    assert cache.get("gibberish") is None


def test_the_backend_unreachable_shape_is_not_cached():
    cache = NLUCache()
    result = {
        "intent": "chat",
        "entities": {},
        "response": "My backend isn't responding right now.",
        "confidence": 0.0,
    }
    assert cache.put("hello", result) is False


def test_zero_confidence_is_not_cached():
    assert NLUCache().put("hello", good_result(confidence=0.0)) is False


def test_missing_or_empty_intent_is_not_cached():
    cache = NLUCache()
    assert cache.put("hello", {"entities": {}, "confidence": 0.9}) is False
    assert cache.put("hello", {"intent": "", "confidence": 0.9}) is False


def test_non_dict_result_is_not_cached():
    assert NLUCache().put("hello", "not a dict") is False


def test_non_numeric_confidence_is_not_cached():
    assert NLUCache().put("hello", {"intent": "chat", "confidence": "high"}) is False


def test_is_cacheable_accepts_a_normal_classification():
    assert NLUCache.is_cacheable(good_result()) is True


# ---------------------------------------------------------------------------
# Isolation (callers mutate what they get back)
# ---------------------------------------------------------------------------

def test_mutating_a_returned_result_does_not_corrupt_the_cache():
    cache = NLUCache()
    cache.put("log my sleep", good_result())
    first = cache.get("log my sleep")
    first["intent"] = "mutated"
    first["entities"]["hours"] = 999
    assert cache.get("log my sleep")["intent"] == "apollo_track_sleep"


def test_mutating_the_stored_source_dict_does_not_change_the_cache():
    cache = NLUCache()
    source = good_result()
    cache.put("log my sleep", source)
    source["intent"] = "mutated"
    assert cache.get("log my sleep")["intent"] == "apollo_track_sleep"


# ---------------------------------------------------------------------------
# Bounds and eviction
# ---------------------------------------------------------------------------

def test_cache_is_bounded_by_max_entries():
    cache = NLUCache(max_entries=5)
    for i in range(20):
        cache.put(f"query {i}", good_result())
    assert cache.stats()["entries"] == 5


def test_eviction_is_least_recently_used():
    cache = NLUCache(max_entries=2)
    cache.put("a", good_result())
    cache.put("b", good_result())
    cache.get("a")               # touch 'a' so 'b' is now the LRU
    cache.put("c", good_result())
    assert cache.get("a") is not None
    assert cache.get("b") is None


def test_re_putting_an_existing_key_does_not_grow_the_cache():
    cache = NLUCache()
    for _ in range(5):
        cache.put("same query", good_result())
    assert cache.stats()["entries"] == 1


# ---------------------------------------------------------------------------
# Stats and invalidation
# ---------------------------------------------------------------------------

def test_stats_track_hits_misses_and_hit_rate():
    cache = NLUCache()
    cache.put("q", good_result())
    cache.get("q")
    cache.get("q")
    cache.get("other")
    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == round(2 / 3, 3)


def test_hit_rate_is_zero_with_no_lookups():
    assert NLUCache().stats()["hit_rate"] == 0.0


def test_invalidate_clears_everything():
    cache = NLUCache()
    cache.put("q", good_result())
    cache.invalidate()
    assert cache.get("q") is None
    assert cache.stats()["entries"] == 0


def test_make_key_is_deterministic():
    assert NLUCache.make_key("Log My Sleep") == NLUCache.make_key("log  my sleep")
