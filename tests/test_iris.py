# tests/test_iris.py
"""
Regression tests for modules/iris/iris_engine.py.

Run with:  python3 tests/test_iris.py
(or:       python -m pytest tests/test_iris.py -v)

These cover three real bugs found and fixed in IrisEngine:

1. handle() branched on the "iris_"-PREFIXED intent strings
   ("iris_search", "iris_ingest", "iris_analyse") — but
   HestiaOrchestrator._strip_module_prefix() strips "iris_" off the intent
   before calling handle() for every normal (non-trigger-tier) dispatch.
   So the intent handle() actually received was "search"/"ingest"/
   "analyse"/"status", none of which matched any branch, and every real
   request silently fell through to a blank, 0.0-confidence response —
   despite can_handle() reporting True for that exact intent.

2. "status" / "iris_status" was accepted by can_handle() but had no
   branch in handle() at all, prefixed or not — always fell through to
   the same blank response.

3. search() always returned a non-empty string, including
   "No photos found matching that." on zero matches. handle()'s
   `result or "No matching media found."` / `0.85 if result else 0.3`
   logic assumed search() returns something falsy when nothing matched,
   so every zero-result search was reported at 0.85 confidence —
   indistinguishable from a real match.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.iris.iris_engine import IrisEngine


def make_engine():
    return IrisEngine()


# ---------------------------------------------------------------------------
# Bug 1 + 2: every declared intent must actually be handled, in both the
# stripped form (what orchestrator normally passes) and the prefixed form
# (what Hecate's Tier-1.5 sees before stripping).
# ---------------------------------------------------------------------------

def test_search_stripped_and_prefixed_both_produce_a_response():
    iris = make_engine()
    for intent in ("search", "iris_search", "query", "iris_query"):
        r = iris.handle(intent, {"raw_query": "some query"}, {})
        assert r["response"], f"blank response for intent={intent!r}"


def test_ingest_stripped_and_prefixed_both_produce_a_response():
    iris = make_engine()
    for intent in ("ingest", "iris_ingest"):
        r = iris.handle(intent, {}, {})
        assert r["response"], f"blank response for intent={intent!r}"
        assert r["confidence"] == 1.0


def test_analyse_stripped_and_prefixed_both_produce_a_response():
    iris = make_engine()
    for intent in ("analyse", "iris_analyse"):
        r = iris.handle(intent, {}, {})
        assert r["response"], f"blank response for intent={intent!r}"


def test_status_stripped_and_prefixed_both_produce_a_response():
    iris = make_engine()
    for intent in ("status", "iris_status"):
        r = iris.handle(intent, {}, {})
        assert r["response"], f"blank response for intent={intent!r}"
        assert "Iris" in r["response"]


def test_unknown_intent_still_falls_through_gracefully():
    iris = make_engine()
    r = iris.handle("totally_unknown_intent", {}, {})
    assert r["confidence"] == 0.0
    assert r["response"]  # non-blank explanatory message, not silently empty


# ---------------------------------------------------------------------------
# Bug 3: confidence must reflect whether anything was actually found.
# ---------------------------------------------------------------------------

def test_zero_results_reported_at_low_confidence():
    iris = make_engine()
    r = iris.handle("search", {"raw_query": "some gibberish query that wont match xyz123"}, {})
    assert r["confidence"] == 0.3
    assert r["response"] == "No matching media found."


if __name__ == "__main__":
    failures = []
    tests = [
        (name, fn) for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures.append(name)
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        sys.exit(1)