# tests/test_query_splitter.py
"""
Tests for core/query_splitter.py (backlog #13).

This module deliberately only finds CANDIDATE split points — the actual
"are these two real, different requests" decision is made by the caller
via real NLU classification (see tests/test_multi_intent.py). What's
tested here is purely the string-level heuristic: does it find a
plausible split, and does it correctly decline to split short fragments
and common non-command "X and Y" phrases.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.query_splitter import candidate_segments, looks_compound


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------

def test_splits_on_plain_and_when_both_sides_look_verby():
    segments = candidate_segments("log my workout and tell me the weather")
    assert segments == ["log my workout", "tell me the weather"]


def test_splits_on_and_also():
    segments = candidate_segments("remind me to call mom and also check my email")
    assert segments == ["remind me to call mom", "check my email"]


def test_splits_on_and_then():
    segments = candidate_segments("log my sleep and then give me a health summary")
    assert segments == ["log my sleep", "give me a health summary"]


def test_splits_on_semicolon():
    segments = candidate_segments("what's the weather; log my workout")
    assert segments == ["what's the weather", "log my workout"]


def test_and_also_takes_priority_over_bare_and():
    # If both connectors are present, the more explicit one should win
    # and produce a clean two-way split rather than a three-way mess.
    segments = candidate_segments("log my workout and also tell me the weather and time")
    assert segments == ["log my workout", "tell me the weather and time"]


def test_looks_compound_true_for_a_splittable_query():
    assert looks_compound("log my workout and tell me the weather") is True


# ---------------------------------------------------------------------------
# Negative cases — must NOT split
# ---------------------------------------------------------------------------

def test_does_not_split_a_plain_single_request():
    assert candidate_segments("what's the weather today") == []


def test_does_not_split_short_noun_list_and():
    # Neither side looks like its own request.
    assert candidate_segments("mac and cheese") == []
    assert candidate_segments("salt and pepper") == []
    assert candidate_segments("bacon and eggs") == []


def test_does_not_split_when_one_side_is_too_short():
    assert candidate_segments("log my workout and go") == []


def test_a_single_clause_with_and_is_still_offered_as_a_candidate():
    # The splitter deliberately over-generates — see its module docstring.
    # "how I felt about it" isn't a real second request, but string-level
    # matching alone can't know that; it's candidate_segments' job only to
    # find a plausible split point. The caller (_try_multi_intent) is what
    # rejects this, by classifying "how I felt about it" and finding it
    # resolves to plain "chat" rather than a concrete registered intent —
    # see tests/test_multi_intent.py for that half of the guarantee.
    segments = candidate_segments("log my workout and how I felt about it")
    assert segments == ["log my workout", "how I felt about it"]


def test_empty_and_whitespace_input():
    assert candidate_segments("") == []
    assert candidate_segments("   ") == []
    assert candidate_segments(None) == []


def test_looks_compound_false_for_a_non_splittable_query():
    assert looks_compound("what's the weather today") is False


# ---------------------------------------------------------------------------
# Segments are trimmed
# ---------------------------------------------------------------------------

def test_segments_are_stripped_of_surrounding_whitespace():
    segments = candidate_segments("  log my workout   and   tell me the weather  ")
    assert segments == ["log my workout", "tell me the weather"]


def test_split_is_case_insensitive_on_the_connector():
    segments = candidate_segments("log my workout AND tell me the weather")
    assert len(segments) == 2
    assert segments[0] == "log my workout"
