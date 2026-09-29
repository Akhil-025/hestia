# tests/test_intent_chains.py
"""
Tests for core/intent_chains.py (backlog #24) — detecting when the second
half of a split compound query refers back to the first half's result
("add it to my reading list") rather than carrying its own content.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.intent_chains import CHAINABLE_TARGETS, apply_chain, detect_chain_reference


# ---------------------------------------------------------------------------
# detect_chain_reference
# ---------------------------------------------------------------------------

def test_add_it_is_detected():
    assert detect_chain_reference("add it to my reading list", "take_note") == "content"


def test_save_that_is_detected():
    assert detect_chain_reference("save that", "take_note") == "content"


def test_note_it_down_is_detected():
    assert detect_chain_reference("note it down", "take_note") == "content"


def test_remember_this_is_detected():
    assert detect_chain_reference("remember this for later", "take_note") == "content"


def test_track_it_is_detected():
    assert detect_chain_reference("track it", "take_note") == "content"


def test_case_insensitive():
    assert detect_chain_reference("ADD IT to my list", "take_note") == "content"


def test_content_bearing_segment_is_not_detected():
    # Carries its own content — not a reference to anything preceding it.
    assert detect_chain_reference("take a note that milk is expensive", "take_note") is None


def test_mid_sentence_it_is_not_enough_on_its_own():
    # Only anchored-at-start references count — a mid-sentence "it" is too
    # weak a signal ("check if it works" is not a chain reference).
    assert detect_chain_reference("check if it still works", "take_note") is None


def test_unrelated_segment_is_not_detected():
    assert detect_chain_reference("what's the weather today", "take_note") is None


def test_unverified_target_intent_never_chains():
    # add_goal isn't in CHAINABLE_TARGETS (its entity shape hasn't been
    # verified against the real handler) — must return None even though
    # the text itself would otherwise match.
    assert detect_chain_reference("add it to my goals", "add_goal") is None


def test_empty_segment_is_not_detected():
    assert detect_chain_reference("", "take_note") is None
    assert detect_chain_reference("   ", "take_note") is None


def test_none_segment_does_not_raise():
    assert detect_chain_reference(None, "take_note") is None


def test_leading_whitespace_is_tolerated():
    assert detect_chain_reference("  add it please", "take_note") == "content"


# ---------------------------------------------------------------------------
# apply_chain
# ---------------------------------------------------------------------------

def test_apply_chain_sets_the_key():
    result = apply_chain({}, "content", "the piped-in summary text")
    assert result == {"content": "the piped-in summary text"}


def test_apply_chain_overwrites_an_existing_value():
    # The whole point: whatever the NLU extracted for "add it" is at best
    # a restatement of "it", not real content.
    result = apply_chain({"content": "it"}, "content", "the real summary")
    assert result["content"] == "the real summary"


def test_apply_chain_preserves_other_entities():
    result = apply_chain({"tag": "reading"}, "content", "summary text")
    assert result == {"tag": "reading", "content": "summary text"}


def test_apply_chain_does_not_mutate_the_input_dict():
    original = {"tag": "reading"}
    apply_chain(original, "content", "summary text")
    assert original == {"tag": "reading"}


def test_apply_chain_with_none_entities():
    result = apply_chain(None, "content", "summary text")
    assert result == {"content": "summary text"}


# ---------------------------------------------------------------------------
# CHAINABLE_TARGETS shape
# ---------------------------------------------------------------------------

def test_take_note_is_the_verified_chainable_target():
    assert CHAINABLE_TARGETS.get("take_note") == "content"
