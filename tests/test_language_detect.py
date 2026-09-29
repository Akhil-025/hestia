# tests/test_language_detect.py
"""
Tests for core/language_detect.py (backlog #27) — deterministic Unicode
script classification, not language identification (see the module
docstring for that distinction).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.language_detect import detect_script


# ---------------------------------------------------------------------------
# Pure scripts
# ---------------------------------------------------------------------------

def test_plain_english_is_latin():
    assert detect_script("what's the weather today") == "latin"


def test_hinglish_written_in_roman_script_is_latin():
    # Script detection can't and doesn't try to distinguish this from
    # English — see the module docstring.
    assert detect_script("mera sleep log karo") == "latin"


def test_pure_devanagari_hindi_is_devanagari():
    assert detect_script("आज मौसम कैसा है") == "devanagari"


def test_pure_devanagari_single_word():
    assert detect_script("नमस्ते") == "devanagari"


# ---------------------------------------------------------------------------
# Mixed / code-switched
# ---------------------------------------------------------------------------

def test_code_switched_query_is_mixed():
    assert detect_script("मुझे 7 baje remind karo") == "mixed"


def test_devanagari_with_a_single_latin_word_is_mixed():
    assert detect_script("आज weather कैसा है") == "mixed"


# ---------------------------------------------------------------------------
# Neither script
# ---------------------------------------------------------------------------

def test_numbers_only_is_other():
    assert detect_script("12345") == "other"


def test_punctuation_only_is_other():
    assert detect_script("???!!!") == "other"


def test_empty_string_is_other():
    assert detect_script("") == "other"


def test_none_input_is_other():
    assert detect_script(None) == "other"


def test_whitespace_only_is_other():
    assert detect_script("   ") == "other"


# ---------------------------------------------------------------------------
# Real query shapes
# ---------------------------------------------------------------------------

def test_a_realistic_hinglish_sentence_with_numbers():
    assert detect_script("kal 6 baje reminder lagao") == "latin"


def test_english_query_with_embedded_numbers_stays_latin():
    assert detect_script("I spent 200 rupees today") == "latin"
