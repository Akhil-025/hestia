# tests/test_entity_confidence.py
"""
Tests for core/entity_confidence.py (backlog #23).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.entity_confidence import (
    lowest_confidence_entity,
    score_entities,
)


# ---------------------------------------------------------------------------
# Email-shaped entities
# ---------------------------------------------------------------------------

def test_valid_email_scores_high():
    scores = score_entities({"to": "raj@example.com"})
    assert scores["to"] >= 0.9


def test_invalid_email_scores_low():
    scores = score_entities({"to": "not an email"})
    assert scores["to"] < 0.5


def test_empty_email_scores_zero():
    scores = score_entities({"to": ""})
    assert scores["to"] == 0.0


def test_recipient_key_variant_is_treated_as_email():
    assert score_entities({"recipient": "a@b.com"})["recipient"] >= 0.9


def test_cc_key_variant_is_treated_as_email():
    assert score_entities({"cc": "team@example.com"})["cc"] >= 0.9


# ---------------------------------------------------------------------------
# Date/time-shaped entities
# ---------------------------------------------------------------------------

def test_parseable_date_scores_reasonably_high():
    # "tomorrow" rather than "next Friday": dateparser's relative-weekday
    # resolution ("next <weekday>") is unreliable across installs/versions
    # without the extra settings modules/chronos/engine.py's own date
    # parsing already has to apply for exactly that reason — this scorer
    # is a lightweight heuristic, not a reimplementation of Chronos's date
    # handling, so it's correct for it to score a phrase dateparser itself
    # can't resolve as low-confidence rather than guessing.
    scores = score_entities({"date": "tomorrow"})
    assert scores["date"] >= 0.7


def test_unparseable_date_scores_low():
    scores = score_entities({"date": "asdkjfh qwerty"})
    assert scores["date"] < 0.5


def test_empty_date_scores_zero():
    scores = score_entities({"date": ""})
    assert scores["date"] == 0.0


def test_deadline_key_variant_is_treated_as_date():
    scores = score_entities({"deadline": "tomorrow"})
    assert scores["deadline"] >= 0.7


def test_a_concrete_date_string_parses_confidently():
    scores = score_entities({"due": "2026-03-15"})
    assert scores["due"] >= 0.7


# ---------------------------------------------------------------------------
# Amount/numeric-shaped entities
# ---------------------------------------------------------------------------

def test_amount_present_in_source_text_scores_high():
    scores = score_entities({"amount": "200"}, raw_query="I spent 200 on lunch")
    assert scores["amount"] >= 0.85


def test_amount_not_in_source_text_scores_moderately():
    scores = score_entities({"amount": "200"}, raw_query="I spent some money")
    assert 0.4 <= scores["amount"] < 0.85


def test_non_numeric_amount_scores_low():
    scores = score_entities({"amount": "a lot"}, raw_query="I spent a lot")
    assert scores["amount"] < 0.5


def test_empty_amount_scores_zero():
    scores = score_entities({"amount": ""})
    assert scores["amount"] == 0.0


def test_hours_key_variant_is_treated_as_amount():
    scores = score_entities({"hours": "7"}, raw_query="I slept 7 hours")
    assert scores["hours"] >= 0.85


def test_none_amount_value_does_not_raise():
    scores = score_entities({"amount": None})
    assert scores["amount"] == 0.0


# ---------------------------------------------------------------------------
# Generic free-text entities
# ---------------------------------------------------------------------------

def test_reasonable_free_text_scores_confidently():
    scores = score_entities({"body": "running ten minutes late"})
    assert scores["body"] >= 0.7


def test_empty_free_text_scores_zero():
    scores = score_entities({"title": ""})
    assert scores["title"] == 0.0


def test_very_short_free_text_scores_low_not_zero():
    scores = score_entities({"note": "x"})
    assert 0.0 < scores["note"] < 0.5


def test_none_free_text_value_does_not_raise():
    scores = score_entities({"note": None})
    assert scores["note"] == 0.0


# ---------------------------------------------------------------------------
# Whole-dict behavior
# ---------------------------------------------------------------------------

def test_empty_entities_dict_returns_empty_scores():
    assert score_entities({}) == {}


def test_none_entities_does_not_raise():
    assert score_entities(None) == {}


def test_every_entity_key_gets_a_score():
    entities = {"to": "a@b.com", "body": "hi", "amount": "10"}
    scores = score_entities(entities, raw_query="hi, 10")
    assert set(scores) == set(entities)


def test_internal_bookkeeping_keys_are_skipped():
    scores = score_entities({"to": "a@b.com", "_confirmed": True})
    assert "_confirmed" not in scores
    assert "to" in scores


def test_a_bad_scorer_input_falls_back_rather_than_raising():
    # An unexpected value type (a list) for an amount-shaped key must not
    # blow up the whole call.
    scores = score_entities({"amount": [1, 2, 3]})
    assert "amount" in scores
    assert 0.0 <= scores["amount"] <= 1.0


def test_all_scores_are_within_zero_to_one():
    entities = {
        "to": "raj@example.com", "date": "Friday", "amount": "42",
        "body": "hello there", "junk": "###???",
    }
    scores = score_entities(entities, raw_query="42")
    assert all(0.0 <= v <= 1.0 for v in scores.values())


# ---------------------------------------------------------------------------
# lowest_confidence_entity
# ---------------------------------------------------------------------------

def test_lowest_confidence_entity_picks_the_minimum():
    scores = {"to": 0.9, "body": 0.3, "subject": 0.6}
    assert lowest_confidence_entity(scores) == ("body", 0.3)


def test_lowest_confidence_entity_of_empty_dict_is_none():
    assert lowest_confidence_entity({}) is None


def test_lowest_confidence_entity_with_a_single_entry():
    assert lowest_confidence_entity({"to": 0.9}) == ("to", 0.9)
