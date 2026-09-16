# tests/test_intent_aliases.py
"""
Tests for core/intent_aliases.py (backlog #22).

Two things matter here beyond "does matching work":

1. The registry stays the source of truth. An alias pointing at an intent
   that isn't in intent_registry.py must be dropped, not loaded — otherwise
   the alias layer becomes a second, silently-diverging definition of what
   intents exist, which is the exact drift the registry was introduced to
   kill.

2. The shipped config/intent_aliases.yaml is validated here, so adding a
   phrase for a misspelled intent fails CI rather than silently doing
   nothing at runtime.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.intent_aliases import IntentAliasResolver, normalise
from modules.hecate.intent_registry import ALL_INTENTS

_ALIAS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "intent_aliases.yaml"
)


def make(aliases):
    return IntentAliasResolver(path=None, aliases=aliases)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_normalise_lowercases_and_collapses_whitespace():
    assert normalise("  LOG   my   Sleep ") == "log my sleep"


def test_normalise_strips_punctuation():
    assert normalise("that's wrong!") == "that s wrong"


def test_normalise_of_empty_input():
    assert normalise("") == ""
    assert normalise(None) == ""


# ---------------------------------------------------------------------------
# Matching modes
# ---------------------------------------------------------------------------

def test_prefix_is_the_default_mode():
    r = make({"apollo_track_sleep": ["log my sleep"]})
    assert r.resolve("log my sleep for 7 hours") == "apollo_track_sleep"


def test_prefix_requires_a_word_boundary():
    # "log my sleeping pills" must not match the "log my sleep" prefix.
    r = make({"apollo_track_sleep": ["log my sleep"]})
    assert r.resolve("log my sleeping pills order") is None


def test_prefix_matches_the_whole_query_exactly():
    r = make({"apollo_track_sleep": ["log my sleep"]})
    assert r.resolve("log my sleep") == "apollo_track_sleep"


def test_exact_mode_does_not_match_a_longer_query():
    r = make({"modules_status": [{"phrase": "module status", "match": "exact"}]})
    assert r.resolve("module status") == "modules_status"
    assert r.resolve("module status please") is None


def test_contains_mode_matches_mid_query():
    r = make({"apollo_track_sleep": [{"phrase": "hours of sleep", "match": "contains"}]})
    assert r.resolve("i got about six hours of sleep last night") == "apollo_track_sleep"


def test_contains_mode_still_requires_word_boundaries():
    r = make({"pluto_log_expense": [{"phrase": "spent", "match": "contains"}]})
    assert r.resolve("i spent 200") == "pluto_log_expense"
    assert r.resolve("unspentbudget remains") is None


def test_no_match_returns_none():
    r = make({"apollo_track_sleep": ["log my sleep"]})
    assert r.resolve("what's the weather like") is None


def test_empty_query_returns_none():
    r = make({"apollo_track_sleep": ["log my sleep"]})
    assert r.resolve("") is None
    assert r.resolve("   ") is None


def test_punctuation_in_the_query_does_not_prevent_a_match():
    r = make({"report_mistake": [{"phrase": "that's wrong", "match": "exact"}]})
    assert r.resolve("That's wrong!") == "report_mistake"


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

def test_exact_beats_prefix():
    r = make({
        "modules_status": [{"phrase": "system status", "match": "exact"}],
        "get_system_info": ["system status"],
    })
    assert r.resolve("system status") == "modules_status"


def test_longer_prefix_wins_over_shorter():
    # Otherwise the order of keys in the YAML file silently decides which
    # of two overlapping phrases wins.
    r = make({
        "apollo_track_sleep": ["log my sleep"],
        "apollo_log_mood": ["log my sleep quality mood"],
    })
    assert r.resolve("log my sleep quality mood today") == "apollo_log_mood"


def test_prefix_beats_contains():
    r = make({
        "pluto_log_expense": ["i spent"],
        "pluto_spending_report": [{"phrase": "spent", "match": "contains"}],
    })
    assert r.resolve("i spent 400 on groceries") == "pluto_log_expense"


# ---------------------------------------------------------------------------
# Registry is the source of truth
# ---------------------------------------------------------------------------

def test_alias_for_an_unregistered_intent_is_dropped():
    r = make({"not_a_real_intent": ["do the thing"]})
    assert r.resolve("do the thing") is None
    assert "not_a_real_intent" in r.dropped
    assert r.count == 0


def test_valid_aliases_survive_alongside_a_dropped_one():
    r = make({
        "not_a_real_intent": ["do the thing"],
        "apollo_track_sleep": ["log my sleep"],
    })
    assert r.resolve("log my sleep") == "apollo_track_sleep"
    assert r.count == 1


def test_resolver_can_be_scoped_to_a_custom_intent_set():
    r = IntentAliasResolver(
        path=None,
        aliases={"custom_intent": ["do x"]},
        valid_intents=frozenset({"custom_intent"}),
    )
    assert r.resolve("do x") == "custom_intent"


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

def test_malformed_entries_are_skipped_without_raising():
    r = make({
        "apollo_track_sleep": [
            "log my sleep",
            "",                       # empty phrase
            {"match": "prefix"},      # no phrase key
            42,                       # wrong type entirely
            None,
        ],
        123: ["nonsense key"],        # non-string intent
    })
    assert r.resolve("log my sleep") == "apollo_track_sleep"
    assert r.count == 1


def test_a_bare_string_instead_of_a_list_is_accepted():
    r = make({"apollo_track_sleep": "log my sleep"})
    assert r.resolve("log my sleep") == "apollo_track_sleep"


def test_unknown_match_mode_falls_back_to_prefix():
    r = make({"apollo_track_sleep": [{"phrase": "log my sleep", "match": "fuzzy"}]})
    assert r.resolve("log my sleep tonight") == "apollo_track_sleep"


def test_missing_alias_file_disables_the_layer_silently():
    r = IntentAliasResolver(path="config/does_not_exist.yaml")
    assert r.count == 0
    assert r.resolve("anything at all") is None


def test_non_mapping_alias_file_is_rejected(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    assert IntentAliasResolver(path=path).count == 0


def test_unparseable_alias_file_does_not_raise(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text("key: [unclosed\n", encoding="utf-8")
    assert IntentAliasResolver(path=path).count == 0


# ---------------------------------------------------------------------------
# NLU-shaped result
# ---------------------------------------------------------------------------

def test_resolve_result_matches_the_nlu_result_shape():
    r = make({"apollo_track_sleep": ["log my sleep"]})
    result = r.resolve_result("log my sleep for 7 hours")
    assert set(result) >= {"intent", "entities", "response", "confidence"}
    assert result["intent"] == "apollo_track_sleep"
    assert result["entities"] == {}       # aliases never extract entities
    assert result["source"] == "alias"


def test_resolve_result_confidence_clears_hecates_high_confidence_bar():
    # Hecate treats >= 0.85 as high confidence; an alias hit must not land
    # below that or it would be forced to the chat fallback.
    r = make({"apollo_track_sleep": ["log my sleep"]})
    assert r.resolve_result("log my sleep")["confidence"] >= 0.85


def test_resolve_result_returns_none_on_miss():
    r = make({"apollo_track_sleep": ["log my sleep"]})
    assert r.resolve_result("what's the weather") is None


# ---------------------------------------------------------------------------
# The shipped alias file
# ---------------------------------------------------------------------------

def test_shipped_alias_file_loads_and_is_non_empty():
    r = IntentAliasResolver(path=_ALIAS_PATH)
    assert r.count > 0


def test_shipped_alias_file_targets_only_registered_intents():
    # This is the test that fails when someone adds a phrase under a
    # misspelled intent name.
    with open(_ALIAS_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    unknown = sorted(set(data) - set(ALL_INTENTS))
    assert not unknown, f"unregistered intents in intent_aliases.yaml: {unknown}"


def test_shipped_alias_file_has_no_duplicate_phrases_across_intents():
    # The same phrase under two intents means one of them can never fire,
    # and which one is decided by dict ordering.
    with open(_ALIAS_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for intent, entries in data.items():
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            phrase = entry if isinstance(entry, str) else entry.get("phrase", "")
            key = normalise(str(phrase))
            if key in seen and seen[key] != intent:
                duplicates.append(f"{key!r} ({seen[key]} and {intent})")
            seen[key] = intent
    assert not duplicates, f"duplicate alias phrases: {duplicates}"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("log my sleep for 8 hours", "apollo_track_sleep"),
        ("i spent 250 on dinner", "pluto_log_expense"),
        ("remind me to call mom at 6", "set_reminder"),
        ("list my habits", "list_habits"),
        ("module status", "modules_status"),
        ("that was wrong", "report_mistake"),
        ("why did you route that to pluto", "explain_routing"),
    ],
)
def test_shipped_aliases_resolve_representative_phrasings(query, expected):
    assert IntentAliasResolver(path=_ALIAS_PATH).resolve(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "what's the weather in mumbai",
        "write me a poem about heat transfer",
        "how are you doing today",
        "search my documents for rankine cycle efficiency",
    ],
)
def test_shipped_aliases_do_not_hijack_unrelated_queries(query):
    # A false alias hit is worse than a miss: it skips the LLM entirely,
    # so there's no second chance to classify correctly.
    assert IntentAliasResolver(path=_ALIAS_PATH).resolve(query) is None
