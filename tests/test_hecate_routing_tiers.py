# tests/test_hecate_routing_tiers.py
"""
Pins every routing tier in HecateEngine.decide() (backlog #214).

Why this file exists: scripts/mutation_check.py was run against
modules/hecate/engine.py with the existing tests (test_hecate.py and
test_registry_contract.py). Only 40% of the deliberate breakages (flipped
confidence thresholds, `and` swapped for `or`, a trigger list that stops
matching) made a test fail, so most of the tier logic was effectively
untested: those tests passed whether or not the code was right. These tests
assert the actual decisions, tier by tier, boundary by boundary.

Re-check with:
    python scripts/mutation_check.py modules/hecate/engine.py \\
        --tests tests/test_hecate.py tests/test_registry_contract.py tests/test_hecate_routing_tiers.py

The one mutant that remains is equivalent: the `athena trigger AND mnemosyne
trigger` clause in the cross-module tier can never decide anything, because an
athena trigger is always taken by Tier 2 first (see
test_cross_module_clause_is_shadowed_by_tier_2).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.hecate import HecateEngine
from modules.hecate.intent_registry import INTENT_MODULE_MAP

ALL = ["core", "mnemosyne", "athena", "iris", "artemis", "hermes", "pluto", "chronos"]
H = HecateEngine()


def decide(query, intent="chat", confidence=0.7, entities=None, active=None):
    nlu = {"intent": intent, "confidence": confidence}
    if entities is not None:
        nlu["entities"] = entities
    return H.decide(query, nlu, ALL if active is None else active)


class TestInterface:
    def test_can_handle_everything(self):
        assert H.can_handle("anything_at_all") is True and H.can_handle("") is True

    def test_get_context_is_an_empty_dict(self):
        assert H.get_context() == {}

    def test_handle_must_not_be_called(self):
        with pytest.raises(NotImplementedError):
            H.handle("x", {}, {})

    def test_decision_shape_and_defaults(self):
        d = decide("hello there")
        assert set(d) == {"primary", "secondary", "confidence", "reason", "synthesize", "intent"}
        assert d["synthesize"] is False and d["intent"] is None and d["secondary"] == []

    def test_missing_confidence_means_exactly_one_half(self):
        d = H.decide("hello there", {"intent": "chat"}, ALL)
        assert d["confidence"] == 0.5 and d["reason"] == "default core"

    @pytest.mark.parametrize("bad", [None, "high", "", float("nan"), [], {}])
    def test_unusable_confidence_does_not_crash_and_means_one_half(self, bad):
        d = H.decide("hello there", {"intent": "chat", "confidence": bad}, ALL)
        assert d["primary"] == "core" and d["confidence"] == 0.5

    def test_numeric_string_confidence_is_accepted(self):
        assert H.decide("hello", {"intent": "chat", "confidence": "0.3"}, ALL)["confidence"] == 0.4


class TestIntentNormalisation:
    def test_intent_case_and_whitespace_are_normalised(self):
        d = decide("any mail", intent="  READ_EMAIL ", confidence=0.95)
        assert d["primary"] == "hermes" and d["intent"] == "read_email"

    def test_non_string_intent_becomes_chat(self):
        assert decide("hello", intent=None)["reason"] == "default core"
        assert decide("hello", intent=42)["reason"] == "default core"

    def test_platform_action_envelope_is_unwrapped(self):
        d = decide("check mail", intent="PLATFORM_ACTION", confidence=0.95,
                   entities={"action": "READ_EMAIL"})
        assert d["primary"] == "hermes" and "read_email" in d["reason"]

    @pytest.mark.parametrize("entities", [None, {}, {"action": ""}, {"action": "   "}, {"action": 5}])
    def test_envelope_without_a_usable_action_is_not_unwrapped(self, entities):
        d = decide("hello", intent="platform_action", confidence=0.95, entities=entities)
        assert d["primary"] == "core" and "registry" not in d["reason"]

    def test_only_the_platform_action_envelope_is_unwrapped(self):
        d = decide("hello", intent="read_email", confidence=0.95, entities={"action": "set_reminder"})
        assert d["primary"] == "hermes"


class TestClarification:
    def test_uncertain_registered_intent_asks_instead_of_acting(self):
        d = decide("send it", intent="send_email", confidence=0.44)
        assert d["primary"] == "core" and d["intent"] == "clarify_intent"
        assert "send_email" in d["reason"] and "hermes" in d["reason"]

    def test_threshold_is_exclusive(self):
        assert decide("send it", intent="send_email", confidence=0.45)["primary"] == "hermes"

    def test_chat_is_never_clarified(self):
        assert decide("hm", intent="chat", confidence=0.1)["intent"] is None

    def test_unregistered_intent_is_not_clarified(self):
        d = decide("hm", intent="totally_made_up", confidence=0.1)
        assert d["intent"] is None and d["reason"] == "low confidence -> chat fallback"

    def test_clarification_keeps_the_given_confidence(self):
        assert decide("x", intent="send_email", confidence=0.2)["confidence"] == 0.2


class TestRegistry:
    def test_routes_to_the_registered_module_with_a_confidence_floor(self):
        d = decide("any mail", intent="read_email", confidence=0.6)
        assert d["primary"] == "hermes" and d["confidence"] == 0.9
        assert d["intent"] == "read_email" and d["synthesize"] is False

    def test_high_nlu_confidence_is_kept_above_the_floor(self):
        assert decide("any mail", intent="read_email", confidence=0.97)["confidence"] == 0.97

    def test_module_prefix_is_stripped_from_the_dispatched_intent(self):
        d = decide("log water", intent="apollo_log_water", confidence=0.9, active=["core", "apollo"])
        assert d["primary"] == "apollo" and d["intent"] == "log_water"

    def test_inactive_module_falls_through_instead_of_misrouting(self):
        d = decide("any mail", intent="read_email", confidence=0.95, active=["core"])
        assert d["primary"] == "core" and "registry" not in d["reason"]

    def test_chat_is_excluded_so_text_triggers_still_run(self):
        assert INTENT_MODULE_MAP["chat"] == "core"
        d = decide("do you remember my name", intent="chat", confidence=0.95)
        assert d["primary"] == "mnemosyne"

    def test_every_registered_intent_reaches_its_module_when_active(self):
        for intent, module in INTENT_MODULE_MAP.items():
            if intent == "chat":
                continue
            d = H.decide("zz", {"intent": intent, "confidence": 0.95}, ALL + [module])
            assert d["primary"] == module, intent


class TestTextTriggers:
    def test_athena_ingest(self):
        d = decide("please ingest my documents now")
        assert (d["primary"], d["intent"], d["confidence"]) == ("athena", "ingest", 1.0)

    def test_athena_ingest_needs_athena(self):
        assert decide("ingest my documents", active=["core", "mnemosyne"])["primary"] == "core"

    def test_ingest_wins_over_the_broader_search_trigger(self):
        assert decide("ingest my notes and then explain from my notes")["intent"] == "ingest"

    def test_athena_search_asks_mnemosyne_for_context_when_present(self):
        d = decide("what does my report say, from my notes")
        assert (d["primary"], d["intent"], d["confidence"]) == ("athena", "search", 1.0)
        assert d["secondary"] == ["mnemosyne"]

    def test_athena_search_without_mnemosyne_has_no_secondary(self):
        d = decide("search my documents for tax", active=["core", "athena"])
        assert d["primary"] == "athena" and d["secondary"] == []

    def test_athena_search_needs_athena(self):
        assert decide("from my notes", active=["core", "mnemosyne"])["primary"] != "athena"

    @pytest.mark.parametrize("q", [
        "what did we talk about yesterday", "What did we discuss today?",
        "what did we just talk about", "what did we talk about earlier",
    ])
    def test_recency_goes_to_core_history(self, q):
        d = decide(q)
        assert (d["primary"], d["intent"], d["confidence"]) == ("core", "get_history", 1.0)

    def test_recency_needs_core(self):
        d = decide("what did we talk about yesterday", active=["athena"])
        assert d["intent"] != "get_history" and "recency" not in d["reason"]

    def test_recency_beats_the_mnemosyne_trigger(self):
        assert decide("do you remember what did we talk about yesterday")["intent"] == "get_history"

    @pytest.mark.parametrize("q", ["do you remember my name", "remind me what I said", "what have i told you"])
    def test_mnemosyne_recall(self, q):
        d = decide(q)
        assert (d["primary"], d["intent"], d["confidence"]) == ("mnemosyne", "recall", 1.0)

    def test_remind_me_to_is_not_a_recall_question(self):
        assert decide("remind me to call mom")["primary"] != "mnemosyne"

    def test_mnemosyne_trigger_needs_mnemosyne(self):
        assert decide("do you remember my name", active=["core"])["primary"] == "core"

    def test_iris_search_and_ingest(self):
        s = decide("find photo of the beach")
        assert (s["primary"], s["intent"], s["confidence"]) == ("iris", "search", 1.0)
        assert decide("please ingest photos")["intent"] == "ingest"
        assert decide("ingest media from my phone")["intent"] == "ingest"

    def test_iris_trigger_needs_iris(self):
        assert decide("find photo of the beach", active=["core"])["primary"] == "core"

    def test_triggers_match_whole_words_only(self):
        assert not H._match("within my notesy", ["in my notes"])
        assert H._match("look in my notes please", ["in my notes"])
        assert not H._match("anything", [])

    def test_matching_is_case_insensitive_via_decide(self):
        assert decide("FROM MY NOTES tell me")["primary"] == "athena"


class TestPrefixFallback:
    def test_unregistered_prefixed_intent_goes_to_that_module(self):
        d = decide("rebalance it", intent="pluto_rebalance", confidence=0.9)
        assert d["primary"] == "pluto" and d["confidence"] == 0.75
        assert "prefix fallback" in d["reason"] and d["intent"] is None

    def test_inactive_prefix_module_falls_through(self):
        d = decide("rebalance it", intent="pluto_rebalance", confidence=0.9, active=["core"])
        assert d["primary"] == "core" and "prefix" not in d["reason"]


class TestKeywordsAndCrossModule:
    @pytest.mark.parametrize("q", ["how is my streak", "I need some motivation", "check my habit"])
    def test_artemis_keywords(self, q):
        d = decide(q)
        assert (d["primary"], d["confidence"], d["intent"]) == ("artemis", 0.9, None)

    def test_artemis_keyword_needs_artemis(self):
        assert decide("how is my streak", active=["core"])["primary"] == "core"

    @pytest.mark.parametrize("q", ["compare spending across months", "this along with that", "a as well as b"])
    def test_cross_module_synthesis(self, q):
        d = decide(q)
        assert (d["primary"], d["secondary"], d["confidence"]) == ("athena", ["mnemosyne"], 0.85)
        assert d["synthesize"] is True and d["intent"] == "search"

    def test_cross_module_needs_both_athena_and_mnemosyne(self):
        assert decide("compare these", active=["core", "athena"])["synthesize"] is False
        assert decide("compare these", active=["core", "mnemosyne"])["synthesize"] is False

    def test_cross_module_clause_is_shadowed_by_tier_2(self):
        d = decide("from my notes, do you remember the plan")
        assert d["reason"] == "athena trigger" and d["synthesize"] is False

    def test_keyword_tier_beats_cross_module(self):
        assert decide("compare my habit streak")["primary"] == "artemis"


class TestConfidenceTiers:
    def test_high_confidence_unregistered_intent_goes_to_core_keeping_confidence(self):
        d = decide("zz", intent="mystery", confidence=0.85)
        assert d["primary"] == "core" and d["confidence"] == 0.85
        assert d["reason"] == "high-confidence intent 'mystery'"

    def test_just_under_high_confidence_is_the_default_route(self):
        d = decide("zz", intent="mystery", confidence=0.84)
        assert d["reason"] == "default core" and d["confidence"] == 0.84

    def test_high_confidence_chat_is_not_a_concrete_intent(self):
        assert decide("zz", intent="chat", confidence=0.99)["reason"] == "default core"

    def test_low_confidence_forces_chat_at_a_fixed_confidence(self):
        d = decide("zz", intent="mystery", confidence=0.3)
        assert (d["primary"], d["confidence"]) == ("core", 0.4)
        assert d["reason"] == "low confidence -> chat fallback"

    def test_half_is_not_low(self):
        assert decide("zz", intent="mystery", confidence=0.5)["reason"] == "default core"
        assert decide("zz", intent="mystery", confidence=0.4999)["reason"].startswith("low confidence")

    def test_route_helper_defaults(self):
        r = HecateEngine._route("core", [], 0.3, "why")
        assert r == {"primary": "core", "secondary": [], "confidence": 0.3, "reason": "why",
                     "synthesize": False, "intent": None}
