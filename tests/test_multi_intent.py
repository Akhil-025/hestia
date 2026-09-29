# tests/test_multi_intent.py
"""
Tests for Hestia._try_multi_intent (backlog #13) — the NLU-backed
acceptance/rejection half of multi-intent splitting. core/query_splitter.py
finds candidate split points; this is what decides whether a candidate is
actually acted on.

Built the same way as tests/test_main.py: object.__new__(main.Hestia) plus
hand-wired mocks for exactly the attributes _try_multi_intent touches
(mnemosyne, nlu, orchestrator, diagnostics), since Hestia.__init__ boots
the whole app and is not meant to be unit-tested directly.
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main


def make_hestia():
    h = object.__new__(main.Hestia)
    h.mnemosyne = MagicMock()
    h.mnemosyne.get_recent.return_value = []
    h.nlu = MagicMock()
    h.orchestrator = MagicMock()
    h.diagnostics = MagicMock()
    h.orchestrator.last_decision = {"primary": "apollo", "reason": "registry"}
    return h


def nlu_result(intent, confidence=0.9):
    return {"intent": intent, "entities": {}, "response": "", "confidence": confidence}


# ---------------------------------------------------------------------------
# Rejects — falls through to single-query handling
# ---------------------------------------------------------------------------

def test_non_compound_query_returns_none_without_calling_nlu():
    h = make_hestia()
    assert h._try_multi_intent("what's the weather today") is None
    h.nlu.understand.assert_not_called()


def test_both_segments_chat_is_rejected():
    h = make_hestia()
    h.nlu.understand.side_effect = [nlu_result("chat"), nlu_result("chat")]
    assert h._try_multi_intent("log my workout and how I felt about it") is None
    h.orchestrator.dispatch.assert_not_called()


def test_one_segment_chat_is_rejected():
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("apollo_log_workout"), nlu_result("chat"),
    ]
    assert h._try_multi_intent("log my workout and how I felt about it") is None
    h.orchestrator.dispatch.assert_not_called()


def test_identical_intents_on_both_sides_is_rejected():
    # Same intent twice reads as a bad split (one request the splitter cut
    # in half), not two independent ones.
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("apollo_log_workout"), nlu_result("apollo_log_workout"),
    ]
    assert h._try_multi_intent("log my workout and log my second workout") is None


def test_unregistered_intent_is_rejected():
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("apollo_log_workout"), nlu_result("totally_made_up_intent"),
    ]
    assert h._try_multi_intent("log my workout and do the unregistered thing") is None


def test_nlu_exception_on_either_segment_falls_through_safely():
    h = make_hestia()
    h.nlu.understand.side_effect = RuntimeError("ollama down")
    assert h._try_multi_intent("log my workout and tell me the weather") is None
    h.orchestrator.dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# Accepts — genuinely different, concrete, registered intents
# ---------------------------------------------------------------------------

def test_two_distinct_registered_intents_are_both_dispatched():
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("apollo_log_workout"), nlu_result("get_weather"),
    ]
    h.orchestrator.dispatch.side_effect = ["Workout logged.", "It's sunny."]
    h._log_routing = MagicMock()

    result = h._try_multi_intent("log my workout and tell me the weather")

    assert result is not None
    response, representative = result
    assert "Workout logged." in response
    assert "It's sunny." in response
    assert h.orchestrator.dispatch.call_count == 2


def test_committed_split_dispatches_each_segment_with_its_own_nlu_result():
    h = make_hestia()
    apollo_result = nlu_result("apollo_log_workout")
    weather_result = nlu_result("get_weather")
    h.nlu.understand.side_effect = [apollo_result, weather_result]
    h.orchestrator.dispatch.side_effect = ["a", "b"]
    h._log_routing = MagicMock()

    h._try_multi_intent("log my workout and tell me the weather")

    calls = h.orchestrator.dispatch.call_args_list
    assert calls[0].args[0] == "log my workout"
    assert calls[0].args[1] == apollo_result
    assert calls[1].args[0] == "tell me the weather"
    assert calls[1].args[1] == weather_result


def test_committed_split_logs_each_segment_individually():
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("apollo_log_workout"), nlu_result("get_weather"),
    ]
    h.orchestrator.dispatch.side_effect = ["a", "b"]
    h._log_routing = MagicMock()

    h._try_multi_intent("log my workout and tell me the weather")

    assert h._log_routing.call_count == 2


def test_representative_result_is_marked_multi_intent_and_never_dispatched_again():
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("apollo_log_workout", 0.9), nlu_result("get_weather", 0.8),
    ]
    h.orchestrator.dispatch.side_effect = ["a", "b"]
    h._log_routing = MagicMock()

    _, representative = h._try_multi_intent("log my workout and tell me the weather")

    assert representative["source"] == "multi_intent"
    assert representative["intent"] == "multi_intent"
    # The lower of the two, so a genuinely uncertain half still shows up
    # as uncertain in the combined record rather than being hidden by an
    # average.
    assert representative["confidence"] == 0.8


def test_a_segment_dispatch_failure_does_not_abort_the_other_segment():
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("apollo_log_workout"), nlu_result("get_weather"),
    ]
    h.orchestrator.dispatch.side_effect = [RuntimeError("boom"), "It's sunny."]
    h._log_routing = MagicMock()

    response, _ = h._try_multi_intent("log my workout and tell me the weather")

    assert "It's sunny." in response
    assert h.orchestrator.dispatch.call_count == 2


# ---------------------------------------------------------------------------
# Intent chaining (backlog #24) — segment 2 refers back to segment 1's
# result rather than carrying its own content.
# ---------------------------------------------------------------------------

def test_anaphoric_second_segment_is_chained_with_first_segments_response():
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("athena_search"), nlu_result("take_note"),
    ]
    h.orchestrator.dispatch.side_effect = [
        "The paper argues X causes Y under condition Z.", "Noted.",
    ]
    h._log_routing = MagicMock()

    response, _ = h._try_multi_intent(
        "summarize this paper and add it to my reading list"
    )

    assert "Noted." in response
    second_call_entities = h.orchestrator.dispatch.call_args_list[1].args[1]["entities"]
    assert second_call_entities["content"] == "The paper argues X causes Y under condition Z."


def test_non_anaphoric_second_segment_is_not_chained():
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("apollo_log_workout"), nlu_result("take_note"),
    ]
    h.orchestrator.dispatch.side_effect = ["Workout logged.", "Noted."]
    h._log_routing = MagicMock()

    h._try_multi_intent("log my workout and take a note that milk is expensive")

    second_call_entities = h.orchestrator.dispatch.call_args_list[1].args[1]["entities"]
    # No chaining happened — segment 2's own (empty, in this fake) entities
    # pass through untouched rather than being overwritten with segment 1's
    # response.
    assert "content" not in second_call_entities


def test_chaining_only_applies_to_verified_chainable_targets():
    # get_weather isn't in CHAINABLE_TARGETS — even an anaphoric-looking
    # second segment must not have entities force-fed into an unverified
    # handler's shape.
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("athena_search"), nlu_result("get_weather"),
    ]
    h.orchestrator.dispatch.side_effect = ["A summary.", "It's sunny."]
    h._log_routing = MagicMock()

    h._try_multi_intent("summarize this and add it to the weather")

    second_call_entities = h.orchestrator.dispatch.call_args_list[1].args[1]["entities"]
    assert second_call_entities == {}


def test_chained_dispatch_still_logs_both_segments():
    h = make_hestia()
    h.nlu.understand.side_effect = [
        nlu_result("athena_search"), nlu_result("take_note"),
    ]
    h.orchestrator.dispatch.side_effect = ["A summary.", "Noted."]
    h._log_routing = MagicMock()

    h._try_multi_intent("summarize this and add it to my reading list")

    assert h._log_routing.call_count == 2
