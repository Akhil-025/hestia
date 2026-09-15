# tests/test_nlu_core.py
"""
Regression tests for core/nlu.py (HestiaNLU).

Run with:  pytest tests/test_nlu_core.py -v

Named test_nlu_core.py rather than test_nlu.py to avoid colliding with
any module-under-test named nlu.py elsewhere in the suite.

Scope: this focuses on the parts of HestiaNLU that don't require a real
Ollama/Anthropic/Gemini backend — the deterministic fast paths, intent
validation against the shared registry, entity alias normalization, the
learn_fact textual-salvage repair, amount-string cleaning, and JSON
response parsing. These are exactly the pieces most likely to silently
regress (a fast-path regex too broad/narrow, an alias table typo, a
parser edge case) without ever calling a model.

The full `understand()` retry loop against a real/mocked LLM provider is
intentionally out of scope here — it's timing- and provider-dependent
enough to deserve its own integration-style test file.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.nlu import HestiaNLU


def make_nlu() -> HestiaNLU:
    # Nonexistent prompt path is intentional: _load_prompt() catches the
    # error and falls back to a minimal built-in prompt, so tests never
    # depend on config/nlu_prompt.txt's 41KB contents.
    return HestiaNLU(prompt_path="/nonexistent/nlu_prompt.txt")


# ---------------------------------------------------------------------------
# Fast-path intents (no LLM call at all)
# ---------------------------------------------------------------------------

def test_fast_path_get_time():
    nlu = make_nlu()
    result = nlu.understand("what time is it")
    assert result["intent"] == "get_time"
    assert result["confidence"] == 0.98


def test_fast_path_get_date_variants():
    nlu = make_nlu()
    for phrasing in ["what's the date", "what day is it", "today's date"]:
        result = nlu.understand(phrasing)
        assert result["intent"] == "get_date", phrasing


def test_fast_path_greeting_routes_to_chat():
    nlu = make_nlu()
    result = nlu.understand("hey")
    assert result["intent"] == "chat"


def test_fast_path_weather():
    nlu = make_nlu()
    result = nlu.understand("what's the weather like")
    assert result["intent"] == "get_weather"


def test_fast_path_save_name_extracts_name():
    nlu = make_nlu()
    result = nlu.understand("my name is Rohan")
    assert result["intent"] == "save_name"
    assert result["entities"] == {"name": "Rohan"}


def test_fast_path_save_name_handles_call_me_phrasing():
    nlu = make_nlu()
    result = nlu.understand("call me Ada")
    assert result["intent"] == "save_name"
    assert result["entities"]["name"] == "Ada"


def test_non_fast_path_query_falls_through_to_backend_check():
    # Not matched by any fast-path pattern, and Ollama is (almost
    # certainly) not reachable in the test environment — understand()
    # should degrade gracefully rather than raising or hanging.
    nlu = make_nlu()
    result = nlu.understand("tell me a joke about pelicans")
    assert result["intent"] in ("chat",)
    assert "entities" in result and "confidence" in result


# ---------------------------------------------------------------------------
# _validate_intent
# ---------------------------------------------------------------------------

def test_validate_intent_accepts_exact_match():
    nlu = make_nlu()
    assert nlu._validate_intent("take_note") == "take_note"


def test_validate_intent_normalizes_case_and_separators():
    nlu = make_nlu()
    assert nlu._validate_intent("Take_Note") == "take_note"
    assert nlu._validate_intent("take-note") == "take_note"
    assert nlu._validate_intent("  take_note  ") == "take_note"


def test_validate_intent_rejects_unknown_intent():
    nlu = make_nlu()
    assert nlu._validate_intent("this_is_not_a_real_intent") is None


def test_validate_intent_rejects_empty_or_none():
    nlu = make_nlu()
    assert nlu._validate_intent("") is None
    assert nlu._validate_intent(None) is None


def test_validate_intent_rejects_non_string():
    nlu = make_nlu()
    assert nlu._validate_intent(123) is None


# ---------------------------------------------------------------------------
# _normalize_entities (alias -> canonical key remapping)
# ---------------------------------------------------------------------------

def test_normalize_entities_renames_alias_key():
    nlu = make_nlu()
    result = nlu._normalize_entities("add_habit", {"content": "run daily"})
    assert result == {"name": "run daily"}


def test_normalize_entities_never_overwrites_existing_canonical_key():
    nlu = make_nlu()
    result = nlu._normalize_entities(
        "add_habit", {"content": "run daily", "name": "existing"}
    )
    assert result == {"content": "run daily", "name": "existing"}


def test_normalize_entities_is_noop_for_intent_without_aliases():
    nlu = make_nlu()
    entities = {"foo": "bar"}
    result = nlu._normalize_entities("get_time", entities)
    assert result == entities


def test_normalize_entities_is_noop_for_empty_entities():
    nlu = make_nlu()
    assert nlu._normalize_entities("add_habit", {}) == {}


# ---------------------------------------------------------------------------
# _clean_amount_entities
# ---------------------------------------------------------------------------

def test_clean_amount_entities_strips_currency_symbol():
    nlu = make_nlu()
    result = nlu._clean_amount_entities("pluto_log_expense", {"amount": "\u20b9500"})
    assert result == {"amount": 500}
    assert isinstance(result["amount"], int)


def test_clean_amount_entities_strips_commas_and_parses_float():
    nlu = make_nlu()
    result = nlu._clean_amount_entities("pluto_log_expense", {"amount": "1,234.50"})
    assert result == {"amount": 1234.5}
    assert isinstance(result["amount"], float)


def test_clean_amount_entities_uses_correct_field_per_intent():
    nlu = make_nlu()
    result = nlu._clean_amount_entities("apollo_log_weight", {"weight": "70kg"})
    assert result == {"weight": 70}


def test_clean_amount_entities_leaves_unparseable_value_untouched():
    nlu = make_nlu()
    result = nlu._clean_amount_entities("pluto_log_expense", {"amount": "a lot"})
    assert result == {"amount": "a lot"}


def test_clean_amount_entities_noop_for_intent_without_amount_field():
    nlu = make_nlu()
    entities = {"name": "run daily"}
    assert nlu._clean_amount_entities("add_habit", entities) == entities


def test_clean_amount_entities_leaves_non_string_amount_untouched():
    nlu = make_nlu()
    result = nlu._clean_amount_entities("pluto_log_expense", {"amount": 500})
    assert result == {"amount": 500}


# ---------------------------------------------------------------------------
# _repair_learn_fact
# ---------------------------------------------------------------------------

def test_repair_learn_fact_noop_when_value_already_present():
    nlu = make_nlu()
    entities = {"key": "favorite_color", "value": "blue"}
    assert nlu._repair_learn_fact("remember my favorite color is blue", entities) == entities


def test_repair_learn_fact_splits_on_last_is():
    nlu = make_nlu()
    result = nlu._repair_learn_fact(
        "remember that my sister's name is Priya", {}
    )
    assert result["value"] == "Priya"
    assert result["key"] == "my_sisters_name"


def test_repair_learn_fact_uses_last_is_not_first_when_ambiguous():
    nlu = make_nlu()
    # "is" appears in the lead-in phrase's remainder here too; only the
    # LAST "is" should be treated as the key/value split point.
    result = nlu._repair_learn_fact(
        "remember that the answer to what my job is is engineer", {}
    )
    assert result["value"] == "engineer"


def test_repair_learn_fact_falls_back_to_colon_split():
    nlu = make_nlu()
    result = nlu._repair_learn_fact("remember: I like my coffee black", {})
    assert result["value"] == "I like my coffee black"
    assert result["key"] == "i_like_my_coffee_black"


def test_repair_learn_fact_uses_model_fact_rephrasing_as_last_resort():
    nlu = make_nlu()
    # No "is" and no ":" anywhere in the salvageable text — fall back to
    # the model's own "fact" field rather than discarding the fact.
    result = nlu._repair_learn_fact(
        "learn this fact", {"fact": "The user likes their coffee black."}
    )
    assert result["value"] == "The user likes their coffee black."


def test_repair_learn_fact_preserves_existing_key_when_present():
    nlu = make_nlu()
    result = nlu._repair_learn_fact(
        "remember that my sister's name is Priya", {"key": "sister_name"}
    )
    assert result["key"] == "sister_name"
    assert result["value"] == "Priya"


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

def test_parse_response_valid_json():
    nlu = make_nlu()
    raw = '{"intent": "chat", "entities": {}, "response": "hi", "confidence": 0.9}'
    parsed, ok = nlu._parse_response(raw)
    assert ok is True
    assert parsed == {
        "intent": "chat", "entities": {}, "response": "hi", "confidence": 0.9
    }


def test_parse_response_extracts_json_embedded_in_surrounding_text():
    nlu = make_nlu()
    raw = 'Sure! {"intent": "chat", "entities": {}, "response": "hi", "confidence": 0.9} Hope that helps.'
    parsed, ok = nlu._parse_response(raw)
    assert ok is True
    assert parsed["intent"] == "chat"


def test_parse_response_strips_markdown_code_fences():
    nlu = make_nlu()
    raw = '```json\n{"intent": "chat", "entities": {}, "response": "hi", "confidence": 0.9}\n```'
    parsed, ok = nlu._parse_response(raw)
    assert ok is True
    assert parsed["intent"] == "chat"


def test_parse_response_no_json_returns_not_ok():
    nlu = make_nlu()
    parsed, ok = nlu._parse_response("no json here")
    assert ok is False
    assert parsed["intent"] == "chat"
    assert parsed["response"] == "no json here"


def test_parse_response_malformed_json_returns_not_ok():
    nlu = make_nlu()
    parsed, ok = nlu._parse_response('{"intent": "chat", "entities": ')
    assert ok is False


def test_parse_response_coerces_wrong_field_types():
    nlu = make_nlu()
    raw = '{"intent": 123, "entities": "not a dict", "response": 456, "confidence": "high"}'
    parsed, ok = nlu._parse_response(raw)
    assert ok is True
    assert parsed["intent"] == "chat"          # non-string intent -> "chat"
    assert parsed["entities"] == {}            # non-dict entities -> {}
    assert parsed["response"] == "456"         # non-string response -> str()
    assert parsed["confidence"] == 0.5         # non-numeric confidence -> 0.5


def test_parse_response_defaults_missing_fields():
    nlu = make_nlu()
    raw = '{"intent": "chat"}'
    parsed, ok = nlu._parse_response(raw)
    assert ok is True
    assert parsed["entities"] == {}
    assert parsed["confidence"] == 0.5


# ---------------------------------------------------------------------------
# _build_schema
# ---------------------------------------------------------------------------

def test_build_schema_constrains_intent_to_registry_enum():
    nlu = make_nlu()
    schema = nlu._schema
    assert schema["properties"]["intent"]["enum"] == sorted(nlu.valid_intents)
    assert schema["required"] == ["intent", "entities", "response", "confidence"]


# ---------------------------------------------------------------------------
# set_memory
# ---------------------------------------------------------------------------

def test_set_memory_requires_context_method():
    import pytest

    nlu = make_nlu()

    class NoContextMethod:
        pass

    with pytest.raises(TypeError):
        nlu.set_memory(NoContextMethod())


def test_set_memory_accepts_valid_memory_object():
    nlu = make_nlu()

    class ValidMemory:
        def get_top_facts_for_context(self):
            return []

    memory = ValidMemory()
    nlu.set_memory(memory)
    assert nlu._memory is memory
