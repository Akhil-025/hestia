"""Regression tests for the Hestia Hardening Plan (Phases 2-4, no LLM)."""
import os
import sys
from datetime import date

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.entity_coerce import coerce_entities
from core.intent_aliases import IntentAliasResolver
from core.nlu import _SAVE_NAME_PATTERN
from core.slot_extract import extract_slots, guard_write_intent
from core.temporal import split_date_time, normalize_event_entities
from modules.mnemosyne.keys import canonical_key, is_junk, salvage_fact

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "hardening_cases.yaml")
_ALIASES = os.path.join(os.path.dirname(__file__), "..", "config", "intent_aliases.yaml")


def _det_intent(text):
    if _SAVE_NAME_PATTERN.match(text):
        return "save_name", {"name": _SAVE_NAME_PATTERN.match(text).group(1)}
    s = extract_slots(text)
    if s:
        return s["intent"], s["entities"]
    return IntentAliasResolver(path=_ALIASES).resolve(text), {}


def _cases():
    with open(_GOLDEN, encoding="utf-8") as fh:
        return [c for c in yaml.safe_load(fh) if c["layer"] == "deterministic"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["utterance"])
def test_golden_deterministic(case):
    intent, ents = _det_intent(case["utterance"])
    assert intent == case["intent"]
    for k, v in (case.get("entities") or {}).items():
        assert ents.get(k) == v


@pytest.mark.parametrize("t", ["I'm tired", "I am hungry", "I'm fine thanks",
                               "i am going home now"])
def test_bare_i_am_does_not_save_name(t):
    assert not _SAVE_NAME_PATTERN.match(t)


@pytest.mark.parametrize("t", ["My name is Akhil", "call me Bob"])
def test_explicit_name_still_saved(t):
    assert _SAVE_NAME_PATTERN.match(t)


def test_question_form_guard():
    assert guard_write_intent("take_note", "What notes do I have?") == "get_notes"
    assert guard_write_intent("learn_fact", "What do you know about me?") == "get_user_info"
    assert guard_write_intent("take_note", "take a note: buy milk") == "take_note"


def test_question_never_extracts_write_slots():
    assert extract_slots("What notes do I have?") is None
    assert extract_slots("how much did I spend on groceries?") is None


@pytest.mark.parametrize("d,t,exp", [
    ("", "tomorrow at 5pm", ("tomorrow", "17:00")),
    ("tomorrow", "5pm", ("tomorrow", "17:00")),
    ("tomorrow at 5pm", "", ("tomorrow", "17:00")),
    ("2026-12-25", "09:00", ("2026-12-25", "09:00")),
    ("", "", ("", "")),
])
def test_temporal_split(d, t, exp):
    assert split_date_time(d, t, today=date(2026, 10, 8)) == exp


def test_temporal_feeds_hermes_parser():
    from zoneinfo import ZoneInfo
    from modules.hermes.engine import _parse_datetime
    d, t = split_date_time("", "tomorrow at 5pm")
    dt = _parse_datetime(d, t, ZoneInfo("Asia/Kolkata"))
    assert (dt.hour, dt.minute) == (17, 0) and dt.tzinfo is None


def test_normalize_event_entities_keeps_other_keys():
    out = normalize_event_entities({"title": "Gym", "time": "tomorrow at 5pm"})
    assert out["title"] == "Gym" and out["time"] == "17:00"


def test_coerce_list_topic_and_keeps_lists_elsewhere():
    out = coerce_entities({"topic": ["gaps"], "ids": ["a", "b"], "x": None})
    assert out == {"topic": "gaps", "ids": ["a", "b"]}


@pytest.mark.parametrize("bad", [None, 5, [], ["gaps"], {"a": 1}])
def test_boundary_fuzz_coerce_never_raises(bad):
    assert isinstance(coerce_entities({"topic": bad, "name": bad}), dict)
    assert coerce_entities(bad) == {}  or isinstance(bad, dict)


def test_ares_topic_list_regression():
    from modules.ares.engine import AresEngine
    topic = AresEngine._topic(AresEngine.__new__(AresEngine), {"topic": ["gaps"]})
    assert topic == "gaps"


def test_sec_company_search_rejects_non_str():
    from core import free_apis
    with pytest.raises(free_apis.FreeAPIError):
        free_apis.sec_company_search(None)


def test_memory_keys():
    assert canonical_key("my name") == "user_name"
    assert canonical_key("user_sister_name") == "sister_name"
    assert is_junk("what: my name", "x") and is_junk("hi my name: akhil", "x")
    assert not is_junk("user_name", "Akhil")
    assert salvage_fact("my sister's name is Priya") == ("sister_name", "Priya")
    assert salvage_fact("what is my name?") is None
