# tests/test_hestia.py
"""
Regression tests for modules/hestia (CoreModule + HestiaOrchestrator).

Run with:  pytest tests/test_hestia.py -v
(or:       python -m pytest tests/test_hestia.py -v
 or:       python3 tests/test_hestia.py)

These cover one real bug found and fixed in CoreModule, plus the
orchestrator's dispatch/registration contract that every module (including
Hermes and Chronos) relies on:

1. CoreModule.get_user_info()/get_system_info() used naive
   datetime.datetime.now() — server-local time — while ChronosEngine and
   HermesEngine both resolve the user's configured IANA timezone via
   ZoneInfo. Since get_user_info() is the exact recovery path the NLU
   falls into when it misclassifies "what's today's date" (see
   test_hecate.py's date-key tests), a server running in a different
   timezone than the user (e.g. UTC vs Asia/Kolkata) would silently answer
   with the wrong date/time — disagreeing with what Chronos would have
   said for the identical question. Fixed by threading a `timezone_name`
   constructor arg through to a ZoneInfo, same as Hermes/Chronos.
"""
import sys
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.hestia.core_module import CoreModule
from modules.hestia.orchestrator import HestiaOrchestrator
from modules.hecate import HecateEngine
from modules.base import BaseModule

_TZ_NAME = "Asia/Kolkata"
_TZ = ZoneInfo(_TZ_NAME)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDB:
    def __init__(self):
        self.facts = {}
        self.rows = []

    def get_fact(self, key):
        return self.facts.get(key)

    def get_by_intent(self, intent, limit):
        return [r for r in self.rows if r["intent"] == intent][:limit]

    def delete_by_intent(self, intent):
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["intent"] != intent]
        return before - len(self.rows)

    def get_recent_interactions_excluding(self, limit, excluded_intents):
        return [r for r in self.rows if r["intent"] not in excluded_intents][:limit]


class FakeMemory:
    def __init__(self):
        self.db = FakeDB()
        self.learned = {}

    def learn(self, key, value):
        self.learned[key] = value
        self.db.facts[key] = value


def make_core(memory=None, timezone_name=_TZ_NAME):
    return CoreModule(memory=memory or FakeMemory(), ollama_cfg={}, timezone_name=timezone_name)


# ---------------------------------------------------------------------------
# Bug fix: timezone-aware date/time answers
# ---------------------------------------------------------------------------

def test_get_user_info_current_date_uses_configured_timezone_not_server_time():
    core = make_core(timezone_name=_TZ_NAME)
    r = core.handle("get_user_info", {"key": "current_date"}, {})
    expected = datetime.now(_TZ).strftime("%A, %B %d, %Y")
    assert expected in r["response"]


def test_get_user_info_current_time_uses_configured_timezone():
    core = make_core(timezone_name=_TZ_NAME)
    r = core.handle("get_user_info", {"key": "current_time"}, {})
    assert re.search(r"\d{1,2}:\d{2} (AM|PM)", r["response"])


def test_get_system_info_uses_configured_timezone():
    core = make_core(timezone_name=_TZ_NAME)
    r = core.handle("get_system_info", {}, {})
    assert re.search(r"\d{1,2}:\d{2} (AM|PM)", r["response"])


def test_unrecognised_timezone_falls_back_to_utc_without_raising():
    core = make_core(timezone_name="Not/ARealZone")
    assert core._tz == ZoneInfo("UTC")


def test_default_timezone_is_utc_when_unspecified():
    core = CoreModule(memory=FakeMemory(), ollama_cfg={})
    assert core._tz == ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# Defensive entity handling
# ---------------------------------------------------------------------------

def test_save_name_handles_explicit_none_without_raising():
    core = make_core()
    r = core.handle("save_name", {"name": None}, {})
    assert r["confidence"] == 0.0
    assert r["response"]


def test_save_name_titlecases_and_persists():
    mem = FakeMemory()
    core = make_core(mem)
    r = core.handle("save_name", {"name": "alice smith"}, {})
    assert mem.learned["user_name"] == "Alice Smith"
    assert "Alice Smith" in r["response"]


def test_get_history_handles_non_numeric_limit_without_raising():
    core = make_core()
    r = core.handle("get_history", {"limit": "not-a-number"}, {})
    assert r["response"]  # falls back to default limit, doesn't crash


def test_get_history_handles_missing_limit():
    core = make_core()
    r = core.handle("get_history", {}, {})
    assert r["response"]


# ---------------------------------------------------------------------------
# get_user_info: date/time keys vs generic fact lookup
# ---------------------------------------------------------------------------

def test_get_user_info_generic_fact_lookup_still_works():
    mem = FakeMemory()
    mem.learn("favourite_colour", "blue")
    core = make_core(mem)
    r = core.handle("get_user_info", {"key": "favourite_colour"}, {})
    assert r["response"] == "blue"
    assert r["confidence"] == 0.85


def test_get_user_info_unknown_key_is_low_confidence_not_an_error():
    core = make_core()
    r = core.handle("get_user_info", {"key": "unknown_thing"}, {})
    assert r["confidence"] == 0.3
    assert r["response"]


# ---------------------------------------------------------------------------
# take_note / get_notes round trip (uses the "Note saved:" prefix contract)
# ---------------------------------------------------------------------------

def test_take_note_extracts_content_from_entities():
    core = make_core()
    r = core.handle("take_note", {"content": "buy milk"}, "")
    assert r["data"]["note"] == "buy milk"
    assert r["response"] == "Note saved: buy milk"


def test_take_note_falls_back_to_stripping_raw_query():
    core = make_core()
    r = core.handle("take_note", {"raw_query": "take a note: buy toy"}, {})
    assert r["data"]["note"] == "buy toy"


def test_get_notes_reads_note_content_back_from_response_prefix():
    mem = FakeMemory()
    mem.db.rows.append({"query": "take a note buy milk", "response": "Note saved: buy milk", "intent": "take_note"})
    core = make_core(mem)
    r = core.handle("get_notes", {}, {})
    assert "buy milk" in r["response"]
    assert "Note saved" not in r["response"]


def test_get_notes_empty_reports_no_notes():
    core = make_core()
    r = core.handle("get_notes", {}, {})
    assert "No notes" in r["response"]


# ---------------------------------------------------------------------------
# can_handle / unknown intent contract
# ---------------------------------------------------------------------------

def test_can_handle_covers_all_declared_intents():
    core = make_core()
    for intent in (
        "save_name", "take_note", "get_notes", "delete_notes",
        "get_history", "set_preference", "get_system_info",
        "get_user_info", "chat",
    ):
        assert core.can_handle(intent)
    assert not core.can_handle("totally_unknown_intent")


def test_unhandled_intent_returns_blank_zero_confidence_not_a_crash():
    core = make_core()
    r = core.handle("totally_unknown_intent", {}, {})
    assert r["confidence"] == 0.0
    assert r["data"] == {}


# ---------------------------------------------------------------------------
# Orchestrator: registration, dispatch, and error containment
# ---------------------------------------------------------------------------

def _build_orchestrator(core=None):
    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(core or make_core())
    return orch


def test_orchestrator_registers_core_and_dispatches_to_it():
    orch = _build_orchestrator()
    resp = orch.dispatch(
        "what is today's date",
        {"intent": "get_user_info", "entities": {"key": "current_date"}, "confidence": 0.9},
    )
    expected = datetime.now(_TZ).strftime("%A, %B %d, %Y")
    assert expected in resp


def test_orchestrator_falls_back_gracefully_when_module_raises():
    class _ExplodingModule(BaseModule):
        name = "core"

        def can_handle(self, intent):
            return True

        def handle(self, intent, entities, context):
            raise RuntimeError("boom")

    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(_ExplodingModule())
    resp = orch.dispatch("anything", {"intent": "chat", "entities": {}, "confidence": 0.9})
    assert resp  # a graceful string, never an unhandled exception


def test_orchestrator_rejects_non_basemodule_registration():
    orch = HestiaOrchestrator()
    try:
        orch.register(object())
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_orchestrator_replacing_a_module_updates_active_modules_once():
    orch = _build_orchestrator()
    orch.register(make_core())  # re-register "core"
    assert orch.registered_modules.count("core") == 1


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
        except Exception as e:
            failures.append(name)
            print(f"ERROR {name}: {e!r}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        sys.exit(1)