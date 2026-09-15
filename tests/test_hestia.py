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

    def get_all_facts(self, limit=100, offset=0):
        # Mirrors MnemosyneDB.get_all_facts()'s dict shape (key/value/...),
        # most-recently-learned first, since that's what real callers get.
        items = list(self.facts.items())[::-1]
        return [{"key": k, "value": v, "source": "user", "confidence": 1.0,
                  "created_at": None, "updated_at": None} for k, v in items[:limit]]

    def get_interaction_stats(self):
        # Mirrors MnemosyneDB.get_interaction_stats()'s shape — used by
        # _delete_notes()'s confirmation preview to report a count without
        # touching any rows.
        notes = len([r for r in self.rows if r["intent"] == "take_note"])
        return {
            "total": len(self.rows),
            "notes": notes,
            "unique_intents": len({r["intent"] for r in self.rows}),
        }


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


def test_get_user_info_no_key_falls_back_to_known_facts():
    # Bug: "What do you know about me?" reaches get_user_info with no
    # resolvable key (NLU sends entities={}), and the handler used to
    # hard-return "I don't have that information yet." even when facts
    # like user_name were already stored, because it never queried
    # get_all_facts(). It should now surface what's actually known.
    mem = FakeMemory()
    mem.learn("user_name", "Akhil")
    mem.learn("i_like_my_coffee_black", "user likes coffee black")
    core = make_core(mem)
    r = core.handle("get_user_info", {}, {})
    assert "Akhil" in r["response"]
    assert "coffee" in r["response"].lower()
    assert r["response"] != "I don't have that information yet."


def test_get_user_info_no_key_and_no_facts_still_admits_ignorance():
    core = make_core()
    r = core.handle("get_user_info", {}, {})
    assert r["response"] == "I don't have that information yet."
    assert r["confidence"] == 0.3


def test_get_user_info_specific_key_missing_does_not_dump_all_facts():
    # A specific (but unknown) key was asked for — this must not fall
    # through to the general "here's everything I know" summary, since
    # that would silently answer a different question than the one asked.
    mem = FakeMemory()
    mem.learn("user_name", "Akhil")
    core = make_core(mem)
    r = core.handle("get_user_info", {"key": "favourite_food"}, {})
    assert r["response"] == "I don't have that information yet."
    assert "Akhil" not in r["response"]


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


def test_get_notes_filters_by_topic_entity():
    # Bug: "query my notes on machine learning" extracted {"topic": "..."}
    # via the NLU, but _get_notes() took no entities at all and always
    # returned the same unfiltered top-10 list regardless of what was
    # asked. It must now actually filter on the topic.
    mem = FakeMemory()
    mem.db.rows.append({"query": "note", "response": "Note saved: buy new headphones", "intent": "take_note"})
    mem.db.rows.append({"query": "note", "response": "Note saved: read a paper on machine learning", "intent": "take_note"})
    core = make_core(mem)

    r = core.handle("get_notes", {"topic": "machine learning"}, {})
    assert "machine learning" in r["response"].lower()
    assert "headphones" not in r["response"].lower()


def test_get_notes_topic_with_no_match_says_so_instead_of_dumping_all():
    mem = FakeMemory()
    mem.db.rows.append({"query": "note", "response": "Note saved: buy new headphones", "intent": "take_note"})
    core = make_core(mem)

    r = core.handle("get_notes", {"topic": "taxes"}, {})
    assert "taxes" in r["response"].lower()
    assert "headphones" not in r["response"].lower()


def test_get_notes_no_topic_is_unchanged_unfiltered_behaviour():
    mem = FakeMemory()
    mem.db.rows.append({"query": "note", "response": "Note saved: buy new headphones", "intent": "take_note"})
    core = make_core(mem)

    r = core.handle("get_notes", {}, {})
    assert "headphones" in r["response"].lower()


# ---------------------------------------------------------------------------
# delete_notes: confirmation gating
#
# Deleting notes is irreversible and wipes ALL of them in one shot, so the
# first call must NOT delete anything — it should only report what would
# happen and ask for confirmation (see HestiaOrchestrator's confirmation
# mechanism in modules/hestia/orchestrator.py). Only a call carrying
# entities["_confirmed"] = True — which only the orchestrator sends, after
# the user's next reply reads as a clear "yes" — actually deletes.
# ---------------------------------------------------------------------------

def test_delete_notes_with_no_notes_reports_nothing_to_delete():
    core = make_core()
    r = core.handle("delete_notes", {}, {})
    assert "don't have any notes" in r["response"].lower()
    assert r["data"] == {}


def test_delete_notes_first_call_asks_for_confirmation_and_deletes_nothing():
    mem = FakeMemory()
    mem.db.rows.append({"query": "note", "response": "Note saved: buy milk", "intent": "take_note"})
    mem.db.rows.append({"query": "note", "response": "Note saved: buy eggs", "intent": "take_note"})
    core = make_core(mem)

    r = core.handle("delete_notes", {}, {})

    assert r.get("needs_confirmation") is True
    assert "2" in r["response"]
    assert len(mem.db.rows) == 2  # nothing deleted yet
    assert r["confirm_intent"] == "delete_notes"


def test_delete_notes_confirmed_call_actually_deletes():
    mem = FakeMemory()
    mem.db.rows.append({"query": "note", "response": "Note saved: buy milk", "intent": "take_note"})
    core = make_core(mem)

    preview = core.handle("delete_notes", {}, {})
    assert preview.get("needs_confirmation") is True

    r = core.handle("delete_notes", {"_confirmed": True}, {})
    assert r["data"]["deleted"] == 1
    assert len(mem.db.rows) == 0


def test_delete_notes_confirmation_preview_survives_db_check_failure_gracefully():
    class _ExplodingDB(FakeDB):
        def get_interaction_stats(self):
            raise RuntimeError("db unavailable")

    mem = FakeMemory()
    mem.db = _ExplodingDB()
    core = make_core(mem)

    r = core.handle("delete_notes", {}, {})
    assert r["confidence"] == 0.0
    assert r.get("needs_confirmation", False) is False


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


# ---------------------------------------------------------------------------
# Orchestrator: confirmation gating end-to-end
#
# CoreModule.delete_notes is one of three handlers (with Hermes.send_email
# and Mnemosyne.forget_fact) that ask the orchestrator to hold an action as
# pending instead of performing it immediately. These tests exercise that
# mechanism through the real orchestrator + real CoreModule + real Hecate,
# not a mock, since the interesting behaviour lives in how dispatch() reacts
# to the *next* query after a needs_confirmation response.
# ---------------------------------------------------------------------------

def _notes_orchestrator(note_count=2):
    mem = FakeMemory()
    for i in range(note_count):
        mem.db.rows.append({"query": "note", "response": f"Note saved: item {i}", "intent": "take_note"})
    core = make_core(mem)
    orch = _build_orchestrator(core)
    return orch, mem


def test_orchestrator_holds_delete_notes_pending_until_confirmed():
    orch, mem = _notes_orchestrator(note_count=3)

    ask = orch.dispatch("delete all my notes", {"intent": "delete_notes", "entities": {}, "confidence": 0.9})
    assert "3" in ask
    assert len(mem.db.rows) == 3  # still nothing deleted

    confirm = orch.dispatch("yes", {"intent": "chat", "entities": {}, "confidence": 0.5})
    assert "3" in confirm or "Deleted" in confirm
    assert len(mem.db.rows) == 0


def test_orchestrator_cancels_pending_delete_notes_on_no():
    orch, mem = _notes_orchestrator(note_count=1)

    orch.dispatch("delete all my notes", {"intent": "delete_notes", "entities": {}, "confidence": 0.9})
    cancel = orch.dispatch("no", {"intent": "chat", "entities": {}, "confidence": 0.5})

    assert "won't" in cancel.lower()
    assert len(mem.db.rows) == 1  # untouched


def test_orchestrator_drops_pending_delete_notes_on_unrelated_next_query():
    # An unrelated follow-up (not a clear yes/no) must abandon the pending
    # action rather than guessing — a later, unrelated "yes" must never be
    # able to reach back and trigger it.
    orch, mem = _notes_orchestrator(note_count=1)

    orch.dispatch("delete all my notes", {"intent": "delete_notes", "entities": {}, "confidence": 0.9})
    orch.dispatch("what's the weather like", {"intent": "chat", "entities": {}, "confidence": 0.9})

    assert len(mem.db.rows) == 1  # nothing deleted

    # A "yes" now is a fresh, unrelated query, not a confirmation — must
    # not delete notes either.
    orch.dispatch("yes", {"intent": "chat", "entities": {}, "confidence": 0.9})
    assert len(mem.db.rows) == 1


def test_orchestrator_pending_confirmation_expires(monkeypatch):
    orch, mem = _notes_orchestrator(note_count=1)

    orch.dispatch("delete all my notes", {"intent": "delete_notes", "entities": {}, "confidence": 0.9})
    assert orch._pending is not None

    # Simulate the confirmation window having passed.
    orch._pending.created_at -= 10_000

    confirm = orch.dispatch("yes", {"intent": "chat", "entities": {}, "confidence": 0.9})
    assert len(mem.db.rows) == 1  # the stale "yes" did NOT delete anything
    assert orch._pending is None


# ---------------------------------------------------------------------------
# CoreModule.stream_chat()
# ---------------------------------------------------------------------------

def test_stream_chat_yields_pieces_from_generate_stream(monkeypatch):
    import modules.hestia.core_module as core_module

    core = make_core()

    def _fake_stream(prompt, model=None, host=None, port=None):
        assert "hello" in prompt
        yield "Hi"
        yield " there"

    monkeypatch.setattr(core_module, "generate_stream", _fake_stream)

    pieces = list(core.stream_chat("hello"))
    assert pieces == ["Hi", " there"]


def test_stream_chat_prefers_llm_instance_generate_stream_when_present(monkeypatch):
    class _FakeLLM:
        def generate_stream(self, prompt, options=None):
            yield "from-llm-instance"

    core = CoreModule(
        memory=FakeMemory(), ollama_cfg={}, llm=_FakeLLM(), timezone_name=_TZ_NAME
    )
    pieces = list(core.stream_chat("hello"))
    assert pieces == ["from-llm-instance"]


def test_stream_chat_falls_back_to_module_level_generate_stream_when_llm_lacks_it(monkeypatch):
    import modules.hestia.core_module as core_module

    class _LLMWithoutStreaming:
        def generate(self, prompt):
            return "blocking-only"

    def _fake_stream(prompt, model=None, host=None, port=None):
        yield "module-level"

    monkeypatch.setattr(core_module, "generate_stream", _fake_stream)
    core = CoreModule(
        memory=FakeMemory(), ollama_cfg={}, llm=_LLMWithoutStreaming(), timezone_name=_TZ_NAME
    )
    pieces = list(core.stream_chat("hello"))
    assert pieces == ["module-level"]


def test_stream_chat_stops_yielding_without_raising_on_exception(monkeypatch):
    import modules.hestia.core_module as core_module

    def _fake_stream(prompt, model=None, host=None, port=None):
        yield "partial"
        raise RuntimeError("connection dropped")

    monkeypatch.setattr(core_module, "generate_stream", _fake_stream)
    core = make_core()

    pieces = list(core.stream_chat("hello"))  # must not raise
    assert pieces == ["partial"]


def test_stream_chat_empty_when_generate_stream_yields_nothing(monkeypatch):
    import modules.hestia.core_module as core_module

    def _fake_stream(prompt, model=None, host=None, port=None):
        return
        yield  # pragma: no cover - makes this a generator function

    monkeypatch.setattr(core_module, "generate_stream", _fake_stream)
    core = make_core()

    assert list(core.stream_chat("hello")) == []


# ---------------------------------------------------------------------------
# HestiaOrchestrator.try_stream_chat()
# ---------------------------------------------------------------------------

def test_try_stream_chat_streams_plain_chat_through_core(monkeypatch):
    import modules.hestia.core_module as core_module

    def _fake_stream(prompt, model=None, host=None, port=None):
        yield "Once"
        yield " upon a time"

    monkeypatch.setattr(core_module, "generate_stream", _fake_stream)
    orch = _build_orchestrator()

    gen = orch.try_stream_chat(
        "tell me something", {"intent": "chat", "entities": {}, "confidence": 0.5}
    )
    assert gen is not None
    assert list(gen) == ["Once", " upon a time"]


def test_try_stream_chat_updates_context_after_streaming_completes(monkeypatch):
    import modules.hestia.core_module as core_module

    monkeypatch.setattr(
        core_module, "generate_stream",
        lambda prompt, model=None, host=None, port=None: iter(["ok"]),
    )
    orch = _build_orchestrator()

    gen = orch.try_stream_chat(
        "hi there", {"intent": "chat", "entities": {}, "confidence": 0.5}
    )
    list(gen)  # drain the generator so the finally-block context push runs
    assert orch._ctx.active_modules == orch.registered_modules  # sanity: still consistent
    # The chat intent should now be reflected in recent intent history.
    assert "chat" in orch._ctx.recent_intents


def test_try_stream_chat_streams_with_synthesized_secondary_context(monkeypatch):
    import modules.hestia.core_module as core_module

    seen_prompts = []

    def _fake_stream(prompt, model=None, host=None, port=None):
        seen_prompts.append(prompt)
        yield "It'll be sunny"

    monkeypatch.setattr(core_module, "generate_stream", _fake_stream)

    class _Weather(BaseModule):
        name = "weather"

        def can_handle(self, intent):
            return False

        def handle(self, intent, entities, context):
            return {"response": "", "data": {}, "confidence": 0.0}

        def get_context(self):
            return {"forecast": "sunny, 75F"}

    orch = _build_orchestrator()
    orch.register(_Weather())
    monkeypatch.setattr(
        orch, "_route",
        lambda raw_query, nlu_result: {
            "primary": "core", "secondary": ["weather"], "synthesize": True,
        },
    )

    gen = orch.try_stream_chat(
        "should I bring an umbrella", {"intent": "chat", "entities": {}, "confidence": 0.5}
    )
    assert gen is not None
    assert list(gen) == ["It'll be sunny"]
    # The secondary module's context made it into the single streamed prompt.
    assert len(seen_prompts) == 1
    assert "sunny, 75F" in seen_prompts[0]


def test_try_stream_chat_streams_plain_when_synthesize_true_but_no_secondary(monkeypatch):
    import modules.hestia.core_module as core_module

    seen_prompts = []

    def _fake_stream(prompt, model=None, host=None, port=None):
        seen_prompts.append(prompt)
        yield "sure"

    monkeypatch.setattr(core_module, "generate_stream", _fake_stream)

    orch = _build_orchestrator()
    monkeypatch.setattr(
        orch, "_route",
        lambda raw_query, nlu_result: {
            "primary": "core", "secondary": [], "synthesize": True,
        },
    )

    gen = orch.try_stream_chat(
        "tell me a joke", {"intent": "chat", "entities": {}, "confidence": 0.5}
    )
    assert gen is not None
    assert list(gen) == ["sure"]
    assert "Relevant context" not in seen_prompts[0]


def test_try_stream_chat_returns_none_when_a_confirmation_is_pending():
    orch, mem = _notes_orchestrator(note_count=1)
    orch.dispatch("delete all my notes", {"intent": "delete_notes", "entities": {}, "confidence": 0.9})

    gen = orch.try_stream_chat(
        "yes", {"intent": "chat", "entities": {}, "confidence": 0.9}
    )
    assert gen is None


def test_try_stream_chat_returns_none_for_non_core_module():
    class _OtherModule(BaseModule):
        name = "other"

        def can_handle(self, intent):
            return intent == "chat"

        def handle(self, intent, entities, context):
            return {"response": "from other", "data": {}, "confidence": 0.9}

    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(make_core())
    orch.register(_OtherModule())

    gen = orch.try_stream_chat(
        "hi", {"intent": "other_chat", "entities": {}, "confidence": 0.9}
    )
    assert gen is None


def test_try_stream_chat_returns_none_for_non_chat_intent():
    orch = _build_orchestrator()
    gen = orch.try_stream_chat(
        "what's my name", {"intent": "get_user_info", "entities": {"key": "name"}, "confidence": 0.9}
    )
    assert gen is None


def test_try_stream_chat_returns_none_when_core_has_no_stream_chat():
    class _StreamlessCore(BaseModule):
        name = "core"

        def can_handle(self, intent):
            return intent == "chat"

        def handle(self, intent, entities, context):
            return {"response": "blocking only", "data": {}, "confidence": 0.7}

    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(_StreamlessCore())

    gen = orch.try_stream_chat(
        "hi", {"intent": "chat", "entities": {}, "confidence": 0.5}
    )
    assert gen is None


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