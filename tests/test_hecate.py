# tests/test_hecate.py
"""
Regression tests for modules/hecate/engine.py.

Run with:  pytest tests/test_hecate.py -v
(or:       python -m pytest tests/test_hecate.py -v)

These cover two real bugs found and fixed in HecateEngine:

1. "get_user_info" was force-routed straight to Mnemosyne, bypassing
   Core's date/time misclassification recovery (Core answers
   "what's today's date" even when the NLU wrongly emits
   {"intent": "get_user_info", "entities": {"key": "current_date"}}).
   Mnemosyne has no such recovery, so the query silently broke.

2. Hecate's text-trigger tiers (athena/mnemosyne/iris "trigger" matches)
   picked a `primary` module based on the raw query text, but never told
   the orchestrator which *intent* that module should handle. Since the
   NLU's own intent for these phrasings is frequently "chat" (not
   declared by athena/mnemosyne/iris), the orchestrator's
   can_handle()-mismatch recovery silently re-routed the query to Core's
   generic chat handler — discarding Hecate's routing decision entirely.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.hecate import HecateEngine
from modules.base import BaseModule
from modules.hestia.orchestrator import HestiaOrchestrator


def make_engine():
    return HecateEngine()


# ---------------------------------------------------------------------------
# Bug 1: get_user_info must NOT be force-routed to mnemosyne
# ---------------------------------------------------------------------------

def test_get_user_info_date_key_routes_to_core():
    h = make_engine()
    nlu = {"intent": "get_user_info", "entities": {"key": "current_date"}, "confidence": 0.99}
    decision = h.decide("what is todays date", nlu, ["core", "mnemosyne", "chronos"])
    assert decision["primary"] == "core"


def test_get_user_info_generic_fact_still_reachable():
    h = make_engine()
    nlu = {"intent": "get_user_info", "entities": {"key": "user_name"}, "confidence": 0.99}
    decision = h.decide("what's my name", nlu, ["core", "mnemosyne"])
    # Core is a superset (forwards to the same fact store) so it should win.
    assert decision["primary"] == "core"


def test_learn_fact_and_forget_fact_still_forced_to_mnemosyne():
    h = make_engine()
    for intent in ("learn_fact", "forget_fact"):
        nlu = {"intent": intent, "entities": {}, "confidence": 0.9}
        decision = h.decide("remember that I like coffee", nlu, ["core", "mnemosyne"])
        assert decision["primary"] == "mnemosyne", intent


# ---------------------------------------------------------------------------
# Bug 2: trigger-tier routing must carry an intent the target module declares
# ---------------------------------------------------------------------------

def test_mnemosyne_trigger_carries_recall_intent():
    h = make_engine()
    nlu = {"intent": "chat", "entities": {}, "confidence": 0.95}
    decision = h.decide("do you remember my birthday", nlu, ["core", "mnemosyne"])
    assert decision["primary"] == "mnemosyne"
    assert decision["intent"] == "recall"


def test_remind_me_to_does_not_collide_with_mnemosyne_recall_trigger():
    """
    Regression: the bare "remind me" trigger used to match reminder-CREATION
    phrasing too ("remind me to call mom tomorrow"), silently routing it to
    Mnemosyne's semantic recall (which has no way to create a reminder)
    instead of leaving it for Chronos's "set_reminder" intent. Since Chronos
    has no Tier-2 text-trigger fallback, this made "remind me to X" produce
    "I don't have any memories about that yet." instead of ever creating
    the reminder, whenever the NLU misclassified the utterance as "chat".
    """
    h = make_engine()
    nlu = {"intent": "chat", "entities": {}, "confidence": 0.9}
    decision = h.decide("remind me to call mom tomorrow", nlu, ["core", "mnemosyne", "chronos"])
    assert decision["primary"] != "mnemosyne"


def test_remind_me_what_still_reaches_mnemosyne_recall():
    """The interrogative form is a genuine recall question and must still route."""
    h = make_engine()
    nlu = {"intent": "chat", "entities": {}, "confidence": 0.9}
    decision = h.decide("remind me what my wifi password is", nlu, ["core", "mnemosyne"])
    assert decision["primary"] == "mnemosyne"
    assert decision["intent"] == "recall"


def test_athena_trigger_carries_search_intent():
    h = make_engine()
    nlu = {"intent": "chat", "entities": {}, "confidence": 0.9}
    decision = h.decide("in my notes what did I write about heat transfer", nlu, ["core", "athena"])
    assert decision["primary"] == "athena"
    assert decision["intent"] == "search"


def test_athena_ingest_trigger_carries_ingest_intent():
    """
    Regression: Athena previously had no text-trigger path at all for
    ingestion (unlike Iris's "ingest my photos" / "iris_ingest" pair), so
    there was no voice/chat-reachable way to index documents.
    """
    h = make_engine()
    nlu = {"intent": "chat", "entities": {}, "confidence": 0.9}
    decision = h.decide("please ingest my documents", nlu, ["core", "athena"])
    assert decision["primary"] == "athena"
    assert decision["intent"] == "ingest"


def test_athena_ingest_intent_direct_routing():
    """Tier 1.5: a reliable 'athena_ingest' NLU intent must route straight
    to athena with intent='ingest', mirroring the existing 'athena_search'
    direct-routing test above."""
    h = make_engine()
    nlu = {"intent": "athena_ingest", "entities": {}, "confidence": 0.95}
    decision = h.decide("index my notes please", nlu, ["core", "athena"])
    assert decision["primary"] == "athena"
    assert decision["intent"] == "ingest"


def test_iris_trigger_distinguishes_ingest_from_search():
    h = make_engine()
    nlu = {"intent": "chat", "entities": {}, "confidence": 0.9}

    d_ingest = h.decide("please ingest photos now", nlu, ["core", "iris"])
    assert d_ingest["primary"] == "iris"
    assert d_ingest["intent"] == "ingest"

    d_search = h.decide("find photo of my dog", nlu, ["core", "iris"])
    assert d_search["primary"] == "iris"
    assert d_search["intent"] == "search"


def test_tier1_and_tierx_routes_leave_intent_unset():
    """Reliable-intent tiers should NOT override the NLU's intent."""
    h = make_engine()
    decision = h.decide(
        "read my email",
        {"intent": "hermes_read_email", "confidence": 0.9},
        ["core", "hermes"],
    )
    assert decision["primary"] == "hermes"
    assert decision.get("intent") is None


# ---------------------------------------------------------------------------
# End-to-end through the orchestrator, with lightweight fake modules
# ---------------------------------------------------------------------------

class _FakeCore(BaseModule):
    name = "core"

    def can_handle(self, intent):
        return intent in {"chat", "get_user_info"}

    def handle(self, intent, entities, context):
        if intent == "get_user_info" and (entities.get("key") or "").lower() == "current_date":
            return {"response": "core: today's date", "data": {}, "confidence": 0.95}
        if intent == "chat":
            return {"response": "core: generic chat", "data": {}, "confidence": 0.5}
        return {"response": "core: no info", "data": {}, "confidence": 0.3}


class _FakeMnemosyne(BaseModule):
    name = "mnemosyne"

    def can_handle(self, intent):
        return intent in {"recall", "learn_fact", "forget_fact", "get_user_info"}

    def handle(self, intent, entities, context):
        if intent == "recall":
            return {"response": "mnemosyne: recalled memory", "data": {}, "confidence": 0.85}
        return {"response": f"mnemosyne: {intent}", "data": {}, "confidence": 0.9}


def _build_orchestrator():
    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(_FakeCore())
    orch.register(_FakeMnemosyne())
    return orch


def test_e2e_date_misclassification_answered_by_core():
    orch = _build_orchestrator()
    resp = orch.dispatch(
        "what is todays date",
        {"intent": "get_user_info", "entities": {"key": "current_date"}, "confidence": 0.99},
    )
    assert resp == "core: today's date"


def test_e2e_mnemosyne_trigger_reaches_mnemosyne_not_core_chat():
    orch = _build_orchestrator()
    resp = orch.dispatch(
        "do you remember my birthday",
        {"intent": "chat", "entities": {}, "confidence": 0.95},
    )
    assert resp == "mnemosyne: recalled memory"


if __name__ == "__main__":
    # Allow running without pytest installed.
    import inspect
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
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        sys.exit(1)