# tests/test_slot_filling.py
"""
Tests for conversational slot-filling (backlog #29):
modules/hermes/engine.py's extended _clarify(slot=, entities=) plus
HestiaOrchestrator's PendingSlotFill / _resolve_pending_slot.

Uses a tiny synthetic module rather than the real HermesEngine for most
cases (no Google API, no OAuth) — the mechanism being tested lives in the
orchestrator, not in Hermes. A handful of tests at the bottom exercise the
real _clarify()/_send_email wiring directly.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.base import BaseModule
from modules.hecate import HecateEngine
from modules.hermes.engine import _clarify
from modules.hestia.orchestrator import HestiaOrchestrator


class _SlotModule(BaseModule):
    """A module whose one intent needs a `to` entity, mirroring send_email."""

    name = "core"

    def can_handle(self, intent):
        return intent == "chat"

    def handle(self, intent, entities, context):
        to = (entities.get("to") or "").strip()
        if not to:
            return _clarify("Who should this go to?", slot="to", entities=entities)
        return {"response": f"Sent to {to}.", "data": {}, "confidence": 0.9}


def make_orchestrator():
    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(_SlotModule())
    return orch


def send_chat(orch, text, entities=None):
    return orch.dispatch(
        text, {"intent": "chat", "entities": entities or {}, "confidence": 0.9}
    )


# ---------------------------------------------------------------------------
# _clarify() itself
# ---------------------------------------------------------------------------

def test_clarify_without_slot_carries_no_slot_data():
    result = _clarify("What date?")
    assert result["data"] == {"needs_clarification": True}


def test_clarify_with_slot_carries_slot_and_entities():
    result = _clarify("Who?", slot="to", entities={"subject": "hi"})
    assert result["data"]["missing_slot"] == "to"
    assert result["data"]["slot_entities"] == {"subject": "hi"}


def test_clarify_with_slot_but_no_entities_defaults_to_empty_dict():
    result = _clarify("Who?", slot="to")
    assert result["data"]["slot_entities"] == {}


# ---------------------------------------------------------------------------
# The happy path: ask, answer, complete
# ---------------------------------------------------------------------------

def test_missing_slot_asks_and_next_reply_completes_it():
    orch = make_orchestrator()
    first = send_chat(orch, "send a message")
    assert "who" in first.lower()

    second = send_chat(orch, "raj@example.com")
    assert second == "Sent to raj@example.com."


def test_the_answer_is_taken_verbatim_not_reclassified():
    # "log my workout" would normally route to a completely different
    # module/intent — but as a slot-fill answer it must be swallowed as
    # the literal value, never sent through NLU/Hecate again.
    orch = make_orchestrator()
    send_chat(orch, "send a message")
    result = send_chat(orch, "log my workout")
    assert result == "Sent to log my workout."


def test_entities_already_known_before_the_missing_one_are_preserved():
    class _TwoSlotModule(BaseModule):
        name = "core"

        def can_handle(self, intent):
            return True

        def handle(self, intent, entities, context):
            to = (entities.get("to") or "").strip()
            subject = (entities.get("subject") or "").strip()
            if not to:
                return _clarify("Who?", slot="to", entities=entities)
            return {
                "response": f"Sent '{subject}' to {to}.",
                "data": {}, "confidence": 0.9,
            }

    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(_TwoSlotModule())
    send_chat(orch, "send a message", entities={"subject": "hello there"})
    result = send_chat(orch, "raj@example.com")
    assert result == "Sent 'hello there' to raj@example.com."


# ---------------------------------------------------------------------------
# Cancellation and expiry
# ---------------------------------------------------------------------------

def test_never_mind_cancels_instead_of_being_taken_as_the_value():
    orch = make_orchestrator()
    send_chat(orch, "send a message")
    result = send_chat(orch, "never mind")
    assert "never mind" in result.lower()
    assert orch._pending_slot is None


def test_empty_reply_reprompts_without_consuming_the_pending_state_incorrectly():
    orch = make_orchestrator()
    send_chat(orch, "send a message")
    result = send_chat(orch, "   ")
    assert "didn't catch" in result.lower()


def test_expired_pending_slot_falls_through_to_normal_routing(monkeypatch):
    import modules.hestia.orchestrator as orch_module

    orch = make_orchestrator()
    fake_now = [1000.0]
    monkeypatch.setattr(orch_module.time, "time", lambda: fake_now[0])

    send_chat(orch, "send a message")
    assert orch._pending_slot is not None

    fake_now[0] = 1000.0 + orch_module._CONFIRMATION_TTL_SECONDS + 1
    result = send_chat(orch, "raj@example.com")
    # Falls through to normal dispatch of a fresh "chat" query, which
    # routes straight back into _SlotModule with an empty "to" again —
    # so it re-asks, rather than "raj@example.com" being silently
    # accepted as the stale slot's answer.
    assert "Sent to" not in result
    assert orch._pending_slot is not None
    assert orch._pending_slot.created_at == fake_now[0]


# ---------------------------------------------------------------------------
# Chaining into the other pending states
# ---------------------------------------------------------------------------

def test_filling_one_slot_can_lead_straight_into_a_confirmation_gate():
    class _ConfirmAfterFillModule(BaseModule):
        name = "core"

        def can_handle(self, intent):
            return True

        def handle(self, intent, entities, context):
            to = (entities.get("to") or "").strip()
            if not to:
                return _clarify("Who?", slot="to", entities=entities)
            if not entities.get("_confirmed"):
                return {
                    "response": f"Send to {to}?",
                    "data": {}, "confidence": 0.9,
                    "needs_confirmation": True,
                    "confirm_intent": "chat",
                    "confirm_entities": {"to": to, "_confirmed": True},
                    "confirm_label": f"send to {to}",
                }
            return {"response": f"Sent to {to}.", "data": {}, "confidence": 0.9}

    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(_ConfirmAfterFillModule())

    send_chat(orch, "send a message")
    ask = send_chat(orch, "raj@example.com")
    assert "?" in ask
    assert orch._pending is not None
    assert orch._pending_slot is None

    result = send_chat(orch, "yes")
    assert result == "Sent to raj@example.com."


def test_filling_one_slot_can_lead_straight_into_another_missing_slot():
    class _TwoMissingModule(BaseModule):
        name = "core"

        def can_handle(self, intent):
            return True

        def handle(self, intent, entities, context):
            to = (entities.get("to") or "").strip()
            body = (entities.get("body") or "").strip()
            if not to:
                return _clarify("Who?", slot="to", entities=entities)
            if not body:
                return _clarify("Say what?", slot="body", entities=entities)
            return {"response": f"Sent '{body}' to {to}.", "data": {}, "confidence": 0.9}

    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(_TwoMissingModule())

    send_chat(orch, "send a message")
    second_question = send_chat(orch, "raj@example.com")
    assert "say what" in second_question.lower()
    assert orch._pending_slot is not None
    assert orch._pending_slot.missing_slot == "body"

    result = send_chat(orch, "running late")
    assert result == "Sent 'running late' to raj@example.com."


def test_pending_slot_and_pending_confirmation_never_coexist():
    orch = make_orchestrator()
    send_chat(orch, "send a message")
    assert orch._pending_slot is not None
    assert orch._pending is None


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_module_unregistered_between_ask_and_answer_falls_through_safely():
    orch = make_orchestrator()
    send_chat(orch, "send a message")
    orch.unregister("core")
    # No module to re-dispatch to — must not raise.
    result = orch._resolve_pending_slot("raj@example.com")
    assert result is None


def test_handler_exception_during_slot_fill_is_caught():
    class _ExplodesOnFill(BaseModule):
        name = "core"

        def can_handle(self, intent):
            return True

        def handle(self, intent, entities, context):
            if not entities.get("to"):
                return _clarify("Who?", slot="to", entities=entities)
            raise RuntimeError("boom")

    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(_ExplodesOnFill())
    send_chat(orch, "send a message")
    result = send_chat(orch, "raj@example.com")
    assert "something went wrong" in result.lower()


# ---------------------------------------------------------------------------
# Real Hermes wiring (no mock module)
# ---------------------------------------------------------------------------

def test_hermes_send_email_missing_recipient_carries_slot_and_body():
    from modules.hermes.engine import HermesEngine

    hermes = HermesEngine.__new__(HermesEngine)  # skip __init__ (needs Google creds)
    result = hermes._send_email({"body": "running 10 min late"})
    assert result["data"]["needs_clarification"] is True
    assert result["data"]["missing_slot"] == "to"
    assert result["data"]["slot_entities"]["body"] == "running 10 min late"


def test_hermes_send_email_missing_body_carries_slot():
    from modules.hermes.engine import HermesEngine

    hermes = HermesEngine.__new__(HermesEngine)
    result = hermes._send_email({"to": "raj@example.com"})
    assert result["data"]["missing_slot"] == "body"
    assert result["data"]["slot_entities"]["to"] == "raj@example.com"


def test_hermes_create_event_missing_title_carries_slot():
    from modules.hermes.engine import HermesEngine

    hermes = HermesEngine.__new__(HermesEngine)
    hermes._tz = __import__("datetime").timezone.utc
    result = hermes._create_event({"raw_query": "schedule something"})
    assert result["data"]["missing_slot"] == "title"
