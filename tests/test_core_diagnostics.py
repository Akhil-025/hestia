# tests/test_core_diagnostics.py
"""
Tests for CoreModule's diagnostic intents (backlog #3, #8, #259):
modules_status, explain_routing, report_mistake.

These are the user-facing half of core/observability.py. What's tested
here is the contract CoreModule has to hold up regardless of what
Diagnostics does internally:

  - can_handle() declares all three (a registry entry alone isn't enough —
    if can_handle() disagrees, dispatch silently falls back to chat);
  - each returns the standard {response, data, confidence} dict;
  - a missing Diagnostics degrades to an honest message instead of raising
    (CoreModule is constructed without one in plenty of tests and in any
    stripped-down process);
  - a Diagnostics whose method raises is caught, because a broken
    diagnostic is the most likely thing to be broken when you reach for
    the diagnostics.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.hestia.core_module import CoreModule

_DIAGNOSTIC_INTENTS = ("modules_status", "explain_routing", "report_mistake")


class _FakeDiagnostics:
    def __init__(self, raises=False):
        self._raises = raises
        self.feedback_notes = []

    def _maybe_raise(self):
        if self._raises:
            raise RuntimeError("diagnostics exploded")

    def status_summary(self):
        self._maybe_raise()
        return "3 module(s) registered.\nready: athena\nunknown: core, chronos"

    def module_status(self):
        self._maybe_raise()
        return {
            "athena": {"registered": True, "state": "ready", "probe": "ready"},
            "core": {"registered": True, "state": "unknown", "probe": None},
            "chronos": {"registered": True, "state": "unknown", "probe": None},
        }

    def explain_last(self):
        self._maybe_raise()
        return "Your last query was 'i spent 200'. Hecate routed it to 'pluto'."

    def last_decision(self):
        self._maybe_raise()
        return {"query": "i spent 200", "intent": "pluto_log_expense", "module": "pluto"}

    def record_feedback(self, note=""):
        self._maybe_raise()
        self.feedback_notes.append(note)
        return "Logged."


def make_core(diagnostics=None):
    return CoreModule(
        memory=None,
        ollama_cfg={"model": "mistral"},
        timezone_name="Asia/Kolkata",
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intent", _DIAGNOSTIC_INTENTS)
def test_can_handle_declares_each_diagnostic_intent(intent):
    assert make_core().can_handle(intent) is True


def test_existing_intents_still_declared():
    core = make_core()
    for intent in ("chat", "take_note", "get_history", "get_user_info"):
        assert core.can_handle(intent) is True


def test_diagnostics_is_an_optional_constructor_argument():
    # Every existing call site (and test) constructs CoreModule without it.
    core = CoreModule(memory=None, ollama_cfg={})
    assert core.can_handle("modules_status") is True


# ---------------------------------------------------------------------------
# modules_status (#8)
# ---------------------------------------------------------------------------

def test_modules_status_reports_the_summary():
    result = make_core(_FakeDiagnostics()).handle("modules_status", {}, {})
    assert "3 module(s) registered" in result["response"]
    assert "athena" in result["data"]["modules"]
    assert result["confidence"] > 0.9


def test_modules_status_without_diagnostics_is_honest():
    result = make_core().handle("modules_status", {}, {})
    assert "aren't wired up" in result["response"]
    assert result["data"] == {}


def test_modules_status_survives_a_raising_diagnostics():
    result = make_core(_FakeDiagnostics(raises=True)).handle("modules_status", {}, {})
    assert "couldn't read" in result["response"].lower()
    assert result["confidence"] < 0.5


# ---------------------------------------------------------------------------
# explain_routing (#3)
# ---------------------------------------------------------------------------

def test_explain_routing_returns_the_explanation():
    result = make_core(_FakeDiagnostics()).handle("explain_routing", {}, {})
    assert "pluto" in result["response"]
    assert result["data"]["intent"] == "pluto_log_expense"


def test_explain_routing_without_diagnostics_is_honest():
    result = make_core().handle("explain_routing", {}, {})
    assert "aren't wired up" in result["response"]


def test_explain_routing_survives_a_raising_diagnostics():
    result = make_core(_FakeDiagnostics(raises=True)).handle("explain_routing", {}, {})
    assert "couldn't reconstruct" in result["response"].lower()


# ---------------------------------------------------------------------------
# report_mistake (#259)
# ---------------------------------------------------------------------------

def test_report_mistake_forwards_the_raw_query_as_the_note():
    diag = _FakeDiagnostics()
    make_core(diag).handle(
        "report_mistake",
        {"raw_query": "that was wrong, i meant my sleep log"},
        {},
    )
    assert "sleep log" in diag.feedback_notes[0]


def test_report_mistake_prefers_an_extracted_note_entity():
    # The NLU prompt's few-shot examples put the useful half in
    # entities.note; that should win over the whole raw utterance.
    diag = _FakeDiagnostics()
    make_core(diag).handle(
        "report_mistake",
        {"note": "meant sleep not workout", "raw_query": "that was wrong, ..."},
        {},
    )
    assert diag.feedback_notes[0] == "meant sleep not workout"


def test_report_mistake_with_no_note_at_all():
    diag = _FakeDiagnostics()
    result = make_core(diag).handle("report_mistake", {}, {})
    assert diag.feedback_notes == [""]
    assert result["confidence"] > 0.9


def test_report_mistake_without_diagnostics_is_honest():
    result = make_core().handle("report_mistake", {"raw_query": "that was wrong"}, {})
    assert "aren't wired up" in result["response"]


def test_report_mistake_survives_a_raising_diagnostics():
    result = make_core(_FakeDiagnostics(raises=True)).handle(
        "report_mistake", {"raw_query": "that was wrong"}, {}
    )
    assert "couldn't save" in result["response"].lower()


# ---------------------------------------------------------------------------
# Response shape (the BaseModule contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intent", _DIAGNOSTIC_INTENTS)
@pytest.mark.parametrize("diagnostics", [None, _FakeDiagnostics(), _FakeDiagnostics(True)])
def test_every_path_returns_the_standard_response_dict(intent, diagnostics):
    result = make_core(diagnostics).handle(intent, {"raw_query": "x"}, {})
    assert set(result) >= {"response", "data", "confidence"}
    assert isinstance(result["response"], str) and result["response"]
    assert isinstance(result["data"], dict)
    assert 0.0 <= result["confidence"] <= 1.0
