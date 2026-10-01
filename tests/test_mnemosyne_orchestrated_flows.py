# tests/test_mnemosyne_orchestrated_flows.py
"""
End-to-end checks that the multi-turn Mnemosyne flows (quiz, study review)
work through the REAL HestiaOrchestrator + Hecate routing, not just when a
test calls engine.handle() by hand: the intents must route to Mnemosyne via
the registry, the question must be held as a pending slot-fill, and the
user's next reply must come back to the same flow verbatim — never
re-classified as some other command.
"""
import json
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_mnemosyne import make_engine  # noqa: E402
from modules.base import BaseModule  # noqa: E402
from modules.hecate import HecateEngine  # noqa: E402
from modules.hestia.orchestrator import HestiaOrchestrator  # noqa: E402

_QUIZ_JSON = json.dumps({"questions": [
    {"question": "Boiling point of water?", "choices": ["50C", "100C", "150C", "200C"], "correct_index": 1},
    {"question": "Symbol for gold?", "choices": ["Ag", "Fe", "Au", "Pb"], "correct_index": 2},
]})


class _LLM:
    def generate(self, prompt, fmt=None):
        return _QUIZ_JSON if fmt == "json" else "ok"


class _Core(BaseModule):
    name = "core"

    def can_handle(self, intent):
        return intent == "chat"

    def handle(self, intent, entities, context):
        return {"response": "CHAT FALLBACK", "data": {}, "confidence": 0.5}


@pytest.fixture
def world():
    tmp = tempfile.mkdtemp()
    engine, _ = make_engine(tmp)
    engine.quiz_engine.llm = _LLM()
    engine.quiz_engine._shuffle = False
    engine.learn("thermo_first_law", "energy is conserved")
    orch = HestiaOrchestrator()
    orch.register_hecate(HecateEngine())
    orch.register(_Core())
    orch.register(engine)
    yield orch, engine
    shutil.rmtree(tmp, ignore_errors=True)


def say(orch, text, intent="chat", entities=None):
    return orch.dispatch(text, {"intent": intent, "entities": entities or {}, "confidence": 0.9})


def test_quiz_runs_through_the_orchestrator_and_replies_are_not_reclassified(world):
    orch, engine = world
    q1 = say(orch, "quiz me on thermo", "start_quiz", {"subject": "thermo"})
    assert "Question 1 of 2" in q1

    # "chat" would route to the core fallback if this reply were re-classified.
    q2 = say(orch, "B")
    assert "Correct!" in q2 and "Question 2 of 2" in q2 and "FALLBACK" not in q2

    done = say(orch, "the third one")
    assert "2 of 2" in done and "100%" in done

    stats = engine.quiz_engine.get_strength_weakness_map()["thermo"]
    assert (stats["attempts"], stats["correct"]) == (2, 2)


def test_cancelling_mid_quiz_uses_the_orchestrators_cancel_phrase(world):
    orch, engine = world
    say(orch, "quiz me on thermo", "start_quiz", {"subject": "thermo"})
    assert say(orch, "never mind") == "Okay, never mind."
    # The abandoned session must not capture later, unrelated chat.
    assert say(orch, "hello there") == "CHAT FALLBACK"


def test_unclear_answer_is_asked_again_through_the_orchestrator(world):
    orch, _ = world
    say(orch, "quiz me on thermo", "start_quiz", {"subject": "thermo"})
    again = say(orch, "uhh not sure what to say")
    assert "didn't catch" in again and "Question 1 of 2" in again
    assert "Correct!" in say(orch, "b")           # still in the same quiz


def test_alias_style_request_with_no_entities_still_finds_its_subject(world):
    orch, _ = world
    first = say(orch, "quiz me on thermo", "start_quiz", {})
    assert "Quiz on thermo" in first


def test_study_review_through_the_orchestrator(world):
    orch, engine = world
    say(orch, "add", "add_study_fact", {"key": "entropy", "value": "disorder"})
    asked = say(orch, "review my study cards", "review_study")
    assert "Card 1 of 1" in asked and "entropy" in asked
    result = say(orch, "disorder")
    assert "Correct!" in result and "recalled 1 of 1" in result
    assert engine.study_store.due_cards() == []


def test_graph_question_routes_to_mnemosyne(world):
    orch, engine = world
    engine.learn("sister_name", "Priya")
    reply = say(orch, "what connects to Priya", "graph_connections", {"entity": "Priya"})
    assert "sister name" in reply


def test_new_intents_are_all_owned_by_mnemosyne_in_the_registry():
    from modules.hecate.intent_registry import module_for_intent
    from modules.mnemosyne.extensions import EXTENSION_INTENTS
    assert EXTENSION_INTENTS
    assert {module_for_intent(i) for i in EXTENSION_INTENTS} == {"mnemosyne"}
