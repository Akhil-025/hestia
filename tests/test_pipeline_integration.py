# tests/test_pipeline_integration.py
"""
Full-path integration tests: text -> HestiaNLU -> HestiaOrchestrator ->
HecateEngine -> a module (backlog #209).

Per-module unit tests cannot catch the failures that happen *between* the
pieces: the NLU emitting an intent the registry routes somewhere unexpected,
Hecate picking a module whose ``can_handle`` then rejects the stripped intent,
an entity the NLU hands over in a shape the module can't parse. These tests run
the real NLU, orchestrator and Hecate with only the model call scripted, and
the real Apollo engine (temporary database) as the module that does real work.
Other modules are small recording stubs, so a test can assert both *where* a
query went and *what it was given*.

The scripted model replaces ``HestiaNLU._call_llm`` and nothing else, so the
retry loop, intent validation, entity normalisation and caching all run for
real. No test touches the network, Ollama or the clock (``time.sleep`` in the
NLU is patched out).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.nlu as nlu_module
from core.nlu import HestiaNLU
from modules.apollo.engine import ApolloEngine
from modules.base import BaseModule
from modules.hecate import HecateEngine
from modules.hestia.orchestrator import HestiaOrchestrator


class _Llm:
    def generate(self, prompt: str) -> str:
        return "Looks fine."


class ScriptedModel:
    """Stands in for the model behind the NLU: returns queued replies in order
    (repeating the last), and records every prompt it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    @staticmethod
    def intent(intent, entities=None, confidence=0.95):
        return json.dumps({"intent": intent, "entities": entities or {},
                           "response": "", "confidence": confidence})

    def __call__(self, prompt):
        self.prompts.append(prompt)
        i = min(len(self.prompts) - 1, len(self.replies) - 1)
        reply = self.replies[i]
        if isinstance(reply, Exception):
            raise reply
        return reply


class Recorder(BaseModule):
    """A module that accepts a fixed set of intents and records its calls."""

    def __init__(self, name, intents, reply="ok", raises=None):
        self.name = name
        self._intents = set(intents)
        self._reply = reply
        self._raises = raises
        self.calls: list[tuple[str, dict]] = []

    def can_handle(self, intent):
        return intent in self._intents

    def handle(self, intent, entities, context):
        self.calls.append((intent, dict(entities)))
        if self._raises:
            raise self._raises
        return {"response": self._reply, "data": {}, "confidence": 0.95}

    def get_context(self):
        return {}


class Pipeline:
    def __init__(self, *replies, extra=()):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.apollo = ApolloEngine(db_path=Path(tmp.name), llm=_Llm())
        self.hermes = Recorder("hermes", {"read_email", "send_email"}, "You have 2 emails.")
        self.chronos = Recorder("chronos", {"get_time", "set_reminder"}, "It is noon.")
        self.model = ScriptedModel(*replies)
        self.nlu = HestiaNLU(prompt_path="/nonexistent/nlu_prompt.txt",
                             alias_path=None, cache_ttl_seconds=0)
        self.nlu._health_check = lambda: True
        self.nlu._call_llm = self.model
        self.orch = HestiaOrchestrator()
        self.orch.register_hecate(HecateEngine())
        for m in (self.apollo, self.hermes, self.chronos, *extra):
            self.orch.register(m)

    def ask(self, text):
        result = self.nlu.understand(text)
        self.last_nlu = result
        return self.orch.dispatch(text, result)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(nlu_module.time, "sleep", lambda *_: None)


I = ScriptedModel.intent


# ------------------------------------------------------------- happy paths

def test_weight_in_kg_reaches_apollo_and_is_stored():
    p = Pipeline(I("apollo_log_weight", {"weight": "72.5", "unit": "kg"}))
    reply = p.ask("I weigh 72.5 kilos")
    assert "72.5 kg" in reply
    assert p.apollo.db.latest_weight()["weight_kg"] == 72.5


def test_weight_in_pounds_is_converted_for_storage():
    p = Pipeline(I("apollo_log_weight", {"weight": "154", "unit": "lb"}))
    p.ask("log 154 pounds")
    assert abs(p.apollo.db.latest_weight()["weight_kg"] - 69.85) < 0.1


def test_an_entity_with_extra_words_still_parses_end_to_end():
    # The NLU often leaves units in the value; the parser has to cope.
    p = Pipeline(I("apollo_log_workout", {"type": "run", "duration": "30 min"}))
    p.ask("I ran for half an hour")
    workouts = p.apollo.db.get_workouts(7)
    assert len(workouts) == 1 and workouts[0]["duration"] == 30


def test_a_registered_non_apollo_intent_goes_to_its_own_module():
    p = Pipeline(I("read_email"))
    assert p.ask("any new mail?") == "You have 2 emails."
    assert [c[0] for c in p.hermes.calls] == ["read_email"]
    assert p.apollo.db.get_workouts(7) == []


def test_the_original_query_text_reaches_the_module():
    p = Pipeline(I("read_email"))
    p.ask("any new mail?")
    assert p.hermes.calls[0][1]["raw_query"] == "any new mail?"


def test_the_platform_action_envelope_fails_closed_at_the_nlu_and_is_retried():
    # PLATFORM_ACTION is not a real intent, so the NLU rejects it and asks the
    # model again; the correct answer on the retry is dispatched normally.
    # (Hecate's own unwrapping is covered in test_hecate_routing_tiers.py; it
    # only matters for callers that bypass the NLU's validation.)
    p = Pipeline(I("PLATFORM_ACTION", {"action": "READ_EMAIL"}), I("read_email"))
    assert p.ask("check my inbox") == "You have 2 emails."
    assert len(p.model.prompts) == 2 and "PLATFORM_ACTION" in p.model.prompts[1]


def test_an_envelope_that_never_resolves_is_not_guessed_into_a_module():
    p = Pipeline(I("PLATFORM_ACTION", {"action": "SEND_EMAIL"}))
    p.ask("check my inbox")
    assert p.hermes.calls == []


def test_fast_path_intents_never_call_the_model():
    p = Pipeline(I("chat"))
    assert p.ask("what time is it") == "It is noon."
    assert p.model.prompts == []
    assert p.chronos.calls[0][0] == "get_time"


# ------------------------------------------------- bad input stays contained

def test_a_negative_duration_is_rejected_and_nothing_is_logged():
    p = Pipeline(I("apollo_log_workout", {"type": "run", "duration": "-30 min"}))
    reply = p.ask("log a -30 minute run")
    assert "between" in reply and "minutes" in reply
    assert p.apollo.db.get_workouts(7) == []


def test_an_absurd_number_is_rejected_without_crashing():
    p = Pipeline(I("apollo_log_workout", {"type": "run", "duration": "9" * 400}))
    reply = p.ask("log a very long run")
    assert isinstance(reply, str) and reply
    assert p.apollo.db.get_workouts(7) == []


def test_an_out_of_range_weight_asks_instead_of_storing():
    p = Pipeline(I("apollo_log_weight", {"weight": "5", "unit": "kg"}))
    p.ask("I weigh 5 kilos")
    assert p.apollo.db.latest_weight() is None


# ---------------------------------------------------------- NLU retry loop

def test_an_unknown_intent_is_retried_with_a_correction_then_succeeds():
    p = Pipeline(I("make_it_so"), I("read_email"))
    assert p.ask("any new mail?") == "You have 2 emails."
    assert len(p.model.prompts) == 2 and "make_it_so" in p.model.prompts[1]


def test_unparseable_replies_are_retried_then_succeed():
    p = Pipeline("not json at all", I("read_email"))
    assert p.ask("any new mail?") == "You have 2 emails."
    assert len(p.model.prompts) == 2


def test_a_model_that_never_answers_properly_degrades_to_chat_not_an_exception():
    p = Pipeline("garbage", "garbage", "garbage")
    p.orch.register(Recorder("core", {"chat", "get_time"}, "Sorry, say that again?"))
    reply = p.ask("blah blah")
    assert reply == "Sorry, say that again?"
    assert len(p.model.prompts) == 3


def test_a_provider_exception_is_survived():
    p = Pipeline(RuntimeError("backend down"), I("read_email"))
    assert p.ask("any new mail?") == "You have 2 emails."


# --------------------------------------------------------- Hecate decisions

def test_a_low_confidence_registered_intent_asks_before_acting():
    p = Pipeline(I("send_email", confidence=0.3))
    p.orch.register(Recorder("core", {"clarify_intent", "chat"}, "Did you mean to send an email?"))
    reply = p.ask("uh send it")
    assert reply == "Did you mean to send an email?"
    assert p.hermes.calls == []                      # never acted on a guess


def test_an_unregistered_module_prefix_falls_back_to_chat_safely():
    p = Pipeline(I("chat"))
    p.orch.register(Recorder("core", {"chat"}, "Just chatting."))
    assert p.ask("tell me a joke") == "Just chatting."
    assert p.hermes.calls == [] and p.apollo.db.get_workouts(7) == []


def test_a_module_that_isnt_registered_does_not_crash_dispatch():
    p = Pipeline(I("read_email"))
    p.orch.unregister("hermes")
    p.orch.register(Recorder("core", {"chat", "read_email"}, "Handled by core."))
    assert isinstance(p.ask("any new mail?"), str)


# ---------------------------------------------------------------- resilience

def test_a_module_that_raises_gives_a_reply_not_an_exception():
    p = Pipeline(I("read_email"))
    p.hermes._raises = RuntimeError("gmail exploded")
    p.orch.register(Recorder("core", {"chat"}, "fallback"))
    assert isinstance(p.ask("any new mail?"), str)


def test_repeated_failures_trip_the_breaker_and_other_modules_keep_working():
    p = Pipeline(I("read_email"))
    p.hermes._raises = RuntimeError("down")
    p.orch.register(Recorder("core", {"chat"}, "fallback"))
    for _ in range(6):
        p.ask("any new mail?")
    p.model.replies = [I("apollo_log_weight", {"weight": "70", "unit": "kg"})]
    p.model.prompts.clear()
    assert "70" in p.ask("I weigh 70 kilos")
    assert p.apollo.db.latest_weight()["weight_kg"] == 70


def test_context_carries_recent_intents_between_turns():
    p = Pipeline(I("read_email"))
    p.ask("any new mail?")
    p.ask("any new mail?")
    assert p.orch._ctx.recent_intents[-2:] == ["read_email", "read_email"]
