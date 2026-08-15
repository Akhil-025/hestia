# tests/test_ares.py
"""
Regression tests for modules/ares/engine.py.

Run with:  pytest tests/test_ares.py -v
(or:       python -m pytest tests/test_ares.py -v)

Background — bug fixed here:

Every Ares analysis prompt (_PLAN_PROMPT, _RISK_PROMPT, _SWOT_PROMPT,
_DECISION_PROMPT, _PREMORTEM_PROMPT, _COMPETITIVE_PROMPT,
_CONTINGENCY_PROMPT, _WAR_ROOM_PROMPT) used to interpolate only a bare
{topic} string and instruct the model to "be specific to the topic",
while every _*() handler's `context: dict` parameter went completely
unused. With a small local model and zero real grounding, the only way
to satisfy "be specific" was to invent plausible-sounding specifics —
observed in production as a fabricated "Job Offer A / Job Offer B" for
decision_support, and a fabricated military "battlefield" scenario for
war_room_briefing on a topic with no prior context.

Fixed by:
  - AresEngine._grounding() assembling real facts/goals/notes from
    Mnemosyne and injecting it as KNOWN CONTEXT into every prompt, with
    an explicit "don't invent specifics beyond this" instruction.
  - war_room_briefing degrading confidence and explicitly refusing to
    invent a situation when no real context exists, instead of silently
    fabricating one at the same 0.9 confidence as a grounded briefing.
  - _ollama_call now passes a low temperature for these analytical calls.

These tests use a FakeLLM that records the prompt it was called with, so
we can assert grounding is actually being threaded through — not just
that the final formatted response looks reasonable.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.ares.engine import AresEngine


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDB:
    def __init__(self):
        self.goals = []
        self.notes = []

    def get_goals(self, status="active"):
        return [g for g in self.goals if g["status"] == status]

    def get_by_intent(self, intent, limit):
        return self.notes[:limit] if intent == "take_note" else []


class FakeMemory:
    """Stand-in for MnemosyneEngine — only the surface Ares actually calls."""
    def __init__(self):
        self.db = FakeDB()
        self.facts_context = ""     # what get_top_facts_for_context() returns
        self.remember_result = ""   # what remember() returns
        self.learned = {}

    def get_top_facts_for_context(self, limit=8):
        return self.facts_context

    def remember(self, query, n=6):
        return self.remember_result

    def learn(self, key, value, source="user"):
        self.learned[key] = value

    def add_reminder(self, text, due_time):
        pass


class FakeLLM:
    """Records the last prompt/options it was called with; returns canned JSON."""
    def __init__(self, json_response: str):
        self.json_response = json_response
        self.last_prompt = None
        self.last_options = None
        self.call_count = 0

    def generate(self, prompt: str, fmt: str = None, options: dict = None) -> str:
        self.last_prompt = prompt
        self.last_options = options
        self.call_count += 1
        return self.json_response


def make_engine(memory=None, json_response='{"strengths": ["a"], "weaknesses": [], "opportunities": [], "threats": []}'):
    llm = FakeLLM(json_response)
    engine = AresEngine(memory=memory or FakeMemory(), ollama_cfg={}, llm=llm)
    return engine, llm


# ---------------------------------------------------------------------------
# Grounding is actually threaded into the prompt
# ---------------------------------------------------------------------------

def test_swot_prompt_includes_known_facts():
    mem = FakeMemory()
    mem.facts_context = "- user_name: Akhil\n- employer: Acme Corp"
    engine, llm = make_engine(mem)

    engine.handle("swot_analysis", {"topic": "switching jobs"}, {})

    assert "Akhil" in llm.last_prompt
    assert "Acme Corp" in llm.last_prompt


def test_swot_prompt_includes_active_goals():
    mem = FakeMemory()
    mem.db.goals = [{"text": "finish the Hestia project by September", "status": "active"}]
    engine, llm = make_engine(mem)

    engine.handle("swot_analysis", {"topic": "my project"}, {})

    assert "finish the Hestia project by September" in llm.last_prompt


def test_swot_prompt_includes_recent_notes():
    mem = FakeMemory()
    mem.db.notes = [{"query": "note", "response": "Note saved: buy new headphones"}]
    engine, llm = make_engine(mem)

    engine.handle("swot_analysis", {"topic": "my week"}, {})

    assert "buy new headphones" in llm.last_prompt


def test_grounding_block_is_none_placeholder_when_memory_is_empty():
    engine, llm = make_engine(FakeMemory())

    engine.handle("swot_analysis", {"topic": "switching jobs"}, {})

    assert "(none)" in llm.last_prompt


def test_grounding_survives_no_memory_at_all():
    # AresEngine(memory=None) must not crash — several call sites construct
    # it without memory (e.g. tests, or a degraded boot).
    llm = FakeLLM('{"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}')
    engine = AresEngine(memory=None, ollama_cfg={}, llm=llm)

    result = engine.handle("swot_analysis", {"topic": "switching jobs"}, {})

    assert result["confidence"] > 0
    assert "(none)" in llm.last_prompt


# ---------------------------------------------------------------------------
# Low temperature is actually requested for analytical calls
# ---------------------------------------------------------------------------

def test_analysis_calls_request_low_temperature():
    engine, llm = make_engine()
    engine.handle("swot_analysis", {"topic": "switching jobs"}, {})
    assert llm.last_options == {"temperature": 0.2}


# ---------------------------------------------------------------------------
# decision_support: never invents options beyond what the user gave
# ---------------------------------------------------------------------------

def test_decision_support_prompt_instructs_against_inventing_options():
    engine, llm = make_engine(
        json_response='{"recommendation": "A", "options_analysis": [], "key_factors": [], "next_step": ""}'
    )
    engine.handle("decision_support", {"topic": "which job to take", "options": "Offer A, Offer B"}, {})
    assert "never invent additional or alternative options" in llm.last_prompt
    assert "Offer A, Offer B" in llm.last_prompt


def test_decision_support_without_options_asks_instead_of_calling_llm():
    engine, llm = make_engine()
    result = engine.handle("decision_support", {"topic": "which job to take"}, {})
    assert llm.call_count == 0
    assert "options" in result["response"].lower()


# ---------------------------------------------------------------------------
# war_room_briefing: the actual reported hallucination bug
# ---------------------------------------------------------------------------

def test_war_room_briefing_with_no_context_does_not_fabricate_a_situation():
    # This is the exact reported bug: topic = a raw command sentence,
    # nothing in memory, and the model previously invented a full
    # "battlefield" scenario anyway. The prompt must now explicitly forbid
    # that, and the handler must reflect the lack of grounding in a lower
    # confidence rather than reporting 0.9 for a fabricated briefing.
    engine, llm = make_engine(
        FakeMemory(),
        json_response=(
            '{"situation": "Nothing on record yet for this topic.", '
            '"priorities": [], "open_risks": [], '
            '"recommended_next_move": "Tell me more about what\'s going on.", '
            '"confidence": "Low"}'
        ),
    )

    result = engine.handle(
        "war_room_briefing",
        {"raw_query": "summarize my recent work and highlight gaps"},
        {},
    )

    assert "do not invent" in llm.last_prompt.lower() or "Do NOT invent" in llm.last_prompt
    assert result["confidence"] == 0.4
    assert "no prior analysis on record" in result["response"].lower()


def test_war_room_briefing_combines_structured_and_semantic_context():
    mem = FakeMemory()
    mem.facts_context = "- employer: Acme Corp"
    mem.remember_result = "Regarding switching jobs: you were leaning towards Offer B."
    engine, llm = make_engine(
        mem,
        json_response=(
            '{"situation": "In progress.", "priorities": ["decide"], '
            '"open_risks": [], "recommended_next_move": "Decide by Friday.", '
            '"confidence": "Medium"}'
        ),
    )

    result = engine.handle("war_room_briefing", {"topic": "switching jobs"}, {})

    assert "Acme Corp" in llm.last_prompt
    assert "Offer B" in llm.last_prompt
    assert result["confidence"] == 0.9
    assert "drawing on prior analysis" in result["response"].lower()


def test_war_room_briefing_uses_recent_intents_from_orchestrator_context():
    engine, llm = make_engine(
        FakeMemory(),
        json_response='{"situation": "x", "priorities": [], "open_risks": [], "recommended_next_move": "y", "confidence": "Low"}',
    )
    engine.handle(
        "war_room_briefing",
        {"topic": "my week"},
        {"recent_intents": ["log_workout", "track_sleep", "log_mood"]},
    )
    assert "log_workout" in llm.last_prompt


# ---------------------------------------------------------------------------
# Existing formatting behaviour is unchanged for a well-grounded response
# ---------------------------------------------------------------------------

def test_swot_analysis_still_formats_a_table():
    engine, llm = make_engine(
        FakeMemory(),
        json_response=(
            '{"strengths": ["S1"], "weaknesses": ["W1"], '
            '"opportunities": ["O1"], "threats": ["T1"]}'
        ),
    )
    result = engine.handle("swot_analysis", {"topic": "switching jobs"}, {})
    assert "STRENGTHS" in result["response"]
    assert "S1" in result["response"]
    assert result["confidence"] == 0.9


def test_unparseable_response_falls_back_gracefully():
    engine, llm = make_engine(FakeMemory(), json_response="not valid json")
    result = engine.handle("swot_analysis", {"topic": "switching jobs"}, {})
    assert result["confidence"] == 0.3