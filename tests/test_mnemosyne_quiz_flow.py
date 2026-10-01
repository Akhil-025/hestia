# tests/test_mnemosyne_quiz_flow.py
"""
Tests for the user-facing quiz layer (backlog #34, #35): the start_quiz /
answer_quiz / quiz_performance intents, spoken-answer parsing, choice
shuffling, and the trend calculation. The generation/scoring engine itself
is covered by tests/test_quiz_engine.py.
"""
import json
import os
import random
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_mnemosyne import make_engine  # noqa: E402
from core.quiz_engine import (  # noqa: E402
    QuizEngine, format_question, is_skip, parse_choice,
)

_QUIZ_JSON = json.dumps({"questions": [
    {"question": "Boiling point of water?", "choices": ["50C", "100C", "150C", "200C"], "correct_index": 1},
    {"question": "Gas plants absorb?", "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "correct_index": 2},
    {"question": "Symbol for gold?", "choices": ["Ag", "Fe", "Au", "Pb"], "correct_index": 2},
]})


class _QuizLLM:
    def __init__(self, response=_QUIZ_JSON):
        self.response = response

    def generate(self, prompt, fmt=None):
        return self.response if fmt == "json" else "ok"


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp()
    eng, _ = make_engine(tmp)
    eng.quiz_engine.llm = _QuizLLM()
    eng.quiz_engine._shuffle = False        # deterministic answer positions
    eng.learn("thermo_first_law", "energy is conserved")      # quiz source material
    yield eng
    shutil.rmtree(tmp, ignore_errors=True)


def _reply(engine, text, session_entities=None):
    ents = dict(session_entities or {"_quiz": True})
    ents["answer"] = text
    ents["raw_query"] = text
    return engine.handle("start_quiz", ents, {})


# ---------------------------------------------------------------------------
# parse_choice
# ---------------------------------------------------------------------------

_CH = ["50C", "100C", "Carbon dioxide", "Hydrogen"]


@pytest.mark.parametrize("text,expected", [
    ("B", 1), ("b)", 1), ("option b", 1), ("the answer is c", 2), ("it's D", 3),
    ("bee", 1), ("see", 2), ("the second one", 1), ("2", 1), ("third", 2),
    ("last", 3), ("Carbon dioxide", 2), ("i think it's carbon dioxide", 2),
    ("carbon dioxid", 2), ("hydrogen gas", 3), ("number 3", 2), ("a", 0),
])
def test_parse_choice_understands_spoken_and_typed_answers(text, expected):
    assert parse_choice(text, _CH) == expected


@pytest.mark.parametrize("text", ["blah", "I like bees", "see you later", "", "   "])
def test_parse_choice_returns_none_instead_of_guessing(text):
    assert parse_choice(text, _CH) is None


def test_parse_choice_rejects_a_letter_beyond_the_choices():
    assert parse_choice("d", ["x", "y"]) is None


def test_parse_choice_ambiguous_text_is_none():
    assert parse_choice("hydrogen carbon dioxide", _CH) is None


def test_is_skip():
    assert is_skip("skip") and is_skip("I don't know.") and not is_skip("b")


def test_format_question_is_readable():
    out = format_question({"question": "Q?", "choices": ["a", "b", "c", "d"]}, 2, 5)
    assert out.splitlines()[0] == "Question 2 of 5: Q?"
    assert "C) c" in out


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------

def test_shuffle_keeps_correct_index_pointing_at_the_right_answer(tmp_path):
    qe = QuizEngine(_QuizLLM(), str(tmp_path / "q.db"), rng=random.Random(7))
    qs = qe.generate_quiz(["material"], "chem", n=3)
    texts = {"Boiling point of water?": "100C", "Gas plants absorb?": "Carbon dioxide",
             "Symbol for gold?": "Au"}
    for q in qs:
        assert q["choices"][q["correct_index"]] == texts[q["question"]]
        stored = qe.get_question(q["id"])
        assert stored["choices"][stored["correct_index"]] == texts[q["question"]]


def test_shuffle_actually_moves_answers(tmp_path):
    positions = set()
    for seed in range(12):
        qe = QuizEngine(_QuizLLM(), str(tmp_path / f"q{seed}.db"), rng=random.Random(seed))
        positions.add(qe.generate_quiz(["m"], "s", n=1)[0]["correct_index"])
    assert len(positions) > 1


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def _attempt(qe, qid, correct):
    q = qe.get_question(qid)
    qe.grade_answer(qid, q["correct_index"] if correct else (q["correct_index"] + 1) % 4)


def test_trend_needs_enough_history(tmp_path):
    qe = QuizEngine(_QuizLLM(), str(tmp_path / "q.db"), shuffle_choices=False)
    qid = qe.generate_quiz(["m"], "s", n=1)[0]["id"]
    for _ in range(5):
        _attempt(qe, qid, True)
    assert qe.get_subject_trend("s") is None


def test_trend_detects_improvement_and_decline(tmp_path):
    qe = QuizEngine(_QuizLLM(), str(tmp_path / "q.db"), shuffle_choices=False)
    qid = qe.generate_quiz(["m"], "s", n=1)[0]["id"]
    for ok in [False] * 5 + [True] * 5:
        _attempt(qe, qid, ok)
    assert qe.get_subject_trend("s")["direction"] == "up"
    for ok in [False] * 5:
        _attempt(qe, qid, ok)
    assert qe.get_subject_trend("s")["direction"] == "down"


def test_trend_flat(tmp_path):
    qe = QuizEngine(_QuizLLM(), str(tmp_path / "q.db"), shuffle_choices=False)
    qid = qe.generate_quiz(["m"], "s", n=1)[0]["id"]
    for ok in [True, False] * 5:
        _attempt(qe, qid, ok)
    assert qe.get_subject_trend("s")["direction"] == "flat"


# ---------------------------------------------------------------------------
# The conversation
# ---------------------------------------------------------------------------

def test_start_quiz_asks_the_first_question_via_the_slot_fill_contract(engine):
    r = engine.handle("start_quiz", {"subject": "thermo"}, {})
    assert r["data"]["needs_clarification"] and r["data"]["missing_slot"] == "answer"
    assert "Question 1 of 3" in r["response"] and "A) 50C" in r["response"]


def test_full_quiz_scores_and_records_attempts(engine):
    engine.handle("start_quiz", {"subject": "thermo"}, {})
    r1 = _reply(engine, "B")                     # correct
    assert "Correct!" in r1["response"] and "Question 2 of 3" in r1["response"]
    r2 = _reply(engine, "A")                     # wrong (answer is C)
    assert "Not quite" in r2["response"] and "C) Carbon dioxide" in r2["response"]
    r3 = _reply(engine, "au")                    # correct by text
    assert "You got 2 of 3 (67%) on thermo" in r3["response"]
    assert not r3["data"].get("needs_clarification")
    assert engine._quiz_session is None
    stats = engine.quiz_engine.get_strength_weakness_map()["thermo"]
    assert (stats["attempts"], stats["correct"]) == (3, 2)


def test_unparseable_answer_reasks_the_same_question_without_grading(engine):
    engine.handle("start_quiz", {"subject": "thermo"}, {})
    r = _reply(engine, "hmm let me think")
    assert r["data"]["needs_clarification"]
    assert "didn't catch" in r["response"] and "Question 1 of 3" in r["response"]
    assert engine.quiz_engine.get_strength_weakness_map() == {}


def test_skip_is_recorded_as_incorrect_and_reveals_the_answer(engine):
    engine.handle("start_quiz", {"subject": "thermo"}, {})
    r = _reply(engine, "skip")
    assert "Skipped. The answer was B) 100C" in r["response"]
    assert engine.quiz_engine.get_strength_weakness_map()["thermo"]["correct"] == 0


def test_quit_stops_with_a_partial_score(engine):
    engine.handle("start_quiz", {"subject": "thermo"}, {})
    _reply(engine, "B")
    r = _reply(engine, "quit")
    assert "stopping the quiz" in r["response"] and "1 of 1" in r["response"]
    assert engine._quiz_session is None


def test_quit_before_answering_anything_does_not_divide_by_zero(engine):
    engine.handle("start_quiz", {"subject": "thermo"}, {})
    r = _reply(engine, "quit")
    assert "No questions answered" in r["response"]


def test_answer_quiz_intent_with_no_quiz_running(engine):
    r = engine.handle("answer_quiz", {"answer": "B"}, {})
    assert "no quiz in progress" in r["response"]


def test_answer_quiz_intent_continues_a_running_quiz(engine):
    engine.handle("start_quiz", {"subject": "thermo"}, {})
    r = engine.handle("answer_quiz", {"answer": "B"}, {})
    assert "Correct!" in r["response"]


def test_subject_and_count_come_from_the_raw_query_for_alias_routed_requests(engine):
    r = engine.handle("start_quiz", {}, {"raw_query": "give me 2 questions on thermo"})
    assert "Quiz on thermo: 3 question" in r["response"] or "Quiz on thermo: 2 question" in r["response"]
    assert engine._quiz_session["subject"] == "thermo"


def test_starting_a_new_quiz_replaces_a_stale_session(engine):
    engine.handle("start_quiz", {"subject": "thermo"}, {})
    _reply(engine, "B")
    engine.handle("start_quiz", {"subject": "thermo"}, {})
    assert engine._quiz_session["index"] == 0


def test_no_source_material_says_so_and_starts_nothing(engine):
    r = engine.handle("start_quiz", {"subject": "astrophysics"}, {})
    assert "couldn't find any notes" in r["response"]
    assert engine._quiz_session is None


def test_model_producing_no_valid_questions_is_reported(engine):
    engine.quiz_engine.llm = _QuizLLM("not json at all")
    r = engine.handle("start_quiz", {"subject": "thermo"}, {})
    assert "couldn't generate" in r["response"]
    assert engine._quiz_session is None


def test_athena_documents_are_used_as_quiz_source(engine):
    class _Res:
        documents = ["Entropy always increases in an isolated system."]

    class _Rag:
        seen = None
        def search(self, q, n_results=None):
            _Rag.seen = q
            return _Res()

    class _Athena:
        rag = _Rag()

    engine.attach_athena(_Athena())
    seen = []
    orig = engine.quiz_engine.generate_quiz
    engine.quiz_engine.generate_quiz = lambda texts, subject, n=5: (seen.extend(texts), orig(texts, subject, n))[1]
    engine.handle("start_quiz", {"subject": "entropy"}, {})
    assert _Rag.seen == "entropy"
    assert any("Entropy always increases" in t for t in seen)


def test_athena_failure_falls_back_to_facts(engine):
    class _Rag:
        def search(self, *a, **k):
            raise RuntimeError("index down")

    class _Athena:
        rag = _Rag()

    engine.attach_athena(_Athena())
    r = engine.handle("start_quiz", {"subject": "thermo"}, {})
    assert "Question 1" in r["response"]


# ---------------------------------------------------------------------------
# quiz_performance (#35)
# ---------------------------------------------------------------------------

def test_performance_with_no_history(engine):
    assert "haven't answered any" in engine.handle("quiz_performance", {}, {})["response"]


def test_performance_ranks_weakest_first_and_names_it(engine):
    qe = engine.quiz_engine
    good = qe.generate_quiz(["m"], "chemistry", n=1)[0]["id"]
    bad = qe.generate_quiz(["m"], "physics", n=1)[0]["id"]
    for _ in range(3):
        _attempt(qe, good, True)
        _attempt(qe, bad, False)
    r = engine.handle("quiz_performance", {}, {})
    assert r["response"].startswith("Weakest: physics.")
    assert r["response"].index("physics") < r["response"].index("chemistry")
    assert set(r["data"]["subjects"]) == {"chemistry", "physics"}


def test_performance_does_not_call_a_thin_sample_a_weakness(engine):
    qe = engine.quiz_engine
    qid = qe.generate_quiz(["m"], "physics", n=1)[0]["id"]
    _attempt(qe, qid, False)
    assert "Not enough attempts" in engine.handle("quiz_performance", {}, {})["response"]
