# tests/test_quiz_engine.py
"""
Tests for core/quiz_store.py and core/quiz_engine.py (backlog #34, #35).
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.quiz_engine import QuizEngine
from core.quiz_store import QuizStore


class _FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, prompt, fmt=None):
        self.calls += 1
        return self.response


_GOOD_RESPONSE = json.dumps({
    "questions": [
        {
            "question": "What is the boiling point of water at sea level?",
            "choices": ["50°C", "100°C", "150°C", "200°C"],
            "correct_index": 1,
        },
        {
            "question": "What gas do plants absorb for photosynthesis?",
            "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"],
            "correct_index": 2,
        },
    ]
})


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "quiz.db")


@pytest.fixture
def engine(db_path):
    return QuizEngine(_FakeLLM(_GOOD_RESPONSE), db_path)


# ---------------------------------------------------------------------------
# QuizStore
# ---------------------------------------------------------------------------

def test_store_add_and_get_question(db_path):
    store = QuizStore(db_path)
    qid = store.add_question("physics", "Q1?", ["a", "b", "c", "d"], 2)
    q = store.get_question(qid)
    assert q["question"] == "Q1?"
    assert q["choices"] == ["a", "b", "c", "d"]
    assert q["correct_index"] == 2
    assert q["subject"] == "physics"


def test_store_get_unknown_question_returns_none(db_path):
    store = QuizStore(db_path)
    assert store.get_question(999) is None


def test_store_get_questions_by_subject(db_path):
    store = QuizStore(db_path)
    store.add_question("physics", "Q1?", ["a", "b", "c", "d"], 0)
    store.add_question("chemistry", "Q2?", ["a", "b", "c", "d"], 1)
    physics_qs = store.get_questions_by_subject("physics")
    assert len(physics_qs) == 1
    assert physics_qs[0]["question"] == "Q1?"


def test_store_record_attempt_and_stats(db_path):
    store = QuizStore(db_path)
    qid = store.add_question("physics", "Q1?", ["a", "b", "c", "d"], 0)
    store.record_attempt(qid, "physics", correct=True, chosen_index=0)
    store.record_attempt(qid, "physics", correct=False, chosen_index=1)
    stats = store.get_subject_stats()
    assert stats == [{"subject": "physics", "attempts": 2, "correct": 1, "accuracy": 0.5}]


def test_store_subject_stats_empty_with_no_attempts(db_path):
    assert QuizStore(db_path).get_subject_stats() == []


def test_store_stats_over_time_chronological(db_path):
    store = QuizStore(db_path)
    qid = store.add_question("physics", "Q1?", ["a", "b", "c", "d"], 0)
    store.record_attempt(qid, "physics", correct=True, chosen_index=0)
    store.record_attempt(qid, "physics", correct=False, chosen_index=1)
    series = store.get_subject_stats_over_time("physics")
    assert len(series) == 2
    assert series[0]["correct"] == 1
    assert series[1]["correct"] == 0


# ---------------------------------------------------------------------------
# QuizEngine.generate_quiz (#34)
# ---------------------------------------------------------------------------

def test_generate_quiz_stores_well_formed_questions(engine):
    questions = engine.generate_quiz(["Water boils at 100C."], subject="physics", n=2)
    assert len(questions) == 2
    assert all("id" in q for q in questions)
    assert questions[0]["subject"] == "physics"


def test_generate_quiz_persists_to_the_store(engine):
    engine.generate_quiz(["Water boils at 100C."], subject="physics", n=2)
    stored = engine.store.get_questions_by_subject("physics")
    assert len(stored) == 2


def test_generate_quiz_respects_n_even_if_model_returns_more(db_path):
    llm = _FakeLLM(_GOOD_RESPONSE)  # returns 2 questions
    engine = QuizEngine(llm, db_path)
    questions = engine.generate_quiz(["material"], subject="physics", n=1)
    assert len(questions) == 1


def test_generate_quiz_with_no_source_material_returns_empty(engine):
    assert engine.generate_quiz([], subject="physics") == []
    assert engine.generate_quiz(["   ", ""], subject="physics") == []


def test_generate_quiz_survives_unparseable_llm_output(db_path):
    engine = QuizEngine(_FakeLLM("not json at all"), db_path)
    assert engine.generate_quiz(["material"], subject="physics") == []


def test_generate_quiz_survives_llm_exception(db_path):
    class _Explodes:
        def generate(self, prompt, fmt=None):
            raise RuntimeError("ollama down")

    engine = QuizEngine(_Explodes(), db_path)
    assert engine.generate_quiz(["material"], subject="physics") == []


def test_generate_quiz_rejects_wrong_choice_count(db_path):
    bad = json.dumps({"questions": [
        {"question": "Q?", "choices": ["a", "b"], "correct_index": 0},
    ]})
    engine = QuizEngine(_FakeLLM(bad), db_path)
    assert engine.generate_quiz(["material"], subject="physics") == []


def test_generate_quiz_rejects_out_of_range_correct_index(db_path):
    bad = json.dumps({"questions": [
        {"question": "Q?", "choices": ["a", "b", "c", "d"], "correct_index": 9},
    ]})
    engine = QuizEngine(_FakeLLM(bad), db_path)
    assert engine.generate_quiz(["material"], subject="physics") == []


def test_generate_quiz_rejects_empty_question_text(db_path):
    bad = json.dumps({"questions": [
        {"question": "  ", "choices": ["a", "b", "c", "d"], "correct_index": 0},
    ]})
    engine = QuizEngine(_FakeLLM(bad), db_path)
    assert engine.generate_quiz(["material"], subject="physics") == []


def test_generate_quiz_skips_bad_questions_but_keeps_good_ones(db_path):
    mixed = json.dumps({"questions": [
        {"question": "Bad", "choices": ["a", "b"], "correct_index": 0},
        {"question": "Good?", "choices": ["a", "b", "c", "d"], "correct_index": 1},
    ]})
    engine = QuizEngine(_FakeLLM(mixed), db_path)
    questions = engine.generate_quiz(["material"], subject="physics")
    assert len(questions) == 1
    assert questions[0]["question"] == "Good?"


def test_generate_quiz_with_non_dict_response_returns_empty(db_path):
    engine = QuizEngine(_FakeLLM(json.dumps(["not", "a", "dict"])), db_path)
    assert engine.generate_quiz(["material"], subject="physics") == []


# ---------------------------------------------------------------------------
# QuizEngine.grade_answer
# ---------------------------------------------------------------------------

def test_grade_answer_correct(engine):
    questions = engine.generate_quiz(["material"], subject="physics", n=1)
    qid = questions[0]["id"]
    correct_index = questions[0]["correct_index"]
    assert engine.grade_answer(qid, correct_index) is True


def test_grade_answer_incorrect(engine):
    questions = engine.generate_quiz(["material"], subject="physics", n=1)
    qid = questions[0]["id"]
    wrong_index = (questions[0]["correct_index"] + 1) % 4
    assert engine.grade_answer(qid, wrong_index) is False


def test_grade_answer_unknown_question_returns_none(engine):
    assert engine.grade_answer(9999, 0) is None


def test_grade_answer_records_the_attempt(engine):
    questions = engine.generate_quiz(["material"], subject="physics", n=1)
    qid = questions[0]["id"]
    engine.grade_answer(qid, questions[0]["correct_index"])
    stats = engine.store.get_subject_stats()
    assert stats[0]["attempts"] == 1


# ---------------------------------------------------------------------------
# Strength/weakness map (#35)
# ---------------------------------------------------------------------------

def _seed_attempts(engine, subject, n_correct, n_wrong):
    for _ in range(n_correct):
        qs = engine.generate_quiz(["material"], subject=subject, n=1)
        engine.grade_answer(qs[0]["id"], qs[0]["correct_index"])
    for _ in range(n_wrong):
        qs = engine.generate_quiz(["material"], subject=subject, n=1)
        wrong = (qs[0]["correct_index"] + 1) % 4
        engine.grade_answer(qs[0]["id"], wrong)


def test_strength_weakness_map_reports_accuracy_per_subject(engine):
    _seed_attempts(engine, "physics", n_correct=3, n_wrong=1)
    _seed_attempts(engine, "chemistry", n_correct=1, n_wrong=3)
    smap = engine.get_strength_weakness_map()
    assert smap["physics"]["accuracy"] == 0.75
    assert smap["chemistry"]["accuracy"] == 0.25


def test_weakest_subjects_sorted_worst_first(engine):
    _seed_attempts(engine, "physics", n_correct=3, n_wrong=1)
    _seed_attempts(engine, "chemistry", n_correct=1, n_wrong=3)
    weakest = engine.get_weakest_subjects()
    assert weakest[0][0] == "chemistry"
    assert weakest[1][0] == "physics"


def test_weakest_subjects_excludes_below_min_attempts(engine):
    _seed_attempts(engine, "physics", n_correct=1, n_wrong=0)  # only 1 attempt
    _seed_attempts(engine, "chemistry", n_correct=1, n_wrong=3)  # 4 attempts
    weakest = engine.get_weakest_subjects(min_attempts=3)
    assert [s for s, _ in weakest] == ["chemistry"]


def test_weakest_subjects_respects_top_n(engine):
    for subj in ("a", "b", "c"):
        _seed_attempts(engine, subj, n_correct=1, n_wrong=2)
    assert len(engine.get_weakest_subjects(top_n=2, min_attempts=1)) == 2


def test_performance_summary_with_no_attempts(engine):
    assert "No quiz attempts" in engine.get_performance_summary()


def test_performance_summary_with_insufficient_samples(engine):
    _seed_attempts(engine, "physics", n_correct=1, n_wrong=0)
    assert "Not enough attempts" in engine.get_performance_summary()


def test_performance_summary_lists_weak_subjects(engine):
    _seed_attempts(engine, "chemistry", n_correct=1, n_wrong=3)
    summary = engine.get_performance_summary()
    assert "chemistry" in summary
    assert "25%" in summary
