"""
core/quiz_engine.py

Quiz generation from ingested content, with score-history tracking and a
per-subject strength/weakness map (backlog #34, #35).

Scope note
------------
This is the generation + storage + scoring engine — the substantive part
of both backlog items. It is NOT yet wired into the orchestrator as user-
facing intents ("quiz me on thermodynamics"): that needs a new registered
module, new intent_registry.py entries, and new config/nlu_prompt.txt
few-shot examples, following the exact checklist CONTRIBUTING.md already
documents for adding an intent. Left as clearly-flagged follow-up work
rather than rushed alongside everything else in this batch — the engine
below is fully functional and tested standalone (see
tests/test_quiz_engine.py) and ready to be called from a handler once
that wiring is done.

Source content
----------------
`generate_quiz` takes source TEXT directly (a list of strings) rather
than reaching into Athena's document index itself — that keeps this
module decoupled from Athena's internals and testable without a real RAG
index. The intended caller (once wired) is Athena's search/query results
feeding this function's `source_texts` parameter, per the backlog's "pull
N facts, generate multiple-choice via Ollama" — "pull" is the caller's
job; "generate" is this module's.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Protocol

from core.quiz_store import QuizStore

logger = logging.getLogger(__name__)


class _LLM(Protocol):
    def generate(self, prompt: str, fmt: Optional[str] = None) -> str: ...


_QUIZ_PROMPT_TEMPLATE = """\
Generate {n} multiple-choice quiz questions based ONLY on the material
below. Each question must have exactly 4 options with exactly one correct
answer. Return ONLY valid JSON, no preamble, in this exact shape:

{{"questions": [
  {{"question": "...", "choices": ["...", "...", "...", "..."], "correct_index": 0}}
]}}

Material:
{material}
"""

# A generated quiz is only as good as the questions actually being
# well-formed — this filters out obviously broken model output (wrong
# choice count, an out-of-range correct_index) rather than trusting the
# JSON shape blindly, since a small local model can and does drift from
# the requested schema under load.
_REQUIRED_CHOICE_COUNT = 4


class QuizEngine:
    def __init__(self, llm: _LLM, db_path: str) -> None:
        self.llm = llm
        self.store = QuizStore(db_path)

    # ------------------------------------------------------------------
    # Generation (#34)
    # ------------------------------------------------------------------

    def generate_quiz(
        self, source_texts: list[str], subject: str, n: int = 5
    ) -> list[dict]:
        """
        Generate up to *n* multiple-choice questions from *source_texts*
        and persist them under *subject*. Returns the stored questions
        (each with an "id"), fewer than *n* if the model produced fewer
        well-formed ones or none at all on a parse/generation failure —
        never raises.
        """
        material = "\n\n".join(t.strip() for t in source_texts if t and t.strip())
        if not material:
            logger.info("generate_quiz: no usable source material for subject=%s", subject)
            return []

        prompt = _QUIZ_PROMPT_TEMPLATE.format(n=n, material=material[:6000])
        try:
            raw = self.llm.generate(prompt, fmt="json")
            parsed = json.loads(raw)
        except Exception:
            logger.exception("generate_quiz: LLM call or JSON parse failed for subject=%s", subject)
            return []

        candidates = parsed.get("questions", []) if isinstance(parsed, dict) else []
        stored: list[dict] = []
        for q in candidates[:n]:
            validated = self._validate_question(q)
            if validated is None:
                continue
            qid = self.store.add_question(
                subject=subject,
                question=validated["question"],
                choices=validated["choices"],
                correct_index=validated["correct_index"],
                source_snippet=material[:500],
            )
            stored.append({**validated, "id": qid, "subject": subject})

        if not stored:
            logger.warning("generate_quiz: model produced no well-formed questions for subject=%s", subject)
        return stored

    @staticmethod
    def _validate_question(q: Any) -> Optional[dict]:
        """Reject a malformed generated question rather than storing garbage."""
        if not isinstance(q, dict):
            return None
        question = str(q.get("question", "")).strip()
        choices = q.get("choices")
        correct_index = q.get("correct_index")
        if not question:
            return None
        if not isinstance(choices, list) or len(choices) != _REQUIRED_CHOICE_COUNT:
            return None
        if not all(isinstance(c, str) and c.strip() for c in choices):
            return None
        if not isinstance(correct_index, int) or not (0 <= correct_index < len(choices)):
            return None
        return {"question": question, "choices": choices, "correct_index": correct_index}

    # ------------------------------------------------------------------
    # Answering / scoring
    # ------------------------------------------------------------------

    def grade_answer(self, question_id: int, chosen_index: int) -> Optional[bool]:
        """
        Record an attempt and return whether it was correct, or None if
        *question_id* doesn't exist (never raises for a bad id — a
        misheard/garbled question reference shouldn't crash the quiz flow).
        """
        question = self.store.get_question(question_id)
        if question is None:
            logger.warning("grade_answer: unknown question_id=%s", question_id)
            return None
        correct = chosen_index == question["correct_index"]
        self.store.record_attempt(
            question_id=question_id, subject=question["subject"],
            correct=correct, chosen_index=chosen_index,
        )
        return correct

    # ------------------------------------------------------------------
    # Strength/weakness map (#35)
    # ------------------------------------------------------------------

    def get_strength_weakness_map(self) -> dict[str, dict]:
        """{subject: {attempts, correct, accuracy}} for every attempted subject."""
        return {s["subject"]: s for s in self.store.get_subject_stats()}

    def get_weakest_subjects(
        self, top_n: int = 3, min_attempts: int = 3
    ) -> list[tuple[str, dict]]:
        """
        The *top_n* lowest-accuracy subjects with at least *min_attempts*
        attempts, worst first — a single quiz answer isn't a "weakness",
        it's noise; this is a rate, not an incident.
        """
        stats = [s for s in self.store.get_subject_stats() if s["attempts"] >= min_attempts]
        ranked = sorted(stats, key=lambda s: (s["accuracy"], -s["attempts"]))
        return [(s["subject"], s) for s in ranked[:top_n]]

    def get_performance_summary(self) -> str:
        """Prose summary — the shape a morning-brief-style digest or a spoken answer would use."""
        weakest = self.get_weakest_subjects()
        if not weakest:
            all_stats = self.store.get_subject_stats()
            if not all_stats:
                return "No quiz attempts recorded yet."
            return "Not enough attempts per subject yet to identify weak areas."

        lines = ["Weakest subject(s):"]
        for subject, stats in weakest:
            lines.append(
                f"  {subject}: {stats['accuracy']:.0%} "
                f"({stats['correct']}/{stats['attempts']})"
            )
        return "\n".join(lines)
