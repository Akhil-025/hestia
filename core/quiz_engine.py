"""
core/quiz_engine.py

Quiz generation from ingested content, with score-history tracking and a
per-subject strength/weakness map (backlog #34, #35).

Wiring
------
The user-facing side lives in ``modules/mnemosyne/extensions.py``: the
``start_quiz`` / ``answer_quiz`` / ``quiz_performance`` intents (registered
in ``modules/hecate/intent_registry.py``) drive this engine, and a quiz is a
multi-turn conversation built on the orchestrator's existing slot-fill
mechanism (each question asks for an ``answer`` slot). This module stays
free of any orchestrator knowledge — it is generation, storage, scoring and
the pure helpers for formatting a question and parsing a spoken answer.

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

import difflib
import json
import logging
import random
import re
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
  {{"question": "...", "choices": ["...", "...", "...", "..."], "correct_index": 2}}
]}}

correct_index is the 0-based position of the correct choice; vary it between
questions.

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
    def __init__(
        self, llm: _LLM, db_path: str,
        rng: Optional[random.Random] = None, shuffle_choices: bool = True,
    ) -> None:
        self.llm = llm
        self.store = QuizStore(db_path)
        # Small local models very often make choice A the right answer
        # (the prompt's own example nudges them that way). Shuffling after
        # validation and remapping correct_index removes that tell, so a
        # learner can't score by always picking A. `rng` is injectable so
        # tests are deterministic.
        self._rng = rng or random.Random()
        self._shuffle = shuffle_choices

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
            if self._shuffle:
                validated = self._shuffled(validated)
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

    def _shuffled(self, q: dict) -> dict:
        """Shuffle the choices and remap correct_index to follow the right answer."""
        order = list(range(len(q["choices"])))
        self._rng.shuffle(order)
        return {
            "question": q["question"],
            "choices": [q["choices"][i] for i in order],
            "correct_index": order.index(q["correct_index"]),
        }

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

    def get_question(self, question_id: int) -> Optional[dict]:
        return self.store.get_question(question_id)

    # ------------------------------------------------------------------
    # Strength/weakness map (#35)
    # ------------------------------------------------------------------

    def get_subject_trend(self, subject: str, window: int = 5) -> Optional[dict]:
        """
        Is the learner improving in *subject*? Compares accuracy over the
        most recent *window* attempts with the *window* before them.
        Returns None until there are at least ``2 * window`` attempts —
        a trend from fewer is noise, the same reasoning as min_attempts in
        get_weakest_subjects.
        """
        series = [bool(a["correct"]) for a in self.store.get_subject_stats_over_time(subject)]
        if len(series) < 2 * window:
            return None
        recent = series[-window:]
        earlier = series[-2 * window:-window]
        r, e = sum(recent) / window, sum(earlier) / window
        direction = "up" if r - e >= 0.2 else "down" if e - r >= 0.2 else "flat"
        return {"recent": round(r, 3), "earlier": round(e, 3), "direction": direction}

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


# ---------------------------------------------------------------------------
# Pure helpers for the conversational layer
# ---------------------------------------------------------------------------

LETTERS = "ABCD"

SKIP_WORDS = frozenset({"skip", "pass", "next", "idk", "i don't know", "i dont know", "no idea"})

# What speech-to-text commonly returns for a spoken letter, accepted only
# when the whole reply is that word (or follows "option"/"answer"/"is"), so
# an ordinary sentence containing "see" or "be" is never read as a choice.
_LETTER_HOMOPHONES = {
    "a": "a", "ay": "a", "eh": "a",
    "b": "b", "be": "b", "bee": "b",
    "c": "c", "see": "c", "sea": "c", "cee": "c",
    "d": "d", "dee": "d",
}
_ORDINALS = {
    "first": 0, "1st": 0, "1": 0, "one": 0,
    "second": 1, "2nd": 1, "2": 1, "two": 1,
    "third": 2, "3rd": 2, "3": 2, "three": 2,
    "fourth": 3, "4th": 3, "4": 3, "four": 3, "last": 3,
}
_LEAD_IN_RE = re.compile(
    r"^(?:(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s+(?:is\s+)?"
    r"|it(?:'s|\s+is)\s+|i\s+(?:think|say|pick|choose|guess|'ll\s+go\s+with)\s+(?:it(?:'s|\s+is)\s+)?"
    r"|(?:go\s+with|going\s+with)\s+|my\s+answer\s+is\s+)+",
)


def format_question(question: dict, number: int, total: int) -> str:
    """'Question 2 of 5: ... A) ... B) ... C) ... D) ...' — readable aloud and on screen."""
    lines = [f"Question {number} of {total}: {question['question']}"]
    for letter, choice in zip(LETTERS, question["choices"]):
        lines.append(f"{letter}) {choice}")
    return "\n".join(lines)


def is_skip(text: str) -> bool:
    return (text or "").strip().lower().strip(".!?") in SKIP_WORDS


def parse_choice(text: str, choices: list[str]) -> Optional[int]:
    """
    Map a spoken/typed reply to a choice index, or None if it can't be
    decided. Understands letters ("B", "option b", "it's c"), STT
    homophones ("bee"), ordinals ("the second one", "3"), and the choice
    text itself (exact, contained, or a close fuzzy match). Ambiguity is
    None — the caller re-asks rather than guessing, since a wrongly
    graded answer would corrupt the strength/weakness map.
    """
    if not text or not choices:
        return None
    cleaned = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None

    stripped = _LEAD_IN_RE.sub("", cleaned).strip()
    # "the second one" / "number 3" -> "second" / "3"
    stripped = re.sub(r"^(?:the|number|no)\s+", "", stripped)
    words = stripped.split()
    if len(words) > 1 and words[-1] == "one":
        stripped = " ".join(words[:-1])

    # Letter (or homophone) as the whole remaining reply.
    if stripped in _LETTER_HOMOPHONES:
        idx = "abcd".index(_LETTER_HOMOPHONES[stripped])
        return idx if idx < len(choices) else None
    # Ordinal / number as the whole remaining reply ("the second one").
    if stripped in _ORDINALS:
        idx = _ORDINALS[stripped]
        return idx if idx < len(choices) else None

    norm_choices = [re.sub(r"[^\w\s']", " ", c.lower()).split() for c in choices]
    norm_choices = [" ".join(c) for c in norm_choices]

    # Exact text.
    exact = [i for i, c in enumerate(norm_choices) if c and c == stripped]
    if len(exact) == 1:
        return exact[0]
    # The reply contains exactly one choice's text ("I think it's carbon dioxide").
    contained = [i for i, c in enumerate(norm_choices) if c and f" {c} " in f" {cleaned} "]
    if len(contained) == 1:
        return contained[0]
    # Close fuzzy match, unambiguous.
    scored = sorted(
        ((difflib.SequenceMatcher(None, stripped, c).ratio(), i) for i, c in enumerate(norm_choices)),
        reverse=True,
    )
    if scored and scored[0][0] >= 0.8 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.1):
        return scored[0][1]
    return None

