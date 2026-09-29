# tests/test_eval_parser.py
"""
Tests for scripts/eval_intents.py's parser and scoring logic (backlog
#21).

These run against the REAL hestia_test_prompts.md file, deterministically
and with no Ollama connection — this is the part of the eval that can and
should run in CI on every commit. The live-model half (actually
classifying each prompt) is a separate, opt-in concern gated by whether
Ollama is reachable; see scripts/eval_intents.py's module docstring for
why that split exists.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.hecate.intent_registry import ALL_INTENTS
from scripts.eval_intents import (
    CaseResult,
    EvalReport,
    EvalCase,
    _resolve_intent,
    parse_golden_dataset,
    run_eval,
)

_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "hestia_test_prompts.md"
)


# ---------------------------------------------------------------------------
# Parsing the real file
# ---------------------------------------------------------------------------

def test_parses_a_substantial_number_of_cases():
    cases = parse_golden_dataset(_DATASET)
    assert len(cases) > 80


def test_finds_module_level_cases_for_every_module():
    cases = parse_golden_dataset(_DATASET)
    modules_seen = {c.expected for c in cases if c.kind == "module"}
    # Every module with a "Per-Module Sanity Checks" section is covered —
    # a module silently dropping out of the parse would shrink this set.
    expected_modules = {
        "core", "mnemosyne", "hermes", "hephaestus", "chronos", "athena",
        "iris", "artemis", "ares", "apollo", "orpheus", "metis",
        "dionysus", "pluto",
    }
    assert expected_modules <= modules_seen


def test_hecate_itself_contributes_no_module_cases():
    # Its section explicitly says "No direct prompts" — nothing should be
    # attributed to it.
    cases = parse_golden_dataset(_DATASET)
    assert not any(c.expected == "hecate" for c in cases if c.kind == "module")


def test_finds_the_known_explicit_intent_cases():
    cases = parse_golden_dataset(_DATASET)
    by_prompt = {c.prompt: c for c in cases if c.kind == "intent"}
    assert by_prompt["List my habits"].expected == "list_habits"
    assert by_prompt["List my habits"].forbidden == "get_goals"
    assert by_prompt["Give up on my goal to learn guitar"].expected == "abandon_goal"


def test_every_intent_level_expected_value_is_a_real_registered_intent():
    # The single most important invariant this parser has to hold: every
    # ground-truth label it produces must be something the NLU can
    # actually emit, or every "failure" the eval reports would be a false
    # one caused by the parser, not the model.
    cases = parse_golden_dataset(_DATASET)
    bad = [
        c for c in cases
        if c.kind == "intent" and c.expected not in ALL_INTENTS
    ]
    assert not bad, [c.prompt for c in bad]


def test_every_module_level_expected_value_is_a_real_registered_module():
    from modules.hecate.intent_registry import INTENT_MODULE_MAP
    real_modules = set(INTENT_MODULE_MAP.values())
    cases = parse_golden_dataset(_DATASET)
    bad = [
        c for c in cases
        if c.kind == "module" and c.expected not in real_modules
    ]
    assert not bad, [c.prompt for c in bad]


def test_ambiguous_hedged_lines_are_not_scored_as_ground_truth():
    # A line containing "? Test both readings" (or similar) must land as
    # "manual", never be forced into a single pass/fail answer.
    cases = parse_golden_dataset(_DATASET)
    hedged = [c for c in cases if "Test both readings" in c.note]
    assert hedged
    assert all(c.kind == "manual" for c in hedged)


def test_manual_cases_still_carry_the_prompt_text():
    # They're not scored, but they should still be runnable/loggable.
    cases = parse_golden_dataset(_DATASET)
    manual = [c for c in cases if c.kind == "manual"]
    assert manual
    assert all(c.prompt for c in manual)


def test_missing_file_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_golden_dataset(tmp_path / "does_not_exist.md")


# ---------------------------------------------------------------------------
# Intent-name resolution (bare markdown form -> registered prefixed form)
# ---------------------------------------------------------------------------

def test_resolve_intent_passes_through_an_already_registered_name():
    assert _resolve_intent("list_habits") == "list_habits"


def test_resolve_intent_adds_a_known_unambiguous_prefix():
    assert _resolve_intent("rewrite_style") == "orpheus_rewrite_style"
    assert _resolve_intent("rewrite_text") == "metis_rewrite_text"


def test_resolve_intent_leaves_an_unresolvable_name_unchanged():
    assert _resolve_intent("not_a_real_intent_at_all") == "not_a_real_intent_at_all"


# ---------------------------------------------------------------------------
# Scoring (run_eval against a fake NLU/Hecate — no Ollama)
# ---------------------------------------------------------------------------

class _FakeNLU:
    """Returns a fixed intent/confidence per exact prompt text."""

    def __init__(self, answers: dict):
        self.answers = answers

    def understand(self, text, context=None):
        return self.answers.get(text, {"intent": "chat", "confidence": 0.5})


class _FakeHecate:
    def __init__(self, module_of: dict):
        self.module_of = module_of

    def decide(self, text, nlu_result, active_modules):
        intent = nlu_result.get("intent", "chat")
        return {"primary": self.module_of.get(intent, "core")}


def test_run_eval_scores_a_correct_module_case():
    cases = [EvalCase(prompt="hi", kind="module", section=1, expected="apollo")]
    nlu = _FakeNLU({"hi": {"intent": "apollo_log_workout", "confidence": 0.9}})
    hecate = _FakeHecate({"apollo_log_workout": "apollo"})
    report = run_eval(nlu, hecate, cases, ["apollo"])
    assert report.accuracy == 1.0
    assert report.results[0].passed is True


def test_run_eval_scores_an_incorrect_module_case():
    cases = [EvalCase(prompt="hi", kind="module", section=1, expected="apollo")]
    nlu = _FakeNLU({"hi": {"intent": "pluto_log_expense", "confidence": 0.9}})
    hecate = _FakeHecate({"pluto_log_expense": "pluto"})
    report = run_eval(nlu, hecate, cases, ["apollo", "pluto"])
    assert report.accuracy == 0.0
    assert report.results[0].passed is False


def test_run_eval_scores_an_intent_case_on_exact_intent_not_module():
    cases = [EvalCase(prompt="hi", kind="intent", section=2, expected="list_habits")]
    nlu = _FakeNLU({"hi": {"intent": "list_habits", "confidence": 0.9}})
    hecate = _FakeHecate({"list_habits": "artemis"})
    report = run_eval(nlu, hecate, cases, ["artemis"])
    assert report.accuracy == 1.0


def test_run_eval_fails_when_the_forbidden_intent_is_chosen():
    cases = [EvalCase(
        prompt="List my habits", kind="intent", section=2,
        expected="list_habits", forbidden="get_goals",
    )]
    nlu = _FakeNLU({"List my habits": {"intent": "get_goals", "confidence": 0.7}})
    hecate = _FakeHecate({"get_goals": "artemis"})
    report = run_eval(nlu, hecate, cases, ["artemis"])
    assert report.accuracy == 0.0


def test_run_eval_never_scores_manual_cases():
    cases = [EvalCase(prompt="hi", kind="manual", section=3)]
    nlu = _FakeNLU({})
    hecate = _FakeHecate({})
    report = run_eval(nlu, hecate, cases, [])
    assert report.results[0].passed is None
    assert report.scored == []
    assert report.accuracy == 1.0  # vacuous — nothing scored, not a failure


def test_run_eval_survives_an_nlu_exception():
    class _Explodes:
        def understand(self, text, context=None):
            raise RuntimeError("ollama down")

    cases = [EvalCase(prompt="hi", kind="module", section=1, expected="apollo")]
    report = run_eval(_Explodes(), _FakeHecate({}), cases, [])
    assert report.results[0].passed is False
    assert "error" in report.results[0].intent


def test_report_format_lists_failures():
    cases = [EvalCase(prompt="hi", kind="module", section=1, expected="apollo")]
    nlu = _FakeNLU({"hi": {"intent": "pluto_log_expense", "confidence": 0.9}})
    hecate = _FakeHecate({"pluto_log_expense": "pluto"})
    report = run_eval(nlu, hecate, cases, ["apollo", "pluto"])
    text = report.format()
    assert "1 failure" in text
    assert "hi" in text


def test_report_format_with_no_failures():
    cases = [EvalCase(prompt="hi", kind="module", section=1, expected="apollo")]
    nlu = _FakeNLU({"hi": {"intent": "apollo_log_workout", "confidence": 0.9}})
    hecate = _FakeHecate({"apollo_log_workout": "apollo"})
    report = run_eval(nlu, hecate, cases, ["apollo"])
    assert "No failures" in report.format()


def test_confusion_matrix_only_covers_intent_level_cases():
    cases = [
        EvalCase(prompt="a", kind="module", section=1, expected="apollo"),
        EvalCase(prompt="b", kind="intent", section=2, expected="list_habits"),
    ]
    nlu = _FakeNLU({
        "a": {"intent": "apollo_log_workout", "confidence": 0.9},
        "b": {"intent": "get_goals", "confidence": 0.6},
    })
    hecate = _FakeHecate({"apollo_log_workout": "apollo", "get_goals": "artemis"})
    report = run_eval(nlu, hecate, cases, ["apollo", "artemis"])
    matrix = report.confusion()
    assert matrix == {"list_habits": {"get_goals": 1}}
    # The module-level case ("a" -> apollo) never enters the intent
    # confusion matrix at all — only "b"'s intent-level row is present.
    assert len(matrix) == 1
