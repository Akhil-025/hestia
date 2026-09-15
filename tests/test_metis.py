# tests/test_metis.py
"""
Extensive regression tests for modules/metis/engine.py — Metis, Hestia's
writing-assistance module (correction, clarity, style, tone, rewriting,
content drafting, summarising/expanding/shortening, outlining, citation
formatting, consistency checks, readability, and writing stats).

This module previously had zero test coverage. These tests cover every
public intent's happy path, clarification path, and error path (mirroring
Orpheus's already-tested LLM-failure/malformed-JSON conventions, since
Metis explicitly shares that design), the module-level pure formatting
helpers, `_normalise`'s prefix-matching behaviour, and MetisDB's schema
and stats aggregation, all via an injected FakeLLM (bypassing
core.ollama_client.generate entirely) plus an isolated temp-file SQLite DB.

Run with:  pytest tests/test_metis.py -v
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.metis.engine import (
    MetisEngine,
    LLMResponseError,
    _extract,
    _normalise,
    _collect_missing,
    _clarify,
    _format_corrections,
    _format_style_suggestions,
    _format_outline,
    _format_consistency,
    _format_readability,
    _format_stats,
    _ok,
    _err,
)
from modules.metis.db import MetisDB


# ===========================================================================
# Fakes / fixtures
# ===========================================================================

class FakeLLM:
    """Stand-in for HestiaLLM — records prompts, returns canned JSON/text."""

    def __init__(self, response: str = ""):
        self.response = response
        self.responses = None  # optional queue for sequential different responses
        self.last_prompt = None
        self.last_fmt = None
        self.call_count = 0

    def generate(self, prompt: str, fmt: str = None) -> str:
        self.last_prompt = prompt
        self.last_fmt = fmt
        self.call_count += 1
        if self.responses is not None:
            return self.responses[self.call_count - 1]
        return self.response


def make_engine(tmp_path, llm_response="", memory=None):
    """Build a MetisEngine with a real temp-file SQLite DB and an injected
    FakeLLM (bypassing core.ollama_client.generate entirely)."""
    engine = MetisEngine(
        ollama_cfg={}, memory=memory,
        db_path=tmp_path / "metis_test.db",
        llm=FakeLLM(llm_response),
    )
    return engine, engine._llm_instance


# ===========================================================================
# BaseModule contract
# ===========================================================================

class TestMetisContract:
    ALL_INTENTS = (
        "correct_text", "improve_clarity", "suggest_style", "detect_tone",
        "rewrite_text", "draft_content", "summarize_text", "expand_text",
        "shorten_text", "generate_outline", "check_plagiarism",
        "generate_citation", "check_consistency", "readability_report",
        "writing_stats",
    )

    def test_can_handle_all_registered_intents(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        for intent in self.ALL_INTENTS:
            assert engine.can_handle(intent) is True

    def test_can_handle_rejects_unknown_intent(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        assert engine.can_handle("not_a_real_intent") is False

    def test_name_is_metis(self):
        assert MetisEngine.name == "metis"

    def test_handle_unknown_intent_returns_zero_confidence(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("bogus_intent", {}, {})
        assert result["confidence"] == 0.0
        assert result["data"] == {}

    def test_handle_never_raises_on_unexpected_exception(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch.object(engine, "_dispatch", side_effect=RuntimeError("boom")):
            result = engine.handle("correct_text", {"text": "x"}, {})
        assert result["confidence"] == 0.0
        assert "went wrong" in result["response"].lower()

    def test_get_context_reports_recent_types_and_total(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps(
            {"corrected": "Fixed.", "changes": []}
        ))
        engine.handle("correct_text", {"text": "fix me"}, {})
        ctx = engine.get_context()
        assert ctx["metis_total_items"] == 1
        assert ctx["metis_recent_types"] == ["correction"]

    def test_get_context_returns_empty_dict_on_db_failure(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch.object(engine.db, "get_all", side_effect=RuntimeError("db down")):
            assert engine.get_context() == {}

    def test_get_context_empty_when_no_activity(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        ctx = engine.get_context()
        assert ctx["metis_total_items"] == 0
        assert ctx["metis_recent_types"] == []


# ===========================================================================
# correct_text
# ===========================================================================

class TestCorrectText:
    def test_happy_path_with_changes(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "corrected": "I have gone home.",
            "changes": [{"type": "grammar", "original": "have went",
                         "suggestion": "have gone"}],
        }))
        result = engine.handle("correct_text", {"text": "I have went home."}, {})
        assert result["confidence"] == 0.93
        assert "I have gone home." in result["response"]
        assert "have went" in result["response"]
        assert result["data"]["corrected"] == "I have gone home."
        assert len(result["data"]["changes"]) == 1

    def test_no_errors_found(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "corrected": "Perfect already.", "changes": [],
        }))
        result = engine.handle("correct_text", {"text": "Perfect already."}, {})
        assert "No errors found." in result["response"]

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("correct_text", {}, {})
        assert result["confidence"] == 0.6
        assert result["data"]["needs_clarification"] is True

    def test_uses_content_entity_fallback(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps(
            {"corrected": "ok", "changes": []}
        ))
        engine.handle("correct_text", {"content": "some text"}, {})
        assert "some text" in llm.last_prompt

    def test_uses_raw_query_fallback(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps(
            {"corrected": "ok", "changes": []}
        ))
        engine.handle("correct_text", {"raw_query": "raw text here"}, {})
        assert "raw text here" in llm.last_prompt

    def test_llm_empty_response_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("correct_text", {"text": "x"}, {})
        assert result["confidence"] == 0.0
        assert "trouble correcting" in result["response"].lower()

    def test_malformed_json_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="not json at all")
        result = engine.handle("correct_text", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_empty_corrected_field_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps(
            {"corrected": "", "changes": []}
        ))
        result = engine.handle("correct_text", {"text": "x"}, {})
        assert result["confidence"] == 0.0
        assert "came back empty" in result["response"].lower()

    def test_non_dict_changes_entries_are_filtered(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "corrected": "ok", "changes": ["not a dict", {"type": "grammar"}],
        }))
        result = engine.handle("correct_text", {"text": "x"}, {})
        assert len(result["data"]["changes"]) == 1

    def test_uses_json_fmt_for_llm_call(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps(
            {"corrected": "ok", "changes": []}
        ))
        engine.handle("correct_text", {"text": "x"}, {})
        assert llm.last_fmt == "json"

    def test_persists_to_db_and_memory_for_worthy_type(self, tmp_path):
        # "correction" is not in _MEMORY_WORTHY_TYPES, so memory should
        # NOT be called for correct_text specifically.
        mem = MagicMock()
        engine, _ = make_engine(tmp_path, llm_response=json.dumps(
            {"corrected": "ok", "changes": []}
        ), memory=mem)
        engine.handle("correct_text", {"text": "x"}, {})
        mem.learn.assert_not_called()
        assert engine.db.get_stats()["total"] == 1

    def test_db_write_failure_does_not_withhold_output(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps(
            {"corrected": "Still returned.", "changes": []}
        ))
        with patch.object(engine.db, "save", side_effect=RuntimeError("disk full")):
            result = engine.handle("correct_text", {"text": "x"}, {})
        assert result["confidence"] == 0.93
        assert "Still returned." in result["response"]


# ===========================================================================
# improve_clarity
# ===========================================================================

class TestImproveClarity:
    def test_happy_path_with_notes(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "revised": "Clear text.", "notes": ["removed redundancy"],
        }))
        result = engine.handle("improve_clarity", {"text": "wordy text"}, {})
        assert result["confidence"] == 0.92
        assert "Clear text." in result["response"]
        assert "removed redundancy" in result["response"]

    def test_happy_path_without_notes(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "revised": "Clear text.", "notes": [],
        }))
        result = engine.handle("improve_clarity", {"text": "wordy text"}, {})
        assert result["response"] == "Clear text."

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("improve_clarity", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("improve_clarity", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_empty_revised_field_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps(
            {"revised": "", "notes": []}
        ))
        result = engine.handle("improve_clarity", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_non_string_notes_are_filtered(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "revised": "ok", "notes": ["good note", 123, None],
        }))
        result = engine.handle("improve_clarity", {"text": "x"}, {})
        assert result["data"]["notes"] == ["good note"]


# ===========================================================================
# suggest_style
# ===========================================================================

class TestSuggestStyle:
    def test_happy_path(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "suggestions": [{"category": "voice", "issue": "too passive",
                              "suggestion": "use active voice"}],
            "revised": "Active voice text.",
        }))
        result = engine.handle("suggest_style", {"text": "text was written"}, {})
        assert result["confidence"] == 0.9
        assert "too passive" in result["response"]
        assert "Active voice text." in result["response"]

    def test_no_suggestions_falls_back_to_revised_or_message(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "suggestions": [], "revised": "",
        }))
        result = engine.handle("suggest_style", {"text": "x"}, {})
        assert result["response"] == "No style issues found."

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("suggest_style", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("suggest_style", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_saves_revised_text_when_present(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "suggestions": [], "revised": "Revised version.",
        }))
        engine.handle("suggest_style", {"text": "original"}, {})
        rows = engine.db.get_all()
        assert rows[0]["content"] == "Revised version."

    def test_saves_original_text_when_no_revised(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "suggestions": [], "revised": "",
        }))
        engine.handle("suggest_style", {"text": "original text"}, {})
        rows = engine.db.get_all()
        assert rows[0]["content"] == "original text"


# ===========================================================================
# detect_tone
# ===========================================================================

class TestDetectTone:
    def test_detection_only_without_target(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "detected_tone": "casual", "description": "reads informally.",
            "suggested_target": "professional",
        }))
        result = engine.handle("detect_tone", {"text": "hey whats up"}, {})
        assert "Detected tone: casual." in result["response"]
        assert "reads informally." in result["response"]
        assert "more professional tone might land better" in result["response"]
        assert llm.call_count == 1  # no tone-shift call made

    def test_detection_with_target_also_rewrites(self, tmp_path):
        engine, llm = make_engine(tmp_path)
        llm.responses = [
            json.dumps({"detected_tone": "casual", "description": "informal",
                        "suggested_target": "formal"}),
            "Formal rewritten text.",
        ]
        result = engine.handle(
            "detect_tone", {"text": "hey", "target_tone": "professional"}, {}
        )
        assert "Formal rewritten text." in result["response"]
        assert result["data"]["rewritten"] == "Formal rewritten text."
        assert llm.call_count == 2

    def test_invalid_target_tone_normalises_to_empty_and_skips_shift(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "detected_tone": "casual", "description": "", "suggested_target": "",
        }))
        engine.handle("detect_tone", {"text": "hey", "target_tone": "gibberish_xyz"}, {})
        assert llm.call_count == 1  # target didn't normalise -> no shift call

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("detect_tone", {}, {})
        assert result["confidence"] == 0.6

    def test_detection_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("detect_tone", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_tone_shift_failure_does_not_break_detection_response(self, tmp_path):
        """If the second (tone-shift) LLM call fails, detection still
        succeeds and the response degrades gracefully rather than erroring
        out entirely."""
        engine, llm = make_engine(tmp_path)
        llm.responses = [
            json.dumps({"detected_tone": "casual", "description": "informal",
                        "suggested_target": "formal"}),
            "",  # tone-shift call returns empty -> raises LLMResponseError, caught
        ]
        result = engine.handle(
            "detect_tone", {"text": "hey", "target_tone": "professional"}, {}
        )
        assert result["confidence"] == 0.9
        assert "Detected tone: casual." in result["response"]
        assert result["data"]["rewritten"] == ""

    def test_non_dict_detect_result_defaults_to_unclear(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps(["not", "a", "dict"]))
        result = engine.handle("detect_tone", {"text": "x"}, {})
        assert result["data"]["detected_tone"] == "unclear"


# ===========================================================================
# rewrite_text
# ===========================================================================

class TestRewriteText:
    def test_happy_path_default_goal(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="Clearer version.")
        result = engine.handle("rewrite_text", {"text": "murky text"}, {})
        assert result["confidence"] == 0.93
        assert result["response"] == "Clearer version."
        assert result["data"]["goal"] == "clarity"
        assert "clarity" in llm.last_prompt

    def test_valid_goal_is_passed_through(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="Confident version.")
        engine.handle("rewrite_text", {"text": "x", "goal": "confidence"}, {})
        assert "confidence" in llm.last_prompt

    def test_invalid_goal_falls_back_to_default(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="x")
        result = engine.handle("rewrite_text", {"text": "x", "goal": "gibberish"}, {})
        assert result["data"]["goal"] == "clarity"

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("rewrite_text", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("rewrite_text", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_rewrite_is_memory_worthy_type(self, tmp_path):
        mem = MagicMock()
        engine, _ = make_engine(tmp_path, llm_response="Result text.", memory=mem)
        engine.handle("rewrite_text", {"text": "x"}, {})
        mem.learn.assert_called_once()

    def test_memory_failure_does_not_break_response(self, tmp_path):
        mem = MagicMock()
        mem.learn.side_effect = RuntimeError("memory down")
        engine, _ = make_engine(tmp_path, llm_response="Result text.", memory=mem)
        result = engine.handle("rewrite_text", {"text": "x"}, {})
        assert result["confidence"] == 0.93

    def test_no_memory_injected_is_a_noop(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="Result text.", memory=None)
        result = engine.handle("rewrite_text", {"text": "x"}, {})
        assert result["confidence"] == 0.93


# ===========================================================================
# draft_content
# ===========================================================================

class TestDraftContent:
    def test_happy_path(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="Dear Sir, ...")
        result = engine.handle(
            "draft_content",
            {"brief": "ask for a raise", "content_type": "email", "tone": "confident"},
            {},
        )
        assert result["confidence"] == 0.92
        assert "Dear Sir, ..." in result["response"]
        assert result["data"]["content_type"] == "email"
        assert "confident" in llm.last_prompt

    def test_default_content_type_is_email(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="x")
        result = engine.handle("draft_content", {"brief": "something"}, {})
        assert result["data"]["content_type"] == "email"

    def test_type_entity_used_as_fallback_for_content_type(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="x")
        result = engine.handle(
            "draft_content", {"brief": "b", "type": "blog"}, {}
        )
        assert result["data"]["content_type"] == "blog"

    def test_title_includes_content_type_and_brief_excerpt(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="body text")
        result = engine.handle(
            "draft_content", {"brief": "quarterly update", "content_type": "report"}, {}
        )
        assert "Report" in result["data"]["title"]
        assert "quarterly update" in result["data"]["title"]

    def test_missing_brief_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("draft_content", {}, {})
        assert result["confidence"] == 0.6

    def test_uses_topic_as_brief_fallback(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="x")
        engine.handle("draft_content", {"topic": "product launch"}, {})
        assert "product launch" in llm.last_prompt

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("draft_content", {"brief": "x"}, {})
        assert result["confidence"] == 0.0

    def test_draft_is_memory_worthy_type(self, tmp_path):
        mem = MagicMock()
        engine, _ = make_engine(tmp_path, llm_response="x", memory=mem)
        engine.handle("draft_content", {"brief": "b"}, {})
        mem.learn.assert_called_once()


# ===========================================================================
# summarize_text / expand_text / shorten_text
# ===========================================================================

class TestSummarizeExpandShorten:
    def test_summarize_happy_path(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="Short summary.")
        result = engine.handle("summarize_text", {"text": "long article " * 20}, {})
        assert result["confidence"] == 0.93
        assert result["response"] == "Short summary."

    def test_summarize_default_length_is_short(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="x")
        engine.handle("summarize_text", {"text": "long text"}, {})
        assert "short" in llm.last_prompt

    def test_summarize_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("summarize_text", {}, {})
        assert result["confidence"] == 0.6

    def test_summarize_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("summarize_text", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_expand_happy_path(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="Expanded with detail.")
        result = engine.handle("expand_text", {"text": "brief note"}, {})
        assert result["confidence"] == 0.9
        assert result["response"] == "Expanded with detail."

    def test_expand_default_length_is_medium(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="x")
        engine.handle("expand_text", {"text": "note"}, {})
        assert "medium" in llm.last_prompt

    def test_expand_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("expand_text", {}, {})
        assert result["confidence"] == 0.6

    def test_shorten_happy_path(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="Short.")
        result = engine.handle("shorten_text", {"text": "long winded text " * 10}, {})
        assert result["confidence"] == 0.9
        assert result["response"] == "Short."

    def test_shorten_default_length_is_short(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="x")
        engine.handle("shorten_text", {"text": "text"}, {})
        assert "short" in llm.last_prompt

    def test_shorten_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("shorten_text", {}, {})
        assert result["confidence"] == 0.6

    def test_shorten_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("shorten_text", {"text": "x"}, {})
        assert result["confidence"] == 0.0


# ===========================================================================
# generate_outline
# ===========================================================================

class TestGenerateOutline:
    def test_happy_path(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "title": "My Outline",
            "sections": [{"heading": "Intro", "points": ["p1", "p2"]}],
        }))
        result = engine.handle("generate_outline", {"topic": "space travel"}, {})
        assert result["confidence"] == 0.92
        assert "My Outline" in result["response"]
        assert "Intro" in result["response"]
        assert "p1" in result["response"]

    def test_title_falls_back_to_topic_title_case(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "title": "", "sections": [{"heading": "H", "points": []}],
        }))
        result = engine.handle("generate_outline", {"topic": "my topic"}, {})
        assert result["data"]["title"] == "My Topic"

    def test_missing_topic_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("generate_outline", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("generate_outline", {"topic": "x"}, {})
        assert result["confidence"] == 0.0

    def test_empty_sections_returns_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps(
            {"title": "T", "sections": []}
        ))
        result = engine.handle("generate_outline", {"topic": "x"}, {})
        assert result["confidence"] == 0.0
        assert "came back empty" in result["response"].lower()

    def test_non_dict_sections_are_filtered(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "title": "T", "sections": ["bad", {"heading": "Good", "points": []}],
        }))
        result = engine.handle("generate_outline", {"topic": "x"}, {})
        assert len(result["data"]["sections"]) == 1


# ===========================================================================
# check_plagiarism
# ===========================================================================

class TestCheckPlagiarism:
    def test_returns_honest_unsupported_response(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("check_plagiarism", {"text": "some essay text"}, {})
        assert result["confidence"] == 0.6
        assert result["data"]["supported"] is False
        assert "don't have a live web index" in result["response"]

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("check_plagiarism", {}, {})
        assert result["confidence"] == 0.6
        assert result["data"].get("needs_clarification") is True

    def test_does_not_call_llm_at_all(self, tmp_path):
        engine, llm = make_engine(tmp_path)
        engine.handle("check_plagiarism", {"text": "x"}, {})
        assert llm.call_count == 0

    def test_saves_to_db(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        engine.handle("check_plagiarism", {"text": "x"}, {})
        rows = engine.db.get_all()
        assert rows[0]["type"] == "plagiarism_check"


# ===========================================================================
# generate_citation
# ===========================================================================

class TestGenerateCitation:
    def test_happy_path(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "reference_entry": "Author, A. (2024). Title. Publisher.",
            "in_text": "(Author, 2024)",
        }))
        result = engine.handle(
            "generate_citation", {"source": "some book", "style": "apa"}, {}
        )
        assert result["confidence"] == 0.85
        assert "APA reference entry:" in result["response"]
        assert "(Author, 2024)" in result["response"]

    def test_default_style_is_apa(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "reference_entry": "entry", "in_text": "",
        }))
        engine.handle("generate_citation", {"source": "book"}, {})
        assert "apa" in llm.last_prompt

    def test_invalid_style_falls_back_to_default(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "reference_entry": "entry", "in_text": "",
        }))
        result = engine.handle(
            "generate_citation", {"source": "book", "style": "not_a_style"}, {}
        )
        assert result["data"]["style"] == "apa"

    def test_mla_style_is_respected(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "reference_entry": "entry", "in_text": "",
        }))
        result = engine.handle(
            "generate_citation", {"source": "book", "style": "mla"}, {}
        )
        assert result["data"]["style"] == "mla"
        assert "MLA reference entry:" in result["response"]

    def test_missing_source_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("generate_citation", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("generate_citation", {"source": "x"}, {})
        assert result["confidence"] == 0.0

    def test_empty_reference_entry_returns_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "reference_entry": "", "in_text": "",
        }))
        result = engine.handle("generate_citation", {"source": "x"}, {})
        assert result["confidence"] == 0.0
        assert "came back empty" in result["response"].lower()

    def test_no_in_text_omits_that_section(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "reference_entry": "entry only", "in_text": "",
        }))
        result = engine.handle("generate_citation", {"source": "x"}, {})
        assert "In-text:" not in result["response"]

    def test_citation_is_memory_worthy_type(self, tmp_path):
        mem = MagicMock()
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "reference_entry": "entry", "in_text": "",
        }), memory=mem)
        engine.handle("generate_citation", {"source": "x"}, {})
        mem.learn.assert_called_once()


# ===========================================================================
# check_consistency
# ===========================================================================

class TestCheckConsistency:
    def test_happy_path_with_issues(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "issues": [{"category": "capitalization",
                        "description": "Inconsistent case",
                        "examples": ["Email vs email"]}],
        }))
        result = engine.handle("check_consistency", {"text": "Email and email"}, {})
        assert result["confidence"] == 0.88
        assert "Inconsistent case" in result["response"]
        assert "Email vs email" in result["response"]

    def test_no_issues_found(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({"issues": []}))
        result = engine.handle("check_consistency", {"text": "consistent text"}, {})
        assert result["response"] == "No consistency issues found."

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("check_consistency", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("check_consistency", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_non_dict_issues_are_filtered(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "issues": ["bad entry", {"category": "dates", "description": "d"}],
        }))
        result = engine.handle("check_consistency", {"text": "x"}, {})
        assert len(result["data"]["issues"]) == 1


# ===========================================================================
# readability_report
# ===========================================================================

class TestReadabilityReport:
    def test_happy_path(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "reading_ease": "easy", "sentence_variety": "high",
            "issues": ["Some passive voice."],
            "suggestions": ["Use more active voice."],
        }))
        result = engine.handle("readability_report", {"text": "some text"}, {})
        assert result["confidence"] == 0.88
        assert "Reading ease: easy" in result["response"]
        assert "Some passive voice." in result["response"]
        assert "Use more active voice." in result["response"]

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("readability_report", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("readability_report", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_non_dict_llm_result_degrades_gracefully(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps(["a", "b"]))
        result = engine.handle("readability_report", {"text": "x"}, {})
        assert result["confidence"] == 0.88
        assert "unknown" in result["response"].lower()


# ===========================================================================
# writing_stats
# ===========================================================================

class TestWritingStats:
    def test_no_activity_yet(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("writing_stats", {}, {})
        assert result["confidence"] == 0.97
        assert "No writing activity logged yet." in result["response"]

    def test_reports_counts_after_activity(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="Result text.")
        engine.handle("rewrite_text", {"text": "hello world"}, {})
        engine.handle("expand_text", {"text": "hi"}, {})
        result = engine.handle("writing_stats", {}, {})
        assert "Total writing items: 2" in result["response"]
        assert "rewrite" in result["response"]
        assert "expansion" in result["response"]

    def test_db_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch.object(engine.db, "get_stats", side_effect=RuntimeError("db down")):
            result = engine.handle("writing_stats", {}, {})
        assert result["confidence"] == 0.0

    def test_does_not_call_llm(self, tmp_path):
        engine, llm = make_engine(tmp_path)
        engine.handle("writing_stats", {}, {})
        assert llm.call_count == 0


# ===========================================================================
# LLM call helpers (_llm_text / _llm_json)
# ===========================================================================

class TestLlmHelpers:
    def test_llm_text_uses_injected_instance_with_no_fmt(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="plain text")
        result = engine._llm_text("prompt")
        assert result == "plain text"
        assert llm.last_fmt is None

    def test_llm_text_strips_result(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="  padded  ")
        assert engine._llm_text("prompt") == "padded"

    def test_llm_text_raises_on_empty(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        with pytest.raises(LLMResponseError):
            engine._llm_text("prompt")

    def test_llm_text_raises_on_whitespace_only(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="   ")
        with pytest.raises(LLMResponseError):
            engine._llm_text("prompt")

    def test_llm_json_uses_json_fmt(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response='{"a": 1}')
        result = engine._llm_json("prompt")
        assert result == {"a": 1}
        assert llm.last_fmt == "json"

    def test_llm_json_raises_on_empty(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        with pytest.raises(LLMResponseError):
            engine._llm_json("prompt")

    def test_llm_json_raises_on_invalid_json(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="{not valid json")
        with pytest.raises(LLMResponseError):
            engine._llm_json("prompt")

    def test_falls_back_to_module_level_generate_without_llm_instance(self, tmp_path):
        engine = MetisEngine(
            ollama_cfg={"model": "mistral", "host": "127.0.0.1", "port": 11434},
            db_path=tmp_path / "m.db",
        )
        with patch("modules.metis.engine.generate", return_value="fallback") as mock_gen:
            result = engine._llm_text("prompt")
        assert result == "fallback"
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("fmt") is None

    def test_falls_back_to_module_level_generate_for_json(self, tmp_path):
        engine = MetisEngine(db_path=tmp_path / "m.db")
        with patch("modules.metis.engine.generate", return_value='{"x": 1}') as mock_gen:
            result = engine._llm_json("prompt")
        assert result == {"x": 1}
        assert mock_gen.call_args.kwargs.get("fmt") == "json"


# ===========================================================================
# Module-level pure helpers
# ===========================================================================

class TestExtractHelper:
    def test_returns_first_nonempty_key(self):
        assert _extract({"a": "", "b": "val"}, "a", "b") == "val"

    def test_returns_empty_when_all_missing(self):
        assert _extract({}, "a", "b") == ""

    def test_strips_whitespace(self):
        assert _extract({"a": "  hi  "}, "a") == "hi"

    def test_coerces_non_string(self):
        assert _extract({"a": 42}, "a") == "42"


class TestNormaliseHelper:
    def test_exact_match(self):
        assert _normalise("professional", {"professional", "casual"}, "neutral") == "professional"

    def test_case_insensitive(self):
        assert _normalise("PROFESSIONAL", {"professional"}, "neutral") == "professional"

    def test_empty_value_returns_default(self):
        assert _normalise("", {"professional"}, "neutral") == "neutral"

    def test_unmatched_value_returns_default(self):
        assert _normalise("xyz_unknown", {"professional"}, "neutral") == "neutral"

    def test_prefix_match(self):
        # "form" is a prefix of "formal"
        assert _normalise("form", {"formal", "casual"}, "neutral") == "formal"

    def test_value_is_prefix_of_valid_entry(self):
        assert _normalise("casual chat", {"casual"}, "neutral") == "casual"


class TestCollectMissingAndClarify:
    def test_collect_missing_returns_questions_for_empty_fields(self):
        result = _collect_missing(
            ("text", "", "What text?"), ("goal", "clarity", "What goal?")
        )
        assert result == ["What text?"]

    def test_collect_missing_returns_empty_when_all_present(self):
        assert _collect_missing(("a", "x", "q1"), ("b", "y", "q2")) == []

    def test_clarify_shape(self):
        result = _clarify(["question 1", "question 2"])
        assert result == {
            "response": "question 1 question 2",
            "data": {"needs_clarification": True},
            "confidence": 0.6,
        }


class TestFormatCorrections:
    def test_no_changes(self):
        assert _format_corrections("Fixed text.", []) == "No errors found.\n\nFixed text."

    def test_with_changes(self):
        changes = [{"type": "spelling", "original": "teh", "suggestion": "the"}]
        result = _format_corrections("Fixed", changes)
        assert "Changes (1):" in result
        assert "[spelling]" in result
        assert '"teh"' in result
        assert '"the"' in result

    def test_missing_change_fields_default_gracefully(self):
        result = _format_corrections("Fixed", [{}])
        assert "[edit]" in result


class TestFormatStyleSuggestions:
    def test_no_suggestions_returns_revised(self):
        assert _format_style_suggestions([], "Revised text") == "Revised text"

    def test_no_suggestions_no_revised_returns_default_message(self):
        assert _format_style_suggestions([], "") == "No style issues found."

    def test_with_suggestions_and_revised(self):
        suggestions = [{"category": "voice", "issue": "passive", "suggestion": "active"}]
        result = _format_style_suggestions(suggestions, "Active text.")
        assert "[voice] passive → active" in result
        assert "Active text." in result

    def test_with_suggestions_no_revised(self):
        suggestions = [{"category": "voice", "issue": "passive", "suggestion": "active"}]
        result = _format_style_suggestions(suggestions, "")
        assert "Revised:" not in result


class TestFormatOutline:
    def test_renders_sections_and_points(self):
        result = _format_outline("Title", [
            {"heading": "Intro", "points": ["p1", "p2"]},
            {"heading": "Body", "points": []},
        ])
        assert result.startswith("Title")
        assert "1. Intro" in result
        assert "   - p1" in result
        assert "2. Body" in result

    def test_missing_heading_uses_default(self):
        result = _format_outline("T", [{"points": ["p"]}])
        assert "1. Section 1" in result


class TestFormatConsistency:
    def test_no_issues(self):
        assert _format_consistency([]) == "No consistency issues found."

    def test_with_issues_and_examples(self):
        issues = [{"category": "dates", "description": "inconsistent formats",
                   "examples": ["2024-01-01", "Jan 1, 2024"]}]
        result = _format_consistency(issues)
        assert "Consistency issues (1):" in result
        assert "[dates] inconsistent formats" in result
        assert "e.g. 2024-01-01" in result


class TestFormatReadability:
    def test_full_result(self):
        result = _format_readability({
            "reading_ease": "medium", "sentence_variety": "low",
            "issues": ["too many long sentences"],
            "suggestions": ["vary sentence length"],
        })
        assert "Reading ease: medium" in result
        assert "Sentence variety: low" in result
        assert "too many long sentences" in result
        assert "vary sentence length" in result

    def test_empty_result_uses_unknown_defaults(self):
        result = _format_readability({})
        assert "Reading ease: unknown" in result
        assert "Sentence variety: unknown" in result

    def test_non_string_issues_are_filtered(self):
        result = _format_readability({"issues": ["ok", 123, None]})
        assert result.count("•") == 1


class TestFormatStats:
    def test_zero_total(self):
        assert _format_stats({"total": 0, "by_type": {}}) == "No writing activity logged yet."

    def test_with_data(self):
        stats = {
            "total": 3,
            "by_type": {
                "rewrite": {"count": 2, "input_chars": 100, "output_chars": 90},
                "draft": {"count": 1, "input_chars": 20, "output_chars": 200},
            },
        }
        result = _format_stats(stats)
        assert "Total writing items: 3" in result
        assert "draft: 1" in result
        assert "rewrite: 2" in result
        assert "≈100 chars in / 90 chars out" in result

    def test_by_type_sorted_alphabetically(self):
        stats = {
            "total": 2,
            "by_type": {
                "z_type": {"count": 1, "input_chars": 0, "output_chars": 0},
                "a_type": {"count": 1, "input_chars": 0, "output_chars": 0},
            },
        }
        result = _format_stats(stats)
        assert result.index("a_type") < result.index("z_type")


class TestResponseShapeHelpers:
    def test_ok_default(self):
        assert _ok("hi") == {"response": "hi", "data": {}, "confidence": 0.9}

    def test_ok_custom(self):
        assert _ok("hi", data={"x": 1}, confidence=0.4) == {
            "response": "hi", "data": {"x": 1}, "confidence": 0.4,
        }

    def test_err(self):
        assert _err("bad") == {"response": "bad", "data": {}, "confidence": 0.0}


# ===========================================================================
# MetisDB
# ===========================================================================

class TestMetisDB:
    def test_save_and_get_all(self, tmp_path):
        db = MetisDB(str(tmp_path / "m.db"))
        db.save("rewrite", "content A", title="T1", input_chars=10, output_chars=9)
        db.save("draft", "content B", title="T2", input_chars=5, output_chars=50)
        rows = db.get_all()
        assert len(rows) == 2
        types = {r["type"] for r in rows}
        assert types == {"rewrite", "draft"}

    def test_get_recent_filters_by_type(self, tmp_path):
        db = MetisDB(str(tmp_path / "m.db"))
        db.save("rewrite", "a")
        db.save("draft", "b")
        db.save("rewrite", "c")
        rows = db.get_recent("rewrite")
        assert len(rows) == 2
        assert all(r["type"] == "rewrite" for r in rows)

    def test_get_recent_respects_limit(self, tmp_path):
        db = MetisDB(str(tmp_path / "m.db"))
        for i in range(5):
            db.save("rewrite", f"content {i}")
        rows = db.get_recent("rewrite", limit=2)
        assert len(rows) == 2

    def test_get_all_respects_limit(self, tmp_path):
        db = MetisDB(str(tmp_path / "m.db"))
        for i in range(5):
            db.save("rewrite", f"content {i}")
        rows = db.get_all(limit=3)
        assert len(rows) == 3

    def test_get_all_orders_most_recent_first(self, tmp_path):
        db = MetisDB(str(tmp_path / "m.db"))
        first_id = db.save("rewrite", "first")
        second_id = db.save("rewrite", "second")
        rows = db.get_all()
        assert rows[0]["id"] == second_id
        assert rows[1]["id"] == first_id

    def test_get_stats_empty_db(self, tmp_path):
        db = MetisDB(str(tmp_path / "m.db"))
        stats = db.get_stats()
        assert stats == {"by_type": {}, "total": 0}

    def test_get_stats_aggregates_correctly(self, tmp_path):
        db = MetisDB(str(tmp_path / "m.db"))
        db.save("rewrite", "a", input_chars=10, output_chars=8)
        db.save("rewrite", "b", input_chars=20, output_chars=15)
        db.save("draft", "c", input_chars=5, output_chars=50)
        stats = db.get_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["rewrite"]["count"] == 2
        assert stats["by_type"]["rewrite"]["input_chars"] == 30
        assert stats["by_type"]["rewrite"]["output_chars"] == 23
        assert stats["by_type"]["draft"]["count"] == 1

    def test_save_returns_incrementing_row_id(self, tmp_path):
        db = MetisDB(str(tmp_path / "m.db"))
        id1 = db.save("rewrite", "a")
        id2 = db.save("rewrite", "b")
        assert id2 == id1 + 1

    def test_save_persists_metadata_and_previews(self, tmp_path):
        db = MetisDB(str(tmp_path / "m.db"))
        db.save(
            "rewrite", "output content", title="My Title",
            input_preview="input snippet", input_chars=100, output_chars=14,
            metadata=json.dumps({"goal": "clarity"}),
        )
        row = db.get_all()[0]
        assert row["title"] == "My Title"
        assert row["input_preview"] == "input snippet"
        assert row["input_chars"] == 100
        assert row["output_chars"] == 14
        assert json.loads(row["metadata"]) == {"goal": "clarity"}

    def test_concurrent_writes_are_thread_safe(self, tmp_path):
        import threading
        db = MetisDB(str(tmp_path / "m.db"))
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    db.save("rewrite", f"item-{n}-{i}")
            except Exception as exc:  # pragma: no cover - failure path only
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert db.get_stats()["total"] == 50


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
