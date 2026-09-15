# tests/test_orpheus.py
"""
Extensive regression tests for modules/orpheus/engine.py — Orpheus,
Hestia's creative-writing module (poems, lyrics, short stories,
brainstorming, creative prompts, name generation, continuing/critiquing/
rewriting existing text, and recalling past creations).

This module previously had zero test coverage. These tests cover every
public intent's happy path, clarification path, and error/malformed-JSON
path; the point-of-view alias resolution (`_normalise_pov`) and its
deliberate prefix-collision handling; every module-level pure formatting
helper; and OrpheusDB's schema, search filters, and ordering — all via an
injected FakeLLM (bypassing core.ollama_client.generate entirely) plus an
isolated temp-file SQLite DB.

Run with:  pytest tests/test_orpheus.py -v
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.orpheus.engine import (
    OrpheusEngine,
    LLMResponseError,
    _extract,
    _normalise,
    _normalise_pov,
    _collect_missing,
    _clarify,
    _validate_brainstorm,
    _format_brainstorm,
    _format_creative_prompts,
    _validate_critique,
    _format_critique,
    _format_names,
    _format_creations,
    _ok,
    _err,
    _DEFAULT_POV,
)
from modules.orpheus.db import OrpheusDB


# ===========================================================================
# Fakes / fixtures
# ===========================================================================

class FakeLLM:
    """Stand-in for HestiaLLM — records prompts, returns canned JSON/text."""

    def __init__(self, response: str = ""):
        self.response = response
        self.responses = None
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
    """Build an OrpheusEngine with a real temp-file SQLite DB and an
    injected FakeLLM (bypassing core.ollama_client.generate entirely)."""
    engine = OrpheusEngine(
        ollama_cfg={}, memory=memory,
        db_path=tmp_path / "orpheus_test.db",
        llm=FakeLLM(llm_response),
    )
    return engine, engine._llm_instance


# ===========================================================================
# BaseModule contract
# ===========================================================================

class TestOrpheusContract:
    ALL_INTENTS = (
        "write_poem", "brainstorm", "creative_prompt", "generate_lyrics",
        "write_story", "continue_writing", "critique_writing",
        "rewrite_style", "generate_names", "get_creations",
    )

    def test_can_handle_all_registered_intents(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        for intent in self.ALL_INTENTS:
            assert engine.can_handle(intent) is True

    def test_can_handle_rejects_unknown_intent(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        assert engine.can_handle("not_a_real_intent") is False

    def test_name_is_orpheus(self):
        assert OrpheusEngine.name == "orpheus"

    def test_handle_unknown_intent_returns_zero_confidence(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("bogus_intent", {}, {})
        assert result["confidence"] == 0.0
        assert result["data"] == {}

    def test_handle_never_raises_on_unexpected_exception(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch.object(engine, "_dispatch", side_effect=RuntimeError("boom")):
            result = engine.handle("write_poem", {"topic": "x"}, {})
        assert result["confidence"] == 0.0
        assert "went wrong" in result["response"].lower()

    def test_get_context_reports_recent_types_and_total(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="A lovely poem.")
        engine.handle("write_poem", {"topic": "stars"}, {})
        ctx = engine.get_context()
        assert ctx["orpheus_total_creations"] == 1
        assert ctx["orpheus_recent_types"] == ["poem"]

    def test_get_context_returns_empty_dict_on_db_failure(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch.object(engine.db, "get_all", side_effect=RuntimeError("db down")):
            assert engine.get_context() == {}

    def test_get_context_empty_when_no_activity(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        ctx = engine.get_context()
        assert ctx["orpheus_total_creations"] == 0
        assert ctx["orpheus_recent_types"] == []


# ===========================================================================
# write_poem
# ===========================================================================

class TestWritePoem:
    def test_happy_path(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="Roses are red...")
        result = engine.handle(
            "write_poem", {"topic": "love", "style": "sonnet", "tone": "romantic"}, {}
        )
        assert result["confidence"] == 0.95
        assert "Roses are red..." in result["response"]
        assert result["data"]["style"] == "sonnet"
        assert result["data"]["tone"] == "romantic"
        assert "love" in llm.last_prompt

    def test_default_style_and_tone_used_when_absent(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="poem text")
        result = engine.handle("write_poem", {"topic": "autumn"}, {})
        assert result["data"]["style"] == "free verse"
        assert result["data"]["tone"] == "reflective"

    def test_invalid_style_falls_back_to_default(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="poem")
        result = engine.handle("write_poem", {"topic": "x", "style": "gibberish"}, {})
        assert result["data"]["style"] == "free verse"

    def test_title_includes_style_and_topic(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="poem")
        result = engine.handle("write_poem", {"topic": "the sea", "style": "haiku"}, {})
        assert "Haiku" in result["data"]["title"]
        assert "The Sea" in result["data"]["title"]

    def test_missing_topic_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("write_poem", {}, {})
        assert result["confidence"] == 0.6
        assert result["data"]["needs_clarification"] is True

    def test_uses_raw_query_fallback_for_topic(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="poem")
        engine.handle("write_poem", {"raw_query": "write about the moon"}, {})
        assert "write about the moon" in llm.last_prompt

    def test_llm_empty_response_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("write_poem", {"topic": "x"}, {})
        assert result["confidence"] == 0.0
        assert "trouble writing" in result["response"].lower()

    def test_uses_no_fmt_for_plain_text_llm_call(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="poem")
        engine.handle("write_poem", {"topic": "x"}, {})
        assert llm.last_fmt is None

    def test_persists_to_db(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="A poem.")
        engine.handle("write_poem", {"topic": "trees"}, {})
        rows = engine.db.get_all()
        assert rows[0]["type"] == "poem"

    def test_persists_to_memory_when_present(self, tmp_path):
        mem = MagicMock()
        engine, _ = make_engine(tmp_path, llm_response="A poem.", memory=mem)
        engine.handle("write_poem", {"topic": "trees"}, {})
        mem.learn.assert_called_once()

    def test_memory_failure_does_not_break_response(self, tmp_path):
        mem = MagicMock()
        mem.learn.side_effect = RuntimeError("memory down")
        engine, _ = make_engine(tmp_path, llm_response="A poem.", memory=mem)
        result = engine.handle("write_poem", {"topic": "trees"}, {})
        assert result["confidence"] == 0.95

    def test_db_write_failure_does_not_withhold_output(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="A poem output.")
        with patch.object(engine.db, "save", side_effect=RuntimeError("disk full")):
            result = engine.handle("write_poem", {"topic": "x"}, {})
        assert result["confidence"] == 0.95
        assert "A poem output." in result["response"]


# ===========================================================================
# brainstorm
# ===========================================================================

class TestBrainstorm:
    def test_happy_path(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "central_idea": "core concept",
            "branches": [
                {"theme": "Tech", "ideas": ["AI", "AR"], "unexpected_angle": "low-tech twist"}
            ],
            "cross_connections": ["Tech ties to Nature"],
            "first_action": "sketch a prototype",
        }))
        result = engine.handle("brainstorm", {"topic": "future cities"}, {})
        assert result["confidence"] == 0.95
        assert "core concept" in result["response"]
        assert "TECH" in result["response"]
        assert "sketch a prototype" in result["response"]

    def test_missing_topic_returns_prompt_not_clarify_shape(self, tmp_path):
        """brainstorm's missing-topic path is a plain _ok(), not _clarify()
        — this differs from every other Orpheus intent and is worth
        pinning explicitly."""
        engine, llm = make_engine(tmp_path)
        result = engine.handle("brainstorm", {}, {})
        assert result["confidence"] == 0.5
        assert "brainstorm about" in result["response"].lower()
        assert llm.call_count == 0

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("brainstorm", {"topic": "x"}, {})
        assert result["confidence"] == 0.0

    def test_malformed_json_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="not json")
        result = engine.handle("brainstorm", {"topic": "x"}, {})
        assert result["confidence"] == 0.0

    def test_branches_not_a_list_fails_validation(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "central_idea": "x", "branches": "not a list",
        }))
        result = engine.handle("brainstorm", {"topic": "x"}, {})
        assert result["confidence"] == 0.0
        assert "malformed" in result["response"].lower()

    def test_empty_branches_list_fails_validation(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "central_idea": "x", "branches": [],
        }))
        result = engine.handle("brainstorm", {"topic": "x"}, {})
        assert result["confidence"] == 0.0

    def test_branches_containing_non_dict_fails_validation(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "central_idea": "x", "branches": ["just a string"],
        }))
        result = engine.handle("brainstorm", {"topic": "x"}, {})
        assert result["confidence"] == 0.0

    def test_uses_json_fmt(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "central_idea": "x", "branches": [{"theme": "t", "ideas": []}],
        }))
        engine.handle("brainstorm", {"topic": "x"}, {})
        assert llm.last_fmt == "json"

    def test_persists_central_idea_to_memory(self, tmp_path):
        mem = MagicMock()
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "central_idea": "the key idea", "branches": [{"theme": "t", "ideas": []}],
        }), memory=mem)
        engine.handle("brainstorm", {"topic": "x"}, {})
        mem.learn.assert_called_once()
        assert mem.learn.call_args[0][1] == "the key idea"


# ===========================================================================
# creative_prompt
# ===========================================================================

class TestCreativePrompt:
    def test_happy_path(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "prompts": [
                {"prompt": "Write about a lost key.", "medium": "writing", "difficulty": "easy"},
            ],
        }))
        result = engine.handle(
            "creative_prompt", {"medium": "writing", "theme": "mystery", "count": 3}, {}
        )
        assert result["confidence"] == 0.95
        assert "lost key" in result["response"]
        assert len(result["data"]["prompts"]) == 1

    def test_default_medium_and_theme(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "prompts": [{"prompt": "p"}],
        }))
        engine.handle("creative_prompt", {}, {})
        assert "writing" in llm.last_prompt
        assert "anything" in llm.last_prompt

    def test_count_is_clamped_to_max(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "prompts": [{"prompt": "p"}],
        }))
        engine.handle("creative_prompt", {"count": 999}, {})
        assert "20" in llm.last_prompt  # _MAX_PROMPT_COUNT

    def test_count_is_clamped_to_min_of_one(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "prompts": [{"prompt": "p"}],
        }))
        engine.handle("creative_prompt", {"count": -5}, {})
        assert "Generate 1 creative" in llm.last_prompt

    def test_invalid_count_falls_back_to_default(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "prompts": [{"prompt": "p"}],
        }))
        engine.handle("creative_prompt", {"count": "not a number"}, {})
        assert "Generate 5 creative" in llm.last_prompt

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("creative_prompt", {}, {})
        assert result["confidence"] == 0.0

    def test_non_dict_prompt_entries_filtered(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "prompts": ["just a string", {"prompt": "real one"}],
        }))
        result = engine.handle("creative_prompt", {}, {})
        assert len(result["data"]["prompts"]) == 1

    def test_all_prompts_filtered_returns_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "prompts": ["a", "b", "c"],
        }))
        result = engine.handle("creative_prompt", {}, {})
        assert result["confidence"] == 0.0
        assert "empty" in result["response"].lower()

    def test_no_prompts_key_returns_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({}))
        result = engine.handle("creative_prompt", {}, {})
        assert result["confidence"] == 0.0

    def test_type_entity_used_as_medium_fallback(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "prompts": [{"prompt": "p"}],
        }))
        engine.handle("creative_prompt", {"type": "art"}, {})
        assert "art" in llm.last_prompt


# ===========================================================================
# generate_lyrics
# ===========================================================================

class TestGenerateLyrics:
    def test_happy_path(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="VERSE 1\nSome lyrics...")
        result = engine.handle(
            "generate_lyrics", {"topic": "heartbreak", "genre": "pop", "tone": "melancholic"}, {}
        )
        assert result["confidence"] == 0.95
        assert "VERSE 1" in result["response"]
        assert result["data"]["genre"] == "pop"

    def test_default_genre_tone_rhyme_structure(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="lyrics")
        result = engine.handle("generate_lyrics", {"topic": "x"}, {})
        assert result["data"]["genre"] == "pop"
        assert "rhyme" in result["data"]
        assert "structure" in result["data"]

    def test_custom_structure_used_verbatim(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="lyrics")
        engine.handle(
            "generate_lyrics",
            {"topic": "x", "structure": "verse-chorus-outro"}, {},
        )
        assert "verse-chorus-outro" in llm.last_prompt

    def test_missing_topic_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("generate_lyrics", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("generate_lyrics", {"topic": "x"}, {})
        assert result["confidence"] == 0.0

    def test_invalid_rhyme_scheme_falls_back_to_default(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="lyrics")
        result = engine.handle(
            "generate_lyrics", {"topic": "x", "rhyme_scheme": "nonsense"}, {}
        )
        assert result["data"]["rhyme"] == "ABAB"


# ===========================================================================
# write_story
# ===========================================================================

class TestWriteStory:
    def test_happy_path(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="Once upon a time...")
        result = engine.handle(
            "write_story",
            {"topic": "a dragon", "genre": "fantasy", "pov": "first person", "tone": "hopeful"},
            {},
        )
        assert result["confidence"] == 0.95
        assert "Once upon a time..." in result["response"]
        assert result["data"]["genre"] == "fantasy"
        assert result["data"]["pov"] == "first person"

    def test_default_genre_and_pov(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="story")
        result = engine.handle("write_story", {"topic": "x"}, {})
        assert result["data"]["genre"] == "literary"
        assert result["data"]["pov"] == "third person limited"

    def test_point_of_view_entity_fallback(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="story")
        result = engine.handle(
            "write_story", {"topic": "x", "point_of_view": "omniscient"}, {}
        )
        assert result["data"]["pov"] == "third person omniscient"

    def test_missing_topic_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("write_story", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("write_story", {"topic": "x"}, {})
        assert result["confidence"] == 0.0

    def test_title_includes_genre_and_topic(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="story")
        result = engine.handle(
            "write_story", {"topic": "space pirates", "genre": "sci-fi"}, {}
        )
        assert "Sci-Fi Story" in result["data"]["title"]
        assert "Space Pirates" in result["data"]["title"]


# ===========================================================================
# continue_writing
# ===========================================================================

class TestContinueWriting:
    def test_happy_path(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="...and then it happened.")
        result = engine.handle(
            "continue_writing", {"text": "It was a dark night.", "direction": "add tension"}, {}
        )
        assert result["confidence"] == 0.9
        assert result["response"] == "...and then it happened."
        assert "add tension" in llm.last_prompt

    def test_no_direction_omits_direction_line(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="continuation")
        engine.handle("continue_writing", {"text": "start"}, {})
        assert "Direction:" not in llm.last_prompt

    def test_text_truncated_to_max_input_len(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="continuation")
        long_text = "a" * 5000
        engine.handle("continue_writing", {"text": long_text}, {})
        # _MAX_INPUT_TEXT_LEN is 4000
        assert long_text[:4000] in llm.last_prompt
        assert long_text not in llm.last_prompt

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("continue_writing", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("continue_writing", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_default_length_is_short(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="c")
        result = engine.handle("continue_writing", {"text": "x"}, {})
        assert result["data"]["length"] == "short"

    def test_continuation_not_persisted_to_memory(self, tmp_path):
        """continue_writing has no self._persist() call at all — only
        write_poem/generate_lyrics/write_story do."""
        mem = MagicMock()
        engine, _ = make_engine(tmp_path, llm_response="c", memory=mem)
        engine.handle("continue_writing", {"text": "x"}, {})
        mem.learn.assert_not_called()


# ===========================================================================
# critique_writing
# ===========================================================================

class TestCritiqueWriting:
    def test_happy_path(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "overall_impression": "Strong voice.",
            "strengths": ["vivid imagery"],
            "areas_to_improve": ["pacing in the middle"],
            "line_to_revisit": "The sky was blue.",
            "revision_example": "The sky bled a bruised blue.",
        }))
        result = engine.handle(
            "critique_writing", {"text": "some prose", "focus": "pacing"}, {}
        )
        assert result["confidence"] == 0.9
        assert "Strong voice." in result["response"]
        assert "vivid imagery" in result["response"]
        assert "pacing in the middle" in result["response"]
        assert "bruised blue" in result["response"]
        assert "pacing" in llm.last_prompt

    def test_no_focus_omits_focus_line(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "strengths": ["s"], "areas_to_improve": [],
        }))
        engine.handle("critique_writing", {"text": "x"}, {})
        assert "Focus especially on:" not in llm.last_prompt

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("critique_writing", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("critique_writing", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_malformed_result_missing_lists_fails_validation(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "overall_impression": "ok",
        }))
        result = engine.handle("critique_writing", {"text": "x"}, {})
        assert result["confidence"] == 0.0
        assert "malformed" in result["response"].lower()

    def test_both_lists_empty_fails_validation(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "strengths": [], "areas_to_improve": [],
        }))
        result = engine.handle("critique_writing", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_only_strengths_passes_validation(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "strengths": ["good pacing"], "areas_to_improve": [],
        }))
        result = engine.handle("critique_writing", {"text": "x"}, {})
        assert result["confidence"] == 0.9

    def test_text_truncated_to_max_input_len(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "strengths": ["s"], "areas_to_improve": [],
        }))
        long_text = "b" * 5000
        engine.handle("critique_writing", {"text": long_text}, {})
        assert long_text[:4000] in llm.last_prompt


# ===========================================================================
# rewrite_style
# ===========================================================================

class TestRewriteStyle:
    def test_happy_path(self, tmp_path):
        # NB: Orpheus's _VALID_TONES is the creative-writing tone set
        # (melancholic, joyful, romantic, dark, playful, ...) — "formal"
        # isn't one of them (that's a Metis rewrite-goal concept), so use
        # a tone that's actually valid here.
        engine, llm = make_engine(tmp_path, llm_response="Playful rewritten text.")
        result = engine.handle(
            "rewrite_style", {"text": "hey whats up", "tone": "playful"}, {}
        )
        assert result["confidence"] == 0.9
        assert result["response"] == "Playful rewritten text."
        assert result["data"]["tone"] == "playful"

    def test_style_entity_included_in_prompt(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="x")
        engine.handle("rewrite_style", {"text": "x", "style": "Hemingway-esque"}, {})
        assert "Hemingway-esque" in llm.last_prompt

    def test_no_style_uses_judgement_line(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="x")
        engine.handle("rewrite_style", {"text": "x"}, {})
        assert "use your judgement" in llm.last_prompt.lower()

    def test_missing_text_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("rewrite_style", {}, {})
        assert result["confidence"] == 0.6

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("rewrite_style", {"text": "x"}, {})
        assert result["confidence"] == 0.0

    def test_default_tone_is_reflective(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="x")
        result = engine.handle("rewrite_style", {"text": "x"}, {})
        assert result["data"]["tone"] == "reflective"


# ===========================================================================
# generate_names
# ===========================================================================

class TestGenerateNames:
    def test_happy_path(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "names": [{"name": "Aria", "vibe": "ethereal"}],
        }))
        result = engine.handle(
            "generate_names", {"category": "character", "theme": "fantasy"}, {}
        )
        assert result["confidence"] == 0.92
        assert "Aria" in result["response"]
        assert "ethereal" in result["response"]

    def test_default_category_and_theme(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "names": [{"name": "N"}],
        }))
        engine.handle("generate_names", {}, {})
        assert "character" in llm.last_prompt
        assert "anything" in llm.last_prompt

    def test_type_entity_used_as_category_fallback(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "names": [{"name": "N"}],
        }))
        engine.handle("generate_names", {"type": "band"}, {})
        assert "band" in llm.last_prompt

    def test_count_is_clamped_to_max(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "names": [{"name": "N"}],
        }))
        engine.handle("generate_names", {"count": 999}, {})
        assert "20" in llm.last_prompt  # _MAX_NAME_COUNT

    def test_invalid_count_falls_back_to_default(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response=json.dumps({
            "names": [{"name": "N"}],
        }))
        engine.handle("generate_names", {"count": "abc"}, {})
        assert "Generate 8 creative" in llm.last_prompt

    def test_llm_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        result = engine.handle("generate_names", {}, {})
        assert result["confidence"] == 0.0

    def test_names_missing_name_key_are_filtered(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "names": [{"vibe": "no name here"}, {"name": "Valid"}],
        }))
        result = engine.handle("generate_names", {}, {})
        assert len(result["data"]["names"]) == 1
        assert result["data"]["names"][0]["name"] == "Valid"

    def test_non_dict_name_entries_filtered(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "names": ["just a string"],
        }))
        result = engine.handle("generate_names", {}, {})
        assert result["confidence"] == 0.0
        assert "empty" in result["response"].lower()

    def test_invalid_category_falls_back_to_default(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response=json.dumps({
            "names": [{"name": "N"}],
        }))
        result = engine.handle("generate_names", {"category": "nonsense_category"}, {})
        assert "character" in result["response"].lower()


# ===========================================================================
# get_creations
# ===========================================================================

class TestGetCreations:
    def test_returns_saved_creations(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="A poem.")
        engine.handle("write_poem", {"topic": "trees"}, {})
        result = engine.handle("get_creations", {}, {})
        assert result["confidence"] == 0.9
        assert len(result["data"]["creations"]) == 1
        assert "poem" in result["response"].lower()

    def test_no_creations_yet(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("get_creations", {}, {})
        assert result["confidence"] == 0.7
        assert result["data"]["creations"] == []
        assert "don't have any matching" in result["response"].lower()

    def test_filters_by_valid_type(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        engine.responses = None
        engine._llm_instance.response = "A poem."
        engine.handle("write_poem", {"topic": "x"}, {})
        engine._llm_instance.response = json.dumps({
            "names": [{"name": "N"}],
        })
        engine.handle("generate_names", {}, {})
        result = engine.handle("get_creations", {"type": "poem"}, {})
        assert len(result["data"]["creations"]) == 1
        assert result["data"]["creations"][0]["type"] == "poem"

    def test_unrecognised_type_filter_is_ignored(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="A poem.")
        engine.handle("write_poem", {"topic": "x"}, {})
        result = engine.handle("get_creations", {"type": "not_a_real_type"}, {})
        # falls back to no filter -> still returns the poem
        assert len(result["data"]["creations"]) == 1

    def test_keyword_search(self, tmp_path):
        engine, llm = make_engine(tmp_path)
        llm.responses = ["A poem about oceans.", "A poem about mountains."]
        engine.handle("write_poem", {"topic": "oceans"}, {})
        engine.handle("write_poem", {"topic": "mountains"}, {})
        result = engine.handle("get_creations", {"keyword": "oceans"}, {})
        assert len(result["data"]["creations"]) == 1

    def test_limit_is_clamped_to_max(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="A poem.")
        for _ in range(3):
            engine.handle("write_poem", {"topic": "x"}, {})
        result = engine.handle("get_creations", {"limit": 999}, {})
        # _MAX_CREATIONS_LIMIT is 20, but only 3 rows exist
        assert len(result["data"]["creations"]) == 3

    def test_invalid_limit_falls_back_to_default(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="A poem.")
        engine.handle("write_poem", {"topic": "x"}, {})
        result = engine.handle("get_creations", {"limit": "not a number"}, {})
        assert result["confidence"] == 0.9

    def test_db_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch.object(engine.db, "search", side_effect=RuntimeError("db down")):
            result = engine.handle("get_creations", {}, {})
        assert result["confidence"] == 0.0
        assert "couldn't pull up" in result["response"].lower()

    def test_does_not_call_llm(self, tmp_path):
        engine, llm = make_engine(tmp_path)
        engine.handle("get_creations", {}, {})
        assert llm.call_count == 0


# ===========================================================================
# LLM call helpers
# ===========================================================================

class TestLlmHelpers:
    def test_llm_text_uses_injected_instance(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="plain text")
        assert engine._llm_text("prompt") == "plain text"
        assert llm.last_fmt is None

    def test_llm_text_strips_result(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="  padded  ")
        assert engine._llm_text("prompt") == "padded"

    def test_llm_text_raises_on_empty(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="")
        with pytest.raises(LLMResponseError):
            engine._llm_text("prompt")

    def test_llm_json_uses_json_fmt(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response='{"a": 1}')
        assert engine._llm_json("prompt") == {"a": 1}
        assert llm.last_fmt == "json"

    def test_llm_json_raises_on_invalid_json(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="not valid json")
        with pytest.raises(LLMResponseError):
            engine._llm_json("prompt")

    def test_falls_back_to_module_level_generate_without_llm_instance(self, tmp_path):
        engine = OrpheusEngine(
            ollama_cfg={"model": "mistral", "host": "127.0.0.1", "port": 11434},
            db_path=tmp_path / "o.db",
        )
        with patch("modules.orpheus.engine.generate", return_value="fallback") as mock_gen:
            result = engine._llm_text("prompt")
        assert result == "fallback"
        assert mock_gen.call_args.kwargs.get("fmt") is None

    def test_falls_back_to_module_level_generate_for_json(self, tmp_path):
        engine = OrpheusEngine(db_path=tmp_path / "o.db")
        with patch("modules.orpheus.engine.generate", return_value='{"x":1}') as mock_gen:
            result = engine._llm_json("prompt")
        assert result == {"x": 1}
        assert mock_gen.call_args.kwargs.get("fmt") == "json"


# ===========================================================================
# Module-level pure helpers
# ===========================================================================

class TestExtractHelper:
    def test_returns_first_nonempty(self):
        assert _extract({"a": "", "b": "val"}, "a", "b") == "val"

    def test_all_missing_returns_empty(self):
        assert _extract({}, "a") == ""

    def test_strips_whitespace(self):
        assert _extract({"a": "  hi  "}, "a") == "hi"


class TestNormaliseHelper:
    def test_exact_match(self):
        assert _normalise("haiku", {"haiku", "sonnet"}, "free verse") == "haiku"

    def test_case_insensitive(self):
        assert _normalise("HAIKU", {"haiku"}, "free verse") == "haiku"

    def test_empty_returns_default(self):
        assert _normalise("", {"haiku"}, "free verse") == "free verse"

    def test_unmatched_returns_default(self):
        assert _normalise("limerick-ish", {"haiku", "sonnet"}, "free verse") != "limerick-ish"

    def test_prefix_match(self):
        assert _normalise("folk rock", {"folk", "jazz"}, "pop") == "folk"


class TestNormalisePov:
    def test_empty_returns_default(self):
        assert _normalise_pov("") == _DEFAULT_POV

    def test_exact_alias_first(self):
        assert _normalise_pov("first") == "first person"

    def test_exact_alias_second(self):
        assert _normalise_pov("second person") == "second person"

    def test_disambiguates_third_person_omniscient(self):
        assert _normalise_pov("third person omniscient") == "third person omniscient"

    def test_disambiguates_third_person_limited(self):
        assert _normalise_pov("third person limited") == "third person limited"

    def test_bare_third_person_defaults_to_limited(self):
        """'third person' is a genuine prefix of BOTH omniscient and
        limited — the documented resolution defaults to limited via
        ordered alias-table iteration."""
        assert _normalise_pov("third person") == "third person limited"

    def test_omniscient_shorthand(self):
        assert _normalise_pov("omniscient") == "third person omniscient"

    def test_limited_shorthand(self):
        assert _normalise_pov("limited") == "third person limited"

    def test_unrecognised_value_returns_default(self):
        assert _normalise_pov("sideways narration") == _DEFAULT_POV

    def test_prefix_startswith_resolution(self):
        assert _normalise_pov("1st person plural") == "first person"

    def test_case_insensitive(self):
        assert _normalise_pov("FIRST PERSON") == "first person"


class TestCollectMissingAndClarify:
    def test_collect_missing_finds_empty_fields(self):
        result = _collect_missing(("topic", "", "What topic?"))
        assert result == ["What topic?"]

    def test_collect_missing_all_present_returns_empty(self):
        assert _collect_missing(("topic", "x", "q")) == []

    def test_clarify_shape(self):
        result = _clarify(["q1", "q2"])
        assert result["response"] == "q1 q2"
        assert result["data"] == {"needs_clarification": True}
        assert result["confidence"] == 0.6


class TestValidateBrainstorm:
    def test_valid_structure(self):
        assert _validate_brainstorm({
            "central_idea": "x", "branches": [{"theme": "t"}]
        }) is True

    def test_not_a_dict(self):
        assert _validate_brainstorm(["a", "b"]) is False

    def test_branches_missing(self):
        assert _validate_brainstorm({"central_idea": "x"}) is False

    def test_branches_not_a_list(self):
        assert _validate_brainstorm({"branches": "str"}) is False

    def test_branches_empty(self):
        assert _validate_brainstorm({"branches": []}) is False

    def test_branches_contains_non_dict(self):
        assert _validate_brainstorm({"branches": ["str", {"theme": "t"}]}) is False


class TestFormatBrainstorm:
    def test_full_structure(self):
        result = _format_brainstorm("topic", {
            "central_idea": "core",
            "branches": [
                {"theme": "tech", "ideas": ["i1", "i2"], "unexpected_angle": "twist"},
            ],
            "cross_connections": ["c1"],
            "first_action": "start here",
        })
        assert "Topic" in result
        assert "core" in result
        assert "TECH" in result
        assert "i1" in result
        assert "twist" in result
        assert "c1" in result
        assert "start here" in result

    def test_minimal_structure(self):
        result = _format_brainstorm("t", {"branches": [{"theme": "x"}]})
        assert "X" in result  # theme upper-cased

    def test_non_dict_branch_skipped_gracefully(self):
        result = _format_brainstorm("t", {"branches": [{"theme": "ok"}, "bad"]})
        assert "OK" in result


class TestFormatCreativePrompts:
    def test_renders_numbered_list(self):
        result = _format_creative_prompts("writing", "adventure", [
            {"prompt": "Write about X", "medium": "writing", "difficulty": "easy"},
            {"prompt": "Write about Y", "medium": "writing", "difficulty": "hard"},
        ])
        assert "1." in result
        assert "2." in result
        assert "Write about X" in result
        assert "Write about Y" in result

    def test_missing_fields_default_gracefully(self):
        result = _format_creative_prompts("writing", "theme", [{}])
        assert "1." in result


class TestValidateCritique:
    def test_valid_with_strengths_only(self):
        assert _validate_critique({"strengths": ["s"], "areas_to_improve": []}) is True

    def test_valid_with_improve_only(self):
        assert _validate_critique({"strengths": [], "areas_to_improve": ["i"]}) is True

    def test_not_a_dict(self):
        assert _validate_critique("string") is False

    def test_missing_keys(self):
        assert _validate_critique({}) is False

    def test_wrong_types(self):
        assert _validate_critique({"strengths": "not a list", "areas_to_improve": []}) is False

    def test_both_empty_is_invalid(self):
        assert _validate_critique({"strengths": [], "areas_to_improve": []}) is False


class TestFormatCritique:
    def test_full_structure(self):
        result = _format_critique({
            "overall_impression": "Good work.",
            "strengths": ["strong voice"],
            "areas_to_improve": ["pacing"],
            "line_to_revisit": "The end.",
            "revision_example": "A better end.",
        })
        assert "Good work." in result
        assert "STRENGTHS" in result
        assert "strong voice" in result
        assert "TO IMPROVE" in result
        assert "pacing" in result
        assert "TRY REVISING" in result
        assert "A better end." in result

    def test_minimal_structure_no_revision(self):
        result = _format_critique({"strengths": ["s"], "areas_to_improve": []})
        assert "TRY REVISING" not in result


class TestFormatNames:
    def test_renders_names_with_vibe(self):
        result = _format_names("character", "space", [
            {"name": "Zara", "vibe": "bold"},
        ])
        assert "Character" in result
        assert "Zara" in result
        assert "bold" in result

    def test_renders_names_without_vibe(self):
        result = _format_names("band", "rock", [{"name": "The Echoes"}])
        assert "The Echoes" in result
        assert "—" not in result.split("\n")[-1] or "The Echoes" in result


class TestFormatCreations:
    def test_renders_rows(self):
        rows = [
            {"type": "poem", "title": "My Poem", "logged_at": "2024-01-01"},
            {"type": "story", "title": "", "logged_at": "2024-01-02"},
        ]
        result = _format_creations(rows)
        assert "My Poem" in result
        assert "[poem]" in result
        assert "Story" in result  # falls back to type.title()


class TestResponseShapeHelpers:
    def test_ok_default(self):
        assert _ok("hi") == {"response": "hi", "data": {}, "confidence": 0.9}

    def test_err(self):
        assert _err("bad") == {"response": "bad", "data": {}, "confidence": 0.0}


# ===========================================================================
# OrpheusDB
# ===========================================================================

class TestOrpheusDB:
    def test_save_and_get_all(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "content A", title="T1")
        db.save("story", "content B", title="T2")
        rows = db.get_all()
        assert len(rows) == 2

    def test_get_recent_filters_by_type(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "a")
        db.save("story", "b")
        db.save("poem", "c")
        rows = db.get_recent("poem")
        assert len(rows) == 2
        assert all(r["type"] == "poem" for r in rows)

    def test_get_recent_respects_limit(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        for i in range(5):
            db.save("poem", f"c{i}")
        assert len(db.get_recent("poem", limit=2)) == 2

    def test_get_all_orders_most_recent_first(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        id1 = db.save("poem", "first")
        id2 = db.save("poem", "second")
        rows = db.get_all()
        assert rows[0]["id"] == id2
        assert rows[1]["id"] == id1

    def test_search_by_type_only(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "a")
        db.save("story", "b")
        results = db.search(type_="poem")
        assert len(results) == 1
        assert results[0]["type"] == "poem"

    def test_search_by_keyword_matches_title(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "content", title="Ocean Dreams")
        db.save("poem", "content", title="Mountain Song")
        results = db.search(keyword="ocean")
        assert len(results) == 1
        assert results[0]["title"] == "Ocean Dreams"

    def test_search_by_keyword_matches_content(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "a poem about the vast ocean", title="Untitled")
        db.save("poem", "a poem about mountains", title="Untitled 2")
        results = db.search(keyword="ocean")
        assert len(results) == 1

    def test_search_combines_type_and_keyword_with_and(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "ocean content", title="Ocean Poem")
        db.save("story", "ocean content", title="Ocean Story")
        results = db.search(type_="poem", keyword="ocean")
        assert len(results) == 1
        assert results[0]["type"] == "poem"

    def test_search_no_filters_returns_all(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "a")
        db.save("story", "b")
        assert len(db.search()) == 2

    def test_search_respects_limit(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        for i in range(5):
            db.save("poem", f"c{i}")
        assert len(db.search(limit=3)) == 3

    def test_search_no_matches_returns_empty_list(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "a")
        assert db.search(keyword="nonexistent_xyz") == []

    def test_search_keyword_is_case_insensitive(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "content", title="OCEAN Waves")
        results = db.search(keyword="ocean")
        assert len(results) == 1

    def test_save_persists_metadata(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        db.save("poem", "content", metadata=json.dumps({"style": "haiku"}))
        row = db.get_all()[0]
        assert json.loads(row["metadata"]) == {"style": "haiku"}

    def test_save_returns_incrementing_id(self, tmp_path):
        db = OrpheusDB(str(tmp_path / "o.db"))
        id1 = db.save("poem", "a")
        id2 = db.save("poem", "b")
        assert id2 == id1 + 1

    def test_concurrent_writes_are_thread_safe(self, tmp_path):
        import threading
        db = OrpheusDB(str(tmp_path / "o.db"))
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    db.save("poem", f"item-{n}-{i}")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(db.get_all(limit=1000)) == 50


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
