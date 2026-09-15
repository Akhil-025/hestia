# tests/test_dionysus.py
"""
Extensive regression tests for modules/dionysus/engine.py — Dionysus,
Hestia's entertainment/lifestyle recommendation module (movies, music,
restaurants, outings, recipes, and dismiss/mark-seen history).

This module previously had zero test coverage. These tests cover every
public intent, the dismiss/mark_seen history feature, and the module's
external-API integration points (OMDB, Spotify, TheAudioDB/MealDB/
CocktailDB/TriviaDB via core.free_apis), all via mocking at the
appropriate boundary (requests / free_apis helpers / the injected LLM).

Run with:  pytest tests/test_dionysus.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.free_apis import FreeAPIError
from modules.dionysus.engine import DionysusEngine
from modules.dionysus.db import DionysusDB


# ===========================================================================
# Fakes / fixtures
# ===========================================================================

class FakeLLM:
    """Stand-in for HestiaLLM — records prompts, returns canned JSON/text."""
    def __init__(self, response: str = ""):
        self.response = response
        self.last_prompt = None
        self.last_fmt = None
        self.call_count = 0

    def generate(self, prompt: str, fmt: str = None) -> str:
        self.last_prompt = prompt
        self.last_fmt = fmt
        self.call_count += 1
        return self.response


class FakeBrowserAgent:
    def __init__(self, search_result: str = ""):
        self.search_result = search_result
        self.queries = []

    def search_web(self, query: str) -> str:
        self.queries.append(query)
        return self.search_result


class FakeMemory:
    def __init__(self, location: str = ""):
        self._location = location

    def get_preference(self, key, default=""):
        if key == "location":
            return self._location
        return default


@pytest.fixture()
def dionysus_db_path(tmp_path):
    return str(tmp_path / "dionysus.db")


def make_engine(tmp_path, llm_response="", browser_result=None, memory=None):
    """Build a DionysusEngine with a real temp-file SQLite DB and an
    injected FakeLLM (bypassing core.ollama_client.generate entirely)."""
    engine = DionysusEngine(
        ollama_cfg={}, browser_agent=FakeBrowserAgent(browser_result) if browser_result is not None else None,
        memory=memory, llm=FakeLLM(llm_response),
    )
    # Redirect the DB to an isolated temp file so tests never share state.
    engine.db = DionysusDB(str(tmp_path / "dionysus_test.db"))
    return engine, engine._llm_instance


# ===========================================================================
# BaseModule contract
# ===========================================================================

class TestDionysusContract:
    def test_can_handle_all_registered_intents(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        for intent in (
            "recommend_movie", "find_restaurant", "recommend_music",
            "plan_outing", "dismiss_recommendation", "mark_seen",
            "recommend_recipe",
        ):
            assert engine.can_handle(intent) is True

    def test_can_handle_rejects_unknown_intent(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        assert engine.can_handle("totally_unknown") is False

    def test_handle_unknown_intent_returns_zero_confidence(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("not_a_real_intent", {}, {})
        assert result["confidence"] == 0.0
        assert result["data"] == {}

    def test_get_context_returns_empty_dict(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        assert engine.get_context() == {}


# ===========================================================================
# dismiss_recommendation / mark_seen
# ===========================================================================

class TestDismissAndMarkSeen:
    def test_dismiss_without_title_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("dismiss_recommendation", {}, {})
        assert result["confidence"] == 0.5
        assert "which recommendation" in result["response"].lower()

    def test_dismiss_marks_title_dismissed_in_db(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        engine.db.log("movie", "Inception", "great movie", 8.5)
        result = engine.handle("dismiss_recommendation", {"title": "Inception"}, {})
        assert result["confidence"] == 0.9
        assert "Inception" in result["response"]
        assert "Inception" in engine.db.dismissed_titles("movie")

    def test_dismiss_uses_raw_query_fallback(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("dismiss_recommendation", {"raw_query": "The Matrix"}, {})
        assert "The Matrix" in result["response"]

    def test_mark_seen_without_title_asks_for_clarification(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("mark_seen", {}, {})
        assert result["confidence"] == 0.5

    def test_mark_seen_updates_existing_recommendation(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        engine.db.log("movie", "Interstellar", "space epic", 9.0)
        result = engine.handle("mark_seen", {"title": "Interstellar"}, {})
        assert result["confidence"] == 0.9
        assert "Interstellar" in engine.db.seen_titles("movie")

    def test_mark_seen_reports_when_title_not_found(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        result = engine.handle("mark_seen", {"title": "Never Logged"}, {})
        assert result["confidence"] == 0.4
        assert "don't have" in result["response"].lower()

    def test_mark_seen_default_type_is_movie(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        engine.db.log("movie", "Dune", "sci-fi", 8.8)
        engine.handle("mark_seen", {"title": "Dune"}, {})
        assert "Dune" in engine.db.seen_titles("movie")

    def test_mark_seen_respects_explicit_type(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        engine.db.log("music", "Bohemian Rhapsody", "classic", None)
        engine.handle("mark_seen", {"title": "Bohemian Rhapsody", "type": "music"}, {})
        assert "Bohemian Rhapsody" in engine.db.seen_titles("music")

    def test_recommend_movie_excludes_both_dismissed_and_seen_titles(self, tmp_path):
        """The bug this feature fixed: previously only dismissed_titles was
        consulted, so a mark_seen()'d movie could still be recommended again."""
        engine, llm = make_engine(
            tmp_path,
            llm_response=json.dumps({"recommendations": [
                {"title": "New Movie", "year": "2024", "reason": "fun"}
            ]}),
        )
        engine.db.log("movie", "Old Dismissed", "x", None)
        engine.db.dismiss("Old Dismissed")
        engine.db.log("movie", "Old Seen", "y", None)
        engine.db.mark_seen("Old Seen", type_="movie")

        engine.handle("recommend_movie", {"mood": "action"}, {})

        assert "Old Dismissed" in llm.last_prompt
        assert "Old Seen" in llm.last_prompt


# ===========================================================================
# recommend_movie
# ===========================================================================

class TestRecommendMovie:
    def test_happy_path_without_omdb_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("modules.dionysus.engine.OMDB_KEY", "")
        engine, llm = make_engine(
            tmp_path,
            llm_response=json.dumps({"recommendations": [
                {"title": "Interstellar", "year": "2014", "reason": "mind-bending sci-fi"}
            ]}),
        )
        result = engine.handle("recommend_movie", {"mood": "space epics"}, {})
        assert result["confidence"] == 0.9
        assert "Interstellar" in result["response"]
        assert "N/A" in result["response"]  # no OMDB key -> rating unavailable
        assert result["data"]["recommendations"][0]["imdb_rating"] == "N/A"
        # Logged to DB for future dismiss/seen tracking.
        history = engine.db.get_history("movie")
        assert any(h["title"] == "Interstellar" for h in history)

    def test_happy_path_with_omdb_enrichment(self, tmp_path, monkeypatch):
        monkeypatch.setattr("modules.dionysus.engine.OMDB_KEY", "fake-key")
        engine, llm = make_engine(
            tmp_path,
            llm_response=json.dumps({"recommendations": [
                {"title": "Interstellar", "year": "2014", "reason": "mind-bending sci-fi"}
            ]}),
        )
        omdb_payload = {
            "Response": "True", "imdbRating": "8.6", "Runtime": "169 min",
            "Rated": "PG-13", "Genre": "Adventure, Drama, Sci-Fi",
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = omdb_payload
        with patch("modules.dionysus.engine.requests.get", return_value=mock_resp):
            result = engine.handle("recommend_movie", {"mood": "space epics"}, {})

        assert "8.6" in result["response"]
        assert "169 min" in result["response"]
        assert "PG-13" in result["response"]
        assert result["data"]["recommendations"][0]["imdb_rating"] == "8.6"

    def test_omdb_lookup_failure_degrades_to_na(self, tmp_path, monkeypatch):
        monkeypatch.setattr("modules.dionysus.engine.OMDB_KEY", "fake-key")
        engine, _ = make_engine(
            tmp_path,
            llm_response=json.dumps({"recommendations": [
                {"title": "Ghost Movie", "year": "2099", "reason": "obscure"}
            ]}),
        )
        with patch("modules.dionysus.engine.requests.get", side_effect=ConnectionError("boom")):
            result = engine.handle("recommend_movie", {"mood": "obscure"}, {})
        assert "N/A" in result["response"]

    def test_omdb_response_not_found_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("modules.dionysus.engine.OMDB_KEY", "fake-key")
        engine, _ = make_engine(
            tmp_path,
            llm_response=json.dumps({"recommendations": [
                {"title": "Nonexistent", "year": "2099", "reason": "x"}
            ]}),
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"Response": "False", "Error": "Movie not found!"}
        with patch("modules.dionysus.engine.requests.get", return_value=mock_resp):
            result = engine.handle("recommend_movie", {"mood": "x"}, {})
        assert "N/A" in result["response"]

    def test_malformed_llm_json_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="not valid json at all")
        result = engine.handle("recommend_movie", {"mood": "comedy"}, {})
        assert result["confidence"] == 0.3
        assert "trouble finding movies" in result["response"].lower()

    def test_defaults_mood_genre_when_absent(self, tmp_path):
        engine, llm = make_engine(
            tmp_path, llm_response=json.dumps({"recommendations": []})
        )
        engine.handle("recommend_movie", {}, {})
        assert "something good" in llm.last_prompt.lower()

    def test_rating_note_includes_rated_and_genre(self):
        note = DionysusEngine._rating_note({"Rated": "PG-13", "Genre": "Drama"})
        assert "PG-13" in note and "Drama" in note

    def test_rating_note_empty_when_no_metadata(self):
        assert DionysusEngine._rating_note({}) == ""


# ===========================================================================
# find_restaurant
# ===========================================================================

class TestFindRestaurant:
    def test_no_browser_agent_returns_unavailable(self, tmp_path):
        engine, _ = make_engine(tmp_path, browser_result=None)
        result = engine.handle("find_restaurant", {"cuisine": "Italian"}, {})
        assert result["confidence"] == 0.0
        assert "not available" in result["response"].lower()

    def test_happy_path_formats_results(self, tmp_path):
        engine, _ = make_engine(
            tmp_path,
            browser_result="Trattoria Roma | Pasta Palace | Little Italy Bistro",
        )
        result = engine.handle(
            "find_restaurant", {"cuisine": "Italian", "area": "Bandra"}, {}
        )
        assert result["confidence"] == 0.85
        assert "Trattoria Roma" in result["response"]
        assert "Pasta Palace" in result["response"]
        assert result["data"]["query"] == "best Italian restaurants in Bandra Mumbai"

    def test_no_results_returns_graceful_message(self, tmp_path):
        engine, _ = make_engine(tmp_path, browser_result="No results found.")
        result = engine.handle("find_restaurant", {"cuisine": "Klingon"}, {})
        assert result["confidence"] == 0.3
        assert "couldn't find" in result["response"].lower()

    def test_defaults_area_to_mumbai(self, tmp_path):
        engine, _ = make_engine(tmp_path, browser_result="Some Place")
        result = engine.handle("find_restaurant", {"cuisine": "cafe"}, {})
        assert "Mumbai" in result["data"]["query"]

    def test_budget_appended_to_query(self, tmp_path):
        engine, _ = make_engine(tmp_path, browser_result="Some Place")
        result = engine.handle(
            "find_restaurant", {"cuisine": "cafe", "budget": "cheap"}, {}
        )
        assert "cheap budget" in result["data"]["query"]

    def test_results_truncated_to_five(self, tmp_path):
        many = " | ".join(f"Place {i}" for i in range(10))
        engine, _ = make_engine(tmp_path, browser_result=many)
        result = engine.handle("find_restaurant", {"cuisine": "any"}, {})
        # Only 5 numbered lines plus header.
        numbered_lines = [l for l in result["response"].splitlines() if l.strip().startswith(tuple("12345"))]
        assert len(numbered_lines) == 5


# ===========================================================================
# recommend_music
# ===========================================================================

class TestRecommendMusic:
    def test_happy_path_with_spotify_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("modules.dionysus.engine.SPOTIFY_CLIENT_ID", "id")
        monkeypatch.setattr("modules.dionysus.engine.SPOTIFY_CLIENT_SECRET", "secret")
        engine, _ = make_engine(
            tmp_path,
            llm_response=json.dumps({"recommendations": [
                {"artist": "Tame Impala", "track": "The Less I Know The Better", "reason": "dreamy"}
            ]}),
        )
        token_resp = MagicMock()
        token_resp.raise_for_status.return_value = None
        token_resp.json.return_value = {"access_token": "tok123"}

        search_resp = MagicMock()
        search_resp.raise_for_status.return_value = None
        search_resp.json.return_value = {
            "tracks": {"items": [{
                "preview_url": "http://preview",
                "external_urls": {"spotify": "http://open.spotify.com/track/x"},
                "popularity": 77,
            }]}
        }

        with patch("modules.dionysus.engine.requests.post", return_value=token_resp), \
             patch("modules.dionysus.engine.requests.get", return_value=search_resp):
            result = engine.handle("recommend_music", {"mood": "chill"}, {})

        assert "Tame Impala" in result["response"]
        assert "77/100" in result["response"]
        assert "open.spotify.com" in result["response"]
        assert result["data"]["recommendations"][0]["popularity"] == 77

    def test_falls_back_to_audiodb_when_spotify_not_configured(self, tmp_path, monkeypatch):
        monkeypatch.setattr("modules.dionysus.engine.SPOTIFY_CLIENT_ID", "")
        monkeypatch.setattr("modules.dionysus.engine.SPOTIFY_CLIENT_SECRET", "")
        engine, _ = make_engine(
            tmp_path,
            llm_response=json.dumps({"recommendations": [
                {"artist": "Some Artist", "track": "Some Track", "reason": "nice"}
            ]}),
        )
        with patch("modules.dionysus.engine._fa_audiodb_artist",
                   return_value={"genre": "Indie Rock"}):
            result = engine.handle("recommend_music", {"mood": "chill"}, {})

        assert "Genre (TheAudioDB): Indie Rock" in result["response"]
        assert result["data"]["recommendations"][0]["popularity"] is None

    def test_audiodb_failure_is_swallowed_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setattr("modules.dionysus.engine.SPOTIFY_CLIENT_ID", "")
        engine, _ = make_engine(
            tmp_path,
            llm_response=json.dumps({"recommendations": [
                {"artist": "X", "track": "Y", "reason": "z"}
            ]}),
        )
        with patch("modules.dionysus.engine._fa_audiodb_artist",
                   side_effect=FreeAPIError("down")):
            result = engine.handle("recommend_music", {"mood": "chill"}, {})
        assert result["confidence"] == 0.9  # still succeeds overall
        assert "X" in result["response"]

    def test_malformed_llm_response_returns_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, llm_response="{broken json")
        result = engine.handle("recommend_music", {"mood": "sad"}, {})
        assert result["confidence"] == 0.3
        assert "trouble finding music" in result["response"].lower()

    def test_dismissed_music_excluded_from_prompt(self, tmp_path):
        engine, llm = make_engine(
            tmp_path, llm_response=json.dumps({"recommendations": []})
        )
        engine.db.log("music", "Old Song", "x", None)
        engine.db.dismiss("Old Song")
        engine.handle("recommend_music", {"mood": "anything"}, {})
        assert "Old Song" in llm.last_prompt


# ===========================================================================
# plan_outing
# ===========================================================================

class TestPlanOuting:
    def test_no_browser_returns_unavailable(self, tmp_path):
        engine, _ = make_engine(tmp_path, browser_result=None)
        result = engine.handle("plan_outing", {"topic": "weekend"}, {})
        assert result["confidence"] == 0.0

    def test_happy_path_builds_itinerary(self, tmp_path):
        engine, llm = make_engine(
            tmp_path,
            browser_result="Cafe Mocha | Marine Drive Cafe",
            llm_response=json.dumps({
                "title": "Chill Weekend",
                "slots": [
                    {"time": "Morning", "place": "Marine Drive", "activity": "walk",
                     "duration": "1h", "travel_to_next": "10 min"}
                ],
                "tips": ["bring water"],
            }),
        )
        with patch("modules.dionysus.engine._fa_meal_suggestion", side_effect=FreeAPIError("n/a")), \
             patch("modules.dionysus.engine._fa_trivia_question", side_effect=FreeAPIError("n/a")):
            result = engine.handle("plan_outing", {"topic": "weekend"}, {})

        assert result["confidence"] == 0.9
        assert "Chill Weekend" in result["response"]
        assert "Marine Drive" in result["response"]
        assert "bring water" in result["response"]

    def test_uses_memory_location_preference(self, tmp_path):
        memory = FakeMemory(location="Bandra")
        engine, llm = make_engine(
            tmp_path, browser_result="Some Place",
            llm_response=json.dumps({"title": "T", "slots": [], "tips": []}),
            memory=memory,
        )
        with patch("modules.dionysus.engine._fa_meal_suggestion", side_effect=FreeAPIError("n/a")), \
             patch("modules.dionysus.engine._fa_trivia_question", side_effect=FreeAPIError("n/a")):
            engine.handle("plan_outing", {"topic": "brunch"}, {})
        assert "Bandra" in llm.last_prompt

    def test_extra_tips_appended_when_free_apis_succeed(self, tmp_path):
        engine, _ = make_engine(
            tmp_path, browser_result="Some Place",
            llm_response=json.dumps({"title": "T", "slots": [], "tips": []}),
        )
        with patch("modules.dionysus.engine._fa_meal_suggestion",
                   return_value={"name": "Pasta", "area": "Italian"}), \
             patch("modules.dionysus.engine._fa_trivia_question",
                   return_value={"question": "What year was X founded?"}):
            result = engine.handle("plan_outing", {"topic": "date night"}, {})
        assert "Food idea: Pasta" in result["response"]
        assert "Icebreaker: What year was X founded?" in result["response"]

    def test_malformed_llm_response_returns_error(self, tmp_path):
        engine, _ = make_engine(tmp_path, browser_result="Some Place", llm_response="not json")
        result = engine.handle("plan_outing", {"topic": "day trip"}, {})
        assert result["confidence"] == 0.3

    def test_no_places_found_still_proceeds(self, tmp_path):
        engine, llm = make_engine(
            tmp_path, browser_result="No results found",
            llm_response=json.dumps({"title": "T", "slots": [], "tips": []}),
        )
        with patch("modules.dionysus.engine._fa_meal_suggestion", side_effect=FreeAPIError("n/a")), \
             patch("modules.dionysus.engine._fa_trivia_question", side_effect=FreeAPIError("n/a")):
            engine.handle("plan_outing", {"topic": "day trip"}, {})
        assert "No places found" in llm.last_prompt


class TestFormatOuting:
    def test_format_outing_renders_slots_and_tips(self):
        result = {
            "title": "Day Out",
            "slots": [
                {"time": "10am", "place": "Park", "activity": "walk",
                 "duration": "1h", "travel_to_next": "15 min"},
            ],
            "tips": ["wear sunscreen"],
        }
        formatted = DionysusEngine._format_outing(result)
        assert "Day Out" in formatted
        assert "Park" in formatted
        assert "wear sunscreen" in formatted
        assert "15 min" in formatted

    def test_format_outing_handles_missing_optional_fields(self):
        result = {"title": "Minimal", "slots": [{"time": "1pm"}], "tips": []}
        formatted = DionysusEngine._format_outing(result)
        assert "Minimal" in formatted  # must not raise on missing place/activity/etc


# ===========================================================================
# recommend_recipe
# ===========================================================================

class TestRecommendRecipe:
    def test_meal_suggestion_happy_path(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch("modules.dionysus.engine._fa_meal_suggestion",
                   return_value={"name": "Butter Chicken", "area": "Indian"}):
            result = engine.handle("recommend_recipe", {}, {})
        assert result["confidence"] == 0.85
        assert "Butter Chicken" in result["response"]
        assert "Indian cuisine" in result["response"]
        assert result["data"]["type"] == "meal"

    def test_cocktail_suggestion_when_category_drink(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch("modules.dionysus.engine._fa_cocktail_suggestion",
                   return_value={"name": "Mojito", "glass": "Highball"}):
            result = engine.handle("recommend_recipe", {"category": "drink"}, {})
        assert "Mojito" in result["response"]
        assert "Highball" in result["response"]
        assert result["data"]["type"] == "cocktail"

    def test_meal_suggestion_with_no_result_found(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch("modules.dionysus.engine._fa_meal_suggestion", return_value=None):
            result = engine.handle("recommend_recipe", {}, {})
        assert result["confidence"] == 0.3
        assert "couldn't find" in result["response"].lower()

    def test_cocktail_suggestion_with_no_result_found(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch("modules.dionysus.engine._fa_cocktail_suggestion", return_value=None):
            result = engine.handle("recommend_recipe", {"category": "drink"}, {})
        assert result["confidence"] == 0.3

    def test_upstream_api_failure_returns_graceful_error(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch("modules.dionysus.engine._fa_meal_suggestion",
                   side_effect=FreeAPIError("service down")):
            result = engine.handle("recommend_recipe", {}, {})
        assert result["confidence"] == 0.2
        assert "unavailable" in result["response"].lower()

    def test_cuisine_entity_passed_through(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        with patch("modules.dionysus.engine._fa_meal_suggestion",
                   return_value={"name": "Pad Thai", "area": "Thai"}) as mock_fn:
            engine.handle("recommend_recipe", {"cuisine": "Thai"}, {})
        mock_fn.assert_called_once_with("Thai")


# ===========================================================================
# LLM call helpers
# ===========================================================================

class TestOllamaCallHelpers:
    def test_uses_injected_llm_instance_when_present(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response='{"a": 1}')
        result = engine._ollama_call("some prompt")
        assert result == '{"a": 1}'
        assert llm.last_fmt == "json"

    def test_ollama_text_uses_no_fmt(self, tmp_path):
        engine, llm = make_engine(tmp_path, llm_response="plain text")
        result = engine._ollama_text("some prompt")
        assert result == "plain text"
        assert llm.last_fmt is None

    def test_falls_back_to_module_level_generate_without_llm_instance(self, tmp_path):
        engine = DionysusEngine(ollama_cfg={"model": "mistral", "host": "127.0.0.1", "port": 11434})
        engine.db = DionysusDB(str(tmp_path / "d.db"))
        with patch("modules.dionysus.engine.generate", return_value="fallback result") as mock_gen:
            result = engine._ollama_call("prompt text")
        assert result == "fallback result"
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("fmt") == "json"

    def test_parse_returns_none_on_invalid_json(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        assert engine._parse("not json", "some_intent") is None

    def test_parse_returns_dict_on_valid_json(self, tmp_path):
        engine, _ = make_engine(tmp_path)
        assert engine._parse('{"x": 1}', "some_intent") == {"x": 1}


# ===========================================================================
# DionysusDB
# ===========================================================================

class TestDionysusDB:
    def test_log_and_get_history(self, tmp_path):
        db = DionysusDB(str(tmp_path / "d.db"))
        db.log("movie", "Movie A", "detail", 8.0)
        db.log("movie", "Movie B", "detail2", 7.5)
        history = db.get_history("movie")
        titles = [h["title"] for h in history]
        assert "Movie A" in titles and "Movie B" in titles

    def test_get_history_excludes_dismissed(self, tmp_path):
        db = DionysusDB(str(tmp_path / "d.db"))
        db.log("movie", "Dismissed One", "x", None)
        db.dismiss("Dismissed One")
        history = db.get_history("movie")
        assert all(h["title"] != "Dismissed One" for h in history)

    def test_mark_seen_case_insensitive(self, tmp_path):
        db = DionysusDB(str(tmp_path / "d.db"))
        db.log("movie", "Inception", "x", None)
        updated = db.mark_seen("inception", type_="movie")
        assert updated is True
        assert "Inception" in db.seen_titles("movie")

    def test_mark_seen_updates_most_recent_match(self, tmp_path):
        db = DionysusDB(str(tmp_path / "d.db"))
        db.log("movie", "Repeat", "first", None)
        db.log("movie", "Repeat", "second", None)
        updated = db.mark_seen("Repeat", type_="movie")
        assert updated is True

    def test_mark_seen_returns_false_for_missing_title(self, tmp_path):
        db = DionysusDB(str(tmp_path / "d.db"))
        assert db.mark_seen("Never Logged", type_="movie") is False

    def test_dismissed_titles_filters_by_type(self, tmp_path):
        db = DionysusDB(str(tmp_path / "d.db"))
        db.log("movie", "M1", "", None)
        db.log("music", "S1", "", None)
        db.dismiss("M1")
        db.dismiss("S1")
        assert db.dismissed_titles("movie") == ["M1"]
        assert db.dismissed_titles("music") == ["S1"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))