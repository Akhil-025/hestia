# tests/test_dionysus_backlog.py
"""
Tests for the Dionysus backlog items (#146, #147, #150, #151, #152, #269):

  #146  find_events         event search, filtering, logging
  #147  dismissal expiry    dismissed titles can come back after N days
  #150  surprise_me         picks deliberately outside recent taste
  #151  cost / budget       budget parsing, restaurant note, outing totals
  #152  schedule_recharge   repeating Chronos reminder (real ChronosEngine)
  #269  more/less like this feedback that steers later recommendations

Run with:  pytest tests/test_dionysus_backlog.py -v
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chronos.engine import ChronosEngine
from modules.dionysus.db import DionysusDB
from modules.dionysus.engine import (
    DionysusEngine,
    _budget_note,
    _outing_total,
    _parse_budget,
    _price_in_text,
    _to_rupees,
)
from modules.hecate.intent_registry import INTENT_MODULE_MAP
from modules.mnemosyne.db import MnemosyneDB

IST = ZoneInfo("Asia/Kolkata")

MOVIES = json.dumps({"recommendations": [
    {"title": "Arrival", "year": "2016", "reason": "thoughtful sci-fi"},
]})
MUSIC = json.dumps({"recommendations": [
    {"artist": "Nils Frahm", "track": "Says", "reason": "calm and building"},
]})


class FakeLLM:
    def __init__(self, response=""):
        self.response = response
        self.prompts = []

    def generate(self, prompt, fmt=None):
        self.prompts.append(prompt)
        return self.response


class FakeBrowser:
    """search_web returns a canned string per call (or the same one each time)."""
    def __init__(self, *results):
        self.results = list(results) or [""]
        self.queries = []

    def search_web(self, query):
        self.queries.append(query)
        i = min(len(self.queries) - 1, len(self.results) - 1)
        return self.results[i]


class FakeMemory:
    def __init__(self, location=""):
        self.location = location

    def get_preference(self, key, default=""):
        return self.location if key == "location" else default


def make_engine(tmp_path, llm_response="", browser=None, memory=None, **kw):
    engine = DionysusEngine(ollama_cfg={}, browser_agent=browser, memory=memory,
                            llm=FakeLLM(llm_response), **kw)
    engine.db = DionysusDB(str(tmp_path / "d.db"))
    return engine, engine._llm_instance


def backdate_dismissal(db, title, days):
    with db._conn:
        db._conn.execute(
            "UPDATE recommendations SET dismissed_at = datetime('now', ?) WHERE title = ?",
            (f"-{days} days", title),
        )


# ===========================================================================
# Plumbing
# ===========================================================================

NEW_INTENTS = ("find_events", "surprise_me", "schedule_recharge",
               "more_like_this", "less_like_this")


def test_every_registered_dionysus_intent_is_handled_by_the_engine(tmp_path):
    from modules.hecate.intent_registry import strip_module_prefix
    engine, _ = make_engine(tmp_path)
    registered = [i for i, m in INTENT_MODULE_MAP.items() if m == "dionysus"]
    assert len(registered) >= 12
    for intent in registered:
        assert engine.can_handle(strip_module_prefix(intent)), intent


def test_new_intents_are_handled_and_registered(tmp_path):
    engine, _ = make_engine(tmp_path)
    for intent in NEW_INTENTS:
        assert engine.can_handle(intent)
        assert INTENT_MODULE_MAP[f"dionysus_{intent}"] == "dionysus"


# ===========================================================================
# DB: migration + expiry (#147)
# ===========================================================================

def test_old_database_is_migrated_without_losing_rows(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, title TEXT NOT NULL,
            detail TEXT, rating REAL, dismissed BOOLEAN DEFAULT 0, seen BOOLEAN DEFAULT 0,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        INSERT INTO recommendations (type, title, dismissed) VALUES ('movie', 'Old Dismissed', 1);
        INSERT INTO recommendations (type, title, seen) VALUES ('movie', 'Old Seen', 1);
    """)
    conn.commit()
    conn.close()

    db = DionysusDB(path)
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(recommendations)")}
    assert {"dismissed_at", "feedback"} <= cols
    assert db.dismissed_titles("movie") == ["Old Dismissed"]
    assert db.seen_titles("movie") == ["Old Seen"]
    # Pre-existing dismissals are dated "now", so a 30-day expiry doesn't drop them.
    assert db.dismissed_titles("movie", expire_days=30) == ["Old Dismissed"]


def test_dismissed_titles_never_expire_by_default(tmp_path):
    db = DionysusDB(str(tmp_path / "d.db"))
    db.log("movie", "Cats")
    db.dismiss("Cats")
    backdate_dismissal(db, "Cats", 5000)
    assert db.dismissed_titles("movie") == ["Cats"]
    assert db.dismissed_titles("movie", expire_days=0) == ["Cats"]


def test_dismissal_expires_after_the_window(tmp_path):
    db = DionysusDB(str(tmp_path / "d.db"))
    db.log("movie", "Cats")
    db.log("movie", "Dune")
    db.dismiss("Cats")
    db.dismiss("Dune")
    backdate_dismissal(db, "Cats", 200)
    backdate_dismissal(db, "Dune", 10)
    assert db.dismissed_titles("movie", expire_days=180) == ["Dune"]


def test_expired_dismissal_is_back_in_the_recommendation_prompt(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES, dismiss_expire_days=180)
    engine.db.log("movie", "Cats")
    engine.db.dismiss("Cats")
    engine.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})
    assert "Cats" in llm.prompts[-1]
    backdate_dismissal(engine.db, "Cats", 400)
    engine.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})
    assert "Avoid these already seen/dismissed: none" in llm.prompts[-1]


def test_expiry_can_be_turned_off(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES, dismiss_expire_days=0)
    engine.db.log("movie", "Cats")
    engine.db.dismiss("Cats")
    backdate_dismissal(engine.db, "Cats", 4000)
    engine.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})
    assert "Cats" in llm.prompts[-1]


def test_watched_titles_never_expire(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES, dismiss_expire_days=1)
    engine.db.log("movie", "Heat")
    engine.db.mark_seen("Heat")
    with engine.db._conn:
        engine.db._conn.execute("UPDATE recommendations SET logged_at = datetime('now','-900 days')")
    engine.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})
    assert "Heat" in llm.prompts[-1]


def test_dismissals_persist_across_restart(tmp_path):
    path = str(tmp_path / "d.db")
    db = DionysusDB(path)
    db.log("movie", "Cats")
    db.dismiss("Cats")
    db._conn.close()
    assert DionysusDB(path).dismissed_titles("movie", expire_days=180) == ["Cats"]


# ===========================================================================
# #269 more / less like this
# ===========================================================================

def test_set_feedback_matches_case_insensitively_and_by_type(tmp_path):
    db = DionysusDB(str(tmp_path / "d.db"))
    db.log("movie", "Inception")
    db.log("music", "Inception")
    assert db.set_feedback("inception", 1, "music") == {"title": "Inception", "type": "music"}
    assert db.feedback_titles("music", 1) == ["Inception"]
    assert db.feedback_titles("movie", 1) == []
    assert db.set_feedback("nothing here", 1) is None


def test_more_like_this_records_feedback_and_asks_for_similar(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES)
    engine.db.log("movie", "Interstellar")
    out = engine.handle("more_like_this", {"title": "interstellar"}, {})
    assert out["response"].startswith("Noted, more like 'Interstellar'.")
    assert "Arrival" in out["response"]
    assert "similar in feel to Interstellar" in llm.prompts[-1]
    assert out["data"]["feedback"] == 1
    assert engine.db.feedback_titles("movie", 1) == ["Interstellar"]


def test_liked_titles_steer_and_are_not_recommended_back(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES)
    engine.db.log("movie", "Interstellar")
    engine.db.set_feedback("Interstellar", 1, "movie")
    engine.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})
    prompt = llm.prompts[-1]
    assert "The user liked these (lean toward similar): Interstellar." in prompt
    assert "Avoid these already seen/dismissed: Interstellar" in prompt


def test_less_like_this_dismisses_and_steers_away(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES)
    engine.db.log("movie", "Cats")
    out = engine.handle("less_like_this", {"title": "Cats"}, {})
    assert "less like 'Cats'" in out["response"]
    assert "steer away from similar" in out["response"]
    assert llm.prompts == []                      # no LLM call needed
    engine.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})
    prompt = llm.prompts[-1]
    assert "The user did not like these (avoid similar): Cats." in prompt
    assert "Avoid these already seen/dismissed: Cats" in prompt


def test_disliked_titles_stay_excluded_after_dismissal_expires(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES, dismiss_expire_days=30)
    engine.db.log("movie", "Cats")
    engine.handle("less_like_this", {"title": "Cats"}, {})
    backdate_dismissal(engine.db, "Cats", 365)
    engine.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})
    assert "Avoid these already seen/dismissed: Cats" in llm.prompts[-1]


@pytest.mark.parametrize("phrase", ["that", "this one", "It", "the last one"])
def test_no_title_or_pronoun_uses_the_latest_recommendation(tmp_path, phrase):
    engine, _ = make_engine(tmp_path, MOVIES)
    engine.db.log("movie", "Older")
    engine.db.log("music", "Nils Frahm — Says")
    out = engine.handle("less_like_this", {"title": phrase}, {})
    assert out["data"]["title"] == "Nils Frahm — Says"
    assert out["data"]["type"] == "music"


def test_latest_can_be_narrowed_by_type(tmp_path):
    engine, _ = make_engine(tmp_path, MOVIES)
    engine.db.log("movie", "A Movie")
    engine.db.log("music", "A Song")
    out = engine.handle("less_like_this", {"type": "film"}, {})
    assert out["data"]["title"] == "A Movie"


def test_feedback_with_nothing_to_refer_to_asks_which(tmp_path):
    engine, _ = make_engine(tmp_path)
    out = engine.handle("more_like_this", {}, {})
    assert out["response"] == "Which recommendation do you mean?"
    assert out["confidence"] == 0.5


def test_more_like_an_unknown_title_still_works_and_is_remembered(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES)
    out = engine.handle("more_like_this", {"title": "Whiplash", "type": "movie"}, {})
    assert out["data"]["known"] is False
    assert "similar in feel to Whiplash" in llm.prompts[-1]
    assert engine.db.feedback_titles("movie", 1) == ["Whiplash"]


def test_more_like_music_uses_the_music_prompt(tmp_path):
    engine, llm = make_engine(tmp_path, MUSIC)
    engine.db.log("music", "Brian Eno — An Ending")
    out = engine.handle("more_like_this", {}, {})
    assert "Nils Frahm" in out["response"]
    assert "Recommend 5 songs" in llm.prompts[-1]


def test_more_like_non_steering_type_only_records(tmp_path):
    engine, llm = make_engine(tmp_path)
    engine.db.log("recipe", "Pad Thai")
    out = engine.handle("more_like_this", {"title": "Pad Thai"}, {})
    assert "Noted, you liked 'Pad Thai'" in out["response"]
    assert llm.prompts == []


def test_less_like_a_restaurant_still_dismisses_it(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.db.log("event", "Comedy night at X")
    engine.handle("less_like_this", {"title": "Comedy night at X"}, {})
    assert engine.db.dismissed_titles("event") == ["Comedy night at X"]


# ===========================================================================
# #150 surprise me
# ===========================================================================

def test_surprise_movie_asks_for_something_outside_recent_taste(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES)
    for t in ("Interstellar", "Arrival"):
        engine.db.log("movie", t)
    engine.db.set_feedback("Interstellar", 1, "movie")
    out = engine.handle("surprise_me", {"raw_query": "surprise me"}, {})
    prompt = llm.prompts[-1]
    assert "SURPRISE REQUEST" in prompt
    assert "Interstellar" in prompt and "Arrival" in prompt
    assert "lean toward similar" not in prompt       # liked titles must not pull it back
    assert out["response"].startswith("Surprise movies")
    assert out["data"]["surprise"] is True


def test_surprise_music_when_the_user_says_music(tmp_path):
    engine, llm = make_engine(tmp_path, MUSIC)
    out = engine.handle("surprise_me", {"raw_query": "surprise me with some music"}, {})
    assert "Recommend 5 songs" in llm.prompts[-1]
    assert out["response"].startswith("Surprise music")


def test_surprise_ignores_the_logged_mood(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES)
    apollo = MagicMock()
    apollo.recent_mood_context.return_value = {"mood": "tired", "low_trend": False}
    engine.attach_apollo(apollo)
    # A generic query, so a normal request WOULD pick up the logged mood.
    engine.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})
    assert "tired" in llm.prompts[-1]
    apollo.recent_mood_context.reset_mock()
    engine.handle("surprise_me", {"raw_query": "recommend something"}, {})
    assert "tired" not in llm.prompts[-1]
    apollo.recent_mood_context.assert_not_called()


def test_surprise_still_excludes_dismissed_and_seen(tmp_path):
    engine, llm = make_engine(tmp_path, MOVIES)
    engine.db.log("movie", "Cats")
    engine.db.dismiss("Cats")
    engine.handle("surprise_me", {"raw_query": "surprise me"}, {})
    assert "Avoid these already seen/dismissed: Cats" in llm.prompts[-1]


# ===========================================================================
# #151 budget / cost
# ===========================================================================

@pytest.mark.parametrize("raw, amount, tier", [
    ("under 1500", 1500, None),
    ("\u20b92,000", 2000, None),
    ("2k", 2000, None),
    ("1.5 lakh", 150000, None),
    ("2000 for two", 1000, None),
    ("cheap", None, "cheap"),
    ("mid-range", None, "mid"),
    ("fine dining", None, "premium"),
    ("cheap, under 400", 400, "cheap"),
])
def test_parse_budget(raw, amount, tier):
    info = _parse_budget(raw)
    assert info["amount"] == amount and info["tier"] == tier


@pytest.mark.parametrize("raw", ["", None, "none", "whatever", "n/a"])
def test_parse_budget_returns_none_without_a_budget(raw):
    assert _parse_budget(raw) is None


@pytest.mark.parametrize("value, expected", [
    (400, 400.0), ("400", 400.0), ("\u20b9300-500", 500.0), ("free", 0.0),
    ("1,200", 1200.0), ("1.5k", 1500.0), (0, 0.0),
    (None, None), ("varies", None), (-5, None), (True, None),
])
def test_to_rupees(value, expected):
    assert _to_rupees(value) == expected


def test_outing_total_skips_slots_without_a_usable_number():
    result = {"slots": [{"cost_per_person": 300}, {"cost_per_person": "n/a"},
                        {"cost_per_person": "free"}, "junk", {}]}
    assert _outing_total(result) == 300.0
    assert _outing_total({"slots": [{"place": "x"}]}) is None
    assert _outing_total({}) is None


def test_price_in_text_per_person_and_for_two():
    assert _price_in_text("Great thali \u20b9400 only") == 400.0
    assert _price_in_text("Bistro, Rs. 1,800 for two") == 900.0
    assert _price_in_text("no price here") is None


def test_budget_note_wording():
    assert "cheap" in _budget_note(_parse_budget("cheap"))
    assert "not live prices" in _budget_note(_parse_budget("cheap"))
    assert "1,500+" in _budget_note(_parse_budget("fine dining"))
    assert "check the menu" in _budget_note(_parse_budget("under 800"))
    assert _budget_note(None) == ""


def test_restaurant_reply_adds_budget_note_and_flags_over_budget(tmp_path):
    browser = FakeBrowser("Cheap Bites \u20b9300 | Posh Place \u20b92,400 per head | Plain Cafe")
    engine, _ = make_engine(tmp_path, browser=browser)
    out = engine.handle("find_restaurant", {"cuisine": "any", "budget": "under 1000"}, {})
    text = out["response"]
    assert "Cheap Bites \u20b9300\n" in text + "\n"
    assert "Posh Place \u20b92,400 per head  (price shown is over your budget)" in text
    assert "Budget: about \u20b91,000 per person" in text
    assert out["data"]["budget"]["amount"] == 1000


def test_restaurant_reply_without_budget_is_unchanged(tmp_path):
    engine, _ = make_engine(tmp_path, browser=FakeBrowser("A | B"))
    out = engine.handle("find_restaurant", {"cuisine": "thai"}, {})
    assert "Budget" not in out["response"]
    assert out["data"]["budget"] is None


OUTING = {
    "title": "Day out", "tips": [],
    "slots": [
        {"time": "Morning", "place": "Park", "activity": "walk", "duration": "2h",
         "cost_per_person": 0, "travel_to_next": "10 min"},
        {"time": "Afternoon", "place": "Cafe", "activity": "lunch", "duration": "1h",
         "cost_per_person": "\u20b9600-800", "travel_to_next": ""},
        {"time": "Evening", "place": "Show", "activity": "comedy", "duration": "2h",
         "cost_per_person": 700},
    ],
}


def outing_engine(tmp_path, outing=OUTING):
    return make_engine(tmp_path, json.dumps(outing), browser=FakeBrowser("A place"))


def test_outing_total_is_added_up_in_code(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.dionysus.engine._fa_meal_suggestion", lambda *a, **k: None)
    monkeypatch.setattr("modules.dionysus.engine._fa_trivia_question", lambda *a, **k: None)
    engine, _ = outing_engine(tmp_path)
    out = engine.handle("plan_outing", {"raw_query": "plan a day"}, {})
    assert out["data"]["estimated_cost_per_person"] == 1500.0
    assert "Estimated total: ~\u20b91,500 per person" in out["response"]
    assert "Cost     : free" in out["response"]
    assert "~\u20b9800 per person" in out["response"]        # range uses the upper end
    assert "over your budget" not in out["response"]


def test_outing_budget_goes_into_prompt_and_flags_overspend(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.dionysus.engine._fa_meal_suggestion", lambda *a, **k: None)
    monkeypatch.setattr("modules.dionysus.engine._fa_trivia_question", lambda *a, **k: None)
    engine, llm = outing_engine(tmp_path)
    out = engine.handle("plan_outing", {"raw_query": "plan", "budget": "under 1000"}, {})
    assert "budget is about \u20b91,000 per person" in llm.prompts[-1]
    assert out["data"]["over_budget_by"] == 500.0
    assert "about \u20b9500 over your budget of \u20b91,000" in out["response"]


def test_outing_tier_budget_uses_the_top_of_the_range(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.dionysus.engine._fa_meal_suggestion", lambda *a, **k: None)
    monkeypatch.setattr("modules.dionysus.engine._fa_trivia_question", lambda *a, **k: None)
    engine, llm = outing_engine(tmp_path)
    out = engine.handle("plan_outing", {"raw_query": "plan", "budget": "cheap"}, {})
    assert "cheap options" in llm.prompts[-1]
    assert out["data"]["budget_per_person"] == 500
    assert out["data"]["over_budget_by"] == 1000.0


def test_outing_without_cost_numbers_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.dionysus.engine._fa_meal_suggestion", lambda *a, **k: None)
    monkeypatch.setattr("modules.dionysus.engine._fa_trivia_question", lambda *a, **k: None)
    plain = {"title": "T", "slots": [{"time": "Morning", "place": "P", "activity": "a",
                                       "duration": "1h"}], "tips": []}
    engine, _ = outing_engine(tmp_path, plain)
    out = engine.handle("plan_outing", {"raw_query": "plan"}, {})
    assert "Estimated total" not in out["response"]
    assert "Cost" not in out["response"]
    assert "estimated_cost_per_person" not in out["data"]


# ===========================================================================
# #146 find_events
# ===========================================================================

def test_events_need_a_browser(tmp_path):
    engine, _ = make_engine(tmp_path)
    out = engine.handle("find_events", {}, {})
    assert out["confidence"] == 0.0


def test_events_query_and_reply(tmp_path):
    browser = FakeBrowser("Jazz Night | Stand-up Show", "Food Fest")
    engine, _ = make_engine(tmp_path, browser=browser)
    out = engine.handle("find_events",
                        {"category": "music", "area": "Bandra", "date": "this weekend"}, {})
    assert browser.queries[0] == "music events in Bandra Mumbai this weekend"
    assert browser.queries[1].endswith("BookMyShow Insider")
    assert "  1. Jazz Night" in out["response"] and "  3. Food Fest" in out["response"]
    assert "not confirmed listings" in out["response"]
    assert out["data"]["events"] == ["Jazz Night", "Stand-up Show", "Food Fest"]


def test_events_near_me_uses_saved_location(tmp_path):
    browser = FakeBrowser("Something")
    engine, _ = make_engine(tmp_path, browser=browser, memory=FakeMemory("Powai"))
    engine.handle("find_events", {"location": "near me"}, {})
    assert browser.queries[0] == "events in Powai Mumbai"


def test_events_default_to_mumbai_without_repeating_it(tmp_path):
    browser = FakeBrowser("Something")
    engine, _ = make_engine(tmp_path, browser=browser)
    engine.handle("find_events", {}, {})
    assert browser.queries[0] == "events in Mumbai"


def test_events_are_deduplicated_and_capped_at_five(tmp_path):
    browser = FakeBrowser("A | B | C", "a | D | E | F | G")
    engine, _ = make_engine(tmp_path, browser=browser)
    out = engine.handle("find_events", {}, {})
    assert out["data"]["events"] == ["A", "B", "C", "D", "E"]


def test_events_hide_dismissed_and_seen_ones(tmp_path):
    engine, _ = make_engine(tmp_path, browser=FakeBrowser("Old Show | New Show | Went Already"))
    engine.db.log("event", "Old Show")
    engine.db.dismiss("Old Show")
    engine.db.log("event", "Went Already")
    engine.db.mark_seen("Went Already", type_="event")
    out = engine.handle("find_events", {}, {})
    assert out["data"]["events"] == ["New Show"]


def test_events_are_logged_so_feedback_can_target_them(tmp_path):
    engine, _ = make_engine(tmp_path, browser=FakeBrowser("Jazz Night"))
    engine.handle("find_events", {}, {})
    out = engine.handle("less_like_this", {}, {})
    assert out["data"]["title"] == "Jazz Night" and out["data"]["type"] == "event"


def test_events_with_no_results(tmp_path):
    engine, _ = make_engine(tmp_path, browser=FakeBrowser("No results found for 'x'."))
    out = engine.handle("find_events", {"category": "opera"}, {})
    assert out["confidence"] == 0.3
    assert "opera" in out["response"]


# ===========================================================================
# #152 schedule_recharge
# ===========================================================================

class ChronosRig:
    def __init__(self, tmp_path):
        mem = MagicMock()
        mem.db = MnemosyneDB(str(tmp_path / "m.db"))
        mem.get_device_location.return_value = None
        now = datetime(2026, 10, 1, 9, 0, tzinfo=IST).astimezone(timezone.utc)   # a Thursday
        self.chronos = ChronosEngine(memory=mem, local_tz="Asia/Kolkata",
                                     clock=lambda: now, notify=MagicMock(),
                                     exports_dir=str(tmp_path))
        self.dionysus, _ = make_engine(tmp_path)
        self.dionysus.attach_chronos(self.chronos)

    def pending(self):
        return self.chronos._svc.list_pending()


def test_recharge_needs_chronos(tmp_path):
    engine, _ = make_engine(tmp_path)
    out = engine.handle("schedule_recharge", {}, {})
    assert "Chronos" in out["response"] and out["confidence"] == 0.2


def test_recharge_creates_a_real_repeating_reminder(tmp_path):
    rig = ChronosRig(tmp_path)
    out = rig.dionysus.handle(
        "schedule_recharge",
        {"raw_query": "schedule a weekend recharge every Saturday at 5pm for 3 hours"}, {})
    assert out["confidence"] == 0.9
    assert out["data"]["recurring"] is True
    assert out["data"]["schedule"] == "every Saturday at 5pm"
    assert out["data"]["duration"] == "3 hours"
    pending = rig.pending()
    assert len(pending) == 1 and "recharge break (3 hours)" in pending[0]["text"]
    assert "5:00 PM" in out["response"] and "cancel reminder recharge" in out["response"]
    saved = rig.dionysus.db.list_routines()
    assert saved[0]["schedule"] == "every Saturday at 5pm"
    assert saved[0]["reminder_id"] == str(out["data"]["id"])


def test_recharge_uses_defaults_when_nothing_is_said(tmp_path):
    rig = ChronosRig(tmp_path)
    out = rig.dionysus.handle("schedule_recharge", {"raw_query": "set up a recharge routine"}, {})
    assert out["data"]["schedule"] == "every Sunday at 4pm"
    assert out["data"]["duration"] == "2 hours"
    assert len(rig.pending()) == 1


def test_recharge_understands_weekend(tmp_path):
    rig = ChronosRig(tmp_path)
    out = rig.dionysus.handle("schedule_recharge", {"raw_query": "block out weekend downtime"}, {})
    assert out["data"]["schedule"] == "every weekend at 4pm"
    assert out["confidence"] == 0.9


def test_recharge_entities_win_over_raw_text(tmp_path):
    rig = ChronosRig(tmp_path)
    out = rig.dionysus.handle(
        "schedule_recharge",
        {"raw_query": "recharge", "schedule": "every Friday at 8pm", "duration": "90 minutes"}, {})
    assert out["data"]["schedule"] == "every Friday at 8pm"
    assert out["data"]["duration"] == "90 minutes"


def test_recharge_is_not_created_twice(tmp_path):
    rig = ChronosRig(tmp_path)
    ask = {"raw_query": "recharge every Sunday at 4pm"}
    rig.dionysus.handle("schedule_recharge", ask, {})
    again = rig.dionysus.handle("schedule_recharge", ask, {})
    assert "already have a recharge routine" in again["response"]
    assert len(rig.pending()) == 1


def test_recharge_passes_chronos_questions_through(tmp_path):
    engine, _ = make_engine(tmp_path)
    chronos = MagicMock()
    chronos.handle.return_value = {"response": "What time?", "data": {}, "confidence": 0.5}
    engine.attach_chronos(chronos)
    out = engine.handle("schedule_recharge", {"raw_query": "recharge"}, {})
    assert out["response"] == "What time?" and out["confidence"] == 0.5
    assert engine.db.list_routines() == []


def test_recharge_survives_a_chronos_crash(tmp_path):
    engine, _ = make_engine(tmp_path)
    chronos = MagicMock()
    chronos.handle.side_effect = RuntimeError("boom")
    engine.attach_chronos(chronos)
    out = engine.handle("schedule_recharge", {"raw_query": "recharge"}, {})
    assert "couldn't set" in out["response"]
    assert out["confidence"] == 0.2
