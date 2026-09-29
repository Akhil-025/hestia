# tests/test_mnemosyne_extended.py
"""
Tests for the Mnemosyne — Memory & Knowledge backlog batch:
  #36 fact expiry/decay            #37 contradiction detection
  #38 confidence/source tracking   #40 weekly/monthly digest
  #41 bulk "forget everything about X"   #42 semantic deduplication
  #44 memory export                #45 importance-weighted ranking
  #48 dated recall                 #49 memory dashboard
  #50 memory provenance

Reuses tests/test_mnemosyne.py's `make_engine` (real MnemosyneDB, a real
or stubbed ChromaDB per that file's `_ensure_stubs()`) rather than
duplicating that fixture. Most tests here exercise pure SQL/Python logic
that doesn't touch the vector store at all — MnemosyneEngine wraps every
vector_store call in try/except, so those are robust regardless of
whether a real or stubbed ChromaDB is active in this pytest session (see
test_mnemosyne.py's docstring for why that can vary by collection order).

For the handful of tests that genuinely need controlled, deterministic
embedding-similarity behavior (deduplication, contradiction interplay,
touch-on-recall), `engine.vector_store` is swapped for a small local fake
whose `search()` result is set explicitly per test, rather than relying
on real embeddings to happen to land above/below a similarity threshold.
"""
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_mnemosyne import make_engine  # noqa: E402
from modules.mnemosyne.engine import MnemosyneEngine  # noqa: E402


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp()
    eng, _ = make_engine(tmp)
    yield eng
    shutil.rmtree(tmp, ignore_errors=True)


class _FakeVectorStore:
    """Deterministic stand-in for MnemosyneVectorStore.search()/add()/delete()."""

    def __init__(self):
        self.added = []
        self.deleted = []
        self._search_results = []

    def set_search_results(self, results):
        self._search_results = results

    def add(self, text, metadata, doc_id):
        self.added.append((text, metadata, doc_id))

    def search(self, query, n_results=5, where=None):
        return list(self._search_results)

    def delete(self, doc_id):
        self.deleted.append(doc_id)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# #38 — confidence/source tracking (schema already had the columns; this
# tests they actually round-trip and are used correctly).
# ---------------------------------------------------------------------------

def test_fact_source_and_confidence_round_trip(engine):
    engine.learn("favorite_food", "biryani", source="user")
    row = engine.db.get_fact_row("favorite_food")
    assert row["source"] == "user"
    assert row["confidence"] == 1.0


def test_fact_source_can_be_inferred(engine):
    engine.learn("commute_mode", "train", source="inferred")
    row = engine.db.get_fact_row("commute_mode")
    assert row["source"] == "inferred"


def test_get_fact_row_returns_none_for_unknown_key(engine):
    assert engine.db.get_fact_row("nonexistent_key") is None


# ---------------------------------------------------------------------------
# #36 — fact expiry / decay
# ---------------------------------------------------------------------------

def test_freshly_learned_fact_is_not_stale(engine):
    engine.learn("favorite_food", "biryani")
    flagged = engine.run_decay_check(months=6)
    assert flagged == []


def test_old_untouched_fact_is_flagged_stale(engine):
    engine.learn("old_fact", "something")
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(days=400))
    with engine.db._lock, engine.db._conn:
        engine.db._conn.execute(
            "UPDATE facts SET last_accessed = ? WHERE key = ?", (old_ts, "old_fact")
        )
    flagged = engine.run_decay_check(months=6)
    assert [f["key"] for f in flagged] == ["old_fact"]


def test_decay_never_deletes_only_flags(engine):
    engine.learn("old_fact", "something")
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(days=400))
    with engine.db._lock, engine.db._conn:
        engine.db._conn.execute(
            "UPDATE facts SET last_accessed = ? WHERE key = ?", (old_ts, "old_fact")
        )
    engine.run_decay_check(months=6)
    # Still readable — decay flags for review, it never deletes.
    assert engine.db.get_fact("old_fact") == "something"


def test_decay_does_not_reflag_an_already_flagged_fact(engine):
    engine.learn("old_fact", "something")
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(days=400))
    with engine.db._lock, engine.db._conn:
        engine.db._conn.execute(
            "UPDATE facts SET last_accessed = ? WHERE key = ?", (old_ts, "old_fact")
        )
    first = engine.run_decay_check(months=6)
    second = engine.run_decay_check(months=6)
    assert len(first) == 1
    assert second == []  # already flagged, not re-reported


def test_touching_a_stale_fact_clears_the_flag(engine):
    engine.learn("old_fact", "something")
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(days=400))
    with engine.db._lock, engine.db._conn:
        engine.db._conn.execute(
            "UPDATE facts SET last_accessed = ? WHERE key = ?", (old_ts, "old_fact")
        )
    engine.run_decay_check(months=6)
    engine.db.touch_fact("old_fact")
    remaining = engine.get_stale_facts_for_review()
    assert remaining == []


def test_get_stale_facts_for_review_lists_flagged_facts(engine):
    engine.learn("old_fact", "something")
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(days=400))
    with engine.db._lock, engine.db._conn:
        engine.db._conn.execute(
            "UPDATE facts SET last_accessed = ? WHERE key = ?", (old_ts, "old_fact")
        )
    engine.run_decay_check(months=6)
    review = engine.get_stale_facts_for_review()
    assert review[0]["key"] == "old_fact"


# ---------------------------------------------------------------------------
# #45 — importance-weighted ranking
# ---------------------------------------------------------------------------

def test_score_fact_prefers_recent_over_old():
    now = datetime.now(timezone.utc)
    recent = {"updated_at": _iso(now), "access_count": 0, "importance": 0.5, "confidence": 1.0}
    old = {"updated_at": _iso(now - timedelta(days=60)), "access_count": 0, "importance": 0.5, "confidence": 1.0}
    assert MnemosyneEngine._score_fact(recent, now) > MnemosyneEngine._score_fact(old, now)


def test_score_fact_prefers_frequently_accessed():
    now = datetime.now(timezone.utc)
    base = {"updated_at": _iso(now), "importance": 0.5, "confidence": 1.0}
    frequent = {**base, "access_count": 20}
    rare = {**base, "access_count": 0}
    assert MnemosyneEngine._score_fact(frequent, now) > MnemosyneEngine._score_fact(rare, now)


def test_score_fact_respects_explicit_importance():
    now = datetime.now(timezone.utc)
    base = {"updated_at": _iso(now), "access_count": 0, "confidence": 1.0}
    important = {**base, "importance": 1.0}
    unimportant = {**base, "importance": 0.0}
    assert MnemosyneEngine._score_fact(important, now) > MnemosyneEngine._score_fact(unimportant, now)


def test_score_fact_handles_missing_or_malformed_fields():
    # Must never raise on a sparse/legacy row.
    score = MnemosyneEngine._score_fact({})
    assert 0.0 <= score <= 1.0


def test_get_top_facts_scored_ranks_important_fact_above_stale_recent_one(engine):
    engine.learn("frequently_used", "value A")
    engine.learn("just_touched_once", "value B")
    for _ in range(15):
        engine.db.touch_fact("frequently_used")
    top = engine.get_top_facts_scored(limit=2)
    keys = [f["key"] for f in top]
    assert keys.index("frequently_used") < keys.index("just_touched_once")


def test_set_fact_importance_affects_ranking(engine):
    engine.learn("low_importance", "value A")
    engine.learn("boosted", "value B")
    engine.set_fact_importance("boosted", 1.0)
    engine.set_fact_importance("low_importance", 0.0)
    top = engine.get_top_facts_scored(limit=2)
    assert top[0]["key"] == "boosted"


def test_set_fact_importance_returns_false_for_unknown_key(engine):
    assert engine.set_fact_importance("nonexistent", 0.8) is False


def test_set_fact_importance_clamps_out_of_range_values(engine):
    engine.learn("k", "v")
    engine.set_fact_importance("k", 5.0)
    assert engine.db.get_fact_row("k")["importance"] == 1.0
    engine.set_fact_importance("k", -3.0)
    assert engine.db.get_fact_row("k")["importance"] == 0.0


def test_get_top_facts_for_context_uses_scored_ranking(engine):
    engine.learn("a", "value A")
    engine.learn("b", "value B")
    text = engine.get_top_facts_for_context(limit=2)
    assert "a: value A" in text
    assert "b: value B" in text


# ---------------------------------------------------------------------------
# #37 — contradiction detection
# ---------------------------------------------------------------------------

def test_similar_key_different_value_is_a_contradiction_candidate(engine):
    engine.learn("favorite_color", "blue")
    conflict = engine.check_for_contradiction("favourite_colour", "red")
    assert conflict == {"key": "favorite_color", "value": "blue"}


def test_similar_key_same_value_is_not_a_contradiction(engine):
    engine.learn("favorite_color", "blue")
    conflict = engine.check_for_contradiction("favourite_colour", "blue")
    assert conflict is None


def test_unrelated_key_is_not_a_contradiction(engine):
    engine.learn("favorite_color", "blue")
    conflict = engine.check_for_contradiction("birthday", "March 5")
    assert conflict is None


def test_updating_the_same_key_is_not_a_contradiction(engine):
    engine.learn("favorite_color", "blue")
    conflict = engine.check_for_contradiction("favorite_color", "red")
    assert conflict is None


def test_no_existing_facts_means_no_contradiction(engine):
    assert engine.check_for_contradiction("anything", "value") is None


def test_learn_fact_surfaces_a_contradiction_note_without_blocking(engine):
    engine.learn("favorite_color", "blue")
    result = engine.handle(
        "learn_fact", {"key": "favourite_colour", "value": "red"}, {}
    )
    assert "differ from what you told me" in result["response"]
    assert "favorite color" in result["response"]
    # Non-blocking: the new fact was still written.
    assert engine.db.get_fact("favourite_colour") == "red"


def test_learn_fact_without_a_conflict_has_a_clean_response(engine):
    result = engine.handle("learn_fact", {"key": "birthday", "value": "March 5"}, {})
    assert "differ" not in result["response"]


# ---------------------------------------------------------------------------
# #42 — semantic deduplication on ingest
# ---------------------------------------------------------------------------

def test_new_key_with_duplicate_value_is_deduplicated(engine):
    fake = _FakeVectorStore()
    engine.vector_store = fake
    engine.db.set_fact("existing_key", "I love hiking", source="user")

    fake.set_search_results([
        {"text": "I love hiking", "metadata": {"type": "fact", "key": "existing_key"}, "score": 0.97, "id": "existing_key"},
    ])

    result = engine.learn("new_key", "I love hiking")
    assert result == {"deduplicated": True, "matched_key": "existing_key"}
    # No second SQL row written under the new key.
    assert engine.db.get_fact("new_key") is None
    # No new vector embedding written either.
    assert fake.added == []


def test_deduplication_bumps_the_existing_facts_access_count(engine):
    fake = _FakeVectorStore()
    engine.vector_store = fake
    engine.db.set_fact("existing_key", "I love hiking", source="user")
    fake.set_search_results([
        {"text": "I love hiking", "metadata": {"type": "fact", "key": "existing_key"}, "score": 0.97, "id": "existing_key"},
    ])
    engine.learn("new_key", "I love hiking")
    assert engine.db.get_fact_row("existing_key")["access_count"] == 1


def test_low_similarity_match_is_not_deduplicated(engine):
    fake = _FakeVectorStore()
    engine.vector_store = fake
    fake.set_search_results([
        {"text": "something unrelated", "metadata": {"type": "fact", "key": "other_key"}, "score": 0.3, "id": "other_key"},
    ])
    result = engine.learn("new_key", "completely different value")
    assert result["deduplicated"] is False
    assert engine.db.get_fact("new_key") == "completely different value"


def test_match_under_the_same_key_is_not_deduplicated(engine):
    # An update to an existing key is a legitimate upsert, not a dup.
    fake = _FakeVectorStore()
    engine.vector_store = fake
    engine.db.set_fact("same_key", "old value")
    fake.set_search_results([
        {"text": "old value", "metadata": {"type": "fact", "key": "same_key"}, "score": 0.99, "id": "same_key"},
    ])
    result = engine.learn("same_key", "new value")
    assert result["deduplicated"] is False
    assert engine.db.get_fact("same_key") == "new value"


def test_dedup_check_is_skipped_without_a_vector_store():
    tmp = tempfile.mkdtemp()
    try:
        eng, _ = make_engine(tmp)
        eng.vector_store = None
        result = eng.learn("k", "v")
        assert result == {"deduplicated": False, "matched_key": None}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_learn_fact_handler_reports_deduplication(engine):
    fake = _FakeVectorStore()
    engine.vector_store = fake
    engine.db.set_fact("existing_key", "I love hiking", source="user")
    fake.set_search_results([
        {"text": "I love hiking", "metadata": {"type": "fact", "key": "existing_key"}, "score": 0.97, "id": "existing_key"},
    ])
    result = engine.handle("learn_fact", {"key": "new_key", "value": "I love hiking"}, {})
    assert "already have that" in result["response"]
    assert result["data"]["deduplicated"] is True


# ---------------------------------------------------------------------------
# #41 — bulk "forget everything about X"
# ---------------------------------------------------------------------------

def test_find_matching_facts_by_substring(engine):
    engine.learn("pluto_favorite_stock", "AAPL")
    engine.learn("pluto_risk_tolerance", "moderate")
    engine.learn("unrelated_fact", "x")
    matches = engine.find_matching_facts("pluto")
    assert set(matches) == {"pluto_favorite_stock", "pluto_risk_tolerance"}


def test_forget_matching_removes_every_match(engine):
    engine.learn("pluto_a", "1")
    engine.learn("pluto_b", "2")
    removed = engine.forget_matching(["pluto_a", "pluto_b"])
    assert set(removed) == {"pluto_a", "pluto_b"}
    assert engine.db.get_fact("pluto_a") is None
    assert engine.db.get_fact("pluto_b") is None


def test_bulk_forget_asks_for_confirmation_first(engine):
    engine.learn("pluto_a", "1")
    engine.learn("pluto_b", "2")
    result = engine.handle("forget_fact", {"pattern": "pluto"}, {})
    assert result.get("needs_confirmation") is True
    assert "2 thing(s)" in result["response"]
    # Nothing deleted yet.
    assert engine.db.get_fact("pluto_a") == "1"


def test_bulk_forget_confirmed_actually_deletes(engine):
    engine.learn("pluto_a", "1")
    engine.learn("pluto_b", "2")
    result = engine.handle("forget_fact", {"pattern": "pluto", "_confirmed": True}, {})
    assert "2 thing(s)" in result["response"]
    assert engine.db.get_fact("pluto_a") is None
    assert engine.db.get_fact("pluto_b") is None


def test_bulk_forget_with_no_matches_reports_nothing_to_forget(engine):
    result = engine.handle("forget_fact", {"pattern": "nonexistent"}, {})
    assert "don't have anything" in result["response"]
    assert result.get("needs_confirmation") is not True


def test_forget_fact_still_supports_single_key_path(engine):
    # Regression: the bulk-pattern addition must not break the original
    # single-key forget flow.
    engine.learn("k", "v")
    result = engine.handle("forget_fact", {"key": "k", "_confirmed": True}, {})
    assert "Forgotten: k." == result["response"]
    assert engine.db.get_fact("k") is None


# ---------------------------------------------------------------------------
# #44 — memory export
# ---------------------------------------------------------------------------

def test_export_json_includes_facts_and_is_parseable(engine):
    import json
    engine.learn("k", "v")
    text = engine.export_memory(fmt="json")
    parsed = json.loads(text)
    assert parsed["facts"][0]["key"] == "k"
    assert "exported_at" in parsed


def test_export_markdown_includes_facts_readably(engine):
    engine.learn("favorite_food", "biryani")
    text = engine.export_memory(fmt="markdown")
    assert "# Hestia memory export" in text
    assert "**favorite_food**: biryani" in text


def test_export_includes_goals(engine):
    engine.db.add_goal("Learn to cook")
    text = engine.export_memory(fmt="json")
    import json
    parsed = json.loads(text)
    assert any(g["text"] == "Learn to cook" for g in parsed["goals"])


def test_export_with_no_data_does_not_raise(engine):
    text = engine.export_memory(fmt="json")
    import json
    parsed = json.loads(text)
    assert parsed["facts"] == []


# ---------------------------------------------------------------------------
# #49 — memory dashboard
# ---------------------------------------------------------------------------

def test_dashboard_reports_fact_count(engine):
    engine.learn("a", "1")
    engine.learn("b", "2")
    dash = engine.get_memory_dashboard()
    assert dash["facts"] == 2


def test_dashboard_reports_db_size_as_positive(engine):
    engine.learn("a", "1")
    dash = engine.get_memory_dashboard()
    assert dash["db_size_bytes"] > 0


def test_dashboard_reports_stale_fact_count(engine):
    engine.learn("old_fact", "something")
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(days=400))
    with engine.db._lock, engine.db._conn:
        engine.db._conn.execute(
            "UPDATE facts SET last_accessed = ? WHERE key = ?", (old_ts, "old_fact")
        )
    engine.run_decay_check(months=6)
    dash = engine.get_memory_dashboard()
    assert dash["stale_facts"] == 1


def test_dashboard_never_raises_when_vector_store_is_none(engine):
    engine.vector_store = None
    dash = engine.get_memory_dashboard()
    assert dash["embedding_count"] is None


# ---------------------------------------------------------------------------
# #48 — dated recall
# ---------------------------------------------------------------------------

def test_recall_on_date_finds_facts_created_that_day(engine):
    engine.learn("k", "v")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = engine.recall_on_date(today)
    assert "your k is v" in text.lower()


def test_recall_on_date_finds_interactions_that_day(engine):
    engine.push("what's the weather", "It's sunny.", "get_weather")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = engine.recall_on_date(today)
    assert "weather" in text.lower()


def test_recall_on_date_with_nothing_returns_empty_string(engine):
    assert engine.recall_on_date("2020-01-01") == ""


def test_recall_on_date_does_not_leak_other_days(engine):
    engine.learn("k", "v")
    text = engine.recall_on_date("2020-01-01")
    assert text == ""


# ---------------------------------------------------------------------------
# #50 — memory provenance
# ---------------------------------------------------------------------------

def test_provenance_phrase_for_today():
    now = datetime.now(timezone.utc)
    phrase = MnemosyneEngine._provenance_phrase(now.isoformat(), "user")
    assert "today" in phrase


def test_provenance_phrase_for_yesterday():
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    phrase = MnemosyneEngine._provenance_phrase(yesterday.isoformat(), "user")
    assert "yesterday" in phrase


def test_provenance_phrase_includes_non_user_source():
    now = datetime.now(timezone.utc)
    phrase = MnemosyneEngine._provenance_phrase(now.isoformat(), "inferred")
    assert "inferred" in phrase


def test_provenance_phrase_omits_plain_user_source():
    now = datetime.now(timezone.utc)
    phrase = MnemosyneEngine._provenance_phrase(now.isoformat(), "user")
    assert "user" not in phrase.replace("today", "")


def test_provenance_phrase_empty_for_no_timestamp():
    assert MnemosyneEngine._provenance_phrase(None, "user") == ""


def test_provenance_phrase_tolerates_malformed_timestamp():
    assert MnemosyneEngine._provenance_phrase("not-a-date", "user") == ""


def test_get_user_info_response_includes_provenance(engine):
    engine.learn("user_name", "Priya")
    result = engine.handle("get_user_info", {"key": "user_name"}, {})
    assert "today" in result["response"]


def test_touching_a_fact_via_get_user_info_increments_access_count(engine):
    engine.learn("user_name", "Priya")
    engine.handle("get_user_info", {"key": "user_name"}, {})
    assert engine.db.get_fact_row("user_name")["access_count"] == 1


# ---------------------------------------------------------------------------
# #40 — weekly/monthly digest
# ---------------------------------------------------------------------------

class _DigestLLM:
    def generate(self, prompt, fmt=None):
        return "This week covered cooking, travel planning, and a few reminders."


def test_generate_periodic_digest_with_no_summaries_returns_none(engine):
    assert engine.generate_periodic_digest("weekly") is None


def test_generate_periodic_digest_produces_and_stores_a_summary(engine):
    engine.hestia_llm = _DigestLLM()
    engine.add_summary("2026-01-01", "2026-01-02", "Talked about cooking.", "food", 5)
    digest = engine.generate_periodic_digest("weekly")
    assert digest == "This week covered cooking, travel planning, and a few reminders."
    recent = engine.db.get_recent_summaries(n=5)
    assert any(s["topic"] == "weekly_digest" for s in recent)


def test_generate_periodic_digest_monthly_uses_a_wider_window(engine):
    engine.hestia_llm = _DigestLLM()
    engine.add_summary("2026-01-01", "2026-01-02", "Talked about cooking.", "food", 5)
    digest = engine.generate_periodic_digest("monthly")
    assert digest is not None
    recent = engine.db.get_recent_summaries(n=5)
    assert any(s["topic"] == "monthly_digest" for s in recent)


def test_generate_periodic_digest_survives_llm_failure(engine):
    class _BrokenLLM:
        def generate(self, prompt, fmt=None):
            raise RuntimeError("ollama down")

    engine.hestia_llm = _BrokenLLM()
    engine.add_summary("2026-01-01", "2026-01-02", "Talked about cooking.", "food", 5)
    assert engine.generate_periodic_digest("weekly") is None


def test_generate_periodic_digest_survives_empty_llm_response(engine):
    class _EmptyLLM:
        def generate(self, prompt, fmt=None):
            return "   "

    engine.hestia_llm = _EmptyLLM()
    engine.add_summary("2026-01-01", "2026-01-02", "Talked about cooking.", "food", 5)
    assert engine.generate_periodic_digest("weekly") is None
