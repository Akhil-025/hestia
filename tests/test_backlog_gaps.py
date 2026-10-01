# tests/test_backlog_gaps.py
"""
Tests for the user-path gaps found auditing the "done" backlog items:
  #36 stale review     #38 confidence recall   #40 periodic digest job
  #44 export           #45 importance          #48 dated recall
  #49 memory stats     #63 source feedback     #64 score breakdown
  #74 EXIF search
Each item already had an engine method; these test that a person can reach it.
"""
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_mnemosyne import make_engine  # noqa: E402
from modules.mnemosyne.dates import resolve_range, local_day_bounds_utc  # noqa: E402
from modules.hecate.intent_registry import INTENT_MODULE_MAP  # noqa: E402

NEW_INTENTS = ("review_stale_facts", "export_memory", "set_fact_importance",
               "recall_on_date", "get_memory_stats")


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp()
    eng, _ = make_engine(tmp)
    eng.vector_store = None            # SQL-only behaviour; no embeddings involved
    yield eng
    shutil.rmtree(tmp, ignore_errors=True)


def _old(engine, key, days):
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with engine.db._lock, engine.db._conn:
        engine.db._conn.execute(
            "UPDATE facts SET last_accessed=?, created_at=?, updated_at=? WHERE key=?", (ts, ts, ts, key))


# --------------------------------------------------------------------- registry
def test_new_intents_registered_and_handled(engine):
    for i in NEW_INTENTS:
        assert INTENT_MODULE_MAP[i] == "mnemosyne"
        assert engine.can_handle(i)


def test_aliases_resolve_to_new_intents():
    from core.intent_aliases import IntentAliasResolver
    r = IntentAliasResolver("config/intent_aliases.yaml")
    assert r.resolve("what did I say last Tuesday") == "recall_on_date"
    assert r.resolve("export my memory as markdown") == "export_memory"
    assert r.resolve("how many facts do you remember") == "get_memory_stats"
    assert r.resolve("keep all the stale facts") == "review_stale_facts"
    assert r.resolve("mark my allergy as very important") == "set_fact_importance"
    assert r.resolve("the second source wasn't relevant") == "athena_mark_feedback"
    assert r.resolve("show scores for entropy") == "athena_search"
    assert r.resolve("what did we talk about yesterday") is None   # stays with get_history


def test_every_new_intent_is_in_the_nlu_prompt():
    text = open(os.path.join(os.path.dirname(__file__), "..", "config", "nlu_prompt.txt"), encoding="utf-8").read()
    for i in NEW_INTENTS + ("athena_mark_feedback",):
        assert i in text


# ------------------------------------------------------------------------ dates
T = date(2026, 10, 1)   # a Thursday


@pytest.mark.parametrize("phrase,start,end", [
    ("what did I say yesterday", "2026-09-30", "2026-09-30"),
    ("last tuesday", "2026-09-29", "2026-09-29"),
    ("on 3 march", "2026-03-03", "2026-03-03"),
    ("march 3rd 2025", "2025-03-03", "2025-03-03"),
    ("photos from March 2024", "2024-03-01", "2024-03-31"),
    ("2 days ago", "2026-09-29", "2026-09-29"),
    ("last week", "2026-09-21", "2026-09-27"),
    ("photos from 2023", "2023-01-01", "2023-12-31"),
    ("2025-12-31", "2025-12-31", "2025-12-31"),
])
def test_resolve_range(phrase, start, end):
    r = resolve_range(phrase, T)
    assert (r.start.isoformat(), r.end.isoformat()) == (start, end)


@pytest.mark.parametrize("phrase", ["what do you know about me", "I may go", "taken in 1850", "feb 30", ""])
def test_resolve_range_rejects_non_dates(phrase):
    assert resolve_range(phrase, T) is None


def test_local_day_bounds_span_exactly_the_days():
    lo, hi = local_day_bounds_utc(date(2026, 9, 30), date(2026, 10, 1))
    a = datetime.strptime(lo, "%Y-%m-%d %H:%M:%S")
    b = datetime.strptime(hi, "%Y-%m-%d %H:%M:%S")
    assert timedelta(hours=47) <= b - a <= timedelta(hours=49)   # 48h, +-1h around a DST change


# ------------------------------------------------------------------ #48 recall
def _log_at(engine, text, local_dt, intent="chat"):
    engine.db.push_interaction(text, "ok", intent)
    utc = local_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with engine.db._lock, engine.db._conn:
        engine.db._conn.execute("UPDATE interaction_log SET pushed_at=? WHERE user_text=?", (utc, text))


def test_dated_recall_finds_yesterday_and_skips_meta_queries(engine):
    y = datetime.combine(date.today() - timedelta(days=1), datetime.min.time()).replace(hour=12).astimezone()
    _log_at(engine, "plan the trip to Goa", y)
    _log_at(engine, "what did I say yesterday", y, intent="recall_on_date")
    r = engine.handle("recall_on_date", {"raw_query": "what did I say yesterday"}, {})
    assert "Goa" in r["response"] and "what did I say" not in r["response"].split("talked about:")[-1]
    assert r["data"]["start"] == (date.today() - timedelta(days=1)).isoformat()


def test_dated_recall_nothing_recorded(engine):
    r = engine.handle("recall_on_date", {"date": "last week"}, {})
    assert "don't have anything" in r["response"]


def test_dated_recall_asks_when_no_date_given(engine):
    r = engine.handle("recall_on_date", {"raw_query": "what did I say"}, {})
    assert r["data"]["needs_clarification"] and r["data"]["missing_slot"] == "answer"
    r2 = engine.handle("recall_on_date", {"answer": "yesterday"}, {})
    assert "needs_clarification" not in r2["data"]


# -------------------------------------------------------------------- #44 / #49
def test_export_has_no_1000_row_cap(engine):
    for i in range(1005):
        engine.db.set_fact(f"k{i:04d}", "v")
    assert len(json.loads(engine.export_memory("json"))["facts"]) == 1005


def test_export_intent_writes_file(engine):
    engine.db.set_fact("pet", "Rex")
    r = engine.handle("export_memory", {"raw_query": "export my memory as markdown"}, {})
    path = r["data"]["path"]
    assert path.endswith(".md") and os.path.exists(path)
    assert "**pet**: Rex" in open(path, encoding="utf-8").read()
    r = engine.handle("export_memory", {}, {})
    assert r["data"]["path"].endswith(".json")


def test_export_cli(engine, capsys):
    engine.db.set_fact("pet", "Rex")
    from scripts.export_memory import main
    assert main(["--db", engine.config.db_path, "--stdout"]) == 0
    assert '"pet"' in capsys.readouterr().out
    out = os.path.join(os.path.dirname(engine.config.db_path), "cli_out")
    assert main(["--db", engine.config.db_path, "--out", out, "--format", "markdown"]) == 0
    assert os.listdir(out)
    assert main(["--db", os.path.join(out, "nope.db")]) == 1


def test_memory_stats_intent(engine):
    engine.db.set_fact("pet", "Rex")
    r = engine.handle("get_memory_stats", {}, {})
    assert "1 fact" in r["response"] and "database takes up" in r["response"]
    assert r["data"]["facts"] == 1


# ------------------------------------------------------------------------ #45
def test_importance_intent_sets_weight(engine):
    engine.learn("peanut_allergy", "peanuts")
    r = engine.handle("set_fact_importance", {"raw_query": "mark my peanut allergy as very important"}, {})
    assert engine.db.get_fact_row("peanut_allergy")["importance"] == 0.9
    assert "peanut allergy" in r["response"]
    r = engine.handle("set_fact_importance", {"key": "peanut allergy", "importance": "critical"}, {})
    assert engine.db.get_fact_row("peanut_allergy")["importance"] == 1.0


def test_importance_ambiguous_and_unknown(engine):
    engine.learn("sister_name", "Priya")
    engine.learn("brother_name", "Raj")
    r = engine.handle("set_fact_importance", {"key": "name", "importance": "high"}, {})
    assert r["data"]["needs_clarification"] and "sister name" in r["response"]
    r = engine.handle("set_fact_importance", {"key": "zzz", "importance": "high"}, {})
    assert "couldn't find" in r["response"]
    r = engine.handle("set_fact_importance", {"key": "sister_name"}, {})
    assert r["data"]["needs_clarification"]            # no level given -> asks


def test_importance_changes_ranking(engine):
    engine.learn("a_fact", "x"); engine.learn("b_fact", "y")
    engine.handle("set_fact_importance", {"key": "a_fact", "importance": "low"}, {})
    engine.handle("set_fact_importance", {"key": "b_fact", "importance": "critical"}, {})
    assert engine.get_top_facts_scored(2)[0]["key"] == "b_fact"


# ------------------------------------------------------------------------ #36
def test_stale_review_lists_then_keeps(engine):
    engine.learn("old_phone", "Nokia"); engine.learn("fresh", "yes")
    _old(engine, "old_phone", 400)
    r = engine.handle("review_stale_facts", {}, {})
    assert "old phone" in r["response"] and "fresh" not in r["response"]
    assert engine.db.count_stale_facts() == 1
    assert "1 remembered fact hasn't come up" in engine.get_stale_brief()
    r = engine.handle("review_stale_facts", {"raw_query": "keep my old phone"}, {})
    assert "Keeping" in r["response"] and engine.db.count_stale_facts() == 0
    assert engine.run_decay_check() == []              # a kept fact is not re-flagged
    assert engine.get_stale_brief() == ""


def test_stale_keep_all_and_empty(engine):
    assert "Nothing is flagged" in engine.handle("review_stale_facts", {}, {})["response"]
    engine.learn("a", "1"); engine.learn("b", "2")
    _old(engine, "a", 400); _old(engine, "b", 400)
    r = engine.handle("review_stale_facts", {"raw_query": "keep all the stale facts"}, {})
    assert "all 2" in r["response"]


def test_stale_facts_are_voiced_in_recall(engine):
    engine.learn("old_phone", "Nokia")
    _old(engine, "old_phone", 400)
    engine.run_decay_check()
    r = engine.handle("get_user_info", {"key": "old_phone"}, {})
    assert "out of date" in r["response"]


def test_morning_brief_speaks_stale_line():
    from core.heartbeat import HestiaHeartbeat
    mn = MagicMock()
    mn.get_study_brief.return_value = ""
    mn.get_stale_brief.return_value = "2 remembered facts haven't come up in a long while."
    hb = HestiaHeartbeat(mnemosyne=mn)
    with patch("core.heartbeat.bus") as bus:
        hb._morning_brief()
    spoken = [c.args[1]["text"] for c in bus.emit.call_args_list if c.args[0] == "speak"]
    assert any("haven't come up" in s for s in spoken)


# ------------------------------------------------------------------------ #38
def test_confidence_by_source_and_hedge(engine):
    engine.learn("guess", "1", source="inferred")
    assert engine.db.get_fact_row("guess")["confidence"] == 0.6
    engine.learn("sure", "1")
    assert engine.db.get_fact_row("sure")["confidence"] == 1.0
    r = engine.handle("learn_fact", {"key": "locker", "value": "42", "raw_query": "I think my locker is 42"}, {})
    assert "weren't sure" in r["response"]
    assert engine.db.get_fact_row("locker")["confidence"] == 0.6
    r = engine.handle("learn_fact", {"key": "bank", "value": "HDFC", "raw_query": "my bank is HDFC", "confidence": "0.3"}, {})
    assert engine.db.get_fact_row("bank")["confidence"] == 0.3


def test_recall_expresses_uncertainty(engine):
    engine.learn("locker", "42", confidence=0.6)
    engine.learn("pin_hint", "x", confidence=0.3)
    engine.learn("sure", "1")
    assert "not fully certain" in engine.handle("get_user_info", {"key": "locker"}, {})["response"]
    assert "not confident" in engine.handle("get_user_info", {"key": "pin_hint"}, {})["response"]
    assert "certain" not in engine.handle("get_user_info", {"key": "sure"}, {})["response"]
    line = engine._format_result({"text": "42", "metadata": {"type": "fact", "key": "locker", "confidence": 0.6}})
    assert "not fully certain" in line


# ------------------------------------------------------------------------ #40
def _summary(engine, days_ago, topic="chat"):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
    sid = engine.db.add_summary(ts, ts, f"talked about {topic} {days_ago}", topic, 5)
    with engine.db._lock, engine.db._conn:
        engine.db._conn.execute("UPDATE summaries SET created_at=? WHERE id=?", (ts, sid))


def test_digest_runs_from_background_jobs_and_is_persisted(engine):
    _summary(engine, 10); _summary(engine, 3)
    assert engine.digest_due("weekly") and not engine.digest_due("monthly")
    ran = engine.run_background_jobs(now=time.time())
    assert ran.get("digest_weekly") is True and "digest_monthly" not in ran
    assert engine.db.get_latest_summary_time("weekly_digest")
    assert not engine.digest_due("weekly")                       # persisted, survives a restart
    assert "digest_weekly" not in engine.run_background_jobs(now=time.time() + 7 * 3600)


def test_digest_excludes_old_summaries_and_prior_digests(engine):
    _summary(engine, 20, "ancient"); _summary(engine, 2, "recent"); _summary(engine, 1, "weekly_digest")
    seen = {}
    engine.hestia_llm = SimpleNamespace(generate=lambda p, fmt=None: seen.setdefault("p", p) and "A digest.")
    assert engine.generate_periodic_digest("weekly") == "A digest."
    assert "recent" in seen["p"] and "ancient" not in seen["p"] and "weekly_digest" not in seen["p"]


def test_digest_not_built_without_history(engine):
    assert not engine.digest_due("weekly")
    assert "digest_weekly" not in engine.run_background_jobs(now=time.time())


def test_background_jobs_run_decay(engine):
    engine.learn("old_phone", "Nokia")
    _old(engine, "old_phone", 400)
    assert engine.run_background_jobs(now=time.time())["decay"] == 1
    assert engine.db.count_stale_facts() == 1


# ----------------------------------------------------------------- #63 / #64
def _athena(sources, metrics=None):
    from modules.athena.engine import AthenaEngine
    a = object.__new__(AthenaEngine)
    a._last_sources, a._debug_scores = [], False
    a.rag = MagicMock()
    docs = [SimpleNamespace(to_dict=lambda include_score_breakdown=False, s=s: dict(
        s, **({"semantic_score": 0.8, "bm25_score": 0.4} if include_score_breakdown else {}))) for s in sources]
    a.query_service = SimpleNamespace(execute=lambda q: SimpleNamespace(answer="The answer.", sources=docs, metrics=metrics))
    return a


SRC = [{"file_name": "a.pdf", "subject": "physics", "module": "thermo", "page": 3, "chunk_number": 1, "score": 0.9},
       {"file_name": "b.pdf", "subject": "physics", "module": "thermo", "page": 7, "chunk_number": 4, "score": 0.7}]


def test_feedback_uses_last_search_sources():
    a = _athena(SRC)
    r = a.handle("mark_feedback", {"raw_query": "the second source wasn't relevant"}, {})
    assert "need the source" in r["response"]                      # nothing searched yet
    a.handle("search", {"query": "entropy"}, {})
    r = a.handle("mark_feedback", {"raw_query": "the second source wasn't relevant"}, {})
    meta, relevant = a.rag.mark_feedback.call_args.args
    assert meta == {"file_name": "b.pdf", "subject": "physics", "module": "thermo", "page_number": 7, "chunk_number": 4}
    assert relevant is False and "b.pdf" in r["response"]
    a.handle("mark_feedback", {"index": "1", "relevant": "true"}, {})
    assert a.rag.mark_feedback.call_args.args[0]["file_name"] == "a.pdf" and a.rag.mark_feedback.call_args.args[1] is True
    a.handle("mark_feedback", {"file_name": "b.pdf", "raw_query": "that was useful"}, {})
    assert a.rag.mark_feedback.call_args.args[1] is True


def test_feedback_asks_or_rejects_when_unclear():
    a = _athena(SRC)
    a.handle("search", {"query": "entropy"}, {})
    assert "Which source" in a.handle("mark_feedback", {"raw_query": "mark that"}, {})["response"]
    assert "only showed 2" in a.handle("mark_feedback", {"index": 5}, {})["response"]
    a.rag.mark_feedback.assert_not_called()


def test_feedback_accepts_page_key_from_a_source_dict():
    a = _athena(SRC)
    a.handle("mark_feedback", dict(SRC[0], relevant=False), {})
    assert a.rag.mark_feedback.call_args.args[0]["page_number"] == 3


def test_citations_use_last_search():
    a = _athena(SRC)
    assert "don't have a recent search" in a.handle("get_citations", {}, {})["response"]
    a.handle("search", {"query": "entropy"}, {})
    assert a.handle("get_citations", {}, {})["data"]["count"] == 2


def test_score_breakdown_in_reply_text():
    a = _athena(SRC, metrics={"avg_score": 0.8})
    plain = a.handle("search", {"query": "entropy"}, {})["response"]
    assert "semantic" not in plain
    shown = a.handle("search", {"query": "show scores for entropy"}, {})["response"]
    assert "The answer." in shown and "semantic 0.80" in shown and "BM25 0.40" in shown
    assert a.handle("search", {"query": "x", "debug": True}, {})["response"].count("Retrieval scores") == 1


def test_score_toggle_persists_for_session():
    a = _athena(SRC)
    assert "on" in a.handle("search", {"raw_query": "turn on retrieval scores"}, {})["response"]
    assert "Retrieval scores:" in a.handle("search", {"query": "entropy"}, {})["response"]
    a.handle("search", {"raw_query": "turn off retrieval scores"}, {})
    assert "Retrieval scores:" not in a.handle("search", {"query": "entropy"}, {})["response"]


def test_web_feedback_and_debug_routes():
    from web_ui import HestiaWebUI
    athena = MagicMock()
    athena.handle.return_value = {"response": "Noted", "data": {}}
    c = HestiaWebUI(memory=MagicMock(), athena=athena).app.test_client()
    assert c.post("/api/athena/feedback", json={"index": 2, "relevant": False}).status_code == 200
    assert athena.handle.call_args.args[:2] == ("mark_feedback", {"index": 2, "relevant": False})
    assert c.post("/api/athena/feedback", json={}).status_code == 400
    c.post("/api/athena/query", json={"query": "q", "debug": True})
    assert athena.handle.call_args.args[1] == {"query": "q", "debug": True}


def test_web_export_and_dashboard(engine):
    from web_ui import HestiaWebUI
    engine.db.set_fact("pet", "Rex")
    c = HestiaWebUI(memory=engine).app.test_client()
    r = c.get("/api/mnemosyne/export")
    assert r.status_code == 200 and json.loads(r.data)["facts"][0]["key"] == "pet"
    assert "attachment" in r.headers["Content-Disposition"]
    assert b"**pet**" in c.get("/api/mnemosyne/export?format=markdown").data
    assert c.get("/api/mnemosyne/dashboard").get_json()["facts"] == 1


# ------------------------------------------------------------------------ #74
@pytest.fixture
def iris(tmp_path):
    from modules.iris.db import IrisDB
    from modules.iris.iris_engine import IrisEngine
    e = object.__new__(IrisEngine)
    e.db, e.embedder, e.vector_index = IrisDB(str(tmp_path / "iris.db")), None, None

    def add(path, caption, **exif):
        fid = e.db.insert_file(path, path, path, 1, "image", "image/jpeg")
        e.db.update_file_analysis(fid, caption, "[]", "[]", "", False, 0.0)
        e.db.update_file_exif(fid, **exif)
    add("goa_beach.jpg", "beach at sunset", date_taken="2024-03-05T10:00:00", camera_make="Canon",
        camera_model="EOS R", gps_lat=15.3, gps_lon=73.9)
    add("pune_dog.jpg", "a dog in the park", date_taken="2024-03-20T09:00:00", camera_make="Apple", camera_model="iPhone 13")
    add("old_beach.jpg", "beach holiday", date_taken="2021-06-01T09:00:00", camera_make="Apple", camera_model="iPhone 8")
    add("no_exif.jpg", "screenshot of a beach map")
    return e


def test_search_by_date_range(iris):
    out = iris.search("photos from March 2024")
    assert "goa_beach" in out and "pune_dog" in out and "old_beach" not in out and "no_exif" not in out
    assert "taken March 2024" in out


def test_search_by_camera(iris):
    out = iris.search("pictures shot on my iPhone")
    assert "pune_dog" in out and "old_beach" in out and "goa_beach" not in out


def test_search_by_location(iris):
    assert "goa_beach" in iris.search("geotagged photos") and "pune_dog" not in iris.search("geotagged photos")
    assert "pune_dog" in iris.search("photos without location")


def test_exif_filter_combines_with_caption_text(iris):
    out = iris.search("beach photos from 2024")
    assert "goa_beach" in out and "old_beach" not in out and "pune_dog" not in out
    assert iris.search("dog photos taken with a canon") is None


def test_plain_caption_search_unchanged(iris):
    out = iris.search("beach")
    assert "goa_beach" in out and "old_beach" in out and "no_exif" in out
    assert iris.search("photos of paris") is None


def test_iris_handle_routes_exif_query(iris):
    r = iris.handle("search", {"raw_query": "photos from March 2024"}, {})
    assert "goa_beach" in r["response"] and r["confidence"] == 0.85
