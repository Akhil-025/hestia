# tests/test_apollo_crossmodule.py
"""
Cross-module pieces of the Apollo backlog:
  Artemis completion history (#126/#183), tension surfacing (#159),
  mood-aware Dionysus (#149), DB maintenance (#233), heartbeat hooks
  (#115/#118/#120/#161/#233) and the dashboard endpoints (#183).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.consensus import ConsensusEngine
from core.db_maintenance import DBMaintenance
from modules.apollo.engine import ApolloEngine
from modules.artemis.tracker import HISTORY_CAP_DAYS, ArtemisTracker, Habit

UTC = timezone.utc
NOW = datetime.now(UTC).replace(hour=15, minute=0, second=0, microsecond=0)


class FakeLLM:
    def generate(self, *a, **k):
        return "ok"


def apollo(tmp_path, **cfg):
    return ApolloEngine(ollama_cfg={}, db_path=Path(tmp_path) / "apollo.db",
                        llm=FakeLLM(), config=cfg or None)


# ===================================================== Artemis history

class TestArtemisHistory:
    def _tracker(self, tmp_path):
        return ArtemisTracker(str(tmp_path / "artemis_state.json")) \
            if "state_path" not in ArtemisTracker.__init__.__code__.co_varnames \
            else ArtemisTracker(state_path=str(tmp_path / "artemis_state.json"))

    def test_completion_dates_are_recorded_and_persist(self, tmp_path):
        t = self._tracker(tmp_path)
        t.add_habit("meditate")
        d1, d2 = date(2026, 9, 1), date(2026, 9, 2)
        t.complete_habit("meditate", today=d1)
        t.complete_habit("meditate", today=d2)
        h = t.habit_history()["meditate"]
        assert h["dates"] == ["2026-09-01", "2026-09-02"] and h["since"] == "2026-09-01"
        reloaded = self._tracker(tmp_path).habit_history()["meditate"]
        assert reloaded["dates"] == h["dates"]

    def test_same_day_twice_does_not_duplicate(self, tmp_path):
        t = self._tracker(tmp_path)
        t.add_habit("walk")
        d = date(2026, 9, 1)
        t.complete_habit("walk", today=d)
        t.complete_habit("walk", today=d)
        assert t.habit_history()["walk"]["dates"] == ["2026-09-01"]

    def test_old_state_file_loads_unchanged_and_has_no_history(self, tmp_path):
        path = tmp_path / "artemis_state.json"
        path.write_text(json.dumps({"habits": {"read": {
            "streak": 4, "last_done": "2026-08-30", "best_streak": 9,
            "total_completions": 40, "created_at": "2026-01-01T00:00:00+00:00"}}}))
        t = self._tracker(tmp_path)
        assert t.get_habit("read").streak == 4
        info = t.habit_history()["read"]
        assert info["dates"] == [] and info["since"] == ""
        # until something is completed, the serialised shape is exactly the old one
        assert set(Habit.from_dict("read", json.loads(path.read_text())["habits"]["read"]).to_dict()) == {
            "streak", "last_done", "best_streak", "total_completions", "created_at"}

    def test_history_is_capped(self):
        many = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(HISTORY_CAP_DAYS + 120)]
        h = Habit.from_dict("x", {"history": many, "history_since": many[0]})
        assert len(h.history) == HISTORY_CAP_DAYS
        assert h.history[-1] == many[-1]                     # newest kept, oldest dropped

    def test_garbage_history_entries_are_ignored(self):
        h = Habit.from_dict("x", {"history": ["2026-09-01", "nonsense", None, 5, "2026-09-01"]})
        assert h.history == ["2026-09-01"]

    def test_completing_a_habit_already_done_today_backfills_history(self, tmp_path):
        path = tmp_path / "artemis_state.json"
        today = date(2026, 9, 1)
        path.write_text(json.dumps({"habits": {"read": {
            "streak": 1, "last_done": today.isoformat(), "best_streak": 1,
            "total_completions": 1, "created_at": "2026-08-01T00:00:00+00:00"}}}))
        t = self._tracker(tmp_path)
        t.complete_habit("read", today=today)
        assert t.habit_history()["read"]["dates"] == ["2026-09-01"]


# ===================================================== Consensus (#159)

def fake_artemis_with_streak(streak=6, done_yesterday=True, goals=None):
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    habit = SimpleNamespace(streak=streak, last_done=yesterday if done_yesterday else "2000-01-01")
    return SimpleNamespace(tracker=SimpleNamespace(
        get_habits=lambda: {"meditate": habit},
        get_at_risk_goals=lambda today=None: goals or {},
    ))


def fake_apollo(*signals):
    return SimpleNamespace(rest_signals=lambda now=None: list(signals))


REST = {"direction": "rest", "weight": 1.0, "source": "apollo",
        "reason": "you've averaged 4.5h of sleep over the last few nights"}


class TestConsensus:
    def test_surfaces_a_real_disagreement_and_explains_it(self):
        c = ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak())
        out = c.annotate("complete_habit", "Habit logged.")
        assert out.startswith("Habit logged.")
        assert "point in different directions" in out
        assert "4.5h of sleep" in out and "'meditate' streak" in out
        assert "not quietly picking one" in out

    def test_original_reply_is_never_altered(self):
        c = ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak())
        text = "Habit logged."
        assert c.annotate("complete_habit", text).startswith(text + "\n\n")

    def test_silent_when_only_one_side_speaks(self):
        assert ConsensusEngine(fake_apollo(), fake_artemis_with_streak()).note_for("complete_habit") is None
        assert ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak(done_yesterday=False)).note_for("complete_habit") is None

    def test_weak_signals_do_not_fire(self):
        weak = dict(REST, weight=0.5)
        assert ConsensusEngine(fake_apollo(weak), fake_artemis_with_streak()).note_for("get_motivation") is None

    def test_short_streaks_are_not_worth_defending(self):
        assert ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak(streak=2)).note_for("complete_habit") is None

    def test_only_whitelisted_intents(self):
        c = ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak())
        for intent in ("complete_habit", "get_motivation", "suggest_exercise"):
            assert c.note_for(intent)
        for intent in ("log_expense", "recommend_movie", "chat"):
            assert c.note_for(intent) is None
            assert c.annotate(intent, "x") == "x"

    def test_kill_switch(self):
        c = ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak(), enabled=False)
        assert c.annotate("complete_habit", "x") == "x"

    def test_lean_wording_reflects_the_weights(self):
        many_rest = [REST, dict(REST, reason="your latest logged mood was 'exhausted'")]
        c = ConsensusEngine(fake_apollo(*many_rest), fake_artemis_with_streak(streak=3))
        assert "rest signals look stronger" in c.note_for("complete_habit")
        even = ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak(streak=14))
        assert "evenly matched" in even.note_for("complete_habit")

    def test_a_broken_module_means_silence_not_a_crash(self):
        boom = SimpleNamespace(rest_signals=MagicMock(side_effect=RuntimeError("db gone")))
        c = ConsensusEngine(boom, fake_artemis_with_streak())
        assert c.annotate("complete_habit", "x") == "x"

    def test_does_not_repeat_itself(self):
        c = ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak())
        once = c.annotate("complete_habit", "x")
        assert c.annotate("complete_habit", once) == once

    def test_goal_signal_counts_as_push(self):
        c = ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak(streak=0, goals={"g": object()}))
        assert "goal(s) are due soon" in c.note_for("get_motivation")


class TestOrchestratorHook:
    def _orch(self, module_reply="Habit logged."):
        from modules.base import BaseModule
        from modules.hecate import HecateEngine
        from modules.hestia.orchestrator import HestiaOrchestrator

        class Mod(BaseModule):
            name = "artemis"
            def can_handle(self, intent): return True
            def handle(self, intent, entities, context):
                return {"response": module_reply, "data": {}, "confidence": 0.95}

        orch = HestiaOrchestrator()
        orch.register_hecate(HecateEngine())
        orch.register(Mod())
        return orch, "artemis_complete_habit"      # routed by the artemis_ prefix

    def _dispatch(self, orch, intent):
        return orch.dispatch("x", {"intent": intent, "entities": {}, "confidence": 0.95})

    def test_off_by_default(self):
        orch, intent = self._orch()
        assert self._dispatch(orch, intent) == "Habit logged."

    def test_note_is_appended_when_attached(self):
        orch, intent = self._orch()
        orch.attach_consensus(ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak()))
        out = self._dispatch(orch, intent)
        assert out.startswith("Habit logged.") and "different directions" in out

    def test_a_crashing_layer_cannot_break_the_reply(self):
        orch, intent = self._orch()
        bad = MagicMock()
        bad.annotate.side_effect = RuntimeError("boom")
        orch.attach_consensus(bad)
        assert self._dispatch(orch, intent) == "Habit logged."

    def test_can_be_detached(self):
        orch, intent = self._orch()
        orch.attach_consensus(ConsensusEngine(fake_apollo(REST), fake_artemis_with_streak()))
        orch.attach_consensus(None)
        assert self._dispatch(orch, intent) == "Habit logged."


# ===================================================== Dionysus (#149)

def make_dionysus(tmp_path, response, mood_ctx, **kw):
    from modules.dionysus.db import DionysusDB
    from modules.dionysus.engine import DionysusEngine

    llm = MagicMock()
    llm.generate.return_value = response
    e = DionysusEngine(ollama_cfg={}, llm=llm, **kw)
    e.db = DionysusDB(str(tmp_path / "d.db"))
    if mood_ctx is not None:
        e.attach_apollo(SimpleNamespace(recent_mood_context=lambda: mood_ctx))
    return e, llm


MOVIES = json.dumps({"recommendations": [{"title": "Paddington", "year": "2014", "reason": "warm"}]})
SONGS = json.dumps({"recommendations": [{"title": "Sunrise", "artist": "Norah", "reason": "soft"}]})


class TestDionysusMoodAware:
    @pytest.fixture(autouse=True)
    def _no_omdb(self, monkeypatch):
        monkeypatch.setattr("modules.dionysus.engine.OMDB_KEY", "")

    def test_uses_logged_mood_when_user_gave_none_and_says_so(self, tmp_path):
        e, llm = make_dionysus(tmp_path, MOVIES, {"mood": "tired", "low_trend": False})
        r = e.handle("recommend_movie", {"raw_query": "recommend me a movie"}, {})
        assert "based on your logged mood: tired" in r["response"]
        assert "tired" in llm.generate.call_args[0][0]

    def test_low_trend_leans_comforting(self, tmp_path):
        e, llm = make_dionysus(tmp_path, MOVIES, {"mood": "sad", "low_trend": True})
        r = e.handle("recommend_movie", {"raw_query": "suggest something to watch"}, {})
        assert "leaned toward comforting" in r["response"]
        assert "comforting" in llm.generate.call_args[0][0]
        assert "avoid forced cheerfulness" in llm.generate.call_args[0][0]

    def test_music_too(self, tmp_path):
        e, llm = make_dionysus(tmp_path, SONGS, {"mood": "stressed", "low_trend": False})
        r = e.handle("recommend_music", {"raw_query": "recommend some music"}, {})
        assert "based on your logged mood: stressed" in r["response"]

    def test_explicit_mood_or_genre_always_wins(self, tmp_path):
        e, llm = make_dionysus(tmp_path, MOVIES, {"mood": "tired", "low_trend": False})
        r = e.handle("recommend_movie", {"mood": "action", "raw_query": "an action movie"}, {})
        assert "logged mood" not in r["response"]
        r = e.handle("recommend_movie", {"genre": "horror"}, {})
        assert "logged mood" not in r["response"]

    def test_specific_query_is_left_alone(self, tmp_path):
        e, _ = make_dionysus(tmp_path, MOVIES, {"mood": "tired", "low_trend": False})
        r = e.handle("recommend_movie", {"raw_query": "a scary movie about sharks"}, {})
        assert "logged mood" not in r["response"]

    def test_no_recent_mood_behaves_as_before(self, tmp_path):
        e, _ = make_dionysus(tmp_path, MOVIES, None)
        r = e.handle("recommend_movie", {"raw_query": "recommend me a movie"}, {})
        assert "logged mood" not in r["response"]
        e2, _ = make_dionysus(tmp_path, MOVIES, None)
        e2.attach_apollo(SimpleNamespace(recent_mood_context=lambda: None))
        assert "logged mood" not in e2.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})["response"]

    def test_opt_out(self, tmp_path):
        e, _ = make_dionysus(tmp_path, MOVIES, {"mood": "tired", "low_trend": False}, mood_aware=False)
        assert "logged mood" not in e.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})["response"]

    def test_apollo_failure_is_ignored(self, tmp_path):
        e, _ = make_dionysus(tmp_path, MOVIES, None)
        e.attach_apollo(SimpleNamespace(recent_mood_context=MagicMock(side_effect=RuntimeError("x"))))
        assert "Paddington" in e.handle("recommend_movie", {"raw_query": "recommend a movie"}, {})["response"]

    def test_end_to_end_with_real_apollo(self, tmp_path):
        a = apollo(tmp_path)
        a.db.log_mood("exhausted", "")
        e, llm = make_dionysus(tmp_path, MOVIES, None)
        e.attach_apollo(a)
        r = e.handle("recommend_movie", {"raw_query": "recommend me a movie"}, {})
        assert "based on your logged mood: exhausted" in r["response"]


# ===================================================== DB maintenance (#233)

def _bloat(path: Path, rows=2500, keep=20):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob BLOB)")
    conn.executemany("INSERT INTO t (blob) VALUES (?)", [(os.urandom(1024),) for _ in range(rows)])
    conn.commit()
    conn.execute("DELETE FROM t WHERE id > ?", (keep,))
    conn.commit()
    conn.close()


def _count(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    finally:
        conn.close()


class TestDBMaintenance:
    @pytest.fixture(autouse=True)
    def _data_dir(self, tmp_path):
        (tmp_path / "data").mkdir(exist_ok=True)

    def _m(self, tmp_path, **kw):
        return DBMaintenance(root=tmp_path, busy_timeout_ms=50, **kw)

    def test_vacuums_a_bloated_database_and_keeps_every_row(self, tmp_path):
        db = tmp_path / "data" / "x.db"
        _bloat(db)
        before = db.stat().st_size
        res = self._m(tmp_path).run()
        assert [r["status"] for r in res] == ["vacuumed"]
        assert db.stat().st_size < before / 4
        assert _count(db) == 20                                 # never deletes data

    def test_healthy_database_is_left_alone(self, tmp_path):
        db = tmp_path / "data" / "ok.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob BLOB)")
        conn.executemany("INSERT INTO t (blob) VALUES (?)", [(b"x" * 100,)] * 50)
        conn.commit()
        conn.close()
        size = db.stat().st_size
        assert self._m(tmp_path).run()[0]["status"] == "healthy"
        assert db.stat().st_size == size

    def test_at_most_weekly_per_database(self, tmp_path):
        db = tmp_path / "data" / "x.db"
        _bloat(db)
        m = self._m(tmp_path)
        t0 = datetime(2026, 9, 1, 3, 0, tzinfo=UTC)
        assert m.run(t0)[0]["status"] == "vacuumed"
        _bloat_more = sqlite3.connect(db)
        _bloat_more.close()
        assert m.run(t0 + timedelta(days=3))[0]["status"] == "recent"
        assert m.run(t0 + timedelta(days=8))[0]["status"] in ("healthy", "vacuumed")

    def test_locked_database_is_skipped_and_retried(self, tmp_path):
        db = tmp_path / "data" / "x.db"
        _bloat(db)
        m = self._m(tmp_path)
        blocker = sqlite3.connect(db, isolation_level=None)
        blocker.execute("BEGIN EXCLUSIVE")
        try:
            res = m.run()
            assert res[0]["status"] == "locked" and DBMaintenance.needs_retry(res)
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        assert _count(db) == 20
        assert m.run()[0]["status"] == "vacuumed"               # retried, not marked done

    def test_insufficient_disk_space_skips(self, tmp_path):
        db = tmp_path / "data" / "x.db"
        _bloat(db)
        with patch("core.db_maintenance.shutil.disk_usage",
                   return_value=SimpleNamespace(total=1, used=1, free=1024)):
            res = self._m(tmp_path).run()
        assert res[0]["status"] == "no_space" and DBMaintenance.needs_retry(res)
        assert _count(db) == 20

    def test_non_sqlite_files_are_never_touched(self, tmp_path):
        fake = tmp_path / "data" / "notes.db"
        fake.write_text("this is not a database")
        assert self._m(tmp_path).run()[0]["status"] == "not_sqlite"
        assert fake.read_text() == "this is not a database"

    def test_discovers_nested_databases_and_extra_paths(self, tmp_path):
        nested = tmp_path / "data" / "apollo" / "apollo.db"
        nested.parent.mkdir(parents=True)
        sqlite3.connect(nested).close()
        extra = tmp_path / "elsewhere.db"
        sqlite3.connect(extra).close()
        found = {p.name for p in DBMaintenance(root=tmp_path, paths=[extra]).databases()}
        assert found == {"apollo.db", "elsewhere.db"}

    def test_state_is_persisted_and_survives_corruption(self, tmp_path):
        m = self._m(tmp_path)
        (tmp_path / "data" / "a.db").write_bytes(b"")
        sqlite3.connect(tmp_path / "data" / "a.db").close()
        m.run()
        assert m.state_path.exists()
        m.state_path.write_text("{corrupt")
        assert isinstance(m.run(), list)                          # tolerated, no crash

    def test_missing_file(self, tmp_path):
        assert self._m(tmp_path).maintain(tmp_path / "data" / "nope.db")["status"] == "missing"


# ===================================================== Heartbeat hooks

class TestHeartbeatHooks:
    def _hb(self, **kw):
        from core.heartbeat import HestiaHeartbeat
        return HestiaHeartbeat(interval=1, mnemosyne=MagicMock(), **kw)

    def test_apollo_checkins_are_spoken(self):
        apollo_ = SimpleNamespace(
            check_hydration_nudge=lambda: "drink",
            check_weekly_summary=lambda: None,
            check_burnout=lambda: "  ",
            check_goal_pace_reminder=lambda: "pace",
        )
        hb = self._hb(apollo=apollo_)
        with patch("core.heartbeat.bus") as bus:
            hb._maybe_run_apollo_checkins()
        spoken = [c.args[1]["text"] for c in bus.emit.call_args_list if c.args[0] == "speak"]
        assert spoken == ["drink", "pace"]

    def test_one_failing_hook_does_not_stop_the_others(self):
        def boom(): raise RuntimeError("db locked")
        apollo_ = SimpleNamespace(check_hydration_nudge=boom, check_weekly_summary=lambda: "summary")
        hb = self._hb(apollo=apollo_)
        with patch("core.heartbeat.bus") as bus:
            hb._maybe_run_apollo_checkins()
        assert [c.args[1]["text"] for c in bus.emit.call_args_list] == ["summary"]

    def test_no_apollo_is_a_no_op(self):
        with patch("core.heartbeat.bus") as bus:
            self._hb()._maybe_run_apollo_checkins()
        bus.emit.assert_not_called()

    def test_maintenance_only_runs_in_the_small_hours(self):
        maint = MagicMock()
        maint.run.return_value = []
        maint.needs_retry.return_value = False
        hb = self._hb(maintenance=maint)
        with patch("core.heartbeat.datetime") as dt:
            dt.now.return_value = datetime(2026, 9, 30, 14, 0)
            hb._maybe_run_db_maintenance()
            maint.run.assert_not_called()
            dt.now.return_value = datetime(2026, 9, 30, 2, 0)
            hb._maybe_run_db_maintenance()
            maint.run.assert_called_once()
            hb._maybe_run_db_maintenance()                        # already done today
            maint.run.assert_called_once()

    def test_busy_database_is_retried_on_the_next_tick(self):
        maint = MagicMock()
        maint.run.return_value = [{"status": "locked"}]
        maint.needs_retry.return_value = True
        hb = self._hb(maintenance=maint)
        with patch("core.heartbeat.datetime") as dt:
            dt.now.return_value = datetime(2026, 9, 30, 2, 0)
            hb._maybe_run_db_maintenance()
            hb._maybe_run_db_maintenance()
        assert maint.run.call_count == 2

    def test_maintenance_crash_does_not_propagate(self):
        maint = MagicMock()
        maint.run.side_effect = RuntimeError("disk on fire")
        hb = self._hb(maintenance=maint)
        with patch("core.heartbeat.datetime") as dt:
            dt.now.return_value = datetime(2026, 9, 30, 2, 0)
            hb._maybe_run_db_maintenance()                        # must not raise

    def test_run_heartbeat_calls_both_hooks(self):
        hb = self._hb()
        hb._maybe_run_apollo_checkins = MagicMock()
        hb._maybe_run_db_maintenance = MagicMock()
        try:
            hb._run_heartbeat()
        except Exception:
            pass
        hb._maybe_run_apollo_checkins.assert_called_once()
        hb._maybe_run_db_maintenance.assert_called_once()


# ===================================================== Web endpoints (#183)

class TestWebEndpoints:
    def _client(self, tmp_path, artemis=None):
        import web_ui

        a = apollo(tmp_path)
        cls = next(getattr(web_ui, n) for n in dir(web_ui)
                   if isinstance(getattr(web_ui, n), type) and hasattr(getattr(web_ui, n), "_register_apollo_routes"))
        ui = cls.__new__(cls)
        from flask import Flask
        ui.app = Flask(__name__)
        for attr in ("mnemosyne", "pluto", "chronos", "athena", "iris", "hestia", "orchestrator"):
            setattr(ui, attr, None)
        ui.apollo, ui.artemis = a, artemis
        ui._register_apollo_routes()
        return ui, a

    def test_dashboard_endpoints(self, tmp_path):
        ui, a = self._client(tmp_path)
        a.db.log_sleep(7.5, "good", "", bed_time="23:00", wake_time="06:30")
        a.db.log_weight(70, "")
        a.db.log_mood("good", "")
        c = ui.app.test_client()
        data = c.get("/api/apollo/dashboard?days=14").get_json()
        assert set(data) >= {"sleep", "weight", "water", "mood", "streaks", "units"}
        assert data["sleep"][0]["score"] is not None
        for key in ("sleep", "weight", "water", "mood", "streaks", "steps"):
            assert c.get(f"/api/apollo/{key}").status_code == 200
        assert c.get("/api/apollo/sleep").get_json()[0]["hours"] == 7.5

    def test_days_param_is_clamped(self, tmp_path):
        ui, _ = self._client(tmp_path)
        assert ui.app.test_client().get("/api/apollo/dashboard?days=99999").get_json()["days"] == 180

    def test_no_apollo_degrades_to_empty(self, tmp_path):
        ui, _ = self._client(tmp_path)
        ui.apollo = None
        c = ui.app.test_client()
        assert c.get("/api/apollo/dashboard").get_json() == {}
        assert c.get("/api/apollo/sleep").get_json() == []

    def test_heatmap_endpoint(self, tmp_path):
        import web_ui
        t = ArtemisTracker(state_path=str(tmp_path / "s.json")) \
            if "state_path" in ArtemisTracker.__init__.__code__.co_varnames \
            else ArtemisTracker(str(tmp_path / "s.json"))
        t.add_habit("meditate")
        t.complete_habit("meditate", today=datetime.now(UTC).date())
        ui, _ = self._client(tmp_path, artemis=SimpleNamespace(tracker=t))
        ui._register_artemis_routes()
        data = ui.app.test_client().get("/api/artemis/heatmap?days=30").get_json()
        assert len(data["days"]) == 30
        h = data["habits"][0]
        assert h["name"] == "meditate" and h["done"] == [data["days"][-1]]
        assert h["since"] == data["days"][-1]


class TestIndexHtml:
    """The UI can't be rendered here, so pin the things that would silently
    break it: JS that parses, and every new tab wired end to end."""

    def _html(self):
        return (Path(__file__).parent.parent / "templates" / "index.html").read_text(encoding="utf-8")

    def test_new_tabs_are_wired(self):
        html = self._html()
        for section in ("health", "heatmap"):
            assert f'data-section="{section}"' in html
            assert f'<section id="{section}"' in html
            assert f"section === '{section}'" in html
        assert "/api/apollo/dashboard" in html and "/api/artemis/heatmap" in html
        assert 'id="plutoCatChart"' in html

    def test_moods_tab_reads_fields_the_api_returns(self):
        html = self._html()
        assert "m.mood || m.valence" in html

    def test_script_blocks_parse(self, tmp_path):
        import re, shutil, subprocess
        node = shutil.which("node")
        if not node:
            pytest.skip("node not available")
        for i, block in enumerate(re.findall(r"<script>(.*?)</script>", self._html(), flags=re.S)):
            f = tmp_path / f"b{i}.js"
            f.write_text(block, encoding="utf-8")
            r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
