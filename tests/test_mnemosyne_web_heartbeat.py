# tests/test_mnemosyne_web_heartbeat.py
"""
Integration points for the Mnemosyne additions: the web API that feeds the
graph page (#32) and learning summary, the heartbeat's morning-brief study
line (#33), and the heartbeat's background-job hook (#39/#43/#47).
"""
import os
import re
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_mnemosyne import make_engine  # noqa: E402
from core.heartbeat import HestiaHeartbeat  # noqa: E402

flask = pytest.importorskip("flask")
from web_ui import HestiaWebUI  # noqa: E402


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp()
    eng, _ = make_engine(tmp)
    yield eng
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def client(engine):
    ui = HestiaWebUI(memory=engine)
    return ui.app.test_client()


# ---------------------------------------------------------------------------
# Web API
# ---------------------------------------------------------------------------

def test_graph_endpoint_returns_nodes_and_links(engine, client):
    engine.learn("sister_name", "Priya")
    data = client.get("/api/mnemosyne/graph").get_json()
    assert {n["name"] for n in data["nodes"]} == {"You", "Priya"}
    assert data["links"][0]["relation"] == "sister name"
    ids = {n["id"] for n in data["nodes"]}
    assert data["links"][0]["source"] in ids and data["links"][0]["target"] in ids


def test_graph_endpoint_is_empty_not_an_error_on_a_fresh_install(client):
    assert client.get("/api/mnemosyne/graph").get_json() == {"nodes": [], "links": []}


def test_graph_endpoint_caps_max_nodes_and_tolerates_junk(engine, client):
    for i in range(6):
        engine.learn(f"friend_{i}", f"Person{i}")
    assert len(client.get("/api/mnemosyne/graph?max_nodes=3").get_json()["nodes"]) == 3
    r = client.get("/api/mnemosyne/graph?max_nodes=banana")
    assert r.status_code == 200 and r.get_json() == {"nodes": [], "links": []}


def test_graph_endpoint_survives_a_memory_without_the_feature():
    ui = HestiaWebUI(memory=MagicMock(spec=["get_stats"]))
    assert ui.app.test_client().get("/api/mnemosyne/graph").get_json() == {"nodes": [], "links": []}


def test_learning_endpoint_reports_study_and_quiz_state(engine, client):
    engine.handle("add_study_fact", {"key": "entropy", "value": "disorder"}, {})
    data = client.get("/api/mnemosyne/learning").get_json()
    assert data["study"]["total"] == 1 and data["study"]["due"] == 1
    assert data["study"]["upcoming"][0]["key"] == "entropy"
    assert data["quiz"] == {}


def test_api_key_guard_covers_the_new_routes(engine):
    ui = HestiaWebUI(memory=engine, api_key="s3cret")
    c = ui.app.test_client()
    assert c.get("/api/mnemosyne/graph").status_code in (401, 403)


def test_graph_page_is_wired_into_the_dashboard_template():
    html = open(os.path.join(os.path.dirname(__file__), "..", "templates", "index.html"),
                encoding="utf-8").read()
    assert 'data-section="graph"' in html and 'id="graph"' in html
    assert "/api/mnemosyne/graph" in html
    nav_sections = set(re.findall(r'<a [^>]*data-section="(\w+)"', html))
    section_ids = set(re.findall(r'<section id="(\w+)" class="section[^"]*"', html))
    assert nav_sections <= section_ids          # every nav link has a page to show


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def _spoken(emit):
    return [c.args[1]["text"] for c in emit.call_args_list if c.args[0] == "speak"]


def test_morning_brief_includes_study_cards_due():
    mn = MagicMock()
    mn.get_study_brief.return_value = "You have 3 study cards due today."
    hb = HestiaHeartbeat(mnemosyne=mn)
    with patch("core.heartbeat.bus") as bus:
        hb._morning_brief()
    assert "You have 3 study cards due today." in _spoken(bus.emit)
    assert any(c.args[0] == "morning_brief_requested" for c in bus.emit.call_args_list)


def test_morning_brief_stays_silent_about_study_when_nothing_is_due():
    mn = MagicMock()
    mn.get_study_brief.return_value = ""
    hb = HestiaHeartbeat(mnemosyne=mn)
    with patch("core.heartbeat.bus") as bus:
        hb._morning_brief()
    assert len(_spoken(bus.emit)) == 2            # greeting + date only


def test_study_brief_failure_never_blocks_the_morning_brief():
    mn = MagicMock()
    mn.get_study_brief.side_effect = RuntimeError("db locked")
    hb = HestiaHeartbeat(mnemosyne=mn)
    with patch("core.heartbeat.bus") as bus:
        hb._morning_brief()
    assert any(c.args[0] == "morning_brief_requested" for c in bus.emit.call_args_list)


def test_morning_brief_works_with_an_engine_that_predates_study_mode():
    hb = HestiaHeartbeat(mnemosyne=MagicMock(spec=["get_due_reminders"]))
    with patch("core.heartbeat.bus") as bus:
        hb._morning_brief()
    assert len(_spoken(bus.emit)) == 2


def test_morning_brief_with_no_memory_at_all():
    with patch("core.heartbeat.bus") as bus:
        HestiaHeartbeat(mnemosyne=None)._morning_brief()
    assert len(_spoken(bus.emit)) == 2


def test_real_engine_study_brief_flows_into_the_spoken_brief(engine):
    engine.handle("add_study_fact", {"key": "entropy", "value": "disorder"}, {})
    hb = HestiaHeartbeat(mnemosyne=engine)
    with patch("core.heartbeat.bus") as bus:
        hb._morning_brief()
    assert any("1 study card due today" in t for t in _spoken(bus.emit))


def test_tick_runs_the_mnemosyne_background_jobs_and_survives_their_failure():
    mn = MagicMock()
    mn.get_due_reminders.return_value = []
    hb = HestiaHeartbeat(mnemosyne=mn)
    hb._run_heartbeat()
    mn.run_background_jobs.assert_called_once()

    mn.run_background_jobs.side_effect = RuntimeError("boom")
    hb._run_heartbeat()                           # must not raise


def test_tick_works_when_the_engine_has_no_background_jobs():
    mn = MagicMock(spec=["get_due_reminders"])
    mn.get_due_reminders.return_value = []
    HestiaHeartbeat(mnemosyne=mn)._run_heartbeat()
