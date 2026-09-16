# tests/test_observability.py
"""
Tests for core/observability.py — request IDs, the routing log, the
decision ring buffer, feedback logging, and module status aggregation
(backlog #3, #5, #8, #17, #259).

Every test redirects the JSONL logs into a tmp_path via the
HESTIA_LOG_DIR env var, reloading the module so the module-level log
directory constant picks it up. That keeps the suite from writing into the
repo's logs/ directory and makes the on-disk assertions real (the point of
the routing log is that it survives the process, so asserting only on the
in-memory ring would miss the half that matters).
"""
import importlib
import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def obs(tmp_path, monkeypatch):
    """core.observability reloaded with its log dir pointed at tmp_path."""
    monkeypatch.setenv("HESTIA_LOG_DIR", str(tmp_path))
    import core.observability as module

    reloaded = importlib.reload(module)
    yield reloaded
    # Handlers hold open file objects in tmp_path; drop them so the next
    # reload doesn't reuse a handler writing to a deleted directory.
    for name in ("hestia.routing", "hestia.feedback"):
        log = logging.getLogger(name)
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)
    monkeypatch.delenv("HESTIA_LOG_DIR", raising=False)
    importlib.reload(module)


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Request IDs (#17)
# ---------------------------------------------------------------------------

def test_request_id_defaults_to_placeholder_outside_a_request(obs):
    # A fresh contextvar must not raise or invent an id — log records
    # emitted at startup have no request to belong to.
    assert obs.current_request_id() == "-"


def test_new_request_id_is_set_and_returned(obs):
    rid = obs.new_request_id()
    assert rid == obs.current_request_id()
    assert len(rid) == 8


def test_new_request_id_differs_per_call(obs):
    assert obs.new_request_id() != obs.new_request_id()


def test_set_request_id_adopts_external_value(obs):
    obs.set_request_id("abc123")
    assert obs.current_request_id() == "abc123"
    # Falsy values fall back to the placeholder rather than an empty string,
    # which would make the log column silently blank.
    obs.set_request_id("")
    assert obs.current_request_id() == "-"


def test_filter_stamps_request_id_onto_records(obs):
    obs.new_request_id()
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", (), None)
    assert obs.RequestIdFilter().filter(record) is True
    assert record.request_id == obs.current_request_id()


def test_filter_does_not_overwrite_an_existing_request_id(obs):
    # A record forwarded from another context (e.g. an API request handled
    # on a worker thread) already carries its own id.
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", (), None)
    record.request_id = "preset00"
    obs.RequestIdFilter().filter(record)
    assert record.request_id == "preset00"


# ---------------------------------------------------------------------------
# Routing log (#5)
# ---------------------------------------------------------------------------

def test_record_classification_writes_one_json_line(obs, tmp_path):
    diag = obs.Diagnostics()
    diag.record_classification(
        query="log my sleep",
        intent="apollo_track_sleep",
        confidence=0.92,
        module="apollo",
        reason="registry",
        latency_ms=12.34,
    )
    rows = _read_jsonl(tmp_path / "routing.jsonl")
    assert len(rows) == 1
    assert rows[0]["intent"] == "apollo_track_sleep"
    assert rows[0]["module"] == "apollo"
    assert rows[0]["confidence"] == 0.92
    assert rows[0]["latency_ms"] == 12.3      # rounded to 1dp
    assert rows[0]["source"] == "nlu"         # default


def test_record_classification_appends_rather_than_truncating(obs, tmp_path):
    diag = obs.Diagnostics()
    for i in range(3):
        diag.record_classification(
            query=f"q{i}", intent="chat", confidence=0.5, module="core"
        )
    assert len(_read_jsonl(tmp_path / "routing.jsonl")) == 3


def test_record_classification_truncates_very_long_queries(obs, tmp_path):
    diag = obs.Diagnostics()
    diag.record_classification(
        query="x" * 5000, intent="chat", confidence=0.5, module="core"
    )
    # Analysis wants the shape of the query, not an unbounded blob per line.
    assert len(_read_jsonl(tmp_path / "routing.jsonl")[0]["query"]) == 500


def test_record_classification_carries_the_request_id(obs, tmp_path):
    diag = obs.Diagnostics()
    rid = obs.new_request_id()
    diag.record_classification(query="q", intent="chat", confidence=0.5, module="core")
    assert _read_jsonl(tmp_path / "routing.jsonl")[0]["request_id"] == rid


def test_record_classification_tolerates_none_confidence(obs):
    # nlu_result.get("confidence") is genuinely absent on some error paths.
    diag = obs.Diagnostics()
    record = diag.record_classification(
        query="q", intent="chat", confidence=None, module="core", latency_ms=None
    )
    assert record["confidence"] == 0.0
    assert record["latency_ms"] == 0.0


def test_unwritable_log_dir_does_not_raise(tmp_path, monkeypatch):
    # Observability must never be the reason a query fails.
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    monkeypatch.setenv("HESTIA_LOG_DIR", str(blocker / "sub"))
    import core.observability as module

    reloaded = importlib.reload(module)
    diag = reloaded.Diagnostics()
    diag.record_classification(query="q", intent="chat", confidence=0.5, module="core")
    # The in-memory ring still works even with no file behind it.
    assert diag.last_decision()["query"] == "q"
    monkeypatch.delenv("HESTIA_LOG_DIR", raising=False)
    importlib.reload(module)


# ---------------------------------------------------------------------------
# Ring buffer and explanations (#3)
# ---------------------------------------------------------------------------

def test_last_decision_is_none_before_anything_is_routed(obs):
    assert obs.Diagnostics().last_decision() is None


def test_explain_last_is_honest_when_nothing_routed_yet(obs):
    assert "haven't routed anything" in obs.Diagnostics().explain_last()


def test_ring_buffer_is_bounded(obs):
    diag = obs.Diagnostics()
    for i in range(obs._RING_SIZE + 25):
        diag.record_classification(
            query=f"q{i}", intent="chat", confidence=0.5, module="core"
        )
    assert len(diag.recent_decisions(limit=0)) == obs._RING_SIZE


def test_recent_decisions_returns_newest_last(obs):
    diag = obs.Diagnostics()
    for i in range(5):
        diag.record_classification(
            query=f"q{i}", intent="chat", confidence=0.5, module="core"
        )
    recent = diag.recent_decisions(limit=2)
    assert [r["query"] for r in recent] == ["q3", "q4"]


def test_recent_decisions_returns_copies(obs):
    diag = obs.Diagnostics()
    diag.record_classification(query="q", intent="chat", confidence=0.5, module="core")
    diag.recent_decisions()[0]["intent"] = "mutated"
    assert diag.last_decision()["intent"] == "chat"


def test_explain_last_describes_the_previous_decision(obs):
    diag = obs.Diagnostics()
    diag.record_classification(
        query="i spent 200 on lunch",
        intent="pluto_log_expense",
        confidence=0.91,
        module="pluto",
        reason="registry: intent 'pluto_log_expense' -> pluto",
        latency_ms=88.0,
    )
    text = diag.explain_last()
    assert "pluto_log_expense" in text
    assert "pluto" in text
    assert "91%" in text
    assert "registry" in text


def test_explain_last_skips_the_explain_query_itself(obs):
    # The "why did you route that" query is itself classified and logged
    # before CoreModule answers it, so [-1] is the question, not the
    # decision being asked about.
    diag = obs.Diagnostics()
    diag.record_classification(
        query="log my workout", intent="apollo_log_workout",
        confidence=0.9, module="apollo",
    )
    diag.record_classification(
        query="why did you route that there", intent="explain_routing",
        confidence=0.96, module="core",
    )
    text = diag.explain_last()
    assert "apollo_log_workout" in text
    assert "explain_routing" not in text


def test_explain_last_mentions_a_non_nlu_source(obs):
    diag = obs.Diagnostics()
    diag.record_classification(
        query="log my sleep", intent="apollo_track_sleep", confidence=0.92,
        module="apollo", source="alias",
    )
    assert "alias" in diag.explain_last()


# ---------------------------------------------------------------------------
# Feedback (#259)
# ---------------------------------------------------------------------------

def test_record_feedback_attaches_to_the_previous_decision(obs, tmp_path):
    diag = obs.Diagnostics()
    diag.record_classification(
        query="log my sleep", intent="apollo_log_workout",
        confidence=0.7, module="apollo",
    )
    message = diag.record_feedback("I meant sleep, not a workout")
    rows = _read_jsonl(tmp_path / "feedback.jsonl")
    assert len(rows) == 1
    assert rows[0]["query"] == "log my sleep"
    assert rows[0]["intent"] == "apollo_log_workout"   # the labelled mistake
    assert "sleep" in rows[0]["note"]
    assert "apollo_log_workout" in message


def test_record_feedback_skips_the_report_query_itself(obs, tmp_path):
    diag = obs.Diagnostics()
    diag.record_classification(
        query="log my sleep", intent="apollo_log_workout",
        confidence=0.7, module="apollo",
    )
    diag.record_classification(
        query="that was wrong", intent="report_mistake",
        confidence=0.96, module="core",
    )
    diag.record_feedback("that was wrong")
    assert _read_jsonl(tmp_path / "feedback.jsonl")[0]["intent"] == "apollo_log_workout"


def test_record_feedback_with_no_prior_turn_says_so(obs):
    message = obs.Diagnostics().record_feedback("that was wrong")
    assert "don't have a previous query" in message


def test_feedback_count_reflects_written_records(obs):
    diag = obs.Diagnostics()
    assert diag.feedback_count() == 0
    diag.record_classification(query="q", intent="chat", confidence=0.5, module="core")
    diag.record_feedback("nope")
    diag.record_feedback("also nope")
    assert diag.feedback_count() == 2


# ---------------------------------------------------------------------------
# Module status (#8)
# ---------------------------------------------------------------------------

class _FakeModule:
    def __init__(self, name, probe=None, value=True, raises=False):
        self.name = name
        self._value = value
        self._raises = raises
        if probe == "ready":
            self.ready = self._probe
        elif probe == "available":
            self.available = self._probe

    def _probe(self):
        if self._raises:
            raise RuntimeError("postgres pool exhausted")
        return self._value


class _FakeOrchestrator:
    def __init__(self, modules):
        self._modules = {m.name: m for m in modules}

    @property
    def registered_modules(self):
        return list(self._modules)


def test_module_status_empty_without_an_orchestrator(obs):
    assert obs.Diagnostics().module_status() == {}


def test_module_status_normalises_ready_and_available(obs):
    diag = obs.Diagnostics(
        _FakeOrchestrator([
            _FakeModule("athena", probe="ready", value=True),
            _FakeModule("iris", probe="available", value=True),
        ])
    )
    status = diag.module_status()
    assert status["athena"] == {"registered": True, "state": "ready", "probe": "ready"}
    assert status["iris"]["probe"] == "available"
    assert status["iris"]["state"] == "ready"


def test_module_status_reports_degraded_for_a_false_probe(obs):
    diag = obs.Diagnostics(
        _FakeOrchestrator([_FakeModule("athena", probe="ready", value=False)])
    )
    assert diag.module_status()["athena"]["state"] == "degraded"


def test_module_status_reports_unknown_when_no_probe_exists(obs):
    # Most modules expose neither method; claiming "ready" for them would
    # be a fabricated health signal.
    diag = obs.Diagnostics(_FakeOrchestrator([_FakeModule("core")]))
    entry = diag.module_status()["core"]
    assert entry["state"] == "unknown"
    assert entry["probe"] is None


def test_module_status_isolates_a_raising_probe(obs):
    diag = obs.Diagnostics(
        _FakeOrchestrator([
            _FakeModule("pluto", probe="available", raises=True),
            _FakeModule("athena", probe="ready", value=True),
        ])
    )
    status = diag.module_status()
    assert status["pluto"]["state"] == "error"
    assert "postgres" in status["pluto"]["detail"]
    # One crashing module must not hide the rest of the report.
    assert status["athena"]["state"] == "ready"


def test_module_status_survives_an_orchestrator_that_raises(obs):
    class _Broken:
        @property
        def registered_modules(self):
            raise RuntimeError("boom")

    assert obs.Diagnostics(_Broken()).module_status() == {}


def test_status_summary_groups_modules_by_state(obs):
    diag = obs.Diagnostics(
        _FakeOrchestrator([
            _FakeModule("athena", probe="ready", value=True),
            _FakeModule("iris", probe="ready", value=False),
            _FakeModule("core"),
        ])
    )
    summary = diag.status_summary()
    assert "3 module(s) registered" in summary
    assert "ready: athena" in summary
    assert "degraded: iris" in summary
    assert "unknown: core" in summary


def test_bind_orchestrator_after_construction(obs):
    # Diagnostics is built before the orchestrator exists (the routing log
    # has to be ready for the first query), so late binding must work.
    diag = obs.Diagnostics()
    assert diag.module_status() == {}
    diag.bind_orchestrator(_FakeOrchestrator([_FakeModule("core")]))
    assert "core" in diag.module_status()


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

def test_timer_measures_a_non_negative_interval(obs):
    with obs.Timer() as t:
        sum(range(1000))
    assert t.ms >= 0.0


def test_timer_records_elapsed_even_when_the_body_raises(obs):
    t = obs.Timer()
    with pytest.raises(ValueError):
        with t:
            raise ValueError("boom")
    # __exit__ runs on the exception path, so latency is still logged for a
    # failed turn — the turns you most want timings for.
    assert t.ms >= 0.0
