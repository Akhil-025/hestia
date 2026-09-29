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


def test_record_classification_includes_the_script_field(obs, tmp_path):
    # backlog #27 — lets a future accuracy breakdown ask "is
    # classification worse for Devanagari input" from the log alone.
    diag = obs.Diagnostics()
    diag.record_classification(
        query="मौसम कैसा है", intent="get_weather", confidence=0.8, module="chronos"
    )
    assert _read_jsonl(tmp_path / "routing.jsonl")[0]["script"] == "devanagari"


def test_record_classification_script_field_for_english(obs, tmp_path):
    diag = obs.Diagnostics()
    diag.record_classification(
        query="what's the weather", intent="get_weather", confidence=0.8, module="chronos"
    )
    assert _read_jsonl(tmp_path / "routing.jsonl")[0]["script"] == "latin"


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
# Per-intent accuracy tracking (#30)
# ---------------------------------------------------------------------------

def _feedback_record(ts, intent, note="wrong"):
    return {"ts": ts, "request_id": "r", "note": note, "intent": intent}


def _write_feedback_line(tmp_path, record):
    with open(tmp_path / "feedback.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def test_per_intent_accuracy_excludes_intents_below_min_samples(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    _write_routing_line(tmp_path, _record(now.isoformat(), 0.9, "a"))
    _write_routing_line(tmp_path, _record(now.isoformat(), 0.9, "b"))
    diag = obs.Diagnostics()
    # Only 2 samples of the "chat" intent — below the default min of 3.
    assert diag.per_intent_accuracy() == {}


def test_per_intent_accuracy_reports_full_accuracy_with_no_feedback(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for i in range(3):
        _write_routing_line(tmp_path, _record(now.isoformat(), 0.9, f"r{i}"))
    diag = obs.Diagnostics()
    result = diag.per_intent_accuracy()
    assert result["chat"]["total"] == 3
    assert result["chat"]["flagged_wrong"] == 0
    assert result["chat"]["accuracy_estimate"] == 1.0


def test_per_intent_accuracy_reflects_flagged_mistakes(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for i in range(4):
        rec = _record(now.isoformat(), 0.9, f"r{i}")
        rec["intent"] = "apollo_log_workout"
        _write_routing_line(tmp_path, rec)
    _write_feedback_line(tmp_path, _feedback_record(now.isoformat(), "apollo_log_workout"))
    diag = obs.Diagnostics()
    result = diag.per_intent_accuracy()
    assert result["apollo_log_workout"]["total"] == 4
    assert result["apollo_log_workout"]["flagged_wrong"] == 1
    assert result["apollo_log_workout"]["accuracy_estimate"] == 0.75


def test_per_intent_accuracy_tracks_multiple_intents_independently(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for i in range(3):
        rec = _record(now.isoformat(), 0.9, f"a{i}")
        rec["intent"] = "apollo_log_workout"
        _write_routing_line(tmp_path, rec)
    for i in range(3):
        rec = _record(now.isoformat(), 0.9, f"p{i}")
        rec["intent"] = "pluto_log_expense"
        _write_routing_line(tmp_path, rec)
    _write_feedback_line(tmp_path, _feedback_record(now.isoformat(), "pluto_log_expense"))
    diag = obs.Diagnostics()
    result = diag.per_intent_accuracy()
    assert result["apollo_log_workout"]["accuracy_estimate"] == 1.0
    assert result["pluto_log_expense"]["flagged_wrong"] == 1


def test_per_intent_accuracy_excludes_records_outside_the_window(obs, tmp_path):
    from datetime import datetime, timezone, timedelta
    old = datetime.now(timezone.utc) - timedelta(days=30)
    for i in range(5):
        rec = _record(old.isoformat(), 0.9, f"r{i}")
        _write_routing_line(tmp_path, rec)
    diag = obs.Diagnostics()
    assert diag.per_intent_accuracy(days=7) == {}


def test_worst_performing_intents_sorted_worst_first(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for i in range(4):
        rec = _record(now.isoformat(), 0.9, f"a{i}")
        rec["intent"] = "apollo_log_workout"
        _write_routing_line(tmp_path, rec)
    for i in range(4):
        rec = _record(now.isoformat(), 0.9, f"p{i}")
        rec["intent"] = "pluto_log_expense"
        _write_routing_line(tmp_path, rec)
    # apollo: 1/4 wrong (75% accurate); pluto: 3/4 wrong (25% accurate)
    _write_feedback_line(tmp_path, _feedback_record(now.isoformat(), "apollo_log_workout"))
    for _ in range(3):
        _write_feedback_line(tmp_path, _feedback_record(now.isoformat(), "pluto_log_expense"))
    diag = obs.Diagnostics()
    worst = diag.worst_performing_intents()
    assert worst[0][0] == "pluto_log_expense"
    assert worst[1][0] == "apollo_log_workout"


def test_worst_performing_intents_respects_top_n(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for intent_name in ("a", "b", "c"):
        for i in range(3):
            rec = _record(now.isoformat(), 0.9, f"{intent_name}{i}")
            rec["intent"] = intent_name
            _write_routing_line(tmp_path, rec)
    diag = obs.Diagnostics()
    assert len(diag.worst_performing_intents(top_n=2)) == 2


def test_weekly_accuracy_summary_reports_no_data(obs):
    summary = obs.Diagnostics().weekly_accuracy_summary()
    assert "not enough" in summary.lower()


def test_weekly_accuracy_summary_reports_no_mistakes(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for i in range(3):
        _write_routing_line(tmp_path, _record(now.isoformat(), 0.9, f"r{i}"))
    summary = obs.Diagnostics().weekly_accuracy_summary()
    assert "no mistakes" in summary.lower()


def test_weekly_accuracy_summary_lists_worst_intents(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for i in range(3):
        rec = _record(now.isoformat(), 0.9, f"r{i}")
        rec["intent"] = "pluto_log_expense"
        _write_routing_line(tmp_path, rec)
    _write_feedback_line(tmp_path, _feedback_record(now.isoformat(), "pluto_log_expense"))
    summary = obs.Diagnostics().weekly_accuracy_summary()
    assert "pluto_log_expense" in summary


# ---------------------------------------------------------------------------
# Nightly low-confidence review (#6)
# ---------------------------------------------------------------------------

def _write_routing_line(tmp_path, record):
    with open(tmp_path / "routing.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _record(ts, confidence, request_id="r1"):
    return {
        "ts": ts, "request_id": request_id, "query": "q",
        "intent": "chat", "confidence": confidence, "module": "core",
        "reason": "", "latency_ms": 1.0, "source": "nlu",
    }


def test_low_confidence_since_returns_recent_low_confidence_records(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    _write_routing_line(tmp_path, _record(now.isoformat(), 0.3, "low"))
    _write_routing_line(tmp_path, _record(now.isoformat(), 0.9, "high"))
    diag = obs.Diagnostics()
    results = diag.low_confidence_since(hours=24, threshold=0.6)
    assert [r["request_id"] for r in results] == ["low"]


def test_low_confidence_since_excludes_entries_outside_the_window(obs, tmp_path):
    from datetime import datetime, timezone, timedelta
    old = datetime.now(timezone.utc) - timedelta(hours=48)
    _write_routing_line(tmp_path, _record(old.isoformat(), 0.3, "old"))
    diag = obs.Diagnostics()
    assert diag.low_confidence_since(hours=24, threshold=0.6) == []


def test_low_confidence_since_with_no_log_file_returns_empty(obs):
    assert obs.Diagnostics().low_confidence_since() == []


def test_low_confidence_since_skips_malformed_lines(obs, tmp_path):
    with open(tmp_path / "routing.jsonl", "a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
    diag = obs.Diagnostics()
    assert diag.low_confidence_since() == []  # must not raise


def test_write_review_queue_appends_new_entries(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    _write_routing_line(tmp_path, _record(now.isoformat(), 0.3, "a"))
    _write_routing_line(tmp_path, _record(now.isoformat(), 0.4, "b"))
    diag = obs.Diagnostics()
    added = diag.write_review_queue(threshold=0.6)
    assert added == 2
    assert len(_read_jsonl(tmp_path / "review_queue.jsonl")) == 2


def test_write_review_queue_deduplicates_by_request_id(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    _write_routing_line(tmp_path, _record(now.isoformat(), 0.3, "a"))
    diag = obs.Diagnostics()
    diag.write_review_queue(threshold=0.6)
    added_again = diag.write_review_queue(threshold=0.6)
    assert added_again == 0
    assert len(_read_jsonl(tmp_path / "review_queue.jsonl")) == 1


def test_write_review_queue_returns_zero_when_nothing_qualifies(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    _write_routing_line(tmp_path, _record(now.isoformat(), 0.95, "a"))
    diag = obs.Diagnostics()
    assert diag.write_review_queue(threshold=0.6) == 0


def test_review_queue_summary_reports_zero_when_empty(obs):
    assert "No low-confidence" in obs.Diagnostics().review_queue_summary()


def test_review_queue_summary_reports_a_count(obs, tmp_path):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    _write_routing_line(tmp_path, _record(now.isoformat(), 0.3, "a"))
    diag = obs.Diagnostics()
    diag.write_review_queue(threshold=0.6)
    summary = diag.review_queue_summary()
    assert "1 low-confidence" in summary


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
